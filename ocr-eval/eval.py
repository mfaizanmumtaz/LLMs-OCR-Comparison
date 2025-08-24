from argparse import ArgumentParser
import os
from tqdm.asyncio import tqdm
import json
import datasets
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import time
import logging
from models import OpenRouterModel, AVAILABLE_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Extract the text in the image. Give me the final text, nothing else."
RESULTS_DIR = "results"
MAX_TOKENS = 500
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Base delay in seconds
CONCURRENT_REQUESTS = 10  # Number of concurrent requests
RATE_LIMIT_DELAY = 0.1  # Delay between batches to respect rate limits

os.makedirs(RESULTS_DIR, exist_ok=True)

ds_ids = [
    "ahmedheakl/arocrbench_patsocr",
    "ahmedheakl/arocrbench_historicalbooks", 
    "ahmedheakl/arocrbench_khattparagraph",
    "ahmedheakl/arocrbench_synthesizear",
    "ahmedheakl/arocrbench_historyar",
    "ahmedheakl/arocrbench_adab",
    "ahmedheakl/arocrbench_muharaf",
    "ahmedheakl/arocrbench_onlinekhatt",
    "ahmedheakl/arocrbench_khatt",
    "ahmedheakl/arocrbench_isippt",
    "ahmedheakl/arocrbench_arabicocr",
    "ahmedheakl/arocrbench_hindawi",
    "ahmedheakl/arocrbench_evarest",
]

class AsyncOpenRouterModel:
    """Async wrapper for OpenRouterModel with retry logic"""
    
    def __init__(self, max_tokens: int, model_name: str):
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.base_model = OpenRouterModel(max_tokens=max_tokens, model_name=model_name)
        
    async def __call__(self, prompt: str, image, max_retries: int = MAX_RETRIES) -> str:
        """Async call with retry logic"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid overwhelming the API
                if attempt > 0:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.1f}s delay")
                
                # Run the synchronous model call in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.base_model, prompt, image
                )
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
        
        # If all retries failed, return empty string
        return ""

def get_async_model(model_name: str, flash_attn: bool) -> AsyncOpenRouterModel:
    """Get async model instance"""
    if model_name == "google":
        return AsyncOpenRouterModel(max_tokens=MAX_TOKENS, model_name="google/gemini-2.5-flash")
    if model_name == "gpt-5-mini":
        return AsyncOpenRouterModel(max_tokens=MAX_TOKENS, model_name="openai/gpt-5-mini")
    if model_name == "gpt-4.1-mini":
        return AsyncOpenRouterModel(max_tokens=MAX_TOKENS, model_name="openai/gpt-4.1-mini")
    if model_name == "mistralai":
        return AsyncOpenRouterModel(max_tokens=MAX_TOKENS, model_name="mistralai/mistral-saba")
    raise ValueError(f"Model {model_name} not found")

async def process_sample(
    model: AsyncOpenRouterModel,
    sample: Dict[str, Any],
    idx: int,
    answer_name: str,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Process a single sample with concurrency control"""
    async with semaphore:
        try:
            img = sample["image"]
            pred = await model(DEFAULT_PROMPT, img)
            gt = sample[answer_name]
            
            result = {"idx": idx, "gt": gt, "pred": pred}
            logger.debug(f"Successfully processed sample {idx}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {str(e)}")
            return {"idx": idx, "gt": sample[answer_name], "pred": ""}

async def process_dataset_async(
    model: AsyncOpenRouterModel,
    ds_id: str,
    model_name: str
) -> None:
    """Process entire dataset asynchronously"""
    ds_name = ds_id.split("_")[-1]
    logger.info(f"Starting evaluation of {ds_name}...")
    
    output_path = f"{RESULTS_DIR}/{model_name}_{ds_name}.json"
    
    # Check if output already exists
    if os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping...")
        return
    
    try:
        ds = datasets.load_dataset(ds_id, split="train")
        logger.info(f"Loaded dataset {ds_name} with {len(ds)} samples")
        
        answer_name = "answer" if ds_id in ds_ids[:3] else "text"
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        # Create tasks for all samples
        tasks = []
        for idx, sample in enumerate(ds):
            task = process_sample(model, sample, idx, answer_name, semaphore)
            tasks.append(task)
        
        # Process with progress bar
        results = []
        with tqdm(total=len(tasks), desc=f"Processing {ds_name}") as pbar:
            # Process in batches to avoid overwhelming the system
            batch_size = CONCURRENT_REQUESTS * 2
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                        results.append({"idx": len(results), "gt": "", "pred": ""})
                    else:
                        results.append(result)
                    pbar.update(1)
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(tasks):
                    await asyncio.sleep(RATE_LIMIT_DELAY)
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x["idx"])
        
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # Report statistics
        successful_predictions = sum(1 for r in results if r["pred"].strip())
        success_rate = (successful_predictions / len(results)) * 100
        logger.info(f"Completed {ds_name}: {successful_predictions}/{len(results)} "
                   f"successful predictions ({success_rate:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to process dataset {ds_name}: {str(e)}")
        raise

async def main_async(args):
    """Main async function"""
    logger.info(f"Starting async evaluation with model: {args.model_name}")
    logger.info(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    logger.info(f"Max retries per sample: {MAX_RETRIES}")
    
    model = get_async_model(args.model_name, args.flash_attn)
    
    start_time = time.time()
    
    # Process datasets concurrently (but with controlled concurrency)
    if args.parallel_datasets:
        # Process multiple datasets in parallel
        dataset_semaphore = asyncio.Semaphore(2)  # Limit concurrent datasets
        
        async def process_with_semaphore(ds_id):
            async with dataset_semaphore:
                await process_dataset_async(model, ds_id, args.model_name)
        
        dataset_tasks = [process_with_semaphore(ds_id) for ds_id in ds_ids]
        await asyncio.gather(*dataset_tasks)
    else:
        # Process datasets sequentially
        for ds_id in ds_ids:
            await process_dataset_async(model, ds_id, args.model_name)
    
    total_time = time.time() - start_time
    logger.info(f"Completed all evaluations in {total_time:.1f} seconds")
    
    # Generate summary report
    generate_summary_report(args.model_name)

def generate_summary_report(model_name: str):
    """Generate a summary report of all processed datasets"""
    logger.info("Generating summary report...")
    
    summary = {
        "model_name": model_name,
        "datasets": {},
        "total_samples": 0,
        "total_successful": 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for ds_id in ds_ids:
        ds_name = ds_id.split("_")[-1]
        output_path = f"{RESULTS_DIR}/{model_name}_{ds_name}.json"
        
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                successful = sum(1 for item in data if item["pred"].strip())
                total = len(data)
                
                summary["datasets"][ds_name] = {
                    "total_samples": total,
                    "successful_predictions": successful,
                    "success_rate": (successful / total * 100) if total > 0 else 0
                }
                
                summary["total_samples"] += total
                summary["total_successful"] += successful
                
            except Exception as e:
                logger.error(f"Failed to read {output_path}: {e}")
                summary["datasets"][ds_name] = {"error": str(e)}
    
    # Calculate overall success rate
    if summary["total_samples"] > 0:
        summary["overall_success_rate"] = (summary["total_successful"] / summary["total_samples"]) * 100
    else:
        summary["overall_success_rate"] = 0
    
    # Save summary
    summary_path = f"{RESULTS_DIR}/{model_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Overall: {summary['total_successful']}/{summary['total_samples']} "
               f"({summary['overall_success_rate']:.1f}%) successful predictions")

def main(args):
    """Synchronous entry point"""
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google", choices=AVAILABLE_MODELS)
    parser.add_argument("--flash_attn", default=False, action="store_true")
    parser.add_argument("--parallel_datasets", default=False, action="store_true",
                       help="Process multiple datasets in parallel (uses more resources)")
    parser.add_argument("--concurrent_requests", type=int, default=CONCURRENT_REQUESTS,
                       help="Number of concurrent requests per dataset")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES,
                       help="Maximum number of retries per failed request")
    
    args = parser.parse_args()
    
    # Update global settings from args
    CONCURRENT_REQUESTS = args.concurrent_requests
    MAX_RETRIES = args.max_retries
    
    main(args)