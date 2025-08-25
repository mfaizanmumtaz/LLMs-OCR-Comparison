from argparse import ArgumentParser
import os
from tqdm.asyncio import tqdm
import json
import datasets
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Any, Optional
import time
import logging
from models import AsyncOpenRouterModel, AVAILABLE_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Extract the text in the image. Give me the final text, nothing else."
RESULTS_DIR = "results"
MAX_TOKENS = 30000
MAX_RETRIES = 5
RETRY_DELAY = 1.0  # Base delay in seconds
CONCURRENT_REQUESTS = 10  # Number of concurrent requests
RATE_LIMIT_DELAY = 0.1  # Delay between batches to respect rate limits

# Create results directory asynchronously
async def ensure_results_dir():
    """Ensure results directory exists"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

ds_ids = [
    # "ahmedheakl/arocrbench_patsocr",
    # "ahmedheakl/arocrbench_historicalbooks", 
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

def get_model(model_name: str) -> AsyncOpenRouterModel:
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

async def load_dataset_async(ds_id: str) -> Any:
    """Load dataset in executor to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: datasets.load_dataset(ds_id, split="train"))

async def file_exists_async(path: str) -> bool:
    """Check if file exists asynchronously"""
    try:
        async with aiofiles.open(path, 'r'):
            pass
        return True
    except FileNotFoundError:
        return False

async def save_json_async(data: Any, path: str):
    """Save JSON data asynchronously"""
    async with aiofiles.open(path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, indent=4, ensure_ascii=False))

async def load_json_async(path: str) -> Any:
    """Load JSON data asynchronously"""
    async with aiofiles.open(path, 'r', encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)

async def process_sample(
    model: AsyncOpenRouterModel,
    sample: Dict[str, Any],
    idx: int,
    answer_name: str,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Process a single sample with concurrency control and enhanced error handling"""
    async with semaphore:
        try:
            img = sample["image"]
            # Call the async model directly
            pred = await model(DEFAULT_PROMPT, img, max_retries=MAX_RETRIES)
            gt = sample[answer_name]
            
            # Log if we got an empty prediction (all retries failed)
            if not pred.strip():
                logger.warning(f"Sample {idx}: All retries failed, got empty prediction")
            
            result = {"idx": idx, "gt": gt, "pred": pred}
            logger.debug(f"Successfully processed sample {idx}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {str(e)}")
            return {"idx": idx, "gt": sample.get(answer_name, ""), "pred": ""}

async def process_dataset_async(
    model: AsyncOpenRouterModel,
    ds_id: str,
    model_name: str
) -> None:
    """Process entire dataset asynchronously"""
    ds_name = ds_id.split("_")[-1]
    logger.info(f"Starting evaluation of {ds_name}...")
    
    output_path = f"{RESULTS_DIR}/{model_name}_{ds_name}.json"
    
    # Check if output already exists (async)
    if await file_exists_async(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping...")
        return
    
    try:
        # Load dataset asynchronously
        ds = await load_dataset_async(ds_id)
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
        
        # Save results asynchronously
        await save_json_async(results, output_path)
        
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
    # Update global settings from args
    global CONCURRENT_REQUESTS, MAX_RETRIES
    CONCURRENT_REQUESTS = args.concurrent_requests
    MAX_RETRIES = args.max_retries
    
    logger.info(f"Starting async evaluation with model: {args.model_name}")
    logger.info(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    logger.info(f"Max retries per sample: {MAX_RETRIES}")
    
    # Ensure results directory exists
    await ensure_results_dir()
    
    model = get_model(args.model_name)
    
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
    
    # Close model session and save timing stats
    await model.close()
    
    total_time = time.time() - start_time
    logger.info(f"Completed all evaluations in {total_time:.1f} seconds")
    
    # Generate summary report
    await generate_summary_report(args.model_name)

async def generate_summary_report(model_name: str):
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
        
        if await file_exists_async(output_path):
            try:
                data = await load_json_async(output_path)
                
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
    
    # Save summary asynchronously
    summary_path = f"{RESULTS_DIR}/{model_name}_summary.json"
    await save_json_async(summary, summary_path)
    
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
    parser.add_argument("--model_name", type=str, default="google", choices=AVAILABLE_MODELS,
                       help="Model to use for evaluation")
    parser.add_argument("--parallel_datasets", default=False, action="store_true",
                       help="Process multiple datasets in parallel (uses more resources)")
    parser.add_argument("--concurrent_requests", type=int, default=10,
                       help="Number of concurrent requests per dataset")
    parser.add_argument("--max_retries", type=int, default=5,
                       help="Maximum number of retries per failed request")
    
    args = parser.parse_args()
    
    main(args)