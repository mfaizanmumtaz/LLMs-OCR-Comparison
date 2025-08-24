import asyncio
from argparse import ArgumentParser
import os
from tqdm.asyncio import tqdm as async_tqdm
import json
from concurrent.futures import ThreadPoolExecutor

import datasets
from bs4 import BeautifulSoup

from models import OpenRouterModel, AVAILABLE_MODELS
from prompts import HTML_PROMPT, DF_PROMPT

RESULTS_DIR = "results"
MAX_TOKENS = 2000
DS_ID = "ahmedheakl/arocrbench_tables"
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on API rate limits
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_table(html):
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    return table

def get_type(meta):
    return eval(meta)['figure_type']

def get_model(model_name: str, flash_attn, **kwargs):
    if model_name == "google":
        return OpenRouterModel(max_tokens=MAX_TOKENS, model_name="google/gemini-2.5-flash")
    if model_name == "gpt-5-mini":
        return OpenRouterModel(max_tokens=MAX_TOKENS, model_name="openai/gpt-5-mini")
    if model_name == "gpt-4.1-mini":
        return OpenRouterModel(max_tokens=MAX_TOKENS, model_name="openai/gpt-4.1-mini")
    if model_name == "mistralai":
        return OpenRouterModel(max_tokens=MAX_TOKENS, model_name="mistralai/mistral-saba")
    raise ValueError(f"Model {model_name} not found")

async def process_sample(model, sample, idx, is_html: bool, semaphore):
    """Process a single sample asynchronously"""
    async with semaphore:  # Limit concurrent requests
        if is_html != ("HTML" in eval(sample['metadata'])["_pipeline"]): 
            return None
            
        img = sample['image']
        try:
            prompt = HTML_PROMPT if is_html else DF_PROMPT
            
            # Run model inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                pred = await loop.run_in_executor(executor, model, prompt, img)
            
            if pred is None:
                print(f"Model returned None for idx {idx}")
                pred = ""
            else:
                if is_html:
                    # For HTML: extract table from HTML response
                    table = get_table(pred)
                    pred = str(table) if table is not None else ""
                    pred = pred.split("```html")[-1].split("```")[0]
                else:
                    # For CSV: use raw response, just clean up code blocks
                    pred = str(pred)
                    pred = pred.split("```csv")[-1].split("```")[0]
                    
        except Exception as e:
            print(f"Skipping {idx} for {e}")
            pred = ""
            
        gt = str(get_table(sample["code"])) if is_html else sample['data']
        return {"idx": idx, "gt": gt, "pred": pred, "type": get_type(sample['metadata'])}

async def eval_ds_async(ds: datasets.Dataset, model, is_html: bool, output_path: str):
    """Evaluate dataset asynchronously"""
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Create tasks for all samples
    tasks = []
    for idx, sample in enumerate(ds):
        task = process_sample(model, sample, idx, is_html, semaphore)
        tasks.append(task)
    
    # Process all tasks with progress bar
    results = []
    desc = f"Evaluating tables {'HTML' if is_html else 'CSV'}"
    
    # Use async_tqdm for progress tracking
    async for result in async_tqdm(
        asyncio.as_completed(tasks), 
        total=len(tasks), 
        desc=desc
    ):
        completed_result = await result
        if completed_result is not None:  # Only add non-None results
            results.append(completed_result)
    
    # Sort results by idx to maintain order
    results.sort(key=lambda x: x["idx"])
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

async def main_async(args):
    """Main async function"""
    # Load dataset once
    ds = datasets.load_dataset(DS_ID, split="train")
    
    # Create tasks for both HTML and CSV evaluation
    tasks = []
    
    # HTML evaluation task
    model_html = get_model(args.model_name, args.flash_attn, is_html=True)
    output_path_html = f"{RESULTS_DIR}/{args.model_name}_html.json"
    html_task = eval_ds_async(ds, model_html, is_html=True, output_path=output_path_html)
    tasks.append(html_task)
    
    # CSV evaluation task
    model_csv = get_model(args.model_name, args.flash_attn, is_html=False)
    output_path_csv = f"{RESULTS_DIR}/{args.model_name}_csv.json"
    csv_task = eval_ds_async(ds, model_csv, is_html=False, output_path=output_path_csv)
    tasks.append(csv_task)
    
    # Run both evaluations concurrently
    await asyncio.gather(*tasks)
    
    # Cleanup
    del model_html, model_csv

def main(args):
    """Synchronous wrapper for async main"""
    # Update global concurrent limit if specified
    global MAX_CONCURRENT_REQUESTS
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    
    asyncio.run(main_async(args))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google", choices=AVAILABLE_MODELS)
    parser.add_argument("--flash_attn", default=False, action="store_true")
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum concurrent requests")
    args = parser.parse_args()
    
    main(args)