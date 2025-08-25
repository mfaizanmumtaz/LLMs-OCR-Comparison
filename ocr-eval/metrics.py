import json
from glob import glob
from tqdm import tqdm
import os
from Levenshtein import distance as lev
from sacrebleu import corpus_bleu, corpus_chrf
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.translate import meteor_score
from torchmetrics.text import CharErrorRate, WordErrorRate
import re

def preprocess_arabic_text(text: str) -> str:
    # Remove newlines
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants to bare alef
    text = re.sub('[إأٱآا]', 'ا', text)
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    # Normalize alef maksura to yeh
    text = text.replace('ى', 'ي')
    # This regex matches one or more tatweel characters (ـ) or any whitespace sequence.
    # The lambda replaces whitespace sequences with a single space,
    # and removes tatweel characters by replacing them with an empty string.
    text = re.sub(r'(ـ+)|\s+', lambda m: ' ' if m.group(0).isspace() else '', text).strip()
    text = ' '.join(text.split())
    return text

def avg(l):
    return 0 if len(l) == 0 else round(sum(l) / len(l), 2)
 
def evaluate_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
   
    references = []
    hypotheses = []
 
    metrics = {
        'edit_distance': [],
        'cer': [],
        'wer': [],
        'meteor': [],
    }
    cr = CharErrorRate()
    wr = WordErrorRate()
    for item in data:
        pred = item['pred']
        if pred is None: 
            pred = ""
        gt = preprocess_arabic_text(item['gt']).strip()
        pred = preprocess_arabic_text(pred).strip()
        edit_dist = lev(gt, pred)
        metrics['edit_distance'].append(edit_dist)
 
        metrics['meteor'].append(
            meteor_score.single_meteor_score(
                gt.split(),
                pred.split(),
                preprocess=lambda x: x.lower()
            )
        )
       
        references.append(gt)
        hypotheses.append(pred)
 
    return {
        'blue': round(corpus_bleu(hypotheses, [references]).score, 2),
        'chrf': round(corpus_chrf(hypotheses, [references]).score, 2),
        'ed': round(avg(metrics['edit_distance']), 2),
        'cer': round(cr(hypotheses, references).item(), 2),
        'wer': round(wr(hypotheses, references).item(), 2),
        'meteor': round(avg(metrics['meteor']), 2),
    }

def evaluate_model(model_name):
    """Evaluate a single model"""
    print(f"\n{'='*50}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*50}")
    
    files = glob(f"results/{model_name}_*.json")
    
    # Filter out summary files
    files = [f for f in files if not f.endswith('_summary.json')]
    
    if not files:
        print(f"No result files found for model: {model_name}")
        print(f"Expected files: results/{model_name}_*.json (excluding summary files)")
        return None
    
    print(f"Found files: {files}")
    
    data = []
    cer_total, wer_total, chrf_total = 0, 0, 0
    
    for file in tqdm(files, desc=f"Evaluating {model_name}"):
        file_name = file.split("/")[-1]
        ds_name = file_name.replace(".json", "").split("_")[-1]  # Extract dataset name
        
        print(f"Processing {ds_name} dataset...")
        results = evaluate_file(file)
        results["dataset"] = ds_name
        data.append(results)
        
        cer_total += results['cer']
        wer_total += results['wer']
        chrf_total += results['chrf']
        
        print(f"  {ds_name.upper()} Results:")
        print(f"    BLEU: {results['blue']}")
        print(f"    CHrF: {results['chrf']}")
        print(f"    CER: {results['cer']}")
        print(f"    WER: {results['wer']}")
        print(f"    METEOR: {results['meteor']}")
        print(f"    Edit Distance: {results['ed']}")
    
    # Calculate averages
    num_datasets = len(data)
    if num_datasets > 0:
        avg_cer = cer_total / num_datasets
        avg_wer = wer_total / num_datasets
        avg_chrf = chrf_total / num_datasets
        
        print(f"\n{model_name} OVERALL AVERAGES:")
        print(f"  CER: {avg_cer:.2f} | WER: {avg_wer:.2f} | CHrF: {avg_chrf:.2f}")
    
    # Save results
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{model_name}_metrics.json", "w") as f:
        json.dump(data, f, indent=4)
    
    return {
        'model_name': model_name,
        'results': data,
        'averages': {
            'cer': avg_cer if num_datasets > 0 else 0,
            'wer': avg_wer if num_datasets > 0 else 0,
            'chrf': avg_chrf if num_datasets > 0 else 0
        } if num_datasets > 0 else {}
    }

def run_all_evaluations():
    """Run evaluation on all specified models"""
    models = ["gpt-5-mini", "gpt-4.1-mini", "google"]
    
    all_results = []
    
    print("Starting evaluation for all models...")
    print(f"Models to evaluate: {models}")
    
    for model_name in models:
        result = evaluate_model(model_name)
        if result:
            all_results.append(result)
    
    # Print summary comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        print(f"{'Model':<15} {'CER':<8} {'WER':<8} {'CHrF':<8}")
        print("-" * 45)
        for result in all_results:
            if result['averages']:
                print(f"{result['model_name']:<15} "
                      f"{result['averages']['cer']:<8.2f} "
                      f"{result['averages']['wer']:<8.2f} "
                      f"{result['averages']['chrf']:<8.2f}")
        
        # Save combined results
        with open("metrics/all_models_comparison.json", "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nDetailed results saved in 'metrics/' directory")
        print(f"Combined results saved in 'metrics/all_models_comparison.json'")
    else:
        print("No results to compare - no model files found!")

if __name__ == "__main__":
    # Run evaluation on all models
    run_all_evaluations()
    
    # You can also run individual models:
    # evaluate_model("google")
    # evaluate_model("gpt-5-mini")
    # evaluate_model("gpt-4.1-mini")