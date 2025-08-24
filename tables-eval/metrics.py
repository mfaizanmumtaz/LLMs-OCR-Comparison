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
import pandas as pd
from io import StringIO
import traceback

# TEDS imports and classes
import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from bs4 import BeautifulSoup

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        try:
            pred = html.fromstring(pred, parser=parser)
            true = html.fromstring(true, parser=parser)
            if pred.xpath('body/table') and true.xpath('body/table'):
                pred = pred.xpath('body/table')[0]
                true = true.xpath('body/table')[0]
                if self.ignore_nodes:
                    etree.strip_tags(pred, *self.ignore_nodes)
                    etree.strip_tags(true, *self.ignore_nodes)
                n_nodes_pred = len(pred.xpath(".//*"))
                n_nodes_true = len(true.xpath(".//*"))
                n_nodes = max(n_nodes_pred, n_nodes_true)
                tree_pred = self.load_html_tree(pred)
                tree_true = self.load_html_tree(true)
                distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
                return 1.0 - (float(distance) / n_nodes)
            else:
                return 0.0
        except Exception as e:
            print(f"TEDS evaluation error: {e}")
            return 0.0

    def batch_evaluate(self, pred_json, true_json, samples):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
        '''
        scores = []
        for pred, gt in tqdm(zip(pred_json, true_json), total=len(pred_json), desc="Computing TEDS"):
            score = self.evaluate(pred, gt)
            scores.append(score)
        scores = dict(zip(samples, scores))
        return scores


def flatten_table(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return html_content
        
    new_table = soup.new_tag('table')
    new_tbody = soup.new_tag('tbody')
    new_table.append(new_tbody)
    
    # Flatten all rows into tbody
    for section in ['thead', 'tbody', 'tfoot']:
        section_tag = soup.find(section)
        if section_tag:
            for row in section_tag.find_all('tr'):
                new_tbody.append(row.extract())
            section_tag.extract()
    
    return str(new_table.prettify())

def arabic_to_english_numerals(input_str):
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'  # Arabic-Indic digits
    english_numerals = '0123456789'  # Western Arabic digits
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def aug_html(s):
    if not s or s.strip() == "":
        return "<html><body><table><tbody></tbody></table></body></html>"
    s = s.replace('dir=\"&lt;built-in function dir&gt;\"', "")
    s = re.sub(r"\s+", ' ', s)
    s = arabic_to_english_numerals(s)
    s = flatten_table(s)
    return f"<html>\n<body>\n{s}\n</body></html>"

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

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Compare two DataFrames cell by cell and return similarity score"""
    # Create the union of indices and columns
    all_index = df1.index.union(df2.index)
    all_columns = df1.columns.union(df2.columns)
    # Align both DataFrames on the same index and columns
    df1_aligned = df1.reindex(index=all_index, columns=all_columns)
    df2_aligned = df2.reindex(index=all_index, columns=all_columns)
    # Compare cell by cell (treating NaNs as equal)
    comparison = (df1_aligned == df2_aligned) | (df1_aligned.isna() & df2_aligned.isna())
    score = comparison.sum().sum() / comparison.size
    return score

def safe_to_df(s: str) -> pd.DataFrame:
    """Safely convert CSV string to DataFrame with error handling"""
    s = arabic_to_english_numerals(s)
    csv_data = StringIO(s)
    try:
        df = pd.read_csv(csv_data)
    except pd.errors.ParserError as e:
        csv_data.seek(0)
        print("ParserError encountered. Falling back to Python engine with on_bad_lines='skip'.")
        df = pd.read_csv(csv_data, engine='python', on_bad_lines='skip')
    
    # --- Ensure unique column labels ---
    # Convert all column names to strings. This converts NaN to 'nan'
    df.columns = df.columns.astype(str)
    # If there are duplicate column names, drop all but the first occurrence.
    if df.columns.duplicated().any():
        print("Warning: Duplicate column labels found. Dropping duplicates.")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # --- (Optional) Ensure unique index labels ---
    if df.index.duplicated().any():
        print("Warning: Duplicate index labels found. Resetting index.")
        df = df.reset_index(drop=True)
    
    return df

def avg(l):
    return 0 if len(l) == 0 else round(sum(l) / len(l), 2)

def evaluate_file_standard(file_path):
    """Standard evaluation metrics (BLEU, CHrF, CER, WER, METEOR, ED)"""
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

def evaluate_file_teds(file_path):
    """TEDS evaluation for HTML tables"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    teds = TEDS(n_jobs=1)  # Using 1 job to avoid multiprocessing issues
    pred_json = []
    true_json = []
    samples = []
    
    for item in data:
        pred = item['pred']
        gt = item['gt']
        
        # Clean predictions
        if pred:
            pred = pred.replace("html\n", "").strip()
        
        pred_html = aug_html(pred if pred else "")
        true_html = aug_html(gt if gt else "")
        
        pred_json.append(pred_html)
        true_json.append(true_html)
        samples.append(item['idx'])
    
    scores = teds.batch_evaluate(pred_json, true_json, samples)
    avg_score = sum(scores.values()) * 100 / len(scores) if scores else 0
    
    return {
        'teds': round(avg_score, 2),
        'individual_scores': scores
    }

def evaluate_file_jaccord(file_path):
    """Jaccord evaluation for CSV tables (DataFrame comparison)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_score = 0
    num_failed = 0
    bad_samples = []
    valid_samples = 0
    
    print("Computing Jaccord CSV score...")
    for item in tqdm(data, desc="Processing CSV samples"):
        try:
            gt_text = item['gt']
            pred_text = item['pred'] if item['pred'] is not None else ""
            
            # Skip empty samples
            if not gt_text.strip():
                continue
                
            gt_df = safe_to_df(gt_text)
            pred_df = safe_to_df(pred_text)
            
            score = compare_dataframes(gt_df, pred_df)
            total_score += score
            valid_samples += 1
            
            if score < 0.1:
                bad_samples.append(item['idx'])
                
        except Exception as e:
            idx = item['idx']
            if "No columns to parse" in str(e): 
                continue
            print(f"Skipping sample {idx}: {str(e)}")
            num_failed += 1
    
    avg_score = (total_score / valid_samples * 100) if valid_samples > 0 else 0
    
    print(f"Jaccord CSV Score: {avg_score:.2f} | Failed: {num_failed}/{len(data)}")
    if bad_samples:
        print(f"Bad samples found at indices: {bad_samples[:10]}...")  # Show first 10
    
    return {
        'jaccord': round(avg_score, 2),
        'num_failed': num_failed,
        'bad_samples': bad_samples,
        'valid_samples': valid_samples
    }

def evaluate_model_complete(model_name):
    """Complete evaluation including standard metrics + TEDS for HTML + Jaccord for CSV"""
    print(f"\n{'='*60}")
    print(f"Complete Evaluation for: {model_name}")
    print(f"{'='*60}")
    
    files = glob(f"results/{model_name}_*.json")
    
    if not files:
        print(f"No result files found for model: {model_name}")
        return None
    
    print(f"Found files: {files}")
    
    all_results = {}
    
    for file in files:
        file_name = file.split("/")[-1]
        ds_name = file_name.replace(".json", "").split("_")[-1]  # html or csv
        
        print(f"\nProcessing {ds_name.upper()} dataset...")
        
        # Standard metrics
        standard_results = evaluate_file_standard(file)
        all_results[ds_name] = standard_results
        all_results[ds_name]['dataset'] = ds_name
        
        # Dataset-specific metrics
        if ds_name == 'html':
            print("  Computing TEDS score...")
            teds_results = evaluate_file_teds(file)
            all_results[ds_name]['teds'] = teds_results['teds']
            print(f"  TEDS Score: {teds_results['teds']}")
            
        elif ds_name == 'csv':
            print("  Computing Jaccord score...")
            jaccord_results = evaluate_file_jaccord(file)
            all_results[ds_name]['jaccord'] = jaccord_results['jaccord']
            all_results[ds_name]['jaccord_failed'] = jaccord_results['num_failed']
            all_results[ds_name]['jaccord_valid'] = jaccord_results['valid_samples']
            print(f"  Jaccord Score: {jaccord_results['jaccord']}")
        
        print(f"  {ds_name.upper()} Standard Metrics:")
        print(f"    BLEU: {standard_results['blue']}")
        print(f"    CHrF: {standard_results['chrf']}")
        print(f"    CER: {standard_results['cer']}")
        print(f"    WER: {standard_results['wer']}")
        print(f"    METEOR: {standard_results['meteor']}")
        print(f"    Edit Distance: {standard_results['ed']}")
    
    # Calculate averages
    datasets = list(all_results.keys())
    if datasets:
        avg_metrics = {}
        for metric in ['blue', 'chrf', 'cer', 'wer', 'meteor', 'ed']:
            avg_metrics[f'avg_{metric}'] = round(
                sum(all_results[ds][metric] for ds in datasets) / len(datasets), 2
            )
        
        print(f"\n{model_name} OVERALL AVERAGES:")
        print(f"  BLEU: {avg_metrics['avg_blue']}")
        print(f"  CHrF: {avg_metrics['avg_chrf']}")
        print(f"  CER: {avg_metrics['avg_cer']}")
        print(f"  WER: {avg_metrics['avg_wer']}")
        print(f"  METEOR: {avg_metrics['avg_meteor']}")
        print(f"  Edit Distance: {avg_metrics['avg_ed']}")
        
        # Dataset-specific scores
        if 'html' in all_results:
            print(f"  TEDS (HTML only): {all_results['html'].get('teds', 'N/A')}")
        if 'csv' in all_results:
            print(f"  Jaccord (CSV only): {all_results['csv'].get('jaccord', 'N/A')}")
        
        all_results['averages'] = avg_metrics
    
    # Save results
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{model_name}_complete_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    return {
        'model_name': model_name,
        'results': all_results
    }

def run_complete_evaluation():
    """Run complete evaluation on all models"""
    models = ["gpt-5-mini", "gpt-4.1-mini", "google"]
    
    all_results = []
    
    print("Starting COMPLETE evaluation for all models...")
    print("This includes standard metrics + TEDS for HTML + Jaccord for CSV")
    print(f"Models to evaluate: {models}")
    
    for model_name in models:
        result = evaluate_model_complete(model_name)
        if result:
            all_results.append(result)
    
    # Print final comparison
    print(f"\n{'='*90}")
    print("FINAL COMPLETE COMPARISON")
    print(f"{'='*90}")
    
    if all_results:
        # Header
        print(f"{'Model':<15} {'BLEU':<8} {'CHrF':<8} {'CER':<8} {'WER':<8} {'METEOR':<8} {'TEDS':<8} {'Jaccord':<8}")
        print("-" * 85)
        
        for result in all_results:
            model_name = result['model_name']
            results = result['results']
            
            if 'averages' in results:
                avg = results['averages']
                teds_score = results.get('html', {}).get('teds', 'N/A')
                jaccord_score = results.get('csv', {}).get('jaccord', 'N/A')
                
                print(f"{model_name:<15} "
                      f"{avg['avg_blue']:<8.2f} "
                      f"{avg['avg_chrf']:<8.2f} "
                      f"{avg['avg_cer']:<8.2f} "
                      f"{avg['avg_wer']:<8.2f} "
                      f"{avg['avg_meteor']:<8.2f} "
                      f"{teds_score if teds_score != 'N/A' else 'N/A':<8} "
                      f"{jaccord_score if jaccord_score != 'N/A' else 'N/A':<8}")
        
        # Save combined results
        with open("metrics/complete_comparison.json", "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nComplete results saved in 'metrics/' directory")
        print(f"Combined comparison saved in 'metrics/complete_comparison.json'")
    else:
        print("No results to compare - no model files found!")

if __name__ == "__main__":
    # Run complete evaluation on all models
    run_complete_evaluation()
    
    # You can also run individual models:
    # evaluate_model_complete("google")
    # evaluate_model_complete("gpt-5-mini")