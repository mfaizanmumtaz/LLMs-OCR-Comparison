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
import matplotlib.pyplot as plt

# TEDS imports and classes
import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """A parallel version of the map function with a progress bar."""
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        
        kwargs = {'total': len(futures), 'unit': 'it', 'unit_scale': True, 'leave': True}
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    
    out = []
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

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
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
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
    '''Tree Edit Distance based Similarity'''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greater than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        '''Tokenizes table cells'''
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
        '''Converts HTML tree to the format required by apted'''
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
        '''Computes TEDS score between prediction and ground truth'''
        if (not pred) or (not true):
            return 0.0
        try:
            parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
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
        '''Computes TEDS score for a batch of samples'''
        if self.n_jobs == 1:
            scores = [self.evaluate(pred, gt) for pred, gt in tqdm(zip(pred_json, true_json), desc="Computing TEDS")]
        else:
            inputs = [{'pred': pred, 'true': gt} for pred, gt in zip(pred_json, true_json)]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores

def flatten_table(html_content):
    """Flatten table structure by moving all rows to tbody"""
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
    
    # Handle direct tr children of table
    if table:
        for row in table.find_all('tr'):
            new_tbody.append(row.extract())
        table.extract()
    
    return str(new_table.prettify())

def arabic_to_english_numerals(input_str):
    """Convert Arabic-Indic digits to Western Arabic digits"""
    arabic_numerals = '٠١٢٣٤٥٦٧٨٩'
    english_numerals = '0123456789'
    translation_table = str.maketrans(arabic_numerals, english_numerals)
    return input_str.translate(translation_table)

def clean_html_tags(table_html):
    """Keep only allowed HTML tags for tables"""
    allowed_tags = {"table", "tr", "th", "tbody", "td", "thead", "tfoot"}
    soup = BeautifulSoup(table_html, "html.parser")
    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()
    return str(soup)

def aug_html(s):
    """Augment and clean HTML string"""
    if not s or s.strip() == "":
        return "<html><body><table><tbody></tbody></table></body></html>"
    
    s = s.replace('dir=\"&lt;built-in function dir&gt;\"', "")
    s = s.replace('dir=\"<built-in function dir>\"', "")
    s = s.replace("border=\"1\"", "")
    s = s.replace("html\n", "")
    s = re.sub(r"\s+", ' ', s)
    s = arabic_to_english_numerals(s)
    s = flatten_table(s)
    s = clean_html_tags(s)
    return f"<html>\n<body>\n{s}\n</body></html>"

def preprocess_arabic_text(text: str) -> str:
    """Preprocess Arabic text by normalizing and cleaning"""
    text = text.replace("<image>", "")
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
    # Remove tatweel and normalize whitespace
    text = re.sub(r'(ـ+)|\s+', lambda m: ' ' if m.group(0).isspace() else '', text).strip()
    text = ' '.join(text.split())
    return text

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Compare two DataFrames cell by cell and return similarity score"""
    all_index = df1.index.union(df2.index)
    all_columns = df1.columns.union(df2.columns)
    df1_aligned = df1.reindex(index=all_index, columns=all_columns)
    df2_aligned = df2.reindex(index=all_index, columns=all_columns)
    comparison = (df1_aligned == df2_aligned) | (df1_aligned.isna() & df2_aligned.isna())
    score = comparison.sum().sum() / comparison.size
    return score

def safe_to_df(s: str) -> pd.DataFrame:
    """Safely convert CSV string to DataFrame with error handling"""
    s = arabic_to_english_numerals(s)
    csv_data = StringIO(s)
    try:
        df = pd.read_csv(csv_data)
    except pd.errors.ParserError:
        csv_data.seek(0)
        df = pd.read_csv(csv_data, engine='python', on_bad_lines='skip')
    
    df.columns = df.columns.astype(str)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)
    
    return df

def avg(l):
    """Calculate average, handling empty lists"""
    return 0 if len(l) == 0 else round(sum(l) / len(l), 2)

def _entry_to_html(entry):
    """Normalize an entry to an HTML string"""
    if isinstance(entry, str):
        return entry.replace("html\n", "")
    
    if isinstance(entry, dict):
        if 'html' in entry and isinstance(entry['html'], str):
            s = entry['html']
        else:
            text = entry.get('text', '') or ''
            tables = ''
            if isinstance(entry.get('tables', None), list):
                tables = ''.join([t for t in entry.get('tables') if isinstance(t, str)])
            s = f"{text}{tables}"
        return s.replace("html\n", "")
    
    return str(entry)

def evaluate_file_standard(file_path):
    """Standard evaluation metrics (BLEU, CHrF, CER, WER, METEOR, ED)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
   
    references = []
    hypotheses = []
 
    metrics = {
        'edit_distance': [],
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
        'bleu': round(corpus_bleu(hypotheses, [references]).score, 2),
        'chrf': round(corpus_chrf(hypotheses, [references]).score, 2),
        'ed': round(avg(metrics['edit_distance']), 2),
        'cer': round(cr(hypotheses, references).item(), 2),
        'wer': round(wr(hypotheses, references).item(), 2),
        'meteor': round(avg(metrics['meteor']), 2),
    }

def evaluate_file_structured_format(file_path):
    """Evaluation for structured format with text and tables (like your document format)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    teds = TEDS(n_jobs=32)
    all_teds_scores = []
    pred_texts = []
    gt_texts = []
    
    print(f"Evaluating structured format: {file_path}")
    
    for sample in tqdm(data, desc="Processing structured samples"):
        # Handle text evaluation
        pred_entry = sample.get('pred', {})
        gt_entry = sample.get('gt', {})
        
        # Extract text content
        if isinstance(pred_entry, dict) and 'text' in pred_entry:
            pred_text = preprocess_arabic_text(pred_entry['text'])
        else:
            pred_text = preprocess_arabic_text(str(pred_entry))
        
        if isinstance(gt_entry, dict) and 'text' in gt_entry:
            gt_text = preprocess_arabic_text(gt_entry['text'])
        else:
            gt_text = preprocess_arabic_text(str(gt_entry))
        
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)
        
        # Handle table evaluation
        gt_tables = gt_entry.get('tables', []) if isinstance(gt_entry, dict) else []
        pred_tables = pred_entry.get('tables', []) if isinstance(pred_entry, dict) else []
        
        if gt_tables and pred_tables:
            # Compare each GT table with each pred table and take max score per GT table
            max_scores = {}
            
            for gt_idx, gt_table in enumerate(gt_tables):
                max_scores[str(gt_idx)] = 0
                for pred_idx, pred_table in enumerate(pred_tables):
                    try:
                        gt_html = aug_html(preprocess_arabic_text(gt_table))
                        pred_html = aug_html(preprocess_arabic_text(pred_table))
                        score = teds.evaluate(pred_html, gt_html)
                        max_scores[str(gt_idx)] = max(max_scores[str(gt_idx)], score)
                    except Exception as e:
                        print(f"Error evaluating table {gt_idx}.{pred_idx}: {e}")
            
            # Calculate average score for this sample
            sample_avg = avg(list(max_scores.values()))
            if sample_avg is not None:
                all_teds_scores.append(sample_avg)
    
    # Calculate CHrF for text
    chrf_score = corpus_chrf(pred_texts, [gt_texts]).score
    
    # Calculate average TEDS for tables
    teds_avg = avg(all_teds_scores) * 100 if all_teds_scores else 0
    
    # Calculate combined average
    all_average = (chrf_score + teds_avg) / 2
    
    return {
        'chrf_text': round(chrf_score, 2),
        'teds_table': round(teds_avg, 2),
        'all_average': round(all_average, 2)
    }

def evaluate_file_jaccord(file_path):
    """Jaccord evaluation for CSV tables"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_score = 0
    num_failed = 0
    bad_samples = []
    valid_samples = 0
    
    for item in tqdm(data, desc="Computing Jaccord"):
        try:
            gt_text = item['gt']
            pred_text = item['pred'] if item['pred'] is not None else ""
            
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
            if "No columns to parse" not in str(e):
                print(f"Skipping sample {item['idx']}: {str(e)}")
                num_failed += 1
    
    avg_score = (total_score / valid_samples * 100) if valid_samples > 0 else 0
    
    return {
        'jaccord': round(avg_score, 2),
        'num_failed': num_failed,
        'valid_samples': valid_samples
    }

def detect_file_format(file_path):
    """Detect the format of the JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return 'empty'
    
    first_item = data[0]
    
    # Check if it's structured format (has text and tables)
    if isinstance(first_item.get('gt'), dict) and 'text' in first_item['gt']:
        return 'structured'
    
    # Check if it's CSV format (gt is plain text that looks like CSV)
    gt_content = first_item.get('gt', '')
    if isinstance(gt_content, str) and (',' in gt_content or '\t' in gt_content):
        # Simple heuristic: if it has commas/tabs and multiple lines, treat as CSV
        if len(gt_content.split('\n')) > 1:
            return 'csv'
    
    # Default to standard format
    return 'standard'

def evaluate_model_complete(model_name):
    """Complete evaluation including all metrics based on file format detection"""
    print(f"\n{'='*60}")
    print(f"Complete Evaluation for: {model_name}")
    print(f"{'='*60}")
    
    files = glob(f"results/{model_name}*.json")
    
    if not files:
        print(f"No result files found for model: {model_name}")
        return None
    
    print(f"Found files: {files}")
    
    all_results = {}
    
    for file in files:
        file_name = os.path.basename(file)
        # Try to extract dataset name from filename
        if '_' in file_name:
            ds_name = file_name.replace(".json", "").split("_")[-1]
        else:
            ds_name = file_name.replace(f"{model_name}_", "").replace(".json", "")
        
        print(f"\nProcessing dataset: {ds_name}")
        
        # Detect file format
        file_format = detect_file_format(file)
        print(f"  Detected format: {file_format}")
        
        all_results[ds_name] = {'dataset': ds_name, 'format': file_format}
        
        if file_format == 'structured':
            # Use structured evaluation (text + tables)
            structured_results = evaluate_file_structured_format(file)
            all_results[ds_name].update(structured_results)
            print(f"  CHrF (Text): {structured_results['chrf_text']}")
            print(f"  TEDS (Table): {structured_results['teds_table']}")
            print(f"  All Average: {structured_results['all_average']}")
            
        elif file_format == 'csv':
            # Use standard + Jaccord evaluation
            standard_results = evaluate_file_standard(file)
            jaccord_results = evaluate_file_jaccord(file)
            all_results[ds_name].update(standard_results)
            all_results[ds_name].update(jaccord_results)
            print(f"  Standard metrics: BLEU={standard_results['bleu']}, CHrF={standard_results['chrf']}")
            print(f"  Jaccord: {jaccord_results['jaccord']}")
            
        else:
            # Use standard evaluation only
            standard_results = evaluate_file_standard(file)
            all_results[ds_name].update(standard_results)
            print(f"  Standard metrics: BLEU={standard_results['bleu']}, CHrF={standard_results['chrf']}")
    
    # Save results
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{model_name}_complete_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    return {
        'model_name': model_name,
        'results': all_results
    }

def create_comparison_table(all_results):
    """Create comparison table and save as image (like your original script)"""
    
    # Extract results for table format
    comparison_data = {}
    
    for result in all_results:
        model_name = result['model_name']
        results = result['results']
        
        # Look for structured format results first (preferred)
        structured_results = None
        for ds_name, ds_results in results.items():
            if ds_results.get('format') == 'structured':
                structured_results = ds_results
                break
        
        if structured_results:
            comparison_data[model_name] = {
                "ChrF (Text)": structured_results.get('chrf_text', 0),
                "TEDS (Table)": structured_results.get('teds_table', 0),
                "All Average": structured_results.get('all_average', 0)
            }
        else:
            # Fallback to calculating averages from available metrics
            chrf_scores = [ds_results.get('chrf', 0) for ds_results in results.values() 
                          if 'chrf' in ds_results]
            jaccord_scores = [ds_results.get('jaccord', 0) for ds_results in results.values() 
                             if 'jaccord' in ds_results]
            
            avg_chrf = avg(chrf_scores) if chrf_scores else 0
            avg_jaccord = avg(jaccord_scores) if jaccord_scores else 0
            
            comparison_data[model_name] = {
                "ChrF (Text)": avg_chrf,
                "TEDS (Table)": avg_jaccord,  # Using Jaccord as table metric
                "All Average": (avg_chrf + avg_jaccord) / 2
            }
    
    # Create DataFrame and save as image
    df = pd.DataFrame(comparison_data).T
    df.index.name = "Models"
    
    # Create and save table image
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    rowLabels=df.index, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig("evaluation_results.png", bbox_inches="tight", dpi=300)
    print("Results saved as evaluation_results.png")
    print(df)
    
    return df

def run_complete_evaluation():
    """Run complete evaluation on all models"""
    models = ["google", "gpt-5-mini", "gpt-4.1-mini"]
    
    all_results = []
    
    print("Starting COMPLETE evaluation for all models...")
    print("This includes format detection and appropriate metrics")
    print(f"Models to evaluate: {models}")
    
    for model_name in models:
        result = evaluate_model_complete(model_name)
        if result:
            all_results.append(result)
    
    # Create comparison table
    if all_results:
        comparison_df = create_comparison_table(all_results)
        
        # Save combined results
        with open("metrics/complete_comparison.json", "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        
        print(f"\nComplete results saved in 'metrics/' directory")
        print(f"Combined comparison saved in 'metrics/complete_comparison.json'")
    else:
        print("No results to compare - no model files found!")

if __name__ == "__main__":
    run_complete_evaluation()