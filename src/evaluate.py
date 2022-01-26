# Evaluate the JSON Data
import enum
import pandas as pd
import numpy as np
import json
import tqdm
import os
import truecase
from mosestokenizer import MosesDetokenizer
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from math import log
from Metrics.mt_utils import (find_corpus, 
                      load_data, 
                      load_metadata, 
                      print_sys_level_correlation, 
                      print_seg_level_correlation,
                      print_seg_level_correlation_wmt17,
                      output_MT_correlation,
                      df_append)
import torch
import json
year = "18"
include_path = 'Metrics/WMT18/'

import transformers
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def process(a, tokenizer):
 return tokenizer.encode(
                 a, add_special_tokens=True, max_length=tokenizer.model_max_length, truncation=True,
            )
            
def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.
    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict

# %%
from scipy.stats.mstats import gmean, hmean
def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )

operations = dict([
    ('mean', (lambda values: [np.mean(values, axis=0)])),
    ('max', (lambda values: [np.max(values, axis=0)])), 
    ('min', (lambda values: [np.min(values, axis=0)])),
    ('p_mean_2', (lambda values: [gen_mean(values, p=2.0).real])),
    ('p_mean_3', (lambda values: [gen_mean(values, p=3.0).real])),
    ("hmean", lambda values: [hmean(values)]),
    ("gmean", (lambda values: [gmean(values)]))
])

def create_token_ids(text):
  token_src = [tokenizer.tokenize(word) for word in text.split()]
  wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
  return list(chain(*wid_src))

# %%
dataset = find_corpus("WMT"+year)
def evaluate_results( operations):
  wmtscores = []
  for pair in dataset.items():
    reference_path, lp = pair
    references = load_data(os.path.join( include_path +'references/', reference_path))
    src, tgt = lp.split('-')
    source_path = reference_path.replace('ref', 'src')
    source_path = source_path.split('.')[0] + '.' + src  
    source = load_data(os.path.join(include_path +'source', source_path))
    all_meta_data = load_metadata(os.path.join(include_path + 'system-outputs', lp))
    with open(f"Results18/simalign{lp}.json") as f:
      data = json.load(f)
    with MosesDetokenizer(src) as detokenize:        
        source = [detokenize(s.split(' ')) for s in source]         
    with MosesDetokenizer(tgt) as detokenize:                
        references = [detokenize(s.split(' ')) for s in references]
        for i in range(len(all_meta_data)):
            path, testset, lp, system = all_meta_data[i]
            
            translations = load_data(path)        
            num_samples = len(references)

            with MosesDetokenizer(tgt) as detokenize:                    
                translations = [detokenize(s.split(' ')) for s in translations]
            translations = [truecase.get_true_case(s) for s in translations]

            idf_dict = get_idf_dict(source, tokenizer)
            ##data_idf_dict[f"{testset}-{lp}-{system}"] = get_idf_dict(translations, tokenizer)
            scores = {}
            for j in tqdm.tqdm(range(len(translations))):
              #ids = create_token_ids(source[j])
              score = []
              sim = data[f"{testset}-{lp}-{system}"][j][0]
              if data[f"{testset}-{lp}-{system}"][j][0] == 0:
                for k,r in enumerate(data[f"{testset}-{lp}-{system}"][j][1]):
                  for l, c in enumerate(r):
                    if data[f"{testset}-{lp}-{system}"][j][1][k][l] > 0:
                      score.append(sim[k][l])
                #calc_scores = np.array(data[f"{testset}-{lp}-{system}"][j]["awesome_align_sim_scores"])
                #weights = np.array([ idf_dict[ids[aligns[0]]] if idf_dict[ids[aligns[0]]] >= 0 else 1  for aligns in data[f"{testset}-{lp}-{system}"][j]["align_subwords"] ])
                #score = (calc_scores*weights) / weights.mean()
                for o in operations: 
                  if o not in scores:
                    scores[o] = []
                  scores[o].append(operations[o](score)[0])
              else:
                for o in operations: 
                  if o not in scores:
                    scores[o] = []
                  scores[o].append(0)
            for o in operations:
              wmtscores.append(df_append('simalign'+o, num_samples, lp, testset, system, scores[o]))
  for o in operations:     
    print_sys_level_correlation('simalign'+o, wmtscores, list(dataset.values()), os.path.join(f"Metrics/WMT{year}/", 'DA-syslevel.csv'))
    print_seg_level_correlation('simalign'+o, wmtscores, list(dataset.values()), os.path.join(f"Metrics/WMT{year}/", 'RR-seglevel.csv'))

evaluate_results(operations)