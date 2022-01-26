# %%
import transformers
model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# %%
import pandas as pd
import numpy as np
import json
import tqdm
import os
import truecase
from mosestokenizer import MosesDetokenizer, MosesTokenizer

from Metrics.mt_utils import (find_corpus, 
                      load_data, 
                      load_metadata, 
                      print_sys_level_correlation, 
                      print_seg_level_correlation,
                      df_append)
import torch

# %%
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from math import log

# %%
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
import itertools
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  return (cosine_similarity(X, Y) + 1.0) / 2.0
# pre-processing

device = torch.device('cuda')
def create_awesome_data(src, trans):

  sent_src, sent_tgt = src.strip().split(), trans.strip().split()
  token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
  wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
  ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
  
  ids_src, ids_tgt = ids_src.to(device), ids_tgt.to(device)
  sub2word_map_src = []
  for i, word_list in enumerate(token_src):
    sub2word_map_src += [i for x in word_list]
  sub2word_map_tgt = []
  for i, word_list in enumerate(token_tgt):
    sub2word_map_tgt += [i for x in word_list]

  # alignment
  align_layer = 8
  threshold = 1e-3
  model.to(device)
  model.eval()
  with torch.no_grad():
    out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
    out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

    dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

    softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
    softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

    softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

  align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
  cos = get_similarity(out_src.cpu(), out_tgt.cpu())
  sims = []
  for i, j in align_subwords:
     sims.append( float(cos[i][j]))
  
  return  {"awesome_align_sim_scores" : sims, "align_subwords" : align_subwords.tolist() }

# %%
include_path = 'Metrics/WMT17/'
data = {}
import json
done =  []
dataset = find_corpus("WMT17")
for pair in dataset.items():
    reference_path, lp = pair
    if lp in done:
      continue
    references = load_data(os.path.join( include_path +'references/', reference_path))
    src, tgt = lp.split('-')
    source_path = reference_path.replace('ref', 'src')
    source_path = source_path.split('.')[0] + '.' + src  
    source = load_data(os.path.join(include_path +'source', source_path))
    all_meta_data = load_metadata(os.path.join(include_path + 'system-outputs', lp))
    with MosesDetokenizer(src) as detokenize:        
        source = [detokenize(s.split(' ')) for s in source]         
    with MosesDetokenizer(tgt) as detokenize:                
        references = [detokenize(s.split(' ')) for s in references]
    trans = []
    for i in range(len(all_meta_data)):
        path, testset, lp, system = all_meta_data[i]
        
        translations = load_data(path)        
        num_samples = len(references)

        with MosesDetokenizer(tgt) as detokenize:                    
            translations = [detokenize(s.split(' ')) for s in translations]
        translations = [truecase.get_true_case(s) for s in translations]
        scores = []
        for j in tqdm.tqdm(range(len(translations))):
          if not len(translations[j]) == 0:
            scores.append(create_awesome_data(source[j], translations[j]))
          else:
            scores.append( {"awesome_align_sim_scores" : [0], "align_subwords" : [0,0] })

        data[f"{testset}-{lp}-{system}"] = scores
    with open("Results17/awesomealign"+lp+".json", "w") as f:
      json.dump(data, f)




