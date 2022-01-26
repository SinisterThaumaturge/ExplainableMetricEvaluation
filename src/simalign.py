# %%
from Metrics.simalignmodified import SentenceAligner
import pandas as pd
import numpy as np
import json
import tqdm
import os
import truecase
from mosestokenizer import MosesDetokenizer
import json
from Metrics.mt_utils import (find_corpus, 
                      load_data, 
                      load_metadata, 
                      print_sys_level_correlation, 
                      print_seg_level_correlation,
                      df_append)
import torch

# %%
simaligner = SentenceAligner(matching_methods="i", device=torch.device('cuda'))

# %%
include_path = 'Metrics/WMT18/'
wmt_xmoverscores = []
dataset = find_corpus("WMT18")
done = []
for pair in dataset.items():
    data = {}
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
    for i in tqdm.tqdm(range(len(all_meta_data))):
        path, testset, lp, system = all_meta_data[i]
        
        translations = load_data(path)        
        num_samples = len(references)

        with MosesDetokenizer(tgt) as detokenize:                    
            translations = [detokenize(s.split(' ')) for s in translations]
        
        translations = [truecase.get_true_case(s) for s in translations]
        scores = []
        for i in range(len(translations)):
          if not len(translations[i]) == 0:
            scores.append(simaligner.get_word_aligns(source[i], translations[i]))
          else:
            scores.append(([0], [0,0], [0]))
        data[f"{testset}-{lp}-{system}"] = scores
    with open("Results18/simalign" +lp+".json", "w") as f:
        json.dump(data, f)