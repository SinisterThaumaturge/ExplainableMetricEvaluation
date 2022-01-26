# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import random
import itertools
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from . import modeling
from . configuration_bert import BertConfig
from . modeling import BertForMaskedLM
from . tokenization_bert import BertTokenizer
from . tokenization_utils import PreTrainedTokenizer
from . modeling_utils import PreTrainedModel


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, translations,sources, offsets=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.translations = translations
        self.sources = sources
        self.offsets = offsets

    def process_line(self, worker_id, sent_src, sent_tgt):

        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids']
        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt) 

    def __iter__(self):
        
        worker_id = 0

        for src, trans in list(zip(self.sources, self.translations)):
            processed = self.process_line(worker_id, src, trans)
            if processed is None:
                print(f'Line "{line.strip()}" (offset in bytes: {f.tell()}) is not in the correct format. Skipping...')
                empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                empty_sent = ''
                yield (worker_id, empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
            else:
                yield processed



def word_align(device, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, sources, translations):

    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    dataset = LineByLineTextDataset(tokenizer, sources, translations)
    dataloader = DataLoader(
        dataset, batch_size=32, collate_fn=collate, num_workers=1
    )

    model.to(device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")

    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, device, 0, 0, align_layer=8, extraction='softmax', softmax_threshold=0.01,test=True, output_prob=False)
            for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                output_str = []
                for word_align in word_aligns:
                    if word_align[0] != -1:
                        print(f'{word_align[0]}-{word_align[1]}')
                        print((f'{sent_src[word_align[0]]}<sep>{sent_tgt[word_align[1]]}'))
            tqdm_iterator.update(len(ids_src))


def create_alignments(sources, translations):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer

    config = config_class()

    tokenizer = tokenizer_class.from_pretrained('bert-base-multilingual-cased')
    

    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id

    model = model_class(config=config)

    word_align(device, model, tokenizer, sources, translations)

