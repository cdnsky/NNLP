import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

#additional special tokens for BERT
MENTION_START_MARKER = '[unused0]'
MENTION_END_MARKER = '[unused1]'

class Mewsli_Dataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer,
                 model_type: str,
                 context_size=32, #number of tokens to keep on the left and right of the mention
                 entity_context_size=32,
                 mapping=None):

        self.data_dir = data_dir
        self.context_size = context_size
        self.entity_context_size = entity_context_size
        self.tokenizer = tokenizer
        self.model_type = model_type
        data_file = os.path.join(data_dir, f'mentions_{split}.json')
        self.mapping=mapping

        with open(data_file) as f:
            self.mentions: List[Dict] = list(json.load(f).values())

    def __len__(self):
        return len(self.mentions)

    def _get_negative_sample(self, idx) -> Dict:
        ni = idx
        while ni == idx:
            ni = randint(0, self.__len__())
        return self.mentions[ni]

    def _get_mention_tokens(self, mention: Dict[str, Any]):
        start_i = mention['start_index']
        end_i = mention['end_index']
        mention_text = mention['mention_its'].lower()
        words = mention['source_document']['text'].lower().split()
        title = mention['source_document']['title'].lower()

        mention_tokens = [MENTION_START_MARKER] + self.tokenizer.tokenize(mention_text) + [MENTION_END_MARKER]
        left_tokens = self.tokenizer.tokenize(' '.join(words[:start_i]))
        right_tokens = self.tokenizer.tokenize(' '.join(words[end_i + 1:]))
        title_tokens = self.tokenizer.tokenize(title)

        keep_left = (self.context_size - 2 - len(mention_tokens)) // 2
        keep_right = (self.context_size - 2 - keep_left - len(mention_tokens))
        ctx_tokens = left_tokens[-keep_left:] + mention_tokens + right_tokens[:keep_right]
        ctx_tokens = ctx_tokens[:self.tokenizer.model_max_length - 2]
        ctx_tokens = ['[CLS]'] + title_tokens + ['[SEP]'] + ctx_tokens + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(ctx_tokens)
        attention_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += [0] * len(padding)

        assert len(input_ids) <= 512

        return torch.LongTensor(input_ids)

    def _get_entity_tokens(self, mention: Dict[str, Any]) -> Dict:

        if self.model_type=='E':
            q_id = mention['mention_id']
            input_ids = [self.mapping[q_id]]
            padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))
            input_ids += padding

        elif self.model_type=='F':
            text = mention['entity_context'].lower().split()

            text_tokens = self.tokenizer.tokenize(' '.join(text[:self.entity_context_size]))
            tokens = ['[CLS]'] + text_tokens[:self.tokenizer.model_max_length - 2] + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))
            input_ids += padding

        assert len(input_ids) <= 512
        return torch.LongTensor(input_ids)

    def __getitem__(self, idx):
        mention = self.mentions[idx]

        mention_inputs = self._get_mention_tokens(mention)
        entity_inputs = self._get_entity_tokens(mention)

        return {
            'qids': mention['mention_id'],
            'mention_inputs': mention_inputs,
            'entity_inputs': entity_inputs,
        }

class Mewsli_Entities_Dataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer,
                 model_type: str,
                 context_size=32,
                 entity_context_size=32,
                 mapping=None):

        self.data_dir = data_dir
        self.context_size = context_size
        self.entity_context_size = entity_context_size
        self.tokenizer = tokenizer
        self.model_type = model_type
        data_file = os.path.join(data_dir, f'mentions_{split}.json')
        self.mapping=mapping

        with open(data_file) as f:
            self.mentions: List[Dict] = list(json.load(f).values())

    def __len__(self):
        return len(self.mentions)

    def _get_entity_tokens(self, mention: Dict[str, Any]) -> Dict:

        if self.model_type=='E':
            q_id = mention['mention_id']
            input_ids = [self.mapping[q_id]]
            padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))
            input_ids += padding

        elif self.model_type=='F':
            text = mention['entity_context'].lower().split()

            text_tokens = self.tokenizer.tokenize(' '.join(text[:self.entity_context_size]))
            tokens = ['[CLS]'] + text_tokens[:self.tokenizer.model_max_length - 2] + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))
            input_ids += padding

        assert len(input_ids) <= 512
        return torch.LongTensor(input_ids)

    def __getitem__(self, idx):
        mention = self.mentions[idx]

        entity_inputs = self._get_entity_tokens(mention)

        return {
            'qids': mention['mention_id'],
            'entity_inputs': entity_inputs,
        }
