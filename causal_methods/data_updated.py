import torch
import re
from baukit import TraceDict
from pytorch_lightning.utilities.data import DataLoader
from datasets import Dataset
import pytorch_lightning as pl
from utils import get_sub_hidden_states, get_hidden_states
import pandas as pd
import numpy as np


def load_dataset(data_path: str, max_examples=None):
    if data_path is None:
        return []
    
    df = pd.read_csv(data_path)
    df = df[df['decoded_phrase'].notna()]
    if max_examples is not None:
        df = df[:min(max_examples, len(df))]
        
    df = df.astype({'decoded_prefix':'string', 'teacher_phrase':'string'})
    prefixes = df['decoded_prefix'].to_list()
    phrases = df['teacher_phrase'].to_list()
    assert len(prefixes) == len(phrases)
    dataset = Dataset.from_dict({
        'prefix': prefixes,
        'phrase': phrases
    })
    return dataset

class SoftPromptDataModule(pl.LightningDataModule):
    
    def __init__(
            self, model, tokenizer, 
            train_file_path=None, 
            training_examples=None, 
            test_file_path=None, 
            train_batch_size=1, 
            test_batch_size=1, 
            max_length=2048, 
            next_token_skip=[0,1,4], # 0 for the currently decoded token
            in_layer=26, 
            out_layer=27,
            metric="precision@k",
            max_n=5,
        ):
        super(SoftPromptDataModule, self).__init__()

        self.model = model 
        self.tokenizer = tokenizer

        self._skip_token_num = next_token_skip
        self._tranplant_layer = in_layer
        self._target_layer = out_layer

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.train_set = load_dataset(train_file_path, training_examples)
        self.test_set = load_dataset(test_file_path)

        self.max_length = max_length
        self.load_raw_context = (metric != "precision@k")
        self.max_n = max_n

    def _get_configs(self, prefixes, phrases):
        texts = ["<|endoftext|>" + str(prefix) + str(phrase) for prefix, phrase in zip(prefixes, phrases)]
        tokenized_texts, tokenized_phrase = [self.tokenizer(x, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt") for x in [texts, phrases]]

        raw_length = torch.LongTensor([tokenized_texts["attention_mask"].shape[1]]).expand(tokenized_texts["attention_mask"].shape[0], -1)
        phrase_length = torch.sum(tokenized_phrase["attention_mask"], dim=-1).unsqueeze(1)
        end_positions = raw_length - phrase_length
        input_configs = list(map(lambda x: [(x.item(), self._tranplant_layer)], end_positions - 1))
        output_configs = list(map(lambda x: [(x.item() + skips, self._target_layer) for skips in self._skip_token_num], end_positions - 1))
        last_input_tokens_encoded = [tokenized_texts["input_ids"][i][input_configs[i][0][0]] for i in range(len(input_configs))]
        #last_input_tokens_encoded = [tokenized_texts["input_ids"][i] for i in range(len(input_configs))]
        #last_input_tokens = [self.tokenizer.decode(curr_tok) for curr_tok in last_input_tokens_encoded]
        return texts, input_configs, output_configs, last_input_tokens_encoded

    def _create_input_output_pair(self, texts, input_configs, output_configs):
        # hs.shape: [num_layers, batch_size, seq_len, hidden_size]
        hs = get_hidden_states(self.model, self.tokenizer, texts, device=self.model.device)
        assert len(input_configs) == len(output_configs) == hs.shape[1]
        if self._tranplant_layer == "Emb":
            inputs = self.tokenizer(texts, padding=True, max_length=2048, truncation=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            input_embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
            source_hs = []
            for i, batch_config in enumerate(input_configs):
                source_hs.append(input_embeddings[i, batch_config[0][0], :])
            source_hs = torch.stack(source_hs, dim=0)
            source_hs = source_hs.unsqueeze(1)
        else:
            source_hs = get_sub_hidden_states(input_configs, hs)
        target_hs = get_sub_hidden_states(output_configs, hs)
        return source_hs, target_hs
    
    def _create_input(self, texts, input_configs):
        hs = get_hidden_states(self.model, self.tokenizer, texts, device=self.model.device)
        assert len(input_configs) == hs.shape[1]
        if self._tranplant_layer == "Emb":
            inputs = self.tokenizer(texts, padding=True, max_length=2048, truncation=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            input_embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
            source_hs = []
            for i, batch_config in enumerate(input_configs):
                source_hs.append(input_embeddings[i, batch_config[0][0], :])
            source_hs = torch.stack(source_hs, dim=0)
            source_hs = source_hs.unsqueeze(1)
        else:
            source_hs = get_sub_hidden_states(input_configs, hs)
        return source_hs

    def train_collate_fn(self, batch):
        batch = [b.values() for b in batch]
        prefixes, phrases = list(zip(*batch))
        assert len(prefixes) == len(phrases)

        # Enter Normal GPT-J Mode
        self.model.unprefix_model()
        texts, input_configs, output_configs, _ = self._get_configs(prefixes, phrases)
        source_hs, target_hs = self._create_input_output_pair(texts, input_configs, output_configs)
        outputs = self.tokenizer(phrases, return_tensors="pt", truncation=True, max_length=max(self._skip_token_num)+1)
        labels = outputs["input_ids"]
        batch = {
            "source_hs": source_hs,
            "target_hs": target_hs,
            "labels": labels,
        }
        return batch
    
    def test_collate_fn(self, batch):
        batch = [b.values() for b in batch]
        prefixes, phrases = list(zip(*batch))
        assert len(prefixes) == len(phrases)

        # Enter Normal GPT-J Mode
        self.model.unprefix_model()

        texts, input_configs, _, last_input_tokens = self._get_configs(prefixes, phrases)  
        source_hs = self._create_input(texts, input_configs)
        outputs = self.tokenizer(phrases, return_tensors="pt", truncation=True, max_length=self.max_n+1)
        labels = outputs["input_ids"]
        # "last_input_tokens": last_input_tokens
        batch = {
            "source_hs": source_hs,
            "labels": labels,
            "last_input_tokens": prefixes
        }
        if self.load_raw_context:
            processed_prefixes = ["<|endoftext|>" + str(prefix) for prefix in prefixes]
            raw_contexts = self.tokenizer(processed_prefixes, return_tensors="pt", padding=True, max_length=self.max_length)
            batch["raw_input_ids"] = raw_contexts["input_ids"]
            batch["raw_attention_mask"] = raw_contexts["attention_mask"]
        
        return batch
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, collate_fn=self.train_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, collate_fn=self.test_collate_fn)





    

    



    

