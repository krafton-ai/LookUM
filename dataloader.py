import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile
import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers
import utils
LOGGER = utils.get_logger(__name__)

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(" \\'(\\w+)", "'\\1", x)
    x = re.sub(' (\\w+) \\. ', ' \\1. ', x)
    x = re.sub(' (\\w+) \\.$', ' \\1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(' \\?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(' \\!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub('\\" ([^\\"]+) \\"', '"\\1"', x)
    x = re.sub("\\' ([^\\']+) \\'", "'\\1'", x)
    x = re.sub('\\( ([^\\(\\)]+) \\)', '(\\1)', x)
    x = re.sub('\\[ ([^\\[\\]]+) \\]', '[\\1]', x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x

def _group_texts(examples, block_size, bos, eos):
    concatenated_examples = list(itertools.chain(*examples['input_ids']))
    total_length = len(concatenated_examples)
    new_block_size = block_size - 2
    total_length = total_length // new_block_size * new_block_size
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append([bos] + concatenated_examples[i:i + new_block_size] + [eos])
        _attn_masks.append(torch.ones(block_size))
    result['input_ids'] = _values
    result['attention_mask'] = _attn_masks
    return result

def get_dataset(dataset_name, tokenizer, wrap, mode, cache_dir, insert_eos=True, block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False):
    eos_tag = '' if insert_eos else '_eosFalse'
    if wrap:
        filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
    else:
        filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat'
    _path = os.path.join(cache_dir, filename)
    if utils.fsspec_exists(_path):
        LOGGER.info(f'Loading data from: {_path}')
        return datasets.load_from_disk(_path).with_format('torch')
    LOGGER.info(f'Generating new data at: {_path}')
    if dataset_name == 'openwebtext-train':
        dataset = datasets.load_dataset('openwebtext', split='train[:-100000]', cache_dir=cache_dir, streaming=streaming, download_config=datasets.DownloadConfig(resume_download=True, max_retries=20))
    else:
        dataset = datasets.load_dataset(dataset_name, split=mode, cache_dir=cache_dir, streaming=streaming)
    if isinstance(dataset, datasets.DatasetDict):
        data = dataset[mode]
    else:
        data = dataset
    if dataset_name == 'lm1b':
        detokenizer = lm1b_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):

        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok
    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        text = example['text']
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        if wrap:
            tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            if insert_eos:
                tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
        else:
            tokens = tokenizer(text, max_length=block_size, padding='max_length', truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True)
        return tokens
    if streaming:
        tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, desc='Tokenizing')
    else:
        tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True, desc='Tokenizing')
    tokenized_dataset = tokenized_dataset.remove_columns('text')
    if not wrap:
        tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format('torch')
    group_texts = functools.partial(_group_texts, block_size=block_size, bos=BOS, eos=EOS)
    if streaming:
        chunked_dataset = tokenized_dataset.map(group_texts, batched=True, desc='Grouping')
    else:
        group_num_proc = min(num_proc, max(1, len(tokenized_dataset) // 100))
        chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=group_num_proc, load_from_cache_file=True, desc='Grouping')
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format('torch')
    return chunked_dataset

def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.data.tokenizer_name_or_path)
    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(tokenizer, transformers.GPT2Tokenizer):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing((tokenizer.bos_token, tokenizer.bos_token_id), (tokenizer.eos_token, tokenizer.eos_token_id))
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(f'Tokenizer must have a bos_token or cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(f'Tokenizer must have a eos_token or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def get_train_dataloader(config, tokenizer):
    devices = config.trainer.devices
    expected = config.loader.batch_size * config.trainer.num_nodes * devices * config.trainer.accumulate_grad_batches
    if config.loader.global_batch_size != expected:
        raise ValueError(f'Expected global batch size {expected}, got {config.loader.global_batch_size}.')
    train_set = get_dataset(config.data.train, tokenizer, mode='train', wrap=config.data.wrap, insert_eos=getattr(config.data, 'insert_train_eos', True), cache_dir=config.data.cache_dir, block_size=config.model.length, num_proc=config.loader.num_workers or 1, streaming=config.data.streaming)
    loader = torch.utils.data.DataLoader(train_set, batch_size=config.loader.batch_size, num_workers=config.loader.num_workers, pin_memory=config.loader.pin_memory, shuffle=not config.data.streaming, drop_last=True, persistent_workers=config.loader.num_workers > 0)
    return loader

class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(), 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)
        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()
        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        for index in indices:
            self.counter += 1
            yield index
        self.counter = 0

class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        for index in indices:
            self.counter += 1
            yield index
        self.counter = 0
