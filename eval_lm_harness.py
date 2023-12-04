import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

import numpy as np
import transformers
from lm_eval import evaluator
from lm_eval.models.huggingface import AutoCausalLM, HuggingFaceAutoLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel


class LMEvalModel(AutoCausalLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device: Optional[Union[int, str]] = "cuda",
    ):
        super(HuggingFaceAutoLM, self).__init__()  # do the BaseLM init
        self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = model.config

        self._add_special_tokens = add_special_tokens
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.max_length

        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]


def eval_gsm8k(model, tokenizer, tasks=('gsm8k',), limit=None):
    lm_eval_model = LMEvalModel(model, tokenizer, batch_size=1)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=list(tasks),
        num_fewshot=0,
        batch_size=1,
        no_cache=True,
        limit=limit,
    )
    return {
        "task": "gsm8k",
        "limit": limit,
        'metric_name': 'accuracy',
        "metric": results['results']['gsm8k']['acc'],
        "details": results,
    }


def eval_piqa(model, tokenizer, tasks=('piqa',), limit=None):
    lm_eval_model = LMEvalModel(model, tokenizer, batch_size=1)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=list(tasks),
        num_fewshot=0,
        batch_size=1,
        no_cache=True,
        limit=limit,
    )
    return {
        "task": "piqa",
        "limit": limit,
        'metric_name': 'accuracy',
        "metric": results['results']['piqa']['acc'],
        "details": results,
    }


def eval_wikitext(model, tokenizer, tasks=('wikitext',), limit=None):
    lm_eval_model = LMEvalModel(model, tokenizer, batch_size=1)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=list(tasks),
        num_fewshot=0,
        batch_size=1,
        no_cache=True,
        limit=limit,
    )
    return {
        "task": "wikitext",
        "limit": limit,
        'metric_name': 'perplexity',
        "metric": results['results']['wikitext']['word_perplexity'],
        "details": results,
    }
