import argparse
import json
from pathlib import Path

import numpy as np
import torch
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
from accelerate.utils.modeling import get_balanced_memory


def eval_awq(model, tokenizer, model_id):
    lm_eval_model = LMEvalAdaptor(model_id, model, tokenizer, 1)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=['wikitext'],
        batch_size=1,
        no_cache=True,
    )
    return evaluator.make_table(results)


def map_model_to_device(model):
    kwargs = {"max_memory": get_balanced_memory(model)}
    for key in list(kwargs['max_memory'].keys()):
        if str(key) == '0':
            kwargs['max_memory'][key] = int(kwargs['max_memory'][key] * 0.75)
        elif str(key).isdigit():
            kwargs['max_memory'][key] = int(kwargs['max_memory'][key] * 0.8)
    device_map = infer_auto_device_map(
        model,
        # TODO: can we remove this?
        no_split_module_classes=[
            "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
        **kwargs
    )
    print(device_map, flush=True)
    model = dispatch_model(model, device_map=device_map)
    return model


def eval_by_awq_metric(model, tokenizer, model_id):
    model = map_model_to_device(model)
    awq_result = eval_awq(model, tokenizer, model_id)
    print('*' * 10)
    print('[awq eval pipeline]', awq_result)
    return awq_result
