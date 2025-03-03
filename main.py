import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def to_encoded_array(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        array = tensor_or_array.cpu().contiguous().numpy()
    else:
        array = tensor_or_array
    shape = array.shape
    encoded_array = np.packbits(array.reshape(-1))
    return encoded_array, list(shape)


def from_encoded_array(encoded_array, shape):
    array = np.unpackbits(encoded_array)
    tensor = torch.from_numpy(array)
    return tensor[:np.prod(shape)].reshape(shape)


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        # device_map="auto" # avoids loading to gpu at this moment
    )

    model.seqlen = 2048
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    from collections import defaultdict
    model.hf_device_map = defaultdict(lambda: 'cpu')
    # if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    print("use device ", device)

    sparsity_ratio_original = check_sparsity(model, report_n_layers=5) # only check 5 layers to save time
    print('original sparsity: ', sparsity_ratio_original)

    mask_dict = None
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            mask_dict = prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if mask_dict is not None:
        print('Saving mask dict...')
        encoded_mask_dict = {}
        for k, v in mask_dict.items():
            encoded_mask_dict[k] = to_encoded_array(v)
        torch.save(encoded_mask_dict, os.path.join(args.save, 'mask_dict.encoded_bytes.pt'))

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################

    try:
        ppl = eval_ppl(model, tokenizer, device)
        print(f"ppl on wikitext {ppl}")
        save_filepath = os.path.join(args.save, "log.txt")
        with open(save_filepath, "w") as f:
            print("actual_sparsity\tppl", file=f, flush=True)
            print(f"{sparsity_ratio:.4f}\t{ppl:.4f}", file=f, flush=True)
    except Exception as e:
        print(e)

    try:
        from eval_by_awq_metric import eval_by_awq_metric
        import json
        awq_result = eval_by_awq_metric(model, tokenizer, args.model)
        save_filepath = os.path.join(args.save, "eval_by_awq_metrics.json")
        with open(save_filepath, "w") as f:
            json.dump(awq_result, f, indent=2)
    except Exception as e:
        print(e)

    if args.save_model:
        print('Saving model...')
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
