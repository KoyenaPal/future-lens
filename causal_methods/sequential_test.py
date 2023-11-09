from test import test
from customized import CustomizedGPTJForCausalLM
import torch
import os
from pytorch_lightning import seed_everything
from utils import TestArgs
import argparse


def init(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    seed_everything(args.seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_layer', default=0, type=int)
    args = parser.parse_args()

    model = CustomizedGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", torch_dtype=torch.bfloat16).cuda()
    in_layer = args.in_layer
    print("Inference on Layer {}".format(args.in_layer))

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[0], text_prefix=False, context_idx=0, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[1], text_prefix=False, context_idx=0, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[2], text_prefix=False, context_idx=0, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[3], text_prefix=False, context_idx=0, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[0], text_prefix=True, context_idx=0, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[0], text_prefix=True, context_idx=1, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[0], text_prefix=True, context_idx=2, metric="surprisal")
    init(args)
    test(args, model)

    args = TestArgs(in_layer=in_layer, out_layer=27, next_token_skip=[0], text_prefix=True, context_idx=3, metric="surprisal")
    init(args)
    test(args, model)

    
    
