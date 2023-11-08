from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from customized import CustomizedGPTJForCausalLM
from data_updated import SoftPromptDataModule
from module_updated import GPTJModule
import torch
import argparse
import json
import os


def test(args, preloaded_model=None):
    torch_dtypes = {
        "fp32": (torch.float32, 32),
        "fp16": (torch.float16, "16-mixed"),
        "bf16": (torch.bfloat16, "bf16-mixed")
    }

    torch_dtype, training_precision = torch_dtypes[args.precision]

    if torch.cuda.get_device_name in ["NVIDIA H100 PCIe", "NVIDIA A100 80GB PCIe"]:
        print("Set float32 matmul precision to medium.")
        torch.set_float32_matmul_precision('medium')

    if "gpt-j-6b" not in args.model_name_or_path:
        raise ValueError("This script is only for GPT-J-6B.")

    model = CustomizedGPTJForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype).cuda() if preloaded_model is None else preloaded_model
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
        model.resize_token_embeddings(len(tokenizer))

    if not os.path.exists(args.prefix_path):
        if not args.text_prefix:
            print("Loading Prefix from {} ...".format(args.prefix_path))
            raise ValueError("Please specify a prefix path that exists")
    else:
        print("Loading Prefix from {} ...".format(args.prefix_path))
        model.load_prefix(args.prefix_path)

    model.eval()

    data_module = SoftPromptDataModule(
        model = model,
        tokenizer = tokenizer, 
        test_file_path = args.test_set, 
        test_batch_size = args.batch_size,
        max_length = args.max_length,
        next_token_skip = args.next_token_skip,
        in_layer = args.in_layer,
        out_layer = args.out_layer,
        metric = args.metric,
        max_n = args.max_n,
    )

    model = GPTJModule(
        model=model,
        tokenizer=tokenizer, 
        batch_size=args.batch_size,
        prefix_length=args.prefix_length, 
        in_layer = args.in_layer,
        tgt_in_layer = args.tgt_in_layer,
        out_layer = args.out_layer,
        next_token_skip = args.next_token_skip,
        max_n=args.max_n,
        output_path=args.output_path,
        text_prefix=args.text_prefix,
        context_idx=args.context_idx,
        metric=args.metric,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        enable_checkpointing=False,
        precision=training_precision,
    )

    trainer.test(model, dataloaders=data_module.test_dataloader())
    print("Done")


def main():
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--model_name_or_path', default="EleutherAI/gpt-j-6b")
    parser.add_argument('--prefix_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--test_set', default="./data/testing_data_teacher_1000.csv")

    # Model Set-ups
    parser.add_argument('--precision', default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_length', default=2048, type=int)

    # Experiment Set-upss
    parser.add_argument('--metric', default="calibrated", choices=["precision@k", "surprisal", "calibrated"])
    parser.add_argument('--text_prefix', default=False, type=bool)
    parser.add_argument('--context_idx', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--prefix_length', default=10, type=int, choices=[10, 30])
    parser.add_argument('--next_token_skip', default=[1], type=int, nargs="+") # 0 for the next token prediction
    parser.add_argument('--in_layer', default=13, type=int)
    parser.add_argument('--tgt_in_layer', default=13, type=int)
    parser.add_argument('--out_layer', default=27, type=int)

    parser.add_argument('--max_n', default=4, type=int)
    parser.add_argument('--top_k', default=1, type=int)

    # Seed
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    tk_list = "".join([str(i) for i in args.next_token_skip])
    if args.prefix_path is None:
        args.prefix_path = "./results/conll_training/layer{}to27_tk{}/soft_prefix.pt".format(args.in_layer, tk_list)
        #args.prefix_path = "./results/conll_training/layer{}to{}_tk{}/soft_prefix.pt".format(args.in_layer, args.tgt_in_layer, tk_list)
    
    if args.output_path is None:
        print(not args.text_prefix)
        context_type = "prefix" if not args.text_prefix else "ctx" + str(args.context_idx)
        print(context_type)
        args.output_path = "./results/temp/layer{}to{}_tk{}".format(args.in_layer, args.tgt_in_layer, tk_list)
        args.output_path = args.output_path + "_{}_{}".format(args.metric, context_type)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    seed_everything(args.seed)
    test(args)

if __name__ == "__main__":
    main()
