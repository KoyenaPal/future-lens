import torch
import argparse
import os
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoTokenizer
from customized import CustomizedGPTJForCausalLM
from data import SoftPromptDataModule
from module_updated import GPTJModule
from pytorch_lightning.loggers import TensorBoardLogger
import json


def train(args):
    torch_dtypes = {
        "fp32": (torch.float32, 32),
        "fp16": (torch.float16, "16-mixed"),
        "bf16": (torch.bfloat16, "bf16-mixed")
    }

    if torch.cuda.get_device_name in ["NVIDIA H100 PCIe", "NVIDIA A100 80GB PCIe"]:
        torch.set_float32_matmul_precision('medium')

    torch_dtype, training_precision = torch_dtypes[args.precision]

    if "gpt-j-6b" not in args.model_name_or_path:
        raise ValueError("This script is only for GPT-J-6B.")
    
    logger = TensorBoardLogger(save_dir=args.output_path, name="logs")
    model = CustomizedGPTJForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
        model.resize_token_embeddings(len(tokenizer))

    data_module = SoftPromptDataModule(
        model = model,
        tokenizer = tokenizer, 
        train_file_path = args.train_set, 
        training_examples = args.training_examples,
        train_batch_size = args.batch_size, 
        max_length = args.max_length,
        next_token_skip = args.next_token_skip,
        in_layer = args.in_layer,
        out_layer = args.out_layer
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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=args.epochs,
        logger=logger,
        enable_checkpointing=False,
        precision=training_precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=200
    )

    logger.log_hyperparams(vars(args))
    trainer.fit(model, train_dataloaders=data_module.train_dataloader())
    model.save(os.path.join(args.output_path, "soft_prefix.pt"))

    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--model_name_or_path', default="EleutherAI/gpt-j-6b")
    parser.add_argument('--output_path', default=None, type=str)
    parser.add_argument('--train_set', default="./data/training_data_teacher_100000.csv")
    parser.add_argument('--training_examples', default=100000, type=int)

    # Experiment Set-ups
    parser.add_argument('--prefix_length', default=10, type=int, choices=[10, 30])
    parser.add_argument('--next_token_skip', default=[2, 3], type=int, nargs="+") # 0 for the currently predicted token, 1 for the next ...
    parser.add_argument('--in_layer', default=14, type=int)
    parser.add_argument('--tgt_in_layer', default=7, type=int)
    parser.add_argument('--out_layer', default=27, type=int)

    # Hyperparameters
    parser.add_argument('--precision', default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_length', default=2048, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int)

    # Seed
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if args.output_path is None:
        tk = "".join([str(i) for i in args.next_token_skip])
        args.output_path = "./results/training/layer{}to{}_tk{}".format(args.in_layer, args.tgt_in_layer, tk)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    seed_everything(args.seed)
    train(args)


if __name__ == "__main__":
    main()
