import torch
import re
from baukit import TraceDict
from datasets import Dataset
import os


def get_sub_hidden_states(configs, hidden_states):
    batch_hs = []
    for i, batch_configs in enumerate(configs):
        data_hs = []
        for token_idx, layer_idx in batch_configs:
            hs = hidden_states[layer_idx, i, token_idx, :]
            data_hs.append(hs)
        data_hs = torch.stack(data_hs)
        batch_hs.append(data_hs)
    return torch.stack(batch_hs)

def get_hidden_states(model, tokenizer, inputs, device):
    inputs = tokenizer(inputs, padding=True, max_length=2048, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    layer_names = [n for n, _ in model.named_modules() if re.match(r'^transformer.h.\d+$', n)]
    with TraceDict(model, layer_names) as tr:
        _ = model(**inputs)['logits']
    return torch.stack([tr[ln].output[0] for ln in layer_names])


class TestArgs:
    """
    This class is used to store the arguments for the model. (For testing only)
    """

    model_name_or_path="EleutherAI/gpt-j-6b"
    test_set="./data/testing_data_teacher_1000.csv"
    precision = "bf16"

    batch_size = 8
    max_length = 2048

    metric = "precision@k"
    
    text_prefix = False
    context_idx = 0
    prefix_length = 10

    max_n = 4
    top_k = 10

    seed = 42    

    def __init__(self, in_layer, out_layer, next_token_skip, context_idx=0, text_prefix=True, metric="precision@k"):
        self.in_layer = in_layer
        self.out_layer = out_layer
        self.next_token_skip = next_token_skip

        self.context_idx = context_idx
        self.text_prefix = text_prefix
        self.metric = metric

        context_type = "prefix" if not self.text_prefix else "ctx" + str(self.context_idx)

        tk_list = "".join([str(x) for x in self.next_token_skip])
        dirc = "layer{}to{}_tk{}".format(
            self.in_layer, self.out_layer, tk_list
        )

        test_dir = dirc + "_{}_{}".format(self.metric, context_type)

        self.prefix_path = os.path.join("./results/training", dirc, "soft_prefix.pt")
        self.output_path = os.path.join("./results/testing", test_dir)

        

class TrainArgs:
    """
    This class is used to store the arguments for the model. (For training only)
    """

    model_name_or_path="EleutherAI/gpt-j-6b"
    train_set="./data/training_data_100000.csv"
    precision = "bf16"

    batch_size = 2
    max_length = 2048
    
    epochs = 1
    lr = 5e-4
    weight_decay = 1e-5
    accumulate_grad_batches = 4

    seed = 42
    prefix_length = 10

    training_examples = 10000

    def __init__(self, in_layer, out_layer, next_token_skip):
        self.in_layer = in_layer
        self.out_layer = out_layer
        self.next_token_skip = next_token_skip

        token_list = "".join([str(x) for x in next_token_skip])

        dir = "layer{}to{}_tk{}".format(
            self.in_layer, self.out_layer, token_list
        )

        self.output_path = os.path.join("./results/training", dir)






    