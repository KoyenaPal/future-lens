import torch
import argparse
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, LambdaLR
import torch, baukit
from baukit import TraceDict
import pandas as pd
import numpy as np
import re
import json
import wandb


torch.cuda.empty_cache()
wandb.login()

MODEL_NAME = None
TEACHER_MODEL = None
TEACHER_TOK = None
DECODER = None


def get_json_input(input_json_file):
    f = open(input_json_file)
    data = json.load(f)
    input_lists = []
    for i in data:
        input_lists.append(i)
    f.close()
    return input_lists


def get_robust_test_json(input_json_file):
    f = open(input_json_file)
    data = json.load(f)
    f.close()
    return data


def get_linear_model_test_input_original_results(full_pile_sample_file, robust_results_dir, token_pos=1, layer=26):
    full_pile_sample = get_robust_test_json(full_pile_sample_file)
    input_lists = []
    for curr_ind in range(len(full_pile_sample)):
        robust_result_json_file = robust_results_dir + "/" + \
        f"robustTest_InputID_{curr_ind}_from_pile_all_layers_[{layer}].json"
        curr_robust_dict = get_robust_test_json(robust_result_json_file)
        flattened_list_of_score_list = [item for sublist in curr_robust_dict['scores'] for item in sublist]
        result = [sum(items) for items in zip(*flattened_list_of_score_list)]
        if (result[token_pos] > 2):
            curr_sample = full_pile_sample[curr_ind]
            input_lists.append(curr_sample)
    return input_lists


def get_training_testing_data(input_json_file, testing_size=0.4, seed=51):
    random.seed(seed)
    input_lists = get_json_input(input_json_file)
    total_test_size = int(testing_size * len(input_lists))
    testing_indices = random.sample(range(len(input_lists)), total_test_size)
    training_data = []
    testing_data = []
    for curr_ind in range(len(input_lists)):
        if curr_ind in testing_indices:
            testing_data.append(input_lists[curr_ind])
        else:
            training_data.append(input_lists[curr_ind])
    training_data = pd.DataFrame(training_data)
    testing_data = pd.DataFrame(testing_data)
    return training_data, testing_data

def get_sub_hs(token_indices, ref_hs, layers=list(range(28))):
    """
    Function that gets the hidden state values of the token indices to use for the edit hs function in TraceDict. 
    
    Parameters:
        token_indices: token index values of hidden states we should take
        ref_hs: referenced hidden states that we will copy from
        layers: the set of layers we want to copy. 
    
    Returns:
        Replacement HS: replacement hidden states
    """
    saved_hs = []
    for layer in layers:
        layer_hs = []
        for token_index in token_indices:
            layer_hs.append(ref_hs[layer, 0, token_index])
        saved_hs.append(layer_hs)
    return saved_hs

def get_hidden_states(model, tok, prefix, device="cuda:0"):
    inp = {k: torch.tensor(v)[None].to(device) for k, v in tok(prefix).items()}
    layer_names = [n for n, _ in model.named_modules()
                    if re.match(r'^transformer.wte|^transformer.h.\d+$', n)]
    with TraceDict(model, layer_names) as tr:
        logits = model(**inp)['logits']
    return torch.stack([tr[ln].output[0][None,:] if ln=="transformer.wte" else tr[ln].output[0] for ln in layer_names])

def create_input_output_pair(model, tok, doc, input_config, output_config, device):
    """
    input_config = [token position, layer] ex. [10,transformer.h.26]
    output_config = [token position, layer] ex. [11,transformer.h.26]
    """   
    hs = get_hidden_states(model, tok, doc, device)
    source_hs = get_sub_hs(input_config[0], hs, layers=input_config[1])
    target_hs = get_sub_hs(output_config[0], hs, layers=output_config[1])
    return source_hs[0][0], target_hs[0][0]

class HiddenStateDataset(Dataset):
    def __init__(self, input_configs, output_configs, texts, device):
        self.input_configs = input_configs
        self.output_configs = output_configs
        self.texts = texts
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        total_text = self.texts[idx]
        curr_in_config = self.input_configs[idx]
        curr_out_config = self.output_configs[idx]
        source_hs, target_hs = create_input_output_pair(TEACHER_MODEL,
                                                        TEACHER_TOK,
                                                        total_text,
                                                        curr_in_config,
                                                        curr_out_config, self.device)
        total_text_tokens = TEACHER_TOK(total_text)['input_ids']
        wanted_token = total_text_tokens[curr_out_config[0][0]+1]
        sample = (source_hs, target_hs, wanted_token)
        return sample
    
class HiddenStateDatasetBatch(Dataset):
    def __init__(self, input_configs, output_configs, texts, device, npy_path, next_token_skip):
        self.input_configs = input_configs
        self.output_configs = output_configs
        self.texts = texts
        self.device = device
        self.npy_path = npy_path
        self.next_token_skip = next_token_skip

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.npy_path is None:
            return (self.input_configs[idx], self.output_configs[idx], self.texts[idx], 
                    self.device, None, self.next_token_skip)
        else:
            return (self.input_configs[idx], self.output_configs[idx], self.texts[idx], 
                    self.device, self.npy_path[idx], self.next_token_skip)
    
    
def dataset_collate_func(batch):
    source_hs_list = []
    target_hs_list = []
    wanted_token_list = []
    for idx in range(len(batch)):
        curr_in_config = batch[idx][0]
        curr_out_config = batch[idx][1]
        total_text = batch[idx][2]
        device = batch[idx][3]
        # should be a list later
        npy_path = batch[idx][4]
        next_token_skip = batch[idx][5]
        source_hs = None
        target_hs = None
        if npy_path is not None and os.path.exists(npy_path[0]):
            source_path = npy_path[0]
            target_path = npy_path[1]
            source_hs = torch.tensor(np.load(source_path)[curr_in_config[1]]).squeeze().cuda()
            target_hs = torch.tensor(np.load(target_path)[next_token_skip + 1]).cuda()
        else:
            source_hs, target_hs = create_input_output_pair(TEACHER_MODEL,
                                                            TEACHER_TOK,
                                                            total_text,
                                                            curr_in_config,
                                                            curr_out_config, device)
        
        total_text_tokens = TEACHER_TOK(total_text)['input_ids']
        wanted_token = total_text_tokens[curr_out_config[0][0]+1]
        source_hs_list.append(source_hs)
        target_hs_list.append(target_hs)
        wanted_token_list.append(torch.tensor(wanted_token))
    source_hs_list = torch.stack(source_hs_list)
    target_hs_list = torch.stack(target_hs_list)
    wanted_token_list = torch.stack(wanted_token_list)
    return (source_hs_list,target_hs_list, wanted_token_list)
    
def create_dataset(tok, data, next_token_skip=0, in_layer=26, out_layer=27, device='cuda:0', npy_paths=None):
    total_texts = []
    input_configs = []
    output_configs = []
    for _, curr_train in data.iterrows():
        total_text = "<|endoftext|>" + str(curr_train['decoded_prefix']) + str(curr_train['decoded_phrase'])
        total_text_tokens = tok(total_text)['input_ids']
        if len(total_text_tokens) > 2048:
            total_text = tok.decode(total_text_tokens[-2048:])
        actual_gen_len = len(tok(str(curr_train['decoded_phrase']))['input_ids'])
        # including +1s to the layer bc -1 is now considered as embedding layer
        input_config = [[-actual_gen_len - 1], [in_layer + 1]]
        output_config = [[-actual_gen_len + next_token_skip], [out_layer + 1]]          
        total_texts.append(total_text)
        input_configs.append(input_config)
        output_configs.append(output_config)
    dataset_obj = HiddenStateDatasetBatch(input_configs, output_configs, total_texts, 
                                          device, npy_paths, next_token_skip)
    return dataset_obj

def get_test_token_output(hs):
    probs = torch.nn.functional.softmax(DECODER(hs), dim=-1)
    _, favorite_tokens = probs.topk(k=1, dim=-1)
    return favorite_tokens 


def criterion(test_hs, expected_hs, T=1):

    p_s = F.log_softmax(DECODER(test_hs)/T, dim=-1)
    p_t = F.softmax(DECODER(expected_hs)/T, dim=-1)
    loss = (F.kl_div(p_s, p_t,reduction='batchmean')* (T**2))
    # so that the model doesn't aim for it to go to negatives
    total_loss = loss
    return total_loss

# Define a function to calculate the accuracy
def calculate_accuracy(outputs, targets, wanted_tokens):
    gt_correct = 0
    correct = 0
    wanted_token = wanted_tokens.item()
    pred = get_test_token_output(outputs).item()
    actual = get_test_token_output(targets).item()
    if pred == actual:
        correct += 1
    if pred == wanted_token:
        gt_correct += 1
    return correct, gt_correct

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output
    
def train(args, model_s, device, train_loader, test_loader,  optimizer, scheduler, epoch, highest_acc=None, highest_model_save=None):
    model_s.train()
    total_loss = 0.0
    correct = 0
    gt_correct = 0
    total_seen = 0
    optimizer.zero_grad()
    for batch_idx, (data, target, wanted_token) in enumerate(train_loader):
        data, target, wanted_token = data.to(device), target.to(device), wanted_token.to(device)
        total_seen += len(wanted_token)
        output_s = model_s(data).to(device)
        output_t = target
        loss_kd = criterion(output_s, output_t)
        loss = loss_kd
        loss.backward()
        total_loss += loss.item() * data.size(0)
        
        batch_train_seen = 0
        batch_correct = 0
        batch_gt_correct = 0
        if batch_idx % 5 == 0 and batch_idx > 0:
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        for i, output_set in enumerate(output_s):
                curr_correct, curr_gt_correct = calculate_accuracy(output_set, target[i], wanted_token[i])
                correct += curr_correct
                gt_correct += curr_gt_correct
            
                batch_correct += curr_correct
                batch_gt_correct += curr_gt_correct
                batch_train_seen += 1
        # Log metrics to wandb for each GPU set
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if "log_file" in args:
                log_file = args["log_file"]
                with open(log_file, 'a') as file:
                    file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\n'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    file.flush()
            args["test_results"] = os.path.join(args["experiment_data_dir"],
                                                f"linear_model_{display_name}_epoch_{epoch}_batch_idx_{batch_idx}_results.csv")

            test_loss, acc, gt_acc = test(args, model_s, device, test_loader)
            model_save_path = os.path.join(args["experiment_data_dir"],
                       f'model_trained_{display_name}_epoch_{epoch}_batch_idx_{batch_idx}.pth')
            if (highest_acc is None) or (highest_acc < acc):
                highest_acc = acc
                highest_model_save = model_save_path

            wandb.log({"test/loss": test_loss, "test/accuracy": acc,
                       "test/groundtruth_accuracy": gt_acc,
                       "test/train-epoch": epoch, "test/train-batches-seen": batch_idx + epoch*batch_idx,
                      "test/highest-acc": highest_acc, "test/highest_model_save": str(highest_model_save)})
            wandb.log({"train/loss": loss.item(), "train/accuracy": 100 * batch_correct / batch_train_seen,
                        "train/groundtruth_accuracy": 100 * batch_gt_correct / batch_train_seen,
                        "train/train-epoch": epoch, "train/train-batches-seen": batch_idx + epoch*batch_idx})
            torch.save(model_s.state_dict(), model_save_path)

            model_s.train()
            scheduler.step(test_loss)

    train_loss = total_loss / total_seen
    train_acc = 100 * correct / total_seen
    train_gt_acc = 100 * gt_correct / total_seen
    return train_loss, train_acc, train_gt_acc, highest_acc, highest_model_save

def train_new(args, model_s, train_loader, device):
    def calculate_linear_regression(X, Y, singular_value_threshold):
        # Calculate the means of X and Y
        X_mean = torch.mean(X, axis=0)
        Y_mean = torch.mean(Y, axis=0)
        # Center X and Y
        X_centered = X - X_mean
        Y_centered = Y - Y_mean        
        # Compute the SVD of centered X
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        # Apply the threshold to the singular values
        S_inv = torch.where(S > singular_value_threshold, 1.0/S, 0.0)
        num_fulfilled = torch.sum(S_inv.nonzero()).item()
        # Compute the pseudoinverse of centered X using the SVD
        X_centered_pinv = torch.matmul(Vt.T, torch.matmul(torch.diag(S_inv), U.T))
        # Compute A using least squares
        A = torch.matmul(X_centered_pinv, Y_centered)
        # Compute bias term b
        b = Y_mean - torch.matmul(X_mean, A)
        return A, b, num_fulfilled


    # Assuming X and Y are your data
    # my batch size is the full dataset
    for _, (data, target, wanted_token) in enumerate(train_loader):
        data, target, wanted_token = data.to(device), target.to(device), wanted_token.to(device)
        singular_value_threshold = 0.0
        A, b, dim_s_inv = calculate_linear_regression(data, target, singular_value_threshold)

        with torch.no_grad():
            model_s.fc.weight[...] = A.T
            model_s.fc.bias[...] = b

        display_name = args["display_name"]
        args["test_results"] = os.path.join(args["experiment_data_dir"],
                               f"linear_model_train_{display_name}_full_math_results.csv")
        
        #train_loss, train_acc, train_gt_acc = test(args, model_s, device, train_loader)
        saved_model_path = os.path.join(args["experiment_data_dir"],f'model_trained_{display_name}_math_sol.pth')
        torch.save(model_s.state_dict(), saved_model_path)
        return
        

def test(args, model, device, test_loader):
    model.eval()
    total_loss = 0
    gt_correct = 0
    correct = 0
    results = []
    with torch.no_grad():
        for _, (data, target, wanted_token) in baukit.pbar(enumerate(test_loader)):
            data, target, wanted_token = data.to(device), target.to(device), wanted_token.to(device)
            output_s = model(data).to(device)

            loss = criterion(output_s, target)
            total_loss += loss.item() * data.size(0)
            for i, output_set in enumerate(output_s):
                pred = get_test_token_output(output_set).item()
                actual = get_test_token_output(target[i]).item()
                curr_result = {"pred_tok": pred, "pred": TEACHER_TOK.decode(pred),
                        "actual_tok": actual, "actual": TEACHER_TOK.decode(actual),
                        "groundtruth_tok": wanted_token[i].item(), "groundtruth": TEACHER_TOK.decode(wanted_token[i])}
                results.append(curr_result)
                if pred == actual:
                    correct += 1
                if pred == wanted_token[i].item():
                    gt_correct += 1
    test_loss = total_loss / len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)
    gt_acc = 100 * gt_correct / len(test_loader.dataset)
    if "test_results" in args:
        csv_file = args["test_results"]
        result_df = pd.DataFrame(results)
        result_df.to_csv(csv_file)
    print('Total correct: {}\n'.format(correct))
    print('Total groundtruth correct: {}\n'.format(gt_correct))
    print('Test Loss: {}  Accuracy: {}%\n'.format(test_loss, acc))
    print('Groundtruth Accuracy: {}%\n'.format(gt_acc))
    if "log_file" in args:
        log_file = args["log_file"]
        with open(log_file, 'a') as file:
            file.write(f'Testing result at {args["log_file"]}\n')
            file.write('Total correct: {}\n'.format(correct))
            file.write('Total groundtruth correct: {}\n'.format(gt_correct))
            file.write('Test Loss: {}  Accuracy: {}%\n'.format(test_loss, acc))
            file.write('Groundtruth Accuracy: {}%\n'.format(gt_acc))
            file.flush()
    return test_loss, acc, gt_acc


def main(args):
    next_token_skip = args["next_token_skip"]
    in_layer = args["in_layer"]
    out_layer = args["out_layer"]
    # Set the device IDs for the GPUs you want to use
    #device_ids = list(range(1,torch.cuda.device_count()))
    device = args["cuda_device"]
    display_name = my_args["display_name"]
    weight_decay = my_args["weight_decay"]
    wandb.init(
        project = args["project_name"],
        name = display_name,
        config = args)
    training_data = pd.read_csv("../data/training_data_100000.csv")
    testing_data = pd.read_csv("../data/testing_data_1000.csv")

    # Loading data cache if available
    train_npy_dir = "../data/train_data_cache/"
    test_npy_dir = "../data/test_data_cache/"
    train_npy_paths = []
    test_npy_paths = []
    for i in range(len(training_data)):
        curr_source_npy_path = train_npy_dir + f"train_idx_{i}_source_layers.npy"
        curr_target_npy_path = train_npy_dir + f"train_idx_{i}_target_layer_next_skip_tokens_-1_to_2.npy"
        train_npy_paths.append([curr_source_npy_path,curr_target_npy_path])
    for i in range(len(testing_data)):
        curr_source_npy_path = test_npy_dir + f"train_idx_{i}_source_layers.npy"
        curr_target_npy_path = test_npy_dir + f"train_idx_{i}_target_layer_next_skip_tokens_-1_to_2.npy"
        test_npy_paths.append([curr_source_npy_path,curr_target_npy_path])
     
    # If there is cache, update npy_paths to train_npy_dir and test_npy_dir respectively
    train_data = create_dataset(TEACHER_TOK, training_data, next_token_skip=next_token_skip,
                                in_layer=in_layer, out_layer=out_layer, device=device, npy_paths=None)
    test_data = create_dataset(TEACHER_TOK, testing_data, next_token_skip=next_token_skip,
                               in_layer=in_layer, out_layer=out_layer, device=device, npy_paths=None)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args["experiment_data_dir"], exist_ok=True)
    batch_size = args["batch_size"]
    learning_rate = args["learning_rate"]
    linear_model = LinearModel(4096,4096)

    linear_model = linear_model.to(device)
    wandb.watch(linear_model)
    models = [linear_model]
 
    if args["test_only"]:
        test_loss, acc, gt_acc = test(args, models[0], device, test_loader)

    train_loader = DataLoader(dataset=train_data, batch_size = 100, collate_fn = dataset_collate_func, drop_last=True)
    train_new(args, models[0], train_loader, device)
    print(models[0].state_dict())
    del train_loader
    test_loader = DataLoader(dataset=test_data, batch_size = 100, collate_fn = dataset_collate_func, drop_last=True)
    test_loss, acc, gt_acc = test(args, models[0], device, test_loader)
    train_loader = DataLoader(dataset=train_data, batch_size = batch_size, collate_fn = dataset_collate_func, drop_last=True)

    # Hard coded optimizer value here
    optimizer = torch.optim.SGD(models[0].parameters(), lr=learning_rate*(1-0.9), momentum=0.9, nesterov=True, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    print('start fine-tuning...') 
    highest_acc = 0.0
    for epoch in range(args["fine_tune_epochs"]):
        print('# Epoch {} #'.format(epoch))
        train_loss, train_acc, train_gt_acc, highest_curr_acc, highest_curr_model_save = train(args, models[0], device, train_loader, test_loader, optimizer, scheduler, epoch, highest_acc)
        highest_acc = highest_curr_acc
        highest_model_save = highest_curr_model_save
        wandb.log({"train/end-loss": train_loss, f"train/end-accuracy": train_acc, f"train/end-groundtruth_accuracy": train_gt_acc, "train/end-epoch": epoch})
        args["test_results"] = os.path.join(args["experiment_data_dir"],
                                            f"linear_model_{display_name}_epoch_{epoch}_results.csv")
        test_loss, acc, gt_acc = test(args, models[0], device, test_loader)
        wandb.log({"test/loss": test_loss, "test/accuracy": acc, "test/groundtruth_accuracy": gt_acc, "test/epoch": epoch})
        torch.save(models[0].state_dict(), os.path.join(args["experiment_data_dir"],
                                                        f'model_trained_{display_name}_epoch_{epoch}.pth'))



    args["test_results"] = os.path.join(args["experiment_data_dir"],
                                        f"linear_model_{display_name}_epoch_{epoch}_complete_results.csv")
    torch.save(models[0].state_dict(), os.path.join(args["experiment_data_dir"], f'model_trained_{display_name}_complete.pth'))
    print('Model trained saved to %s' % args["experiment_data_dir"])
    test_loss, acc, gt_acc = test(args, models[0], device, test_loader)
    wandb.log({"test/loss": test_loss, "test/accuracy": acc, "test/groundtruth_accuracy": gt_acc, "test/epoch": args["fine_tune_epochs"]})
    wandb.finish()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--cuda_device', type=str, default='cuda',
                        help='path to the pretrained teacher model checkpoint')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='learning rate for training')
    parser.add_argument('--fine_tune_epochs', type=int, default=3,
                        help='epochs to fine tune')
    parser.add_argument('--experiment_data_dir', type=str, default= "../linear-models-gpt-j-TESTER",
                        help='For saving output checkpoints')
    parser.add_argument('--project_name', type=str, default= "linear-model-gpt-j-TESTER",
                        help='For saving output checkpoints')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--in_layer', type=int, default=28, metavar='N',
                        help='which layer of the source token would you like to transplant from')
    parser.add_argument('--out_layer', type=int, default=28, metavar='N',
                        help='which layer of the target token would you like to approximate')
    parser.add_argument('--next_token_skip', type=int, default=-1, metavar='N',
                        help='how many tokens ahead of the next token would you like to see')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='run test only')
    parser.add_argument('--test_data_split', type=float, default=0.4,
                        help='test data split')

    args = parser.parse_args()
    my_args = {}
    my_args["weight_decay"] = args.weight_decay
    my_args["new_train"] = True
    my_args["test_only"] = args.test_only
    my_args["cuda_device"] = args.cuda_device
    # Layers start from -1 (Embedding) to 27 (Layer 28) in code
    my_args["in_layer"] = args.in_layer - 1
    my_args["out_layer"] = args.out_layer - 1
    my_args["next_token_skip"] = args.next_token_skip
    in_layer = my_args["in_layer"]
    out_layer = my_args["out_layer"]
    next_token_skip = my_args["next_token_skip"]
    display_name = f"PREV_L{in_layer}_OUT_L{out_layer}_NEXT_T{next_token_skip}"
    my_args["display_name"] = display_name
    my_args["project_name"] = args.project_name
    my_args["experiment_data_dir"] = args.experiment_data_dir
    experiment_data_dir = my_args["experiment_data_dir"]
    my_args["test_results"] = f"{experiment_data_dir}/linear_model_{display_name}.csv"
    my_args["log_file"] = f"{experiment_data_dir}/log_{display_name}.txt"
    my_args["log_interval"] = args.log_interval
    my_args["fine_tune_epochs"] = args.fine_tune_epochs
    my_args["batch_size"] = args.batch_size
    my_args["learning_rate"] = args.learning_rate
    my_args["test_data_split"] = args.test_data_split

    MODEL_NAME = "EleutherAI/gpt-j-6b"
    TEACHER_MODEL, TEACHER_TOK = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).cuda(),
    AutoTokenizer.from_pretrained(MODEL_NAME) 
    )
    baukit.set_requires_grad(False, TEACHER_MODEL)
    DECODER = torch.nn.Sequential(TEACHER_MODEL.transformer.ln_f, TEACHER_MODEL.lm_head)

    print("current args setting", my_args)
    main(my_args)
    
    
