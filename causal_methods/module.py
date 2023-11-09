import pytorch_lightning as pl
from metric import PrecisionAtKMetric, SurprisalMetric
from torch import optim
import re
from torch.nn import KLDivLoss
import torch.nn.functional as F
from baukit import TraceDict
import baukit
import torch
import json
import os


class GPTJModule(pl.LightningModule):

    # Natural, normal context as a comparision to the the trained soft prompt prefix
    _COMPARE_CONTEXT = {
        10: [
            "Hello! Could you please tell me more about \"",
            "The multi-tokens present here are \"",
            "The concepts in this hidden state listed are: (",
            "<|endoftext|> This state is describing about the following concept:"
            ],
    }

    def __init__(
            self, model, tokenizer, batch_size, in_layer, out_layer, next_token_skip, 
            lr=None, text_prefix=True, weight_decay=None, max_n=None, 
            prefix_length=10, top_k=10, output_path=None, context_idx=0, metric="precision@k"
        ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = dict()
        self.prefix_length = prefix_length
        self.batch_size = batch_size

        self.output_path = output_path

        self.text_prefix = text_prefix
        self.context_idx = context_idx

        self.next_token_skip = next_token_skip
        self.in_layer = in_layer
        self.out_layer = out_layer
        
        self.top_k_preds = []
        self.decoded_labels = []

        self.kl_loss = KLDivLoss(reduction="batchmean", log_target=True)
        self.layer_names = [n for n, _ in self.model.named_modules() if re.match(r'^transformer.h.\d+$', n)]
        self.max_n = max_n
        self.top_k = top_k

        self.metrics = None

        if max_n is not None:
            if metric == "precision@k":
                self.metrics = []
                for i in range(0, max_n):
                    self.metrics.append(PrecisionAtKMetric(i, top_k))
            else:
                self.metrics = SurprisalMetric(max_n)

        self.hyper_parameters = {
            "lr": lr,
            "weight_decay": weight_decay
        }

        self.optimized_param = []
        for name, param in self.model.named_parameters():
            if name == 'transformer.prefix_embedding':
                self.optimized_param.append(param)
    
    def save(self, saving_path: str):
        self.model.save_prefix(saving_path)

    def _get_transplant_layers(self, token_indices, layers=list(range(28))):
        layer_names = [n for n, _ in self.model.named_modules() if re.match(r'^transformer.h.\d+$', n)]
        input_layers = [layer_names[layer] for layer in layers]
        return_indices = []
        for input_layer in input_layers:
            return_indices.append((input_layer, token_indices))
        return return_indices
    
    def _replace_hidden_layer_logits(self, replacement_hs_values):

        def rep_out(batch_outputs, layer, _):
            if not self.in_layer == "Emb":
                layer_names = self.layer_names[self.in_layer]
                if layer == layer_names:
                    batch_outputs[0][:, self.prefix_length - 1, :] = replacement_hs_values.squeeze()
            return batch_outputs
        
        return rep_out
    
    def _predict_with_topk(self, input_ids, source_hs, labels, generation_length, top_k, batch_size_when_no_input):
        top_k_output = []
        with TraceDict(self.model, self.layer_names, edit_output=self._replace_hidden_layer_logits(source_hs)) as tr:
            for i in range(generation_length):
                outputs = self.model(input_ids=input_ids, batch_size_when_no_input=batch_size_when_no_input)
                pred_logits = outputs.logits[:, -1, :] # Batch, Vocab dim unchanged, takes the last token logits
                pred_probs = torch.softmax(pred_logits, dim=-1)
                pred_top_k = torch.topk(pred_probs, top_k, dim=-1).indices

                next_token = labels[:, i].unsqueeze(1)
                if input_ids is None:
                    input_ids = next_token
                    batch_size_when_no_input = None
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                top_k_output.append(pred_top_k)

        top_k_output = torch.stack(top_k_output, dim=1)
        return top_k_output
    
    def _predict_with_suprisal(self, input_ids, source_hs, labels, raw_input_ids, raw_attention_mask, generation_length, batch_size_when_no_input):
        suprisal_values = []
        decoded_output = []

        with TraceDict(self.model, self.layer_names, edit_output=self._replace_hidden_layer_logits(source_hs)) as tr:
            for i in range(generation_length):
                outputs = self.model(input_ids=input_ids, batch_size_when_no_input=batch_size_when_no_input)
                pred_logits = outputs.logits[:, -1, :] # Batch, Vocab dim unchanged, takes the last token logits
                pred_probs = torch.softmax(pred_logits, dim=-1)
                pred_tokens = torch.argmax(pred_probs, dim=-1)

                next_token = labels[:, i].unsqueeze(1)
                if input_ids is None:
                    input_ids = next_token
                    batch_size_when_no_input = None
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                decoded_output.append(pred_tokens)
            tr.close()
        
        decoded_output = torch.stack(decoded_output, dim=1)
        assert decoded_output.shape[1] == generation_length # Sanity check for the generation length

        self.model.unprefix_model()
        for position_id in range(generation_length):
            outputs = self.model(input_ids=raw_input_ids, attention_mask=raw_attention_mask)
            pred_logits = outputs.logits[:, -1, :] # Batch, Vocab dim unchanged, takes the last token logits
            pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
            max_likely_tokens = decoded_output[:, position_id]
            likely_token_log_prob = torch.gather(pred_log_probs, dim=-1, index=max_likely_tokens.unsqueeze(-1))
            next_token = labels[:, position_id].unsqueeze(-1)
            
            raw_input_ids = torch.cat([raw_input_ids, next_token], dim=-1)
            raw_attention_mask = torch.cat([raw_attention_mask, torch.ones_like(next_token)], dim=-1)
            suprisal_values.append(-likely_token_log_prob)
        
        suprisal_values = torch.stack(suprisal_values, dim=1)
        return suprisal_values.squeeze(-1)
    
    def _get_equal_length_context(self):
        return self.tokenizer(self._COMPARE_CONTEXT[self.prefix_length][self.context_idx], return_tensors="pt")["input_ids"].to(self.device)
    
    def on_train_start(self) -> None:
        self.model.train()
        return super().on_train_start()
    
    def training_step(self, batch, batch_idx):
        source_hs, target_hs, labels = batch["source_hs"], batch["target_hs"], batch["labels"]
        with TraceDict(self.model, [*self.layer_names, "transformer.wte"], edit_output=self._replace_hidden_layer_logits(source_hs)) as tr:
            loss = None
            target_hs_idx = 0

            input_ids = None
            batch_size_when_no_input = self.batch_size

            for i in range(max(self.next_token_skip) + 1):
                self.model.prefix_model()
                _ = self.model(input_ids=input_ids, batch_size_when_no_input=batch_size_when_no_input)
                if i in self.next_token_skip:
                    gen_hidden_states = torch.stack([tr[ln].output[0] for ln in self.layer_names])
                    gen_hidden_states = gen_hidden_states[self.out_layer, :, -1, :]

                    gen_logits = self.model.lm_head(self.model.transformer.ln_f(gen_hidden_states))
                    desire_logits = self.model.lm_head(self.model.transformer.ln_f(target_hs[:, target_hs_idx, :]))

                    gen_prob = torch.log_softmax(gen_logits, dim=-1)
                    desire_prob = torch.log_softmax(desire_logits, dim=-1)

                    if loss is None:
                        loss = self.kl_loss(gen_prob, desire_prob)
                    else:
                        loss += self.kl_loss(gen_prob, desire_prob)
                    target_hs_idx += 1
                
                next_token = labels[:, i].unsqueeze(1)
                if input_ids is None:
                    batch_size_when_no_input = None
                    input_ids = next_token
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    
        loss = loss / len(self.next_token_skip)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def on_test_start(self) -> None:
        self.model.eval()
        if not isinstance(self.metrics, SurprisalMetric):
            for i in range(len(self.metrics)):
                self.metrics[i].reset()
                self.metrics[i] = self.metrics[i].to(self.device)
        else:
            self.metrics.reset()
            self.metrics = self.metrics.to(self.device)
        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        source_hs, labels = batch["source_hs"], batch["labels"]
        batch_size = source_hs.shape[0]

        if self.text_prefix:
            input_ids = self._get_equal_length_context().repeat((batch_size, 1))
            batch_size_when_no_input = None
            self.model.unprefix_model()
        else:
            input_ids = None
            batch_size_when_no_input = batch_size
            self.model.prefix_model()

        if isinstance(self.metrics, SurprisalMetric):
            raw_input_ids, raw_attention_mask = batch["raw_input_ids"], batch["raw_attention_mask"]
            surprisal_matrix = self._predict_with_suprisal(input_ids, source_hs, labels, raw_input_ids, raw_attention_mask, self.max_n, batch_size_when_no_input)
            self.metrics(surprisal_matrix)
        else:
            top_k_output = self._predict_with_topk(input_ids, source_hs, labels, self.max_n, self.top_k, batch_size_when_no_input)
            golden_output = labels[:, :self.max_n]
            for metric in self.metrics:
                metric(top_k_output, golden_output)
            
            self.top_k_preds.extend(top_k_output.cpu().numpy().tolist())
            self.decoded_labels.extend(golden_output.cpu().numpy().tolist())
        
    def on_test_end(self) -> None:
        results = {}
        if isinstance(self.metrics, SurprisalMetric):
            surprisal_for_n = self.metrics.compute()
            results["surprisal"] = surprisal_for_n.cpu().numpy().tolist()
            print("Surprisal: ", results["surprisal"])
        else:
            for i, metric in enumerate(self.metrics):
                prec_at_k = metric.compute()
                results[str(i)] = prec_at_k.cpu().numpy().tolist()

            for key in results: 
                print_values = ["{:0.2f}".format(r*100) for r in results[key]]        
                print(key, print_values)

            json.dump(self.top_k_preds, open(os.path.join(self.output_path, "top_k_preds.json"), "w"))
            json.dump(self.decoded_labels, open(os.path.join(self.output_path, "decoded_labels.json"), "w"))
            self.top_k_preds, self.decoded_labels = [], []

        json.dump(results, open(os.path.join(self.output_path, "test_results.json"), "w"))
        return results
    
    def configure_optimizers(self):
        print("Num of Optimized Params: {}".format(len(self.optimized_param)))
        return optim.AdamW(params=self.optimized_param, **self.hyper_parameters)
