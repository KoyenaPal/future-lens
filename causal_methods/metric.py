from torchmetrics import Metric
import torch


class PrecisionAtKMetric(Metric):
    def __init__(self, n=0, topk=10):
        super().__init__()
        self.n = n
        self.topk = topk
        self.add_state(f"correct", default=torch.zeros(topk), dist_reduce_fx="sum")
        self.add_state(f"total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_top_k, golds):
        candidates = preds_top_k[:, self.n, :]
        golds = golds[:, self.n]
        hits = (candidates == golds.unsqueeze(-1))
        hits = torch.where(hits, 1, 0)
        hits = hits.sum(dim=0)
        self.correct += hits
        self.total += golds.shape[0]

    def compute(self):
        self.correct = torch.cumsum(self.correct, dim=0)
        return self.correct.float() / self.total.float()


class CalibratedAccuracy(Metric):
    def __init__(self, n=0, topk=10):
        super().__init__()
        self.n = n
        self.topk = topk
        self.add_state(f"calibrated_90_higher_tokens", default=torch.tensor([]), dist_reduce_fx=None)
        self.add_state(f"result_verbose", default=[], dist_reduce_fx=None)
        # self.add_state(f"calibrated_60_higher_tokens", default=torch.tensor([]), dist_reduce_fx=None)
        self.add_state(f"calibrated_correct", default=torch.zeros(4), dist_reduce_fx="sum")
        self.add_state(f"calibrated_total", default=torch.zeros(4), dist_reduce_fx="sum")
        self.add_state(f"correct", default=torch.zeros(topk,1), dist_reduce_fx="sum")
        self.add_state(f"total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs, pred, golds, init_input_ids, gen_input_ids):
        confidence_ranges = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0)]
        grouped_accuracies = [[] for _ in confidence_ranges]
        grouped_correct_90_higher_tokens = []
        result_verbose_list = []
        # grouped_correct_60_higher_tokens = []
        probs = probs[:, self.n, :]
        candidates = pred[:, self.n, :]
        golds = golds[:, self.n]
        accuracies = torch.eq(candidates, golds.unsqueeze(-1)).float()
        hits = (candidates == golds.unsqueeze(-1))
        for overall_idx, (prob, accuracy, cands) in enumerate(zip(probs, accuracies, candidates)):
            for idx, (start, end) in enumerate(confidence_ranges):
                if start <= prob.item() < end:
                    grouped_accuracies[idx].append(accuracy.item())
                    correct = False
                    if accuracy.item() != 0.0:
                        correct = True                
                    result_verbose_list.append((cands.item(), 
                                                prob.item(), 
                                                correct, 
                                                init_input_ids[overall_idx],
                                                gen_input_ids[overall_idx]))
                    if accuracy.item() != 0.0 and idx == (len(confidence_ranges) - 1):
                        grouped_correct_90_higher_tokens.append(cands.item())
                    # elif accuracy.item() > 0.0 and idx == (len(confidence_ranges) - 2):
                    #     grouped_correct_60_higher_tokens.append(cands.item())
                    break
                    
        self.calibrated_correct += torch.tensor([sum(group) if group else 0 for group in grouped_accuracies]).cuda()
        self.calibrated_total += torch.tensor([len(group) if group else 0 for group in grouped_accuracies]).cuda()
        self.calibrated_90_higher_tokens = torch.cat((self.calibrated_90_higher_tokens, torch.tensor(grouped_correct_90_higher_tokens).cuda()),dim=0).cuda()
        #self.result_verbose = torch.cat((self.result_verbose, torch.tensor(result_verbose_list).cuda()),dim=0).cuda()
        self.result_verbose.extend(result_verbose_list)
        # self.calibrated_60_higher_tokens = torch.cat((self.calibrated_60_higher_tokens, torch.tensor(grouped_correct_60_higher_tokens).cuda()),dim=0).cuda()
        # print("---RESULT VERBOSE---")
        # print(self.result_verbose)
        hits = torch.where(hits, 1, 0)
        hits = hits.sum(dim=0)
        
        self.correct += hits
        self.total += golds.shape[0]

    def compute(self):
        self.correct = torch.cumsum(self.correct, dim=0)
        return (self.correct.float() / self.total.float()), (self.calibrated_correct.float() / self.calibrated_total.float()), self.calibrated_90_higher_tokens, self.result_verbose




class SurprisalMetric(Metric):
    def __init__(self, max_n=0):
        super().__init__()
        self.max_n = max_n
        self.add_state(f"surprisal", default=torch.zeros(max_n), dist_reduce_fx="sum")
        self.add_state(f"total", default=torch.zeros(max_n), dist_reduce_fx="sum")

    def update(self, surprisal_matrix):
        self.surprisal += surprisal_matrix.sum(dim=0)
        self.total += surprisal_matrix.shape[0]

    def compute(self):
        return self.surprisal.float() / self.total.float()

