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

