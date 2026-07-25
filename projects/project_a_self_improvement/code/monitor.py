"""

monitor.py —The failure-prediction Monitor.



This is the CORE NOVELTY of Project A.



The Monitor M takes an episode history d_hist (a flat vector built by

``EpisodeLog.history_vector``) and outputs a probability that the current

episode will end in failure.



KEY INSIGHT (the decoupling assumption):

  M is trained ONLY on rollouts collected by a FROZEN policy pi.

  No gradient flows from M back into pi. This breaks the "self-play

  collapse" loop that hurts joint-trained self-critics.



REVIEW-ME:

- Architecture is intentionally simple (MLP). The paper appendix

  will ablate against LSTM / Transformer to show robustness.

- Failure labels are heuristic —see ``envs.is_failure_episode``.

- The Monitor class is a normal ``nn.Module`` with BCE loss.

"""



from __future__ import annotations

from dataclasses import dataclass

from typing import List



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader





@dataclass

class MonitorConfig:

    history_dim: int        # size of the flat vector from EpisodeLog.history_vector

    hidden: int = 64

    lr: float = 3e-4

    batch_size: int = 32

    epochs: int = 5

    seed: int = 0





class FailureDataset(Dataset):

    """

    Build (history_vector, label) pairs from a list of completed episodes.

    Label = 1.0 if the episode is a failure, 0.0 otherwise.



    REVIEW-ME: We sample ONE vector per episode (using the full history).

    The paper appendix will ablate using multi-sample per episode.

    """



    def __init__(self, episodes: List, history_len: int = 32, n_actions: int = 2, threshold: float = None):

        self.X = []

        self.y = []

        from envs import is_failure_episode

        # Auto-detect n_actions from the max action seen (capped at 16)

        max_action = 1

        for ep in episodes:

            for tr in ep.transitions:

                if tr.action > max_action:

                    max_action = tr.action

        detected = max(2, max_action + 1)

        n_actions = max(detected, n_actions)

        for ep in episodes:

            vec = ep.history_vector(history_len=history_len, n_actions=n_actions)

            self.X.append(vec)

            if threshold is None:
                self.y.append(1.0 if is_failure_episode(ep) else 0.0)
            else:
                self.y.append(1.0 if ep.total_reward < threshold else 0.0)

        self.X = np.stack(self.X).astype(np.float32)

        self.y = np.array(self.y, dtype=np.float32)



    def __len__(self):

        return len(self.X)



    def __getitem__(self, i):

        return self.X[i], self.y[i]





class FailureMonitor(nn.Module):

    """

    MLP-based monitor: history_vector -> P(failure).



    In v1 we use a 2-layer MLP. In v2 we'll ablate against:

      - larger MLP

      - 1-layer LSTM (sequential)

      - small Transformer encoder

    """



    def __init__(self, cfg: MonitorConfig):

        super().__init__()

        torch.manual_seed(cfg.seed)

        self.net = nn.Sequential(

            nn.Linear(cfg.history_dim, cfg.hidden),

            nn.ReLU(),

            nn.Linear(cfg.hidden, cfg.hidden),

            nn.ReLU(),

            nn.Linear(cfg.hidden, 1),

        )



    def forward(self, h: torch.Tensor) -> torch.Tensor:

        """Return P(failure) in [0, 1]."""

        return torch.sigmoid(self.net(h)).squeeze(-1)



    def predict(self, h_vec: np.ndarray) -> float:

        with torch.no_grad():

            t = torch.as_tensor(h_vec, dtype=torch.float32).unsqueeze(0)

            return float(self.forward(t).item())





def train_monitor(


    cfg: MonitorConfig,

    episodes: List,

    history_len: int = 32,

    verbose: bool = True,

    threshold: float = None,

) -> tuple[FailureMonitor, dict]:
    """
    and a small dict of metrics (final loss, AUROC).



    REVIEW-ME: We do early stopping based on train loss (no val split)

    for v1 to keep the code minimal. For the paper we'll add a held-out

    set in `experiments/` and report AUROC there.

    """

    ds = FailureDataset(episodes, history_len=history_len, threshold=threshold)

    if verbose:

        print(f"  Monitor dataset: {len(ds)} episodes "

              f"({int(ds.y.sum())} failures, {int((1 - ds.y).sum())} successes)")

    if int(ds.y.sum()) == 0 or int((1 - ds.y).sum()) == 0:

        if verbose:

            print("  Skipping monitor training: only one class present.")

        # return untrained monitor

        return FailureMonitor(cfg), {"final_loss": 0.0, "auroc": 0.5, "n": len(ds)}



    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = FailureMonitor(cfg)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)



    last_loss = 0.0

    for epoch in range(cfg.epochs):

        epoch_loss = 0.0

        n = 0

        for x, y in loader:

            logits = model(x)

            loss = F.binary_cross_entropy(logits, y)

            opt.zero_grad()

            loss.backward()

            opt.step()

            epoch_loss += float(loss.item()) * len(x)

            n += len(x)

        last_loss = epoch_loss / max(1, n)

        if verbose:

            print(f"  Epoch {epoch + 1}/{cfg.epochs}  loss={last_loss:.4f}")



    # simple AUROC

    with torch.no_grad():

        preds = model(torch.as_tensor(ds.X, dtype=torch.float32)).numpy()

    auroc = _quick_auroc(ds.y, preds)

    if verbose:

        print(f"  Monitor AUROC (on train set, for sanity): {auroc:.3f}")



    return model, {"final_loss": last_loss, "auroc": auroc, "n": len(ds)}





def _quick_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:

    """Quick AUROC for binary labels —used for sanity, not for paper claims."""

    from itertools import product

    pos = y_score[y_true == 1]

    neg = y_score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:

        return 0.5

    n_concord = 0

    n_pairs = 0

    for p in pos:

        n_concord += (neg < p).sum() + 0.5 * (neg == p).sum()

        n_pairs += len(neg)

    return float(n_concord / max(1, n_pairs))










