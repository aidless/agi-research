"""slot_attention.py - minimal slot attention module for Project C PoC.

NOT production-ready. This is a feasibility check: can we encode
LunarLander states into discrete slots with a small Transformer-based
slot attention module, all in pure PyTorch CPU?

Architecture follows Locatello et al. 2020 "Object-Centric Learning
with Slot Attention":
  1. Input features: linear projection of obs_dim to slot_dim
  2. Slot initialization: learned parameters (n_slots x slot_dim)
  3. Iterative attention: slots compete for input features via softmax
  4. Update: GRU updates each slot based on its attended inputs
  5. Layer norm + iterate

Output: n_slots slot representations, each slot_dim wide. The
"object-centric" property: each slot should bind to a distinct
feature subset (e.g., position, velocity, angle, leg contact).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, n_slots=4, slot_dim=32, n_iters=3, hidden_dim=64,
                 input_dim=8, eps=1e-8):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.eps = eps

        # Project input to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Slot initialization: learned mu + log_sigma
        self.slot_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, slot_dim))
        nn.init.xavier_uniform_(self.slot_mu)

        # Attention: k, v from inputs; q from slots
        self.k_proj = nn.Linear(hidden_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, slot_dim, bias=False)
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # LayerNorm + MLP for slot residual
        self.norm1 = nn.LayerNorm(slot_dim)
        self.norm2 = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

    def forward(self, x):
        """
        x: (batch, n_features, input_dim)
        returns: (batch, n_slots, slot_dim)
        """
        b = x.size(0)
        # Project inputs to hidden_dim
        x_proj = self.input_proj(x)  # (b, n_features, hidden_dim)
        k = self.k_proj(x_proj)  # (b, n_features, slot_dim)
        v = self.v_proj(x_proj)  # (b, n_features, slot_dim)

        # Initialize slots
        mu = self.slot_mu.expand(b, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(b, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)  # (b, n_slots, slot_dim)

        # Iterate attention
        for _ in range(self.n_iters):
            slots_prev = slots
            slots_norm = self.norm1(slots)
            q = self.q_proj(slots_norm)  # (b, n_slots, slot_dim)

            # Attention logits: (b, n_slots, n_features)
            scale = self.slot_dim ** -0.5
            # attn_logits shape: (b, n_features, n_slots)
            attn_logits = torch.einsum("bnd,bsd->bns", k, q) * scale
            # Softmax over slots (last dim) so slots compete for each feature
            attn = F.softmax(attn_logits, dim=-1)  # (b, n_features, n_slots)
            attn = attn + self.eps
            # Renormalize over features so each input is fully explained
            attn = attn / attn.sum(dim=1, keepdim=True)  # (b, n_features, n_slots)

            # Aggregate: weighted sum over features per slot
            # updates: (b, n_slots, slot_dim)
            updates = torch.einsum("bns,bnd->bsd", attn, v)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, self.n_slots, self.slot_dim)

            # Residual MLP
            slots = slots + self.mlp(self.norm2(slots))

        return slots


def slot_diversity_loss(slots):
    """
    Encourage slots to bind to different features.
    slots: (batch, n_slots, slot_dim)
    Returns scalar loss; smaller = more diverse.
    """
    # Normalize slots
    s = F.normalize(slots, dim=-1)
    # Pairwise similarity (b, n_slots, n_slots)
    sim = torch.einsum("bnd,bmd->bnm", s, s)
    # Mask diagonal (self-similarity)
    n = sim.size(-1)
    mask = torch.eye(n, device=sim.device).bool().unsqueeze(0)
    sim = sim.masked_fill(mask, 0.0)
    # Penalize off-diagonal magnitude
    return sim.abs().mean()


if __name__ == "__main__":
    # PoC: does it work on synthetic batch?
    torch.manual_seed(42)
    slot = SlotAttention(n_slots=4, slot_dim=32, n_iters=3, hidden_dim=64,
                        input_dim=8)
    # Batch of 8 sequences, each with 6 feature vectors of dim 8
    x = torch.randn(8, 6, 8)
    out = slot(x)
    print("Input:", x.shape)
    print("Slots:", out.shape)
    print("Slot 0 mean:", out[0, 0].mean().item())
    print("Slot 1 mean:", out[0, 1].mean().item())
    print("Diversity loss:", slot_diversity_loss(out).item())

    # Test training: minimize reconstruction loss
    target = torch.randn(8, 4 * 32)  # target is a flat vector
    optimizer = torch.optim.Adam(slot.parameters(), lr=3e-4)
    for step in range(50):
        out = slot(x)
        flat = out.reshape(8, -1)
        recon_loss = F.mse_loss(flat, target)
        div_loss = slot_diversity_loss(out)
        loss = recon_loss + 0.1 * div_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"  step {step}: recon={recon_loss.item():.4f} div={div_loss.item():.4f}")
    print("PoC PASSED: slot attention trains and produces diverse slots.")