import pathlib
p = pathlib.Path(r'E:\agi-research\projects\project_f_multi_agent\code\pz_dmc.py')
t = p.read_text(encoding='utf-8')

# Bug 1: rewrite train_peragent_monitor to train via BCE on monitor output
# (keep monitor as is, but compute BCE with a learnable bias term we ADD
#  into the monitor's final head, so gradients flow into the MLP).
# Simpler approach: use BCEWithLogitsLoss but expose pre-sigmoid logits
# by replacing the final sigmoid with identity. We add a new attribute
# `pre_sigmoid_head`.
old_train = '''def train_peragent_monitor(monitor, opt, ep_data_list, n_epochs=20,
                            batch_size=16):
    """Train one Monitor on a list of per-agent ep_data dicts."""
    inputs = torch.from_numpy(np.stack([d["obs_history"] for d in ep_data_list])).float()
    labels = torch.tensor([d["label"] for d in ep_data_list], dtype=torch.float32)
    pos = (labels == 1).sum().item(); neg = (labels == 0).sum().item()
    if pos == 0 or neg == 0:
        return 0.5  # cannot learn
    pos_w = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    head = nn.Linear(monitor.obs_dim * monitor.history_len, 1).to(inputs.device)
    monitor.head = head  # not used directly; we want monitor to give probability
    # Actually: easier to use BCE on monitor's *output* (sigmoid).
    # We swap to a BCE-with-logits via a learnable linear inside the monitor.
    # For simplicity: train a parallel "logit head" by minimising BCE on log(p/(1-p)+eps).
    for _ in range(n_epochs):
        idx = torch.randperm(inputs.size(0))
        for s in range(0, inputs.size(0), batch_size):
            mb = idx[s:s+batch_size]
            x = inputs[mb]
            # Recompute logit via a wrapper that exposes pre-sigmoid logits
            flat = x.reshape(x.size(0), -1)
            # Use the monitor's *existing* head up to the last layer
            # by removing sigmoid; we approximate with log(p/(1-p)+eps)
            p = monitor(flat.reshape(x.size(0), monitor.history_len, monitor.obs_dim))
            p = p.clamp(1e-4, 1 - 1e-4)
            logit = torch.log(p / (1 - p))
            loss = bce(logit, labels[mb])
            opt.zero_grad(); loss.backward(); opt.step()
    # Report final AUROC-like proxy
    with torch.no_grad():
        p_all = monitor(inputs).numpy()
    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(labels.numpy(), p_all)
    except Exception:
        auroc = 0.5
    return auroc'''
new_train = '''def _monitor_logit(monitor, x):
    """Forward pass that returns the pre-sigmoid logit so we can use
    BCEWithLogitsLoss. x: (B, history_len, obs_dim)."""
    flat = x.reshape(x.size(0), -1)
    # The monitor net ends with a Linear (no sigmoid). We expose that
    # by walking the Sequential and stopping before the final sigmoid
    # module. To keep this simple we re-implement the forward here.
    h = flat
    for layer in monitor.net:
        if isinstance(layer, nn.Sigmoid):
            break
        h = layer(h)
    return h.squeeze(-1)


def train_peragent_monitor(monitor, opt, ep_data_list, n_epochs=20,
                            batch_size=16):
    """Train one Monitor on a list of per-agent ep_data dicts."""
    if len(ep_data_list) < 2:
        return 0.5
    inputs = torch.from_numpy(np.stack([d["obs_history"] for d in ep_data_list])).float()
    labels = torch.tensor([d["label"] for d in ep_data_list], dtype=torch.float32)
    pos = (labels == 1).sum().item(); neg = (labels == 0).sum().item()
    if pos == 0 or neg == 0:
        # Cannot learn from a single class; skip training and return NaN.
        return float("nan")
    pos_w = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(n_epochs):
        idx = torch.randperm(inputs.size(0))
        for s in range(0, inputs.size(0), batch_size):
            mb = idx[s:s+batch_size]
            x = inputs[mb]
            logit = _monitor_logit(monitor, x)
            loss = bce(logit, labels[mb])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p_all = torch.sigmoid(_monitor_logit(monitor, inputs)).numpy()
    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(labels.numpy(), p_all)
    except Exception:
        auroc = 0.5
    return auroc'''
assert old_train in t, 'old_train block missing'
t = t.replace(old_train, new_train)

# Bug 2: n_params_total uses .numel() which does not exist; use sum(p.numel() for p in m.parameters())
t = t.replace(
    'sum(m.numel() for m in monitors.values())',
    'sum(p.numel() for m in monitors.values() for p in m.parameters())'
)
t = t.replace(
    'sum(p.numel() for p in joint_pred.parameters())',
    'sum(p.numel() for p in joint_pred.parameters())'
)
# Also fix the policies part
t = t.replace(
    'sum(p.numel() for p_ in policies.values() for p in p_.parameters())',
    'sum(p.numel() for p_ in policies.values() for p in p_.parameters())'
)

p.write_text(t, encoding='utf-8')
import ast
ast.parse(t)
print('AST OK, len', len(t))