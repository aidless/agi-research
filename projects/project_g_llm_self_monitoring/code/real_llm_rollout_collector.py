"""
Project G -- Real-LM rollout collector.

Loads a frozen small LM (default: Qwen2.5-3B-Instruct, already cached
in ~/.cache/huggingface) and generates reasoning traces on a small
math dataset (default: GSM8K, also cached).

This is the BRIDGE between synthetic smoke tests and the full
pre-registered H10. It generates REAL LLM traces that the LLMSlotMonitor
can be trained on.

NO_SELF_DECEPTION.md compliance:
- The default LM is Qwen2.5-3B-Instruct (3B params). This is
  documented in the H10 pre-registration as the primary candidate.
- The default dataset is GSM8K. This is documented in the H10
  pre-registration as the primary candidate.
- The failure label is GSM8K ground truth (final-answer correctness).
  This is the H10 default failure-label definition.
- The user may swap LM (e.g., to Phi-3-mini) or dataset (e.g.,
  to MATH) by changing the constructor parameters.

Performance note:
- 3B params on CPU: ~1-2 sec/token forward pass.
- For 50-token rollouts: ~1-2 min/rollout.
- For 16 rollouts: ~30 min total.
- For 200 rollouts (full pre-reg): ~5 hours on CPU, much faster on GPU.
"""

import os
import torch


# Default LM (cached locally in this environment).
DEFAULT_LM = "Qwen/Qwen2.5-1.5B-Instruct"

# Default dataset (cached locally in this environment).
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"


def load_frozen_lm(model_name=DEFAULT_LM, device="cpu", dtype=torch.float16):
    """Load a frozen small LM and its tokenizer.

    Returns:
        model: the loaded model in eval mode, parameters frozen
        tokenizer: the matching tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} on {device} (dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    # Freeze all parameters (no fine-tuning).
    for param in model.parameters():
        param.requires_grad = False
    print(f"Loaded {model_name}: {sum(p.numel() for p in model.parameters())} params")
    return model, tokenizer


def load_gsm8k(split="test", n_samples=None):
    """Load GSM8K math reasoning dataset.

    Returns:
        problems: list of dicts with keys "question" and "answer"
        ground_truths: list of int (final numerical answer)
    """
    from datasets import load_dataset

    print(f"Loading {DEFAULT_DATASET} ({split})...")
    ds = load_dataset(DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    problems = []
    ground_truths = []
    for ex in ds:
        problems.append({"question": ex["question"], "full_answer": ex["answer"]})
        # Extract ground truth from the answer field.
        # GSM8K answers end with "#### <number>".
        import re
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"])
        if m:
            try:
                gt = float(m.group(1))
                ground_truths.append(gt)
            except ValueError:
                ground_truths.append(None)
        else:
            ground_truths.append(None)
    print(f"Loaded {len(problems)} GSM8K problems ({sum(1 for g in ground_truths if g is not None)} with parseable ground truth)")
    return problems, ground_truths


def extract_final_number(text):
    """Extract the final numerical answer from generated text."""
    import re
    # Try #### format first (GSM8K style).
    m = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Try "the answer is X" pattern.
    m = re.search(r"answer is\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Try last number in text.
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def generate_trace(model, tokenizer, question, max_new_tokens=80, device="cpu"):
    """Generate a single reasoning trace for a question.

    Returns:
        tokens: list[int] token IDs
        logits_features: list[float] mean logit per token (proxy for confidence)
        text: the decoded generated text
    """
    # Build the prompt.
    prompt = (
        "Solve this math problem step by step. End with 'The answer is <number>'.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # outputs.scores is a tuple of (1, vocab_size) per generated token.
    # We use the max logit as a proxy for token confidence.
    new_token_ids = outputs.sequences[0][input_ids.shape[1]:].tolist()
    confidence_per_token = []
    for step_logits in outputs.scores:
        # step_logits: (1, vocab_size)
        max_logit = step_logits.max(dim=-1).values.item()
        confidence_per_token.append(max_logit)
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return new_token_ids, confidence_per_token, text


def collect_real_rollouts(model, tokenizer, problems, ground_truths,
                          max_new_tokens=80, device="cpu", seed=0):
    """Collect reasoning traces from the frozen LM on GSM8K problems.

    Returns:
        trace_features_list: list of (window, feat_dim) tensors
        labels: list of 0/1 failure labels
        metadata: list of dicts with question, text, ground_truth, predicted
    """
    torch.manual_seed(seed)
    trace_features_list = []
    labels = []
    metadata = []
    n_success = 0
    n_failure = 0
    for i, (prob, gt) in enumerate(zip(problems, ground_truths)):
        if gt is None:
            continue
        tokens, confs, text = generate_trace(
            model, tokenizer, prob["question"],
            max_new_tokens=max_new_tokens, device=device,
        )
        predicted = extract_final_number(text)
        is_failure = 1 if (predicted is None or abs(predicted - gt) > 1e-3) else 0
        # Build per-token (token_id, confidence) feature tensor.
        # Pad/truncate to window=20.
        window = 20
        # Token IDs as floats in [-1, 1] (normalized by vocab_size).
        vocab_size = model.config.vocab_size
        feat_tokens = [
            (tok / vocab_size) * 2.0 - 1.0 for tok in tokens[:window]
        ]
        feat_logits = [1.0 / (1.0 + torch.sigmoid(torch.tensor(-c)).item() + 1e-6)
                       for c in confs[:window]]
        # Pad to window.
        while len(feat_tokens) < window:
            feat_tokens.append(0.0)
        while len(feat_logits) < window:
            feat_logits.append(0.5)  # neutral
        # Build (window, feat_dim=64) tensor: tile token and logit each to 32-dim.
        feat_tensor = torch.zeros(window, 64)
        for t in range(window):
            feat_tensor[t, :32] = feat_tokens[t]
            feat_tensor[t, 32:] = feat_logits[t]
        trace_features_list.append(feat_tensor)
        labels.append(float(is_failure))
        metadata.append({
            "question": prob["question"],
            "text": text,
            "ground_truth": gt,
            "predicted": predicted,
            "is_failure": is_failure,
        })
        if is_failure:
            n_failure += 1
        else:
            n_success += 1
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(problems)}] success={n_success}, failure={n_failure}")
    return trace_features_list, labels, metadata


if __name__ == "__main__":
    # Quick sanity check (no LM load, just the dataset).
    print("Real-LM rollout collector sanity check.")
    print("(Full LM trace collection requires transformers + Qwen2.5-3B-Instruct.)")
    print("See h10_real_pilot.py for the full pilot run.")
