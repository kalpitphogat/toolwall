"""Fine-tune a sequence classifier on the leak-free splits (`toolwall train`).

A plain PyTorch loop rather than HF Trainer, so every step is visible:
  * windows built with the same `chunk_texts` used at inference; payload spans decide
    which windows of an injected document are positive
  * class-balanced cross-entropy, AdamW (no decay on bias/LayerNorm), linear warmup/decay
  * bf16 autocast on GPUs that support it, gradient clipping, gradient accumulation
  * document-level validation (max over windows) every half epoch; best checkpoint by
    validation PR-AUC, early stopping on patience
  * afterwards: temperature scaling and thresholds at target false-positive rates, both
    fitted on validation only, saved to calibration.json next to the weights
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from ..data.schema import read_jsonl
from ..eval.metrics import average_precision, roc_auc, tpr_at_fpr
from .calibrate import fit_temperature, sigmoid, threshold_at_fpr
from .chunking import chunk_label, chunk_texts


def _windows(tokenizer, records, max_length, stride):
    chunks = chunk_texts(tokenizer, [r.text for r in records], max_length, stride)
    per_doc: dict[int, int] = {}
    for c in chunks:
        per_doc[c.doc] = per_doc.get(c.doc, 0) + 1
    kept, dropped = [], 0
    for c in chunks:
        r = records[c.doc]
        lab = chunk_label(r.label, r.span, c.char_span)
        if lab is None:
            # injected document without a span that spans several windows: we cannot tell which window holds it
            if per_doc[c.doc] == 1:
                lab = r.label
            else:
                dropped += 1
                continue
        kept.append((c.input_ids, c.attention_mask, lab))
    return kept, dropped


def _batches(items, batch_size, rng, pad_id):
    """Length-bucketed batches: sort within shuffled buckets so padding stays small."""
    idx = list(range(len(items)))
    rng.shuffle(idx)
    bucket = batch_size * 50
    batches = []
    for b in range(0, len(idx), bucket):
        part = sorted(idx[b : b + bucket], key=lambda i: len(items[i][0]))
        batches += [part[k : k + batch_size] for k in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    for bt in batches:
        width = max(len(items[i][0]) for i in bt)
        ids = [items[i][0] + [pad_id] * (width - len(items[i][0])) for i in bt]
        mask = [items[i][1] + [0] * (width - len(items[i][1])) for i in bt]
        yield ids, mask, [items[i][2] for i in bt]


def doc_margins(model, tokenizer, texts, device, max_length, stride, batch_size=32):
    """Max-over-windows logit margin per document (same aggregation as TransformerScorer)."""
    import torch

    chunks = chunk_texts(tokenizer, texts, max_length, stride)
    out = np.full(len(texts), -np.inf)
    pad = tokenizer.pad_token_id or 0
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].input_ids))
    model.eval()
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            width = max(len(chunks[i].input_ids) for i in idx)
            ids = torch.tensor([chunks[i].input_ids + [pad] * (width - len(chunks[i].input_ids)) for i in idx], device=device)
            mask = torch.tensor([chunks[i].attention_mask + [0] * (width - len(chunks[i].attention_mask)) for i in idx], device=device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(input_ids=ids, attention_mask=mask).logits.float()
            m = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            for j, i in enumerate(idx):
                out[chunks[i].doc] = max(out[chunks[i].doc], m[j])
    model.train()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base-model", default="microsoft/deberta-v3-small")
    p.add_argument("--out", default="models/toolwall-deberta-v3-small")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--patience", type=int, default=3, help="evaluations without improvement before stopping")
    p.add_argument("--target-fprs", default="0.01,0.05")
    p.add_argument("--max-train", type=int, default=0, help="subsample training docs (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("w", encoding="utf-8")

    def emit(**kw):
        kw["time"] = round(time.time(), 1)
        log.write(json.dumps(kw) + "\n")
        log.flush()
        print(json.dumps(kw), flush=True)

    train = list(read_jsonl(Path(args.data) / "train.jsonl"))
    val = list(read_jsonl(Path(args.data) / "val.jsonl"))
    if args.max_train:
        rng = random.Random(args.seed)
        train = rng.sample(train, min(args.max_train, len(train)))
        val = rng.sample(val, min(max(200, args.max_train // 5), len(val)))

    tok = AutoTokenizer.from_pretrained(args.base_model)
    items, dropped = _windows(tok, train, args.max_length, args.stride)
    n_pos = sum(lab for _, _, lab in items)
    emit(event="data", train_docs=len(train), train_windows=len(items), positive_windows=n_pos,
         dropped_unlabelable_windows=dropped, val_docs=len(val))

    # fp32 master weights: transformers >= 5 loads checkpoints in their stored dtype (fp16 for DeBERTa-v3),
    # and AdamW on fp16 weights diverges to NaN after the first step (eps underflows). bf16 autocast gives the speed.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label={0: "BENIGN", 1: "INJECTION"}, label2id={"BENIGN": 0, "INJECTION": 1},
        dtype=torch.float32,
    ).to(device)
    model.train()

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    groups = [
        {"params": [p_ for n, p_ in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p_ for n, p_ in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(items) / args.batch_size / args.grad_accum)
    total = max(1, int(steps_per_epoch * args.epochs))
    warm = int(total * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm))
    )
    weights = torch.tensor([len(items) / (2 * (len(items) - n_pos)), len(items) / (2 * max(1, n_pos))], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    val_texts, val_y = [r.text for r in val], np.array([r.label for r in val])
    eval_every = max(1, steps_per_epoch // 2)
    best, bad_evals, step = -1.0, 0, 0
    rng = random.Random(args.seed)
    pad = tok.pad_token_id or 0
    t_start = time.time()
    done = False
    epoch = 0
    while not done:
        running, micro = 0.0, 0
        for ids, mask, labels in _batches(items, args.batch_size, rng, pad):
            ids_t = torch.tensor(ids, device=device)
            mask_t = torch.tensor(mask, device=device)
            y_t = torch.tensor(labels, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(input_ids=ids_t, attention_mask=mask_t).logits
            loss = loss_fn(logits.float(), y_t) / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; refusing to continue")
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                emit(event="train", step=step, epoch=round(step / steps_per_epoch, 3), loss=round(running / 50, 5),
                     lr=sched.get_last_lr()[0])
                running = 0.0
            if step % eval_every == 0 or step >= total:
                m = doc_margins(model, tok, val_texts, device, args.max_length, args.stride)
                pr = average_precision(val_y, m)
                emit(event="val", step=step, pr_auc=pr, roc_auc=roc_auc(val_y, m), tpr_at_1pct_fpr=tpr_at_fpr(val_y, m, 0.01),
                     elapsed_min=round((time.time() - t_start) / 60, 2))
                if pr > best:
                    best, bad_evals = pr, 0
                    model.save_pretrained(out)
                    tok.save_pretrained(out)
                else:
                    bad_evals += 1
                if bad_evals >= args.patience or step >= total:
                    done = True
                    break
        epoch += 1
        if epoch > math.ceil(args.epochs):
            done = True

    # calibration on validation with the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(out, dtype=torch.float32).to(device)
    m = doc_margins(model, tok, val_texts, device, args.max_length, args.stride)
    temperature = fit_temperature(m, val_y)
    probs = sigmoid(m / temperature)
    fprs = [float(x) for x in args.target_fprs.split(",")]
    calibration = {
        "base_model": args.base_model,
        "positive_index": 1,
        "temperature": temperature,
        "thresholds": {f"fpr_{f:g}": threshold_at_fpr(probs[val_y == 0], f) for f in fprs},
        "max_length": args.max_length,
        "stride": args.stride,
        "fitted_on": "val",
        "val": {"n": int(len(val_y)), "pr_auc": average_precision(val_y, probs), "roc_auc": roc_auc(val_y, probs)},
        "train_minutes": round((time.time() - t_start) / 60, 1),
        "args": vars(args),
    }
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    emit(event="done", best_val_pr_auc=best, temperature=temperature, thresholds=calibration["thresholds"])


if __name__ == "__main__":
    main()
