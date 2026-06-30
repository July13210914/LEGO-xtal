#!/usr/bin/env python3
"""Train the fast tetrahedron-constrained local SiO4 prior."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyxtal.symmetry import Group
from torch.utils.data import DataLoader, Dataset, Sampler

from O_local_prior import (
    LocalOxygenPrior,
    cell_matrix_from_parameters,
    local_proposal_loss,
    merge_local_proposals,
    periodic_cartesian_delta,
    stable_distance,
)

BASE_COLUMNS = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
LOSS_KEYS = ("loss", "vector", "tetra", "bridge", "bond", "osi2", "osi3", "oo2")


def op_parts(op):
    rot = getattr(op, "rotation_matrix", None)
    trans = getattr(op, "translation_vector", None)
    if rot is None or trans is None:
        affine = np.asarray(op.affine_matrix, dtype=float)
        rot, trans = affine[:3, :3], affine[:3, 3]
    return np.asarray(rot, dtype=float), np.asarray(trans, dtype=float)


def site_count(df):
    return sum(str(c).startswith("wp") for c in df.columns)


def expand_species(row, nslots, target_cn):
    group = Group(int(row["spg"]))
    positions = []
    for slot in range(nslots):
        if int(row[f"target_coord{slot}"]) != int(target_cn):
            continue
        wp_index = int(row[f"wp{slot}"])
        if wp_index < 0:
            continue
        generator = np.asarray([row[f"x{slot}"], row[f"y{slot}"], row[f"z{slot}"]], dtype=float)
        wp = group[wp_index]
        for op in wp.ops:
            rotation, translation = op_parts(op)
            positions.append((rotation @ generator + translation) % 1.0)
    return np.asarray(positions, dtype=np.float32) if positions else np.zeros((0, 3), dtype=np.float32)


def load_atom_sets(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    nslots = site_count(df)
    records, skipped = [], 0
    for source_index, row in df.iterrows():
        try:
            si = expand_species(row, nslots, 4)
            oxygen = expand_species(row, nslots, 2)
            if len(si) == 0 or len(oxygen) < 4 or len(oxygen) != 2 * len(si):
                skipped += 1
                continue
            records.append({
                "source_index": int(source_index),
                "spg": int(row["spg"]),
                "cell_parameters": np.asarray([float(row[c]) for c in BASE_COLUMNS[1:]], dtype=np.float32),
                "si": si,
                "o": oxygen,
            })
        except Exception as exc:
            skipped += 1
            print(f"Skipping row {source_index}: {type(exc).__name__}: {exc}")
    print(f"Loaded {len(records)}/{len(df)} usable teacher structures from {path}" + (f"; skipped {skipped}" if skipped else ""))
    return records, len(df)


def split_records_by_chunk(records, total_source_rows, chunk_size, val_fraction, seed):
    populated = sorted({r["source_index"] // chunk_size for r in records})
    if len(populated) < 2:
        raise RuntimeError("Need at least two populated chunks for validation")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(populated, dtype=np.int64)
    rng.shuffle(shuffled)
    n_val = min(max(1, int(round(len(shuffled) * val_fraction))), len(shuffled) - 1)
    val_chunks = set(map(int, shuffled[:n_val]))
    train_chunks = set(map(int, shuffled[n_val:]))
    train = [r for r in records if r["source_index"] // chunk_size in train_chunks]
    val = [r for r in records if r["source_index"] // chunk_size in val_chunks]
    return train, val, {
        "total_source_rows": int(total_source_rows), "chunk_size": int(chunk_size),
        "train_chunks": sorted(train_chunks), "validation_chunks": sorted(val_chunks),
        "train_structures": len(train), "validation_structures": len(val),
    }


class RecordDataset(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.records[index]


class BucketBatchSampler(Sampler):
    def __init__(self, records, batch_size, shuffle, seed, bucket_multiplier=20):
        self.records, self.batch_size, self.shuffle, self.seed = records, int(batch_size), bool(shuffle), int(seed)
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_multiplier))
        self.epoch = 0
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return math.ceil(len(self.records) / self.batch_size)
    def __iter__(self):
        indices = list(range(len(self.records)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle: rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=lambda i: len(self.records[i]["si"]))
            batches.extend(bucket[j:j + self.batch_size] for j in range(0, len(bucket), self.batch_size))
        if self.shuffle: rng.shuffle(batches)
        yield from batches


def collate_records(batch):
    max_si, max_o, bsz = max(len(r["si"]) for r in batch), max(len(r["o"]) for r in batch), len(batch)
    si = torch.zeros((bsz, max_si, 3), dtype=torch.float32)
    sm = torch.zeros((bsz, max_si), dtype=torch.bool)
    oxygen = torch.zeros((bsz, max_o, 3), dtype=torch.float32)
    om = torch.zeros((bsz, max_o), dtype=torch.bool)
    cp = torch.zeros((bsz, 6), dtype=torch.float32)
    idx = torch.zeros(bsz, dtype=torch.long)
    spg = torch.zeros(bsz, dtype=torch.long)
    for i, rec in enumerate(batch):
        ns, no = len(rec["si"]), len(rec["o"])
        si[i, :ns], oxygen[i, :no] = torch.from_numpy(rec["si"]), torch.from_numpy(rec["o"])
        sm[i, :ns], om[i, :no] = True, True
        cp[i], idx[i], spg[i] = torch.from_numpy(rec["cell_parameters"]), rec["source_index"], rec["spg"]
    return si, sm, oxygen, om, cp, idx, spg


def make_loader(records, batch_size, shuffle, num_workers, pin_memory, seed, bucket_multiplier):
    sampler = BucketBatchSampler(records, batch_size, shuffle, seed, bucket_multiplier)
    loader = DataLoader(RecordDataset(records), batch_sampler=sampler, collate_fn=collate_records,
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return loader, sampler


def autocast_context(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(enabled and device.type == "cuda"))


def run_epoch(model, loader, device, args, optimizer=None, scaler=None, density_scale=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {k: 0.0 for k in LOSS_KEYS}
    seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for si, sm, oxygen, om, cp, _, _ in loader:
            si, sm, oxygen, om, cp = [x.to(device, non_blocking=True) for x in (si, sm, oxygen, om, cp)]
            if training: optimizer.zero_grad(set_to_none=True)
            # Keep the inexpensive MLP in AMP, but evaluate periodic geometry
            # and all distance-based losses in float32. FP16 sqrt/top-k geometry
            # can produce singular or nonfinite gradients near coincident proposals.
            with autocast_context(device, args.amp):
                vectors, _ = model(si, sm, cp)
            with torch.autocast(device_type="cuda", enabled=False):
                loss, pieces = local_proposal_loss(
                    vectors.float(), sm, oxygen.float(), om, si.float(), cp.float(),
                    vector_weight=args.vector_weight,
                    tetrahedral_weight=args.tetra_weight,
                    bridge_weight=args.bridge_weight,
                    bond_weight=args.bond_weight,
                    osi_second_weight=args.osi_second_weight * density_scale,
                    osi_third_weight=args.osi_third_weight * density_scale,
                    oo_second_weight=args.oo_second_weight * density_scale,
                )
            if not torch.isfinite(loss):
                detail = ", ".join(f"{k}={v}" for k, v in pieces.items())
                raise FloatingPointError(f"Nonfinite local-proposal loss ({detail})")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            bsz = len(si); seen += bsz
            totals["loss"] += float(loss.detach()) * bsz
            for k in LOSS_KEYS[1:]: totals[k] += pieces[k] * bsz
    return {k: v / seen for k, v in totals.items()}


def nearest_metrics(si, oxygen, cell, cutoff=2.4):
    so = stable_distance(periodic_cartesian_delta(si, oxygen, cell))
    sorted_si, sorted_o = torch.sort(so, dim=1).values, torch.sort(so, dim=0).values
    si4 = (so < cutoff).sum(dim=1).eq(4).float().mean()
    o2 = (so < cutoff).sum(dim=0).eq(2).float().mean()
    oo = stable_distance(periodic_cartesian_delta(oxygen, oxygen, cell))
    oo.fill_diagonal_(float("inf")); oo_nn = oo.min(dim=1).values
    return {
        "Si4": float(si4), "O2": float(o2), "joint_proxy": float(0.5 * (si4 + o2)),
        "SiO_first4_mean": float(sorted_si[:, :4].mean()),
        "SiO_fifth_mean": float(sorted_si[:, 4].mean()) if sorted_si.shape[1] > 4 else float("nan"),
        "OSi_first2_mean": float(sorted_o[:2].mean()),
        "OSi_third_mean": float(sorted_o[2].mean()) if sorted_o.shape[0] > 2 else float("nan"),
        "OO_nearest_mean": float(oo_nn.mean()),
        "OO_nearest_lt1_fraction": float((oo_nn < 1.0).float().mean()),
    }


def evaluate_structures(model, loader, device, amp, output_dir):
    model.eval(); rows = []
    with torch.no_grad():
        for si, sm, oxygen, om, cp, idx, spg in loader:
            si, sm, oxygen, om, cp = [x.to(device) for x in (si, sm, oxygen, om, cp)]
            with autocast_context(device, amp):
                proposals, pmask, parents = model.proposals_fractional(si, sm, cp)
            cells = cell_matrix_from_parameters(cp.float())
            for i in range(len(si)):
                s = si[i, sm[i]].float(); ref = oxygen[i, om[i]].float()
                p = proposals[i][pmask[i]].float(); par = parents[i][pmask[i]].long()
                merged = merge_local_proposals(p, par, cells[i])
                row = {"source_index": int(idx[i]), "spg": int(spg[i]), "n_si": len(s), "n_o": len(merged)}
                row.update({f"pred_{k}": v for k, v in nearest_metrics(s, merged, cells[i]).items()})
                row.update({f"teacher_{k}": v for k, v in nearest_metrics(s, ref, cells[i]).items()})
                rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(output_dir / "validation_structure_metrics.csv", index=False)
    summary = {"n_structures": int(len(df))}
    for col in df.columns:
        if col.startswith("pred_") or col.startswith("teacher_"): summary[col] = float(df[col].mean())
    with open(output_dir / "validation_structure_summary.json", "w") as f: json.dump(summary, f, indent=2)
    return summary


def save_checkpoint(path, model, optimizer, scaler, epoch, best, args, split):
    torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "epoch": epoch,
                "best_validation_loss": best, "args": vars(args), "split": split}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="data/train/sio2_tetra_shuffled_seed42.csv")
    p.add_argument("--chunk-size", type=int, default=100); p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2); p.add_argument("--bucket-multiplier", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--token-dim", type=int, default=128)
    p.add_argument("--neighbor-count", type=int, default=12); p.add_argument("--max-vector", type=float, default=2.8)
    p.add_argument("--bond-center", type=float, default=1.62, help="Central Si--O bond length for the preset tetrahedron")
    p.add_argument("--bond-range", type=float, default=0.35, help="Maximum learned bond-length deviation")
    p.add_argument("--max-distortion", type=float, default=0.12, help="Maximum tangential distortion of each tetrahedral direction")
    p.add_argument("--max-rotation", type=float, default=math.pi, help="Maximum axis-angle rotation magnitude")
    p.add_argument("--vector-weight", type=float, default=1.0); p.add_argument("--tetra-weight", type=float, default=0.2)
    p.add_argument("--bridge-weight", type=float, default=0.2); p.add_argument("--bond-weight", type=float, default=0.2)
    p.add_argument("--osi-second-weight", type=float, default=0.2, help="Match the proposal O--Si second-neighbor scale")
    p.add_argument("--osi-third-weight", type=float, default=0.3, help="Keep the proposal O--Si third neighbor outside the teacher shell")
    p.add_argument("--oo-second-weight", type=float, default=0.3, help="Repel the second cross-parent O proposal beyond the teacher nearest-O scale")
    p.add_argument("--density-warmup-epochs", type=int, default=50, help="Linearly ramp OSi2/OSi3/OO2 weights from zero")
    p.add_argument("--grad-clip", type=float, default=5.0); p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min-delta", type=float, default=0.01); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="o_prior_local"); p.add_argument("--no-amp", action="store_true")
    p.add_argument("--evaluate-only", action="store_true", help="Skip training and evaluate an existing checkpoint")
    p.add_argument("--checkpoint", default=None, help="Checkpoint for --evaluate-only; defaults to OUTPUT/O_local_prior_best.pt")
    args = p.parse_args(); args.amp = not args.no_amp

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    records, total_rows = load_atom_sets(args.train_data)
    train_records, val_records, split = split_records_by_chunk(records, total_rows, args.chunk_size, args.val_fraction, args.seed)
    print(f"Chunk split: {len(train_records)} train / {len(val_records)} validation structures; chunk_size={args.chunk_size}")
    max_si, max_o = max(len(r["si"]) for r in records), max(len(r["o"]) for r in records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Architecture: local Si neighborhoods -> rotated bounded SiO4 tetrahedron; K={args.neighbor_count}")
    print(f"Tetrahedron: bond={args.bond_center:.3f}+/-{args.bond_range:.3f} A; max_distortion={args.max_distortion:.3f}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "") + f"; AMP={'on' if args.amp else 'off'}")

    train_loader, train_sampler = make_loader(train_records, args.batch_size, True, args.num_workers, device.type == "cuda", args.seed, args.bucket_multiplier)
    val_loader, _ = make_loader(val_records, args.batch_size, False, args.num_workers, device.type == "cuda", args.seed, args.bucket_multiplier)
    model = LocalOxygenPrior(
        max_si, max_o, token_dim=args.token_dim, neighbor_count=args.neighbor_count,
        max_vector=args.max_vector, bond_center=args.bond_center, bond_range=args.bond_range,
        max_distortion=args.max_distortion, max_rotation=args.max_rotation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with open(output / "split.json", "w") as f: json.dump(split, f, indent=2)

    if args.evaluate_only:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else output / "O_local_prior_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        summary = evaluate_structures(model, val_loader, device, args.amp, output)
        print(f"Evaluated checkpoint: {checkpoint_path}")
        print(f"Held-out structural metrics: {output / 'validation_structure_summary.json'}")
        return

    history, best, best_epoch, stale = [], float("inf"), 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        density_scale = min(1.0, epoch / max(1, args.density_warmup_epochs))
        train_m = run_epoch(model, train_loader, device, args, optimizer, scaler, density_scale=density_scale)
        validate = epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs
        if validate:
            val_m = run_epoch(model, val_loader, device, args, density_scale=density_scale)
            improved = val_m["loss"] < best - args.min_delta
            if improved:
                best, best_epoch, stale = val_m["loss"], epoch, 0
                save_checkpoint(output / "O_local_prior_best.pt", model, optimizer, scaler, epoch, best, args, split)
            else: stale += 1
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:4d} | train {train_m['loss']:.4f} (vec {train_m['vector']:.4f}, bridge {train_m['bridge']:.4f}, bond {train_m['bond']:.4f}, tet {train_m['tetra']:.4f}, OSi2 {train_m['osi2']:.4f}, OSi3 {train_m['osi3']:.4f}, OO2 {train_m['oo2']:.4f}) | val {val_m['loss']:.4f} (vec {val_m['vector']:.4f}, bridge {val_m['bridge']:.4f}, bond {val_m['bond']:.4f}, tet {val_m['tetra']:.4f}, OSi2 {val_m['osi2']:.4f}, OSi3 {val_m['osi3']:.4f}, OO2 {val_m['oo2']:.4f}) | best {best:.4f} @ {best_epoch} | stale {stale}/{args.patience}")
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
            history.append(row); pd.DataFrame(history).to_csv(output / "prior_loss.csv", index=False)
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break

    checkpoint = torch.load(output / "O_local_prior_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    summary = evaluate_structures(model, val_loader, device, args.amp, output)
    save_checkpoint(output / "O_local_prior_last.pt", model, optimizer, scaler, checkpoint["epoch"], checkpoint["best_validation_loss"], args, split)
    with open(output / "training_summary.json", "w") as f:
        json.dump({"best_epoch": checkpoint["epoch"], "best_validation_loss": checkpoint["best_validation_loss"], "elapsed_seconds": time.time() - start, "structure_summary": summary}, f, indent=2)
    print(f"Best checkpoint: {output / 'O_local_prior_best.pt'}")
    print(f"Held-out structural metrics: {output / 'validation_structure_summary.json'}")
    print(f"Training script completed in {(time.time() - start) / 60.0:.1f} minutes.")


if __name__ == "__main__": main()

