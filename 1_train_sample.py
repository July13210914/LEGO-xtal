#!/usr/bin/env python3
"""Train and sample a factorized SiO2 LEGO-Xtal VAE.

This version canonicalizes independent sites as Si(CN4) -> O(CN2) -> padding,
then trains three blocks:
    global: space group and cell
    Si:     Si Wyckoff skeleton and generating coordinates
    O:      O Wyckoff skeleton and generating coordinates

The O block is decoded conditionally on the generated global and Si blocks.
Hard Wyckoff-multiplicity masking is intentionally deferred to v2.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from lego.VAE_factorized import FactorizedVAE


BASE_COLUMNS = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
SI_CN = 4
O_CN = 2


def find_indexed_columns(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    return sorted(
        int(match.group(1))
        for column in columns
        if (match := pattern.match(str(column)))
    )


def validate_layout(df):
    missing = set(BASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing base columns: {sorted(missing)}")
    wp_indices = find_indexed_columns(df.columns, "wp")
    target_indices = find_indexed_columns(df.columns, "target_coord")
    if not wp_indices or wp_indices != list(range(len(wp_indices))):
        raise ValueError(f"wp columns must be contiguous from wp0; found {wp_indices}")
    if target_indices != wp_indices:
        raise ValueError(
            f"target_coord indices do not match wp indices: {target_indices} vs {wp_indices}"
        )
    for i in wp_indices:
        required = {f"wp{i}", f"x{i}", f"y{i}", f"z{i}", f"target_coord{i}"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing site columns for slot {i}: {sorted(missing)}")
    return len(wp_indices)


def canonicalize_species_order(df, num_wps):
    """Reorder each row as all CN4 sites, then all CN2 sites, then padding."""
    output = df.copy()
    allowed = {0, SI_CN, O_CN}
    n_si_max = 0
    n_o_max = 0
    rows = []

    for row_index, row in df.iterrows():
        si_sites = []
        o_sites = []
        for i in range(num_wps):
            wp = int(row[f"wp{i}"])
            cn = int(row[f"target_coord{i}"])
            xyz = tuple(float(row[f"{axis}{i}"]) for axis in "xyz")
            if cn not in allowed:
                raise ValueError(
                    f"Row {row_index}, slot {i}: unsupported target_coord={cn}; "
                    "factorized SiO2 v1 expects only 4, 2, or 0."
                )
            if wp == -1:
                if cn != 0:
                    raise ValueError(
                        f"Row {row_index}, slot {i}: wp=-1 requires target_coord=0."
                    )
                continue
            if cn == SI_CN:
                si_sites.append((wp, xyz))
            elif cn == O_CN:
                o_sites.append((wp, xyz))
            else:
                raise ValueError(
                    f"Row {row_index}, slot {i}: occupied wp={wp} has target_coord=0."
                )

        if not si_sites or not o_sites:
            raise ValueError(
                f"Row {row_index} lacks one species: n_Si={len(si_sites)}, n_O={len(o_sites)}"
            )
        n_si_max = max(n_si_max, len(si_sites))
        n_o_max = max(n_o_max, len(o_sites))
        rows.append((si_sites, o_sites))

    for row_pos, (si_sites, o_sites) in enumerate(rows):
        ordered = [(wp, xyz, SI_CN) for wp, xyz in si_sites]
        ordered += [(wp, xyz, O_CN) for wp, xyz in o_sites]
        ordered += [(-1, (-1.0, -1.0, -1.0), 0)] * (num_wps - len(ordered))
        idx = output.index[row_pos]
        for i, (wp, xyz, cn) in enumerate(ordered):
            output.at[idx, f"wp{i}"] = wp
            output.at[idx, f"target_coord{i}"] = cn
            for axis, value in zip("xyz", xyz):
                output.at[idx, f"{axis}{i}"] = value

    return output, n_si_max, n_o_max


def encode_wp_token(values):
    return "|".join(str(int(value)) for value in values)


def decode_wp_token(token, expected_slots, label):
    parts = str(token).strip().split("|")
    if len(parts) != expected_slots:
        raise ValueError(
            f"Malformed {label} token {token!r}: expected {expected_slots} entries."
        )
    try:
        return [int(value) for value in parts]
    except ValueError as exc:
        raise ValueError(f"Malformed integer in {label} token {token!r}") from exc


def build_factorized_blocks(df, num_wps, n_si_max, n_o_max):
    global_df = df[BASE_COLUMNS].copy()

    si_records = []
    o_records = []
    for _, row in df.iterrows():
        si_slots = []
        o_slots = []
        for i in range(num_wps):
            cn = int(row[f"target_coord{i}"])
            site = {
                "wp": int(row[f"wp{i}"]),
                "x": float(row[f"x{i}"]),
                "y": float(row[f"y{i}"]),
                "z": float(row[f"z{i}"]),
            }
            if cn == SI_CN:
                si_slots.append(site)
            elif cn == O_CN:
                o_slots.append(site)

        si_slots += [{"wp": -1, "x": -1.0, "y": -1.0, "z": -1.0}] * (
            n_si_max - len(si_slots)
        )
        o_slots += [{"wp": -1, "x": -1.0, "y": -1.0, "z": -1.0}] * (
            n_o_max - len(o_slots)
        )

        si_record = {"si_skeleton_token": encode_wp_token(s["wp"] for s in si_slots)}
        o_record = {"o_skeleton_token": encode_wp_token(s["wp"] for s in o_slots)}
        for i, site in enumerate(si_slots):
            for axis in "xyz":
                si_record[f"si_{axis}{i}"] = site[axis]
        for i, site in enumerate(o_slots):
            for axis in "xyz":
                o_record[f"o_{axis}{i}"] = site[axis]
        si_records.append(si_record)
        o_records.append(o_record)

    return global_df, pd.DataFrame(si_records), pd.DataFrame(o_records)


def blocks_to_lego_rows(global_df, si_df, o_df, num_wps, n_si_max, n_o_max):
    """Convert factorized samples to LEGO rows and reject slot overflow.

    The maximum Si and O block capacities are learned independently across the
    dataset and may sum to more than ``num_wps``. That is valid internally, but
    a sampled Si/O combination is retained only when its total number of
    occupied independent sites fits the original LEGO row width.
    """
    if not (len(global_df) == len(si_df) == len(o_df)):
        raise ValueError("Sampled block row counts differ.")

    records = []
    rejected_overflow = 0
    for row_index in range(len(global_df)):
        global_row = global_df.iloc[row_index]
        si_row = si_df.iloc[row_index]
        o_row = o_df.iloc[row_index]
        si_wps = decode_wp_token(
            si_row["si_skeleton_token"], n_si_max, "Si skeleton"
        )
        o_wps = decode_wp_token(
            o_row["o_skeleton_token"], n_o_max, "O skeleton"
        )

        sites = []
        for i, wp in enumerate(si_wps):
            xyz = [float(si_row[f"si_{axis}{i}"]) for axis in "xyz"]
            sites.append((wp, xyz, SI_CN))
        for i, wp in enumerate(o_wps):
            xyz = [float(o_row[f"o_{axis}{i}"]) for axis in "xyz"]
            sites.append((wp, xyz, O_CN))

        occupied = []
        for wp, xyz, cn in sites:
            if wp == -1:
                continue
            occupied.append((wp, [float(v) % 1.0 for v in xyz], cn))

        if len(occupied) > num_wps:
            rejected_overflow += 1
            continue

        record = {column: global_row[column] for column in BASE_COLUMNS}
        padded = [(-1, [-1.0, -1.0, -1.0], 0)] * (num_wps - len(occupied))
        for i, (wp, xyz, cn) in enumerate(occupied + padded):
            record[f"wp{i}"] = int(wp)
            record[f"x{i}"], record[f"y{i}"], record[f"z{i}"] = xyz
            record[f"target_coord{i}"] = int(cn)
        records.append(record)

    return pd.DataFrame(records), rejected_overflow


def restore_dtypes(df, training_df, num_wps, discrete_cell, discrete_coordinates):
    output = df.copy()
    integer_columns = ["spg"]
    if discrete_cell:
        integer_columns += ["a", "b", "c", "alpha", "beta", "gamma"]
    for i in range(num_wps):
        integer_columns += [f"wp{i}", f"target_coord{i}"]
        if discrete_coordinates:
            integer_columns += [f"x{i}", f"y{i}", f"z{i}"]
    for column in output.columns:
        output[column] = pd.to_numeric(output[column], errors="raise")
        if column in integer_columns:
            output[column] = np.rint(output[column]).astype(int)
        else:
            output[column] = output[column].astype(float)
    return output.loc[:, training_df.columns]


def main():
    parser = argparse.ArgumentParser(
        description="Factorized global -> Si -> O VAE for SiO2 LEGO-Xtal data"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--nbatch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--sample", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--context-end", type=float, default=0.8)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(args.data)
    if args.epochs <= 0 or args.nbatch <= 0 or args.sample <= 0:
        raise ValueError("--epochs, --nbatch, and --sample must be positive.")
    if not 0.0 <= args.context_end <= 1.0:
        raise ValueError("--context-end must lie in [0, 1].")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(args.data)
    df.columns = df.columns.astype(str).str.strip()
    if args.cutoff is not None:
        df = df.iloc[: args.cutoff].copy()
    if df.empty:
        raise ValueError("Training data is empty.")

    num_wps = validate_layout(df)
    canonical_df, n_si_max, n_o_max = canonicalize_species_order(df, num_wps)
    global_df, si_df, o_df = build_factorized_blocks(
        canonical_df, num_wps, n_si_max, n_o_max
    )

    first_a = float(global_df["a"].iloc[0])
    discrete_cell = abs(first_a - round(first_a)) < 1e-2
    coordinate_columns = [c for c in si_df.columns if re.match(r"si_[xyz]\d+$", c)]
    coordinate_columns += [c for c in o_df.columns if re.match(r"o_[xyz]\d+$", c)]
    coordinate_max = max(
        pd.to_numeric(pd.concat([si_df[c] for c in si_df.columns if c != "si_skeleton_token"] +
                                [o_df[c] for c in o_df.columns if c != "o_skeleton_token"]),
                      errors="coerce").max(),
        0.0,
    )
    discrete_coordinates = coordinate_max >= 2.5 + 1e-3

    global_discrete = ["spg"]
    if discrete_cell:
        global_discrete += ["a", "b", "c", "alpha", "beta", "gamma"]
    si_discrete = ["si_skeleton_token"]
    o_discrete = ["o_skeleton_token"]
    if discrete_coordinates:
        si_discrete += [c for c in si_df.columns if c != "si_skeleton_token"]
        o_discrete += [c for c in o_df.columns if c != "o_skeleton_token"]

    data_name = os.path.splitext(os.path.basename(args.data))[0]
    model_folder = os.path.join("models", data_name, "FactorizedVAE_v2")
    sample_folder = os.path.join("data", "sample")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(sample_folder, exist_ok=True)

    print(f"Rows: {len(df)}")
    print(f"Original slots: {num_wps}")
    print(f"Si block capacity: {n_si_max}")
    print(f"O block capacity: {n_o_max}")
    print(f"Global columns: {global_df.columns.tolist()}")
    print(f"Si columns: {si_df.columns.tolist()}")
    print(f"O columns: {o_df.columns.tolist()}")
    print(f"Discrete cell: {discrete_cell}")
    print(f"Discrete coordinates: {discrete_coordinates}")

    model = FactorizedVAE(
        embedding_dim=128,
        compress_dims=(512, 512),
        decompress_dims=(512, 512),
        context_dim=128,
        l2scale=1e-5,
        batch_size=args.nbatch,
        epochs=args.epochs,
        loss_factor=2.0,
        kl_weight=1.0,
        kl_warmup_epochs=min(50, args.epochs),
        predicted_context_start=0.0,
        predicted_context_end=args.context_end,
        cuda=torch.cuda.is_available(),
        verbose=True,
        folder=model_folder,
    )

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"FactorizedVAE device: {model._device}")
    
    if model._device.type == "cuda":
        device_index = torch.cuda.current_device()
        print(f"CUDA device index: {device_index}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device_index)}")

    model.fit(
        global_df,
        si_df,
        o_df,
        global_discrete_columns=global_discrete,
        si_discrete_columns=si_discrete,
        o_discrete_columns=o_discrete,
    )

    accepted_batches = []
    accepted_count = 0
    total_generated = 0
    rejected_overflow = 0
    max_sample_rounds = 20

    for sample_round in range(1, max_sample_rounds + 1):
        remaining = args.sample - accepted_count
        if remaining <= 0:
            break

        draw_size = max(remaining, int(np.ceil(remaining * 1.25)))
        sampled_global, sampled_si, sampled_o = model.sample(
            draw_size, temperature=args.temperature, hard=True
        )
        total_generated += draw_size
        valid_batch, rejected = blocks_to_lego_rows(
            sampled_global,
            sampled_si,
            sampled_o,
            num_wps,
            n_si_max,
            n_o_max,
        )
        rejected_overflow += rejected
        if not valid_batch.empty:
            accepted_batches.append(valid_batch)
            accepted_count += len(valid_batch)

        print(
            f"Sampling round {sample_round}: generated {draw_size}, "
            f"accepted {len(valid_batch)}, cumulative "
            f"{accepted_count}/{args.sample}"
        )

    if accepted_count < args.sample:
        acceptance = accepted_count / total_generated if total_generated else 0.0
        raise RuntimeError(
            "Could not obtain the requested number of samples that fit the "
            f"original {num_wps}-slot LEGO layout after {max_sample_rounds} "
            f"rounds: accepted {accepted_count}/{args.sample}; "
            f"acceptance={acceptance:.1%}."
        )

    synthetic = pd.concat(accepted_batches, ignore_index=True).iloc[: args.sample].copy()
    synthetic = restore_dtypes(
        synthetic,
        canonical_df,
        num_wps,
        discrete_cell,
        discrete_coordinates,
    )
    print(
        "Sampling summary:\n"
        f"  Requested rows: {args.sample}\n"
        f"  Total generated: {total_generated}\n"
        f"  Rejected slot-overflow combinations: {rejected_overflow}\n"
        f"  Retained fraction: {accepted_count / total_generated:.1%}"
    )

    output = os.path.join(
        sample_folder,
        f"{data_name}-FactorizedVAE-v2-seed{args.seed}-{args.sample}.csv",
    )
    synthetic.to_csv(output, index=False)
    final_model = os.path.join(model_folder, "models", "FactorizedVAE_final.pkl")
    model.save(final_model)
    print(f"Saved samples: {output}")
    print(f"Saved model: {final_model}")


if __name__ == "__main__":
    main()

