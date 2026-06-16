#!/usr/bin/env python3

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from lego.GAN import GAN
from lego.VAE import VAE


BASE_COLUMNS = [
    "spg",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
]


def find_indexed_columns(columns, prefix):
    """Return sorted integer indices for columns named f'{prefix}<index>'."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    indices = []

    for column in columns:
        match = pattern.match(str(column))
        if match:
            indices.append(int(match.group(1)))

    return sorted(indices)


def require_contiguous_indices(indices, prefix):
    """Require indexed columns to start at zero and contain no gaps."""
    if not indices:
        raise ValueError(f"No {prefix}<index> columns were found.")

    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"{prefix} columns must be contiguous from {prefix}0 to "
            f"{prefix}{indices[-1]}; found indices {indices}."
        )


def validate_joint_layout(df):
    """Validate LEGO structural columns and site-specific target labels."""
    missing_base = set(BASE_COLUMNS) - set(df.columns)
    if missing_base:
        raise ValueError(
            "Input CSV is missing required base columns: "
            f"{sorted(missing_base)}"
        )

    wp_indices = find_indexed_columns(df.columns, "wp")
    target_indices = find_indexed_columns(df.columns, "target_coord")

    require_contiguous_indices(wp_indices, "wp")
    require_contiguous_indices(target_indices, "target_coord")

    if wp_indices != target_indices:
        raise ValueError(
            "Wyckoff and target-coordination columns do not match. "
            f"wp indices={wp_indices}; target_coord indices={target_indices}."
        )

    num_wps = len(wp_indices)

    for index in range(num_wps):
        required_site_columns = {
            f"wp{index}",
            f"x{index}",
            f"y{index}",
            f"z{index}",
            f"target_coord{index}",
        }
        missing_site = required_site_columns - set(df.columns)
        if missing_site:
            raise ValueError(
                f"Missing columns for Wyckoff slot {index}: "
                f"{sorted(missing_site)}"
            )

    allowed_columns = set(BASE_COLUMNS)
    for index in range(num_wps):
        allowed_columns.update(
            {
                f"wp{index}",
                f"x{index}",
                f"y{index}",
                f"z{index}",
                f"target_coord{index}",
            }
        )

    extra_columns = [column for column in df.columns if column not in allowed_columns]
    if extra_columns:
        raise ValueError(
            "Unexpected columns were found in the training CSV: "
            f"{extra_columns}. This joint-generation script currently expects "
            "only LEGO structural columns plus target_coord columns."
        )

    return num_wps


def validate_target_coordination(df, num_wps):
    """Validate consistency between occupied WP slots and CN3/CN4 labels."""
    errors = []

    for index in range(num_wps):
        wp_col = f"wp{index}"
        target_col = f"target_coord{index}"

        wp_values = pd.to_numeric(df[wp_col], errors="coerce")
        target_values = pd.to_numeric(df[target_col], errors="coerce")

        if wp_values.isna().any():
            rows = wp_values[wp_values.isna()].index[:10].tolist()
            errors.append(f"{wp_col} contains non-numeric values at rows {rows}")

        if target_values.isna().any():
            rows = target_values[target_values.isna()].index[:10].tolist()
            errors.append(
                f"{target_col} contains non-numeric values at rows {rows}"
            )

        occupied = wp_values != -1
        empty = ~occupied

        invalid_occupied = occupied & ~target_values.isin([3, 4])
        if invalid_occupied.any():
            rows = invalid_occupied[invalid_occupied].index[:10].tolist()
            errors.append(
                f"{target_col} must be 3 or 4 for occupied {wp_col} slots; "
                f"invalid rows {rows}"
            )

        invalid_empty = empty & (target_values != 0)
        if invalid_empty.any():
            rows = invalid_empty[invalid_empty].index[:10].tolist()
            errors.append(
                f"{target_col} must be 0 when {wp_col}=-1; invalid rows {rows}"
            )

    if errors:
        raise ValueError(
            "Invalid site-specific target coordination data:\n- "
            + "\n- ".join(errors)
        )


def print_target_statistics(df, num_wps):
    """Print compact diagnostics for the joint CN3/CN4 training labels."""
    target_columns = [f"target_coord{i}" for i in range(num_wps)]
    values = df[target_columns].to_numpy(dtype=int)

    occupied = values > 0
    cn3_count = int(np.sum(values == 3))
    cn4_count = int(np.sum(values == 4))
    occupied_count = int(np.sum(occupied))

    rows_with_cn3 = np.any(values == 3, axis=1)
    rows_with_cn4 = np.any(values == 4, axis=1)
    mixed_rows = rows_with_cn3 & rows_with_cn4

    patterns, counts = np.unique(values, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]

    print("Target-coordination statistics:")
    print(f"  Occupied site labels: {occupied_count}")
    print(f"  CN3 labels: {cn3_count}")
    print(f"  CN4 labels: {cn4_count}")
    print(f"  Mixed CN3/CN4 rows: {int(np.sum(mixed_rows))} / {len(df)}")
    print(f"  Unique padded target patterns: {len(patterns)}")
    print("  Most common target patterns:")

    for pattern_index in order[:10]:
        pattern = patterns[pattern_index].tolist()
        count = int(counts[pattern_index])
        print(f"    {pattern}: {count}")

    print()



def encode_skeleton_token(df, num_wps):
    """Bind space group, all Wyckoff slots, and all target CN labels."""
    encoded = df.copy()
    token_parts = [
        pd.to_numeric(encoded["spg"], errors="raise").astype(int).astype(str)
    ]
    drop_columns = ["spg"]

    for index in range(num_wps):
        wp_col = f"wp{index}"
        target_col = f"target_coord{index}"
        wp_values = pd.to_numeric(encoded[wp_col], errors="raise").astype(int)
        target_values = pd.to_numeric(
            encoded[target_col], errors="raise"
        ).astype(int)
        token_parts.append(
            wp_values.astype(str) + ":" + target_values.astype(str)
        )
        drop_columns.extend([wp_col, target_col])

    skeleton = token_parts[0]
    for part in token_parts[1:]:
        skeleton = skeleton + "|" + part

    encoded["skeleton_token"] = skeleton
    encoded = encoded.drop(columns=drop_columns)
    return encoded


def decode_skeleton_token(df, num_wps):
    """Split generated skeleton tokens back into spg, WP, and target CN."""
    decoded = df.copy()
    token_col = "skeleton_token"

    if token_col not in decoded.columns:
        raise ValueError(
            "Synthetic data is missing required column skeleton_token."
        )

    token_values = decoded[token_col].astype(str).str.strip()
    expected_pattern = r"^(-?\d+)" + "".join(
        [r"\|(-?\d+):([034])" for _ in range(num_wps)]
    ) + r"$"
    parts = token_values.str.extract(expected_pattern)

    invalid = parts.isna().any(axis=1)
    if invalid.any():
        rows = invalid[invalid].index[:10].tolist()
        examples = token_values.loc[rows].tolist()
        raise ValueError(
            f"Malformed skeleton_token values at rows {rows}: {examples}"
        )

    decoded["spg"] = parts[0].astype(int)
    for index in range(num_wps):
        decoded[f"wp{index}"] = parts[1 + 2 * index].astype(int)
        decoded[f"target_coord{index}"] = parts[2 + 2 * index].astype(int)

    decoded = decoded.drop(columns=[token_col])
    return decoded


def restore_original_column_order(df, original_columns):
    """Return decoded samples in exactly the original training CSV order."""
    missing = [column for column in original_columns if column not in df.columns]
    extra = [column for column in df.columns if column not in original_columns]

    if missing or extra:
        raise ValueError(
            "Decoded synthetic columns do not match the training CSV. "
            f"Missing={missing}; extra={extra}."
        )

    return df.loc[:, list(original_columns)].copy()



def get_contiguous_site_mask(df, num_wps):
    """Return rows whose occupied WP slots form a contiguous prefix."""
    wp_columns = [f"wp{i}" for i in range(num_wps)]
    wp_values = df[wp_columns].apply(pd.to_numeric, errors="coerce")

    valid = ~wp_values.isna().any(axis=1)
    seen_empty = np.zeros(len(df), dtype=bool)
    contiguous = np.ones(len(df), dtype=bool)

    for column in wp_columns:
        occupied = wp_values[column].to_numpy() != -1
        contiguous &= ~(seen_empty & occupied)
        seen_empty |= ~occupied

    return pd.Series(valid.to_numpy() & contiguous, index=df.index)


def validate_contiguous_sites(df, num_wps, context="data"):
    """Require occupied WP slots to precede all padded empty slots."""
    mask = get_contiguous_site_mask(df, num_wps)
    if not mask.all():
        rows = mask.index[~mask][:10].tolist()
        raise ValueError(
            f"Non-contiguous occupied Wyckoff slots found in {context}; "
            f"example rows {rows}. Once wp_i=-1, all later slots must be -1."
        )


def decode_and_validate_samples(
    sampled,
    num_wps,
    original_columns,
    training_df,
    output_discrete_columns,
):
    """Decode tokens, restore dtypes, and return only contiguous valid rows."""
    sampled = sampled.copy()
    sampled.columns = sampled.columns.astype(str).str.replace(
        " ", "", regex=False
    )
    sampled = decode_skeleton_token(sampled, num_wps)
    sampled = restore_original_column_order(sampled, original_columns)
    sampled = restore_sample_dtypes(
        sampled, training_df, output_discrete_columns
    )
    validate_target_coordination(sampled, num_wps)

    contiguous_mask = get_contiguous_site_mask(sampled, num_wps)
    sampled = sampled.loc[contiguous_mask].copy()
    sampled, wrapped_values, reset_values = normalize_sample_coordinates(
        sampled, num_wps
    )

    return (
        sampled,
        int((~contiguous_mask).sum()),
        wrapped_values,
        reset_values,
    )


def normalize_sample_coordinates(df, num_wps):
    """Wrap occupied fractional coordinates and reset padded slots.

    Occupied-site coordinates are periodic fractional coordinates, so values
    outside [0, 1) are wrapped with modulo 1. Empty padded slots are restored
    exactly to the LEGO sentinel coordinates (-1, -1, -1).
    """
    output = df.copy()
    wrapped_values = 0
    reset_values = 0

    for index in range(num_wps):
        wp_col = f"wp{index}"
        coord_cols = [f"x{index}", f"y{index}", f"z{index}"]

        wp_values = pd.to_numeric(output[wp_col], errors="raise")
        occupied = wp_values != -1
        empty = ~occupied

        for coord_col in coord_cols:
            values = pd.to_numeric(output[coord_col], errors="coerce")

            invalid_occupied = occupied & ~np.isfinite(values)
            if invalid_occupied.any():
                rows = invalid_occupied[invalid_occupied].index[:10].tolist()
                raise ValueError(
                    f"{coord_col} contains non-finite values for occupied "
                    f"sites at rows {rows}."
                )

            occupied_values = values.loc[occupied].to_numpy(dtype=float)
            wrapped = np.mod(occupied_values, 1.0)
            wrapped_values += int(
                np.count_nonzero(~np.isclose(occupied_values, wrapped))
            )
            output.loc[occupied, coord_col] = wrapped

            reset_values += int(
                np.count_nonzero(
                    ~np.isclose(values.loc[empty].to_numpy(dtype=float), -1.0)
                )
            )
            output.loc[empty, coord_col] = -1.0

    return output, wrapped_values, reset_values

def restore_sample_dtypes(df_synthetic, df_training, discrete_columns):
    """Restore stable numeric dtypes after synthesis without stringifying data."""
    output = df_synthetic.copy()

    for column in output.columns:
        if column not in df_training.columns:
            continue

        output[column] = pd.to_numeric(output[column], errors="raise")

        if column in discrete_columns:
            output[column] = np.rint(output[column]).astype(int)
        else:
            output[column] = output[column].astype(float)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="LEGO crystal synthesizer with a protected categorical skeleton"
    )

    parser.add_argument("--data", required=True, help="Input CSV dataset.")
    parser.add_argument(
        "--model",
        choices=["GAN", "VAE", "gan", "vae"],
        default="GAN",
        help="Generative model: GAN or VAE.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of training epochs. Default: 250.",
    )
    parser.add_argument(
        "--nbatch",
        type=int,
        default=500,
        help="Training batch size. Default: 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
        help="Optional maximum number of training rows.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100000,
        help="Number of synthetic rows to generate. Default: 100000.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Input CSV does not exist: {args.data}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.nbatch <= 0:
        raise ValueError("--nbatch must be greater than zero.")
    if args.cutoff is not None and args.cutoff <= 0:
        raise ValueError("--cutoff must be greater than zero.")
    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be greater than zero.")

    model = args.model.upper()
    data_name = os.path.splitext(os.path.basename(args.data))[0]

    model_root = os.path.join("models", data_name)
    sample_root = os.path.join("data", "sample")

    os.makedirs(model_root, exist_ok=True)
    os.makedirs(sample_root, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
        print("CUDA is available.")

    print(f"CUDA available: {cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if cuda:
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        print(f"Device name: {torch.cuda.get_device_name(current_device)}")

    df = pd.read_csv(args.data)

    if args.cutoff is not None and len(df) > args.cutoff:
        print(f"Selecting the first {args.cutoff} of {len(df)} rows for training.")
        df = df.iloc[: args.cutoff].copy()

    if df.empty:
        raise ValueError("Input dataset is empty after applying --cutoff.")

    df.columns = df.columns.astype(str).str.strip()
    num_wps = validate_joint_layout(df)
    validate_target_coordination(df, num_wps)
    validate_contiguous_sites(df, num_wps, context="training data")

    print(f"Data shape: {df.shape}\n")
    print(f"Data head:\n{df.head()}\n")

    original_columns = df.columns.tolist()
    df_model = encode_skeleton_token(df, num_wps)

    dis_cols = ["skeleton_token"]

    first_a = float(df_model["a"].iloc[0])
    if abs(first_a - round(first_a)) < 1e-2:
        discrete_cell = True
        dis_cols.extend(["a", "b", "c", "alpha", "beta", "gamma"])
    else:
        discrete_cell = False

    discrete_coordinates = (
        pd.to_numeric(df_model["x0"], errors="raise").max() >= 2.5 + 1e-3
    )

    for index in range(num_wps):
        x_col = f"x{index}"
        y_col = f"y{index}"
        z_col = f"z{index}"

        if discrete_coordinates:
            dis_cols.extend([x_col, y_col, z_col])

    print(f"Number of Wyckoff slots: {num_wps}")
    print(f"Discrete cell parameters: {discrete_cell}")
    print(f"Discrete coordinates: {discrete_coordinates}")
    print(f"Number of discrete columns: {len(dis_cols)}")
    print(f"Discrete columns: {dis_cols}\n")

    print_target_statistics(df, num_wps)
    skeleton_counts = df_model["skeleton_token"].value_counts()
    print("Categorical-skeleton statistics:")
    print(f"  Unique observed skeletons: {len(skeleton_counts)}")
    print(f"  Most frequent skeleton count: {int(skeleton_counts.iloc[0])}")
    print()

    model_folder = os.path.join(model_root, model)
    os.makedirs(model_folder, exist_ok=True)

    if model == "GAN":
        synthesizer = GAN(
            embedding_dim=128,
            generator_dim=(512, 512),
            discriminator_dim=(512, 512),
            generator_lr=2e-4,
            generator_decay=1e-6,
            discriminator_lr=2e-4,
            discriminator_decay=1e-6,
            batch_size=args.nbatch,
            discriminator_steps=1,
            log_frequency=True,
            verbose=True,
            epochs=args.epochs,
            pac=10,
            cuda=cuda,
            folder=model_folder,
        )
    elif model == "VAE":
        synthesizer = VAE(
            embedding_dim=128,
            compress_dims=(512, 512),
            decompress_dims=(512, 512),
            l2scale=1e-5,
            loss_factor=2,
            epochs=args.epochs,
            verbose=True,
            cuda=cuda,
            batch_size=args.nbatch,
            folder=model_folder,
        )
    else:
        raise RuntimeError(f"Only GAN and VAE are supported, not {args.model}")

    print(f"Training {model} model.")
    print(f"Model output directory: {model_folder}")

    synthesizer.fit(df_model, discrete_columns=dis_cols)

    synthetic_data_size = len(df) if args.sample is None else args.sample

    output_discrete_columns = ["spg"]
    if discrete_cell:
        output_discrete_columns.extend(
            ["a", "b", "c", "alpha", "beta", "gamma"]
        )
    for index in range(num_wps):
        output_discrete_columns.extend(
            [f"wp{index}", f"target_coord{index}"]
        )
        if discrete_coordinates:
            output_discrete_columns.extend(
                [f"x{index}", f"y{index}", f"z{index}"]
            )

    print(
        f"Generating {synthetic_data_size} valid synthetic rows with "
        "a protected space-group/Wyckoff/target skeleton token."
    )

    accepted_batches = []
    accepted_count = 0
    total_generated = 0
    rejected_noncontiguous = 0
    wrapped_coordinate_values = 0
    reset_padded_coordinate_values = 0
    max_sample_rounds = 20

    for sample_round in range(1, max_sample_rounds + 1):
        remaining = synthetic_data_size - accepted_count
        if remaining <= 0:
            break

        batch_size = max(remaining, int(np.ceil(remaining * 1.25)))
        sampled = synthesizer.sample(samples=batch_size)
        if sampled is None or len(sampled) == 0:
            raise RuntimeError(
                f"The synthesizer returned no samples in round {sample_round}."
            )

        total_generated += len(sampled)
        (
            valid_batch,
            rejected,
            wrapped_values,
            reset_values,
        ) = decode_and_validate_samples(
            sampled,
            num_wps,
            original_columns,
            df,
            output_discrete_columns,
        )
        rejected_noncontiguous += rejected
        wrapped_coordinate_values += wrapped_values
        reset_padded_coordinate_values += reset_values

        if not valid_batch.empty:
            accepted_batches.append(valid_batch)
            accepted_count += len(valid_batch)

        print(
            f"  Sampling round {sample_round}: generated {len(sampled)}, "
            f"accepted {len(valid_batch)}, cumulative {accepted_count}/"
            f"{synthetic_data_size}"
        )

    if accepted_count < synthetic_data_size:
        acceptance = accepted_count / total_generated if total_generated else 0.0
        raise RuntimeError(
            "Could not obtain the requested number of contiguous synthetic "
            f"rows after {max_sample_rounds} rounds: accepted "
            f"{accepted_count}/{synthetic_data_size}; overall acceptance "
            f"rate={acceptance:.1%}."
        )

    df_synthetic = pd.concat(accepted_batches, ignore_index=True)
    df_synthetic = df_synthetic.iloc[:synthetic_data_size].copy()

    acceptance = accepted_count / total_generated
    print(
        "Synthetic sampling summary:\n"
        f"  Requested valid rows: {synthetic_data_size}\n"
        f"  Total generated rows: {total_generated}\n"
        f"  Rejected invalid/non-contiguous rows: {rejected_noncontiguous}\n"
        f"  Wrapped occupied coordinate values: "
        f"{wrapped_coordinate_values}\n"
        f"  Reset padded coordinate values: "
        f"{reset_padded_coordinate_values}\n"
        f"  Effective retained fraction: {acceptance:.1%}"
    )
    print("Synthetic data sample:\n" f"{df_synthetic.head(10)}\n")

    output_file = os.path.join(
        sample_root,
        (
            f"{data_name}-{model}-skeleton-dis{len(dis_cols)}-"
            f"seed{args.seed}-{synthetic_data_size}.csv"
        ),
    )

    output_parent = os.path.dirname(output_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    print(f"Saving {synthetic_data_size} samples to {output_file}")
    df_synthetic.to_csv(output_file, index=False, header=True)

    print(f"Saved synthetic dataset: {output_file}")


if __name__ == "__main__":
    main()

