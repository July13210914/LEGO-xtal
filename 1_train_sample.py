#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import torch

from lego.GAN import GAN
from lego.VAE import VAE


def main():
    parser = argparse.ArgumentParser(description="Table Synthesizer")

    parser.add_argument(
        "--data",
        required=True,
        help="Input CSV dataset.",
    )
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

    # Reproducibility
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

    # Read dataset
    df = pd.read_csv(args.data)

    if args.cutoff is not None and len(df) > args.cutoff:
        print(
            f"Selecting the first {args.cutoff} of "
            f"{len(df)} rows for training."
        )
        df = df.iloc[:args.cutoff].copy()

    if df.empty:
        raise ValueError("Input dataset is empty after applying --cutoff.")

    required_columns = {
        "spg",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "x0",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"Data shape: {df.shape}\n")
    print(f"Data head:\n{df.head()}\n")

    # Determine the number of Wyckoff-site groups.
    #
    # Expected layout:
    # 7 lattice/symmetry columns +
    # 4 columns per Wyckoff site:
    # wp_i, x_i, y_i, z_i
    non_base_columns = len(df.columns) - 7

    if non_base_columns < 4 or non_base_columns % 4 != 0:
        raise ValueError(
            "Unexpected CSV column layout. Expected 7 base columns plus "
            "4 columns per Wyckoff site, but found "
            f"{len(df.columns)} total columns."
        )

    num_wps = non_base_columns // 4

    # Set up categorical/discrete columns
    dis_cols = ["spg"]

    first_a = df["a"].iloc[0]

    if abs(first_a - round(first_a)) < 1e-2:
        discrete_cell = True
        dis_cols.extend(
            ["a", "b", "c", "alpha", "beta", "gamma"]
        )
    else:
        discrete_cell = False

    discrete_coordinates = df["x0"].max() >= 2.5 + 1e-3

    for i in range(num_wps):
        wp_col = f"wp{i}"
        x_col = f"x{i}"
        y_col = f"y{i}"
        z_col = f"z{i}"

        site_columns = {wp_col, x_col, y_col, z_col}
        missing_site_columns = site_columns - set(df.columns)

        if missing_site_columns:
            raise ValueError(
                f"Missing columns for Wyckoff site {i}: "
                f"{sorted(missing_site_columns)}"
            )

        dis_cols.append(wp_col)

        if discrete_coordinates:
            dis_cols.extend([x_col, y_col, z_col])

    print(f"Number of Wyckoff slots: {num_wps}")
    print(f"Discrete cell parameters: {discrete_cell}")
    print(f"Discrete coordinates: {discrete_coordinates}")
    print(f"Number of discrete columns: {len(dis_cols)}")
    print(f"Discrete columns: {dis_cols}\n")

    # Initialize synthesizer
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
        raise RuntimeError(
            f"Only GAN and VAE are supported, not {args.model}"
        )

    print(f"Training {model} model.")
    print(f"Model output directory: {model_folder}")

    synthesizer.fit(
        df,
        discrete_columns=dis_cols,
    )

    # Generate synthetic rows
    synthetic_data_size = (
        len(df) if args.sample is None else args.sample
    )

    print(f"Generating {synthetic_data_size} synthetic rows.")

    df_synthetic = synthesizer.sample(
        samples=synthetic_data_size
    )

    if df_synthetic is None or len(df_synthetic) == 0:
        raise RuntimeError("The synthesizer returned no synthetic samples.")

    print(
        "Synthetic data sample:\n"
        f"{df_synthetic.head(10)}\n"
    )

    output_file = os.path.join(
        sample_root,
        (
            f"{data_name}-{model}-dis{len(dis_cols)}-"
            f"seed{args.seed}-{synthetic_data_size}.csv"
        ),
    )

    # Create the exact parent directory before saving.
    output_parent = os.path.dirname(output_file)

    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    print(
        f"Saving {synthetic_data_size} samples to "
        f"{output_file}"
    )

    df_synthetic.columns = (
        df_synthetic.columns
        .astype(str)
        .str.replace(" ", "", regex=False)
    )

    # Replace commas embedded inside cell values so they do not interfere
    # with the CSV representation.
    df_synthetic = df_synthetic.map(
        lambda value: str(value).replace(",", " ")
    )

    df_synthetic.to_csv(
        output_file,
        index=False,
        header=True,
    )

    print(f"Saved synthetic dataset: {output_file}")


if __name__ == "__main__":
    main()
