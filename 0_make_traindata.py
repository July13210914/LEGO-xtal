#!/usr/bin/env python3

import argparse
import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pyxtal.db import database_topology
from pyxtal.lego.builder import builder
from pyxtal.util import new_struc_wo_energy


def build_output_filename(output_dir, tag, discrete, discrete_cell):
    """Return the training CSV path for the selected representation mode."""
    if discrete and discrete_cell:
        suffix = "-discell.csv"
    elif discrete:
        suffix = "-dis.csv"
    else:
        suffix = ".csv"

    return os.path.join(output_dir, f"{tag}{suffix}")


def make_builder(prototype="graphite", cn=3, rcut=2.0):
    """Create and configure a LEGO builder for carbon environments."""
    reference = pyxtal()
    reference.from_prototype(prototype)
    reference_structure = reference.to_pymatgen()

    bu = builder(["C"], [1], verbose=False)
    bu.set_descriptor_calculator(mykwargs={"rcut": rcut})
    bu.set_reference_enviroments(reference_structure)
    bu.set_criteria(CN={"C": [cn]})
    return bu


def make_csv(
    total_reps,
    include_energy,
    include_label,
    discrete,
    discrete_cell,
    n_wp,
    filename,
):
    """Convert tabular representations to a typed CSV file."""
    total_reps = np.asarray(total_reps)

    if total_reps.ndim != 2:
        raise ValueError(
            f"Expected a 2D representation array, got shape {total_reps.shape}"
        )

    column_names = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
    float_cols = set()

    if not discrete_cell:
        float_cols.update(range(1, 7))

    for i in range(n_wp):
        base = 7 + 4 * i
        column_names.extend([f"wp{i}", f"x{i}", f"y{i}", f"z{i}"])

        if not discrete:
            float_cols.update([base + 1, base + 2, base + 3])

    if include_energy:
        column_names.append("energy")
        float_cols.add(len(column_names) - 1)

    if include_label:
        column_names.append("label")

    if total_reps.shape[1] != len(column_names):
        raise ValueError(
            "Representation width does not match the expected CSV layout: "
            f"{total_reps.shape[1]} values versus {len(column_names)} columns."
        )

    data = {}
    for i, column in enumerate(column_names):
        if i in float_cols:
            data[column] = total_reps[:, i].astype(float)
        else:
            data[column] = total_reps[:, i].astype(int)

    df = pd.DataFrame(data, columns=column_names)

    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} representations to {filename}")


def process_one_xtal(item, params, base_seed):
    """Process one source structure inside a multiprocessing worker."""
    source_index, xtal = item

    seed = base_seed + source_index
    random.seed(seed)
    np.random.seed(seed)

    try:
        reps = get_reps_from_xtal(xtal, params)
        return source_index, reps, None
    except Exception as exc:
        return source_index, [], f"{type(exc).__name__}: {exc}"


def get_reps_from_xtal(xtal, params):
    """Generate tabular representations for one structure and its subgroups."""
    (
        max_dof,
        n_atoms_min,
        n_atoms_max,
        max_energy,
        min_spg,
        n_wp,
        max_per_structure,
        include_energy,
        discrete,
        discrete_cell,
        discrete_resolution,
        subgroup_eps,
        prototype,
        cn,
        rcut,
    ) = params

    bu = make_builder(prototype=prototype, cn=cn, rcut=rcut)
    xtal_reps = []

    atom_count = sum(xtal.numIons)
    ff_energy = getattr(xtal, "ff_energy", None)
    filter_by_energy = np.isfinite(max_energy)

    energy_is_valid = (
        not filter_by_energy
        or (ff_energy is not None and ff_energy <= max_energy)
    )

    if not (
        xtal.dof <= max_dof
        and n_atoms_min <= atom_count <= n_atoms_max
        and energy_is_valid
        and xtal.group.number >= min_spg
        and len(xtal.atom_sites) <= n_wp
    ):
        return xtal_reps

    current_energy = ff_energy if include_energy else None

    xtal_opt, _, _ = bu.optimize_xtal(xtal, add_db=False)
    if xtal_opt is None or not xtal_opt.check_validity(bu.criteria):
        return xtal_reps

    n_wps = len(xtal_opt.atom_sites)
    n_max_initial = max(
        1,
        int(0.6 * max_per_structure * np.ceil(n_wps / n_wp)),
    )

    reps_initial = xtal_opt.get_tabular_representations(
        N_wp=n_wp,
        N_max=n_max_initial,
        discrete=discrete,
        discrete_cell=discrete_cell,
        N_grids=discrete_resolution,
    ) or []

    if include_energy and current_energy is not None:
        reps_initial = [
            np.append(rep, current_energy)
            for rep in reps_initial
        ]

    xtal_reps.extend(reps_initial)

    max_cell_factor = max(n_atoms_max / sum(xtal_opt.numIons), 1.0)
    trial_xtals_cache = [xtal_opt]

    for group_type in ("t", "k"):
        for _ in range(20):
            if len(xtal_reps) >= max_per_structure:
                return xtal_reps[:max_per_structure]

            xtal_sub = xtal_opt.subgroup_once(
                eps=subgroup_eps,
                group_type=group_type,
                max_cell=max_cell_factor,
                mut_lat=False,
            )

            if xtal_sub is None:
                xtal0 = xtal_opt.subgroup_once(group_type="t")
                if xtal0 is not None:
                    xtal_sub = xtal0.subgroup_once(
                        eps=subgroup_eps,
                        group_type="t",
                        max_cell=max_cell_factor,
                        mut_lat=False,
                    )

            if xtal_sub is None:
                continue

            lattice_parameters = xtal_sub.lattice.get_para(degree=True)
            lengths = lattice_parameters[:3]
            angles = lattice_parameters[3:]

            valid_geometry = (
                xtal_sub.get_dof() <= max_dof
                and len(xtal_sub.atom_sites) <= n_wp
                and max(lengths) < 50
                and min(angles) > 30
                and max(angles) < 150
            )

            if not valid_geometry:
                continue

            is_new = new_struc_wo_energy(
                xtal_sub,
                trial_xtals_cache,
                0.025,
                0.025,
                1.0,
            )

            if not is_new:
                continue

            try:
                xtal_sub_opt, _, _ = bu.optimize_xtal(
                    xtal_sub,
                    add_db=False,
                )
            except Exception as exc:
                spg = getattr(
                    getattr(xtal_sub, "group", None),
                    "number",
                    "unknown",
                )
                print(
                    f"Skipping subgroup with space group {spg}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if (
                xtal_sub_opt is None
                or not xtal_sub_opt.check_validity(bu.criteria)
            ):
                continue

            trial_xtals_cache.append(xtal_sub_opt)

            n_wps_sub = len(xtal_sub_opt.atom_sites)
            n_max_sub = max(
                1,
                int(
                    0.2
                    * max_per_structure
                    * np.ceil(n_wps_sub / n_wp)
                ),
            )

            reps_sub = xtal_sub_opt.get_tabular_representations(
                N_wp=n_wp,
                N_max=n_max_sub,
                discrete=discrete,
                discrete_cell=discrete_cell,
                N_grids=discrete_resolution,
            ) or []

            if include_energy and current_energy is not None:
                reps_sub = [
                    np.append(rep, current_energy)
                    for rep in reps_sub
                ]

            xtal_reps.extend(reps_sub)

    return xtal_reps[:max_per_structure]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate tabular crystal representations from a PyXtal database."
        )
    )

    parser.add_argument(
        "--database",
        default="data/source/sp2_sacada.db",
        help="Input PyXtal database.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag used to name the output CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/train",
        help="Output directory for the training CSV.",
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=500,
        help="Maximum atoms per unit cell.",
    )
    parser.add_argument(
        "--min_spg",
        type=int,
        default=0,
        help="Minimum allowed space-group number.",
    )
    parser.add_argument(
        "--max_dof",
        type=int,
        default=24,
        help="Maximum allowed structural degrees of freedom.",
    )
    parser.add_argument(
        "--max_wp",
        type=int,
        default=8,
        help="Maximum number of Wyckoff positions.",
    )
    parser.add_argument(
        "--max_energy",
        type=float,
        default=0.0,
        help=(
            "Maximum ff_energy for source structures. "
            "Use 'inf' to disable energy filtering."
        ),
    )
    parser.add_argument(
        "--max_per_struc",
        type=int,
        default=500,
        help="Maximum representations generated per source structure.",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Add a source-structure label column.",
    )
    parser.add_argument(
        "--energy",
        action="store_true",
        help="Include ff_energy in the output CSV.",
    )
    parser.add_argument(
        "--discrete",
        type=int,
        metavar="N_GRIDS",
        help="Discretize Wyckoff coordinates using N_GRIDS bins.",
    )
    parser.add_argument(
        "--discrete_cell",
        action="store_true",
        help="Discretize cell parameters; requires --discrete.",
    )
    parser.add_argument(
        "--prototype",
        default="graphite",
        help="Reference prototype for the local environment.",
    )
    parser.add_argument(
        "--CN",
        dest="cn",
        type=int,
        default=3,
        help="Target carbon coordination number.",
    )
    parser.add_argument(
        "--rcut",
        type=float,
        default=2.0,
        help="SO3 descriptor cutoff radius.",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Source structures assigned to a worker per task batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.database):
        raise FileNotFoundError(f"Database not found: {args.database}")

    if args.max_atoms < 1:
        raise ValueError("--max_atoms must be at least 1.")
    if args.max_wp < 1:
        raise ValueError("--max_wp must be at least 1.")
    if args.max_per_struc < 1:
        raise ValueError("--max_per_struc must be at least 1.")
    if args.ncpu < 1:
        raise ValueError("--ncpu must be at least 1.")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1.")
    if args.discrete is not None and args.discrete < 2:
        raise ValueError("--discrete must be at least 2.")

    if args.discrete_cell and args.discrete is None:
        print(
            "Warning: --discrete_cell requires --discrete; "
            "cell parameters will remain continuous."
        )
        args.discrete_cell = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    use_discrete = args.discrete is not None
    discrete_resolution = args.discrete if use_discrete else None

    output_file = build_output_filename(
        args.output_dir,
        args.tag,
        use_discrete,
        args.discrete_cell,
    )

    # Validate/create the destination before the expensive processing begins.
    os.makedirs(args.output_dir, exist_ok=True)
    write_test = os.path.join(args.output_dir, ".write_test")
    try:
        with open(write_test, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(write_test)
    except OSError as exc:
        raise RuntimeError(
            f"Output directory is not writable: {args.output_dir}"
        ) from exc

    print("--- Configuration ---")
    print(f"Database: {args.database}")
    print(f"Output CSV: {output_file}")
    print(f"Max atoms: {args.max_atoms}")
    print(f"Minimum SPG: {args.min_spg}")
    print(f"Maximum DoF: {args.max_dof}")
    print(f"Maximum WP: {args.max_wp}")
    print(f"Maximum energy: {args.max_energy}")
    print(f"Maximum representations/source: {args.max_per_struc}")
    print(f"Include energy: {args.energy}")
    print(f"Include label: {args.label}")
    print(
        f"Discrete representation: {use_discrete} "
        f"(resolution: {discrete_resolution})"
    )
    print(f"Discrete cell: {args.discrete_cell}")
    print(f"Reference prototype: {args.prototype}")
    print(f"Target CN: {args.cn}")
    print(f"SO3 cutoff: {args.rcut}")
    print(f"CPU processes: {args.ncpu}")
    print(f"Multiprocessing chunksize: {args.chunksize}")
    print(f"Random seed: {args.seed}")
    print("---------------------")

    filter_by_energy = np.isfinite(args.max_energy)
    load_energy = args.energy or filter_by_energy

    db = database_topology(args.database)
    xtals_all = db.get_all_xtals(include_energy=load_energy)
    print(f"Loaded {len(xtals_all)} structures from {args.database}")

    if filter_by_energy:
        xtals_filtered = [
            xtal
            for xtal in xtals_all
            if (
                getattr(xtal, "ff_energy", None) is not None
                and xtal.ff_energy <= args.max_energy
            )
        ]
        print(
            f"Filtered to {len(xtals_filtered)} structures with "
            f"ff_energy <= {args.max_energy}"
        )
    else:
        xtals_filtered = xtals_all
        print("Energy filtering disabled.")

    params = (
        args.max_dof,
        1,
        args.max_atoms,
        args.max_energy,
        args.min_spg,
        args.max_wp,
        args.max_per_struc,
        args.energy,
        use_discrete,
        args.discrete_cell,
        discrete_resolution,
        5e-4,
        args.prototype,
        args.cn,
        args.rcut,
    )

    indexed_xtals = list(enumerate(xtals_filtered))
    worker = partial(
        process_one_xtal,
        params=params,
        base_seed=args.seed,
    )

    total_reps = []
    usable_sources = 0
    failed_sources = 0

    print(
        f"Processing {len(indexed_xtals)} structures "
        f"with {args.ncpu} process(es)..."
    )

    if args.ncpu == 1:
        results = map(worker, indexed_xtals)
        for source_index, source_reps, error in results:
            if error is not None:
                failed_sources += 1
                print(f"Source {source_index} failed: {error}")
                continue

            if not source_reps:
                continue

            usable_sources += 1

            if args.label:
                label = source_index + 1
                source_reps = [
                    np.append(rep, label)
                    for rep in source_reps
                ]

            total_reps.extend(source_reps)
            print(
                f"Completed source {source_index}: "
                f"{len(source_reps)} representations; "
                f"{usable_sources} usable sources; "
                f"{len(total_reps)} representations total."
            )
    else:
        with Pool(processes=args.ncpu) as pool:
            results = pool.imap_unordered(
                worker,
                indexed_xtals,
                chunksize=args.chunksize,
            )

            for source_index, source_reps, error in results:
                if error is not None:
                    failed_sources += 1
                    print(f"Source {source_index} failed: {error}")
                    continue

                if not source_reps:
                    continue

                usable_sources += 1

                if args.label:
                    label = source_index + 1
                    source_reps = [
                        np.append(rep, label)
                        for rep in source_reps
                    ]

                total_reps.extend(source_reps)
                print(
                    f"Completed source {source_index}: "
                    f"{len(source_reps)} representations; "
                    f"{usable_sources} usable sources; "
                    f"{len(total_reps)} representations total."
                )

    print("\nFinished processing.")
    print(f"Usable source structures: {usable_sources}")
    print(f"Failed source structures: {failed_sources}")
    print(f"Total representations: {len(total_reps)}")

    if not total_reps:
        print("No representations were generated; no CSV was written.")
        return

    make_csv(
        total_reps=total_reps,
        include_energy=args.energy,
        include_label=args.label,
        discrete=use_discrete,
        discrete_cell=args.discrete_cell,
        n_wp=args.max_wp,
        filename=output_file,
    )

    print("Script finished.")


if __name__ == "__main__":
    main()

