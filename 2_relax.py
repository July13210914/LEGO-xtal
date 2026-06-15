#!/usr/bin/env python3
import argparse
import os
import shutil
from functools import partial
from multiprocessing import Pool
from time import time

import pandas as pd
from ase.db import connect
from pyxtal import pyxtal
from pyxtal.db import database_topology
from tqdm import tqdm

from lego.builder import builder


def trim_representation(rep):
    """Remove optional trailing energy/label columns from a tabular row."""
    mod = (len(rep) - 7) % 4
    if mod > 0:
        rep = rep[:-mod]
    return rep


def rep_to_xtal(rep, discrete, discrete_cell, discrete_res):
    """Decode one tabular representation into a valid PyXtal structure."""
    rep = trim_representation(rep)

    xtal = pyxtal()
    xtal.from_tabular_representation(
        rep,
        normalize=False,
        discrete=discrete,
        discrete_cell=discrete_cell,
        N_grids=discrete_res,
    )

    if xtal.valid and len(xtal.atom_sites) > 0:
        return xtal

    return None


def process_rep(rep, discrete, discrete_cell, discrete_res):
    """Validate one tabular representation for multiprocessing."""
    try:
        xtal = rep_to_xtal(
            rep,
            discrete=discrete,
            discrete_cell=discrete_cell,
            discrete_res=discrete_res,
        )

        if xtal is not None:
            clean_rep = trim_representation(rep)
            return clean_rep, sum(xtal.numIons)

    except Exception as exc:
        print(f"Failed to process representation: {type(exc).__name__}: {exc}")

    return None


def save_pre_so3_database(
    reps,
    output_path,
    source_csv,
    discrete,
    discrete_cell,
    discrete_res,
):
    """Save decoded structures before SO3 relaxation to an ASE database."""
    if os.path.exists(output_path):
        os.remove(output_path)

    db = connect(output_path)
    saved = 0

    for row_index, rep in enumerate(
        tqdm(reps, desc="Saving pre-SO3 structures")
    ):
        try:
            xtal = rep_to_xtal(
                rep,
                discrete=discrete,
                discrete_cell=discrete_cell,
                discrete_res=discrete_res,
            )

            if xtal is None:
                continue

            atoms = xtal.to_ase(resort=False)

            db.write(
                atoms,
                stage="pre_so3",
                source_csv=source_csv,
                row_index=row_index,
                num_atoms=len(atoms),
                data={
                    "rep": (
                        rep.tolist()
                        if hasattr(rep, "tolist")
                        else list(rep)
                    )
                },
            )
            saved += 1

        except Exception as exc:
            print(
                f"Failed to save pre-SO3 structure {row_index}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(f"Saved {saved} pre-SO3 structures to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode and SO3-relax sampled LEGO-Xtal structures."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Input sampled CSV file.",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of CPU worker processes.",
    )
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
        help="First CSV row to process.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Exclusive final CSV row; -1 means process to the end.",
    )
    parser.add_argument(
        "--source",
        default="data/source/sp2_sacada.db",
        help="Reference source database used for overlap checking.",
    )
    parser.add_argument(
        "--prototype",
        default="graphite",
        help="PyXtal prototype used as the SO3 reference environment.",
    )
    parser.add_argument(
        "--CN",
        dest="cn",
        type=int,
        default=3,
        help="Required carbon coordination number.",
    )
    parser.add_argument(
        "--rcut",
        type=float,
        default=2.1,
        help="SO3 descriptor cutoff radius.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: basename of the input CSV "
            "without extension."
        ),
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip topology calculation and cleaning.",
    )
    parser.add_argument(
        "--skip-energy",
        action="store_true",
        help="Skip GULP/ReaxFF energy calculation.",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip overlap checking against the source database.",
    )
    parser.add_argument(
        "--save-pre-so3-db",
        action="store_true",
        help="Save decoded structures before SO3 relaxation.",
    )
    parser.add_argument(
        "--topology-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each topology calculation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    if args.ncpu < 1:
        raise ValueError("--ncpu must be at least 1.")
    if args.begin < 0:
        raise ValueError("--begin must be non-negative.")
    if args.end != -1 and args.end <= args.begin:
        raise ValueError("--end must be greater than --begin, or -1.")
    if args.cn < 1:
        raise ValueError("--CN must be at least 1.")
    if args.rcut <= 0:
        raise ValueError("--rcut must be positive.")

    rank = 0

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.splitext(os.path.basename(args.csv))[0]

    os.makedirs(output_dir, exist_ok=True)

    write_test = os.path.join(output_dir, ".write_test")
    try:
        with open(write_test, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(write_test)
    except OSError as exc:
        raise RuntimeError(
            f"Output directory is not writable: {output_dir}"
        ) from exc

    print("--- Configuration ---")
    print(f"Input CSV: {args.csv}")
    print(f"Output directory: {output_dir}")
    print(f"Rows: {args.begin} to {args.end}")
    print(f"CPU processes: {args.ncpu}")
    print(
        "SLURM_CPUS_PER_TASK: "
        f"{os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}"
    )
    print(f"Reference prototype: {args.prototype}")
    print(f"Target CN: {args.cn}")
    print(f"SO3 cutoff: {args.rcut}")
    print(f"Skip topology: {args.skip_topology}")
    print(f"Skip energy: {args.skip_energy}")
    print(f"Save pre-SO3 DB: {args.save_pre_so3_db}")
    print("---------------------")

    reference = pyxtal()
    reference.from_prototype(args.prototype)
    reference_structure = reference.to_pymatgen()

    df = pd.read_csv(args.csv)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    required_columns = {"spg", "a", "b", "c", "x0"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    coordinate_max = df["x0"].max()
    if coordinate_max < 5 + 1e-3:
        discrete = False
        discrete_res = None
    elif coordinate_max < 50 + 1e-3:
        discrete = True
        discrete_res = 50
    else:
        discrete = True
        discrete_res = 100

    first_a = df["a"].iloc[0]
    first_c = df["c"].iloc[0]
    discrete_cell = (
        abs(first_a - round(first_a)) < 1e-6
        and abs(first_c - round(first_c)) < 1e-6
    )

    full_data = df.to_numpy()
    if args.end == -1:
        data = full_data[args.begin:]
    else:
        data = full_data[args.begin:args.end]

    n_total = len(data)

    print(f"Total CSV rows: {len(df)}")
    print(f"Selected rows: {n_total}")
    print(
        f"Representation mode: discrete={discrete}, "
        f"resolution={discrete_res}, "
        f"discrete_cell={discrete_cell}"
    )

    worker = partial(
        process_rep,
        discrete=discrete,
        discrete_cell=discrete_cell,
        discrete_res=discrete_res,
    )

    valid_rows = []

    if args.ncpu == 1:
        iterator = map(worker, data)
        for result in tqdm(
            iterator,
            total=n_total,
            desc="Decoding sampled rows",
        ):
            if result is not None:
                valid_rows.append(result)
    else:
        with Pool(processes=args.ncpu) as pool:
            iterator = pool.imap(worker, data, chunksize=1)
            for result in tqdm(
                iterator,
                total=n_total,
                desc="Decoding sampled rows",
            ):
                if result is not None:
                    valid_rows.append(result)

    valid_rows.sort(key=lambda item: item[1])
    reps = [item[0] for item in valid_rows]
    n_decoded = len(reps)

    print(f"Successfully decoded structures: {n_decoded}/{n_total}")

    if not reps:
        raise RuntimeError("No sampled rows could be decoded by PyXtal.")

    if args.save_pre_so3_db:
        save_pre_so3_database(
            reps=reps,
            output_path=os.path.join(output_dir, "pre_so3.db"),
            source_csv=args.csv,
            discrete=discrete,
            discrete_cell=discrete_cell,
            discrete_res=discrete_res,
        )

    bu = builder(
        ["C"],
        [1],
        rank=rank,
        prefix=os.path.join(output_dir, "mof"),
    )
    bu.set_descriptor_calculator(
        mykwargs={"rcut": args.rcut}
    )
    bu.set_reference_enviroments(reference_structure)
    bu.set_criteria(CN={"C": [args.cn]})

    print(
        f"Optimizing {n_decoded} structures with "
        f"{args.ncpu} CPU process(es)."
    )

    optimization_start = time()
    optimized_xtals = bu.optimize_reps(
        reps,
        ncpu=args.ncpu,
        minimizers=[
            ("Nelder-Mead", 100),
            ("L-BFGS-B", 400),
            ("L-BFGS-B", 200),
        ],
        N_grids=discrete_res,
    )
    optimization_time = time() - optimization_start

    n_optimized = len(optimized_xtals)
    print(
        f"Valid optimized structures: "
        f"{n_optimized}/{n_decoded}"
    )
    print(
        f"Optimization time: "
        f"{optimization_time / 60:.2f} minutes"
    )

    topology_time = 0.0
    energy_time = 0.0
    n_unique = n_optimized

    if not args.skip_topology:
        topology_start = time()

        bu.db.update_row_topology(
            overwrite=False,
            prefix=os.path.join(output_dir, "mof-0"),
            timeout=args.topology_timeout,
        )
        bu.db.clean_structures_spg_topology(dim=3)

        topology_time = time() - topology_start
        print(
            f"Topology processing time: "
            f"{topology_time / 60:.2f} minutes"
        )
    else:
        print("Skipping topology processing.")

    if not args.skip_energy:
        energy_start = time()

        bu.db.update_row_energy(
            "GULP",
            ncpu=args.ncpu,
            calc_folder=os.path.join(output_dir, f"gulp_{rank}"),
        )

        energy_time = time() - energy_start
        print(
            f"Energy calculation time: "
            f"{energy_time / 60:.2f} minutes"
        )
    else:
        print("Skipping energy calculation.")

    final_db = os.path.join(output_dir, "final.db")

    if os.path.exists(final_db):
        os.remove(final_db)

    if not args.skip_topology:
        unique_db = os.path.join(output_dir, f"unique_{rank}.db")

        if args.skip_energy:
            n_unique = bu.db.get_db_unique_topology(
                unique_db,
                update_topology=False,
            )
        else:
            n_unique = bu.db.get_db_unique_topology(
                unique_db,
                update_topology=False,
                key="ff_energy",
            )

        if not os.path.exists(unique_db):
            raise RuntimeError(
                f"Unique topology database was not created: {unique_db}"
            )

        shutil.move(unique_db, final_db)

    else:
        candidate_raw_dbs = [
            os.path.join(output_dir, f"mof-{rank}.db"),
            os.path.join(output_dir, "mof.db"),
        ]
        
        raw_db = next(
            (path for path in candidate_raw_dbs if os.path.exists(path)),
            None,
        )
        
        if raw_db is None:
            existing_dbs = [
                filename
                for filename in os.listdir(output_dir)
                if filename.endswith(".db")
            ]
            raise RuntimeError(
                "Could not locate the raw optimized database. "
                f"Checked: {candidate_raw_dbs}. "
                f"Existing databases: {existing_dbs}"
            )
        
        shutil.copy2(raw_db, final_db)
        print(f"Copied {raw_db} to {final_db}")

    n_overlap = -1

    if args.skip_overlap:
        print("Skipping source-overlap check.")
    elif args.skip_topology:
        print(
            "Skipping source-overlap check because topology "
            "processing was disabled."
        )
    elif not os.path.isfile(args.source):
        print(
            f"Skipping source-overlap check: "
            f"source database not found: {args.source}"
        )
    else:
        try:
            overlap_log = os.path.join(output_dir, "overlap.log")
            source_db = database_topology(
                args.source,
                log_file=overlap_log,
            )
            overlaps = source_db.check_overlap(final_db)
            n_overlap = len(overlaps)
        except Exception as exc:
            print(
                f"Overlap check failed: "
                f"{type(exc).__name__}: {exc}"
            )

    total_time = time() - start_time

    metric_path = os.path.join(output_dir, "metric.txt")
    with open(metric_path, "w", encoding="utf-8") as handle:
        handle.write(f"Source data: {args.csv}\n")
        handle.write(f"Reference prototype: {args.prototype}\n")
        handle.write(f"Target CN: {args.cn}\n")
        handle.write(f"SO3 cutoff: {args.rcut}\n")
        handle.write(f"Skip topology: {args.skip_topology}\n")
        handle.write(f"Skip energy: {args.skip_energy}\n")
        handle.write(
            f"Optimization time minutes: "
            f"{optimization_time / 60:12.2f}\n"
        )
        handle.write(
            f"Topology time minutes: "
            f"{topology_time / 60:12.2f}\n"
        )
        handle.write(
            f"Energy calculation time minutes: "
            f"{energy_time / 60:12.2f}\n"
        )
        handle.write(
            f"Total time minutes: "
            f"{total_time / 60:12.2f}\n"
        )
        handle.write(f"N_parallel_cpus: {args.ncpu:12d}\n")
        handle.write(f"N_total_count: {n_total:12d}\n")
        handle.write(f"N_valid_xtal: {n_decoded:12d}\n")
        handle.write(f"N_valid_env: {n_optimized:12d}\n")
        handle.write(f"N_unique_xtal: {n_unique:12d}\n")
        handle.write(f"N_train_overlap: {n_overlap:12d}\n")

    print(
        f"N0/N1/N2/N3: "
        f"{n_total}/{n_decoded}/{n_optimized}/{n_unique}"
    )
    print(f"Final database: {final_db}")
    print(f"Metrics: {metric_path}")
    print(
        f"Total wall time: "
        f"{total_time / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()

