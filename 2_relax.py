#!/usr/bin/env python3
"""Decode, mixed-CN SO3-relax, and optionally post-process LEGO samples.

This version supports unconditional joint samples containing
``target_coord0 ... target_coordN``.  Each occupied independent site is
assigned a target coordination of 3 or 4.  The LEGO builder then compares
CN3 sites with a graphite SO3 reference and CN4 sites with a diamond SO3
reference, and performs strict post-optimization coordination validation.
"""

import argparse
import os
import shutil
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
from ase.db import connect
from pyxtal import pyxtal
from pyxtal.db import database_topology
from tqdm import tqdm

from lego.builder import builder, set_target_coordination


BASE_COLUMNS = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]


def discover_site_columns(df):
    """Return ordered representation and target columns for contiguous slots."""
    site_indices = []
    index = 0
    while f"wp{index}" in df.columns:
        site_indices.append(index)
        index += 1

    if not site_indices:
        raise ValueError("Input CSV contains no Wyckoff columns (wp0, wp1, ...).")

    representation_columns = list(BASE_COLUMNS)
    target_columns = []

    for index in site_indices:
        required = [
            f"wp{index}",
            f"x{index}",
            f"y{index}",
            f"z{index}",
            f"target_coord{index}",
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns for site slot {index}: {missing}"
            )

        representation_columns.extend(
            [f"wp{index}", f"x{index}", f"y{index}", f"z{index}"]
        )
        target_columns.append(f"target_coord{index}")

    # Detect gaps such as wp0, wp1, wp3.
    unexpected_wp = sorted(
        column for column in df.columns
        if column.startswith("wp")
        and column[2:].isdigit()
        and int(column[2:]) not in site_indices
    )
    if unexpected_wp:
        raise ValueError(
            "Wyckoff slot columns must be contiguous from wp0. Found: "
            f"{unexpected_wp}"
        )

    return site_indices, representation_columns, target_columns


def validate_and_extract_targets(row, site_indices):
    """Validate padding/contiguity and return targets for occupied slots."""
    targets = []
    seen_empty = False

    for index in site_indices:
        wp = int(round(float(row[f"wp{index}"])))
        target = int(round(float(row[f"target_coord{index}"])))

        if wp == -1:
            seen_empty = True
            if target != 0:
                raise ValueError(
                    f"slot {index}: empty wp=-1 requires target 0, got {target}"
                )
            continue

        if seen_empty:
            raise ValueError(
                f"slot {index}: occupied site occurs after an empty slot"
            )

        if target not in (3, 4):
            raise ValueError(
                f"slot {index}: occupied site requires target 3 or 4, "
                f"got {target}"
            )

        targets.append(target)

    if not targets:
        raise ValueError("row contains no occupied sites")

    return targets



def get_site_wp_index(site):
    """Best-effort extraction of the decoded PyXtal Wyckoff index."""
    wp = getattr(site, "wp", None)
    for obj in (wp, site):
        if obj is None:
            continue
        for name in ("index", "wp_index"):
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def get_site_position(site):
    """Return the decoded generating fractional coordinate, when available."""
    for name in ("position", "pos"):
        value = getattr(site, name, None)
        if value is not None:
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size >= 3:
                return np.mod(array[:3], 1.0)
    return None


def periodic_distance(a, b):
    """Fractional-coordinate distance under minimum-image wrapping."""
    delta = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    delta = np.minimum(delta, 1.0 - delta)
    return float(np.linalg.norm(delta))


def requested_site_records(row, site_indices):
    """Collect occupied requested sites with WP, target, and raw position."""
    records = []
    for index in site_indices:
        wp = int(round(float(row[f"wp{index}"])))
        if wp == -1:
            break
        target = int(round(float(row[f"target_coord{index}"])))
        position = np.mod(
            np.asarray(
                [row[f"x{index}"], row[f"y{index}"], row[f"z{index}"]],
                dtype=float,
            ),
            1.0,
        )
        records.append(
            {
                "slot": index,
                "wp": wp,
                "target": target,
                "position": position,
            }
        )
    return records


def map_targets_to_decoded_sites(xtal, requested, policy):
    """Map requested targets onto PyXtal-repaired independent sites.

    ``strict`` requires the repaired structure to retain every requested site.
    ``map-survivors`` allows PyXtal to drop/merge sites, but every surviving
    decoded site must map uniquely to a requested site with the same WP index.
    """
    decoded = []
    for site_index, site in enumerate(xtal.atom_sites):
        decoded.append(
            {
                "site_index": site_index,
                "site": site,
                "wp": get_site_wp_index(site),
                "position": get_site_position(site),
            }
        )

    requested_wps = [item["wp"] for item in requested]
    decoded_wps = [item["wp"] for item in decoded]

    if policy == "strict" and len(decoded) != len(requested):
        raise ValueError(
            "decoded independent-site count does not match target count: "
            f"{len(decoded)} versus {len(requested)}; "
            f"requested_wps={requested_wps}; decoded_wps={decoded_wps}"
        )

    by_wp = defaultdict(list)
    for record in requested:
        by_wp[record["wp"]].append(record)

    used_slots = set()
    assignments = []
    for item in decoded:
        if item["wp"] is None:
            raise ValueError(
                "could not determine decoded Wyckoff index for target mapping"
            )

        candidates = [
            record for record in by_wp.get(item["wp"], [])
            if record["slot"] not in used_slots
        ]
        if not candidates:
            raise ValueError(
                "no unused requested site matches decoded Wyckoff index "
                f"{item['wp']}; requested_wps={requested_wps}; "
                f"decoded_wps={decoded_wps}"
            )

        if len(candidates) == 1 or item["position"] is None:
            chosen = candidates[0]
        else:
            chosen = min(
                candidates,
                key=lambda record: periodic_distance(
                    record["position"], item["position"]
                ),
            )

        used_slots.add(chosen["slot"])
        assignments.append((item["site"], chosen))

    if policy == "strict" and len(used_slots) != len(requested):
        raise ValueError(
            "strict target mapping left requested sites unmatched: "
            f"matched={len(used_slots)} requested={len(requested)}"
        )

    dropped = [
        record["slot"] for record in requested
        if record["slot"] not in used_slots
    ]
    for site, record in assignments:
        set_target_coordination(site, record["target"])

    mapped_targets = [record["target"] for _site, record in assignments]
    return mapped_targets, dropped, requested_wps, decoded_wps


def decode_row(
    item,
    representation_columns,
    site_indices,
    discrete,
    discrete_cell,
    discrete_res,
    site_repair_policy,
):
    """Decode one DataFrame row and attach target coordination metadata."""
    row_index, row_dict = item

    try:
        validate_and_extract_targets(row_dict, site_indices)
        requested = requested_site_records(row_dict, site_indices)
        rep = np.asarray(
            [row_dict[column] for column in representation_columns],
            dtype=float,
        )

        xtal = pyxtal()
        xtal.from_tabular_representation(
            rep,
            normalize=False,
            discrete=discrete,
            discrete_cell=discrete_cell,
            N_grids=discrete_res,
        )

        if not xtal.valid or len(xtal.atom_sites) == 0:
            return row_index, None, None, None, (
                "PyXtal returned an invalid or empty structure"
            ), None

        targets, dropped_slots, requested_wps, decoded_wps = (
            map_targets_to_decoded_sites(
                xtal,
                requested,
                policy=site_repair_policy,
            )
        )

        repair_info = {
            "policy": site_repair_policy,
            "requested_site_count": len(requested),
            "decoded_site_count": len(xtal.atom_sites),
            "dropped_slots": dropped_slots,
            "requested_wps": requested_wps,
            "decoded_wps": decoded_wps,
        }

        return (
            row_index, xtal, rep, targets, sum(xtal.numIons), repair_info
        )

    except Exception as exc:
        return row_index, None, None, None, (
            f"{type(exc).__name__}: {exc}"
        ), None


def save_pre_so3_database(decoded, output_path, source_csv):
    """Save decoded structures and target vectors before SO3 relaxation."""
    if os.path.exists(output_path):
        os.remove(output_path)

    db = connect(output_path)
    saved = 0

    for row_index, xtal, rep, targets, _num_atoms, repair_info in tqdm(
        decoded,
        desc="Saving pre-SO3 structures",
    ):
        atoms = xtal.to_ase(resort=False)
        db.write(
            atoms,
            stage="pre_so3",
            source_csv=source_csv,
            row_index=int(row_index),
            num_atoms=len(atoms),
            data={
                "rep": rep.tolist(),
                "target_coordination": [int(value) for value in targets],
                "site_repair": repair_info,
            },
        )
        saved += 1

    print(f"Saved {saved} pre-SO3 structures to {output_path}")


def write_prototype_cif(name, output_path):
    """Write a PyXtal built-in prototype to CIF for the builder reference bank."""
    reference = pyxtal()
    reference.from_prototype(name)
    reference.to_pymatgen().to(filename=output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decode and site-specifically SO3-relax mixed CN3/CN4 "
            "LEGO-Xtal samples."
        )
    )

    parser.add_argument("--csv", default=None, help="Input sampled CSV file.")
    parser.add_argument(
        "--ncpu", type=int, default=1, help="Number of CPU worker processes."
    )
    parser.add_argument(
        "--begin", type=int, default=0, help="First CSV row to process."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Exclusive final CSV row; -1 means process to the end.",
    )
    parser.add_argument(
        "--source",
        default="./mixed_34_sacada.db",
        help="Reference source database used for overlap checking.",
    )
    parser.add_argument(
        "--prototype-cn3",
        default="graphite",
        help="PyXtal prototype used for target CN3 sites.",
    )
    parser.add_argument(
        "--prototype-cn4",
        default="diamond",
        help="PyXtal prototype used for target CN4 sites.",
    )
    parser.add_argument(
        "--rcut",
        type=float,
        default=2.4,
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
        "--site-repair-policy",
        choices=["strict", "map-survivors"],
        default="strict",
        help=(
            "How to handle PyXtal site merging/removal. 'strict' requires "
            "one-to-one preservation; 'map-survivors' transfers targets to "
            "uniquely matched surviving sites and records dropped slots."
        ),
    )
    parser.add_argument(
        "--resume-db",
        default=None,
        help=(
            "Resume topology/energy/finalization from an existing optimized "
            "ASE database (for example output_dir/mof-0.db). When supplied, "
            "CSV decoding and SO3 optimization are skipped."
        ),
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

    if args.resume_db is None:
        if args.csv is None:
            raise ValueError("--csv is required unless --resume-db is supplied.")
        if not os.path.isfile(args.csv):
            raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    if args.resume_db is not None and not os.path.isfile(args.resume_db):
        raise FileNotFoundError(f"Resume database not found: {args.resume_db}")
    if args.ncpu < 1:
        raise ValueError("--ncpu must be at least 1.")
    if args.begin < 0:
        raise ValueError("--begin must be non-negative.")
    if args.end != -1 and args.end <= args.begin:
        raise ValueError("--end must be greater than --begin, or -1.")
    if args.rcut <= 0:
        raise ValueError("--rcut must be positive.")

    rank = 0
    output_dir = args.output_dir
    if output_dir is None:
        if args.resume_db is not None:
            output_dir = os.path.dirname(os.path.abspath(args.resume_db)) or "."
        else:
            output_dir = os.path.splitext(os.path.basename(args.csv))[0]
    os.makedirs(output_dir, exist_ok=True)

    print("--- Configuration ---")
    print(f"Input CSV: {args.csv}")
    print(f"Resume database: {args.resume_db}")
    print(f"Output directory: {output_dir}")
    print(f"Rows: {args.begin} to {args.end}")
    print(f"CPU processes: {args.ncpu}")
    print(f"CN3 reference prototype: {args.prototype_cn3}")
    print(f"CN4 reference prototype: {args.prototype_cn4}")
    print(f"SO3 cutoff: {args.rcut}")
    print(f"Site repair policy: {args.site_repair_policy}")
    print(f"Skip topology: {args.skip_topology}")
    print(f"Skip energy: {args.skip_energy}")
    print("---------------------")

    if args.resume_db is not None:
        print("Resume mode enabled: skipping CSV decoding and SO3 optimization.")
        work_db = database_topology(
            args.resume_db,
            log_file=os.path.join(output_dir, "resume_postprocess.log"),
        )
        n_optimized = work_db.db.count()
        n_total = n_optimized
        n_decoded = n_optimized
        n_repaired = 0
        n_dropped_sites = 0
        optimization_time = 0.0
        print(f"Loaded optimized structures: {n_optimized}")
    else:
        df = pd.read_csv(args.csv)
        if df.empty:
            raise ValueError("Input CSV is empty.")

        required_base = set(BASE_COLUMNS + ["x0"])
        missing_base = required_base - set(df.columns)
        if missing_base:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing_base)}"
            )

        site_indices, representation_columns, target_columns = discover_site_columns(df)
        print(f"Detected Wyckoff slots: {len(site_indices)}")
        print(f"Target columns: {target_columns}")

        coordinate_max = pd.to_numeric(df["x0"], errors="coerce").max()
        if coordinate_max < 5 + 1e-3:
            discrete = False
            discrete_res = None
        elif coordinate_max < 50 + 1e-3:
            discrete = True
            discrete_res = 50
        else:
            discrete = True
            discrete_res = 100

        first_a = float(df["a"].iloc[0])
        first_c = float(df["c"].iloc[0])
        discrete_cell = (
            abs(first_a - round(first_a)) < 1e-6
            and abs(first_c - round(first_c)) < 1e-6
        )

        stop = None if args.end == -1 else args.end
        selected = df.iloc[args.begin:stop].copy()
        n_total = len(selected)
        if n_total == 0:
            raise ValueError("Selected CSV row range is empty.")

        print(f"Total CSV rows: {len(df)}")
        print(f"Selected rows: {n_total}")
        print(
            f"Representation mode: discrete={discrete}, "
            f"resolution={discrete_res}, discrete_cell={discrete_cell}"
        )

        items = [
            (int(index), row.to_dict())
            for index, row in selected.iterrows()
        ]
        worker = partial(
            decode_row,
            representation_columns=representation_columns,
            site_indices=site_indices,
            discrete=discrete,
            discrete_cell=discrete_cell,
            discrete_res=discrete_res,
            site_repair_policy=args.site_repair_policy,
        )

        decoded = []
        failures = []

        if args.ncpu == 1:
            iterator = map(worker, items)
            iterator = tqdm(iterator, total=n_total, desc="Decoding sampled rows")
            for result in iterator:
                if result is None:
                    failures.append((-1, "decode worker returned None"))
                    continue
                if result[1] is None:
                    failures.append((result[0], result[4]))
                else:
                    decoded.append(result)
        else:
            with Pool(processes=args.ncpu) as pool:
                iterator = pool.imap(worker, items, chunksize=1)
                for result in tqdm(
                    iterator,
                    total=n_total,
                    desc="Decoding sampled rows",
                ):
                    if result is None:
                        failures.append((-1, "decode worker returned None"))
                        continue
                    if result[1] is None:
                        failures.append((result[0], result[4]))
                    else:
                        decoded.append(result)

        decoded.sort(key=lambda item: (item[4], item[0]))
        n_decoded = len(decoded)
        print(f"Successfully decoded structures: {n_decoded}/{n_total}")
        n_repaired = sum(
            1 for item in decoded if item[5]["dropped_slots"]
        )
        n_dropped_sites = sum(
            len(item[5]["dropped_slots"]) for item in decoded
        )
        print(
            "Decoded with dropped/merged requested sites: "
            f"{n_repaired}/{n_decoded}; dropped requested slots: {n_dropped_sites}"
        )

        if failures:
            failure_log = os.path.join(output_dir, "decode_failures.log")
            with open(failure_log, "w", encoding="utf-8") as handle:
                for row_index, message in failures:
                    handle.write(f"row {row_index}: {message}\n")
            print(f"Decode failures: {len(failures)}; details: {failure_log}")

        if not decoded:
            raise RuntimeError("No sampled rows could be decoded by PyXtal.")

        if args.save_pre_so3_db:
            save_pre_so3_database(
                decoded,
                os.path.join(output_dir, "pre_so3.db"),
                args.csv,
            )

        xtals = [item[1] for item in decoded]

        bu = builder(
            ["C"],
            [1],
            rank=rank,
            prefix=os.path.join(output_dir, "mof"),
        )
        bu.set_descriptor_calculator(mykwargs={"rcut": args.rcut})

        reference_dir = os.path.join(output_dir, "references")
        os.makedirs(reference_dir, exist_ok=True)
        cn3_cif = write_prototype_cif(
            args.prototype_cn3,
            os.path.join(reference_dir, "cn3_reference.cif"),
        )
        cn4_cif = write_prototype_cif(
            args.prototype_cn4,
            os.path.join(reference_dir, "cn4_reference.cif"),
        )
        bu.set_target_coordination_references({3: cn3_cif, 4: cn4_cif})

        print(
            f"Optimizing {n_decoded} structures with site-specific CN3/CN4 "
            f"references using {args.ncpu} CPU process(es)."
        )

        optimization_start = time()
        optimized_xtals = bu.optimize_xtals(
            xtals,
            ncpu=args.ncpu,
            minimizers=[
                ("Nelder-Mead", 100),
                ("L-BFGS-B", 400),
                ("L-BFGS-B", 200),
            ],
        )
        optimization_time = time() - optimization_start

        n_optimized = len(optimized_xtals)
        print(f"Valid optimized structures: {n_optimized}/{n_decoded}")
        print(f"Optimization time: {optimization_time / 60:.2f} minutes")
        work_db = bu.db

    topology_time = 0.0
    energy_time = 0.0
    n_unique = n_optimized

    if n_optimized == 0:
        print("No optimized structures; skipping topology processing.")
    elif not args.skip_topology:
        topology_start = time()
        work_db.update_row_topology(
            overwrite=False,
            prefix=os.path.join(output_dir, "mof-0"),
            timeout=args.topology_timeout,
        )
        work_db.clean_structures_spg_topology(dim=3)
        topology_time = time() - topology_start
        print(f"Topology processing time: {topology_time / 60:.2f} minutes")
    else:
        print("Skipping topology processing.")

    if n_optimized == 0:
        print("No optimized structures; skipping energy calculation.")
    elif not args.skip_energy:
        energy_start = time()
        work_db.update_row_energy(
            "GULP",
            ncpu=args.ncpu,
            calc_folder=os.path.join(output_dir, f"gulp_{rank}"),
        )
        energy_time = time() - energy_start
        print(f"Energy calculation time: {energy_time / 60:.2f} minutes")
    else:
        print("Skipping energy calculation.")

    final_db = os.path.join(output_dir, "final.db")
    if os.path.exists(final_db):
        os.remove(final_db)

    if n_optimized == 0:
        # A zero-yield batch is a valid scientific outcome, not a pipeline error.
        # Create an empty ASE database so downstream bookkeeping has a stable
        # final_db path, then continue to write metrics and exit normally.
        empty_db = connect(final_db)
        empty_db.metadata = {
            "stage": "final",
            "status": "empty",
            "reason": "no structures passed mixed-coordination optimization",
            "source_csv": args.csv,
        }
        n_unique = 0
        print(f"Created empty final database: {final_db}")
    elif not args.skip_topology:
        unique_db = os.path.join(output_dir, f"unique_{rank}.db")
        if args.skip_energy:
            n_unique = work_db.get_db_unique_topology(
                unique_db,
                update_topology=False,
            )
        else:
            n_unique = work_db.get_db_unique_topology(
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
        candidate_raw_dbs = []
        if args.resume_db is not None:
            candidate_raw_dbs.append(args.resume_db)
        candidate_raw_dbs.extend([
            os.path.join(output_dir, f"mof-{rank}.db"),
            os.path.join(output_dir, "mof.db"),
        ])
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
                f"Checked: {candidate_raw_dbs}. Existing: {existing_dbs}"
            )
        shutil.copy2(raw_db, final_db)
        print(f"Copied {raw_db} to {final_db}")

    n_overlap = -1
    if args.skip_overlap:
        print("Skipping source-overlap check.")
    elif args.skip_topology:
        print("Skipping source-overlap check because topology was disabled.")
    elif not os.path.isfile(args.source):
        print(
            "Skipping source-overlap check: source database not found: "
            f"{args.source}"
        )
    else:
        try:
            overlap_log = os.path.join(output_dir, "overlap.log")
            source_db = database_topology(args.source, log_file=overlap_log)
            overlaps = source_db.check_overlap(final_db)
            n_overlap = len(overlaps)
        except Exception as exc:
            print(f"Overlap check failed: {type(exc).__name__}: {exc}")

    total_time = time() - start_time
    metric_path = os.path.join(output_dir, "metric.txt")
    with open(metric_path, "w", encoding="utf-8") as handle:
        handle.write(f"Source data: {args.csv}\n")
        handle.write(f"Resume database: {args.resume_db}\n")
        handle.write(f"CN3 reference prototype: {args.prototype_cn3}\n")
        handle.write(f"CN4 reference prototype: {args.prototype_cn4}\n")
        handle.write(f"SO3 cutoff: {args.rcut}\n")
        handle.write(f"Site repair policy: {args.site_repair_policy}\n")
        handle.write(f"N_repaired_xtal: {n_repaired:12d}\n")
        handle.write(f"N_dropped_requested_sites: {n_dropped_sites:12d}\n")
        handle.write(f"Skip topology: {args.skip_topology}\n")
        handle.write(f"Skip energy: {args.skip_energy}\n")
        handle.write(
            f"Optimization time minutes: {optimization_time / 60:12.2f}\n"
        )
        handle.write(f"Topology time minutes: {topology_time / 60:12.2f}\n")
        handle.write(
            f"Energy calculation time minutes: {energy_time / 60:12.2f}\n"
        )
        handle.write(f"Total time minutes: {total_time / 60:12.2f}\n")
        handle.write(f"N_parallel_cpus: {args.ncpu:12d}\n")
        handle.write(f"N_total_count: {n_total:12d}\n")
        handle.write(f"N_valid_xtal: {n_decoded:12d}\n")
        handle.write(f"N_valid_env: {n_optimized:12d}\n")
        handle.write(f"N_unique_xtal: {n_unique:12d}\n")
        handle.write(f"N_train_overlap: {n_overlap:12d}\n")

    print(f"N0/N1/N2/N3: {n_total}/{n_decoded}/{n_optimized}/{n_unique}")
    print(f"Final database: {final_db}")
    print(f"Metrics: {metric_path}")
    print(f"Total wall time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()

