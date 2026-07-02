#!/usr/bin/env python3
"""Finalize factorized TiO2 samples with direct SO3 relaxation and GULP ranking.

Pipeline
--------
CSV -> strict PyXtal decoding -> cheap exact-representation deduplication
    -> direct element-specific SO3 optimization -> SO3-energy selection
    -> symmetry-constrained GULP relaxation -> periodic TiO6/OTi3 integrity analysis
    -> parallel StructureMatcher deduplication -> integrity-first candidate ranking.

The legacy target_coord columns are used only to identify Ti and O slots in the
current CSV representation. They are not used as coordination constraints,
coordination filters, or SO3-reference selectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from time import time

# Multiprocessing provides the parallelism. Prevent every worker from also
# spawning a full BLAS/OpenMP thread team. Users may override these explicitly.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from ase.db import connect
from pyxtal import pyxtal
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal.db import database_topology
from pyxtal.lattice import Lattice
from pyxtal.symmetry import Group
from tqdm import tqdm

from lego.builder import builder


BASE_COLUMNS = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
SPECIES_FROM_LEGACY_LABEL = {6: "Ti", 3: "O"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-SO3 relaxation and GULP ranking for factorized TiO2 samples."
    )
    parser.add_argument("--csv", required=True, help="Factorized sampled CSV.")
    parser.add_argument("--reference-tio2", required=True, help="Reference TiO2 CIF containing both Ti and O environments, e.g. rutile.cif.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--begin", type=int, default=0,
        help="Start index in the post-deduplication, pre-SO3-valid unique-structure list. Default: 0.",
    )
    parser.add_argument(
        "--end", type=int, default=-1,
        help="Stop index in the post-deduplication, pre-SO3-valid unique-structure list. Default: all.",
    )
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--rcut", type=float, default=3.0)
    parser.add_argument(
        "--discrete-coordinates", action="store_true",
        help="Interpret Wyckoff coordinates as discrete grid indices. Default: continuous coordinates.",
    )
    parser.add_argument(
        "--discrete-cell", action="store_true",
        help="Interpret cell parameters as discrete grid values. Default: continuous cell parameters.",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Grid resolution required by --discrete-coordinates or --discrete-cell.",
    )
    parser.add_argument(
        "--min-ti-ti", type=float, default=1.5,
        help="Catastrophic-overlap preflight floor for Ti-Ti distances in angstrom. Default: 1.5.",
    )
    parser.add_argument(
        "--min-ti-o", type=float, default=1.2,
        help="Catastrophic-overlap preflight floor for Ti-O distances in angstrom. Default: 1.2.",
    )
    parser.add_argument(
        "--min-o-o", type=float, default=1.0,
        help="Catastrophic-overlap preflight floor for O-O distances in angstrom. Default: 1.0.",
    )
    parser.add_argument(
        "--tio-coordination-cutoff", type=float, default=2.6,
        help="Ti-O cutoff used only for diagnostic coordination statistics. Default: 2.6 angstrom.",
    )
    parser.add_argument(
        "--max-so3-energy",
        type=float,
        default=float("inf"),
        help=(
            "Optional maximum final SO3 objective passed to GULP. "
            "The default is 'inf', so every successfully SO3-relaxed structure "
            "is passed to GULP."
        ),
    )
    parser.add_argument(
        "--so3-stop-energy",
        type=float,
        default=100.0,
        help=(
            "Stop the remaining SO3 minimizer stages once this objective is "
            "reached. This accelerates relaxation but does not filter the "
            "GULP handoff. Default: 100."
        ),
    )
    parser.add_argument(
        "--max-initial-so3-energy",
        type=float,
        default=float("inf"),
        help=(
            "Skip expensive SO3 optimization when the initial per-atom SO3 "
            "objective exceeds this value. Use 'inf' to disable. Default: inf."
        ),
    )
    parser.add_argument(
        "--so3-fill-batch-size",
        type=int,
        default=200,
        help=(
            "Number of pre-SO3 candidates optimized per refill cycle while "
            "building the requested post-SO3 unique pool. Default: 200."
        ),
    )
    parser.add_argument(
        "--nm-steps",
        type=int,
        default=50,
        help="Maximum Nelder-Mead iterations in the fast SO3 schedule.",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=150,
        help="Maximum L-BFGS-B iterations after Nelder-Mead.",
    )
    parser.add_argument(
        "--ff-lib", required=True,
        help="Confirmed Ti-O GULP force-field library name/path.",
    )
    parser.add_argument("--skip-gulp", action="store_true")
    parser.add_argument(
        "--dedup-decimals",
        type=int,
        default=8,
        help="Decimal precision for cheap pre-SO3 representation deduplication.",
    )
    parser.add_argument(
        "--post-so3-match-ltol", type=float, default=0.10,
        help="Post-SO3 absolute-scale lattice tolerance. Default: 0.10.",
    )
    parser.add_argument(
        "--post-so3-match-stol", type=float, default=0.15,
        help="Post-SO3 absolute-scale site tolerance. Default: 0.15.",
    )
    parser.add_argument(
        "--post-so3-match-angle-tol", type=float, default=3.0,
        help="Post-SO3 angle tolerance in degrees. Default: 3.",
    )
    parser.add_argument(
        "--match-ltol", type=float, default=0.20,
        help="StructureMatcher fractional lattice tolerance. Default: 0.20.",
    )
    parser.add_argument(
        "--match-stol", type=float, default=0.30,
        help="StructureMatcher normalized site tolerance. Default: 0.30.",
    )
    parser.add_argument(
        "--match-angle-tol", type=float, default=5.0,
        help="StructureMatcher angle tolerance in degrees. Default: 5.",
    )
    parser.add_argument(
        "--match-energy-window", type=float, default=0.05,
        help=(
            "Cheap pair prefilter in eV/atom before direct structure matching. "
            "It never defines a duplicate. Use 'inf' to disable. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--match-volume-tol", type=float, default=0.25,
        help=(
            "Cheap relative volume-per-atom prefilter before StructureMatcher. "
            "It never defines a duplicate. Use 'inf' to disable. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--match-chunksize", type=int, default=8,
        help="Multiprocessing chunksize for pairwise StructureMatcher calls.",
    )
    parser.add_argument(
        "--pre-integrity-neighbor-search-radius",
        type=float,
        default=6.0,
        help="Periodic-neighbor search radius for loose pre-SO3 TiO6/OTi3 screening. Default: 6.0 A.",
    )
    parser.add_argument(
        "--pre-integrity-max-ti-o",
        type=float,
        default=3.0,
        help="Loose maximum sixth Ti-O distance before SO3. Default: 3.0 A.",
    )
    parser.add_argument(
        "--pre-integrity-max-o-ti",
        type=float,
        default=3.0,
        help="Loose maximum third O-Ti distance before SO3. Default: 3.0 A.",
    )
    parser.add_argument(
        "--pre-integrity-max-angle-rms",
        type=float,
        default=35.0,
        help="Loose maximum TiO6 angular RMS before SO3. Default: 35 degrees.",
    )
    parser.add_argument(
        "--pre-integrity-min-ti-pass-fraction",
        type=float,
        default=0.8,
        help="Loose minimum fraction of Ti sites passing before SO3. Default: 0.8.",
    )
    parser.add_argument(
        "--pre-integrity-min-o-pass-fraction",
        type=float,
        default=0.8,
        help="Loose minimum fraction of O sites passing before SO3. Default: 0.8.",
    )
    parser.add_argument(
        "--integrity-neighbor-search-radius",
        type=float,
        default=6.0,
        help="Periodic-neighbor search radius for relaxed TiO6/OTi3 analysis. Default: 6.0 A.",
    )
    parser.add_argument(
        "--integrity-max-ti-o",
        type=float,
        default=2.6,
        help="Maximum allowed sixth Ti-O distance in a passing Ti site. Default: 2.6 A.",
    )
    parser.add_argument(
        "--integrity-max-o-ti",
        type=float,
        default=2.6,
        help="Maximum allowed third O-Ti distance in a passing O site. Default: 2.6 A.",
    )
    parser.add_argument(
        "--integrity-max-angle-rms",
        type=float,
        default=22.0,
        help="Maximum TiO6 O-Ti-O angular RMS from an ideal octahedron. Default: 22 degrees.",
    )
    parser.add_argument(
        "--integrity-min-ti-pass-fraction",
        type=float,
        default=1.0,
        help="Minimum fraction of Ti sites passing TiO6 checks. Default: 1.0.",
    )
    parser.add_argument(
        "--integrity-min-o-pass-fraction",
        type=float,
        default=1.0,
        help="Minimum fraction of O sites passing OTi3 checks. Default: 1.0.",
    )
    parser.add_argument(
        "--training-db",
        default=None,
        help=(
            "Optional ASE database containing the training/reference structures. "
            "Post-GULP unique candidates are compared directly against this database."
        ),
    )
    parser.add_argument(
        "--skip-training-overlap",
        action="store_true",
        help="Skip comparison of final unique candidates against --training-db.",
    )
    return parser.parse_args()


def discover_site_indices(df: pd.DataFrame) -> list[int]:
    indices = []
    i = 0
    while f"wp{i}" in df.columns:
        required = [f"wp{i}", f"x{i}", f"y{i}", f"z{i}", f"target_coord{i}"]
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for slot {i}: {missing}")
        indices.append(i)
        i += 1
    if not indices:
        raise ValueError("No contiguous wp0, wp1, ... site columns were found.")
    extra = [
        name for name in df.columns
        if name.startswith("wp") and name[2:].isdigit() and int(name[2:]) not in indices
    ]
    if extra:
        raise ValueError(f"Non-contiguous Wyckoff slots found: {sorted(extra)}")
    return indices


def infer_representation_mode(df: pd.DataFrame) -> tuple[bool, int | None, bool]:
    coordinate_max = pd.to_numeric(df["x0"], errors="coerce").max()
    if coordinate_max < 5 + 1e-3:
        discrete, resolution = False, None
    elif coordinate_max < 50 + 1e-3:
        discrete, resolution = True, 50
    else:
        discrete, resolution = True, 100

    first_a = float(df["a"].iloc[0])
    first_c = float(df["c"].iloc[0])
    discrete_cell = (
        abs(first_a - round(first_a)) < 1e-6
        and abs(first_c - round(first_c)) < 1e-6
    )
    return discrete, resolution, discrete_cell


def requested_sites(row: dict, indices: list[int]) -> list[dict]:
    records = []
    seen_empty = False
    for slot in indices:
        wp_index = int(round(float(row[f"wp{slot}"])))
        legacy_label = int(round(float(row[f"target_coord{slot}"])))
        if wp_index == -1:
            seen_empty = True
            if legacy_label != 0:
                raise ValueError(f"slot {slot}: empty site requires target_coord=0")
            continue
        if seen_empty:
            raise ValueError(f"slot {slot}: occupied site follows an empty slot")
        if legacy_label not in SPECIES_FROM_LEGACY_LABEL:
            raise ValueError(
                f"slot {slot}: target_coord is used only as a Ti/O label and must be 6 or 3; got {legacy_label}"
            )
        records.append(
            {
                "slot": slot,
                "wp_index": wp_index,
                "species": SPECIES_FROM_LEGACY_LABEL[legacy_label],
                "position": np.mod(
                    np.asarray([row[f"x{slot}"], row[f"y{slot}"], row[f"z{slot}"]], dtype=float),
                    1.0,
                ),
            }
        )
    if not records:
        raise ValueError("row contains no occupied independent sites")
    return records


def periodic_fractional_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.abs(np.asarray(a) - np.asarray(b))
    delta = np.minimum(delta, 1.0 - delta)
    return float(np.linalg.norm(delta))


def build_tio2(
    row: dict,
    indices: list[int],
    discrete: bool,
    resolution: int | None,
    discrete_cell: bool,
) -> tuple[pyxtal, np.ndarray, list[dict]]:
    records = requested_sites(row, indices)
    rep_values = [row[name] for name in BASE_COLUMNS]
    for slot in indices:
        rep_values.extend([row[f"wp{slot}"], row[f"x{slot}"], row[f"y{slot}"], row[f"z{slot}"]])
    rep = np.asarray(rep_values, dtype=float)

    number = int(round(float(row["spg"])))
    if not 1 <= number <= 230:
        raise ValueError(f"invalid space-group number {number}")
    group = Group(number)

    a, b, c, alpha, beta, gamma = [float(row[name]) for name in BASE_COLUMNS[1:]]
    if discrete_cell:
        if resolution is None:
            raise ValueError("discrete cell detected without a grid resolution")
        a, b, c = [value / resolution * 50.0 for value in (a, b, c)]
        alpha, beta, gamma = [value / resolution * 180.0 for value in (alpha, beta, gamma)]
    else:
        alpha, beta, gamma = np.degrees([alpha, beta, gamma])

    lattice = Lattice.from_para(
        a, b, c, alpha, beta, gamma,
        ltype=group.lattice_type,
        force_symmetry=True,
    )

    sites = {"Ti": [], "O": []}
    multiplicities = {"Ti": 0, "O": 0}
    expected = []
    for record in records:
        wp_index = record["wp_index"]
        if wp_index < 0 or wp_index >= len(group):
            raise ValueError(f"slot {record['slot']}: invalid Wyckoff index {wp_index}")
        wp = group[wp_index]
        xyz = record["position"].copy()
        if discrete:
            if resolution is None:
                raise ValueError("discrete coordinates detected without a grid resolution")
            xyz = np.asarray(wp.from_discrete_grid(xyz.tolist(), resolution), dtype=float)
        generator = wp.search_generator(
            xyz.tolist(),
            tol=0.1 if discrete else 0.01,
            symmetrize=True,
        )
        if generator is None:
            raise ValueError(f"slot {record['slot']}: Wyckoff generator search failed")
        generator = np.mod(np.asarray(generator, dtype=float), 1.0)
        symbol = record["species"]
        label = wp.get_label()
        sites[symbol].append((label, *[float(value) for value in generator]))
        multiplicities[symbol] += int(wp.multiplicity)
        expected.append({**record, "wp_label": label, "generator": generator})

    if not sites["Ti"] or not sites["O"]:
        raise ValueError("decoding lost the Ti or O sublattice")
    if multiplicities["O"] != 2 * multiplicities["Ti"]:
        raise ValueError(
            f"Wyckoff multiplicities are not TiO2: Ti={multiplicities['Ti']} O={multiplicities['O']}"
        )

    xtal = pyxtal()
    xtal.build(
        group,
        ["Ti", "O"],
        [multiplicities["Ti"], multiplicities["O"]],
        lattice,
        [sites["Ti"], sites["O"]],
    )
    if not xtal.valid or not xtal.atom_sites:
        raise ValueError("PyXtal returned an invalid or empty structure")
    if int(xtal.group.number) != number:
        raise ValueError(f"space group changed during decoding: {number} -> {xtal.group.number}")
    if len(xtal.atom_sites) != len(expected):
        raise ValueError(
            f"independent-site count changed: requested={len(expected)} decoded={len(xtal.atom_sites)}"
        )

    # Strict one-to-one species/Wyckoff/generator matching.
    available = list(range(len(expected)))
    for site in xtal.atom_sites:
        symbol = str(site.specie)
        label = site.wp.get_label()
        position = np.mod(np.asarray(site.position, dtype=float), 1.0)
        candidates = [
            idx for idx in available
            if expected[idx]["species"] == symbol and expected[idx]["wp_label"] == label
        ]
        if not candidates:
            raise ValueError(f"decoded site {symbol} {label} cannot be mapped to a requested site")
        chosen = min(candidates, key=lambda idx: periodic_fractional_distance(position, expected[idx]["generator"]))
        if periodic_fractional_distance(position, expected[chosen]["generator"]) > 0.05:
            raise ValueError(f"decoded generator moved unexpectedly for {symbol} {label}")
        available.remove(chosen)
    if available:
        raise ValueError(f"unmatched requested independent sites: {available}")

    composition = dict(zip(map(str, xtal.species), map(int, xtal.numIons)))
    if composition.get("Ti", 0) <= 0 or composition.get("O", 0) != 2 * composition.get("Ti", 0):
        raise ValueError(f"decoded structure is not TiO2: {composition}")
    return xtal, rep, expected


def decode_one(payload):
    row_index, row, indices, discrete, resolution, discrete_cell = payload
    try:
        xtal, rep, expected = build_tio2(row, indices, discrete, resolution, discrete_cell)
        return row_index, xtal, rep, expected, None
    except Exception as exc:
        return row_index, None, None, None, f"{type(exc).__name__}: {exc}"


def representation_key(xtal: pyxtal, decimals: int) -> str:
    x = np.round(np.asarray(xtal.get_1d_rep_x(), dtype=float), decimals).tolist()
    site_signature = [
        (str(site.specie), site.wp.get_label())
        for site in xtal.atom_sites
    ]
    payload = {
        "spg": int(xtal.group.number),
        "sites": site_signature,
        "x": x,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deduplicate(decoded: list[tuple], decimals: int):
    groups: dict[str, list[tuple]] = defaultdict(list)
    for item in decoded:
        groups[representation_key(item[1], decimals)].append(item)

    unique = []
    duplicate_rows = []
    for key, members in groups.items():
        members.sort(key=lambda item: item[0])
        representative = members[0]
        source_rows = [int(item[0]) for item in members]
        row_index, xtal, rep, expected, _ = representative
        xtal.tag = {
            "source_row": int(row_index),
            "representative_source_row": int(row_index),
            "source_rows": source_rows,
            "generation_count": len(source_rows),
            "dedup_key": key,
        }
        unique.append((row_index, xtal, rep, expected, source_rows, key))
        for duplicate in source_rows[1:]:
            duplicate_rows.append(
                {
                    "source_row": duplicate,
                    "representative_source_row": int(row_index),
                    "status": "duplicate_before_so3",
                    "dedup_key": key,
                }
            )
    unique.sort(key=lambda item: item[0])
    duplicate_rows.sort(key=lambda item: item["source_row"])
    return unique, duplicate_rows


def geometry_diagnostics(xtal: pyxtal, tio_cutoff: float) -> dict:
    """Return non-filtering TiO2 distance and coordination diagnostics."""
    atoms = xtal.to_ase(resort=False)
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    n = len(atoms)

    def pair_min(symbol_a: str, symbol_b: str) -> float:
        values = []
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = symbols[i], symbols[j]
                if symbol_a == symbol_b:
                    matched = si == symbol_a and sj == symbol_b
                else:
                    matched = (si == symbol_a and sj == symbol_b) or (si == symbol_b and sj == symbol_a)
                if matched:
                    values.append(float(distances[i, j]))
        return min(values) if values else math.inf

    ti_indices = np.where(symbols == "Ti")[0]
    o_indices = np.where(symbols == "O")[0]
    ti_cn = [int(np.sum(distances[i, o_indices] <= tio_cutoff)) for i in ti_indices]
    o_cn = [int(np.sum(distances[i, ti_indices] <= tio_cutoff)) for i in o_indices]

    def summarize(values: list[int], prefix: str) -> dict:
        if not values:
            return {
                f"{prefix}_mean": math.nan,
                f"{prefix}_std": math.nan,
                f"{prefix}_min": math.nan,
                f"{prefix}_max": math.nan,
                f"{prefix}_hist_json": "{}",
            }
        arr = np.asarray(values, dtype=float)
        return {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()),
            f"{prefix}_min": int(arr.min()),
            f"{prefix}_max": int(arr.max()),
            f"{prefix}_hist_json": json.dumps(dict(sorted(Counter(values).items())), separators=(",", ":")),
        }

    result = {
        "num_atoms": n,
        "num_ti": int(len(ti_indices)),
        "num_o": int(len(o_indices)),
        "min_ti_ti": pair_min("Ti", "Ti"),
        "min_ti_o": pair_min("Ti", "O"),
        "min_o_o": pair_min("O", "O"),
        "tio_coordination_cutoff": float(tio_cutoff),
    }
    result.update(summarize(ti_cn, "ti_o_cn"))
    result.update(summarize(o_cn, "o_ti_cn"))
    return result


def preflight_unique_structures(
    unique: list[tuple],
    min_ti_ti: float,
    min_ti_o: float,
    min_o_o: float,
    tio_cutoff: float,
) -> tuple[list[tuple], list[dict], list[dict]]:
    """Reject only catastrophic overlaps and retain diagnostics for every structure."""
    passed, rejected, diagnostics = [], [], []
    for item in unique:
        row_index, xtal, _rep, _expected, source_rows, key = item
        record = {
            "source_row": int(row_index),
            "generation_count": len(source_rows),
            "source_rows_json": json.dumps(source_rows, separators=(",", ":")),
            "dedup_key": key,
            **geometry_diagnostics(xtal, tio_cutoff),
        }
        reasons = []
        if record["min_ti_ti"] < min_ti_ti:
            reasons.append(f"Ti-Ti {record['min_ti_ti']:.6f} < {min_ti_ti:.6f}")
        if record["min_ti_o"] < min_ti_o:
            reasons.append(f"Ti-O {record['min_ti_o']:.6f} < {min_ti_o:.6f}")
        if record["min_o_o"] < min_o_o:
            reasons.append(f"O-O {record['min_o_o']:.6f} < {min_o_o:.6f}")
        record["preflight_passed"] = not reasons
        diagnostics.append(record)
        if reasons:
            rejected.append(record)
        else:
            passed.append(item)
    return passed, rejected, diagnostics


def apply_pre_so3_integrity_screen(
    unique: list[tuple],
    integrity_kwargs: dict,
) -> tuple[list[tuple], list[dict], list[dict]]:
    """Apply a loose, generator-facing TiO6/OTi3 screen before SO3."""
    passed, rejected, diagnostics = [], [], []
    for item in unique:
        row_index, xtal, _rep, _expected, source_rows, key = item
        record = {
            "source_row": int(row_index),
            "generation_count": len(source_rows),
            "source_rows_json": json.dumps(source_rows, separators=(",", ":")),
            "dedup_key": key,
        }
        try:
            structure = xtal.to_pymatgen()
            metrics = relaxed_integrity_metrics(structure, **integrity_kwargs)
            record.update(metrics)
            record["pre_integrity_valid"] = bool(metrics["integrity_valid"])
        except Exception as exc:
            record.update({
                "pre_integrity_valid": False,
                "integrity_valid": False,
                "local_reference_score": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })
        diagnostics.append(record)
        if record["pre_integrity_valid"]:
            passed.append(item)
        else:
            rejected.append(record)
    return passed, rejected, diagnostics


def write_so3_geometry_diagnostics(db_path: Path, output_path: Path, tio_cutoff: float) -> pd.DataFrame:
    rows = []
    if db_path.is_file():
        with connect(db_path) as db:
            for row in db.select():
                try:
                    xtal = pyxtal()
                    xtal.from_seed(row.toatoms())
                    metrics = geometry_diagnostics(xtal, tio_cutoff)
                    rows.append({
                        "so3_db_row": int(row.id),
                        "source_row": int(row.source_row) if hasattr(row, "source_row") else None,
                        "initial_so3_energy": float(row.similarity0) if hasattr(row, "similarity0") else math.nan,
                        "final_so3_energy": float(row.similarity) if hasattr(row, "similarity") else math.nan,
                        **metrics,
                        "error": None,
                    })
                except Exception as exc:
                    rows.append({
                        "so3_db_row": int(row.id),
                        "source_row": int(row.source_row) if hasattr(row, "source_row") else None,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def write_pre_so3_db(path: Path, unique: list[tuple], source_csv: str) -> None:
    if path.exists():
        path.unlink()
    db = connect(path)
    for row_index, xtal, rep, expected, source_rows, key in tqdm(unique, desc="Saving pre-SO3"):
        db.write(
            xtal.to_ase(resort=False),
            stage="pre_so3",
            source_csv=source_csv,
            source_row=int(row_index),
            representative_source_row=int(row_index),
            generation_count=len(source_rows),
            source_rows_json=json.dumps(source_rows, separators=(",", ":")),
            dedup_key=key,
            data={
                "representation": rep.tolist(),
                "requested_sites": [
                    {
                        "slot": int(item["slot"]),
                        "species": item["species"],
                        "wp_index": int(item["wp_index"]),
                        "wp_label": item["wp_label"],
                    }
                    for item in expected
                ],
            },
        )


def copy_selected_so3_rows(source_db: Path, target_db: Path, max_so3: float) -> tuple[int, list[dict]]:
    if target_db.exists():
        target_db.unlink()
    selected = 0
    records = []
    with connect(source_db) as src, connect(target_db) as dst:
        for row in src.select():
            sim = float(row.similarity) if hasattr(row, "similarity") else math.inf
            accepted = math.isfinite(sim) and sim <= max_so3
            record = {
                "so3_db_row": int(row.id),
                "source_row": int(row.source_row) if hasattr(row, "source_row") else None,
                "representative_source_row": int(row.representative_source_row) if hasattr(row, "representative_source_row") else None,
                "generation_count": int(row.generation_count) if hasattr(row, "generation_count") else 1,
                "source_rows_json": row.source_rows_json if hasattr(row, "source_rows_json") else "[]",
                "initial_so3_energy": float(row.similarity0),
                "final_so3_energy": sim,
                "passed_so3_filter": accepted,
            }
            records.append(record)
            if accepted:
                kvp = dict(row.key_value_pairs)
                kvp["so3_db_row"] = int(row.id)
                # Direct element-specific SO3 uses no atom-site target metadata.
                # Removing it prevents irrelevant Wyckoff-label remapping errors
                # when a relaxed orbit is equivalently relabelled (e.g. 48e->48g).
                for key in (
                    "site_properties_json",
                    "cn_labels_json",
                    "cn_wp_labels",
                ):
                    kvp.pop(key, None)
                dst.write(row.toatoms(), key_value_pairs=kvp, data=row.data)
                selected += 1
    return selected, records



_MATCH_STRUCTURES = None
_MATCHER = None


def _init_structure_match_worker(structures, matcher_kwargs):
    global _MATCH_STRUCTURES, _MATCHER
    _MATCH_STRUCTURES = structures
    _MATCHER = StructureMatcher(
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
        allow_subset=False,
        comparator=ElementComparator(),
        **matcher_kwargs,
    )


def _match_structure_pair(pair):
    i, j = pair
    try:
        matched = bool(_MATCHER.fit(_MATCH_STRUCTURES[i], _MATCH_STRUCTURES[j]))
        rms = max_dist = None
        if matched:
            distances = _MATCHER.get_rms_dist(_MATCH_STRUCTURES[i], _MATCH_STRUCTURES[j])
            if distances is not None:
                rms, max_dist = map(float, distances)
        return i, j, matched, rms, max_dist, None
    except Exception as exc:
        return i, j, False, None, None, f"{type(exc).__name__}: {exc}"


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _candidate_match_pairs(successful, structures, energy_window, volume_tol):
    pairs = []
    for i in range(len(successful)):
        rec_i = successful[i][0]
        s_i = structures[i]
        formula_i = s_i.composition.reduced_formula
        vpa_i = s_i.volume / max(s_i.num_sites, 1)
        for j in range(i + 1, len(successful)):
            rec_j = successful[j][0]
            s_j = structures[j]
            if formula_i != s_j.composition.reduced_formula:
                continue
            if math.isfinite(energy_window):
                if abs(rec_i["ff_energy_eV_per_atom"] - rec_j["ff_energy_eV_per_atom"]) > energy_window:
                    continue
            if math.isfinite(volume_tol):
                vpa_j = s_j.volume / max(s_j.num_sites, 1)
                rel = abs(vpa_i - vpa_j) / max(vpa_i, vpa_j, 1e-12)
                if rel > volume_tol:
                    continue
            pairs.append((i, j))
    return pairs



IDEAL_OCTAHEDRAL_ANGLES = np.asarray([90.0] * 12 + [180.0] * 3, dtype=float)


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return math.nan
    cosine = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _periodic_species_neighbors(
    structure: Structure,
    center_index: int,
    neighbor_symbol: str,
    radius: float,
) -> list[tuple[float, np.ndarray]]:
    center = structure[center_index]
    neighbors = []
    for neighbor in structure.get_neighbors(
        center,
        radius,
        include_index=True,
        include_image=True,
    ):
        if neighbor.specie.symbol != neighbor_symbol:
            continue
        vector = np.asarray(neighbor.coords - center.coords, dtype=float)
        neighbors.append((float(neighbor.nn_distance), vector))
    neighbors.sort(key=lambda item: item[0])
    return neighbors


def relaxed_integrity_metrics(
    structure: Structure,
    search_radius: float,
    max_ti_o: float,
    max_o_ti: float,
    max_angle_rms: float,
    min_ti_fraction: float,
    min_o_fraction: float,
    reference_ti_o_6th_A: float | None = None,
    reference_o_ti_3rd_A: float | None = None,
    reference_tio6_angle_rms_deg: float | None = None,
) -> dict:
    """Evaluate periodic TiO6 and OTi3 integrity for one relaxed structure."""
    ti_distances = []
    ti_angle_rms = []
    ti_pass = []
    o_distances = []
    o_pass = []
    failures = []

    for index, site in enumerate(structure):
        symbol = site.specie.symbol
        if symbol == "Ti":
            neighbors = _periodic_species_neighbors(structure, index, "O", search_radius)
            if len(neighbors) < 6:
                ti_distances.append(math.inf)
                ti_angle_rms.append(math.inf)
                ti_pass.append(False)
                continue
            shell = neighbors[:6]
            sixth = float(shell[-1][0])
            vectors = [item[1] for item in shell]
            angles = np.sort(np.asarray([
                _vector_angle_deg(vectors[i], vectors[j])
                for i in range(6) for j in range(i + 1, 6)
            ], dtype=float))
            angle_rms = float(np.sqrt(np.mean((angles - IDEAL_OCTAHEDRAL_ANGLES) ** 2)))
            ti_distances.append(sixth)
            ti_angle_rms.append(angle_rms)
            ti_pass.append(
                math.isfinite(sixth)
                and math.isfinite(angle_rms)
                and sixth <= max_ti_o
                and angle_rms <= max_angle_rms
            )
        elif symbol == "O":
            neighbors = _periodic_species_neighbors(structure, index, "Ti", search_radius)
            if len(neighbors) < 3:
                o_distances.append(math.inf)
                o_pass.append(False)
                continue
            third = float(neighbors[2][0])
            o_distances.append(third)
            o_pass.append(math.isfinite(third) and third <= max_o_ti)

    ti_fraction = float(np.mean(ti_pass)) if ti_pass else 0.0
    o_fraction = float(np.mean(o_pass)) if o_pass else 0.0
    ti_sixth_max = max(ti_distances, default=math.inf)
    ti_sixth_mean = float(np.mean(ti_distances)) if ti_distances else math.inf
    angle_max = max(ti_angle_rms, default=math.inf)
    angle_mean = float(np.mean(ti_angle_rms)) if ti_angle_rms else math.inf
    o_third_max = max(o_distances, default=math.inf)
    o_third_mean = float(np.mean(o_distances)) if o_distances else math.inf

    if ti_fraction < min_ti_fraction:
        failures.append("ti_fraction")
    if o_fraction < min_o_fraction:
        failures.append("o_fraction")
    valid = not failures

    refs = (
        reference_ti_o_6th_A,
        reference_o_ti_3rd_A,
        reference_tio6_angle_rms_deg,
    )
    vals = (ti_sixth_mean, o_third_mean, angle_mean)
    if all(v is not None and math.isfinite(float(v)) for v in refs) and all(
        math.isfinite(float(v)) for v in vals
    ):
        ti_width = max(0.10, float(max_ti_o) - float(reference_ti_o_6th_A))
        o_width = max(0.10, float(max_o_ti) - float(reference_o_ti_3rd_A))
        angle_width = max(
            5.0,
            float(max_angle_rms) - float(reference_tio6_angle_rms_deg),
        )
        d2 = (
            ((ti_sixth_mean - float(reference_ti_o_6th_A)) / ti_width) ** 2
            + ((o_third_mean - float(reference_o_ti_3rd_A)) / o_width) ** 2
            + ((angle_mean - float(reference_tio6_angle_rms_deg)) / angle_width) ** 2
            + (1.0 - ti_fraction) ** 2
            + (1.0 - o_fraction) ** 2
        ) / 5.0
        local_reference_score = float(math.exp(-0.5 * d2))
    else:
        local_reference_score = 0.0

    return {
        "integrity_valid": bool(valid),
        "local_reference_score": local_reference_score,
        "ti_o6_pass_fraction": ti_fraction,
        "o_ti3_pass_fraction": o_fraction,
        "ti_o_6th_max_A": float(ti_sixth_max),
        "ti_o_6th_mean_A": float(ti_sixth_mean),
        "tio6_angle_rms_max_deg": float(angle_max),
        "tio6_angle_rms_mean_deg": float(angle_mean),
        "o_ti_3rd_max_A": float(o_third_max),
        "o_ti_3rd_mean_A": float(o_third_mean),
    }


def _integrity_ranking_key(record: dict) -> tuple:
    """Rank by continuous local-reference similarity, then physical energy."""
    return (
        -float(record.get("local_reference_score", 0.0)),
        float(record.get("ff_energy_eV_per_atom", math.inf)),
        float(record.get("final_so3_energy", math.inf)),
        int(record.get("representative_source_row", 10**12)),
    )


def write_ranked_outputs(
    gulp_db_path: Path,
    output_dir: Path,
    ncpu: int,
    matcher_kwargs: dict,
    energy_window: float,
    volume_tol: float,
    chunksize: int,
    integrity_kwargs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank all GULP minima and deduplicate by direct periodic-structure matching.

    Energy and volume are used only as cheap pair prefilters. Duplicate identity
    is decided exclusively by pymatgen StructureMatcher.fit(). Pair comparisons
    are independent and therefore distributed over CPU workers; matched pairs
    are merged into connected components with a deterministic union-find pass.
    """
    candidates_dir = output_dir / "candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True)

    successful = []
    failed = []
    parse_failed = []
    with connect(gulp_db_path) as db:
        for row in db.select():
            base = {
                "gulp_db_row": int(row.id),
                "so3_db_row": int(row.so3_db_row) if hasattr(row, "so3_db_row") else None,
                "source_row": int(row.source_row) if hasattr(row, "source_row") else None,
                "representative_source_row": int(row.representative_source_row) if hasattr(row, "representative_source_row") else None,
                "generation_count": int(row.generation_count) if hasattr(row, "generation_count") else 1,
                "source_rows_json": row.source_rows_json if hasattr(row, "source_rows_json") else "[]",
                "space_group": int(row.space_group_number),
                "wyckoff_pattern": str(row.wps) if hasattr(row, "wps") else "",
                "pearson_symbol": row.pearson_symbol,
                "num_atoms": int(row.natoms),
                "initial_so3_energy": float(row.similarity0),
                "final_so3_energy": float(row.similarity),
            }
            if hasattr(row, "ff_energy") and row.ff_energy is not None and hasattr(row, "ff_relaxed"):
                cif_text = str(row.ff_relaxed)
                try:
                    structure = Structure.from_str(cif_text, fmt="cif")
                except Exception as exc:
                    parse_failed.append({
                        **base,
                        "failure_reason": f"Cannot parse ff_relaxed CIF: {type(exc).__name__}: {exc}",
                    })
                    continue
                base["ff_energy_eV_per_atom"] = float(row.ff_energy)
                base["relaxed_formula"] = structure.composition.reduced_formula
                base["relaxed_num_sites"] = int(structure.num_sites)
                base["relaxed_volume_per_atom"] = float(structure.volume / structure.num_sites)
                base.update(relaxed_integrity_metrics(structure, **integrity_kwargs))
                successful.append((base, cif_text, structure))
            else:
                failed.append({**base, "failure_reason": "GULP result missing"})

    energy_order = {
        item[0]["gulp_db_row"]: rank
        for rank, item in enumerate(
            sorted(successful, key=lambda value: value[0]["ff_energy_eV_per_atom"]),
            start=1,
        )
    }
    successful.sort(key=lambda item: _integrity_ranking_key(item[0]))
    all_ranked_df = pd.DataFrame([
        {
            "all_rank": rank,
            "energy_rank": energy_order[record["gulp_db_row"]],
            **record,
        }
        for rank, (record, _cif, _structure) in enumerate(successful, start=1)
    ])
    _strip_reason_columns(all_ranked_df).to_csv(output_dir / "ranked_all_candidates.csv", index=False)

    structures = [item[2] for item in successful]
    pairs = _candidate_match_pairs(successful, structures, energy_window, volume_tol)
    print(
        f"Post-GULP StructureMatcher: {len(successful)} structures; "
        f"{len(pairs)} candidate pairs; {min(ncpu, max(len(pairs), 1))} worker(s)"
    )

    uf = _UnionFind(len(successful))
    match_rows = []
    if pairs:
        worker_count = min(ncpu, len(pairs))
        if worker_count == 1:
            _init_structure_match_worker(structures, matcher_kwargs)
            results = map(_match_structure_pair, pairs)
            for i, j, matched, rms, max_dist, error in tqdm(
                results, total=len(pairs), desc="Structure matching"
            ):
                if matched:
                    uf.union(i, j)
                match_rows.append({
                    "gulp_db_row_a": successful[i][0]["gulp_db_row"],
                    "gulp_db_row_b": successful[j][0]["gulp_db_row"],
                    "matched": matched,
                    "normalized_rms_distance": rms,
                    "normalized_max_distance": max_dist,
                    "error": error,
                })
        else:
            from multiprocessing import Pool
            with Pool(
                processes=worker_count,
                initializer=_init_structure_match_worker,
                initargs=(structures, matcher_kwargs),
            ) as pool:
                results = pool.imap_unordered(
                    _match_structure_pair, pairs, chunksize=max(1, chunksize)
                )
                for i, j, matched, rms, max_dist, error in tqdm(
                    results, total=len(pairs), desc="Structure matching"
                ):
                    if matched:
                        uf.union(i, j)
                    match_rows.append({
                        "gulp_db_row_a": successful[i][0]["gulp_db_row"],
                        "gulp_db_row_b": successful[j][0]["gulp_db_row"],
                        "matched": matched,
                        "normalized_rms_distance": rms,
                        "normalized_max_distance": max_dist,
                        "error": error,
                    })

    components = defaultdict(list)
    for i in range(len(successful)):
        components[uf.find(i)].append(i)

    retained = []
    duplicate_rows = []
    for component_id, member_indices in enumerate(components.values(), start=1):
        member_indices.sort(key=lambda idx: _integrity_ranking_key(successful[idx][0]))
        rep_idx = member_indices[0]
        representative, cif_text, _ = successful[rep_idx]
        member_gulp_rows = [successful[idx][0]["gulp_db_row"] for idx in member_indices]
        member_source_rows = [successful[idx][0]["representative_source_row"] for idx in member_indices]
        retained.append(({
            **representative,
            "structure_match_component": component_id,
            "post_gulp_multiplicity": len(member_indices),
            "post_gulp_gulp_rows_json": json.dumps(member_gulp_rows, separators=(",", ":")),
            "post_gulp_source_rows_json": json.dumps(member_source_rows, separators=(",", ":")),
        }, cif_text))
        for idx in member_indices[1:]:
            duplicate = successful[idx][0]
            duplicate_rows.append({
                "structure_match_component": component_id,
                "representative_source_row": duplicate["representative_source_row"],
                "retained_source_row": representative["representative_source_row"],
                "gulp_db_row": duplicate["gulp_db_row"],
                "retained_gulp_db_row": representative["gulp_db_row"],
                "ff_energy_eV_per_atom": duplicate["ff_energy_eV_per_atom"],
                "retained_ff_energy_eV_per_atom": representative["ff_energy_eV_per_atom"],
                "final_so3_energy": duplicate["final_so3_energy"],
                "space_group": duplicate["space_group"],
                "wyckoff_pattern": duplicate["wyckoff_pattern"],
                "duplicate_decision": "pymatgen StructureMatcher connected component",
            })

    retained.sort(key=lambda item: _integrity_ranking_key(item[0]))
    ranked = []
    for rank, (record, cif_text) in enumerate(retained, start=1):
        cif_name = f"rank_{rank:04d}_row_{record['representative_source_row']}.cif"
        cif_path = candidates_dir / cif_name
        cif_path.write_text(cif_text)
        ranked.append({"rank": rank, **record, "cif_path": str(cif_path)})

    ranked_df = pd.DataFrame(ranked)
    duplicate_df = pd.DataFrame(duplicate_rows)
    match_df = pd.DataFrame(match_rows)
    failed_df = pd.DataFrame(failed + parse_failed)
    _strip_reason_columns(ranked_df).to_csv(output_dir / "ranked_candidates.csv", index=False)
    if ranked_df.empty:
        valid_ranked_df = ranked_df.copy()
        invalid_ranked_df = ranked_df.copy()
    else:
        valid_ranked_df = ranked_df[ranked_df["integrity_valid"] == True].copy()
        invalid_ranked_df = ranked_df[ranked_df["integrity_valid"] == False].copy()
        valid_ranked_df.insert(0, "integrity_valid_rank", range(1, len(valid_ranked_df) + 1))
        invalid_ranked_df.insert(0, "integrity_invalid_rank", range(1, len(invalid_ranked_df) + 1))
    _strip_reason_columns(valid_ranked_df).to_csv(output_dir / "ranked_valid_candidates.csv", index=False)
    _strip_reason_columns(invalid_ranked_df).to_csv(output_dir / "ranked_invalid_candidates.csv", index=False)
    _strip_reason_columns(duplicate_df).to_csv(output_dir / "post_gulp_duplicates.csv", index=False)
    _strip_reason_columns(match_df).to_csv(output_dir / "structure_match_pairs.csv", index=False)
    _strip_reason_columns(failed_df).to_csv(output_dir / "gulp_failures.csv", index=False)
    return ranked_df, all_ranked_df, duplicate_df, match_df, failed_df

def compare_ranked_to_training_set(
    ranked_df: pd.DataFrame,
    training_db_path: Path,
    output_dir: Path,
    ncpu: int,
    matcher_kwargs: dict,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate retained candidates that reproduce structures in the training DB.

    The original LEGO-Xtal pipeline performed a source-database overlap check.
    Here the same scientific check is applied directly to the final GULP-relaxed
    candidates with pymatgen StructureMatcher. Comparisons are parallel and
    matches are reported, not removed.
    """
    empty_columns = [
        "candidate_rank", "candidate_source_row", "candidate_gulp_db_row",
        "training_db_row", "training_label", "normalized_rms_distance",
        "normalized_max_distance", "error",
    ]
    if ranked_df.empty:
        match_df = pd.DataFrame(columns=empty_columns)
        _strip_reason_columns(match_df).to_csv(output_dir / "training_set_matches.csv", index=False)
        return ranked_df, match_df

    candidate_structures = []
    valid_candidate_rows = []
    for _, row in ranked_df.iterrows():
        try:
            candidate_structures.append(Structure.from_file(str(row["cif_path"])))
            valid_candidate_rows.append(row)
        except Exception as exc:
            print(
                f"Cannot parse ranked candidate {row.get('rank')}: "
                f"{type(exc).__name__}: {exc}"
            )

    training_structures = []
    training_meta = []
    adaptor = AseAtomsAdaptor()
    with connect(training_db_path) as db:
        for row in db.select():
            try:
                structure = adaptor.get_structure(row.toatoms())
            except Exception as exc:
                print(
                    f"Cannot load training DB row {row.id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            label = None
            for key in ("name", "label", "prototype", "pearson_symbol"):
                if hasattr(row, key):
                    value = getattr(row, key)
                    if value is not None:
                        label = str(value)
                        break
            training_structures.append(structure)
            training_meta.append({
                "training_db_row": int(row.id),
                "training_label": label or f"row_{row.id}",
            })

    if not candidate_structures or not training_structures:
        match_df = pd.DataFrame(columns=empty_columns)
        _strip_reason_columns(match_df).to_csv(output_dir / "training_set_matches.csv", index=False)
        annotated = ranked_df.copy()
        annotated["in_training_set"] = False
        annotated["training_match_count"] = 0
        annotated["training_db_rows_json"] = "[]"
        annotated["best_training_db_row"] = None
        annotated["best_training_rms"] = None
        annotated["best_training_max_dist"] = None
        _strip_reason_columns(annotated).to_csv(output_dir / "ranked_candidates.csv", index=False)
        return annotated, match_df

    combined = candidate_structures + training_structures
    offset = len(candidate_structures)
    pairs = []
    for i, cand in enumerate(candidate_structures):
        formula = cand.composition.reduced_formula
        for j, train in enumerate(training_structures):
            if formula == train.composition.reduced_formula:
                pairs.append((i, offset + j))

    print(
        f"Training-set overlap: {len(candidate_structures)} candidates x "
        f"{len(training_structures)} references; {len(pairs)} formula-compatible "
        f"pairs; {min(ncpu, max(len(pairs), 1))} worker(s)"
    )

    raw_results = []
    if pairs:
        worker_count = min(ncpu, len(pairs))
        if worker_count == 1:
            _init_structure_match_worker(combined, matcher_kwargs)
            iterator = map(_match_structure_pair, pairs)
            for result in tqdm(iterator, total=len(pairs), desc="Training overlap"):
                raw_results.append(result)
        else:
            from multiprocessing import Pool
            with Pool(
                processes=worker_count,
                initializer=_init_structure_match_worker,
                initargs=(combined, matcher_kwargs),
            ) as pool:
                iterator = pool.imap_unordered(
                    _match_structure_pair, pairs, chunksize=max(1, chunksize)
                )
                for result in tqdm(iterator, total=len(pairs), desc="Training overlap"):
                    raw_results.append(result)

    matches_by_candidate = defaultdict(list)
    match_rows = []
    for i, combined_j, matched, rms, max_dist, error in raw_results:
        j = combined_j - offset
        if not matched:
            continue
        candidate_row = valid_candidate_rows[i]
        meta = training_meta[j]
        record = {
            "candidate_rank": int(candidate_row["rank"]),
            "candidate_source_row": int(candidate_row["representative_source_row"]),
            "candidate_gulp_db_row": int(candidate_row["gulp_db_row"]),
            "training_db_row": meta["training_db_row"],
            "training_label": meta["training_label"],
            "normalized_rms_distance": rms,
            "normalized_max_distance": max_dist,
            "error": error,
        }
        match_rows.append(record)
        matches_by_candidate[int(candidate_row["rank"])].append(record)

    annotated = ranked_df.copy()
    annotations = []
    for _, row in annotated.iterrows():
        records = matches_by_candidate.get(int(row["rank"]), [])
        records.sort(key=lambda rec: (
            float("inf") if rec["normalized_rms_distance"] is None else rec["normalized_rms_distance"],
            rec["training_db_row"],
        ))
        best = records[0] if records else None
        annotations.append({
            "in_training_set": bool(records),
            "training_match_count": len(records),
            "training_db_rows_json": json.dumps(
                [rec["training_db_row"] for rec in records], separators=(",", ":")
            ),
            "best_training_db_row": None if best is None else best["training_db_row"],
            "best_training_label": None if best is None else best["training_label"],
            "best_training_rms": None if best is None else best["normalized_rms_distance"],
            "best_training_max_dist": None if best is None else best["normalized_max_distance"],
        })
    annotation_df = pd.DataFrame(annotations)
    annotated = pd.concat([annotated.reset_index(drop=True), annotation_df], axis=1)
    match_df = pd.DataFrame(match_rows, columns=empty_columns)
    _strip_reason_columns(annotated).to_csv(output_dir / "ranked_candidates.csv", index=False)
    _strip_reason_columns(match_df).to_csv(output_dir / "training_set_matches.csv", index=False)
    return annotated, match_df



def _is_post_so3_duplicate(structure, retained_structures, matcher):
    """Return the retained index matching structure, or None when unique."""
    for index, retained in enumerate(retained_structures):
        try:
            if matcher.fit(retained, structure):
                return index
        except Exception:
            continue
    return None


def _write_retained_so3_db(
    work_db_path: Path,
    final_db_path: Path,
    retained_source_rows: list[int],
) -> int:
    """Copy only retained post-SO3 representatives into the production DB."""
    if final_db_path.exists():
        final_db_path.unlink()
    retained_order = {int(source_row): rank for rank, source_row in enumerate(retained_source_rows)}
    rows = []
    with connect(work_db_path) as src:
        for row in src.select():
            source_row = int(row.source_row) if hasattr(row, "source_row") else None
            if source_row in retained_order:
                rows.append((retained_order[source_row], row))
    rows.sort(key=lambda item: item[0])
    with connect(final_db_path) as dst:
        for _, row in rows:
            dst.write(
                row.toatoms(),
                key_value_pairs=dict(row.key_value_pairs),
                data=row.data,
            )
    return len(rows)


def _strip_reason_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            column for column in frame.columns
            if "reason" in str(column).lower()
        ],
        errors="ignore",
    )


def main() -> None:
    args = parse_args()
    start = time()

    if args.ncpu < 1:
        raise ValueError("--ncpu must be at least 1")
    if args.begin < 0 or (args.end != -1 and args.end <= args.begin):
        raise ValueError("invalid --begin/--end range")
    if args.rcut <= 0:
        raise ValueError("--rcut must be positive")
    if min(args.min_ti_ti, args.min_ti_o, args.min_o_o, args.tio_coordination_cutoff) <= 0:
        raise ValueError("geometry floors and --tio-coordination-cutoff must be positive")
    if args.dedup_decimals < 3:
        raise ValueError("--dedup-decimals must be at least 3")
    if args.match_ltol <= 0 or args.match_stol <= 0 or args.match_angle_tol <= 0:
        raise ValueError("post-GULP StructureMatcher tolerances must be positive")
    if (
        args.post_so3_match_ltol <= 0
        or args.post_so3_match_stol <= 0
        or args.post_so3_match_angle_tol <= 0
    ):
        raise ValueError("post-SO3 StructureMatcher tolerances must be positive")
    if args.match_chunksize < 1:
        raise ValueError("--match-chunksize must be at least 1")
    if args.so3_fill_batch_size < 1:
        raise ValueError("--so3-fill-batch-size must be at least 1")
    if min(
        args.pre_integrity_neighbor_search_radius,
        args.pre_integrity_max_ti_o,
        args.pre_integrity_max_o_ti,
        args.pre_integrity_max_angle_rms,
    ) <= 0:
        raise ValueError("pre-integrity distances/radius/angle threshold must be positive")
    for name, value in (
        ("pre_integrity_min_ti_pass_fraction", args.pre_integrity_min_ti_pass_fraction),
        ("pre_integrity_min_o_pass_fraction", args.pre_integrity_min_o_pass_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")

    if min(
        args.integrity_neighbor_search_radius,
        args.integrity_max_ti_o,
        args.integrity_max_o_ti,
        args.integrity_max_angle_rms,
    ) <= 0:
        raise ValueError("integrity distances/radius/angle threshold must be positive")
    for name, value in (
        ("integrity_min_ti_pass_fraction", args.integrity_min_ti_pass_fraction),
        ("integrity_min_o_pass_fraction", args.integrity_min_o_pass_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")

    csv_path = Path(args.csv)
    ref_path = Path(args.reference_tio2)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)

    output_dir = Path(args.output_dir or csv_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_structure = Structure.from_file(ref_path)
    reference_metrics = relaxed_integrity_metrics(
        reference_structure,
        search_radius=max(
            args.pre_integrity_neighbor_search_radius,
            args.integrity_neighbor_search_radius,
        ),
        max_ti_o=max(args.pre_integrity_max_ti_o, args.integrity_max_ti_o),
        max_o_ti=max(args.pre_integrity_max_o_ti, args.integrity_max_o_ti),
        max_angle_rms=max(
            args.pre_integrity_max_angle_rms,
            args.integrity_max_angle_rms,
        ),
        min_ti_fraction=0.0,
        min_o_fraction=0.0,
    )
    local_reference_targets = {
        "reference_ti_o_6th_A": reference_metrics["ti_o_6th_mean_A"],
        "reference_o_ti_3rd_A": reference_metrics["o_ti_3rd_mean_A"],
        "reference_tio6_angle_rms_deg": reference_metrics[
            "tio6_angle_rms_mean_deg"
        ],
    }

    print("--- TiO2 final relaxation ---")
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"CPU workers: {args.ncpu}")
    print(f"SO3 reference: {ref_path}")
    print(f"SO3 cutoff: {args.rcut}")
    print(
        "Representation mode: "
        f"coordinates={'discrete' if args.discrete_coordinates else 'continuous'}, "
        f"cell={'discrete' if args.discrete_cell else 'continuous'}, "
        f"resolution={args.resolution}"
    )
    print(
        "Catastrophic-distance preflight: "
        f"Ti-Ti>={args.min_ti_ti}, Ti-O>={args.min_ti_o}, O-O>={args.min_o_o} A"
    )
    print(f"Diagnostic Ti-O coordination cutoff: {args.tio_coordination_cutoff} A")
    print(f"Maximum SO3 energy passed to GULP: {args.max_so3_energy}")
    print(f"SO3 stage-stop energy: {args.so3_stop_energy}")
    print(f"Initial SO3 pre-screen: {args.max_initial_so3_energy}")
    print(f"SO3 schedule: Nelder-Mead {args.nm_steps} + L-BFGS-B {args.lbfgs_steps}")
    print(f"GULP force field: {args.ff_lib}")
    print(
        "Loose pre-SO3 TiO6/OTi3 screen: "
        f"Ti-O6<={args.pre_integrity_max_ti_o} A, "
        f"O-Ti3<={args.pre_integrity_max_o_ti} A, "
        f"TiO6 RMS<={args.pre_integrity_max_angle_rms} deg, "
        f"pass fractions={args.pre_integrity_min_ti_pass_fraction}/"
        f"{args.pre_integrity_min_o_pass_fraction}"
    )
    print(
        "Post-GULP TiO6/OTi3 integrity: "
        f"Ti-O6<={args.integrity_max_ti_o} A, "
        f"O-Ti3<={args.integrity_max_o_ti} A, "
        f"TiO6 RMS<={args.integrity_max_angle_rms} deg, "
        f"pass fractions={args.integrity_min_ti_pass_fraction}/"
        f"{args.integrity_min_o_pass_fraction}"
    )
    print(
        "Local-reference targets: "
        f"Ti-O6={local_reference_targets['reference_ti_o_6th_A']:.4f} A, "
        f"O-Ti3={local_reference_targets['reference_o_ti_3rd_A']:.4f} A, "
        f"TiO6 angular RMS="
        f"{local_reference_targets['reference_tio6_angle_rms_deg']:.3f} deg"
    )
    print(
        "Post-SO3 GULP-pool matching: "
        f"scale=False, ltol={args.post_so3_match_ltol}, "
        f"stol={args.post_so3_match_stol}, "
        f"angle_tol={args.post_so3_match_angle_tol}"
    )
    print(
        "Post-GULP direct structure matching: "
        f"ltol={args.match_ltol}, stol={args.match_stol}, "
        f"angle_tol={args.match_angle_tol}, workers={args.ncpu}"
    )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("input CSV is empty")
    indices = discover_site_indices(df)
    discrete = bool(args.discrete_coordinates)
    discrete_cell = bool(args.discrete_cell)
    resolution = args.resolution
    if (discrete or discrete_cell) and (resolution is None or resolution <= 0):
        raise ValueError("--resolution must be positive when discrete coordinates or cell mode is enabled")
    # Decode the complete CSV first. --begin/--end are intentionally applied
    # only after exact-representation deduplication and loose pre-SO3 screening,
    # so --end=200 means 200 unique, generator-valid structures rather than the
    # first 200 raw CSV rows.
    payloads = [
        (int(idx), row.to_dict(), indices, discrete, resolution, discrete_cell)
        for idx, row in df.iterrows()
    ]

    from multiprocessing import Pool

    decoded, decode_failures = [], []
    if args.ncpu == 1:
        iterator = map(decode_one, payloads)
        for result in tqdm(iterator, total=len(payloads), desc="Strict decoding"):
            if result[1] is None:
                decode_failures.append({"source_row": result[0], "failure_reason": result[4]})
            else:
                decoded.append(result)
    else:
        with Pool(processes=args.ncpu) as pool:
            iterator = pool.imap(decode_one, payloads, chunksize=1)
            for result in tqdm(iterator, total=len(payloads), desc="Strict decoding"):
                if result[1] is None:
                    decode_failures.append({"source_row": result[0], "failure_reason": result[4]})
                else:
                    decoded.append(result)

    decoded.sort(key=lambda item: item[0])
    pd.DataFrame(decode_failures).to_csv(output_dir / "decode_failures.csv", index=False)
    if not decoded:
        raise RuntimeError("no rows survived strict decoding")

    unique, duplicates = deduplicate(decoded, args.dedup_decimals)
    deduplicated_count = len(unique)
    unique, geometry_rejected, pre_geometry = preflight_unique_structures(
        unique,
        args.min_ti_ti,
        args.min_ti_o,
        args.min_o_o,
        args.tio_coordination_cutoff,
    )
    _strip_reason_columns(pd.DataFrame(pre_geometry)).to_csv(output_dir / "pre_so3_geometry.csv", index=False)
    _strip_reason_columns(pd.DataFrame(geometry_rejected)).to_csv(output_dir / "geometry_preflight_failures.csv", index=False)
    if not unique:
        raise RuntimeError("no structures survived catastrophic-distance preflight")

    unique, pre_integrity_rejected, pre_integrity_diagnostics = apply_pre_so3_integrity_screen(
        unique,
        {
            "search_radius": args.pre_integrity_neighbor_search_radius,
            "max_ti_o": args.pre_integrity_max_ti_o,
            "max_o_ti": args.pre_integrity_max_o_ti,
            "max_angle_rms": args.pre_integrity_max_angle_rms,
            "min_ti_fraction": args.pre_integrity_min_ti_pass_fraction,
            "min_o_fraction": args.pre_integrity_min_o_pass_fraction,
            **local_reference_targets,
        },
    )
    _strip_reason_columns(pd.DataFrame(pre_integrity_diagnostics)).to_csv(
        output_dir / "pre_so3_integrity.csv", index=False
    )
    _strip_reason_columns(pd.DataFrame(pre_integrity_rejected)).to_csv(
        output_dir / "pre_so3_integrity_failures.csv", index=False
    )
    print(
        f"Decoded {len(decoded)}/{len(payloads)}; unique before geometry preflight: {deduplicated_count}; "
        f"obvious duplicates removed: {len(duplicates)}; "
        f"catastrophic overlaps rejected: {len(geometry_rejected)}; "
        f"loose pre-SO3 integrity rejected: {len(pre_integrity_rejected)}; "
        f"unique available before requested range: {len(unique)}"
    )
    if not unique:
        raise RuntimeError("no structures survived loose pre-SO3 integrity screening")

    unique_available = len(unique)
    candidate_pool = unique[args.begin:]
    if not candidate_pool:
        raise ValueError(
            "selected unique-structure range is empty after deduplication and "
            f"pre-SO3 screening: available={unique_available}, begin={args.begin}"
        )

    target_post_so3_unique = (
        len(candidate_pool)
        if args.end == -1
        else args.end - args.begin
    )
    if target_post_so3_unique < 1:
        raise ValueError("requested post-SO3 unique target must be positive")

    print(
        "Post-SO3 unique-pool target: "
        f"{target_post_so3_unique}; source candidates available from index "
        f"{args.begin}: {len(candidate_pool)}; refill batch="
        f"{args.so3_fill_batch_size}"
    )

    provenance = []
    for failure in decode_failures:
        provenance.append({
            "source_row": int(failure["source_row"]),
            "representative_source_row": None,
            "status": "decode_failed",
            "failure_reason": failure["failure_reason"],
        })
    provenance.extend(duplicates)
    for rejection in geometry_rejected:
        provenance.append({
            "source_row": int(rejection["source_row"]),
            "representative_source_row": int(rejection["source_row"]),
            "status": "geometry_preflight_failed",
        })
    for rejection in pre_integrity_rejected:
        provenance.append({
            "source_row": int(rejection["source_row"]),
            "representative_source_row": int(rejection["source_row"]),
            "status": "pre_so3_integrity_failed",
        })

    so3_work_prefix = output_dir / "so3_work"
    so3_work_db_path = output_dir / "so3_work-0.db"
    so3_db_path = output_dir / "so3-0.db"
    for path in (
        so3_work_db_path,
        Path(str(so3_work_prefix) + "-0.log"),
        so3_db_path,
    ):
        if path.exists():
            path.unlink()

    bu = builder(["Ti", "O"], [1, 2], rank=0, prefix=str(so3_work_prefix))
    bu.set_descriptor_calculator(mykwargs={"rcut": args.rcut})
    bu.set_reference_enviroments(str(ref_path))

    post_so3_matcher = StructureMatcher(
        primitive_cell=True,
        scale=False,
        attempt_supercell=True,
        allow_subset=False,
        comparator=ElementComparator(),
        ltol=args.post_so3_match_ltol,
        stol=args.post_so3_match_stol,
        angle_tol=args.post_so3_match_angle_tol,
    )

    retained_xtals = []
    retained_structures = []
    retained_source_rows = []
    post_so3_duplicate_rows = []
    all_so3_results = []
    attempted_items = []
    cursor = 0
    so3_start = time()

    while (
        len(retained_xtals) < target_post_so3_unique
        and cursor < len(candidate_pool)
    ):
        batch = candidate_pool[cursor:cursor + args.so3_fill_batch_size]
        cursor += len(batch)
        attempted_items.extend(batch)

        for row_index, _xtal, _rep, _expected, source_rows, key in batch:
            provenance.append({
                "source_row": int(row_index),
                "representative_source_row": int(row_index),
                "status": "submitted_to_so3",
                "generation_count": len(source_rows),
                "source_rows_json": json.dumps(source_rows, separators=(",", ":")),
                "dedup_key": key,
            })

        print(
            f"SO3 refill cycle: optimizing {len(batch)} candidates; "
            f"retained unique={len(retained_xtals)}/"
            f"{target_post_so3_unique}; consumed={cursor}/"
            f"{len(candidate_pool)}"
        )
        optimized_batch = bu.optimize_xtals(
            [item[1] for item in batch],
            ncpu=args.ncpu,
            early_quit=args.so3_stop_energy,
            max_initial_similarity=args.max_initial_so3_energy,
            minimizers=[
                ("Nelder-Mead", args.nm_steps),
                ("L-BFGS-B", args.lbfgs_steps),
            ],
        )
        batch_results = list(getattr(bu, "last_optimization_results", []))
        all_so3_results.extend(batch_results)
        result_by_source = {
            int(record["source_row"]): record
            for record in batch_results
            if record.get("source_row") is not None
        }

        for xtal in optimized_batch:
            source_tag = getattr(xtal, "tag", {}) or {}
            source_row = source_tag.get("source_row")
            if source_row is None:
                continue
            source_row = int(source_row)
            result = result_by_source.get(source_row, {})
            final_similarity = result.get("similarity")
            if (
                final_similarity is None
                or not math.isfinite(float(final_similarity))
                or float(final_similarity) > args.max_so3_energy
            ):
                provenance.append({
                    "source_row": source_row,
                    "representative_source_row": source_row,
                    "status": "post_so3_energy_failed",
                    "final_so3_energy": final_similarity,
                })
                continue

            structure = xtal.to_pymatgen()
            duplicate_index = _is_post_so3_duplicate(
                structure,
                retained_structures,
                post_so3_matcher,
            )
            if duplicate_index is not None:
                retained_source = retained_source_rows[duplicate_index]
                post_so3_duplicate_rows.append({
                    "source_row": source_row,
                    "retained_source_row": retained_source,
                    "final_so3_energy": float(final_similarity),
                })
                provenance.append({
                    "source_row": source_row,
                    "representative_source_row": retained_source,
                    "status": "post_so3_duplicate",
                })
                continue

            retained_xtals.append(xtal)
            retained_structures.append(structure)
            retained_source_rows.append(source_row)
            provenance.append({
                "source_row": source_row,
                "representative_source_row": source_row,
                "status": "post_so3_unique_retained",
            })
            if len(retained_xtals) >= target_post_so3_unique:
                break

        print(
            f"SO3 refill result: retained unique={len(retained_xtals)}/"
            f"{target_post_so3_unique}; post-SO3 duplicates="
            f"{len(post_so3_duplicate_rows)}"
        )

    so3_minutes = (time() - so3_start) / 60.0
    if len(retained_xtals) < target_post_so3_unique:
        print(
            "Warning: source pool exhausted before filling requested post-SO3 "
            f"unique target: retained={len(retained_xtals)}, "
            f"target={target_post_so3_unique}"
        )

    write_pre_so3_db(
        output_dir / "pre_so3.db",
        attempted_items,
        str(csv_path),
    )
    copied = _write_retained_so3_db(
        so3_work_db_path,
        so3_db_path,
        retained_source_rows,
    )
    if copied != len(retained_source_rows):
        raise RuntimeError(
            "retained SO3 DB row count mismatch: "
            f"copied={copied}, retained={len(retained_source_rows)}"
        )

    so3_results = pd.DataFrame(all_so3_results)
    if so3_results.empty:
        so3_results = pd.DataFrame(
            columns=[
                "task_id", "source_row", "similarity0",
                "similarity", "status", "error",
            ]
        )
    _strip_reason_columns(so3_results).to_csv(output_dir / "so3_results.csv", index=False)
    so3_failures = (
        so3_results[
            so3_results.get("status", pd.Series(dtype=bool)) == False
        ]
        if not so3_results.empty else so3_results
    )
    _strip_reason_columns(so3_failures).to_csv(output_dir / "so3_failures.csv", index=False)
    _strip_reason_columns(pd.DataFrame(post_so3_duplicate_rows)).to_csv(
        output_dir / "post_so3_duplicates.csv",
        index=False,
    )
    post_so3_geometry_df = write_so3_geometry_diagnostics(
        so3_db_path,
        output_dir / "post_so3_geometry.csv",
        args.tio_coordination_cutoff,
    )

    print(
        f"SO3 stage completed in {so3_minutes:.2f} min: "
        f"attempted={len(attempted_items)}, "
        f"retained post-SO3 unique={len(retained_source_rows)}, "
        f"duplicates removed={len(post_so3_duplicate_rows)}"
    )

    gulp_db_path = output_dir / "gulp_candidates.db"
    selected_count, so3_db_records = copy_selected_so3_rows(
        so3_db_path,
        gulp_db_path,
        args.max_so3_energy,
    )
    pd.DataFrame(so3_db_records).to_csv(output_dir / "so3_selection.csv", index=False)
    print(f"SO3-energy selection passed {selected_count}/{len(so3_db_records)} structures to GULP")

    gulp_minutes = 0.0
    if selected_count and not args.skip_gulp:
        gulp_start = time()
        gulp_db = database_topology(
            str(gulp_db_path),
            log_file=str(output_dir / "gulp.log"),
        )
        gulp_db.update_row_energy(
            "GULP",
            ncpu=args.ncpu,
            ff_lib=args.ff_lib,
            overwrite=False,
            calc_folder=str(output_dir / "gulp_calc"),
        )
        gulp_minutes = (time() - gulp_start) / 60.0
        matcher_kwargs = {
            "ltol": args.match_ltol,
            "stol": args.match_stol,
            "angle_tol": args.match_angle_tol,
        }
        ranked_df, ranked_all_df, post_gulp_duplicates_df, structure_match_pairs_df, gulp_failed_df = write_ranked_outputs(
            gulp_db_path,
            output_dir,
            args.ncpu,
            matcher_kwargs,
            args.match_energy_window,
            args.match_volume_tol,
            args.match_chunksize,
            {
                "search_radius": args.integrity_neighbor_search_radius,
                "max_ti_o": args.integrity_max_ti_o,
                "max_o_ti": args.integrity_max_o_ti,
                "max_angle_rms": args.integrity_max_angle_rms,
                "min_ti_fraction": args.integrity_min_ti_pass_fraction,
                "min_o_fraction": args.integrity_min_o_pass_fraction,
                **local_reference_targets,
            },
        )
        print(
            f"GULP-successful candidates: {len(ranked_all_df)}/{selected_count}; "
            f"apparent duplicates removed: {len(post_gulp_duplicates_df)}; "
            f"ranked unique candidates: {len(ranked_df)}; "
            f"integrity-valid: "
            f"{int(ranked_df['integrity_valid'].sum()) if not ranked_df.empty else 0}"
        )
        training_matches_df = pd.DataFrame()
        if args.skip_training_overlap:
            print("Skipping training-set overlap check.")
        elif args.training_db is None:
            print("Training-set overlap not requested; use --training-db PATH.")
        else:
            training_db_path = Path(args.training_db)
            if not training_db_path.is_file():
                raise FileNotFoundError(training_db_path)
            ranked_df, training_matches_df = compare_ranked_to_training_set(
                ranked_df,
                training_db_path,
                output_dir,
                args.ncpu,
                matcher_kwargs,
                args.match_chunksize,
            )
            print(
                f"Training-set overlap: {int(ranked_df['in_training_set'].sum())}/"
                f"{len(ranked_df)} unique candidates matched known structures"
            )
    else:
        ranked_df = pd.DataFrame()
        ranked_all_df = pd.DataFrame()
        post_gulp_duplicates_df = pd.DataFrame()
        structure_match_pairs_df = pd.DataFrame()
        gulp_failed_df = pd.DataFrame()
        training_matches_df = pd.DataFrame()
        _strip_reason_columns(ranked_df).to_csv(output_dir / "ranked_candidates.csv", index=False)
        _strip_reason_columns(ranked_df).to_csv(output_dir / "ranked_valid_candidates.csv", index=False)
        _strip_reason_columns(ranked_df).to_csv(output_dir / "ranked_invalid_candidates.csv", index=False)
        _strip_reason_columns(ranked_all_df).to_csv(output_dir / "ranked_all_candidates.csv", index=False)
        _strip_reason_columns(post_gulp_duplicates_df).to_csv(output_dir / "post_gulp_duplicates.csv", index=False)
        _strip_reason_columns(structure_match_pairs_df).to_csv(output_dir / "structure_match_pairs.csv", index=False)
        _strip_reason_columns(gulp_failed_df).to_csv(output_dir / "gulp_failures.csv", index=False)
        _strip_reason_columns(training_matches_df).to_csv(output_dir / "training_set_matches.csv", index=False)
        if args.skip_gulp:
            print("GULP skipped by request.")

    _strip_reason_columns(
        pd.DataFrame(provenance).sort_values("source_row")
    ).to_csv(
        output_dir / "provenance.csv", index=False
    )

    total_minutes = (time() - start) / 60.0
    summary = {
        "source_csv": str(csv_path),
        "range_semantics": "target_count_of_post_so3_structurematcher_unique_structures",
        "input_rows": len(df),
        "unique_available_before_range": unique_available,
        "unique_selection_begin": args.begin,
        "unique_selection_end": args.end,
        "unique_selected_for_so3": len(attempted_items),
        "decoded": len(decoded),
        "decode_failed": len(decode_failures),
        "unique_before_geometry_preflight": deduplicated_count,
        "geometry_preflight_failed": len(geometry_rejected),
        "pre_so3_integrity_failed": len(pre_integrity_rejected),
        "unique_before_so3": len(attempted_items),
        "duplicates_before_so3": len(duplicates),
        "post_so3_duplicates_before_gulp": len(post_so3_duplicate_rows),
        "post_so3_unique_target": target_post_so3_unique,
        "post_so3_match_scale": False,
        "post_so3_match_ltol": args.post_so3_match_ltol,
        "post_so3_match_stol": args.post_so3_match_stol,
        "post_so3_match_angle_tol": args.post_so3_match_angle_tol,
        "reference_ti_o_6th_A": local_reference_targets["reference_ti_o_6th_A"],
        "reference_o_ti_3rd_A": local_reference_targets["reference_o_ti_3rd_A"],
        "reference_tio6_angle_rms_deg": local_reference_targets[
            "reference_tio6_angle_rms_deg"
        ],
        "source_candidates_consumed_for_so3": len(attempted_items),
        "so3_valid": len(retained_source_rows),
        "max_so3_energy": args.max_so3_energy,
        "so3_stop_energy": args.so3_stop_energy,
        "max_initial_so3_energy": args.max_initial_so3_energy,
        "so3_minimizers": [["Nelder-Mead", args.nm_steps], ["L-BFGS-B", args.lbfgs_steps]],
        "passed_to_gulp": selected_count,
        "gulp_successful": len(ranked_all_df),
        "post_gulp_duplicates": len(post_gulp_duplicates_df),
        "ranked_unique_candidates": len(ranked_df),
        "integrity_valid_candidates": (
            int(ranked_df["integrity_valid"].sum())
            if not ranked_df.empty and "integrity_valid" in ranked_df.columns else 0
        ),
        "training_db": args.training_db,
        "training_overlap_matches": len(training_matches_df),
        "candidates_in_training_set": (
            int(ranked_df["in_training_set"].sum())
            if not ranked_df.empty and "in_training_set" in ranked_df.columns else 0
        ),
        "gulp_failed": len(gulp_failed_df),
        "so3_minutes": so3_minutes,
        "gulp_minutes": gulp_minutes,
        "total_minutes": total_minutes,
        "so3_rcut": args.rcut,
        "ff_lib": args.ff_lib,
        "representation_mode": {
            "discrete_coordinates": discrete,
            "discrete_cell": discrete_cell,
            "resolution": resolution,
        },
        "coordination_filtering": True,
        "catastrophic_geometry_preflight": True,
        "loose_pre_so3_integrity_screen": True,
        "pre_so3_integrity_thresholds": {
            "neighbor_search_radius_A": args.pre_integrity_neighbor_search_radius,
            "max_sixth_Ti_O_A": args.pre_integrity_max_ti_o,
            "max_third_O_Ti_A": args.pre_integrity_max_o_ti,
            "max_TiO6_angle_rms_deg": args.pre_integrity_max_angle_rms,
            "min_Ti_site_pass_fraction": args.pre_integrity_min_ti_pass_fraction,
            "min_O_site_pass_fraction": args.pre_integrity_min_o_pass_fraction,
        },
        "geometry_preflight_floors_A": {
            "Ti-Ti": args.min_ti_ti,
            "Ti-O": args.min_ti_o,
            "O-O": args.min_o_o,
        },
        "tio_coordination_diagnostics_cutoff_A": args.tio_coordination_cutoff,
        "post_gulp_integrity_ranking": True,
        "post_gulp_integrity_thresholds": {
            "neighbor_search_radius_A": args.integrity_neighbor_search_radius,
            "max_sixth_Ti_O_A": args.integrity_max_ti_o,
            "max_third_O_Ti_A": args.integrity_max_o_ti,
            "max_TiO6_angle_rms_deg": args.integrity_max_angle_rms,
            "min_Ti_site_pass_fraction": args.integrity_min_ti_pass_fraction,
            "min_O_site_pass_fraction": args.integrity_min_o_pass_fraction,
        },
        "topology_filtering": False,
        "structure_matcher_deduplication": True,
        "structure_matcher_parallel_workers": args.ncpu,
        "training_overlap_structure_matcher": True,
        "structure_matcher": {
            "ltol": args.match_ltol,
            "stol": args.match_stol,
            "angle_tol": args.match_angle_tol,
            "primitive_cell": True,
            "scale": True,
            "attempt_supercell": True,
            "comparator": "ElementComparator",
        },
        "structure_match_pair_prefilters": {
            "energy_window_eV_per_atom": args.match_energy_window,
            "relative_volume_per_atom_tolerance": args.match_volume_tol,
            "same_reduced_formula": True,
        },
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Loose pre-SO3 integrity diagnostics: {output_dir / 'pre_so3_integrity.csv'}")
    print(f"Ranked candidates: {output_dir / 'ranked_candidates.csv'}")
    print(f"Integrity-valid ranking: {output_dir / 'ranked_valid_candidates.csv'}")
    print(f"Integrity-invalid diagnostics: {output_dir / 'ranked_invalid_candidates.csv'}")
    if args.training_db is not None and not args.skip_training_overlap:
        print(f"Training-set matches: {output_dir / 'training_set_matches.csv'}")
    print(f"Candidate CIFs: {output_dir / 'candidates'}")
    print(f"Total wall time: {total_minutes:.2f} min")


if __name__ == "__main__":
    main()

