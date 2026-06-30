#!/usr/bin/env python3
"""Finalize factorized SiO2 samples with direct SO3 relaxation and GULP ranking.

Pipeline
--------
CSV -> strict PyXtal decoding -> cheap exact-representation deduplication
    -> direct element-specific SO3 optimization -> SO3-energy selection
    -> symmetry-constrained GULP relaxation -> parallel StructureMatcher deduplication
    -> candidates ranked by eV/atom.

The legacy target_coord columns are used only to identify Si and O slots in the
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
SPECIES_FROM_LEGACY_LABEL = {4: "Si", 2: "O"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-SO3 relaxation and GULP ranking for factorized SiO2 samples."
    )
    parser.add_argument("--csv", required=True, help="Factorized sampled CSV.")
    parser.add_argument("--reference-sio2", required=True, help="Reference SiO2 CIF, e.g. alpha_quartz.cif.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--rcut", type=float, default=3.0)
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
            "objective exceeds this value. Use 'inf' to disable. Default: 1000."
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
    parser.add_argument("--ff-lib", default="reaxff", help="GULP force-field library name/path.")
    parser.add_argument("--skip-gulp", action="store_true")
    parser.add_argument(
        "--dedup-decimals",
        type=int,
        default=8,
        help="Decimal precision for cheap pre-SO3 representation deduplication.",
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
                f"slot {slot}: target_coord is used only as a Si/O label and must be 4 or 2; got {legacy_label}"
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


def build_sio2(
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

    sites = {"Si": [], "O": []}
    multiplicities = {"Si": 0, "O": 0}
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

    if not sites["Si"] or not sites["O"]:
        raise ValueError("decoding lost the Si or O sublattice")
    if multiplicities["O"] != 2 * multiplicities["Si"]:
        raise ValueError(
            f"Wyckoff multiplicities are not SiO2: Si={multiplicities['Si']} O={multiplicities['O']}"
        )

    xtal = pyxtal()
    xtal.build(
        group,
        ["Si", "O"],
        [multiplicities["Si"], multiplicities["O"]],
        lattice,
        [sites["Si"], sites["O"]],
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
    if composition.get("Si", 0) <= 0 or composition.get("O", 0) != 2 * composition.get("Si", 0):
        raise ValueError(f"decoded structure is not SiO2: {composition}")
    return xtal, rep, expected


def decode_one(payload):
    row_index, row, indices, discrete, resolution, discrete_cell = payload
    try:
        xtal, rep, expected = build_sio2(row, indices, discrete, resolution, discrete_cell)
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


def write_ranked_outputs(
    gulp_db_path: Path,
    output_dir: Path,
    ncpu: int,
    matcher_kwargs: dict,
    energy_window: float,
    volume_tol: float,
    chunksize: int,
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
                successful.append((base, cif_text, structure))
            else:
                failed.append({**base, "failure_reason": "GULP result missing"})

    successful.sort(key=lambda item: item[0]["ff_energy_eV_per_atom"])
    all_ranked_df = pd.DataFrame([
        {"all_rank": rank, **record}
        for rank, (record, _cif, _structure) in enumerate(successful, start=1)
    ])
    all_ranked_df.to_csv(output_dir / "ranked_all_candidates.csv", index=False)

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
        member_indices.sort(key=lambda idx: (
            successful[idx][0]["ff_energy_eV_per_atom"],
            successful[idx][0]["final_so3_energy"],
            successful[idx][0]["representative_source_row"],
        ))
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

    retained.sort(key=lambda item: item[0]["ff_energy_eV_per_atom"])
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
    ranked_df.to_csv(output_dir / "ranked_candidates.csv", index=False)
    duplicate_df.to_csv(output_dir / "post_gulp_duplicates.csv", index=False)
    match_df.to_csv(output_dir / "structure_match_pairs.csv", index=False)
    failed_df.to_csv(output_dir / "gulp_failures.csv", index=False)
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
        match_df.to_csv(output_dir / "training_set_matches.csv", index=False)
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
        match_df.to_csv(output_dir / "training_set_matches.csv", index=False)
        annotated = ranked_df.copy()
        annotated["in_training_set"] = False
        annotated["training_match_count"] = 0
        annotated["training_db_rows_json"] = "[]"
        annotated["best_training_db_row"] = None
        annotated["best_training_rms"] = None
        annotated["best_training_max_dist"] = None
        annotated.to_csv(output_dir / "ranked_candidates.csv", index=False)
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
    annotated.to_csv(output_dir / "ranked_candidates.csv", index=False)
    match_df.to_csv(output_dir / "training_set_matches.csv", index=False)
    return annotated, match_df


def main() -> None:
    args = parse_args()
    start = time()

    if args.ncpu < 1:
        raise ValueError("--ncpu must be at least 1")
    if args.begin < 0 or (args.end != -1 and args.end <= args.begin):
        raise ValueError("invalid --begin/--end range")
    if args.rcut <= 0:
        raise ValueError("--rcut must be positive")
    if args.dedup_decimals < 3:
        raise ValueError("--dedup-decimals must be at least 3")
    if args.match_ltol <= 0 or args.match_stol <= 0 or args.match_angle_tol <= 0:
        raise ValueError("StructureMatcher tolerances must be positive")
    if args.match_chunksize < 1:
        raise ValueError("--match-chunksize must be at least 1")

    csv_path = Path(args.csv)
    ref_path = Path(args.reference_sio2)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)

    output_dir = Path(args.output_dir or csv_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- SiO2 final relaxation ---")
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"CPU workers: {args.ncpu}")
    print(f"SO3 reference: {ref_path}")
    print(f"SO3 cutoff: {args.rcut}")
    print(f"Maximum SO3 energy passed to GULP: {args.max_so3_energy}")
    print(f"SO3 stage-stop energy: {args.so3_stop_energy}")
    print(f"Initial SO3 pre-screen: {args.max_initial_so3_energy}")
    print(f"SO3 schedule: Nelder-Mead {args.nm_steps} + L-BFGS-B {args.lbfgs_steps}")
    print(f"GULP force field: {args.ff_lib}")
    print(
        "Post-GULP direct structure matching: "
        f"ltol={args.match_ltol}, stol={args.match_stol}, "
        f"angle_tol={args.match_angle_tol}, workers={args.ncpu}"
    )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("input CSV is empty")
    indices = discover_site_indices(df)
    discrete, resolution, discrete_cell = infer_representation_mode(df)
    stop = None if args.end == -1 else args.end
    selected_df = df.iloc[args.begin:stop]
    if selected_df.empty:
        raise ValueError("selected row range is empty")

    payloads = [
        (int(idx), row.to_dict(), indices, discrete, resolution, discrete_cell)
        for idx, row in selected_df.iterrows()
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
    print(
        f"Decoded {len(decoded)}/{len(payloads)}; unique before SO3: {len(unique)}; "
        f"obvious duplicates removed: {len(duplicates)}"
    )
    write_pre_so3_db(output_dir / "pre_so3.db", unique, str(csv_path))

    provenance = []
    for failure in decode_failures:
        provenance.append({
            "source_row": int(failure["source_row"]),
            "representative_source_row": None,
            "status": "decode_failed",
            "failure_reason": failure["failure_reason"],
        })
    provenance.extend(duplicates)
    for row_index, _xtal, _rep, _expected, source_rows, key in unique:
        provenance.append({
            "source_row": int(row_index),
            "representative_source_row": int(row_index),
            "status": "submitted_to_so3",
            "generation_count": len(source_rows),
            "source_rows_json": json.dumps(source_rows, separators=(",", ":")),
            "dedup_key": key,
        })

    so3_prefix = output_dir / "so3"
    so3_db_path = output_dir / "so3-0.db"
    for suffix in ("-0.db", "-0.log"):
        path = Path(str(so3_prefix) + suffix)
        if path.exists():
            path.unlink()

    bu = builder(["Si", "O"], [1, 2], rank=0, prefix=str(so3_prefix))
    bu.set_descriptor_calculator(mykwargs={"rcut": args.rcut})
    bu.set_reference_enviroments(str(ref_path))

    xtals = [item[1] for item in unique]
    print(f"Direct SO3 optimization: {len(xtals)} unique structures")
    so3_start = time()
    stage_stop = args.so3_stop_energy
    optimized = bu.optimize_xtals(
        xtals,
        ncpu=args.ncpu,
        early_quit=stage_stop,
        max_initial_similarity=args.max_initial_so3_energy,
        minimizers=[
            ("Nelder-Mead", args.nm_steps),
            ("L-BFGS-B", args.lbfgs_steps),
        ],
    )
    so3_minutes = (time() - so3_start) / 60.0
    print(f"SO3-valid structures: {len(optimized)}/{len(unique)} in {so3_minutes:.2f} min")

    so3_results = pd.DataFrame(getattr(bu, "last_optimization_results", []))
    if so3_results.empty and args.ncpu == 1:
        # Serial builder versions may not expose detailed results; the DB remains authoritative.
        so3_results = pd.DataFrame(columns=["task_id", "source_row", "similarity0", "similarity", "status", "error"])
    so3_results.to_csv(output_dir / "so3_results.csv", index=False)
    so3_failures = so3_results[so3_results.get("status", pd.Series(dtype=bool)) == False] if not so3_results.empty else so3_results
    so3_failures.to_csv(output_dir / "so3_failures.csv", index=False)

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
        )
        print(
            f"GULP-successful candidates: {len(ranked_all_df)}/{selected_count}; "
            f"apparent duplicates removed: {len(post_gulp_duplicates_df)}; "
            f"ranked unique candidates: {len(ranked_df)}"
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
        ranked_df.to_csv(output_dir / "ranked_candidates.csv", index=False)
        ranked_all_df.to_csv(output_dir / "ranked_all_candidates.csv", index=False)
        post_gulp_duplicates_df.to_csv(output_dir / "post_gulp_duplicates.csv", index=False)
        structure_match_pairs_df.to_csv(output_dir / "structure_match_pairs.csv", index=False)
        gulp_failed_df.to_csv(output_dir / "gulp_failures.csv", index=False)
        training_matches_df.to_csv(output_dir / "training_set_matches.csv", index=False)
        if args.skip_gulp:
            print("GULP skipped by request.")

    pd.DataFrame(provenance).sort_values("source_row").to_csv(
        output_dir / "provenance.csv", index=False
    )

    total_minutes = (time() - start) / 60.0
    summary = {
        "source_csv": str(csv_path),
        "input_rows": len(payloads),
        "decoded": len(decoded),
        "decode_failed": len(decode_failures),
        "unique_before_so3": len(unique),
        "duplicates_before_so3": len(duplicates),
        "so3_valid": len(optimized),
        "max_so3_energy": args.max_so3_energy,
        "so3_stop_energy": args.so3_stop_energy,
        "max_initial_so3_energy": args.max_initial_so3_energy,
        "so3_minimizers": [["Nelder-Mead", args.nm_steps], ["L-BFGS-B", args.lbfgs_steps]],
        "passed_to_gulp": selected_count,
        "gulp_successful": len(ranked_all_df),
        "post_gulp_duplicates": len(post_gulp_duplicates_df),
        "ranked_unique_candidates": len(ranked_df),
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
        "coordination_filtering": False,
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

    print(f"Ranked candidates: {output_dir / 'ranked_candidates.csv'}")
    if args.training_db is not None and not args.skip_training_overlap:
        print(f"Training-set matches: {output_dir / 'training_set_matches.csv'}")
    print(f"Candidate CIFs: {output_dir / 'candidates'}")
    print(f"Total wall time: {total_minutes:.2f} min")


if __name__ == "__main__":
    main()

