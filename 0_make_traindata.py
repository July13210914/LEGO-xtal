#!/usr/bin/env python3
"""Generate LEGO-Xtal training rows for a configurable binary chemistry.

The first species is the building-center block and the second species is the
conditioned sublattice. ``target_coordination`` remains the persistent role /
SO(3)-reference label consumed by the finalized factorized workflow; it is not
recomputed as a hard coordination-number filter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from itertools import combinations
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyxtal.db import database_topology
from pyxtal.symmetry import Group
from lego.builder import builder
from pyxtal.util import new_struc_wo_energy


def build_output_filename(output_dir, tag, discrete, discrete_cell):
    suffix = "-discell.csv" if discrete and discrete_cell else "-dis.csv" if discrete else ".csv"
    return os.path.join(output_dir, f"{tag}{suffix}")


def load_coord_ref_config(inline_json: str | None, json_file: str | None):
    if json_file:
        path = Path(json_file)
        if not path.is_file():
            raise FileNotFoundError(f"Coordination-reference file not found: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            config = json.loads(inline_json or "")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --coord-ref-dict JSON: {exc}") from exc

    if not isinstance(config, dict) or len(config) != 2:
        raise ValueError("This factorized workflow requires exactly two species entries.")

    normalized = {}
    used_labels = set()
    for species, entry in config.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Configuration for {species} must be an object.")
        missing = {"neighbor_species", "coordination", "reference"} - set(entry)
        if missing:
            raise ValueError(f"Configuration for {species} is missing {sorted(missing)}.")
        coordination = int(entry["coordination"])
        if coordination <= 0 or coordination in used_labels:
            raise ValueError("Coordination/role labels must be distinct positive integers.")
        used_labels.add(coordination)
        reference = Path(str(entry["reference"])).expanduser()
        if not reference.is_file():
            raise FileNotFoundError(f"Reference CIF for {species} not found: {reference}")
        normalized[str(species)] = {
            "neighbor_species": str(entry["neighbor_species"]),
            "coordination": coordination,
            "reference": str(reference.resolve()),
        }

    species = list(normalized)
    for symbol, entry in normalized.items():
        if entry["neighbor_species"] not in normalized:
            raise ValueError(f"Unknown neighbor species for {symbol}: {entry['neighbor_species']}")
    return normalized, species


def parse_composition(value: str, species: list[str]) -> list[int]:
    try:
        parts = [int(x.strip()) for x in value.split(",")]
    except ValueError as exc:
        raise ValueError("--composition must be comma-separated positive integers.") from exc
    if len(parts) != len(species) or any(x <= 0 for x in parts):
        raise ValueError(f"--composition must contain {len(species)} positive integers.")
    return parts


def set_site_target(site, value: int):
    if hasattr(site, "set_target_coordination"):
        site.set_target_coordination(int(value))
    else:
        if not hasattr(site, "property") or site.property is None:
            site.property = {}
        site.property["target_coordination"] = int(value)
        site.target_coordination = int(value)


def assign_configured_templates(xtal, config):
    for site in xtal.atom_sites:
        species = str(site.specie)
        if species not in config:
            raise ValueError(f"Unsupported species {species!r}; expected {list(config)}.")
        set_site_target(site, config[species]["coordination"])


def make_builder(config, species, composition, rcut):
    """Create the binary builder in direct element-specific SO3 mode.

    The configured coordination values remain attached to atom sites for the
    factorized CSV block labels, but they do not route the SO3 objective and do
    not activate builder.check_target_coordination().  A shared reference CIF
    supplies one SO3 descriptor per element through the builder's established
    multi-element pathway.
    """
    bu = builder(species, composition, verbose=False)
    bu.set_descriptor_calculator(mykwargs={"rcut": float(rcut)})

    reference_paths = {
        os.path.realpath(config[symbol]["reference"])
        for symbol in species
    }
    if len(reference_paths) != 1:
        raise ValueError(
            "The current multi-element builder accepts one shared reference "
            "structure containing all configured species. Received distinct "
            f"reference paths: {sorted(reference_paths)}"
        )

    reference_cif = next(iter(reference_paths))
    bu.set_reference_enviroments(reference_cif)
    return bu




def _angle_deg(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1.0e-12 or n2 <= 1.0e-12:
        return np.nan
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _ideal_angle_rms(vectors, coordination):
    """Return RMS angular deviation for supported ideal polyhedra.

    CN=4 is compared with a tetrahedron (six 109.471-degree angles).
    CN=6 is compared with an octahedron (twelve 90-degree and three
    180-degree angles). Other coordination labels receive no angular test.
    """
    if coordination not in (4, 6):
        return np.nan
    angles = np.sort(np.asarray([
        _angle_deg(vectors[i], vectors[j])
        for i in range(coordination)
        for j in range(i + 1, coordination)
    ], dtype=float))
    if not np.all(np.isfinite(angles)):
        return np.nan
    if coordination == 4:
        ideal = np.full(6, 109.47122063449069, dtype=float)
    else:
        ideal = np.asarray([90.0] * 12 + [180.0] * 3, dtype=float)
    return float(np.sqrt(np.mean((angles - ideal) ** 2)))


def _periodic_species_neighbors(structure, center_index, neighbor_species, radius):
    center = structure[center_index]
    neighbors = []
    for neighbor in structure.get_neighbors(
        center, radius, include_index=True, include_image=True
    ):
        if str(neighbor.specie.symbol) != str(neighbor_species):
            continue
        neighbors.append({
            "distance": float(neighbor.nn_distance),
            "vector": np.asarray(neighbor.coords - center.coords, dtype=float),
        })
    neighbors.sort(key=lambda item: item["distance"])
    return neighbors


def evaluate_local_integrity(
    xtal,
    config,
    species,
    similarity,
    integrity,
    source_index,
    stage,
):
    """Evaluate periodic first-shell integrity after SO3 optimization."""
    structure = xtal.to_pymatgen()
    symbols = [str(site.specie.symbol) for site in structure]
    report = {
        "source_index": int(source_index),
        "stage": str(stage),
        "space_group": int(xtal.group.number),
        "natoms": int(len(structure)),
        "similarity": float(similarity) if similarity is not None else np.nan,
        "accepted": False,
        "rejection_reasons": "",
    }
    reasons = []

    first, second = species
    first_count = symbols.count(first)
    report["similarity_per_building_center"] = (
        float(similarity) / first_count
        if similarity is not None and first_count > 0 else np.nan
    )

    for species_index, symbol in enumerate(species):
        entry = config[symbol]
        neighbor_symbol = entry["neighbor_species"]
        coordination = int(entry["coordination"])
        site_ids = [i for i, value in enumerate(symbols) if value == symbol]
        site_records = []

        for site_id in site_ids:
            neighbors = _periodic_species_neighbors(
                structure,
                site_id,
                neighbor_symbol,
                integrity["neighbor_search_radius"],
            )
            if len(neighbors) < coordination:
                site_records.append({
                    "pass": False,
                    "nth_distance": np.nan,
                    "shell_gap": np.nan,
                    "angle_rms": np.nan,
                })
                continue

            shell = neighbors[:coordination]
            nth_distance = float(shell[-1]["distance"])
            shell_gap = (
                float(neighbors[coordination]["distance"] - nth_distance)
                if len(neighbors) > coordination else np.nan
            )
            angle_rms = _ideal_angle_rms(
                [item["vector"] for item in shell], coordination
            )

            max_distance = (
                integrity["center_max_neighbor_distance"]
                if species_index == 0
                else integrity["neighbor_max_center_distance"]
            )
            min_gap = (
                integrity["center_min_shell_gap"]
                if species_index == 0
                else integrity["neighbor_min_shell_gap"]
            )
            max_angle_rms = (
                integrity["center_max_angle_rms"]
                if species_index == 0
                else integrity["neighbor_max_angle_rms"]
            )

            passed = nth_distance <= max_distance
            if min_gap > 0:
                passed = passed and np.isfinite(shell_gap) and shell_gap >= min_gap
            if np.isfinite(max_angle_rms) and coordination in (4, 6):
                passed = passed and np.isfinite(angle_rms) and angle_rms <= max_angle_rms

            site_records.append({
                "pass": bool(passed),
                "nth_distance": nth_distance,
                "shell_gap": shell_gap,
                "angle_rms": angle_rms,
            })

        prefix = "building_center" if species_index == 0 else "conditioned_sublattice"
        pass_fraction = (
            float(np.mean([record["pass"] for record in site_records]))
            if site_records else 0.0
        )
        nth_values = np.asarray(
            [record["nth_distance"] for record in site_records], dtype=float
        )
        gap_values = np.asarray(
            [record["shell_gap"] for record in site_records], dtype=float
        )
        angle_values = np.asarray(
            [record["angle_rms"] for record in site_records], dtype=float
        )
        report[f"{prefix}_species"] = symbol
        report[f"{prefix}_target_coordination"] = coordination
        report[f"{prefix}_site_count"] = len(site_records)
        report[f"{prefix}_pass_fraction"] = pass_fraction
        report[f"{prefix}_nth_distance_max"] = (
            float(np.nanmax(nth_values)) if np.any(np.isfinite(nth_values)) else np.nan
        )
        report[f"{prefix}_nth_distance_mean"] = (
            float(np.nanmean(nth_values)) if np.any(np.isfinite(nth_values)) else np.nan
        )
        report[f"{prefix}_shell_gap_min"] = (
            float(np.nanmin(gap_values)) if np.any(np.isfinite(gap_values)) else np.nan
        )
        report[f"{prefix}_shell_gap_mean"] = (
            float(np.nanmean(gap_values)) if np.any(np.isfinite(gap_values)) else np.nan
        )
        report[f"{prefix}_angle_rms_max"] = (
            float(np.nanmax(angle_values)) if np.any(np.isfinite(angle_values)) else np.nan
        )
        report[f"{prefix}_angle_rms_mean"] = (
            float(np.nanmean(angle_values)) if np.any(np.isfinite(angle_values)) else np.nan
        )

        required_fraction = (
            integrity["min_center_pass_fraction"]
            if species_index == 0
            else integrity["min_neighbor_pass_fraction"]
        )
        if pass_fraction + 1.0e-12 < required_fraction:
            reasons.append(
                f"{prefix}_pass_fraction={pass_fraction:.3f}<"
                f"{required_fraction:.3f}"
            )

    max_so3 = integrity["max_so3_per_center"]
    if (
        np.isfinite(max_so3)
        and np.isfinite(report["similarity_per_building_center"])
        and report["similarity_per_building_center"] > max_so3
    ):
        reasons.append(
            "similarity_per_building_center="
            f"{report['similarity_per_building_center']:.6g}>{max_so3:.6g}"
        )

    report["accepted"] = len(reasons) == 0
    report["rejection_reasons"] = ";".join(reasons)
    return bool(report["accepted"]), report



def coordination_label(value):
    """Return the integer coordination label from PyXtal scalar/tuple storage."""
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("Empty target_coordination tuple/list.")
        # PyXtal may store (neighbor_species, coordination) or equivalent.
        numeric = [item for item in value if isinstance(item, (int, float, np.integer, np.floating))]
        if not numeric:
            raise ValueError(f"No numeric coordination label in {value!r}.")
        value = numeric[-1]
    return int(value)

def target_coordination_vector(xtal, n_wp, missing_value=0):
    sites = getattr(xtal, "atom_sites", [])
    if len(sites) > n_wp:
        raise ValueError(f"Structure has {len(sites)} atom sites, exceeding N_wp={n_wp}.")
    values = []
    for index, site in enumerate(sites):
        value = getattr(site, "target_coordination", None)
        if value is None:
            value = (getattr(site, "property", {}) or {}).get("target_coordination")
        if value is None:
            raise ValueError(
                f"Missing target_coordination for site {index} "
                f"({site.specie}, {site.wp.get_label()})."
            )
        label = coordination_label(value)
        if label <= 0:
            raise ValueError(
                f"Invalid target_coordination={value!r} for site {index} "
                f"({site.specie}, {site.wp.get_label()})."
            )
        values.append(label)
    values.extend([int(missing_value)] * (n_wp - len(values)))
    return np.asarray(values, dtype=int)


def _cell_matrix_from_representation(representation):
    """Return a row-vector cell matrix from one continuous LEGO row."""
    a, b, c, alpha, beta, gamma = map(float, representation[1:7])
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)
    if abs(sg) < 1.0e-12:
        raise ValueError("Degenerate gamma in tabular representation.")
    y3 = c * (ca - cb * cg) / sg
    z3_sq = c * c - (c * cb) ** 2 - y3 ** 2
    if z3_sq <= 1.0e-12:
        raise ValueError("Degenerate cell metric in tabular representation.")
    return np.asarray([
        [a, 0.0, 0.0],
        [b * cg, b * sg, 0.0],
        [c * cb, y3, np.sqrt(z3_sq)],
    ], dtype=float)


def _deduplicate_fractional(frac, tol=1.0e-6):
    frac = np.asarray(frac, dtype=float).reshape(-1, 3) % 1.0
    unique = []
    for point in frac:
        if not any(
            np.linalg.norm((point - other) - np.round(point - other)) <= tol
            for other in unique
        ):
            unique.append(point)
    return np.asarray(unique, dtype=float).reshape(-1, 3)


def _expand_representation_orbits(representation, n_wp):
    """Expand each occupied independent Wyckoff slot without assigning species."""
    spg = int(round(float(representation[0])))
    group = Group(spg)
    orbits = []
    for slot in range(n_wp):
        base = 7 + 4 * slot
        wp_index = int(round(float(representation[base])))
        if wp_index < 0:
            continue
        if wp_index >= len(group):
            raise ValueError(f"Invalid Wyckoff index {wp_index} for space group {spg}.")
        generator = np.asarray(representation[base + 1:base + 4], dtype=float)
        wp = group[wp_index]
        frac = np.asarray([op.operate(generator) for op in wp.ops], dtype=float) % 1.0
        frac = _deduplicate_fractional(frac)
        if len(frac) != int(wp.multiplicity):
            raise ValueError(
                f"Slot {slot} ({wp.get_label()}) expands to {len(frac)} atoms; "
                f"expected {wp.multiplicity}."
            )
        orbits.append((slot, frac))
    return orbits


def _species_distance_spectrum(frac_by_species, cell):
    """Periodic species-resolved distance spectrum invariant to origin/setting."""
    shifts = np.asarray(
        [[i, j, k] for i in (-1, 0, 1)
         for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    zero_shift = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
    names = list(frac_by_species)
    spectrum = {}
    for i, name_a in enumerate(names):
        a = np.asarray(frac_by_species[name_a], dtype=float).reshape(-1, 3)
        for j in range(i, len(names)):
            name_b = names[j]
            b = np.asarray(frac_by_species[name_b], dtype=float).reshape(-1, 3)
            delta = a[:, None, None, :] - b[None, :, None, :] + shifts[None, None, :, :]
            cart = np.einsum("...i,ij->...j", delta, cell)
            dist = np.linalg.norm(cart, axis=-1)
            if name_a == name_b:
                ids = np.arange(len(a))
                dist[ids, ids, zero_shift] = np.inf
            values = np.sort(dist[np.isfinite(dist)].reshape(-1))
            spectrum[(name_a, name_b)] = values
    return spectrum


def _source_distance_spectrum(xtal, species):
    atoms = xtal.to_ase(resort=False, add_vaccum=False)
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    frac_by_species = {}
    for symbol in species:
        frac_by_species[symbol] = scaled[symbols == symbol]
        if len(frac_by_species[symbol]) == 0:
            raise ValueError(f"Source structure contains no atoms of species {symbol}.")
    return _species_distance_spectrum(frac_by_species, cell), {
        symbol: len(frac_by_species[symbol]) for symbol in species
    }


def _spectra_match(candidate, reference, atol=2.0e-4, rtol=2.0e-5):
    if candidate.keys() != reference.keys():
        return False
    for key in reference:
        a, b = candidate[key], reference[key]
        if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol):
            return False
    return True


def _labels_for_valid_representation(
    representation, xtal, n_wp, species, config,
):
    """Find the site labels that make a row exactly equivalent to ``xtal``.

    ``get_tabular_representations`` may reorder independent sites.  More
    importantly, some independently selected equivalent generators are not
    mutually compatible because they correspond to different common origin or
    setting choices.  Enumerate the binary species assignment allowed by the
    atom counts and accept only a species-resolved periodic distance spectrum
    identical to the source structure.
    """
    orbits = _expand_representation_orbits(representation, n_wp)
    if not orbits:
        return None
    cell = _cell_matrix_from_representation(representation)
    reference, counts = _source_distance_spectrum(xtal, species)
    first, second = species
    required_first = counts[first]
    all_ids = range(len(orbits))

    for n_first_sites in range(1, len(orbits)):
        for chosen in combinations(all_ids, n_first_sites):
            chosen = set(chosen)
            if sum(len(orbits[i][1]) for i in chosen) != required_first:
                continue
            frac_first = np.concatenate([orbits[i][1] for i in chosen], axis=0)
            frac_second = np.concatenate(
                [orbits[i][1] for i in all_ids if i not in chosen], axis=0
            )
            if len(frac_second) != counts[second]:
                continue
            candidate = _species_distance_spectrum(
                {first: frac_first, second: frac_second}, cell
            )
            if not _spectra_match(candidate, reference):
                continue
            labels = np.zeros(n_wp, dtype=int)
            for orbit_id, (slot, _) in enumerate(orbits):
                symbol = first if orbit_id in chosen else second
                labels[slot] = int(config[symbol]["coordination"])
            return labels
    return None


def append_target_coordination(
    representations, xtal, n_wp, species, config, diagnostics=None,
):
    """Append representation-specific role labels and reject invalid augmentation."""
    expected_width = 7 + 4 * n_wp
    output = []
    rejected = 0
    for row_index, representation in enumerate(representations):
        representation = np.asarray(representation)
        if representation.ndim != 1 or len(representation) != expected_width:
            raise ValueError(
                f"Representation {row_index} has shape {representation.shape}; "
                f"expected width {expected_width}."
            )
        labels = _labels_for_valid_representation(
            representation, xtal, n_wp, species, config
        )
        if labels is None:
            rejected += 1
            continue
        output.append(np.concatenate((representation, labels)))
    if diagnostics is not None:
        diagnostics["augmentation_candidates"] = diagnostics.get(
            "augmentation_candidates", 0
        ) + len(representations)
        diagnostics["augmentation_rejected"] = diagnostics.get(
            "augmentation_rejected", 0
        ) + rejected
    return output


def make_csv(total_reps, include_energy, include_label, discrete, discrete_cell, n_wp, filename):
    total_reps = np.asarray(total_reps)
    if total_reps.ndim != 2:
        raise ValueError(f"Expected 2D representation array, got {total_reps.shape}")
    columns = ["spg", "a", "b", "c", "alpha", "beta", "gamma"]
    float_cols = set() if discrete_cell else set(range(1, 7))
    for i in range(n_wp):
        base = 7 + 4 * i
        columns.extend([f"wp{i}", f"x{i}", f"y{i}", f"z{i}"])
        if not discrete:
            float_cols.update([base + 1, base + 2, base + 3])
    columns.extend([f"target_coord{i}" for i in range(n_wp)])
    if include_energy:
        columns.append("energy")
        float_cols.add(len(columns) - 1)
    if include_label:
        columns.append("label")
    if total_reps.shape[1] != len(columns):
        raise ValueError(f"CSV width mismatch: {total_reps.shape[1]} vs {len(columns)}")
    data = {
        name: total_reps[:, i].astype(float if i in float_cols else int)
        for i, name in enumerate(columns)
    }
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(filename, index=False)
    print(f"Saved {len(total_reps)} representations to {filename}")


def process_one_xtal(item, params, base_seed):
    source_index, xtal = item
    random.seed(base_seed + source_index)
    np.random.seed(base_seed + source_index)
    try:
        reps, reports = get_reps_from_xtal(xtal, params, source_index)
        return source_index, reps, reports, None
    except Exception as exc:
        return source_index, [], [], f"{type(exc).__name__}: {exc}"


def get_reps_from_xtal(xtal, params, source_index):
    (
        max_dof, n_atoms_min, n_atoms_max, max_energy, min_spg, n_wp,
        max_per_structure, include_energy, discrete, discrete_cell,
        discrete_resolution, subgroup_eps, config, species, composition, rcut,
        integrity,
    ) = params

    assign_configured_templates(xtal, config)
    bu = make_builder(config, species, composition, rcut)
    atom_count = sum(xtal.numIons)
    ff_energy = getattr(xtal, "ff_energy", None)
    filter_by_energy = np.isfinite(max_energy)
    if not (
        xtal.dof <= max_dof
        and n_atoms_min <= atom_count <= n_atoms_max
        and (not filter_by_energy or (ff_energy is not None and ff_energy <= max_energy))
        and xtal.group.number >= min_spg
        and len(xtal.atom_sites) <= n_wp
    ):
        return [], []
    target_coordination_vector(xtal, n_wp)
    current_energy = ff_energy if include_energy else None

    xtal_opt, sim_parent, _ = bu.optimize_xtal(xtal, add_db=False)
    if xtal_opt is None or not xtal_opt.check_validity(bu.criteria):
        return [], []

    integrity_reports = []
    if integrity["enabled"]:
        accepted, report = evaluate_local_integrity(
            xtal_opt, config, species, sim_parent, integrity,
            source_index=source_index, stage="parent",
        )
        integrity_reports.append(report)
        if not accepted:
            print(
                f"Source {source_index} rejected after SO3: "
                f"{report['rejection_reasons']}"
            )
            return [], integrity_reports

    reps = []
    augmentation_diagnostics = {}
    n_wps = len(xtal_opt.atom_sites)
    n_max_initial = max(1, int(0.6 * max_per_structure * np.ceil(n_wps / n_wp)))
    initial = xtal_opt.get_tabular_representations(
        N_wp=n_wp, N_max=n_max_initial, discrete=discrete,
        discrete_cell=discrete_cell, N_grids=discrete_resolution,
    ) or []
    initial = append_target_coordination(
        initial, xtal_opt, n_wp, species, config, augmentation_diagnostics
    )
    if include_energy and current_energy is not None:
        initial = [np.append(rep, current_energy) for rep in initial]
    reps.extend(initial)

    max_cell_factor = max(n_atoms_max / sum(xtal_opt.numIons), 1.0)
    trial_cache = [xtal_opt]
    for group_type in ("t", "k"):
        for _ in range(20):
            if len(reps) >= max_per_structure:
                return reps[:max_per_structure], integrity_reports
            xtal_sub = xtal_opt.subgroup_once(
                eps=subgroup_eps, group_type=group_type,
                max_cell=max_cell_factor, mut_lat=False,
            )
            if xtal_sub is None:
                xtal0 = xtal_opt.subgroup_once(group_type="t")
                if xtal0 is not None:
                    xtal_sub = xtal0.subgroup_once(
                        eps=subgroup_eps, group_type="t",
                        max_cell=max_cell_factor, mut_lat=False,
                    )
            if xtal_sub is None:
                continue
            para = xtal_sub.lattice.get_para(degree=True)
            if not (
                xtal_sub.get_dof() <= max_dof
                and len(xtal_sub.atom_sites) <= n_wp
                and max(para[:3]) < 50
                and min(para[3:]) > 30
                and max(para[3:]) < 150
            ):
                continue
            target_coordination_vector(xtal_sub, n_wp)
            if not new_struc_wo_energy(xtal_sub, trial_cache, 0.025, 0.025, 1.0):
                continue
            try:
                xtal_sub_opt, sim_sub, _ = bu.optimize_xtal(xtal_sub, add_db=False)
            except Exception:
                continue
            if xtal_sub_opt is None or not xtal_sub_opt.check_validity(bu.criteria):
                continue
            if integrity["enabled"]:
                accepted, report = evaluate_local_integrity(
                    xtal_sub_opt, config, species, sim_sub, integrity,
                    source_index=source_index, stage=f"subgroup_{group_type}",
                )
                integrity_reports.append(report)
                if not accepted:
                    continue
            trial_cache.append(xtal_sub_opt)
            n_max_sub = max(
                1, int(0.2 * max_per_structure * np.ceil(len(xtal_sub_opt.atom_sites) / n_wp))
            )
            sub = xtal_sub_opt.get_tabular_representations(
                N_wp=n_wp, N_max=n_max_sub, discrete=discrete,
                discrete_cell=discrete_cell, N_grids=discrete_resolution,
            ) or []
            sub = append_target_coordination(
                sub, xtal_sub_opt, n_wp, species, config,
                augmentation_diagnostics,
            )
            if include_energy and current_energy is not None:
                sub = [np.append(rep, current_energy) for rep in sub]
            reps.extend(sub)
    if augmentation_diagnostics.get("augmentation_rejected", 0):
        print(
            "Augmentation validation: rejected "
            f"{augmentation_diagnostics['augmentation_rejected']}/"
            f"{augmentation_diagnostics['augmentation_candidates']} "
            "incompatible tabular representations."
        )
    return reps[:max_per_structure], integrity_reports


def parse_args():
    parser = argparse.ArgumentParser(description="Generate factorized LEGO-Xtal training CSV data.")
    parser.add_argument("--database", default="data/source/tio2.db")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default="data/train")
    parser.add_argument("--max_atoms", type=int, default=500)
    parser.add_argument("--min_spg", type=int, default=0)
    parser.add_argument("--max_dof", type=int, default=24)
    parser.add_argument("--max_wp", type=int, default=8)
    parser.add_argument("--max_energy", type=float, default=float("inf"))
    parser.add_argument("--max_per_struc", type=int, default=500)
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--discrete", type=int, metavar="N_GRIDS")
    parser.add_argument("--discrete_cell", action="store_true")
    parser.add_argument("--rcut", type=float, default=2.4)
    parser.add_argument(
        "--no-integrity-filter", action="store_true",
        help="Disable the post-SO3 local coordination/polyhedron filter.",
    )
    parser.add_argument("--integrity-neighbor-radius", type=float, default=6.0)
    parser.add_argument("--center-max-neighbor-distance", type=float, default=2.6)
    parser.add_argument("--center-min-shell-gap", type=float, default=0.15)
    parser.add_argument("--center-max-angle-rms", type=float, default=20.0)
    parser.add_argument("--neighbor-max-center-distance", type=float, default=2.6)
    parser.add_argument("--neighbor-min-shell-gap", type=float, default=0.10)
    parser.add_argument(
        "--neighbor-max-angle-rms", type=float, default=float("inf"),
        help="Optional tetrahedral/octahedral angular RMS limit for species 2.",
    )
    parser.add_argument("--min-center-pass-fraction", type=float, default=1.0)
    parser.add_argument("--min-neighbor-pass-fraction", type=float, default=1.0)
    parser.add_argument(
        "--max-so3-per-center", type=float, default=5.0,
        help="Maximum multiplicity-weighted LEGO SO3 objective per expanded building-center atom.",
    )
    parser.add_argument(
        "--integrity-report", default=None,
        help="CSV audit path. Default: <output-dir>/<tag>-integrity.csv",
    )
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coord-ref-dict")
    group.add_argument("--coord-ref-file")
    parser.add_argument(
        "--composition", default="1,2",
        help="Stoichiometric coefficients in coord-reference species order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config, species = load_coord_ref_config(args.coord_ref_dict, args.coord_ref_file)
    composition = parse_composition(args.composition, species)
    if not os.path.isfile(args.database):
        raise FileNotFoundError(args.database)
    for name, value in (("max_atoms", args.max_atoms), ("max_wp", args.max_wp),
                        ("max_per_struc", args.max_per_struc), ("ncpu", args.ncpu),
                        ("chunksize", args.chunksize)):
        if value < 1:
            raise ValueError(f"--{name} must be positive.")
    if args.discrete is not None and args.discrete < 2:
        raise ValueError("--discrete must be at least 2.")
    if args.discrete_cell and args.discrete is None:
        args.discrete_cell = False


    for name, value in (
        ("integrity_neighbor_radius", args.integrity_neighbor_radius),
        ("center_max_neighbor_distance", args.center_max_neighbor_distance),
        ("neighbor_max_center_distance", args.neighbor_max_center_distance),
    ):
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name, value in (
        ("center_min_shell_gap", args.center_min_shell_gap),
        ("neighbor_min_shell_gap", args.neighbor_min_shell_gap),
    ):
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    for name, value in (
        ("min_center_pass_fraction", args.min_center_pass_fraction),
        ("min_neighbor_pass_fraction", args.min_neighbor_pass_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, 1].")

    integrity = {
        "enabled": not args.no_integrity_filter,
        "neighbor_search_radius": float(args.integrity_neighbor_radius),
        "center_max_neighbor_distance": float(args.center_max_neighbor_distance),
        "center_min_shell_gap": float(args.center_min_shell_gap),
        "center_max_angle_rms": float(args.center_max_angle_rms),
        "neighbor_max_center_distance": float(args.neighbor_max_center_distance),
        "neighbor_min_shell_gap": float(args.neighbor_min_shell_gap),
        "neighbor_max_angle_rms": float(args.neighbor_max_angle_rms),
        "min_center_pass_fraction": float(args.min_center_pass_fraction),
        "min_neighbor_pass_fraction": float(args.min_neighbor_pass_fraction),
        "max_so3_per_center": float(args.max_so3_per_center),
    }
    integrity_report_file = (
        args.integrity_report
        or os.path.join(args.output_dir, f"{args.tag}-integrity.csv")
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    use_discrete = args.discrete is not None
    output_file = build_output_filename(args.output_dir, args.tag, use_discrete, args.discrete_cell)
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Configuration ---")
    print(f"Database: {args.database}")
    print(f"Output CSV: {output_file}")
    print(f"Species/composition: {species} / {composition}")
    print(f"Role/SO3 labels: { {s: config[s]['coordination'] for s in species} }")
    print(f"References: { {s: config[s]['reference'] for s in species} }")
    print(f"SO3 cutoff: {args.rcut}")
    print(f"Integrity filter: {integrity}")
    print(f"Integrity report: {integrity_report_file}")
    print("---------------------")

    filter_by_energy = np.isfinite(args.max_energy)
    db = database_topology(args.database)
    xtals = db.get_all_xtals(include_energy=args.energy or filter_by_energy)
    if filter_by_energy:
        xtals = [x for x in xtals if getattr(x, "ff_energy", None) is not None and x.ff_energy <= args.max_energy]
    params = (
        args.max_dof, 1, args.max_atoms, args.max_energy, args.min_spg,
        args.max_wp, args.max_per_struc, args.energy, use_discrete,
        args.discrete_cell, args.discrete if use_discrete else None, 5e-4,
        config, species, composition, args.rcut, integrity,
    )
    worker = partial(process_one_xtal, params=params, base_seed=args.seed)
    indexed = list(enumerate(xtals))
    total_reps, integrity_reports, usable, failed = [], [], 0, 0

    results = map(worker, indexed) if args.ncpu == 1 else None
    if args.ncpu == 1:
        iterator = results
        pool = None
    else:
        pool = Pool(processes=args.ncpu)
        iterator = pool.imap_unordered(worker, indexed, chunksize=args.chunksize)
    try:
        for source_index, source_reps, source_reports, error in iterator:
            integrity_reports.extend(source_reports)
            if error:
                failed += 1
                print(f"Source {source_index} failed: {error}")
                continue
            if not source_reps:
                continue
            usable += 1
            if args.label:
                source_reps = [np.append(rep, source_index + 1) for rep in source_reps]
            total_reps.extend(source_reps)
            print(f"Completed source {source_index}: {len(source_reps)} representations; total={len(total_reps)}")
    finally:
        if pool is not None:
            pool.close(); pool.join()

    if integrity_reports:
        os.makedirs(os.path.dirname(integrity_report_file) or ".", exist_ok=True)
        pd.DataFrame(integrity_reports).sort_values(
            ["source_index", "stage"], kind="stable"
        ).to_csv(integrity_report_file, index=False)
        accepted_count = sum(bool(row["accepted"]) for row in integrity_reports)
        print(
            f"Integrity checks accepted {accepted_count}/{len(integrity_reports)}; "
            f"report={integrity_report_file}"
        )

    print(f"Usable source structures: {usable}")
    print(f"Failed source structures: {failed}")
    print(f"Total representations: {len(total_reps)}")
    if total_reps:
        make_csv(total_reps, args.energy, args.label, use_discrete,
                 args.discrete_cell, args.max_wp, output_file)


if __name__ == "__main__":
    main()

