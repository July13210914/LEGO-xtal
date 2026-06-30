#!/usr/bin/env python3
"""Train and sample a factorized SiO2 LEGO-Xtal VAE.

This version confirms the Si framework first, then constructs oxygen one
Wyckoff orbit at a time using a narrowed pooled Si--O/O--O ionic probability
field. Cached Sobol orbit pools concentrate exact
search inside promising free-parameter regions while retaining exploration.  The decoder stages are:
    global:       space group and cell
    Si skeleton:  Si Wyckoff occupancy pattern
    Si parameters: site-wise free parameters conditioned on G, Si skeleton, and prior Si sites
    O skeleton:   O Wyckoff occupancy conditioned on the complete Si block
    O parameters: free Wyckoff parameters conditioned on sampled O skeleton

The sampled free parameters are mapped deterministically through PyXtal to exact
Wyckoff generating coordinates before the standard LEGO CSV is written.
Sampling applies hard space-group, stoichiometry, and slot-capacity masks.
Confirmed Si frameworks are dispatched one per CPU task to a dynamically
scheduled manager-worker pool for cached ionic-field oxygen construction.
"""

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
import contextlib
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from lego.VAE_factorized import FactorizedVAE
from pyxtal.symmetry import Group


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


def _wyckoff_free_parameters(spg, wp_index, xyz, row_label):
    """Convert a generating coordinate to a padded 3-vector of free parameters.

    PyXtal defines the exact parameterization through ``get_free_xyzs`` and
    ``get_position_from_free_xyzs``. The returned vector is padded with zeros;
    only the first ``wp.get_dof()`` entries are used during reconstruction.
    """
    group = Group(int(spg))
    if wp_index < 0 or wp_index >= len(group):
        raise ValueError(
            f"{row_label}: Wyckoff index {wp_index} is invalid for space group {spg}."
        )
    wp = group[int(wp_index)]
    xyz = np.asarray(xyz, dtype=float)
    generator = wp.search_generator(xyz, tol=1e-2, symmetrize=True)
    if generator is None:
        # Training coordinates may contain finite-precision displacement from
        # the exact manifold. Project once, then require an exact generator.
        projected = wp.project(xyz)
        generator = wp.search_generator(projected, tol=1e-6, symmetrize=True)
    if generator is None:
        raise ValueError(
            f"{row_label}: cannot map coordinate {xyz.tolist()} to "
            f"{wp.get_label()} in space group {spg}."
        )
    free = np.asarray(wp.get_free_xyzs(generator), dtype=float) % 1.0
    dof = int(wp.get_dof())
    if len(free) != dof:
        raise RuntimeError(
            f"{row_label}: PyXtal returned {len(free)} free parameters for "
            f"{wp.get_label()}, expected {dof}."
        )
    padded = np.zeros(3, dtype=float)
    padded[:dof] = free
    return padded


def _wyckoff_position_from_parameters(spg, wp_index, parameters):
    """Reconstruct an exact generating coordinate from free parameters."""
    group = Group(int(spg))
    if wp_index < 0 or wp_index >= len(group):
        raise ValueError(
            f"Wyckoff index {wp_index} is invalid for space group {spg}."
        )
    wp = group[int(wp_index)]
    dof = int(wp.get_dof())
    free = np.asarray(parameters, dtype=float)[:dof] % 1.0
    xyz = np.asarray(wp.get_position_from_free_xyzs(free), dtype=float) % 1.0

    # This is an invariant of the representation, not a tolerance-based repair.
    check = wp.search_generator(xyz, tol=1e-7, symmetrize=False)
    if check is None:
        raise RuntimeError(
            f"Internal Wyckoff reconstruction failure for spg={spg}, "
            f"wp={wp_index}, dof={dof}, parameters={free.tolist()}."
        )
    return xyz


def build_factorized_blocks(df, num_wps, n_si_max, n_o_max):
    """Build global/species blocks using free Wyckoff parameters.

    The three continuous columns per site are retained for compatibility with
    the existing VAE block layout, but they now mean ``u0,u1,u2``. Unused
    entries are zero for occupied special positions and -1 for padded sites.
    """
    global_df = df[BASE_COLUMNS].copy()

    si_records = []
    o_records = []
    for row_index, row in df.iterrows():
        spg = int(row["spg"])
        si_slots = []
        o_slots = []
        for i in range(num_wps):
            cn = int(row[f"target_coord{i}"])
            wp_index = int(row[f"wp{i}"])
            if wp_index == -1:
                continue
            xyz = [float(row[f"{axis}{i}"]) for axis in "xyz"]
            params = _wyckoff_free_parameters(
                spg, wp_index, xyz, f"row {row_index}, slot {i}"
            )
            site = {"wp": wp_index, "u0": params[0], "u1": params[1], "u2": params[2]}
            if cn == SI_CN:
                si_slots.append(site)
            elif cn == O_CN:
                o_slots.append(site)

        pad = {"wp": -1, "u0": -1.0, "u1": -1.0, "u2": -1.0}
        si_slots += [pad.copy() for _ in range(n_si_max - len(si_slots))]
        o_slots += [pad.copy() for _ in range(n_o_max - len(o_slots))]

        si_record = {"si_skeleton_token": encode_wp_token(s["wp"] for s in si_slots)}
        o_record = {"o_skeleton_token": encode_wp_token(s["wp"] for s in o_slots)}
        for i, site in enumerate(si_slots):
            for j in range(3):
                si_record[f"si_u{j}_{i}"] = site[f"u{j}"]
        for i, site in enumerate(o_slots):
            for j in range(3):
                o_record[f"o_u{j}_{i}"] = site[f"u{j}"]
        si_records.append(si_record)
        o_records.append(o_record)

    return global_df, pd.DataFrame(si_records), pd.DataFrame(o_records)


def blocks_to_si_rows(global_df, si_df, num_wps, n_si_max):
    """Reconstruct only the Si block for pre-oxygen screening.

    Returns the LEGO-like rows and their source positions in the input blocks.
    """
    if len(global_df) != len(si_df):
        raise ValueError("Sampled global and Si block row counts differ.")
    records, source_positions = [], []
    rejected_reconstruction = 0
    for row_index in range(len(global_df)):
        global_row = global_df.iloc[row_index]
        spg = int(round(float(global_row["spg"])))
        si_row = si_df.iloc[row_index]
        try:
            si_wps = decode_wp_token(
                si_row["si_skeleton_token"], n_si_max, "Si skeleton"
            )
            reconstructed = []
            for i, wp_index in enumerate(si_wps):
                if wp_index < 0:
                    continue
                params = [float(si_row[f"si_u{j}_{i}"]) for j in range(3)]
                xyz = _wyckoff_position_from_parameters(spg, wp_index, params)
                reconstructed.append((wp_index, xyz.tolist(), SI_CN))
            if not reconstructed or len(reconstructed) > num_wps:
                raise ValueError("Invalid occupied Si-site count.")
            record = {column: global_row[column] for column in BASE_COLUMNS}
            padded = [(-1, [-1.0, -1.0, -1.0], 0)] * (
                num_wps - len(reconstructed)
            )
            for i, (wp_index, xyz, cn) in enumerate(reconstructed + padded):
                record[f"wp{i}"] = int(wp_index)
                record[f"x{i}"], record[f"y{i}"], record[f"z{i}"] = xyz
                record[f"target_coord{i}"] = int(cn)
            records.append(record)
            source_positions.append(row_index)
        except Exception:
            rejected_reconstruction += 1
    return pd.DataFrame(records), np.asarray(source_positions, dtype=int), rejected_reconstruction


def blocks_to_lego_rows_with_map(
    global_df, si_df, o_df, num_wps, n_si_max, n_o_max
):
    """Full reconstruction plus source-row mapping for repeated O proposals."""
    if not (len(global_df) == len(si_df) == len(o_df)):
        raise ValueError("Sampled block row counts differ.")
    records, source_positions = [], []
    rejected_overflow = rejected_reconstruction = 0
    for row_index in range(len(global_df)):
        global_row = global_df.iloc[row_index]
        spg = int(round(float(global_row["spg"])))
        si_row, o_row = si_df.iloc[row_index], o_df.iloc[row_index]
        try:
            si_wps = decode_wp_token(si_row["si_skeleton_token"], n_si_max, "Si skeleton")
            o_wps = decode_wp_token(o_row["o_skeleton_token"], n_o_max, "O skeleton")
            sites = []
            for i, wp_index in enumerate(si_wps):
                params = [float(si_row[f"si_u{j}_{i}"]) for j in range(3)]
                sites.append((wp_index, params, SI_CN))
            for i, wp_index in enumerate(o_wps):
                params = [float(o_row[f"o_u{j}_{i}"]) for j in range(3)]
                sites.append((wp_index, params, O_CN))
            occupied = [(wp, params, cn) for wp, params, cn in sites if wp != -1]
            if len(occupied) > num_wps:
                rejected_overflow += 1
                continue
            reconstructed = []
            for wp_index, params, cn in occupied:
                xyz = _wyckoff_position_from_parameters(spg, wp_index, params)
                reconstructed.append((wp_index, xyz.tolist(), cn))
            record = {column: global_row[column] for column in BASE_COLUMNS}
            padded = [(-1, [-1.0, -1.0, -1.0], 0)] * (num_wps - len(reconstructed))
            for i, (wp_index, xyz, cn) in enumerate(reconstructed + padded):
                record[f"wp{i}"] = int(wp_index)
                record[f"x{i}"], record[f"y{i}"], record[f"z{i}"] = xyz
                record[f"target_coord{i}"] = int(cn)
            records.append(record)
            source_positions.append(row_index)
        except Exception:
            rejected_reconstruction += 1
    return (pd.DataFrame(records), np.asarray(source_positions, dtype=int),
            rejected_overflow, rejected_reconstruction)


def subset_si_state(state, indices):
    """Take a stable subset of a fixed-Si sampler state."""
    idx = np.asarray(indices, dtype=int)
    return {
        "z": np.asarray(state["z"])[idx],
        "global_x": np.asarray(state["global_x"])[idx],
        "si_x": np.asarray(state["si_x"])[idx],
        "global_df": state["global_df"].iloc[idx].reset_index(drop=True),
        "si_df": state["si_df"].iloc[idx].reset_index(drop=True),
        "valid_mask": np.ones(len(idx), dtype=bool),
        "stats": state.get("stats", {}),
        "max_independent_sites": state.get("max_independent_sites"),
    }


def blocks_to_lego_rows(global_df, si_df, o_df, num_wps, n_si_max, n_o_max):
    """Reconstruct exact Wyckoff coordinates and emit standard LEGO rows."""
    if not (len(global_df) == len(si_df) == len(o_df)):
        raise ValueError("Sampled block row counts differ.")

    records = []
    rejected_overflow = 0
    rejected_reconstruction = 0
    for row_index in range(len(global_df)):
        global_row = global_df.iloc[row_index]
        spg = int(round(float(global_row["spg"])))
        si_row = si_df.iloc[row_index]
        o_row = o_df.iloc[row_index]
        si_wps = decode_wp_token(si_row["si_skeleton_token"], n_si_max, "Si skeleton")
        o_wps = decode_wp_token(o_row["o_skeleton_token"], n_o_max, "O skeleton")

        sites = []
        for i, wp_index in enumerate(si_wps):
            params = [float(si_row[f"si_u{j}_{i}"]) for j in range(3)]
            sites.append((wp_index, params, SI_CN))
        for i, wp_index in enumerate(o_wps):
            params = [float(o_row[f"o_u{j}_{i}"]) for j in range(3)]
            sites.append((wp_index, params, O_CN))

        occupied = [(wp, params, cn) for wp, params, cn in sites if wp != -1]
        if len(occupied) > num_wps:
            rejected_overflow += 1
            continue

        reconstructed = []
        try:
            for wp_index, params, cn in occupied:
                xyz = _wyckoff_position_from_parameters(spg, wp_index, params)
                reconstructed.append((wp_index, xyz.tolist(), cn))
        except (ValueError, RuntimeError, IndexError):
            rejected_reconstruction += 1
            continue

        record = {column: global_row[column] for column in BASE_COLUMNS}
        padded = [(-1, [-1.0, -1.0, -1.0], 0)] * (num_wps - len(reconstructed))
        for i, (wp_index, xyz, cn) in enumerate(reconstructed + padded):
            record[f"wp{i}"] = int(wp_index)
            record[f"x{i}"], record[f"y{i}"], record[f"z{i}"] = xyz
            record[f"target_coord{i}"] = int(cn)
        records.append(record)

    return pd.DataFrame(records), rejected_overflow, rejected_reconstruction

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



class SiAwareOxygenFirstShellGeometry:
    """Differentiable Si-fixed oxygen first-shell objective.

    The training row supplies the crystallographic skeleton and the target
    ordered first-shell distances.  Predicted cell and Si coordinates are
    reconstructed, then detached.  Gradients therefore flow only through the
    oxygen free parameters.  The objective matches Si->O d1/d4/d5 and O->Si
    d1/d2/d3, directly representing SiO4 completion and OSi2 bridging while
    keeping the first unwanted neighbour outside the shell.
    """

    METRIC_WEIGHTS = {
        "si_d1": 0.5, "si_d4": 1.0, "si_d5": 1.0,
        "o_d1": 0.5, "o_d2": 1.0, "o_d3": 1.0,
    }

    def __init__(self, canonical_df, n_si_max, n_o_max):
        self.rows = []
        self.n_si_max = int(n_si_max)
        self.n_o_max = int(n_o_max)
        self._template_cache = {}
        nslots = sum(str(c).startswith("wp") for c in canonical_df.columns)
        for _, row in canonical_df.iterrows():
            spg = int(row["spg"])
            si_wps, o_wps, teacher_si, teacher_o = [], [], [], []
            group = Group(spg)
            for i in range(nslots):
                wp_index = int(row.get(f"wp{i}", -1))
                cn = int(row.get(f"target_coord{i}", 0))
                if wp_index < 0:
                    continue
                generator = np.asarray(
                    [row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]], dtype=float
                )
                wp = group[wp_index]
                positions = np.asarray(
                    [op.operate(generator) for op in wp.ops], dtype=float
                ) % 1.0
                positions = _deduplicate_fractional_positions(positions)
                if cn == SI_CN:
                    si_wps.append(wp_index); teacher_si.append(positions)
                elif cn == O_CN:
                    o_wps.append(wp_index); teacher_o.append(positions)
            if not teacher_si or not teacher_o:
                self.rows.append(None)
                continue
            cell = _cell_matrix_numpy(row)
            teacher_si = np.concatenate(teacher_si, axis=0).astype(np.float32)
            teacher_o = np.concatenate(teacher_o, axis=0).astype(np.float32)
            targets = self._ordered_numpy(teacher_si, teacher_o, cell)
            if targets is None:
                self.rows.append(None)
                continue
            self.rows.append((spg, si_wps, o_wps, targets))

    @staticmethod
    def _op_parts(op):
        rot = getattr(op, "rotation_matrix", None)
        trans = getattr(op, "translation_vector", None)
        if rot is None or trans is None:
            affine = np.asarray(getattr(op, "affine_matrix"), dtype=float)
            rot, trans = affine[:3, :3], affine[:3, 3]
        return np.asarray(rot, dtype=float), np.asarray(trans, dtype=float)

    def _site_template(self, spg, wp_index):
        key = (int(spg), int(wp_index))
        if key in self._template_cache:
            return self._template_cache[key]
        wp = Group(int(spg))[int(wp_index)]
        dof = int(wp.get_dof())
        u0 = np.full(dof, 0.271, dtype=float)
        base = np.asarray(wp.get_position_from_free_xyzs(u0), dtype=float)
        A = np.zeros((3, 3), dtype=float)
        eps = 1e-5
        for j in range(dof):
            uj = u0.copy(); uj[j] += eps
            pos = np.asarray(wp.get_position_from_free_xyzs(uj), dtype=float)
            delta = pos - base; delta -= np.round(delta)
            A[:, j] = delta / eps
        b = base - A[:, :dof] @ u0
        mats, offs = [], []
        for op in wp.ops:
            R, t = self._op_parts(op)
            mats.append(R @ A); offs.append(R @ b + t)
        result = (np.asarray(mats, np.float32), np.asarray(offs, np.float32))
        self._template_cache[key] = result
        return result

    @staticmethod
    def _continuous_layout(transformer):
        layout, st = {}, 0
        for info in transformer._column_transform_info_list:
            ed = st + info.output_dimensions
            if info.column_type == "continuous":
                layout[info.column_name] = (st, ed, info.transform)
            st = ed
        return layout

    @staticmethod
    def _gm_parameters(gm):
        bgm = getattr(gm, "_bgm_transformer", None) or getattr(gm, "_model", None)
        means = np.asarray(getattr(bgm, "means_"), dtype=float).reshape(-1)
        cov = np.asarray(getattr(bgm, "covariances_"), dtype=float).reshape(-1)
        valid = np.asarray(gm.valid_component_indicator, dtype=bool)
        return means[valid], np.sqrt(cov[valid])

    def _raw_parameters(self, logits, transformer, prefix, nslots):
        layout = self._continuous_layout(transformer)
        values = []
        for i in range(nslots):
            slot = []
            for j in range(3):
                st, ed, gm = layout[f"{prefix}_u{j}_{i}"]
                norm = torch.tanh(logits[:, st])
                probs = torch.softmax(logits[:, st + 1:ed], dim=-1)
                means, stds = self._gm_parameters(gm)
                means = torch.as_tensor(means, device=logits.device, dtype=logits.dtype)
                stds = torch.as_tensor(stds, device=logits.device, dtype=logits.dtype)
                raw = norm[:, None] * (4.0 * stds[None, :]) + means[None, :]
                slot.append((probs * raw).sum(dim=1))
            values.append(torch.stack(slot, dim=1))
        return torch.stack(values, dim=1)

    def _raw_continuous_columns(self, logits, transformer, names):
        layout = self._continuous_layout(transformer)
        values = []
        for name in names:
            st, ed, gm = layout[name]
            norm = torch.tanh(logits[:, st])
            probs = torch.softmax(logits[:, st + 1:ed], dim=-1)
            means, stds = self._gm_parameters(gm)
            means = torch.as_tensor(means, device=logits.device, dtype=logits.dtype)
            stds = torch.as_tensor(stds, device=logits.device, dtype=logits.dtype)
            raw = norm[:, None] * (4.0 * stds[None, :]) + means[None, :]
            values.append((probs * raw).sum(dim=1))
        return torch.stack(values, dim=1)

    @staticmethod
    def _torch_cell_matrix(parameters):
        a, b, c, alpha, beta, gamma = parameters.unbind(dim=-1)
        a, b, c = a.clamp_min(.25), b.clamp_min(.25), c.clamp_min(.25)
        ca, cb, cg, sg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma), torch.sin(gamma)
        sg = torch.where(sg.abs() < 1e-4, torch.sign(sg + 1e-8) * 1e-4, sg)
        vt = (1 + 2*ca*cb*cg - ca.square() - cb.square() - cg.square()).clamp_min(1e-8)
        z = torch.zeros_like(a)
        return torch.stack([
            torch.stack([a,z,z],-1),
            torch.stack([b*cg,b*sg,z],-1),
            torch.stack([c*cb,c*(ca-cb*cg)/sg,c*torch.sqrt(vt)/sg],-1),
        ], dim=-2)

    @staticmethod
    def _periodic_distance_tensor(sf, of, cell):
        """Distances for every Si/O pair and all 27 periodic images.

        The image axis must remain explicit until neighbour ranking.  Taking a
        minimum over images first is incorrect for primitive cells: several
        physical first-shell neighbours may be periodic copies of the same
        atom stored in the central cell.
        """
        shifts = torch.as_tensor(
            [[i, j, k] for i in (-1, 0, 1)
             for j in (-1, 0, 1) for k in (-1, 0, 1)],
            device=sf.device,
            dtype=sf.dtype,
        )
        delta = (
            sf[:, None, None, :]
            - of[None, :, None, :]
            + shifts[None, None, :, :]
        )
        cart = torch.einsum("ijsq,qr->ijsr", delta, cell)
        return torch.linalg.norm(cart, dim=-1)

    @classmethod
    def _ordered_torch(cls, sf, of, cell):
        dist = cls._periodic_distance_tensor(sf, of, cell)
        # For each central Si, rank all O atoms in all 27 images.
        si_all = dist.reshape(dist.shape[0], -1)
        # For each central O, rank all Si atoms in all 27 images.
        o_all = dist.permute(1, 0, 2).reshape(dist.shape[1], -1)
        if si_all.shape[1] < 5 or o_all.shape[1] < 3:
            return None
        si5 = torch.topk(si_all, k=5, dim=1, largest=False, sorted=True).values
        o3 = torch.topk(o_all, k=3, dim=1, largest=False, sorted=True).values
        return {
            "si_d1": si5[:, 0], "si_d4": si5[:, 3], "si_d5": si5[:, 4],
            "o_d1": o3[:, 0], "o_d2": o3[:, 1], "o_d3": o3[:, 2],
        }

    @classmethod
    def _ordered_numpy(cls, sf, of, cell):
        shifts = np.asarray(
            [[i, j, k] for i in (-1, 0, 1)
             for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=float,
        )
        delta = (
            sf[:, None, None, :]
            - of[None, :, None, :]
            + shifts[None, None, :, :]
        )
        dist = np.linalg.norm(
            np.einsum("...i,ij->...j", delta, cell), axis=-1
        )
        si_all = dist.reshape(len(sf), -1)
        o_all = np.transpose(dist, (1, 0, 2)).reshape(len(of), -1)
        if si_all.shape[1] < 5 or o_all.shape[1] < 3:
            return None
        si = np.sort(si_all, axis=1)[:, :5]
        oo = np.sort(o_all, axis=1)[:, :3]
        return {
            "si_d1": si[:, 0], "si_d4": si[:, 3], "si_d5": si[:, 4],
            "o_d1": oo[:, 0], "o_d2": oo[:, 1], "o_d3": oo[:, 2],
        }

    def __call__(self, row_ids, global_logits, si_logits, o_logits,
                 global_transformer, si_transformer, o_transformer, device):
        cells = self._torch_cell_matrix(self._raw_continuous_columns(
            global_logits, global_transformer,
            ["a","b","c","alpha","beta","gamma"],
        )).detach()
        # Detach Si: oxygen must adapt to the already established framework.
        si_u = self._raw_parameters(si_logits, si_transformer, "si", self.n_si_max).detach()
        o_u = self._raw_parameters(o_logits, o_transformer, "o", self.n_o_max)
        losses, teacher_scales = [], []
        for local, rid in enumerate(row_ids.detach().cpu().tolist()):
            row = self.rows[int(rid)]
            if row is None:
                continue
            spg, si_wps, o_wps, target = row
            si_pos, o_pos = [], []
            for slot, wp_index in enumerate(si_wps):
                M, q = self._site_template(spg, wp_index)
                M = torch.as_tensor(M, device=device, dtype=o_logits.dtype)
                q = torch.as_tensor(q, device=device, dtype=o_logits.dtype)
                si_pos.append(torch.einsum("aij,j->ai", M, si_u[local,slot]) + q)
            for slot, wp_index in enumerate(o_wps):
                M, q = self._site_template(spg, wp_index)
                M = torch.as_tensor(M, device=device, dtype=o_logits.dtype)
                q = torch.as_tensor(q, device=device, dtype=o_logits.dtype)
                o_pos.append(torch.einsum("aij,j->ai", M, o_u[local,slot]) + q)
            if not si_pos or not o_pos:
                continue
            predicted = self._ordered_torch(torch.cat(si_pos).detach(), torch.cat(o_pos), cells[local])
            if predicted is None:
                continue
            terms = []
            for name, weight in self.METRIC_WEIGHTS.items():
                target_tensor = torch.as_tensor(target[name], device=device, dtype=o_logits.dtype)
                if predicted[name].numel() != target_tensor.numel():
                    continue
                scale = target_tensor.std(unbiased=False).clamp_min(0.10)
                terms.append(weight * torch.nn.functional.smooth_l1_loss(
                    predicted[name] / scale, target_tensor / scale, reduction="mean"
                ))
            if terms:
                losses.append(sum(terms) / sum(self.METRIC_WEIGHTS.values()))
                teacher_scales.append(o_logits.new_zeros(()))
        if not losses:
            zero = o_logits.sum() * 0.0
            return zero, zero
        return torch.stack(losses).mean(), torch.stack(teacher_scales).mean()


def _cell_matrix_numpy(row):
    a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
    alpha, beta, gamma = (
        float(row["alpha"]), float(row["beta"]), float(row["gamma"])
    )
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)
    if abs(sg) < 1.0e-10:
        raise ValueError("Degenerate gamma angle.")
    y3 = c * (ca - cb * cg) / sg
    z3_sq = c * c - (c * cb) ** 2 - y3 ** 2
    if z3_sq <= 1.0e-10:
        raise ValueError("Degenerate cell metric.")
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cb, y3, np.sqrt(z3_sq)],
        ],
        dtype=np.float32,
    )


def _periodic_nearest_numpy(frac, cell, shift_range=2):
    shifts = np.asarray(
        [
            [i, j, k]
            for i in range(-shift_range, shift_range + 1)
            for j in range(-shift_range, shift_range + 1)
            for k in range(-shift_range, shift_range + 1)
        ],
        dtype=float,
    )
    delta = frac[:, None, None, :] - frac[None, :, None, :]
    delta = delta + shifts[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, cell)
    distances = np.linalg.norm(cart, axis=-1)
    zero_shift = np.where(np.all(shifts == 0, axis=1))[0][0]
    ids = np.arange(len(frac))
    distances[ids, ids, zero_shift] = np.inf
    return distances.reshape(len(frac), -1).min(axis=1)


def _deduplicate_fractional_positions(frac, tol=1.0e-5):
    """Merge periodically equivalent fractional coordinates.

    PyXtal Wyckoff operation lists can contain multiple operations that map a
    special-position generator onto the same physical atom.  These duplicates
    must not be interpreted as zero-distance neighbours.
    """
    frac = np.asarray(frac, dtype=float).reshape(-1, 3) % 1.0
    unique = []
    for position in frac:
        duplicate = False
        for existing in unique:
            delta = position - existing
            delta -= np.round(delta)
            if np.linalg.norm(delta) <= tol:
                duplicate = True
                break
        if not duplicate:
            unique.append(position)
    if not unique:
        return np.empty((0, 3), dtype=float)
    return np.asarray(unique, dtype=float)


def _expand_si_from_lego_row(row, num_wps, dedup_tol=1.0e-5):
    """Return unique symmetry-expanded Si coordinates and the cell matrix."""
    spg = int(round(float(row["spg"])))
    group = Group(spg)
    positions = []
    expected_count = 0
    for slot in range(num_wps):
        if int(row[f"target_coord{slot}"]) != SI_CN:
            continue
        wp_index = int(row[f"wp{slot}"])
        if wp_index < 0 or wp_index >= len(group):
            continue
        generator = np.asarray(
            [row[f"x{slot}"], row[f"y{slot}"], row[f"z{slot}"]],
            dtype=float,
        )
        wp = group[wp_index]
        site_positions = np.asarray(
            [op.operate(generator) for op in wp.ops], dtype=float
        ) % 1.0
        site_positions = _deduplicate_fractional_positions(
            site_positions, tol=dedup_tol
        )
        # A correctly reconstructed generating coordinate should expand to the
        # Wyckoff multiplicity.  Keep the unique positions even if finite input
        # precision causes a mismatch, but reject a completely collapsed site.
        if len(site_positions) == 0:
            raise ValueError(
                f"Si Wyckoff site {wp_index} in space group {spg} expanded to no atoms."
            )
        expected_count += int(wp.multiplicity)
        positions.extend(site_positions.tolist())

    positions = _deduplicate_fractional_positions(positions, tol=dedup_tol)
    if len(positions) < 2:
        raise ValueError("Need at least two unique expanded Si atoms.")
    if expected_count > 0 and len(positions) > expected_count:
        raise RuntimeError(
            f"Expanded {len(positions)} unique Si atoms, exceeding expected "
            f"Wyckoff multiplicity sum {expected_count}."
        )
    return positions, _cell_matrix_numpy(row)


def _periodic_pair_distances_numpy(frac, cell, shift_range=1):
    """Full minimum-image pair-distance matrix under periodic translations."""
    shifts = np.asarray(
        [
            [i, j, k]
            for i in range(-shift_range, shift_range + 1)
            for j in range(-shift_range, shift_range + 1)
            for k in range(-shift_range, shift_range + 1)
        ],
        dtype=float,
    )
    delta = frac[:, None, None, :] - frac[None, :, None, :]
    delta = delta + shifts[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.linalg.norm(cart, axis=-1).min(axis=2)
    np.fill_diagonal(dist, np.inf)
    # Numerical or symmetry-equivalent duplicates must never enter the local
    # density or hard-contact statistics as physical neighbours.
    dist[dist < 1.0e-5] = np.inf
    return dist





# Process-local caches: initialized independently in each CPU worker.
_WORKER_GROUP_CACHE = {}
_WORKER_OPS_CACHE = {}


def _cached_group(spg):
    spg = int(spg)
    group = _WORKER_GROUP_CACHE.get(spg)
    if group is None:
        group = Group(spg)
        _WORKER_GROUP_CACHE[spg] = group
    return group


def _cached_ops(spg, wp_index):
    key = (int(spg), int(wp_index))
    cached = _WORKER_OPS_CACHE.get(key)
    if cached is not None:
        return cached
    wp = _cached_group(spg)[int(wp_index)]
    rotations, translations = [], []
    for op in wp.ops:
        rot = getattr(op, "rotation_matrix", None)
        trans = getattr(op, "translation_vector", None)
        if rot is None or trans is None:
            affine = np.asarray(op.affine_matrix, dtype=float)
            rot, trans = affine[:3, :3], affine[:3, 3]
        rotations.append(np.asarray(rot, dtype=np.float32))
        translations.append(np.asarray(trans, dtype=np.float32))
    cached = (np.asarray(rotations), np.asarray(translations))
    _WORKER_OPS_CACHE[key] = cached
    return cached


def _expand_one_species_worker(payload):
    """Expand only one requested species; CUDA is never touched here."""
    position, row_dict, num_wps, target_cn = payload
    try:
        spg = int(round(float(row_dict["spg"])))
        group = _cached_group(spg)
        points = []
        for slot in range(int(num_wps)):
            if int(row_dict[f"target_coord{slot}"]) != int(target_cn):
                continue
            wp_index = int(row_dict[f"wp{slot}"])
            if wp_index < 0 or wp_index >= len(group):
                raise ValueError(f"Invalid Wyckoff index {wp_index} for spg {spg}.")
            generator = np.asarray(
                [row_dict[f"x{slot}"], row_dict[f"y{slot}"], row_dict[f"z{slot}"]],
                dtype=np.float32,
            )
            rotations, translations = _cached_ops(spg, wp_index)
            expanded = np.einsum("aij,j->ai", rotations, generator) + translations
            expanded = _deduplicate_fractional_positions(expanded % 1.0)
            if len(expanded) == 0:
                raise ValueError(f"Collapsed Wyckoff site {wp_index} for spg {spg}.")
            points.extend(expanded.tolist())
        points = _deduplicate_fractional_positions(points)
        minimum = 2 if int(target_cn) == SI_CN else 1
        if len(points) < minimum:
            raise ValueError(f"Insufficient expanded atoms for CN label {target_cn}: {len(points)}.")
        return position, True, np.asarray(points, dtype=np.float32), ""
    except Exception as exc:
        return position, False, np.empty((0, 3), dtype=np.float32), f"{type(exc).__name__}: {exc}"


def prepare_species_candidates(batch_df, num_wps, target_cn, workers=0, executor=None):
    """CPU-expand one species while preserving input row order."""
    records = batch_df.to_dict(orient="records")
    payloads = [(i, row, int(num_wps), int(target_cn)) for i, row in enumerate(records)]
    if int(workers) <= 1:
        results = [_expand_one_species_worker(item) for item in payloads]
    else:
        owns_executor = executor is None
        pool = executor or ProcessPoolExecutor(max_workers=int(workers))
        try:
            chunksize = max(1, len(payloads) // max(1, int(workers) * 4))
            results = list(pool.map(_expand_one_species_worker, payloads, chunksize=chunksize))
        finally:
            if owns_executor:
                pool.shutdown(wait=True)
    output = [None] * len(records)
    for position, ok, coords, error in results:
        output[int(position)] = {"ok": bool(ok), "coords": coords, "error": error}
    return output


def combine_prepared_species(batch_df, si_prepared, o_prepared=None):
    """Attach validated cell matrices and combine prepared species arrays.

    Geometry failures are represented per candidate and must never abort an
    otherwise valid sampling round.  In particular, decoded cell parameters
    can occasionally produce a singular metric even when Wyckoff expansion
    itself succeeds.
    """
    if len(batch_df) != len(si_prepared) or (
        o_prepared is not None and len(batch_df) != len(o_prepared)
    ):
        raise ValueError("Prepared geometry count does not match batch rows.")

    empty_xyz = np.empty((0, 3), dtype=np.float32)
    empty_cell = np.empty((3, 3), dtype=np.float32)
    combined = []
    for position, (_, row) in enumerate(batch_df.iterrows()):
        si_item = si_prepared[position]
        o_item = None if o_prepared is None else o_prepared[position]

        species_ok = bool(si_item.get("ok", False)) and (
            o_item is None or bool(o_item.get("ok", False))
        )
        error = ""
        if not bool(si_item.get("ok", False)):
            error = si_item.get("error", "Si expansion failed")
        elif o_item is not None and not bool(o_item.get("ok", False)):
            error = o_item.get("error", "O expansion failed")

        cell = empty_cell
        cell_ok = False
        if species_ok:
            try:
                cell = np.asarray(_cell_matrix_numpy(row), dtype=np.float32)
                if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
                    raise ValueError("Non-finite or malformed cell matrix.")
                det = float(np.linalg.det(cell))
                if not np.isfinite(det) or abs(det) <= 1.0e-8:
                    raise ValueError(f"Singular cell matrix with determinant {det}.")
                cell_ok = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        ok = bool(species_ok and cell_ok)
        combined.append({
            "ok": ok,
            "si": (
                np.asarray(si_item["coords"], dtype=np.float32)
                if bool(si_item.get("ok", False)) else empty_xyz
            ),
            "o": (
                empty_xyz if o_item is None
                else np.asarray(o_item["coords"], dtype=np.float32)
                if bool(o_item.get("ok", False)) else empty_xyz
            ),
            "cell": cell if ok else empty_cell,
            "error": error,
        })
    return combined


class OnlineOxygenFirstShellSelector:
    """Online empirical population matching for the SiO4/OSi2 first shell.

    Six marginal order-statistic populations are matched simultaneously:
    Si->O d1/d4/d5 and O->Si d1/d2/d3.  All are extracted from one exact
    27-image batched Si-O distance matrix.  No angular or second-shell target
    is used; d5 and d3 only mark the first unwanted neighbour.
    """

    METRICS = ("si_d1", "si_d4", "si_d5", "o_d1", "o_d2", "o_d3")

    def __init__(self, training_prepared, requested_structures, bins=40,
                 overfill_penalty=2.0, selection_temperature=0.003,
                 weights=None, seed=42, device="cuda", gpu_batch=1024,
                 memory_gib=1.5, cn_cutoff=2.2):
        self.bins = int(bins)
        self.overfill_penalty = float(overfill_penalty)
        self.selection_temperature = float(selection_temperature)
        self.weights = dict(weights or {
            "si_d1": 0.5, "si_d4": 1.0, "si_d5": 1.0,
            "o_d1": 0.5, "o_d2": 1.0, "o_d3": 1.0,
        })
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gpu_batch = max(1, int(gpu_batch))
        self.max_bytes = int(float(memory_gib) * (1024 ** 3))
        self.cn_cutoff = float(cn_cutoff)
        self.failed_geometry = 0
        self.raw_structures = 0
        self.accepted_structures = 0
        self.cn_si4 = []
        self.cn_o2 = []
        self.cn_both = []
        training_desc = self.describe(training_prepared, update_raw=False)
        if not training_desc:
            raise RuntimeError("No valid training Si-O first-shell environments.")
        values = {name: [] for name in self.METRICS}
        nsi, no = [], []
        for item in training_desc:
            for name in self.METRICS:
                values[name].extend(item["values"][name].tolist())
            nsi.append(len(item["values"]["si_d1"]))
            no.append(len(item["values"]["o_d1"]))
        self.edges, self.target_probability, self.target_counts = {}, {}, {}
        self.training_values = {}
        for name in self.METRICS:
            arr = np.asarray(values[name], dtype=float)
            self.training_values[name] = arr
            q001, q999 = np.percentile(arr, [0.1, 99.9])
            pad = max(0.05, 0.1 * (q999 - q001))
            low = min(float(arr.min()), q001 - pad)
            high = max(float(arr.max()), q999 + pad)
            if high <= low:
                high = low + 1.0
            edges = np.linspace(low, high, self.bins + 1)
            hist, _ = np.histogram(np.clip(arr, edges[0] + 1e-12, edges[-1] - 1e-12), bins=edges)
            prob = hist.astype(float) / max(hist.sum(), 1)
            mean_env = float(np.mean(nsi if name.startswith("si_") else no))
            self.edges[name] = edges
            self.target_probability[name] = prob
            self.target_counts[name] = prob * requested_structures * mean_env
        self.accepted_counts = {name: np.zeros(self.bins, dtype=float) for name in self.METRICS}
        self.raw_counts = {name: np.zeros(self.bins, dtype=float) for name in self.METRICS}
        self.accepted_values = {name: [] for name in self.METRICS}
        self.raw_values = {name: [] for name in self.METRICS}

    @staticmethod
    def _shifts(device):
        return torch.as_tensor([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)], device=device, dtype=torch.float32)

    def _hist(self, name, arr):
        edges = self.edges[name]
        clipped = np.clip(np.asarray(arr, dtype=float), edges[0] + 1e-12, edges[-1] - 1e-12)
        return np.histogram(clipped, bins=edges)[0].astype(float)

    def describe(self, prepared, update_raw=True):
        valid = [
            i for i, item in enumerate(prepared)
            if item.get("ok", False)
            and len(item["si"]) >= 1
            and len(item["o"]) >= 1
        ]
        self.failed_geometry += len(prepared) - len(valid) if update_raw else 0
        descriptions = []
        buckets = {}
        for idx in valid:
            item = prepared[idx]
            buckets.setdefault((len(item["si"]), len(item["o"])), []).append(idx)
        shifts = self._shifts(self.device)
        with torch.inference_mode():
            for (nsi, no), ids_all in buckets.items():
                bytes_per = max(nsi * no * 27 * 3 * 4 * 3, 1)
                dynamic = max(1, min(self.gpu_batch, self.max_bytes // bytes_per))
                for start in range(0, len(ids_all), dynamic):
                    ids = ids_all[start:start + dynamic]
                    sf = torch.stack([torch.as_tensor(prepared[i]["si"], device=self.device) for i in ids]).float()
                    of = torch.stack([torch.as_tensor(prepared[i]["o"], device=self.device) for i in ids]).float()
                    cell = torch.stack([torch.as_tensor(prepared[i]["cell"], device=self.device) for i in ids]).float()
                    delta = (
                        sf[:, :, None, None, :]
                        - of[:, None, :, None, :]
                        + shifts[None, None, None, :, :]
                    )
                    cart = torch.einsum("bijsq,bqr->bijsr", delta, cell)
                    sio_images = torch.linalg.norm(cart, dim=-1)
                    # Preserve the 27 image axis while ranking/counting.  A
                    # primitive-cell atom may contribute several distinct
                    # periodic neighbours to the first shell.
                    si_all = sio_images.reshape(len(ids), nsi, no * 27)
                    o_all = (
                        sio_images.permute(0, 2, 1, 3)
                        .reshape(len(ids), no, nsi * 27)
                    )
                    si5 = torch.topk(
                        si_all, k=5, dim=2, largest=False, sorted=True
                    ).values
                    o3 = torch.topk(
                        o_all, k=3, dim=2, largest=False, sorted=True
                    ).values
                    si_cn = (si_all <= self.cn_cutoff).sum(dim=2)
                    o_cn = (o_all <= self.cn_cutoff).sum(dim=2)
                    for j, original in enumerate(ids):
                        vals = {
                            "si_d1": si5[j,:,0].cpu().numpy(),
                            "si_d4": si5[j,:,3].cpu().numpy(),
                            "si_d5": si5[j,:,4].cpu().numpy(),
                            "o_d1": o3[j,:,0].cpu().numpy(),
                            "o_d2": o3[j,:,1].cpu().numpy(),
                            "o_d3": o3[j,:,2].cpu().numpy(),
                        }
                        hists = (
                            {name: self._hist(name, vals[name]) for name in self.METRICS}
                            if hasattr(self, "edges") and self.edges else {}
                        )
                        desc = {"position": int(original), "values": vals, "hist": hists,
                                "frac_si4": float((si_cn[j] == 4).float().mean().item()),
                                "frac_o2": float((o_cn[j] == 2).float().mean().item()),
                                "all_both": bool((si_cn[j] == 4).all().item() and (o_cn[j] == 2).all().item())}
                        descriptions.append(desc)
                        if update_raw:
                            for name in self.METRICS:
                                self.raw_counts[name] += hists[name]
                                self.raw_values[name].extend(vals[name].tolist())
                            self.raw_structures += 1
                    del sf, of, cell, delta, cart, sio_images, si_all, o_all
                    del si5, o3, si_cn, o_cn
        descriptions.sort(key=lambda x: x["position"])
        return descriptions

    def _score(self, desc, current):
        score = 0.0
        for name in self.METRICS:
            hist = desc["hist"][name]
            target = self.target_counts[name]
            denom = np.maximum(target, 1.0)
            deficit = np.maximum(target - current[name], 0.0)
            fill = np.sum(hist * deficit / denom)
            over = np.sum(np.maximum(current[name] + hist - target, 0.0) / denom)
            score += self.weights[name] * (fill - self.overfill_penalty * over)
        if self.selection_temperature > 0:
            score += float(self.rng.gumbel(0.0, self.selection_temperature))
        return score

    def candidate_mean_tv(self, desc):
        values = []
        for name in self.METRICS:
            hist = np.asarray(desc["hist"][name], dtype=float)
            values.append(self._tv(hist, self.target_probability[name]))
        return float(np.mean(values))

    def candidate_metric_tvs(self, desc):
        return {
            name: self._tv(np.asarray(desc["hist"][name], dtype=float),
                          self.target_probability[name])
            for name in self.METRICS
        }

    def candidate_max_tv(self, desc):
        return float(max(self.candidate_metric_tvs(desc).values()))

    def feedback_vector(self, desc):
        """Compact local failure signal for error-conditioned O regeneration."""
        features = []
        for name in self.METRICS:
            train = self.training_values[name]
            scale = max(float(np.std(train)), 0.10)
            value = float(np.mean(desc["values"][name]))
            features.append((value - float(np.mean(train))) / scale)
        si_gap = float(np.mean(desc["values"]["si_d5"] - desc["values"]["si_d4"]))
        o_gap = float(np.mean(desc["values"]["o_d3"] - desc["values"]["o_d2"]))
        train_si_gap = float(np.mean(self.training_values["si_d5"]) -
                             np.mean(self.training_values["si_d4"]))
        train_o_gap = float(np.mean(self.training_values["o_d3"]) -
                            np.mean(self.training_values["o_d2"]))
        features.extend([
            (si_gap - train_si_gap) / max(float(np.std(self.training_values["si_d5"])), 0.10),
            (o_gap - train_o_gap) / max(float(np.std(self.training_values["o_d3"])), 0.10),
        ])
        return np.asarray(features, dtype=np.float32)

    def select(self, descriptions, remaining, oversample_factor=1.0, commit=True,
               block_size=64, min_score=0.0, max_selected=None):
        """Ranked block-greedy six-population selection.

        Scores are relative ranking criteria, not pass/fail thresholds.  The
        previous positive-score gate collapsed to a one-candidate fallback once
        early target bins were filled.  This routine always returns the best
        requested number of candidates, rescoring after each committed block.
        ``min_score`` is accepted but does not alter the ranking score.
        """
        if not descriptions or remaining <= 0:
            return np.zeros(0, dtype=int)
        available = np.arange(len(descriptions))
        current = {name: self.accepted_counts[name].copy() for name in self.METRICS}
        selected = []
        limit = min(int(remaining), len(descriptions), int(max_selected or len(descriptions)))
        while available.size and len(selected) < limit:
            scores = np.asarray([self._score(descriptions[int(i)], current) for i in available])
            order = np.argsort(scores)[::-1]
            ntake = min(int(block_size), limit - len(selected), available.size)
            take = order[:ntake]
            chosen = available[take]
            selected.extend(chosen.tolist())
            for idx in chosen:
                for name in self.METRICS:
                    current[name] += descriptions[int(idx)]["hist"][name]
            keep = np.ones(available.size, dtype=bool)
            keep[take] = False
            available = available[keep]
        if commit:
            self.commit(descriptions, selected)
        return np.asarray(selected, dtype=int)

    def commit(self, descriptions, selected):
        for chosen in selected:
            item = descriptions[int(chosen)]
            for name in self.METRICS:
                self.accepted_counts[name] += item["hist"][name]
                self.accepted_values[name].extend(item["values"][name].tolist())
            self.cn_si4.append(item["frac_si4"])
            self.cn_o2.append(item["frac_o2"])
            self.cn_both.append(item["all_both"])
        self.accepted_structures += len(selected)

    @staticmethod
    def _tv(counts, target):
        return float(0.5 * np.abs(counts / max(counts.sum(), 1.0) - target).sum())

    def concise(self):
        tv = np.mean([self._tv(self.accepted_counts[n], self.target_probability[n]) for n in self.METRICS])
        return (f"accepted={self.accepted_structures}, mean_TV={tv:.4f}, "
                f"Si_CN4={np.mean(self.cn_si4) if self.cn_si4 else np.nan:.3f}, "
                f"O_CN2={np.mean(self.cn_o2) if self.cn_o2 else np.nan:.3f}, "
                f"all_both={np.mean(self.cn_both) if self.cn_both else np.nan:.3f}")

    def metric_summary(self, scope="accepted"):
        if scope == "accepted":
            source = self.accepted_values
            count_source = self.accepted_counts
        elif scope == "raw":
            source = self.raw_values
            count_source = self.raw_counts
        elif scope == "training":
            source = self.training_values
            count_source = None
        else:
            raise ValueError("scope must be accepted, raw, or training")
        out = {}
        for name in self.METRICS:
            arr = np.asarray(source[name], dtype=float)
            if arr.size:
                q05, q50, q95 = np.percentile(arr, [5, 50, 95])
                out[name] = {
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "q05": float(q05), "q50": float(q50), "q95": float(q95),
                    "tv": self._tv(
                        count_source[name] if count_source is not None else np.histogram(
                            np.clip(arr, self.edges[name][0] + 1e-12,
                                    self.edges[name][-1] - 1e-12),
                            bins=self.edges[name]
                        )[0].astype(float),
                        self.target_probability[name],
                    ),
                }
            else:
                out[name] = {k: float("nan") for k in ("mean","std","q05","q50","q95","tv")}
        return out

    def format_metric_summary(self, scope="accepted", prefix=""):
        summary = self.metric_summary(scope=scope)
        parts = []
        for name in self.METRICS:
            x = summary[name]
            parts.append(
                f"{name}={x['mean']:.3f}+/-{x['std']:.3f} "
                f"[{x['q05']:.3f},{x['q50']:.3f},{x['q95']:.3f}] "
                f"TV={x['tv']:.3f}"
            )
        return prefix + "; ".join(parts)

    def diagnostics_frame(self):
        rows = []
        for name in self.METRICS:
            raw = self.raw_counts[name] / max(self.raw_counts[name].sum(), 1.0)
            accepted = self.accepted_counts[name] / max(self.accepted_counts[name].sum(), 1.0)
            for i in range(self.bins):
                rows.append({"metric": name, "bin_left": self.edges[name][i], "bin_right": self.edges[name][i+1],
                             "training_probability": self.target_probability[name][i],
                             "raw_probability": raw[i], "accepted_probability": accepted[i]})
        return pd.DataFrame(rows)



class OnlineOConditionalAdapter(torch.nn.Module):
    """Persistent error-conditioned proposal adapter trained during sampling.

    It predicts the mean of the oxygen-only noise distribution from the frozen
    complete-Si context and the latest eight-dimensional shell-error vector.
    Online updates imitate the best improving noise vectors observed so far;
    the pretrained VAE remains frozen.
    """

    def __init__(self, context_dim, feedback_dim, noise_dim, hidden_dim=128,
                 lr=3.0e-4, replay_size=2048, device="cuda", seed=42):
        super().__init__()
        self.context_dim = int(context_dim)
        self.feedback_dim = int(feedback_dim)
        self.noise_dim = int(noise_dim)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.context_dim + self.feedback_dim, int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), self.noise_dim),
        ).to(self.device)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(lr), weight_decay=1e-6)
        self.replay = deque(maxlen=int(replay_size))
        self.rng = np.random.default_rng(seed)
        self.update_steps = 0
        self.last_loss = float("nan")

    def _inputs(self, contexts, feedback):
        x = np.concatenate([
            np.asarray(contexts, dtype=np.float32),
            np.asarray(feedback, dtype=np.float32),
        ], axis=1)
        return torch.as_tensor(x, device=self.device)

    def propose(self, contexts, feedback, proposals_per_parent, exploration=1.0):
        contexts = np.asarray(contexts, dtype=np.float32)
        feedback = np.asarray(feedback, dtype=np.float32)
        with torch.no_grad():
            mean = self.net(self._inputs(contexts, feedback)).cpu().numpy()
        k = int(proposals_per_parent)
        noise = np.repeat(mean, k, axis=0)
        noise += self.rng.normal(
            0.0, float(exploration), size=noise.shape
        ).astype(np.float32)
        return noise.astype(np.float32, copy=False), mean.astype(np.float32, copy=False)

    def remember(self, context, feedback, target_noise, weight):
        self.replay.append((
            np.asarray(context, dtype=np.float32).copy(),
            np.asarray(feedback, dtype=np.float32).copy(),
            np.asarray(target_noise, dtype=np.float32).copy(),
            float(max(weight, 1.0e-3)),
        ))

    def update(self, steps=1, batch_size=64):
        if not self.replay:
            return float("nan")
        losses = []
        self.train()
        for _ in range(max(1, int(steps))):
            n = min(int(batch_size), len(self.replay))
            ids = self.rng.choice(len(self.replay), size=n, replace=False)
            batch = [self.replay[int(i)] for i in ids]
            c = np.stack([x[0] for x in batch])
            f = np.stack([x[1] for x in batch])
            y = torch.as_tensor(np.stack([x[2] for x in batch]), device=self.device)
            w = torch.as_tensor(np.asarray([x[3] for x in batch], dtype=np.float32), device=self.device)
            pred = self.net(self._inputs(c, f))
            per = (pred - y).square().mean(dim=1)
            loss = (per * w / w.mean().clamp_min(1e-6)).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.update_steps += 1
        self.eval()
        self.last_loss = float(np.mean(losses))
        return self.last_loss


class OnlinePerSiNearestNeighborSelector:
    """Match the accepted population of per-Si nearest-neighbour distances.

    Every symmetry-expanded Si atom contributes exactly one periodic nearest-
    neighbour distance.  The target is a Gaussian fitted to the complete
    training population.  Candidate structures are selected online to fill
    deficits in the accepted global histogram; a broad explicit lower floor
    is used only to reject catastrophic contacts.
    """

    def __init__(
        self,
        training_df,
        num_wps,
        requested_structures,
        nn_bins=40,
        shift_range=1,
        overfill_penalty=1.0,
        selection_temperature=0.01,
        safety_floor=1.67,
        target_sigma_scale=1.0,
        histogram_sigma_span=4.0,
        seed=42,
        gpu_device="cuda",
        gpu_contact_batch=128,
        gpu_shift_chunk=27,
        gpu_max_memory_gib=0.75,
    ):
        if requested_structures <= 0 or nn_bins < 4:
            raise ValueError("Requested structures must be positive and nn_bins >= 4.")
        if safety_floor <= 0 or target_sigma_scale <= 0 or histogram_sigma_span <= 1:
            raise ValueError("Invalid NN support or Gaussian-width settings.")
        self.num_wps = int(num_wps)
        self.nn_bins = int(nn_bins)
        self.shift_range = max(1, int(shift_range))
        self.overfill_penalty = float(overfill_penalty)
        self.selection_temperature = float(selection_temperature)
        self.safety_floor = float(safety_floor)
        self.target_sigma_scale = float(target_sigma_scale)
        self.histogram_sigma_span = float(histogram_sigma_span)
        self.rng = np.random.default_rng(seed)
        self.gpu_device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")
        self.gpu_contact_batch = max(1, int(gpu_contact_batch))
        self.gpu_shift_chunk = max(1, int(gpu_shift_chunk))
        self.gpu_max_bytes = int(float(gpu_max_memory_gib) * (1024 ** 3))

        training_values = []
        nsi = []
        skipped = 0
        for _, row in training_df.iterrows():
            try:
                frac, cell = _expand_si_from_lego_row(row, self.num_wps)
                nearest = _periodic_nearest_numpy(frac, cell, shift_range=self.shift_range)
                nearest = nearest[np.isfinite(nearest)]
                if len(nearest) != len(frac):
                    raise ValueError("Incomplete periodic nearest-neighbour vector.")
                training_values.extend(nearest.tolist())
                nsi.append(len(nearest))
            except Exception:
                skipped += 1
        if not training_values:
            raise RuntimeError("No valid per-Si nearest-neighbour training environments.")

        self.training_values = np.asarray(training_values, dtype=float)
        self.training_structures = len(nsi)
        self.skipped_training = int(skipped)
        self.mean_training_nsi = float(np.mean(nsi))
        self.target_mean = float(np.mean(self.training_values))
        raw_std = float(np.std(self.training_values))
        self.target_std = max(raw_std * self.target_sigma_scale, 1.0e-3)
        self.training_quantiles = {
            name: float(value) for name, value in zip(
                ("q01", "q05", "q25", "q50", "q75", "q95", "q99"),
                np.percentile(self.training_values, [1, 5, 25, 50, 75, 95, 99]),
            )
        }

        low = min(
            float(np.min(self.training_values)),
            self.target_mean - self.histogram_sigma_span * self.target_std,
        )
        high = max(
            float(np.max(self.training_values)),
            self.target_mean + self.histogram_sigma_span * self.target_std,
        )
        low = min(low, self.safety_floor)
        if high <= low:
            high = low + 1.0
        self.edges = np.linspace(low, high, self.nn_bins + 1)
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        gaussian = np.exp(-0.5 * ((centers - self.target_mean) / self.target_std) ** 2)
        gaussian[centers <= self.safety_floor] = 0.0
        if gaussian.sum() <= 0:
            raise RuntimeError("Gaussian NN target has zero probability in all bins.")
        self.target_probability = gaussian / gaussian.sum()
        train_hist, _ = np.histogram(self.training_values, bins=self.edges)
        self.training_empirical_probability = train_hist / max(train_hist.sum(), 1)
        expected_environments = requested_structures * self.mean_training_nsi
        self.target_counts = self.target_probability * expected_environments

        self.raw_counts = np.zeros(self.nn_bins, dtype=float)
        self.accepted_counts = np.zeros(self.nn_bins, dtype=float)
        self.raw_values = []
        self.accepted_values = []
        self.raw_structures = 0
        self.accepted_structures = 0
        self.rejected_safety = 0
        self.failed_geometry = 0

    def _gpu_nearest_vectors(self, expanded):
        """Return one periodic nearest-neighbour vector per candidate structure."""
        if not expanded:
            return []
        output = [None] * len(expanded)
        order = sorted(range(len(expanded)), key=lambda i: len(expanded[i][1]))
        r = self.shift_range
        shifts_all = np.asarray(
            [[i, j, k] for i in range(-r, r + 1)
             for j in range(-r, r + 1) for k in range(-r, r + 1)],
            dtype=np.float32,
        )
        with torch.inference_mode():
            for base in range(0, len(order), self.gpu_contact_batch):
                ids = order[base:base + self.gpu_contact_batch]
                nmax = max(len(expanded[i][1]) for i in ids)
                bytes_per_row = max(nmax * nmax * self.gpu_shift_chunk * 3 * 4, 1)
                dynamic = max(1, min(len(ids), self.gpu_max_bytes // bytes_per_row))
                for sub0 in range(0, len(ids), dynamic):
                    sub = ids[sub0:sub0 + dynamic]
                    b = len(sub)
                    frac = torch.zeros((b, nmax, 3), device=self.gpu_device, dtype=torch.float32)
                    mask = torch.zeros((b, nmax), device=self.gpu_device, dtype=torch.bool)
                    metric = torch.zeros((b, 3, 3), device=self.gpu_device, dtype=torch.float32)
                    lengths = []
                    for j, idx in enumerate(sub):
                        f = torch.as_tensor(expanded[idx][1], device=self.gpu_device, dtype=torch.float32)
                        c = torch.as_tensor(expanded[idx][2], device=self.gpu_device, dtype=torch.float32)
                        lengths.append(len(f))
                        frac[j, :len(f)] = f
                        mask[j, :len(f)] = True
                        metric[j] = c @ c.T
                    run = torch.full((b, nmax, nmax), float("inf"), device=self.gpu_device)
                    for sh0 in range(0, len(shifts_all), self.gpu_shift_chunk):
                        shifts = torch.as_tensor(
                            shifts_all[sh0:sh0 + self.gpu_shift_chunk],
                            device=self.gpu_device,
                        )
                        delta = (
                            frac[:, :, None, None, :]
                            - frac[:, None, :, None, :]
                            + shifts[None, None, None, :, :]
                        )
                        d2 = torch.einsum("bijsq,bqr,bijsr->bijs", delta, metric, delta)
                        run = torch.minimum(run, d2.amin(dim=3))
                        del delta, d2
                    pairmask = mask[:, :, None] & mask[:, None, :]
                    eye = torch.eye(nmax, device=self.gpu_device, dtype=torch.bool)[None]
                    run.masked_fill_(~pairmask | eye, float("inf"))
                    nearest = torch.sqrt(run.amin(dim=2).clamp_min(0)).cpu().numpy()
                    for j, idx in enumerate(sub):
                        output[idx] = nearest[j, :lengths[j]].astype(float, copy=True)
                    del frac, mask, metric, run, nearest
        return output

    def _histogram(self, values):
        values = np.asarray(values, dtype=float)
        clipped = np.clip(values, self.edges[0] + 1.0e-12, self.edges[-1] - 1.0e-12)
        hist, _ = np.histogram(clipped, bins=self.edges)
        return hist.astype(float)

    def describe_batch(self, batch_df, prepared=None):
        expanded = []
        if prepared is None:
            for position, (_, row) in enumerate(batch_df.iterrows()):
                try:
                    frac, cell = _expand_si_from_lego_row(row, self.num_wps)
                    expanded.append((position, frac, cell))
                except Exception:
                    self.failed_geometry += 1
        else:
            if len(prepared) != len(batch_df):
                raise ValueError("Prepared geometry count does not match batch rows.")
            for position, item in enumerate(prepared):
                if not item.get("ok", False):
                    self.failed_geometry += 1
                    continue
                expanded.append((position, item["si"], item["cell"]))
        vectors = self._gpu_nearest_vectors(expanded)
        keep_rows = []
        descriptions = []
        for item, nearest in zip(expanded, vectors):
            position, frac, _ = item
            if nearest is None or len(nearest) != len(frac) or not np.all(np.isfinite(nearest)):
                self.failed_geometry += 1
                continue
            if float(np.min(nearest)) <= self.safety_floor:
                self.rejected_safety += 1
                continue
            hist = self._histogram(nearest)
            self.raw_counts += hist
            self.raw_values.extend(nearest.tolist())
            self.raw_structures += 1
            keep_rows.append(position)
            descriptions.append({"hist": hist, "nn": nearest})
        return keep_rows, descriptions

    def _candidate_scores(self, histograms, current_counts):
        deficit = np.maximum(self.target_counts - current_counts, 0.0)
        denominator = np.maximum(self.target_counts, 1.0)
        fill = (histograms * (deficit / denominator)[None, :]).sum(axis=1)
        over = np.maximum(current_counts[None, :] + histograms - self.target_counts[None, :], 0.0)
        over = (over / denominator[None, :]).sum(axis=1)
        score = fill - self.overfill_penalty * over
        if self.selection_temperature > 0:
            score += self.rng.gumbel(0.0, self.selection_temperature, size=score.shape)
        return score

    def select(self, descriptions, remaining, oversample_factor=1.0, commit=True,
               block_size=64, min_score=0.0, max_selected=None):
        """Ranked block-greedy selection for the single Si-NN population.

        Always return the best requested pool.  Absolute score sign is not a
        valid pass/fail condition after some histogram bins have filled.
        ``min_score`` is accepted but does not alter the ranking score.
        """
        if not descriptions or remaining <= 0:
            return np.zeros(0, dtype=int)
        histograms = np.stack([item["hist"] for item in descriptions])
        available = np.arange(len(descriptions))
        current = self.accepted_counts.copy()
        selected = []
        limit = min(int(remaining), len(descriptions), int(max_selected or len(descriptions)))
        while available.size and len(selected) < limit:
            scores = self._candidate_scores(histograms[available], current)
            order = np.argsort(scores)[::-1]
            ntake = min(int(block_size), limit - len(selected), available.size)
            take = order[:ntake]
            chosen = available[take]
            selected.extend(chosen.tolist())
            current += histograms[chosen].sum(axis=0)
            keep = np.ones(available.size, dtype=bool)
            keep[take] = False
            available = available[keep]
        if commit:
            self.commit(descriptions, selected)
        return np.asarray(selected, dtype=int)

    def commit(self, descriptions, selected):
        for chosen in selected:
            self.accepted_counts += descriptions[int(chosen)]["hist"]
            self.accepted_values.extend(descriptions[int(chosen)]["nn"].tolist())
        self.accepted_structures += len(selected)

    @staticmethod
    def _tv(counts, target):
        counts = np.asarray(counts, dtype=float)
        if counts.sum() <= 0:
            return float("nan")
        return float(0.5 * np.abs(counts / counts.sum() - target).sum())

    def histogram_distance(self, counts=None):
        values = self.accepted_counts if counts is None else np.asarray(counts, dtype=float)
        return self._tv(values, self.target_probability)

    @staticmethod
    def summarize_values(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "q05": float("nan"), "q50": float("nan"), "q95": float("nan")}
        q05, q50, q95 = np.percentile(values, [5, 50, 95])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
        }

    def accepted_summary(self):
        return self.summarize_values(self.accepted_values)

    def diagnostics_frame(self):
        raw = self.raw_counts / max(self.raw_counts.sum(), 1.0)
        accepted = self.accepted_counts / max(self.accepted_counts.sum(), 1.0)
        rows = []
        for i in range(self.nn_bins):
            rows.append({
                "bin_left": self.edges[i],
                "bin_right": self.edges[i + 1],
                "gaussian_target_probability": self.target_probability[i],
                "training_empirical_probability": self.training_empirical_probability[i],
                "raw_generated_probability": raw[i],
                "accepted_probability": accepted[i],
            })
        return pd.DataFrame(rows)



def _periodic_cross_distances_numpy(a_frac, b_frac, cell):
    """Return all 27-image distances from central a atoms to b images."""
    a_frac = np.asarray(a_frac, dtype=float).reshape(-1, 3)
    b_frac = np.asarray(b_frac, dtype=float).reshape(-1, 3)
    shifts = np.asarray(
        [[i, j, k] for i in (-1, 0, 1)
         for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    delta = a_frac[:, None, None, :] - b_frac[None, :, None, :] + shifts[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, np.asarray(cell, dtype=float))
    return np.linalg.norm(cart, axis=-1)


def _expand_o_orbit(spg, wp_index, free_parameters):
    group = Group(int(spg))
    wp = group[int(wp_index)]
    dof = int(wp.get_dof())
    free = np.asarray(free_parameters, dtype=float)[:dof] % 1.0
    generator = np.asarray(wp.get_position_from_free_xyzs(free), dtype=float) % 1.0
    orbit = np.asarray([op.operate(generator) for op in wp.ops], dtype=float) % 1.0
    return _deduplicate_fractional_positions(orbit), generator


class PooledIonicDistanceTarget:
    """Two narrowed ionic-distance distributions used for O construction.

    Si--O pools Si->O1..O4 and O->Si1..Si2 into one distribution.
    O--O uses the nearest periodic O neighbour for each O atom.  Both are fit
    by Gaussians whose widths are intentionally compressed relative to the
    measured training widths, suppressing elongated and diffuse environments.
    """

    def __init__(self, training_prepared, sio_sigma_scale=0.50,
                 oo_sigma_scale=0.50, bins=48, hard_sio_min=1.20,
                 hard_oo_min=1.20):
        self.bins = max(12, int(bins))
        self.hard_sio_min = float(hard_sio_min)
        self.hard_oo_min = float(hard_oo_min)
        sio_all, oo_all = [], []
        self.skipped_training = 0
        for item in training_prepared:
            if not item.get("ok", False):
                self.skipped_training += 1
                continue
            try:
                values = self.extract(item["si"], item["o"], item["cell"])
            except Exception:
                self.skipped_training += 1
                continue
            sio_all.extend(values["sio"].tolist())
            oo_all.extend(values["oo"].tolist())
        if not sio_all or not oo_all:
            raise RuntimeError("No valid pooled ionic-distance training environments.")
        self.training = {
            "sio": np.asarray(sio_all, dtype=float),
            "oo": np.asarray(oo_all, dtype=float),
        }
        scales = {"sio": float(sio_sigma_scale), "oo": float(oo_sigma_scale)}
        self.mu, self.training_sigma, self.sigma = {}, {}, {}
        self.edges, self.target_probability = {}, {}
        for name in ("sio", "oo"):
            arr = self.training[name]
            self.mu[name] = float(np.mean(arr))
            self.training_sigma[name] = max(float(np.std(arr)), 1.0e-3)
            self.sigma[name] = max(self.training_sigma[name] * scales[name], 0.03)
            low = min(float(np.min(arr)), self.mu[name] - 5.0 * self.sigma[name])
            high = max(float(np.max(arr)), self.mu[name] + 5.0 * self.sigma[name])
            self.edges[name] = np.linspace(low, high, self.bins + 1)
            centers = 0.5 * (self.edges[name][:-1] + self.edges[name][1:])
            prob = np.exp(-0.5 * ((centers - self.mu[name]) / self.sigma[name]) ** 2)
            prob /= prob.sum()
            self.target_probability[name] = prob
        self.accepted_counts = {
            name: np.zeros(self.bins, dtype=float) for name in ("sio", "oo")
        }
        self.accepted_values = {"sio": [], "oo": []}
        self.raw_counts = {name: np.zeros(self.bins, dtype=float) for name in ("sio", "oo")}
        self.raw_values = {"sio": [], "oo": []}
        self.accepted_structures = 0
        self.raw_structures = 0

    @staticmethod
    def _oo_nearest(o_frac, cell):
        o_frac = np.asarray(o_frac, dtype=float).reshape(-1, 3)
        shifts = np.asarray(
            [[i, j, k] for i in (-1, 0, 1)
             for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
        delta = o_frac[:, None, None, :] - o_frac[None, :, None, :] + shifts[None, None, :, :]
        cart = np.einsum("...i,ij->...j", delta, np.asarray(cell, dtype=float))
        dist = np.linalg.norm(cart, axis=-1)
        zero = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
        ids = np.arange(len(o_frac))
        dist[ids, ids, zero] = np.inf
        nearest = np.min(dist.reshape(len(o_frac), -1), axis=1)
        return nearest[np.isfinite(nearest)]

    @classmethod
    def extract(cls, si_frac, o_frac, cell):
        si_frac = np.asarray(si_frac, dtype=float).reshape(-1, 3)
        o_frac = np.asarray(o_frac, dtype=float).reshape(-1, 3)
        if len(si_frac) == 0 or len(o_frac) == 0:
            raise ValueError("Empty species in pooled ionic-distance extraction.")
        dist = _periodic_cross_distances_numpy(si_frac, o_frac, cell)
        si_all = np.sort(dist.reshape(len(si_frac), -1), axis=1)
        o_all = np.sort(np.transpose(dist, (1, 0, 2)).reshape(len(o_frac), -1), axis=1)
        sio_parts = [si_all[:, :min(4, si_all.shape[1])].reshape(-1),
                     o_all[:, :min(2, o_all.shape[1])].reshape(-1)]
        sio = np.concatenate(sio_parts)
        oo = cls._oo_nearest(o_frac, cell)
        return {"sio": sio[np.isfinite(sio)], "oo": oo[np.isfinite(oo)]}

    def _hist(self, name, values):
        arr = np.asarray(values, dtype=float)
        arr = np.clip(arr, self.edges[name][0] + 1e-12, self.edges[name][-1] - 1e-12)
        return np.histogram(arr, bins=self.edges[name])[0].astype(float)

    @staticmethod
    def _tv(counts, target):
        counts = np.asarray(counts, dtype=float)
        if counts.sum() <= 0:
            return float("nan")
        return float(0.5 * np.abs(counts / counts.sum() - target).sum())

    def describe(self, si_frac, o_frac, cell, update_raw=False):
        values = self.extract(si_frac, o_frac, cell)
        hists = {name: self._hist(name, values[name]) for name in ("sio", "oo")}
        z2 = {
            name: float(np.mean(((values[name] - self.mu[name]) / self.sigma[name]) ** 2))
            for name in ("sio", "oo")
        }
        tv = {
            name: self._tv(hists[name], self.target_probability[name])
            for name in ("sio", "oo")
        }
        desc = {"values": values, "hist": hists, "z2": z2, "tv": tv,
                "mean_z2": 0.5 * (z2["sio"] + z2["oo"]),
                "mean_tv": 0.5 * (tv["sio"] + tv["oo"])}
        if update_raw:
            for name in ("sio", "oo"):
                self.raw_counts[name] += hists[name]
                self.raw_values[name].extend(values[name].tolist())
            self.raw_structures += 1
        return desc

    def partial_score(self, si_frac, o_frac, cell, fraction):
        desc = self.describe(si_frac, o_frac, cell, update_raw=False)
        # Local probability field: average Gaussian negative log-likelihood.
        loss = 0.5 * desc["z2"]["sio"] + 0.5 * desc["z2"]["oo"]
        # Population-shape term prevents all candidates collapsing exactly at mu.
        loss += 0.20 * (desc["tv"]["sio"] + desc["tv"]["oo"])
        # Early partial states should not dominate merely because they contain
        # fewer O atoms; fraction only supplies a weak completion preference.
        loss += 0.05 * (1.0 - float(fraction))
        return float(loss)

    def commit(self, desc):
        for name in ("sio", "oo"):
            self.accepted_counts[name] += desc["hist"][name]
            self.accepted_values[name].extend(desc["values"][name].tolist())
        self.accepted_structures += 1

    def summary(self, scope="accepted"):
        source = self.accepted_values if scope == "accepted" else self.raw_values
        counts = self.accepted_counts if scope == "accepted" else self.raw_counts
        out = {}
        for name in ("sio", "oo"):
            arr = np.asarray(source[name], dtype=float)
            if arr.size:
                q = np.percentile(arr, [5, 50, 95])
                out[name] = {
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]),
                    "tv": self._tv(counts[name], self.target_probability[name]),
                }
            else:
                out[name] = {k: float("nan") for k in ("mean","std","q05","q50","q95","tv")}
        return out


class SequentialOxygenConstructor:
    """Construct oxygen sites from cached ionic-field orbit probes."""

    def __init__(self, ionic_target, n_o_max, beam_width=6,
                 candidates_per_site=32, max_skeletons=2, seed=42):
        self.ionic_target = ionic_target
        self.n_o_max = int(n_o_max)
        self.beam_width = max(1, int(beam_width))
        self.candidates_per_site = max(8, int(candidates_per_site))
        self.max_skeletons = max(1, int(max_skeletons))
        self.probe_pool_size = max(64, 3 * self.candidates_per_site)
        self.explore_fraction = 0.15
        self.temperature = 0.35
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

    def _orbit_is_allowed(self, orbit, si_frac, existing_o, cell):
        sio = _periodic_cross_distances_numpy(orbit, si_frac, cell)
        if float(np.min(sio)) <= self.ionic_target.hard_sio_min:
            return False
        if len(existing_o):
            oo = _periodic_cross_distances_numpy(orbit, existing_o, cell)
            if float(np.min(oo)) <= self.ionic_target.hard_oo_min:
                return False
        if len(orbit) > 0:
            self_oo = _periodic_cross_distances_numpy(orbit, orbit, cell)
            flat = self_oo.reshape(len(orbit), len(orbit), 27)
            shifts = np.asarray([[i, j, k] for i in (-1, 0, 1)
                                 for j in (-1, 0, 1) for k in (-1, 0, 1)])
            zero = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
            ids = np.arange(len(orbit))
            flat[ids, ids, zero] = np.inf
            if float(np.min(flat)) <= self.ionic_target.hard_oo_min:
                return False
        return True

    def _sobol_points(self, dof, count, scramble_seed):
        if dof == 0:
            return np.zeros((1, 0), dtype=float)
        engine = torch.quasirandom.SobolEngine(
            dimension=dof, scramble=True,
            seed=int(scramble_seed) % (2**31 - 1),
        )
        return engine.draw(int(count)).cpu().numpy().astype(float, copy=False)

    def _build_probe_pool(self, spg, wp_index, dof, si_frac, cell, cache_seed):
        requested = 1 if dof == 0 else self.probe_pool_size
        points = self._sobol_points(dof, requested, cache_seed)
        pool = []
        for free in points:
            try:
                orbit, _ = _expand_o_orbit(spg, wp_index, free)
            except Exception:
                continue
            if len(orbit) == 0 or not self._orbit_is_allowed(
                    orbit, si_frac, np.empty((0, 3), dtype=float), cell):
                continue
            prior = self.ionic_target.partial_score(
                si_frac, orbit, cell, fraction=0.0
            )
            pool.append((np.asarray(free, dtype=float), orbit, float(prior)))
        return pool, requested

    def _conditional_proxy(self, pool_item, existing_o, cell):
        _, orbit, prior = pool_item
        if len(existing_o) == 0:
            return prior
        oo = _periodic_cross_distances_numpy(orbit, existing_o, cell)
        nearest = np.min(oo.reshape(len(orbit), -1), axis=1)
        if float(np.min(nearest)) <= self.ionic_target.hard_oo_min:
            return float("inf")
        z2 = np.mean(((nearest - self.ionic_target.mu["oo"]) /
                      self.ionic_target.sigma["oo"]) ** 2)
        return float(prior + 0.5 * z2)

    def _select_cached_probes(self, pool, existing_o, cell):
        proxies = np.asarray([
            self._conditional_proxy(item, existing_o, cell) for item in pool
        ], dtype=float)
        valid_ids = np.flatnonzero(np.isfinite(proxies))
        if len(valid_ids) == 0:
            return []
        n_total = min(self.candidates_per_site, len(valid_ids))
        n_explore = min(max(1, int(round(n_total * self.explore_fraction))), n_total)
        n_guided = n_total - n_explore
        chosen = []
        if n_guided:
            scores = proxies[valid_ids]
            shifted = scores - np.min(scores)
            log_weights = -shifted / self.temperature

            # Stable weighted sampling without replacement.  Using
            # Generator.choice(..., replace=False, p=weights) can fail when
            # softmax underflow leaves fewer positive entries than requested.
            # Gumbel-top-k works directly with log weights and therefore does
            # not require exponentiation or a positive-probability count.
            uniforms = np.clip(
                self.rng.random(len(valid_ids)),
                np.finfo(float).tiny,
                1.0 - np.finfo(float).eps,
            )
            gumbels = -np.log(-np.log(uniforms))
            ranking = log_weights + gumbels
            if np.all(np.isfinite(ranking)):
                local_ids = np.argsort(ranking)[-n_guided:][::-1]
            else:
                # Defensive fallback for any unexpected numerical pathology.
                local_ids = self.rng.choice(
                    len(valid_ids), size=n_guided, replace=False
                )
            guided_ids = valid_ids[np.asarray(local_ids, dtype=int)]
            chosen.extend((int(i), "guided") for i in guided_ids)
        used = {idx for idx, _ in chosen}
        remaining = np.asarray([i for i in valid_ids if int(i) not in used], dtype=int)
        if n_explore and len(remaining):
            take = min(n_explore, len(remaining))
            explore_ids = self.rng.choice(remaining, size=take, replace=False)
            chosen.extend((int(i), "explore") for i in explore_ids)
        return [(pool[idx], source) for idx, source in chosen]

    def construct(self, global_row, si_frac, cell, skeleton_tokens,
                  progress_label="O search", verbose=True):
        spg = int(round(float(global_row["spg"])))
        group = Group(spg)
        unique_tokens = []
        for token in skeleton_tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)
            if len(unique_tokens) >= self.max_skeletons:
                break
        best_complete = None
        stats = {"attempted_states": 0, "guided_attempted": 0,
                 "explore_attempted": 0, "guided_valid": 0,
                 "explore_valid": 0, "guided_retained": 0,
                 "explore_retained": 0, "probe_total": 0,
                 "probe_valid": 0}
        probe_cache = {}
        search_start = time.perf_counter()
        if verbose:
            print(
                f"{progress_label}: start, skeletons={len(unique_tokens)}, "
                f"beam={self.beam_width}, cached_candidates/site={self.candidates_per_site}",
                flush=True,
            )

        for skeleton_id, token in enumerate(unique_tokens):
            try:
                wps = decode_wp_token(token, self.n_o_max, "O skeleton")
            except Exception:
                continue
            occupied = [(slot, int(wp)) for slot, wp in enumerate(wps) if int(wp) >= 0]
            if not occupied:
                continue
            occupied.sort(key=lambda item: int(group[item[1]].multiplicity), reverse=True)
            total_atoms = sum(int(group[wp].multiplicity) for _, wp in occupied)
            beam = [(0.0, [], np.empty((0, 3), dtype=float), "seed")]
            if verbose:
                print(
                    f"{progress_label}: skeleton {skeleton_id + 1}/{len(unique_tokens)}, "
                    f"independent_sites={len(occupied)}, expanded_O={total_atoms}",
                    flush=True,
                )

            for site_order, (slot, wp_index) in enumerate(occupied):
                site_start = time.perf_counter()
                dof = int(group[wp_index].get_dof())
                key = (wp_index, dof)
                if key not in probe_cache:
                    pool, requested = self._build_probe_pool(
                        spg, wp_index, dof, si_frac, cell,
                        self.seed + 1009 * skeleton_id + 97 * slot + 17 * site_order,
                    )
                    probe_cache[key] = pool
                    stats["probe_total"] += requested
                    stats["probe_valid"] += len(pool)
                pool = probe_cache[key]
                if not pool:
                    beam = []
                    if verbose:
                        print(
                            f"{progress_label}: site {site_order + 1}/{len(occupied)} "
                            f"wp={wp_index} dof={dof}, no valid cached probes",
                            flush=True,
                        )
                    break

                next_states = []
                for _, params_list, existing_o, _ in beam:
                    for pool_item, source in self._select_cached_probes(
                            pool, existing_o, cell):
                        stats[f"{source}_attempted"] += 1
                        free, orbit, _ = pool_item
                        if not self._orbit_is_allowed(orbit, si_frac, existing_o, cell):
                            continue
                        stats[f"{source}_valid"] += 1
                        new_o = orbit if len(existing_o) == 0 else np.concatenate(
                            [existing_o, orbit], axis=0
                        )
                        fraction = min(1.0, len(new_o) / max(total_atoms, 1))
                        score = self.ionic_target.partial_score(
                            si_frac, new_o, cell, fraction
                        )
                        padded = np.zeros(3, dtype=float)
                        padded[:dof] = np.asarray(free, dtype=float)[:dof] % 1.0
                        next_states.append((
                            score,
                            params_list + [(slot, wp_index, padded)],
                            new_o,
                            source,
                        ))
                        stats["attempted_states"] += 1
                if not next_states:
                    beam = []
                    if verbose:
                        print(
                            f"{progress_label}: site {site_order + 1}/{len(occupied)} "
                            f"wp={wp_index}, beam exhausted after "
                            f"{time.perf_counter() - site_start:.1f}s",
                            flush=True,
                        )
                    break
                next_states.sort(key=lambda item: item[0])
                beam = next_states[:self.beam_width]
                for state in beam:
                    if state[3] in ("guided", "explore"):
                        stats[f"{state[3]}_retained"] += 1
                if verbose:
                    print(
                        f"{progress_label}: site {site_order + 1}/{len(occupied)} "
                        f"wp={wp_index} dof={dof}, probes={len(pool)}, "
                        f"expanded_states={len(next_states)}, beam={len(beam)}, "
                        f"best={beam[0][0]:.4f}, "
                        f"elapsed={time.perf_counter() - site_start:.1f}s",
                        flush=True,
                    )

            if not beam:
                continue
            score, params_list, oxygen, _ = min(beam, key=lambda item: item[0])
            if best_complete is None or score < best_complete[0]:
                best_complete = (score, token, params_list, oxygen)

        elapsed = time.perf_counter() - search_start
        if best_complete is None:
            if verbose:
                print(f"{progress_label}: failed after {elapsed:.1f}s", flush=True)
            return None
        score, token, params_list, oxygen = best_complete
        if verbose:
            print(
                f"{progress_label}: complete, score={score:.4f}, "
                f"attempted_states={stats['attempted_states']}, elapsed={elapsed:.1f}s",
                flush=True,
            )
        record = {"o_skeleton_token": token}
        for slot in range(self.n_o_max):
            for j in range(3):
                record[f"o_u{j}_{slot}"] = -1.0
        for slot, _, padded in params_list:
            for j in range(3):
                record[f"o_u{j}_{slot}"] = float(padded[j])
        return {"score": float(score), "o_df": pd.DataFrame([record]),
                "oxygen": np.asarray(oxygen, dtype=np.float32),
                "attempted_states": int(stats["attempted_states"]),
                "proposal_stats": stats}


_O_WORKER_TARGET = None
_O_WORKER_CONFIG = None


def _init_o_worker(ionic_target, config):
    global _O_WORKER_TARGET, _O_WORKER_CONFIG
    _O_WORKER_TARGET = ionic_target
    _O_WORKER_CONFIG = dict(config)
    torch.set_num_threads(1)


def _search_one_si_framework(task):
    parent_index, global_row, si_frac, cell, tokens, task_seed = task
    started = time.perf_counter()
    try:
        constructor = SequentialOxygenConstructor(
            _O_WORKER_TARGET,
            n_o_max=_O_WORKER_CONFIG["n_o_max"],
            beam_width=_O_WORKER_CONFIG["beam_width"],
            candidates_per_site=_O_WORKER_CONFIG["candidates_per_site"],
            max_skeletons=_O_WORKER_CONFIG["max_skeletons"],
            seed=int(task_seed),
        )
        with open(os.devnull, "w") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            result = constructor.construct(
                global_row,
                np.asarray(si_frac, dtype=np.float32),
                np.asarray(cell, dtype=np.float32),
                list(tokens),
                progress_label="worker",
                verbose=False,
            )
        return {
            "parent_index": int(parent_index),
            "result": result,
            "elapsed": float(time.perf_counter() - started),
            "error": None,
        }
    except Exception as exc:
        return {
            "parent_index": int(parent_index),
            "result": None,
            "elapsed": float(time.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Wyckoff-parameterized factorized VAE for SiO2 LEGO-Xtal data"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--nbatch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--sample", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--context-end", type=float, default=0.8)
    parser.add_argument("--selection-block-size", type=int, default=64,
        help="Candidates committed per adaptive selector update.")
    parser.add_argument("--selection-min-score", type=float, default=0.0,
        help=argparse.SUPPRESS)
    parser.add_argument("--nn-bins", type=int, default=40,
        help="Histogram bins for per-Si periodic nearest-neighbour distances.")
    parser.add_argument("--nn-oversample-factor", type=float, default=1.0,
        help=argparse.SUPPRESS)
    parser.add_argument("--nn-round-size", type=int, default=2000,
        help="Maximum number of candidates generated in one online-selection round.")
    parser.add_argument("--nn-overfill-penalty", type=float, default=1.0)
    parser.add_argument("--nn-selection-temperature", type=float, default=0.01)
    parser.add_argument("--si-min-terminal", type=float, default=1.67,
        help="Broad zero-probability Si-Si safety floor in angstrom.")
    parser.add_argument("--nn-target-sigma-scale", type=float, default=1.0,
        help="Multiply the measured training per-Si NN standard deviation by this factor.")
    parser.add_argument("--nn-histogram-sigma-span", type=float, default=4.0,
        help="Minimum histogram span on each side of the Gaussian mean, in target sigma.")
    parser.add_argument("--gpu-contact-batch", type=int, default=128)
    parser.add_argument("--gpu-shift-chunk", type=int, default=27)
    parser.add_argument("--gpu-geometry-memory-gib", type=float, default=0.75,
        help="Approximate additional GPU-memory ceiling for NN screening.")
    parser.add_argument("--min-terminal-round-size", type=int, default=1000)
    parser.add_argument("--max-sample-rounds", type=int, default=100)
    parser.add_argument("--nn-image-range", type=int, default=1,
        help="Periodic translation range used for exact per-Si nearest neighbours.")
    parser.add_argument("--geometry-workers", type=int, default=3,
        help="CPU worker processes for symmetry expansion; 0 or 1 keeps serial behavior.")
    parser.add_argument("--o-shell-bins", type=int, default=40)
    parser.add_argument("--o-sequential-beam-width", type=int, default=6,
        help="Beam width for sequential independent-O-site construction.")
    parser.add_argument("--o-sequential-candidates-per-site", type=int, default=32,
        help="Cached Wyckoff orbit candidates retained per beam state and O site.")
    parser.add_argument("--o-sequential-skeletons", type=int, default=2,
        help="Distinct O skeletons sampled from the conditional decoder per fixed Si.")
    parser.add_argument("--o-sequential-oo-min", type=float, default=1.20,
        help="Catastrophic O-O distance floor used only as an exclusion constraint.")
    parser.add_argument("--o-sequential-sio-min", type=float, default=1.20,
        help="Catastrophic Si-O distance floor during constructive placement.")
    parser.add_argument("--ionic-sio-sigma-scale", type=float, default=0.50,
        help="Compress the pooled Si-O training standard deviation by this factor.")
    parser.add_argument("--ionic-oo-sigma-scale", type=float, default=0.50,
        help="Compress the nearest O-O training standard deviation by this factor.")
    parser.add_argument("--ionic-max-mean-z2", type=float, default=3.0,
        help="Maximum mean squared narrowed-Gaussian z score for final acceptance.")
    parser.add_argument("--ionic-max-tv", type=float, default=0.70,
        help="Maximum mean TV distance of the pooled Si-O and O-O histograms.")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(args.data)
    if args.epochs <= 0 or args.nbatch <= 0 or args.sample <= 0:
        raise ValueError("--epochs, --nbatch, and --sample must be positive.")
    if not 0.0 <= args.context_end <= 1.0:
        raise ValueError("--context-end must lie in [0, 1].")
    if args.selection_block_size <= 0:
        raise ValueError("--selection-block-size must be positive.")
    if args.nn_bins < 4 or args.nn_oversample_factor < 1:
        raise ValueError("NN bins must be >=4 and oversample factor >=1.")
    if args.nn_round_size <= 0 or args.max_sample_rounds <= 0:
        raise ValueError("Round size and max sample rounds must be positive.")
    if args.nn_image_range != 1:
        raise ValueError(
            "The exact periodic geometry uses 27 images; require --nn-image-range 1."
        )
    if args.geometry_workers < 0:
        raise ValueError("--geometry-workers must be nonnegative.")
    if args.o_shell_bins < 4:
        raise ValueError("--o-shell-bins must be at least 4.")
    if args.o_sequential_beam_width < 1 or args.o_sequential_candidates_per_site < 8:
        raise ValueError("Sequential O beam width must be positive and candidates/site >= 8.")
    if args.o_sequential_skeletons < 1:
        raise ValueError("--o-sequential-skeletons must be positive.")
    if args.o_sequential_oo_min <= 0 or args.o_sequential_sio_min <= 0:
        raise ValueError("Sequential O distance floors must be positive.")
    if args.ionic_sio_sigma_scale <= 0 or args.ionic_oo_sigma_scale <= 0:
        raise ValueError("Ionic Gaussian sigma scales must be positive.")
    if args.ionic_max_mean_z2 <= 0:
        raise ValueError("--ionic-max-mean-z2 must be positive.")
    if not 0.0 <= args.ionic_max_tv <= 1.0:
        raise ValueError("--ionic-max-tv must lie in [0,1].")
    if args.nn_overfill_penalty < 0 or args.nn_selection_temperature < 0:
        raise ValueError("NN selection penalties must be nonnegative.")
    if args.nn_target_sigma_scale <= 0 or args.nn_histogram_sigma_span <= 1:
        raise ValueError("NN Gaussian width settings must be positive.")
    if args.gpu_contact_batch < 1 or args.gpu_shift_chunk < 1:
        raise ValueError("GPU chunk sizes must be positive.")
    if not 0 < args.gpu_geometry_memory_gib <= 1.5:
        raise ValueError("GPU geometry memory must lie in (0, 1.5] GiB.")
    if args.min_terminal_round_size < 1:
        raise ValueError("Minimum terminal round size must be positive.")
    if args.si_min_terminal <= 0:
        raise ValueError("--si-min-terminal must be positive.")


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
    # Free Wyckoff parameters are continuous in [0, 1); -1 marks padding.
    discrete_coordinates = False

    global_discrete = ["spg"]
    if discrete_cell:
        global_discrete += ["a", "b", "c", "alpha", "beta", "gamma"]
    si_discrete = ["si_skeleton_token"]
    o_discrete = ["o_skeleton_token"]
    if discrete_coordinates:
        si_discrete += [c for c in si_df.columns if c != "si_skeleton_token"]
        o_discrete += [c for c in o_df.columns if c != "o_skeleton_token"]

    data_name = os.path.splitext(os.path.basename(args.data))[0]
    model_folder = os.path.join("models", data_name, "FactorizedVAE_v40_cached_ionic_field")
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
    print(
        "O construction: crystallographic skeleton prior with cached pooled "
        "Si-O and O-O ionic probability fields."
    )
    geometry_loss = None
    density_selector = OnlinePerSiNearestNeighborSelector(
        canonical_df,
        num_wps=num_wps,
        requested_structures=args.sample,
        nn_bins=args.nn_bins,
        shift_range=args.nn_image_range,
        overfill_penalty=args.nn_overfill_penalty,
        selection_temperature=args.nn_selection_temperature,
        safety_floor=args.si_min_terminal,
        target_sigma_scale=args.nn_target_sigma_scale,
        histogram_sigma_span=args.nn_histogram_sigma_span,
        seed=args.seed,
        gpu_device=("cuda" if torch.cuda.is_available() else "cpu"),
        gpu_contact_batch=args.gpu_contact_batch,
        gpu_shift_chunk=args.gpu_shift_chunk,
        gpu_max_memory_gib=args.gpu_geometry_memory_gib,
    )
    tq = density_selector.training_quantiles
    print(
        "Online per-Si nearest-neighbour selector: "
        f"bins={args.nn_bins}, adaptive_block={args.selection_block_size}, "
        f"round_size={args.nn_round_size}, safety_floor={args.si_min_terminal:.3f} A, "
        f"gpu_geometry_cap={args.gpu_geometry_memory_gib:.2f} GiB"
    )
    print(
        f"Geometry pipeline: workers={args.geometry_workers}, "
        f"exact_images={(2 * args.nn_image_range + 1) ** 3}, "
        f"O_search=beam{args.o_sequential_beam_width}, "
        f"cached_candidates/site={args.o_sequential_candidates_per_site}, "
        f"skeletons={args.o_sequential_skeletons}, "
        f"SiO_sigma_scale={args.ionic_sio_sigma_scale:g}, "
        f"OO_sigma_scale={args.ionic_oo_sigma_scale:g}, "
        f"ionic_max_z2={args.ionic_max_mean_z2:g}, "
        f"ionic_max_TV={args.ionic_max_tv:g}, "
        f"SiO/OO floors={args.o_sequential_sio_min:g}/"
        f"{args.o_sequential_oo_min:g} A"
    )
    print(
        "Per-Si NN training reference:\n"
        f"  valid structures={density_selector.training_structures}/{len(canonical_df)}, "
        f"skipped={density_selector.skipped_training}, "
        f"Si environments={len(density_selector.training_values)}, "
        f"mean Si/structure={density_selector.mean_training_nsi:.2f}\n"
        f"  mean/std={density_selector.target_mean:.4f}/"
        f"{np.std(density_selector.training_values):.4f} A\n"
        f"  q01/q05/q25/q50/q75/q95/q99="
        f"{tq['q01']:.4f}/{tq['q05']:.4f}/{tq['q25']:.4f}/"
        f"{tq['q50']:.4f}/{tq['q75']:.4f}/{tq['q95']:.4f}/{tq['q99']:.4f} A\n"
        f"  Gaussian target: mu={density_selector.target_mean:.4f} A, "
        f"sigma={density_selector.target_std:.4f} A"
    )

    t_train_expand = time.perf_counter()
    training_si = prepare_species_candidates(
        canonical_df, num_wps, SI_CN, workers=args.geometry_workers
    )
    training_o = prepare_species_candidates(
        canonical_df, num_wps, O_CN, workers=args.geometry_workers
    )
    training_prepared = combine_prepared_species(canonical_df, training_si, training_o)
    training_expand_seconds = time.perf_counter() - t_train_expand
    ionic_target = PooledIonicDistanceTarget(
        training_prepared,
        sio_sigma_scale=args.ionic_sio_sigma_scale,
        oo_sigma_scale=args.ionic_oo_sigma_scale,
        bins=args.o_shell_bins,
        hard_sio_min=args.o_sequential_sio_min,
        hard_oo_min=args.o_sequential_oo_min,
    )
    print(
        "Narrowed pooled ionic targets: "
        "Si-O = Si->O1..O4 plus O->Si1..Si2; O-O = nearest periodic O neighbour"
    )
    for name, label in (("sio", "Si-O pooled"), ("oo", "O-O nearest")):
        arr = ionic_target.training[name]
        q = np.percentile(arr, [5, 50, 95])
        print(
            f"  {label}: count={len(arr)}, training mean/std="
            f"{ionic_target.mu[name]:.4f}/{ionic_target.training_sigma[name]:.4f} A, "
            f"target sigma={ionic_target.sigma[name]:.4f} A, "
            f"q05/q50/q95={q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f} A"
        )

    sequential_o = SequentialOxygenConstructor(
        ionic_target,
        n_o_max=n_o_max,
        beam_width=args.o_sequential_beam_width,
        candidates_per_site=args.o_sequential_candidates_per_site,
        max_skeletons=args.o_sequential_skeletons,
        seed=args.seed + 29,
    )

    shell_batch_size = 16
    o_noise_dim = 32
    o_noise_scale = 1.0

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
        shell_loss_weight=0.0,
        shell_batch_size=shell_batch_size,
        o_noise_dim=o_noise_dim,
        o_noise_scale=o_noise_scale,
        cuda=torch.cuda.is_available(),
        verbose=True,
        folder=model_folder,
    )
    if model._device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(model._device)})")
    else:
        print("Device: cpu")

    model.fit(
        global_df,
        si_df,
        o_df,
        global_discrete_columns=global_discrete,
        si_discrete_columns=si_discrete,
        o_discrete_columns=o_discrete,
        geometry_data=geometry_loss,
    )

    print(
        "Training complete. Starting Si sampling and cached ionic-field O construction.",
        flush=True,
    )


    accepted_batches = []
    accepted_count = 0
    total_generated = 0
    total_o_proposals = 0
    rejected_overflow = 0
    rejected_multiplicity = 0
    rejected_reconstruction = 0
    nn_safety_valid = 0
    nn_selected_total = 0
    confirmed_si_total = 0
    exhausted_si_total = 0
    sequential_total_states = 0
    sequential_completed_total = 0
    mask_totals = {
        "invalid_space_group": 0,
        "no_compatible_si_skeleton": 0,
        "invalid_si_skeleton": 0,
        "no_packing_feasible_si_skeleton": 0,
        "packing_conditioned_si_coordinates": 0,
        "no_compatible_o_skeleton": 0,
    }
    timing = {
        "vae_si_generation": 0.0,
        "cpu_si_expansion": 0.0,
        "gpu_si_nn": 0.0,
        "si_selection_scoring": 0.0,
        "vae_o_generation": 0.0,
        "cpu_o_expansion": 0.0,
        "gpu_o_first_shell": 0.0,
        "o_selection_scoring": 0.0,
    }

    geometry_executor = (
        ProcessPoolExecutor(max_workers=args.geometry_workers)
        if args.geometry_workers > 1 else None
    )
    o_worker_config = {
        "n_o_max": n_o_max,
        "beam_width": args.o_sequential_beam_width,
        "candidates_per_site": args.o_sequential_candidates_per_site,
        "max_skeletons": args.o_sequential_skeletons,
    }
    o_executor = (
        ProcessPoolExecutor(
            max_workers=args.geometry_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_o_worker,
            initargs=(ionic_target, o_worker_config),
        )
        if args.geometry_workers > 1 else None
    )

    for sample_round in range(1, args.max_sample_rounds + 1):
        remaining = args.sample - accepted_count
        print(
            f"Sampling round {sample_round} start: remaining={remaining}, "
            f"cumulative={accepted_count}/{args.sample}",
            flush=True,
        )
        if remaining <= 0:
            break

        draw_size = min(
            args.nn_round_size,
            max(
                int(np.ceil(remaining * args.nn_oversample_factor)),
                min(args.min_terminal_round_size, args.nn_round_size),
            ),
        )

        # Stage 1: sample only G and Si.  Oxygen is not decoded yet.
        t_stage = time.perf_counter()
        si_state_all = model.sample_si(
            draw_size,
            temperature=args.temperature,
            hard=True,
            enforce_sio2_multiplicity=True,
            max_independent_sites=num_wps,
            si_skeleton_feasibility=None,
        )
        timing["vae_si_generation"] += time.perf_counter() - t_stage
        total_generated += draw_size
        multiplicity_valid = np.asarray(si_state_all["valid_mask"], dtype=bool)
        rejected_multiplicity += int((~multiplicity_valid).sum())
        for key in mask_totals:
            mask_totals[key] += int(si_state_all["stats"].get(key, 0))
        valid_state_indices = np.flatnonzero(multiplicity_valid)
        if len(valid_state_indices) == 0:
            print(
                f"Sampling round {sample_round}: generated={draw_size}, "
                "no multiplicity-valid Si candidates."
            )
            continue
        si_state = subset_si_state(si_state_all, valid_state_indices)

        # Reconstruct only Si rows, then expand/check Si geometry.
        t_stage = time.perf_counter()
        si_rows, si_source_positions, rejected_si_recon = blocks_to_si_rows(
            si_state["global_df"], si_state["si_df"], num_wps, n_si_max
        )
        rejected_reconstruction += rejected_si_recon
        if si_rows.empty:
            timing["cpu_si_expansion"] += time.perf_counter() - t_stage
            print(
                f"Sampling round {sample_round}: generated={draw_size}, "
                "no reconstructable Si candidates."
            )
            continue
        si_state = subset_si_state(si_state, si_source_positions)
        si_prepared = prepare_species_candidates(
            si_rows, num_wps, SI_CN,
            workers=args.geometry_workers,
            executor=geometry_executor,
        )
        si_combined = combine_prepared_species(si_rows, si_prepared)
        timing["cpu_si_expansion"] += time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        keep_rows, si_descriptions = density_selector.describe_batch(
            si_rows, prepared=si_combined
        )
        timing["gpu_si_nn"] += time.perf_counter() - t_stage
        safety_valid = len(si_descriptions)
        nn_safety_valid += safety_valid
        if not si_descriptions:
            print(
                f"Sampling round {sample_round}: generated={draw_size}, "
                f"Si_rows={len(si_rows)}, safety_valid=0."
            )
            continue

        # Confirm exactly the best Si frameworks currently needed.  They are
        # frozen and retained while oxygen is repeatedly sampled around them.
        t_stage = time.perf_counter()
        si_local = density_selector.select(
            si_descriptions,
            remaining=remaining,
            commit=False,
            block_size=args.selection_block_size,
            max_selected=remaining,
        )
        timing["si_selection_scoring"] += time.perf_counter() - t_stage
        confirmed_row_positions = np.asarray(
            [keep_rows[int(i)] for i in si_local], dtype=int
        )
        confirmed_si_total += len(confirmed_row_positions)
        if len(confirmed_row_positions) == 0:
            continue
        confirmed_state = subset_si_state(si_state, confirmed_row_positions)
        confirmed_si_geometry = [si_combined[int(i)] for i in confirmed_row_positions]
        confirmed_si_desc_indices = np.asarray(si_local, dtype=int)
        unresolved = list(range(len(confirmed_row_positions)))
        accepted_this_round = []
        sequential_attempted = 0
        sequential_completed = 0
        sequential_passed = 0
        sequential_scores = []
        proposal_stats_round = {
            "guided_attempted": 0, "explore_attempted": 0,
            "guided_valid": 0, "explore_valid": 0,
            "guided_retained": 0, "explore_retained": 0,
            "probe_total": 0, "probe_valid": 0,
        }

        t_stage = time.perf_counter()
        skeleton_result = model.sample_o_from_si(
            confirmed_state,
            proposals_per_si=args.o_sequential_skeletons,
            temperature=args.temperature,
            hard=True,
            enforce_sio2_multiplicity=True,
            max_independent_sites=num_wps,
        )
        timing["vae_o_generation"] += time.perf_counter() - t_stage
        skeleton_parent = np.asarray(skeleton_result["parent"], dtype=int)
        skeleton_valid = np.asarray(skeleton_result["valid_mask"], dtype=bool)
        skeleton_tokens = {i: [] for i in unresolved}
        for row_pos in np.flatnonzero(skeleton_valid):
            parent = int(skeleton_parent[row_pos])
            if parent in skeleton_tokens:
                token = str(
                    skeleton_result["o_df"].iloc[int(row_pos)]["o_skeleton_token"]
                )
                if token not in skeleton_tokens[parent]:
                    skeleton_tokens[parent].append(token)

        def process_framework_result(parent_global, result):
            nonlocal sequential_completed, sequential_attempted, sequential_passed
            if result is None:
                return False
            sequential_completed += 1
            sequential_attempted += int(result["attempted_states"])
            sequential_scores.append(float(result["score"]))
            for key, value in result.get("proposal_stats", {}).items():
                if key in proposal_stats_round:
                    proposal_stats_round[key] += int(value)

            one_global = confirmed_state["global_df"].iloc[[parent_global]].reset_index(drop=True)
            one_si = confirmed_state["si_df"].iloc[[parent_global]].reset_index(drop=True)
            full_rows, reconstruction_map, rej_over, rej_recon = blocks_to_lego_rows_with_map(
                one_global, one_si, result["o_df"], num_wps, n_si_max, n_o_max
            )
            nonlocal rejected_overflow, rejected_reconstruction
            rejected_overflow += rej_over
            rejected_reconstruction += rej_recon
            if full_rows.empty:
                return False

            t_desc = time.perf_counter()
            si_item = confirmed_si_geometry[parent_global]
            desc = ionic_target.describe(
                np.asarray(si_item["si"], dtype=np.float32),
                np.asarray(result["oxygen"], dtype=np.float32),
                np.asarray(si_item["cell"], dtype=np.float32),
                update_raw=True,
            )
            timing["gpu_o_first_shell"] += time.perf_counter() - t_desc
            if (
                accepted_count + len(accepted_this_round) < args.sample
                and desc["mean_z2"] <= args.ionic_max_mean_z2
                and desc["mean_tv"] <= args.ionic_max_tv
            ):
                quality = (-desc["mean_z2"], -desc["mean_tv"])
                accepted_this_round.append(
                    (parent_global, desc, full_rows.iloc[[0]].copy(), quality)
                )
                ionic_target.commit(desc)
                density_selector.commit(
                    si_descriptions,
                    [int(confirmed_si_desc_indices[parent_global])],
                )
                if parent_global in unresolved:
                    unresolved.remove(parent_global)
                sequential_passed += 1
                return True
            return False

        search_tasks = []
        for parent_global in list(unresolved):
            tokens = skeleton_tokens.get(parent_global, [])
            if not tokens:
                continue
            si_item = confirmed_si_geometry[parent_global]
            search_tasks.append((
                int(parent_global),
                confirmed_state["global_df"].iloc[parent_global].to_dict(),
                np.asarray(si_item["si"], dtype=np.float32),
                np.asarray(si_item["cell"], dtype=np.float32),
                list(tokens),
                int(args.seed + 1000003 * sample_round + 7919 * parent_global),
            ))

        manager_start = time.perf_counter()
        manager_completed = len(confirmed_row_positions) - len(search_tasks)
        manager_failed = manager_completed
        manager_errors = 0
        progress_step = max(1, len(confirmed_row_positions) // 100)
        print(
            f"Sampling round {sample_round}: O manager start, "
            f"frameworks={len(search_tasks)}, workers="
            f"{args.geometry_workers if o_executor is not None else 1}",
            flush=True,
        )

        if o_executor is None:
            _init_o_worker(ionic_target, o_worker_config)
            for task in search_tasks:
                payload = _search_one_si_framework(task)
                manager_completed += 1
                total_o_proposals += int(
                    payload["result"]["attempted_states"]
                    if payload["result"] is not None else 0
                )
                accepted = process_framework_result(
                    payload["parent_index"], payload["result"]
                )
                if not accepted:
                    manager_failed += 1
                if payload["error"]:
                    manager_errors += 1
                elapsed = time.perf_counter() - manager_start
                rate = manager_completed / max(elapsed, 1e-9)
                print(
                    f"Sampling round {sample_round}: O manager "
                    f"{manager_completed}/{len(confirmed_row_positions)}, "
                    f"accepted={sequential_passed}, failed={manager_failed}, "
                    f"running=0, rate={rate:.2f}/s, elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if accepted_count + len(accepted_this_round) >= args.sample:
                    break
        else:
            task_iter = iter(search_tasks)
            running = {}
            for _ in range(min(args.geometry_workers, len(search_tasks))):
                task = next(task_iter, None)
                if task is None:
                    break
                future = o_executor.submit(_search_one_si_framework, task)
                running[future] = task[0]

            last_reported = manager_completed
            stop_submitting = False
            while running:
                done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in done:
                    running.pop(future, None)
                    payload = future.result()
                    manager_completed += 1
                    result = payload["result"]
                    if result is not None:
                        total_o_proposals += int(result["attempted_states"])
                    accepted = process_framework_result(
                        payload["parent_index"], result
                    )
                    if not accepted:
                        manager_failed += 1
                    if payload["error"]:
                        manager_errors += 1

                    if accepted_count + len(accepted_this_round) >= args.sample:
                        stop_submitting = True
                    if not stop_submitting:
                        task = next(task_iter, None)
                        if task is not None:
                            new_future = o_executor.submit(_search_one_si_framework, task)
                            running[new_future] = task[0]

                if (
                    manager_completed - last_reported >= progress_step
                    or not running
                    or stop_submitting
                ):
                    elapsed = time.perf_counter() - manager_start
                    rate = manager_completed / max(elapsed, 1e-9)
                    print(
                        f"Sampling round {sample_round}: O manager "
                        f"{manager_completed}/{len(confirmed_row_positions)}, "
                        f"accepted={sequential_passed}, failed={manager_failed}, "
                        f"running={len(running)}, rate={rate:.2f}/s, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    last_reported = manager_completed

                if stop_submitting:
                    for future in running:
                        future.cancel()
                    break

        timing["cpu_o_expansion"] += time.perf_counter() - manager_start
        if manager_errors:
            print(
                f"Sampling round {sample_round}: O manager worker_errors={manager_errors}",
                flush=True,
            )

        sequential_total_states += sequential_attempted
        sequential_completed_total += sequential_completed
        exhausted_si_total += len(unresolved)
        if accepted_this_round:
            round_rows = pd.concat(
                [item[2] for item in accepted_this_round], ignore_index=True
            )
            # Never exceed the exact final output count.
            n_take = min(len(round_rows), args.sample - accepted_count)
            if n_take < len(round_rows):
                raise RuntimeError(
                    "Internal fixed-Si proposal logic accepted more structures "
                    "than the remaining output quota."
                )
            accepted_batches.append(round_rows.iloc[:n_take].copy())
            accepted_count += n_take
            nn_selected_total += n_take

        nn_summary = density_selector.accepted_summary()
        ionic_round = ionic_target.summary(scope="accepted")
        metric_line = (
            f"SiO={ionic_round['sio']['mean']:.3f}/"
            f"{ionic_round['sio']['std']:.3f},TV={ionic_round['sio']['tv']:.3f} "
            f"OO={ionic_round['oo']['mean']:.3f}/"
            f"{ionic_round['oo']['std']:.3f},TV={ionic_round['oo']['tv']:.3f}"
        )
        print(
            f"Sampling round {sample_round}: generated_Si={draw_size}, "
            f"Si_rows={len(si_rows)}, safety_valid={safety_valid}, "
            f"confirmed_Si={len(confirmed_row_positions)}, "
            f"O_accepted={len(accepted_this_round)}, O_exhausted={len(unresolved)}, "
            f"O_proposals_total={total_o_proposals}, "
            f"cumulative={accepted_count}/{args.sample}, "
            f"NN_TV={density_selector.histogram_distance():.4f}, "
            f"NN_mean/std={nn_summary['mean']:.3f}/{nn_summary['std']:.3f} A; "
            f"Ionic target accepted={ionic_target.accepted_structures}\n"
            f"  accepted pooled metrics: {metric_line}\n"
            f"  Sequential O construction: completed={sequential_completed}/{len(confirmed_row_positions)}, "
            f"passed={sequential_passed}, attempted_orbit_states={sequential_attempted}, "
            f"mean_constructive_score={np.mean(sequential_scores) if sequential_scores else np.nan:.4f}\n"
            f"  Guided proposals: attempted={proposal_stats_round['guided_attempted']}, "
            f"valid={proposal_stats_round['guided_valid']}, retained={proposal_stats_round['guided_retained']}; "
            f"exploratory attempted={proposal_stats_round['explore_attempted']}, "
            f"valid={proposal_stats_round['explore_valid']}, retained={proposal_stats_round['explore_retained']}; "
            f"Cached probes valid={proposal_stats_round['probe_valid']}/{proposal_stats_round['probe_total']}",
            flush=True,
        )

    if geometry_executor is not None:
        geometry_executor.shutdown(wait=True)
    if o_executor is not None:
        o_executor.shutdown(wait=True, cancel_futures=True)

    if accepted_count < args.sample:
        acceptance = accepted_count / total_generated if total_generated else 0.0
        raise RuntimeError(
            "Could not obtain the requested number of samples that fit the "
            f"original {num_wps}-slot LEGO layout after {args.max_sample_rounds} "
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
        f"  Rejected multiplicity-mask rows: {rejected_multiplicity}\n"
        f"    invalid sampled space group: {mask_totals['invalid_space_group']}\n"
        f"    no compatible Si skeleton: "
        f"{mask_totals['no_compatible_si_skeleton']}\n"
        f"    invalid sampled Si skeleton: {mask_totals['invalid_si_skeleton']}\n"
        f"    no Si skeleton passing external feasibility gate: "
        f"{mask_totals['no_packing_feasible_si_skeleton']}\n"
        f"    Si rows using externally conditioned coordinates: "
        f"{mask_totals['packing_conditioned_si_coordinates']}\n"
        f"    no compatible O skeleton: {mask_totals['no_compatible_o_skeleton']}\n"
        f"  Rejected slot-overflow combinations: {rejected_overflow}\n"
        f"  Rejected Wyckoff reconstructions: {rejected_reconstruction}\n"
        f"  Retained fraction: {accepted_count / total_generated:.1%}"
    )

    nn_diagnostics = os.path.join(
        model_folder, "online_per_si_nn_probability.csv"
    )
    density_selector.diagnostics_frame().to_csv(nn_diagnostics, index=False)
    raw_summary = density_selector.summarize_values(density_selector.raw_values)
    accepted_summary = density_selector.accepted_summary()
    print(
        "Online per-Si nearest-neighbour probability selection:\n"
        f"  Safety-valid candidates: {nn_safety_valid}\n"
        f"  Rejected terminal Si-Si contacts: {density_selector.rejected_safety}\n"
        f"  Failed Si geometry expansions: {density_selector.failed_geometry}\n"
        f"  NN-probability-selected structures: {nn_selected_total}\n"
        f"  Raw histogram TV distance: "
        f"{density_selector.histogram_distance(density_selector.raw_counts):.4f}\n"
        f"  Accepted histogram TV distance: {density_selector.histogram_distance():.4f}\n"
        f"  Training Gaussian mean/std: "
        f"{density_selector.target_mean:.4f}/{density_selector.target_std:.4f} A\n"
        f"  Raw NN mean/std: {raw_summary['mean']:.4f}/{raw_summary['std']:.4f} A\n"
        f"  Accepted NN mean/std: "
        f"{accepted_summary['mean']:.4f}/{accepted_summary['std']:.4f} A\n"
        f"  Accepted q05/q50/q95: "
        f"{accepted_summary['q05']:.4f}/{accepted_summary['q50']:.4f}/"
        f"{accepted_summary['q95']:.4f} A\n"
        f"  NN diagnostics: {nn_diagnostics}"
    )

    raw_ionic = ionic_target.summary(scope="raw")
    accepted_ionic = ionic_target.summary(scope="accepted")
    timing_rows = [
        {"stage": "training_cpu_species_expansion", "seconds": training_expand_seconds},
    ] + [{"stage": key, "seconds": value} for key, value in timing.items()]
    timing_path = os.path.join(model_folder, "sampling_stage_timing.csv")
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
    print(
        "Pooled ionic-distance construction:\n"
        f"  Confirmed Si frameworks: {confirmed_si_total}\n"
        f"  Completed constructive searches: {sequential_completed_total}\n"
        f"  Evaluated orbit states: {sequential_total_states}\n"
        f"  Accepted structures: {ionic_target.accepted_structures}\n"
        f"  Si-O target mu/sigma: {ionic_target.mu['sio']:.4f}/"
        f"{ionic_target.sigma['sio']:.4f} A\n"
        f"  O-O target mu/sigma: {ionic_target.mu['oo']:.4f}/"
        f"{ionic_target.sigma['oo']:.4f} A\n"
        f"  Raw Si-O mean/std/TV: {raw_ionic['sio']['mean']:.4f}/"
        f"{raw_ionic['sio']['std']:.4f}/{raw_ionic['sio']['tv']:.4f}\n"
        f"  Accepted Si-O mean/std/TV: {accepted_ionic['sio']['mean']:.4f}/"
        f"{accepted_ionic['sio']['std']:.4f}/{accepted_ionic['sio']['tv']:.4f}\n"
        f"  Raw O-O mean/std/TV: {raw_ionic['oo']['mean']:.4f}/"
        f"{raw_ionic['oo']['std']:.4f}/{raw_ionic['oo']['tv']:.4f}\n"
        f"  Accepted O-O mean/std/TV: {accepted_ionic['oo']['mean']:.4f}/"
        f"{accepted_ionic['oo']['std']:.4f}/{accepted_ionic['oo']['tv']:.4f}\n"
        f"  Timing: {timing_path}"
    )

    output = os.path.join(
        sample_folder,
        f"{data_name}-FactorizedVAE-v40-cached-ionic-field-seed{args.seed}-{args.sample}.csv",
    )
    synthetic.to_csv(output, index=False)
    final_model = os.path.join(model_folder, "models", "FactorizedVAE_final.pkl")
    model.save(final_model)
    print(f"Saved samples: {output}")
    print(f"Saved model: {final_model}")


if __name__ == "__main__":
    main()

