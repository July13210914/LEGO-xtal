#!/usr/bin/env python3
"""Train and sample a factorized SiO2 LEGO-Xtal VAE.

This version canonicalizes independent sites as Si(CN4) -> O(CN2) -> padding,
then trains a five-stage decoder:
    global:       space group and cell
    Si skeleton:  Si Wyckoff occupancy pattern
    Si parameters: site-wise free parameters conditioned on G, Si skeleton, and prior Si sites
    O skeleton:   O Wyckoff occupancy conditioned on the complete Si block
    O parameters: free Wyckoff parameters conditioned on sampled O skeleton

The sampled free parameters are mapped deterministically through PyXtal to exact
Wyckoff generating coordinates before the standard LEGO CSV is written.
Sampling applies hard space-group, stoichiometry, and slot-capacity masks.
"""

import argparse
import os
import re

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



class CrystallographicChemistryGeometry:
    """Differentiable local crystallographic-chemistry constraints.

    Space group and Wyckoff skeleton are teacher-forced only to expand predicted
    free parameters for the optional cross-sublattice coordination diagnostic.
    No Si-Si RDF objective is used in v15.
    """

    def __init__(
        self,
        canonical_df,
        n_si_max,
        n_o_max,
    ):
        self.rows = []
        self.n_si_max = int(n_si_max)
        self.n_o_max = int(n_o_max)
        for _, row in canonical_df.iterrows():
            spg = int(row["spg"])
            si_wps, o_wps = [], []
            for i in range(sum(c.startswith("wp") for c in canonical_df.columns)):
                wp = int(row.get(f"wp{i}", -1))
                cn = int(row.get(f"target_coord{i}", 0))
                if wp < 0:
                    continue
                (si_wps if cn == SI_CN else o_wps).append(wp)
            cell = self._cell_matrix(
                float(row["a"]), float(row["b"]), float(row["c"]),
                float(row["alpha"]), float(row["beta"]), float(row["gamma"]),
            )
            # Cache exact teacher symmetry-expanded positions for calibration.
            teacher_si, teacher_o = [], []
            for i in range(sum(c.startswith("wp") for c in canonical_df.columns)):
                wp_index = int(row.get(f"wp{i}", -1))
                cn = int(row.get(f"target_coord{i}", 0))
                if wp_index < 0:
                    continue
                generator = np.array(
                    [float(row[f"x{i}"]), float(row[f"y{i}"]), float(row[f"z{i}"])],
                    dtype=float,
                )
                wp = Group(spg)[wp_index]
                positions = []
                for op in wp.ops:
                    R, t = self._op_parts(op)
                    positions.append((R @ generator + t) % 1.0)
                (teacher_si if cn == SI_CN else teacher_o).append(
                    np.asarray(positions, dtype=np.float32)
                )
            teacher_si = np.concatenate(teacher_si, axis=0)
            teacher_o = np.concatenate(teacher_o, axis=0)
            self.rows.append((spg, si_wps, o_wps, cell, teacher_si, teacher_o))

        self._template_cache = {}

    @staticmethod
    def _cell_matrix(a, b, c, alpha, beta, gamma):
        # Input angles are radians in the continuous LEGO representation.
        ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sg = np.sin(gamma)
        if abs(sg) < 1e-8:
            sg = 1e-8
        volume_term = max(1e-12, 1 + 2*ca*cb*cg - ca*ca - cb*cb - cg*cg)
        return np.array([
            [a, 0.0, 0.0],
            [b*cg, b*sg, 0.0],
            [c*cb, c*(ca-cb*cg)/sg, c*np.sqrt(volume_term)/sg],
        ], dtype=np.float32)

    @staticmethod
    def _torch_cell_matrix(parameters):
        """Construct a differentiable cell matrix from [a,b,c,alpha,beta,gamma]."""
        a, b, c, alpha, beta, gamma = parameters.unbind(dim=-1)
        # Keep sampled lengths positive without flattening useful gradients.
        a = a.clamp_min(0.25)
        b = b.clamp_min(0.25)
        c = c.clamp_min(0.25)
        ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sg = torch.sin(gamma)
        sg_safe = torch.where(
            sg.abs() < 1e-4,
            torch.where(sg >= 0, torch.full_like(sg, 1e-4), torch.full_like(sg, -1e-4)),
            sg,
        )
        volume_term = 1 + 2 * ca * cb * cg - ca.square() - cb.square() - cg.square()
        volume_term = volume_term.clamp_min(1e-8)
        zeros = torch.zeros_like(a)
        row0 = torch.stack([a, zeros, zeros], dim=-1)
        row1 = torch.stack([b * cg, b * sg_safe, zeros], dim=-1)
        row2 = torch.stack([
            c * cb,
            c * (ca - cb * cg) / sg_safe,
            c * torch.sqrt(volume_term) / sg_safe,
        ], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

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
            delta = pos - base
            delta -= np.round(delta)
            A[:, j] = delta / eps
        b = base - A[:, :dof] @ u0
        mats, offs = [], []
        for op in wp.ops:
            R, t = self._op_parts(op)
            mats.append(R @ A)
            offs.append(R @ b + t)
        out = (np.asarray(mats, dtype=np.float32), np.asarray(offs, dtype=np.float32))
        self._template_cache[key] = out
        return out

    @staticmethod
    def _continuous_layout(transformer):
        layout = {}
        st = 0
        for info in transformer._column_transform_info_list:
            ed = st + info.output_dimensions
            if info.column_type == "continuous":
                layout[info.column_name] = (st, ed, info.transform)
            st = ed
        return layout

    @staticmethod
    def _gm_parameters(gm):
        bgm = getattr(gm, "_bgm_transformer", None)
        if bgm is None:
            bgm = getattr(gm, "_model", None)
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
                probs = torch.softmax(logits[:, st+1:ed], dim=-1)
                means, stds = self._gm_parameters(gm)
                means = torch.as_tensor(means, device=logits.device, dtype=logits.dtype)
                stds = torch.as_tensor(stds, device=logits.device, dtype=logits.dtype)
                raw_components = norm[:, None] * (4.0 * stds[None, :]) + means[None, :]
                slot.append((probs * raw_components).sum(dim=1))
            values.append(torch.stack(slot, dim=1))
        return torch.stack(values, dim=1)

    def _raw_continuous_columns(self, logits, transformer, column_names):
        """Differentiably invert selected continuous RDT columns."""
        layout = self._continuous_layout(transformer)
        values = []
        for name in column_names:
            if name not in layout:
                raise ValueError(
                    f"Crystallographic-chemistry loss requires continuous column {name!r}."
                )
            st, ed, gm = layout[name]
            norm = torch.tanh(logits[:, st])
            probs = torch.softmax(logits[:, st + 1:ed], dim=-1)
            means, stds = self._gm_parameters(gm)
            means = torch.as_tensor(means, device=logits.device, dtype=logits.dtype)
            stds = torch.as_tensor(stds, device=logits.device, dtype=logits.dtype)
            raw_components = norm[:, None] * (4.0 * stds[None, :]) + means[None, :]
            values.append((probs * raw_components).sum(dim=1))
        return torch.stack(values, dim=1)

    @staticmethod
    def _soft_cn(dist, onset, cutoff):
        """Unit bond count through onset, cosine taper to zero at cutoff."""
        if not 0.0 <= onset < cutoff:
            raise ValueError(
                f"Cross-loss onset must satisfy 0 <= onset < cutoff; got {onset}, {cutoff}."
            )
        taper = 0.5 * (
            torch.cos(torch.pi * (dist - onset) / (cutoff - onset)) + 1.0
        )
        return torch.where(
            dist <= onset,
            torch.ones_like(dist),
            torch.where(dist < cutoff, taper, torch.zeros_like(dist)),
        )

    @classmethod
    def _coordination_penalty(cls, sf, of, cell, onset, cutoff):
        delta = sf[:, None, :] - of[None, :, :]
        delta = delta - torch.round(delta)
        cart = torch.einsum("...i,ij->...j", delta, cell)
        dist = torch.linalg.norm(cart, dim=-1).clamp_min(1e-6)
        weights = cls._soft_cn(dist, float(onset), float(cutoff))
        cn_si = weights.sum(dim=1)
        cn_o = weights.sum(dim=0)
        return ((cn_si - 4.0) ** 2).mean() + ((cn_o - 2.0) ** 2).mean()


    def __call__(self, row_ids, global_logits, si_logits, o_logits,
                 global_transformer, si_transformer, o_transformer,
                 onset, cutoff, device):
        cell_parameters = self._raw_continuous_columns(
            global_logits,
            global_transformer,
            ["a", "b", "c", "alpha", "beta", "gamma"],
        )
        predicted_cells = self._torch_cell_matrix(cell_parameters)
        si_u = self._raw_parameters(si_logits, si_transformer, "si", self.n_si_max)
        o_u = self._raw_parameters(o_logits, o_transformer, "o", self.n_o_max)
        losses = []
        teacher_losses = []
        for local, rid in enumerate(row_ids.detach().cpu().tolist()):
            spg, si_wps, o_wps, cell_np, teacher_si_np, teacher_o_np = self.rows[int(rid)]
            si_pos, o_pos = [], []
            for slot, wp_index in enumerate(si_wps):
                M, q = self._site_template(spg, wp_index)
                M = torch.as_tensor(M, device=device, dtype=si_logits.dtype)
                q = torch.as_tensor(q, device=device, dtype=si_logits.dtype)
                si_pos.append(torch.einsum("aij,j->ai", M, si_u[local, slot]) + q)
            for slot, wp_index in enumerate(o_wps):
                M, q = self._site_template(spg, wp_index)
                M = torch.as_tensor(M, device=device, dtype=o_logits.dtype)
                q = torch.as_tensor(q, device=device, dtype=o_logits.dtype)
                o_pos.append(torch.einsum("aij,j->ai", M, o_u[local, slot]) + q)
            if not si_pos or not o_pos:
                continue
            sf = torch.cat(si_pos, dim=0)
            of = torch.cat(o_pos, dim=0)
            teacher_cell = torch.as_tensor(cell_np, device=device, dtype=sf.dtype)
            pred_cell = predicted_cells[local].to(dtype=sf.dtype)
            losses.append(
                self._coordination_penalty(sf, of, pred_cell, onset, cutoff)
            )

            teacher_si = torch.as_tensor(
                teacher_si_np, device=device, dtype=sf.dtype
            )
            teacher_o = torch.as_tensor(
                teacher_o_np, device=device, dtype=of.dtype
            )
            teacher_losses.append(
                self._coordination_penalty(
                    teacher_si, teacher_o, teacher_cell, onset, cutoff
                )
            )
        if not losses:
            zero = si_logits.sum() * 0.0
            return zero, zero
        return (
            torch.stack(losses).mean(),
            torch.stack(teacher_losses).mean(),
        )



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


def estimate_training_si_nn_gaussian(
    canonical_df,
    num_wps,
    histogram_bin=0.05,
    fit_window=0.75,
    sigma_floor=0.20,
    shift_range=2,
):
    """Fit a Gaussian to the first Si-Si nearest-neighbor peak."""
    nearest_all = []

    for _, row in canonical_df.iterrows():
        try:
            spg = int(row["spg"])
            group = Group(spg)
            cell = _cell_matrix_numpy(row)
            positions = []
            for slot in range(num_wps):
                if int(row[f"target_coord{slot}"]) != 4:
                    continue
                wp_index = int(row[f"wp{slot}"])
                if wp_index < 0 or wp_index >= len(group):
                    continue
                generator = np.asarray(
                    [row[f"x{slot}"], row[f"y{slot}"], row[f"z{slot}"]],
                    dtype=float,
                )
                wp = group[wp_index]
                positions.extend(
                    np.asarray([op.operate(generator) for op in wp.ops], dtype=float)
                    % 1.0
                )
            if len(positions) < 2:
                continue
            frac = np.asarray(positions, dtype=float)
            nearest = _periodic_nearest_numpy(
                frac, cell, shift_range=shift_range
            )
            nearest_all.extend(
                nearest[np.isfinite(nearest) & (nearest > 0.0)].tolist()
            )
        except Exception:
            continue

    values = np.asarray(nearest_all, dtype=float)
    if values.size < 10:
        raise ValueError(
            "Could not collect enough training Si nearest-neighbor distances."
        )

    lo, hi = float(values.min()), float(values.max())
    bins = np.arange(lo, hi + histogram_bin, histogram_bin)
    if bins.size < 3:
        bins = np.linspace(lo, hi + 1.0e-6, 4)
    counts, edges = np.histogram(values, bins=bins)
    peak_bin = int(np.argmax(counts))
    peak_center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])

    peak_values = values[np.abs(values - peak_center) <= fit_window]
    if peak_values.size < 10:
        peak_values = values

    mu = float(np.mean(peak_values))
    sigma = max(float(np.std(peak_values, ddof=1)), float(sigma_floor))

    return {
        "mu": mu,
        "sigma": sigma,
        "peak_center": float(peak_center),
        "n_all": int(values.size),
        "n_peak": int(peak_values.size),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


class SiGaussianPackingConditioner:
    """One narrow/broad Gaussian-mixture gate followed by smooth R0 refinement.

    The gate acts only before optimization.  It compares the initial structure
    dmin with the Gaussian fitted to the first training Si-Si nearest-neighbor
    peak.  Surviving candidates are smoothly reweighted and refined; there is no
    second hard acceptance cutoff after optimization.
    """

    def __init__(
        self,
        nn_mu,
        nn_sigma,
        restarts=4,
        optimize_steps=12,
        optimize_lr=0.03,
        trust_radius=0.10,
        gate_density_min=0.005,
        gate_density_max=0.05,
        broad_fraction=0.05,
        broad_sigma_scale=3.0,
        logit_beta=1.0,
        final_noise_std=0.01,
        shift_range=2,
        seed=42,
        device="cpu",
        progress_every=25,
    ):
        if nn_sigma <= 0:
            raise ValueError("nn_sigma must be positive.")
        if restarts < 1 or optimize_steps < 0:
            raise ValueError("restarts must be positive and optimize_steps nonnegative.")
        if optimize_lr <= 0 or trust_radius < 0:
            raise ValueError("optimize_lr must be positive and trust_radius nonnegative.")
        if not 0 < gate_density_min <= gate_density_max <= 1:
            raise ValueError("Gate density fractions must satisfy 0 < min <= max <= 1.")
        if not 0 <= broad_fraction <= 1:
            raise ValueError("broad_fraction must lie in [0,1].")
        if broad_sigma_scale < 1.0:
            raise ValueError("broad_sigma_scale must be at least 1.")
        if logit_beta < 0 or final_noise_std < 0:
            raise ValueError("logit_beta and final_noise_std must be nonnegative.")

        self.nn_mu = float(nn_mu)
        self.nn_sigma = float(nn_sigma)
        self.restarts = int(restarts)
        self.optimize_steps = int(optimize_steps)
        self.optimize_lr = float(optimize_lr)
        self.trust_radius = float(trust_radius)
        self.gate_density_min = float(gate_density_min)
        self.gate_density_max = float(gate_density_max)
        self.broad_fraction = float(broad_fraction)
        self.broad_sigma_scale = float(broad_sigma_scale)
        self.logit_beta = float(logit_beta)
        self.final_noise_std = float(final_noise_std)
        self.shift_range = int(shift_range)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.progress_every = int(progress_every)
        self._group_cache = {}
        self._template_cache = {}

    def _group(self, spg):
        spg = int(spg)
        if spg not in self._group_cache:
            self._group_cache[spg] = Group(spg)
        return self._group_cache[spg]

    @staticmethod
    def _op_parts(op):
        rot = getattr(op, "rotation_matrix", None)
        trans = getattr(op, "translation_vector", None)
        if rot is None or trans is None:
            affine = np.asarray(op.affine_matrix, dtype=float)
            rot, trans = affine[:3, :3], affine[:3, 3]
        return np.asarray(rot, dtype=float), np.asarray(trans, dtype=float)

    def _site_template(self, spg, wp_index):
        key = (int(spg), int(wp_index))
        if key in self._template_cache:
            return self._template_cache[key]

        wp = self._group(spg)[int(wp_index)]
        dof = int(wp.get_dof())
        if dof == 0:
            u0 = np.zeros(0, dtype=float)
            base = np.asarray(wp.get_position_from_free_xyzs(u0), dtype=float)
            jacobian = np.zeros((3, 3), dtype=float)
        else:
            u0 = np.full(dof, 0.271, dtype=float)
            base = np.asarray(wp.get_position_from_free_xyzs(u0), dtype=float)
            jacobian = np.zeros((3, 3), dtype=float)
            eps = 1.0e-5
            for axis in range(dof):
                shifted_u = u0.copy()
                shifted_u[axis] += eps
                shifted = np.asarray(
                    wp.get_position_from_free_xyzs(shifted_u), dtype=float
                )
                delta = shifted - base
                delta -= np.round(delta)
                jacobian[:, axis] = delta / eps

        intercept = base - jacobian[:, :dof] @ u0
        matrices, offsets = [], []
        for op in wp.ops:
            rotation, translation = self._op_parts(op)
            matrices.append(rotation @ jacobian)
            offsets.append(rotation @ intercept + translation)

        result = (
            torch.as_tensor(np.asarray(matrices, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(offsets, dtype=np.float32), device=self.device),
            dof,
        )
        self._template_cache[key] = result
        return result

    def _expanded_positions(self, spg, wp_indices, parameters):
        positions = []
        for site_index, wp_index in enumerate(wp_indices):
            matrices, offsets, _ = self._site_template(spg, wp_index)
            positions.append(
                torch.einsum("aij,j->ai", matrices, parameters[site_index])
                + offsets
            )
        return torch.cat(positions, dim=0)

    def _nearest_distances(self, frac, cell):
        r = self.shift_range
        shifts = torch.tensor(
            [
                [i, j, k]
                for i in range(-r, r + 1)
                for j in range(-r, r + 1)
                for k in range(-r, r + 1)
            ],
            device=frac.device,
            dtype=frac.dtype,
        )
        delta = frac[:, None, None, :] - frac[None, :, None, :]
        delta = delta + shifts[None, None, :, :]
        cart = torch.einsum("...i,ij->...j", delta, cell)
        distances = torch.linalg.vector_norm(cart, dim=-1)
        zero_shift = int(((shifts == 0).all(dim=1)).nonzero()[0].item())
        ids = torch.arange(frac.shape[0], device=frac.device)
        distances = distances.clone()
        distances[ids, ids, zero_shift] = float("inf")
        return distances.reshape(frac.shape[0], -1).amin(dim=1)

    def _gaussian_energy(self, nearest, effective_sigma):
        z = (nearest - self.nn_mu) / effective_sigma
        # Smooth-L1 prevents a tiny number of severe contacts from dominating
        # every gradient while still exerting a strong restoring force.
        return torch.nn.functional.smooth_l1_loss(
            z, torch.zeros_like(z), beta=1.0, reduction="mean"
        )

    def _relative_density(self, dmin, effective_sigma):
        z = (float(dmin) - self.nn_mu) / effective_sigma
        return float(np.exp(-0.5 * z * z))

    @staticmethod
    def _periodic_delta(current, initial):
        delta = current - initial
        return delta - torch.round(delta)

    def _project_(self, parameters, initial, dof_mask):
        with torch.no_grad():
            delta = self._periodic_delta(parameters, initial)
            norms = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            scale = torch.clamp(
                self.trust_radius / norms.clamp_min(1.0e-12), max=1.0
            )
            delta = delta * scale
            parameters.copy_((initial + delta).remainder(1.0))
            parameters.mul_(dof_mask)

    def _initial_parameters(self, dofs):
        values = np.zeros((len(dofs), 3), dtype=np.float32)
        for site, dof in enumerate(dofs):
            if dof > 0:
                # Uniform crystallographic exploration plus a Gaussian
                # perturbation, wrapped periodically.
                values[site, :dof] = (
                    self.rng.random(dof)
                    + self.rng.normal(0.0, 0.05, size=dof)
                ) % 1.0
        return values

    def _condition_candidate(self, spg, token, cell_np):
        wp_indices = [int(wp) for wp in token if int(wp) >= 0]
        if not wp_indices:
            return None

        group = self._group(spg)
        dofs = []
        for wp_index in wp_indices:
            if wp_index >= len(group):
                return None
            _, _, dof = self._site_template(spg, wp_index)
            dofs.append(dof)

        dof_mask = torch.zeros(
            (len(wp_indices), 3), device=self.device, dtype=torch.float32
        )
        for site, dof in enumerate(dofs):
            dof_mask[site, :dof] = 1.0

        cell = torch.as_tensor(cell_np, device=self.device, dtype=torch.float32)

        # Sample the chemistry-prior component once per candidate skeleton.
        # Most candidates use the fitted narrow Gaussian; a controlled minority
        # uses a broader Gaussian that remains centered on the same NN peak.
        broad_component = bool(self.rng.random() < self.broad_fraction)
        effective_sigma = (
            self.nn_sigma * self.broad_sigma_scale
            if broad_component
            else self.nn_sigma
        )

        gate_threshold = float(
            np.exp(
                self.rng.uniform(
                    np.log(self.gate_density_min),
                    np.log(self.gate_density_max),
                )
            )
        )

        passing_starts = []
        best_raw = None

        for _ in range(self.restarts):
            initial_np = self._initial_parameters(dofs)
            initial = torch.as_tensor(
                initial_np, device=self.device, dtype=torch.float32
            )
            with torch.no_grad():
                frac = self._expanded_positions(spg, wp_indices, initial)
                nearest = self._nearest_distances(frac, cell)
                dmin = float(nearest.min().cpu())
                density = self._relative_density(dmin, effective_sigma)
                energy = float(
                    self._gaussian_energy(nearest, effective_sigma).cpu()
                )

            if density >= gate_threshold:
                passing_starts.append((initial_np, energy, dmin, density))
            if best_raw is None or energy < best_raw[1]:
                best_raw = (initial_np, energy, dmin, density)

        if not passing_starts:
            return None

        best = None
        for initial_np, raw_energy, raw_dmin, density in passing_starts:
            initial = torch.tensor(
                initial_np,
                device=self.device,
                dtype=torch.float32,
            )
            parameters = initial.detach().clone().requires_grad_(True)

            if sum(dofs) > 0 and self.optimize_steps > 0:
                optimizer = torch.optim.Adam(
                    [parameters], lr=self.optimize_lr
                )
                for _ in range(self.optimize_steps):
                    optimizer.zero_grad()
                    frac = self._expanded_positions(
                        spg, wp_indices, parameters
                    )
                    nearest = self._nearest_distances(frac, cell)
                    energy = self._gaussian_energy(
                        nearest, effective_sigma
                    )
                    if not energy.requires_grad or not torch.isfinite(energy):
                        break
                    energy.backward()
                    torch.nn.utils.clip_grad_norm_([parameters], 5.0)
                    optimizer.step()
                    self._project_(parameters, initial, dof_mask)

            with torch.no_grad():
                # Preserve a small Gaussian source of diversity after the
                # smooth drift; there is intentionally no second hard gate.
                if self.final_noise_std > 0 and sum(dofs) > 0:
                    noise = torch.randn_like(parameters) * self.final_noise_std
                    parameters.add_(noise * dof_mask)
                    self._project_(parameters, initial, dof_mask)

                frac = self._expanded_positions(
                    spg, wp_indices, parameters
                )
                nearest = self._nearest_distances(frac, cell)
                final_energy = float(
                    self._gaussian_energy(
                        nearest, effective_sigma
                    ).cpu()
                )
                final_dmin = float(nearest.min().cpu())
                final_free = parameters.cpu().numpy().copy()

            candidate = {
                "free": final_free,
                "energy": final_energy,
                "dmin": final_dmin,
                "raw_dmin": raw_dmin,
                "raw_density": density,
                "broad_component": broad_component,
                "effective_sigma": effective_sigma,
                "gate_threshold": gate_threshold,
            }
            if best is None or final_energy < best["energy"]:
                best = candidate

        return best

    def __call__(
        self,
        global_batch_df,
        parsed_si_tokens,
        preallowed,
        si_category_logits,
        topk,
    ):
        n_rows, n_categories = preallowed.shape
        n_sites = max(len(token) for token in parsed_si_tokens)
        allowed = np.zeros_like(preallowed, dtype=bool)
        best_free = np.full(
            (n_rows, n_categories, n_sites, 3),
            np.nan,
            dtype=np.float32,
        )
        best_score = np.full(
            (n_rows, n_categories), np.inf, dtype=np.float32
        )
        best_dmin = np.full(
            (n_rows, n_categories), np.nan, dtype=np.float32
        )
        logit_bias = np.full(
            (n_rows, n_categories), -1.0e9, dtype=np.float32
        )

        tested = passed = broad_passed = 0
        topk = max(1, min(int(topk), n_categories))

        for row_index, (_, row) in enumerate(global_batch_df.iterrows()):
            if (
                self.progress_every > 0
                and row_index > 0
                and row_index % self.progress_every == 0
            ):
                print(
                    f"  Si Gaussian conditioning: {row_index}/{n_rows} rows; "
                    f"rows with candidate={int(allowed[:row_index].any(axis=1).sum())}"
                )

            try:
                spg = int(round(float(row["spg"])))
                cell = _cell_matrix_numpy(row)
            except Exception:
                continue
            if not 1 <= spg <= 230:
                continue

            candidate_ids = np.where(preallowed[row_index])[0]
            if candidate_ids.size == 0:
                continue
            ranked = candidate_ids[
                np.argsort(si_category_logits[row_index, candidate_ids])[::-1]
            ][:topk]

            for category in ranked:
                tested += 1
                result = self._condition_candidate(
                    spg, parsed_si_tokens[int(category)], cell
                )
                if result is None:
                    continue
                passed += 1
                broad_passed += int(result["broad_component"])
                allowed[row_index, int(category)] = True
                free = result["free"]
                best_free[
                    row_index, int(category), : free.shape[0], :
                ] = free
                best_score[row_index, int(category)] = result["energy"]
                best_dmin[row_index, int(category)] = result["dmin"]
                logit_bias[row_index, int(category)] = (
                    -self.logit_beta * result["energy"]
                )

        print(
            "  Si Gaussian gate: "
            f"tested={tested}, passed={passed}, "
            f"broad_component_passed={broad_passed}; "
            f"mu={self.nn_mu:.3f} A, sigma={self.nn_sigma:.3f} A, "
            f"broad_sigma={self.nn_sigma * self.broad_sigma_scale:.3f} A"
        )

        return {
            "allowed": allowed,
            "best_free": best_free,
            "best_score": best_score,
            "best_dmin": best_dmin,
            "logit_bias": logit_bias,
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
    parser.add_argument(
        "--cross-loss-weight", type=float, default=0.1,
        help=("Weight of the differentiable Si->O4/O->Si2 first-shell penalty. "
              "Default 0.1 is conservative; use 0 to disable."),
    )
    parser.add_argument(
        "--cross-onset", type=float, default=2.0,
        help="Distance below which each Si-O pair contributes exactly one to soft CN.",
    )
    parser.add_argument("--cross-cutoff", type=float, default=2.4)
    parser.add_argument("--cross-batch-size", type=int, default=16)
    parser.add_argument(
        "--si-nn-hist-bin", type=float, default=0.05,
        help="Histogram bin width for locating the first training Si-Si NN peak.",
    )
    parser.add_argument(
        "--si-nn-fit-window", type=float, default=0.75,
        help="Half-width around the first NN peak used for Gaussian fitting.",
    )
    parser.add_argument(
        "--si-nn-sigma-floor", type=float, default=0.20,
        help="Minimum Gaussian width in angstrom.",
    )
    parser.add_argument(
        "--si-gate-density-min", type=float, default=0.005,
        help="Minimum sampled relative Gaussian-density threshold.",
    )
    parser.add_argument(
        "--si-gate-density-max", type=float, default=0.05,
        help="Maximum sampled relative Gaussian-density threshold.",
    )
    parser.add_argument(
        "--si-gate-broad-fraction", type=float, default=0.05,
        help=(
            "Fraction of candidate skeletons using the broad chemistry-centered "
            "Gaussian component."
        ),
    )
    parser.add_argument(
        "--si-gate-broad-sigma-scale", type=float, default=3.0,
        help="Width multiplier for the broad Gaussian component.",
    )
    parser.add_argument(
        "--si-pack-restarts", type=int, default=4,
        help="Random Gaussian-perturbed R0 starts per candidate skeleton.",
    )
    parser.add_argument(
        "--si-pack-opt-steps", type=int, default=12,
        help="Short smooth packing-refinement steps; zero disables refinement.",
    )
    parser.add_argument(
        "--si-pack-opt-lr", type=float, default=0.03,
        help="Adam learning rate for smooth R0 refinement.",
    )
    parser.add_argument(
        "--si-pack-trust-radius", type=float, default=0.10,
        help="Maximum periodic displacement from each sampled R0 start.",
    )
    parser.add_argument(
        "--si-pack-logit-beta", type=float, default=1.0,
        help="Strength of smooth W0 logit reweighting by packing energy.",
    )
    parser.add_argument(
        "--si-pack-final-noise", type=float, default=0.01,
        help="Gaussian free-coordinate noise retained after smooth refinement.",
    )
    parser.add_argument(
        "--si-pack-shift-range", type=int, default=2,
        help="Periodic image range used for Si nearest-neighbor distances.",
    )
    parser.add_argument("--si-pack-topk", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(args.data)
    if args.epochs <= 0 or args.nbatch <= 0 or args.sample <= 0:
        raise ValueError("--epochs, --nbatch, and --sample must be positive.")
    if not 0.0 <= args.context_end <= 1.0:
        raise ValueError("--context-end must lie in [0, 1].")
    if args.cross_loss_weight < 0 or args.cross_batch_size <= 0:
        raise ValueError("Cross-loss weight must be nonnegative and batch size positive.")
    if not 0.0 <= args.cross_onset < args.cross_cutoff:
        raise ValueError("Require 0 <= --cross-onset < --cross-cutoff.")
    if args.si_nn_hist_bin <= 0 or args.si_nn_fit_window <= 0:
        raise ValueError("NN histogram bin and fit window must be positive.")
    if args.si_nn_sigma_floor <= 0:
        raise ValueError("--si-nn-sigma-floor must be positive.")
    if not (
        0 < args.si_gate_density_min
        <= args.si_gate_density_max
        <= 1
    ):
        raise ValueError(
            "Require 0 < gate-density-min <= gate-density-max <= 1."
        )
    if not 0 <= args.si_gate_broad_fraction <= 1:
        raise ValueError("--si-gate-broad-fraction must lie in [0,1].")
    if args.si_gate_broad_sigma_scale < 1.0:
        raise ValueError("--si-gate-broad-sigma-scale must be at least 1.")
    if args.si_pack_restarts < 1 or args.si_pack_opt_steps < 0:
        raise ValueError("Packing restarts must be positive and steps nonnegative.")
    if args.si_pack_opt_lr <= 0 or args.si_pack_trust_radius < 0:
        raise ValueError("Packing LR must be positive and trust radius nonnegative.")
    if args.si_pack_logit_beta < 0 or args.si_pack_final_noise < 0:
        raise ValueError("Logit beta and final noise must be nonnegative.")
    if args.si_pack_shift_range < 1 or args.si_pack_topk < 1:
        raise ValueError("Shift range and top-k must be positive.")


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
    model_folder = os.path.join("models", data_name, "FactorizedVAE_v22")
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
    print(f"Coordination loss: weight={args.cross_loss_weight}, "
          f"onset={args.cross_onset} A, cutoff={args.cross_cutoff} A, "
          f"rows/batch={args.cross_batch_size}")
    nn_prior = estimate_training_si_nn_gaussian(
        canonical_df,
        num_wps,
        histogram_bin=args.si_nn_hist_bin,
        fit_window=args.si_nn_fit_window,
        sigma_floor=args.si_nn_sigma_floor,
        shift_range=args.si_pack_shift_range,
    )
    print(
        "Si factorization: P(G), Gaussian-gated and smoothly reweighted "
        "P(W_Si|G,C_NN), noisy-refined P(R_Si|G,W_Si,C_NN)"
    )
    print(
        "Training Si NN Gaussian: "
        f"peak={nn_prior['peak_center']:.3f} A, "
        f"mu={nn_prior['mu']:.3f} A, "
        f"sigma={nn_prior['sigma']:.3f} A; "
        f"all q05/q50/q95={nn_prior['q05']:.3f}/"
        f"{nn_prior['q50']:.3f}/{nn_prior['q95']:.3f} A; "
        f"Npeak/Nall={nn_prior['n_peak']}/{nn_prior['n_all']}"
    )
    print(
        "Si Gaussian gate/refinement: "
        f"density threshold={args.si_gate_density_min:g}-"
        f"{args.si_gate_density_max:g}; "
        f"broad_fraction={args.si_gate_broad_fraction:.3f}; "
        f"broad_sigma_scale={args.si_gate_broad_sigma_scale:.2f}; "
        f"restarts={args.si_pack_restarts}; "
        f"steps={args.si_pack_opt_steps}; "
        f"trust={args.si_pack_trust_radius:.3f}; "
        f"final_noise={args.si_pack_final_noise:.3f}; "
        f"topk={args.si_pack_topk}"
    )
    geometry_loss = CrystallographicChemistryGeometry(
        canonical_df,
        n_si_max,
        n_o_max,
    )

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
        cross_loss_weight=args.cross_loss_weight,
        cross_onset=args.cross_onset,
        cross_cutoff=args.cross_cutoff,
        cross_batch_size=args.cross_batch_size,
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

    si_packing_feasibility = SiGaussianPackingConditioner(
        nn_mu=nn_prior["mu"],
        nn_sigma=nn_prior["sigma"],
        restarts=args.si_pack_restarts,
        optimize_steps=args.si_pack_opt_steps,
        optimize_lr=args.si_pack_opt_lr,
        trust_radius=args.si_pack_trust_radius,
        gate_density_min=args.si_gate_density_min,
        gate_density_max=args.si_gate_density_max,
        broad_fraction=args.si_gate_broad_fraction,
        broad_sigma_scale=args.si_gate_broad_sigma_scale,
        logit_beta=args.si_pack_logit_beta,
        final_noise_std=args.si_pack_final_noise,
        shift_range=args.si_pack_shift_range,
        seed=args.seed,
        device="cpu",
    )


    accepted_batches = []
    accepted_count = 0
    total_generated = 0
    rejected_overflow = 0
    rejected_multiplicity = 0
    rejected_reconstruction = 0
    mask_totals = {
        "invalid_space_group": 0,
        "no_compatible_si_skeleton": 0,
        "invalid_si_skeleton": 0,
        "no_packing_feasible_si_skeleton": 0,
        "packing_conditioned_si_coordinates": 0,
        "no_compatible_o_skeleton": 0,
    }
    max_sample_rounds = 20

    for sample_round in range(1, max_sample_rounds + 1):
        remaining = args.sample - accepted_count
        if remaining <= 0:
            break

        draw_size = max(remaining, int(np.ceil(remaining * 1.25)))
        (
            sampled_global,
            sampled_si,
            sampled_o,
            multiplicity_valid,
            mask_stats,
        ) = model.sample(
            draw_size,
            temperature=args.temperature,
            hard=True,
            enforce_sio2_multiplicity=True,
            max_independent_sites=num_wps,
            si_skeleton_feasibility=si_packing_feasibility,
            si_feasibility_topk=args.si_pack_topk,
        )
        total_generated += draw_size
        rejected_multiplicity += int((~multiplicity_valid).sum())
        for key in mask_totals:
            mask_totals[key] += int(mask_stats.get(key, 0))

        sampled_global = sampled_global.loc[multiplicity_valid].reset_index(drop=True)
        sampled_si = sampled_si.loc[multiplicity_valid].reset_index(drop=True)
        sampled_o = sampled_o.loc[multiplicity_valid].reset_index(drop=True)

        valid_batch, rejected, rejected_recon = blocks_to_lego_rows(
            sampled_global,
            sampled_si,
            sampled_o,
            num_wps,
            n_si_max,
            n_o_max,
        )
        rejected_overflow += rejected
        rejected_reconstruction += rejected_recon
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
        f"  Rejected multiplicity-mask rows: {rejected_multiplicity}\n"
        f"    invalid sampled space group: {mask_totals['invalid_space_group']}\n"
        f"    no compatible Si skeleton: "
        f"{mask_totals['no_compatible_si_skeleton']}\n"
        f"    invalid sampled Si skeleton: {mask_totals['invalid_si_skeleton']}\n"
        f"    no Si skeleton passing Gaussian gate: "
        f"{mask_totals['no_packing_feasible_si_skeleton']}\n"
        f"    Si rows using Gaussian-conditioned R0: "
        f"{mask_totals['packing_conditioned_si_coordinates']}\n"
        f"    no compatible O skeleton: {mask_totals['no_compatible_o_skeleton']}\n"
        f"  Rejected slot-overflow combinations: {rejected_overflow}\n"
        f"  Rejected Wyckoff reconstructions: {rejected_reconstruction}\n"
        f"  Retained fraction: {accepted_count / total_generated:.1%}"
    )

    output = os.path.join(
        sample_folder,
        f"{data_name}-FactorizedVAE-v22-seed{args.seed}-{args.sample}.csv",
    )
    synthetic.to_csv(output, index=False)
    final_model = os.path.join(model_folder, "models", "FactorizedVAE_final.pkl")
    model.save(final_model)
    print(f"Saved samples: {output}")
    print(f"Saved model: {final_model}")


if __name__ == "__main__":
    main()

