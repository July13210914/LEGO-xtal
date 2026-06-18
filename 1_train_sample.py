#!/usr/bin/env python3
"""Train and sample a factorized SiO2 LEGO-Xtal VAE.

This version canonicalizes independent sites as Si(CN4) -> O(CN2) -> padding,
then trains a five-stage decoder:
    global:       space group and cell
    Si skeleton:  Si Wyckoff occupancy pattern
    Si parameters: free Wyckoff parameters conditioned on sampled Si skeleton
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



class SoftCoordinationGeometry:
    """Differentiable first-shell Si/O coordination loss.

    The discrete space group and Wyckoff skeleton are teacher-forced from the
    training row. Predicted free parameters are converted back from the RDT
    normalized representation, mapped through cached affine Wyckoff operations,
    and expanded to all symmetry-equivalent atoms. No bond angles or second
    shells are used.
    """

    def __init__(self, canonical_df, n_si_max, n_o_max):
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

    def __call__(self, row_ids, si_logits, o_logits, si_transformer, o_transformer,
                 onset, cutoff, device):
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
            cell = torch.as_tensor(cell_np, device=device, dtype=sf.dtype)
            losses.append(self._coordination_penalty(sf, of, cell, onset, cutoff))

            teacher_si = torch.as_tensor(
                teacher_si_np, device=device, dtype=sf.dtype
            )
            teacher_o = torch.as_tensor(
                teacher_o_np, device=device, dtype=of.dtype
            )
            teacher_losses.append(
                self._coordination_penalty(
                    teacher_si, teacher_o, cell, onset, cutoff
                )
            )
        if not losses:
            zero = si_logits.sum() * 0.0
            return zero, zero
        return torch.stack(losses).mean(), torch.stack(teacher_losses).mean()


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
    model_folder = os.path.join("models", data_name, "FactorizedVAE_v9")
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
    print(f"Cross loss: weight={args.cross_loss_weight}, onset={args.cross_onset} A, "
          f"cutoff={args.cross_cutoff} A, rows/batch={args.cross_batch_size}")

    geometry_loss = SoftCoordinationGeometry(canonical_df, n_si_max, n_o_max)

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
        f"    no compatible O skeleton: {mask_totals['no_compatible_o_skeleton']}\n"
        f"  Rejected slot-overflow combinations: {rejected_overflow}\n"
        f"  Rejected Wyckoff reconstructions: {rejected_reconstruction}\n"
        f"  Retained fraction: {accepted_count / total_generated:.1%}"
    )

    output = os.path.join(
        sample_folder,
        f"{data_name}-FactorizedVAE-v9-seed{args.seed}-{args.sample}.csv",
    )
    synthetic.to_csv(output, index=False)
    final_model = os.path.join(model_folder, "models", "FactorizedVAE_final.pkl")
    model.save(final_model)
    print(f"Saved samples: {output}")
    print(f"Saved model: {final_model}")


if __name__ == "__main__":
    main()

