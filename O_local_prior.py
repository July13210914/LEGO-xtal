"""Fast tetrahedron-constrained local oxygen prior for SiO2.

Each symmetry-expanded Si atom encodes a fixed-size local Si neighborhood and
predicts only a rotation, four bounded Si--O bond lengths, and small bounded
distortions of a preset regular SiO4 tetrahedron. Physical O atoms are recovered
after prediction by pairing proposals from different Si centers and averaging
each pair under periodic boundary conditions.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Tuple

import torch
from torch import nn


def cell_matrix_from_parameters(parameters: torch.Tensor) -> torch.Tensor:
    a, b, c, alpha, beta, gamma = parameters.unbind(dim=-1)
    a, b, c = a.clamp_min(0.25), b.clamp_min(0.25), c.clamp_min(0.25)
    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sg = torch.sin(gamma)
    sg = torch.where(sg.abs() < 1.0e-4, torch.full_like(sg, 1.0e-4), sg)
    volume_term = (1 + 2 * ca * cb * cg - ca.square() - cb.square() - cg.square()).clamp_min(1.0e-8)
    zeros = torch.zeros_like(a)
    row0 = torch.stack([a, zeros, zeros], dim=-1)
    row1 = torch.stack([b * cg, b * sg, zeros], dim=-1)
    row2 = torch.stack([
        c * cb,
        c * (ca - cb * cg) / sg,
        c * torch.sqrt(volume_term) / sg,
    ], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def batched_periodic_cartesian_delta(a_frac: torch.Tensor, b_frac: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Return a-b Cartesian deltas for batched fractional point sets."""
    delta = a_frac[:, :, None, :] - b_frac[:, None, :, :]
    delta = delta - torch.round(delta)
    return torch.einsum("b...i,bij->b...j", delta, cells)


def periodic_cartesian_delta(a_frac: torch.Tensor, b_frac: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    delta = a_frac[:, None, :] - b_frac[None, :, :]
    delta = delta - torch.round(delta)
    return torch.einsum("...i,ij->...j", delta, cell)


def stable_distance(cart: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.sqrt(torch.sum(cart * cart, dim=-1) + eps)


def cart_to_frac(cart: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Convert batched Cartesian row vectors to fractional row vectors."""
    inv = torch.linalg.inv(cells)
    return torch.einsum("b...j,bji->b...i", cart, inv)


class LocalOxygenPrior(nn.Module):
    """Per-Si model placing a bounded distortion of a preset SiO4 tetrahedron."""

    def __init__(
        self,
        max_si_atoms: int,
        max_o_atoms: int,
        token_dim: int = 128,
        heads: int = 4,
        layers: int = 3,
        neighbor_count: int = 12,
        max_vector: float = 2.8,
        bond_center: float = 1.62,
        bond_range: float = 0.35,
        max_distortion: float = 0.12,
        max_rotation: float = math.pi,
    ):
        super().__init__()
        del heads, layers, max_vector  # retained for workflow compatibility
        self.max_si_atoms = int(max_si_atoms)
        self.max_o_atoms = int(max_o_atoms)
        self.token_dim = int(token_dim)
        self.neighbor_count = int(neighbor_count)
        self.bond_center = float(bond_center)
        self.bond_range = float(bond_range)
        self.max_distortion = float(max_distortion)
        self.max_rotation = float(max_rotation)

        tetra = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=torch.float32) / math.sqrt(3.0)
        self.register_buffer("tetrahedron", tetra)

        # rotation vector (3), bond corrections (4), tangent distortions (12)
        in_dim = 6 + self.neighbor_count * 4
        hidden = token_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 19),
        )

    def _local_features(self, si_frac, si_mask, cell_parameters):
        cells = cell_matrix_from_parameters(cell_parameters)
        delta = batched_periodic_cartesian_delta(si_frac, si_frac, cells)
        dist = stable_distance(delta)
        valid_pair = si_mask[:, :, None] & si_mask[:, None, :]
        eye = torch.eye(si_frac.shape[1], device=si_frac.device, dtype=torch.bool)[None]
        dist = dist.masked_fill(~valid_pair | eye, float("inf"))
        k = min(self.neighbor_count, max(1, si_frac.shape[1] - 1))
        values, indices = torch.topk(dist, k=k, dim=-1, largest=False)
        gather_idx = indices[..., None].expand(-1, -1, -1, 3)
        neighbor_vec = torch.gather(delta, 2, gather_idx)
        finite = torch.isfinite(values)
        neighbor_vec = torch.where(finite[..., None], neighbor_vec, torch.zeros_like(neighbor_vec))
        values = torch.where(finite, values, torch.zeros_like(values))
        if k < self.neighbor_count:
            pad = self.neighbor_count - k
            neighbor_vec = torch.nn.functional.pad(neighbor_vec, (0, 0, 0, pad))
            values = torch.nn.functional.pad(values, (0, pad))
        local = torch.cat([neighbor_vec, values[..., None]], dim=-1).reshape(si_frac.shape[0], si_frac.shape[1], -1)
        cell_scale = torch.cat([cell_parameters[..., :3] / 10.0, torch.cos(cell_parameters[..., 3:])], dim=-1)
        return torch.cat([local, cell_scale[:, None, :].expand(-1, si_frac.shape[1], -1)], dim=-1)

    @staticmethod
    def _rotation_matrix(rotvec: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation from bounded axis-angle vectors."""
        angle = stable_distance(rotvec).clamp_min(1.0e-8)
        axis = rotvec / angle[..., None]
        x, y, z = axis.unbind(dim=-1)
        zero = torch.zeros_like(x)
        k = torch.stack([
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ], dim=-1).reshape(*rotvec.shape[:-1], 3, 3)
        eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
        eye = eye.expand(*rotvec.shape[:-1], 3, 3)
        sin, cos = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
        return eye + sin * k + (1.0 - cos) * torch.matmul(k, k)

    def forward(self, si_frac, si_mask, cell_parameters, o_count=None):
        del o_count
        features = self._local_features(si_frac, si_mask, cell_parameters)
        raw = self.network(features)
        rotvec = torch.tanh(raw[..., :3])
        rot_norm = stable_distance(rotvec).clamp_min(1.0e-8)
        rotvec = rotvec / rot_norm[..., None] * (torch.tanh(rot_norm)[..., None] * self.max_rotation)
        lengths = self.bond_center + self.bond_range * torch.tanh(raw[..., 3:7])
        distortion = self.max_distortion * torch.tanh(raw[..., 7:].reshape(*raw.shape[:-1], 4, 3))

        rotation = self._rotation_matrix(rotvec)
        base = torch.einsum("...ij,vj->...vi", rotation, self.tetrahedron)
        # Remove radial distortion so only angular/tangential changes are learned.
        distortion = distortion - (distortion * base).sum(dim=-1, keepdim=True) * base
        directions = torch.nn.functional.normalize(base + distortion, dim=-1)
        vectors_cart = directions * lengths[..., None]
        vectors_cart = vectors_cart * si_mask[:, :, None, None].to(vectors_cart.dtype)
        return vectors_cart, si_mask

    def proposals_fractional(self, si_frac, si_mask, cell_parameters):
        vectors_cart, _ = self.forward(si_frac, si_mask, cell_parameters)
        cells = cell_matrix_from_parameters(cell_parameters)
        vectors_frac = cart_to_frac(vectors_cart, cells)
        proposals = (si_frac[:, :, None, :] + vectors_frac) % 1.0
        proposal_mask = si_mask[:, :, None].expand(-1, -1, 4)
        parent = torch.arange(si_frac.shape[1], device=si_frac.device)[None, :, None].expand(si_frac.shape[0], -1, 4)
        return proposals, proposal_mask, parent


_PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)


def local_proposal_loss(
    predicted_vectors: torch.Tensor,
    si_mask: torch.Tensor,
    reference_o: torch.Tensor,
    reference_mask: torch.Tensor,
    si_frac: torch.Tensor,
    cell_parameters: torch.Tensor,
    vector_weight: float = 1.0,
    tetrahedral_weight: float = 0.2,
    bridge_weight: float = 0.2,
    bond_weight: float = 0.2,
    osi_second_weight: float = 0.2,
    osi_third_weight: float = 0.3,
    oo_second_weight: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Fully batched local loss.

    Targets are the four nearest teacher O atoms around each Si.  The 4-point
    assignment is solved exactly by evaluating all 24 permutations on GPU.
    ``bridge`` encourages each proposal to coincide with a proposal from a
    different Si center, expressing two-Si ownership of a physical O atom.
    """
    cells = cell_matrix_from_parameters(cell_parameters)
    # O-Si Cartesian target vectors: O - Si.
    delta = reference_o[:, None, :, :] - si_frac[:, :, None, :]
    delta = delta - torch.round(delta)
    target_cart_all = torch.einsum("bsoi,bij->bsoj", delta, cells)
    target_dist = stable_distance(target_cart_all)
    valid = si_mask[:, :, None] & reference_mask[:, None, :]
    target_dist = target_dist.masked_fill(~valid, float("inf"))
    _, nearest_idx = torch.topk(target_dist, k=4, dim=-1, largest=False)
    gather_idx = nearest_idx[..., None].expand(-1, -1, -1, 3)
    target4 = torch.gather(target_cart_all, 2, gather_idx)

    perms = _PERMS.to(predicted_vectors.device)
    # [B,S,24,4,3]
    target_perm = target4[:, :, perms, :]
    sq = (predicted_vectors[:, :, None, :, :] - target_perm).square().sum(dim=(-1, -2))
    best = sq.min(dim=-1).values / 4.0
    active = si_mask.to(best.dtype)
    vector_loss = (best * active).sum() / active.sum().clamp_min(1.0)

    pred_len = stable_distance(predicted_vectors)
    target_len = stable_distance(target4)
    bond_loss = (((torch.sort(pred_len, dim=-1).values - torch.sort(target_len, dim=-1).values).square().mean(dim=-1)) * active).sum() / active.sum().clamp_min(1.0)

    unit = predicted_vectors / pred_len[..., None].clamp_min(1.0e-6)
    dots = torch.matmul(unit, unit.transpose(-1, -2))
    tri = torch.triu(torch.ones((4, 4), device=dots.device, dtype=torch.bool), diagonal=1)
    tetra_per_si = (dots[..., tri] + 1.0 / 3.0).square().mean(dim=-1)
    tetra_loss = (tetra_per_si * active).sum() / active.sum().clamp_min(1.0)

    # Cross-parent proposal pairing.  This remains dense and batched, but has
    # no learned global attention and scales only over the 4N local proposals.
    vectors_frac = cart_to_frac(predicted_vectors, cells)
    proposals = (si_frac[:, :, None, :] + vectors_frac) % 1.0
    bsz, n_si = si_frac.shape[:2]
    flat = proposals.reshape(bsz, 4 * n_si, 3)
    flat_mask = si_mask[:, :, None].expand(-1, -1, 4).reshape(bsz, 4 * n_si)
    parents = torch.arange(n_si, device=flat.device)[None, :, None].expand(bsz, -1, 4).reshape(bsz, 4 * n_si)
    pair_delta = batched_periodic_cartesian_delta(flat, flat, cells)
    pair_dist2 = pair_delta.square().sum(dim=-1)
    allowed = flat_mask[:, :, None] & flat_mask[:, None, :] & (parents[:, :, None] != parents[:, None, :])
    pair_dist2 = pair_dist2.masked_fill(~allowed, float("inf"))
    nearest_two_cross = torch.topk(pair_dist2, k=2, dim=-1, largest=False).values
    nearest_cross = nearest_two_cross[..., 0]
    second_cross = nearest_two_cross[..., 1]
    bridge_loss = torch.where(flat_mask, nearest_cross, torch.zeros_like(nearest_cross)).sum() / flat_mask.sum().clamp_min(1)

    # Every proposal should represent an O atom bonded to its parent Si and one
    # additional Si.  Match the second O--Si distance to the teacher and keep
    # the third Si shell separated.  These terms operate before hard merging.
    proposal_si_delta = batched_periodic_cartesian_delta(flat, si_frac, cells)
    proposal_si_dist = stable_distance(proposal_si_delta)
    proposal_si_valid = flat_mask[:, :, None] & si_mask[:, None, :]
    proposal_si_dist = proposal_si_dist.masked_fill(~proposal_si_valid, float("inf"))
    pred_osi = torch.topk(proposal_si_dist, k=3, dim=-1, largest=False).values

    ref_osi_delta = batched_periodic_cartesian_delta(reference_o, si_frac, cells)
    ref_osi_dist = stable_distance(ref_osi_delta)
    ref_osi_valid = reference_mask[:, :, None] & si_mask[:, None, :]
    ref_osi_dist = ref_osi_dist.masked_fill(~ref_osi_valid, float("inf"))
    target_osi = torch.topk(ref_osi_dist, k=3, dim=-1, largest=False).values
    si_count = si_mask.sum(dim=1)
    ref_count = reference_mask.sum(dim=1)
    has_second_si = si_count >= 2
    has_third_si = si_count >= 3
    target_second_raw = target_osi[..., 1]
    target_third_raw = target_osi[..., 2]
    target_second_safe = torch.where(torch.isfinite(target_second_raw), target_second_raw, torch.zeros_like(target_second_raw))
    target_third_safe = torch.where(torch.isfinite(target_third_raw), target_third_raw, torch.zeros_like(target_third_raw))
    target_second_mean = torch.where(reference_mask, target_second_safe, torch.zeros_like(target_second_safe)).sum(dim=1) / ref_count.clamp_min(1)
    target_third_mean = torch.where(reference_mask, target_third_safe, torch.zeros_like(target_third_safe)).sum(dim=1) / ref_count.clamp_min(1)
    pred_second = torch.where(torch.isfinite(pred_osi[..., 1]), pred_osi[..., 1], torch.zeros_like(pred_osi[..., 1]))
    pred_third = torch.where(torch.isfinite(pred_osi[..., 2]), pred_osi[..., 2], torch.zeros_like(pred_osi[..., 2]))
    second_mask = flat_mask & has_second_si[:, None]
    third_mask = flat_mask & has_third_si[:, None]
    osi_second_loss = torch.where(second_mask, (pred_second - target_second_mean[:, None]).square(), torch.zeros_like(pred_second)).sum() / second_mask.sum().clamp_min(1)
    # One-sided third-shell barrier: only penalize a third Si that is closer
    # than the teacher structure's mean third-neighbor distance.
    osi_third_loss = torch.where(third_mask, torch.relu(target_third_mean[:, None] - pred_third).square(), torch.zeros_like(pred_third)).sum() / third_mask.sum().clamp_min(1)

    # The closest cross-parent proposal is the intended bridge mate.  The
    # second-closest cross-parent proposal should remain outside the teacher
    # nearest-O shell, preventing unrelated O proposals from collapsing.
    ref_oo_delta = batched_periodic_cartesian_delta(reference_o, reference_o, cells)
    ref_oo_dist = stable_distance(ref_oo_delta)
    ref_eye = torch.eye(reference_o.shape[1], device=reference_o.device, dtype=torch.bool)[None]
    ref_oo_valid = reference_mask[:, :, None] & reference_mask[:, None, :] & ~ref_eye
    ref_oo_dist = ref_oo_dist.masked_fill(~ref_oo_valid, float("inf"))
    target_oo_nn = ref_oo_dist.min(dim=-1).values
    target_oo_mean = torch.where(reference_mask, target_oo_nn, torch.zeros_like(target_oo_nn)).sum(dim=1) / reference_mask.sum(dim=1).clamp_min(1)
    second_cross_dist = torch.where(
        torch.isfinite(second_cross),
        torch.sqrt(second_cross.clamp_min(0.0) + 1.0e-8),
        torch.zeros_like(second_cross),
    )
    oo_second_loss = torch.where(flat_mask, torch.relu(target_oo_mean[:, None] - second_cross_dist).square(), torch.zeros_like(second_cross_dist)).sum() / flat_mask.sum().clamp_min(1)

    loss = (
        float(vector_weight) * vector_loss
        + float(tetrahedral_weight) * tetra_loss
        + float(bridge_weight) * bridge_loss
        + float(bond_weight) * bond_loss
        + float(osi_second_weight) * osi_second_loss
        + float(osi_third_weight) * osi_third_loss
        + float(oo_second_weight) * oo_second_loss
    )
    metrics = {
        "vector": float(vector_loss.detach()),
        "tetra": float(tetra_loss.detach()),
        "bridge": float(bridge_loss.detach()),
        "bond": float(bond_loss.detach()),
        "osi2": float(osi_second_loss.detach()),
        "osi3": float(osi_third_loss.detach()),
        "oo2": float(oo_second_loss.detach()),
    }
    return loss, metrics


def merge_local_proposals(
    proposals_frac: torch.Tensor,
    parent_ids: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Pair all local proposals into physical O positions.

    The previous nearest-pair greedy algorithm could strand two proposals from
    the same Si parent even when a complete legal pairing existed.  Here every
    Si contributes two proposals to each side of a bipartite assignment.  A
    minimum-cost Hungarian assignment then pairs every proposal exactly once
    while forbidding same-parent matches.  This guarantees exactly 2*N_Si
    merged O positions for every structure containing at least two Si atoms.
    """
    n = int(len(proposals_frac))
    if n == 0:
        return proposals_frac
    if n % 4 != 0:
        raise ValueError(f"Expected four proposals per Si, got {n} proposals")

    unique_parents = torch.unique(parent_ids, sorted=True)
    if len(unique_parents) < 2:
        raise RuntimeError("At least two Si parents are required to form bridging O pairs")

    left_parts = []
    right_parts = []
    for parent in unique_parents:
        ids = torch.nonzero(parent_ids == parent, as_tuple=False).flatten()
        if len(ids) != 4:
            raise ValueError(
                f"Expected four proposals for Si parent {int(parent)}, got {len(ids)}"
            )
        left_parts.append(ids[:2])
        right_parts.append(ids[2:])

    left = torch.cat(left_parts)
    right = torch.cat(right_parts)
    left_pos = proposals_frac[left]
    right_pos = proposals_frac[right]
    left_parent = parent_ids[left]
    right_parent = parent_ids[right]

    delta_cart = periodic_cartesian_delta(left_pos, right_pos, cell)
    cost = stable_distance(delta_cart)
    forbidden = left_parent[:, None] == right_parent[None, :]

    # A very large finite penalty is safer than inf for scipy's assignment.
    finite_max = float(cost.detach().max().cpu()) if cost.numel() else 1.0
    penalty = max(1.0e6, finite_max * 1.0e6)
    cost_cpu = cost.detach().float().cpu().numpy()
    forbidden_cpu = forbidden.detach().cpu().numpy()
    cost_cpu[forbidden_cpu] = penalty

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "merge_local_proposals requires scipy.optimize.linear_sum_assignment"
        ) from exc

    row_ind, col_ind = linear_sum_assignment(cost_cpu)
    if len(row_ind) != len(left):
        raise RuntimeError(
            f"Assignment returned {len(row_ind)} pairs for {len(left)} left proposals"
        )
    if forbidden_cpu[row_ind, col_ind].any():
        raise RuntimeError("No complete cross-parent proposal assignment was found")

    left_sel = left_pos[torch.as_tensor(row_ind, device=proposals_frac.device)]
    right_sel = right_pos[torch.as_tensor(col_ind, device=proposals_frac.device)]
    delta = right_sel - left_sel
    delta = delta - torch.round(delta)
    return (left_sel + 0.5 * delta) % 1.0


