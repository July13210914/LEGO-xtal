#!/usr/bin/env python3
"""Juliette v18 chemistry learner: building-block + Xn templates with learned
physical nonbonded exclusion walls.

The binary ASE-database path retains the v9 logic (coordination shell learning,
local-geometry health filtering, canonical Xn medoids, and weak same-element
framework diagnostics) and adds a separate physical nonbonded model.

v12 preserves the joint 3D geometry of the retained same-element
first-shell neighbour cloud.  Earlier framework models kept ranked radii and
shell-pair angle marginals, which can lose correlations and permit planar/slab
realisations.  v11 learns rotation-invariant directional Gram matrices, second-
moment eigenvalues, and planarity/anisotropy statistics from accepted structures.

For every accepted reference structure and every unordered physical-element
pair, v10 records the closest *nonbonded* periodic contact.  External-neighbour
first-shell contacts are excluded from the cross-element statistic.  The hard
exclusion wall is the most compressed accepted reference contact minus a small
user-controlled safety margin; therefore every accepted training structure is
admissible by construction.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

EPS = 1.0e-12


@dataclass
class LocalEnvironment:
    center_element: str
    neighbor_element: str
    vectors: np.ndarray
    source: str

    @property
    def distances(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def angles(self) -> np.ndarray:
        return _angles(self.vectors)


def _angles(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float).reshape(-1, 3)
    out = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            ni = float(np.linalg.norm(arr[i])); nj = float(np.linalg.norm(arr[j]))
            if ni <= EPS or nj <= EPS:
                raise ValueError("Template vectors must have nonzero length")
            c = float(np.dot(arr[i], arr[j]) / (ni * nj))
            out.append(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))
    return np.sort(np.asarray(out, dtype=float))


def _require_runtime_packages():
    try:
        from ase.db import connect  # noqa: F401
        from ase.io import read  # noqa: F401
        from pymatgen.io.ase import AseAtomsAdaptor  # noqa: F401
    except Exception as exc:
        raise RuntimeError("This script requires ASE and pymatgen in the Juliette runtime environment") from exc


def _periodic_neighbors(structure, site_id: int, species: str, radius: float) -> list[dict]:
    center = structure[site_id]
    out = []
    for nn in structure.get_neighbors(center, radius, include_index=True, include_image=True):
        if str(nn.specie.symbol) != str(species):
            continue
        d = float(nn.nn_distance)
        if d <= EPS:
            continue
        out.append({
            "distance": d,
            "vector": np.asarray(nn.coords - center.coords, dtype=float),
            "index": int(nn.index),
            "image": tuple(int(x) for x in nn.image),
        })
    out.sort(key=lambda x: (x["distance"], x["index"], x["image"]))
    return out


def _first_shell_cutoff(per_site_distances: list[list[float]], r_search: float,
                        grid_size: int = 1600, smooth_sigma: float = 7.0,
                        prominence_fraction: float = 0.025,
                        valley_fraction: float = 0.06,
                        min_valley_width: float = 0.08) -> dict:
    edges = np.linspace(0.0, float(r_search), int(grid_size) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(int(grid_size), dtype=float)
    for distances in per_site_distances:
        arr = np.asarray(distances, dtype=float)
        arr = arr[(arr > EPS) & (arr <= r_search)]
        hist += np.histogram(arr, bins=edges)[0]
    hist /= max(len(per_site_distances), 1)
    smooth = gaussian_filter1d(hist, sigma=float(smooth_sigma), mode="nearest")
    maximum = float(np.max(smooth))
    peaks, _ = find_peaks(smooth, prominence=max(maximum * prominence_fraction, EPS))
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(smooth))])
    first_peak = int(np.min(peaks))
    threshold = max(float(smooth[first_peak]) * float(valley_fraction), EPS)
    dr = float(edges[1] - edges[0])
    minimum_bins = max(1, int(math.ceil(min_valley_width / dr)))
    selected = None; run_start = None
    for idx in range(first_peak + 1, len(smooth)):
        if smooth[idx] <= threshold and run_start is None:
            run_start = idx
        if (smooth[idx] > threshold or idx == len(smooth) - 1) and run_start is not None:
            run_end = idx - 1 if smooth[idx] > threshold else idx
            if run_end - run_start + 1 >= minimum_bins:
                selected = run_start; break
            run_start = None
    if selected is None:
        later = [int(x) for x in peaks if int(x) > first_peak]
        right = later[0] if later else len(smooth) - 1
        if right <= first_peak + 1:
            raise RuntimeError("Could not separate the first coordination shell")
        selected = first_peak + int(np.argmin(smooth[first_peak:right + 1]))
    return {
        "cutoff_A": float(edges[selected]),
        "first_peak_A": float(centers[first_peak]),
        "selection": "first_persistent_valley_after_first_peak",
        "grid": {"r_A": centers.tolist(), "density_raw": hist.tolist(), "density_smooth": smooth.tolist()},
    }


def _rotation_invariant_signature(env: LocalEnvironment) -> np.ndarray:
    distances = np.sort(env.distances)
    dscale = max(float(np.mean(distances)), 1.0e-6)
    return np.concatenate([distances / dscale, env.angles / 180.0])


def _medoid_environment(environments: list[LocalEnvironment]) -> LocalEnvironment:
    if not environments:
        raise ValueError("No environments supplied")
    sig = np.vstack([_rotation_invariant_signature(x) for x in environments])
    centre = np.median(sig, axis=0)
    scale = np.median(np.abs(sig - centre), axis=0) + 1.0e-6
    score = np.mean(np.abs((sig - centre) / scale), axis=1)
    return environments[int(np.argmin(score))]


def _learn_local_so3_distribution(environments: list[LocalEnvironment], nrad: int = 5, lmax: int = 4) -> dict:
    """Learn a rigid covariance-whitened SO(3) basin for an external Xn environment.

    Unlike a distance to one canonical fingerprint, this retains the empirical
    narrowness of every descriptor direction.  Scores are raw mean-per-dimension
    Mahalanobis distances; no log/robust compression is applied.
    """
    if not environments:
        raise ValueError("No environments for local SO3 distribution")
    allr=np.concatenate([e.distances for e in environments])
    rlo=float(np.quantile(allr,0.01)); rhi=float(np.quantile(allr,0.99))
    # Extend the learned shell slightly so large radial distortions remain visible
    # rather than falling completely outside every radial basis function.
    pad=max(0.20,0.20*(rhi-rlo))
    centers=np.linspace(max(0.2,rlo-pad),rhi+pad,int(nrad))
    spacing=float((centers[-1]-centers[0])/max(len(centers)-1,1))
    width=max(0.12,0.65*spacing)
    arr=np.vstack([_so3_power_fingerprint(e.vectors,centers,width,lmax) for e in environments])
    mu=arr.mean(axis=0); cen=arr-mu[None,:]
    cov=(cen.T@cen)/max(len(arr)-1,1)
    evals,evecs=np.linalg.eigh(cov); order=np.argsort(evals)[::-1]; evals=evals[order]; evecs=evecs[:,order]
    positive=evals[evals>1e-12]
    # External coordination geometry is intentionally rigid.  Use a substantially
    # smaller covariance floor than the broad Ti-sublattice model, while retaining
    # enough regularization for finite forces in numerically null directions.
    spectral=float(np.median(positive)) if len(positive) else 1e-6
    floor=max(1e-10, spectral*1e-3)
    reg=np.maximum(evals,floor)
    z=(cen@evecs)/np.sqrt(reg[None,:]); md2=np.mean(z*z,axis=1)
    return {
        "representation":"normalized_radial_gaussian_legendre_power_spectrum_SO3_mahalanobis",
        "neighbor_count":int(len(environments[0].vectors)),"n_radial":int(nrad),"lmax":int(lmax),
        "radial_centers_A":centers.tolist(),"radial_width_A":float(width),
        "fingerprint_mean":mu.tolist(),"covariance_eigenvalues":evals.tolist(),
        "regularized_eigenvalues":reg.tolist(),"covariance_eigenvectors":evecs.tolist(),
        "mahalanobis_mean_per_dim_median":float(np.median(md2)),
        "mahalanobis_mean_per_dim_q90":float(np.quantile(md2,.90)),
        "mahalanobis_mean_per_dim_q95":float(np.quantile(md2,.95)),
        "mahalanobis_mean_per_dim_q99":float(np.quantile(md2,.99)),
        "mahalanobis_mean_per_dim_max":float(np.max(md2)),
        "site_count":int(len(arr)),
        "regularization_floor":float(floor),
        "notes":"Rigid external Xn SO3 distribution; raw Mahalanobis score, never log-compressed."
    }


def _template_statistics(environments: list[LocalEnvironment], radial_sigma_floor: float,
                         angular_sigma_floor: float, n_width: float,
                         max_radial_reconstruction_error: float,
                         max_angular_reconstruction_error: float) -> dict:
    medoid = _medoid_environment(environments)
    canonical = np.asarray(medoid.vectors, dtype=float).copy()
    radial = np.concatenate([x.distances for x in environments])
    angular_rows = np.vstack([x.angles for x in environments])
    radial_mean = float(np.mean(radial))
    radial_sigma = max(float(np.std(radial)), float(radial_sigma_floor))
    angular_mean = np.median(angular_rows, axis=0)
    amad = np.median(np.abs(angular_rows - angular_mean[None, :]), axis=0)
    angular_sigma = np.maximum(1.4826 * amad, float(angular_sigma_floor))
    q01 = float(np.quantile(radial, 0.01)); q99 = float(np.quantile(radial, 0.99))
    radial_min = max(0.0, min(q01, radial_mean - n_width * radial_sigma))
    radial_max = max(q99, radial_mean + n_width * radial_sigma)
    cr = np.linalg.norm(canonical, axis=1); ca = np.sort(_angles(canonical))
    radial_mae = float(np.mean(np.abs(cr - radial_mean)))
    angular_mae = float(np.mean(np.abs(ca - np.sort(angular_mean))))
    if radial_mae > max_radial_reconstruction_error:
        raise RuntimeError(f"Canonical medoid radial reconstruction MAE {radial_mae:.6f} A exceeds limit; source={medoid.source}")
    if angular_mae > max_angular_reconstruction_error:
        raise RuntimeError(f"Canonical medoid angular reconstruction MAE {angular_mae:.6f} deg exceeds limit; source={medoid.source}")
    return {
        "coordination_number": int(len(canonical)), "canonical_vectors_A": canonical.tolist(),
        "radial_mean_A": radial_mean, "radial_sigma_A": radial_sigma,
        "radial_min_A": radial_min, "radial_max_A": radial_max,
        "angular_mean_deg": angular_mean.tolist(), "angular_sigma_deg": angular_sigma.tolist(),
        "template_source": medoid.source, "environment_count": int(len(environments)),
        "canonical_self_consistency": {
            "radius_mean_A": float(np.mean(cr)), "radius_spread_A": float(np.std(cr)),
            "radial_reconstruction_mae_A": radial_mae, "angular_reconstruction_mae_deg": angular_mae,
        },
        "so3_environment_distribution": _learn_local_so3_distribution(environments),
        "local_soap_density": _learn_local_soap_density(environments),
    }


def _atomic_block(element: str) -> dict:
    return {"kind": "atomic", "atoms": [{"element": str(element), "position_A": [0.0, 0.0, 0.0]}],
            "reference_center_A": [0.0, 0.0, 0.0], "internal_degrees_of_freedom": []}


def _species_record(label: str, element: str, partner: str, template: dict, source: str) -> dict:
    return {
        "label": str(label), "final_formula": str(element), "building_block": _atomic_block(element),
        "external_template": {
            "kind": f"X{template['coordination_number']}",
            "coordination_number": int(template["coordination_number"]),
            "canonical_vectors_A": template["canonical_vectors_A"], "allowed_partner_labels": [],
            "allowed_partner_elements": [str(partner)], "radial_mean_A": float(template["radial_mean_A"]),
            "radial_sigma_A": float(template["radial_sigma_A"]),
            "radial_sampling_min_A": float(template["radial_min_A"]),
            "radial_sampling_max_A": float(template["radial_max_A"]),
            "first_shell_cutoff_A": float(template["first_shell_cutoff_A"]),
            "angular_mean_deg": template["angular_mean_deg"], "angular_sigma_deg": template["angular_sigma_deg"],
            "so3_environment_distribution": template["so3_environment_distribution"],
            "local_soap_density": template["local_soap_density"],
            "deformation": {"allow_rotation": True, "allow_uniform_radial_scale": True, "allow_vector_distortion": True},
        },
        "source": str(source),
        "diagnostics": {"environment_count": int(template["environment_count"]),
                        "template_source": template["template_source"],
                        "canonical_self_consistency": template["canonical_self_consistency"]},
    }


def _environment_geometry_error(env: LocalEnvironment, reference: LocalEnvironment) -> dict:
    r = np.sort(env.distances); rr = np.sort(reference.distances)
    a = np.sort(env.angles); ar = np.sort(reference.angles)
    if r.shape != rr.shape or a.shape != ar.shape:
        return {"radial_mae_A": float("inf"), "radial_max_A": float("inf"),
                "angular_mae_deg": float("inf"), "angular_max_deg": float("inf")}
    return {"radial_mae_A": float(np.mean(np.abs(r - rr))), "radial_max_A": float(np.max(np.abs(r - rr))),
            "angular_mae_deg": float(np.mean(np.abs(a - ar))), "angular_max_deg": float(np.max(np.abs(a - ar)))}



def _legendre_values(x: np.ndarray, lmax: int) -> list[np.ndarray]:
    """P_l(x), l=0..lmax, using a stable recurrence."""
    out=[np.ones_like(x)]
    if lmax == 0: return out
    out.append(x)
    for l in range(2,lmax+1):
        out.append(((2*l-1)*x*out[-1]-(l-1)*out[-2])/l)
    return out


def _so3_power_fingerprint(v: np.ndarray, radial_centers: np.ndarray, radial_width: float, lmax: int) -> np.ndarray:
    """Compact SOAP/SO(3)-style power spectrum via the spherical-harmonic addition theorem.

    This is rotation invariant and smooth in both radius and direction, but avoids a
    dependency on a particular spherical-harmonic implementation.  For each l and
    radial pair n<=n', p_nn'l is proportional to sum_jk R_n(r_j)R_n'(r_k)P_l(u_j.u_k).
    """
    r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],EPS); dots=np.clip(u@u.T,-1.0,1.0)
    rb=np.exp(-0.5*((r[:,None]-radial_centers[None,:])/max(float(radial_width),EPS))**2)
    vals=[]
    for Pl in _legendre_values(dots,int(lmax)):
        for n in range(len(radial_centers)):
            for m in range(n,len(radial_centers)):
                vals.append(float(np.einsum('j,k,jk->',rb[:,n],rb[:,m],Pl)))
    x=np.asarray(vals,float)
    # Density amplitude is not chemistry here; normalize so shape drives matching.
    return x/max(float(np.linalg.norm(x)),EPS)


def _astronomy_fingerprint(v: np.ndarray, radial_centers: np.ndarray, radial_width: float, lmax: int) -> np.ndarray:
    """Cheap full-sky descriptor: radial occupancy + rotation-invariant angular power.

    Unlike the historical SO3 fingerprint, density amplitude is retained explicitly.
    The angular block is normalized so density and sky shape remain separable.
    """
    r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],EPS); dots=np.clip(u@u.T,-1.0,1.0)
    rb=np.exp(-0.5*((r[:,None]-radial_centers[None,:])/max(float(radial_width),EPS))**2)
    occ=rb.sum(axis=0)/max(float(len(v)),1.0)
    vals=[]
    for Pl in _legendre_values(dots,int(lmax)):
        for n in range(len(radial_centers)):
            for m in range(n,len(radial_centers)):
                vals.append(float(np.einsum('j,k,jk->',rb[:,n],rb[:,m],Pl)))
    power=np.asarray(vals,float); power=power/max(float(np.linalg.norm(power)),EPS)
    return np.r_[occ,power]


def _soap_fingerprint(v: np.ndarray, radial_centers: np.ndarray, radial_width: float, lmax: int) -> np.ndarray:
    """SOAP power-spectrum implementation for a single-species Ti neighbour density.

    Uses Gaussian radial channels and the spherical-harmonic addition theorem.
    The 1/K and 1/K^2 factors retain physically meaningful density amplitude while
    keeping feature magnitudes stable across K.
    """
    r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],EPS); dots=np.clip(u@u.T,-1.0,1.0)
    rb=np.exp(-0.5*((r[:,None]-radial_centers[None,:])/max(float(radial_width),EPS))**2)
    K=max(float(len(v)),1.0); occ=rb.sum(axis=0)/K
    vals=[]
    for Pl in _legendre_values(dots,int(lmax)):
        for n in range(len(radial_centers)):
            for m in range(n,len(radial_centers)):
                vals.append(float(np.einsum('j,k,jk->',rb[:,n],rb[:,m],Pl))/(K*K))
    return np.r_[occ,np.asarray(vals,float)]


def _ace_fingerprint(v: np.ndarray, radial_centers: np.ndarray, radial_width: float, lmax: int) -> np.ndarray:
    """Compact low-body-order ACE-style invariant basis.

    Includes one-body radial density, the SOAP/two-neighbour power invariants,
    and explicit three-neighbour rotational invariants.  It is intentionally a
    small self-contained ACE basis for a cheap generator benchmark, not a full
    external ACE package.
    """
    base=_soap_fingerprint(v,radial_centers,radial_width,lmax)
    r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],EPS); d=np.clip(u@u.T,-1.0,1.0)
    rb=np.exp(-0.5*((r[:,None]-radial_centers[None,:])/max(float(radial_width),EPS))**2)
    K=max(float(len(v)),1.0)
    # Three-neighbour invariants couple different parts of the celestial sphere.
    # Keep only same radial channel to hold dimensionality/cost down.
    tri=[]
    for n in range(len(radial_centers)):
        w=rb[:,n]
        tri.append(float(np.einsum('i,j,k,ij,jk,ki->',w,w,w,d,d,d))/(K**3))
        # Even-parity area/planarity moment: |u_j x u_k|^2 = 1-(u_j.u_k)^2.
        tri.append(float(np.einsum('i,j,k,ij,jk->',w,w,w,1.0-d*d,d*d))/(K**3))
    return np.r_[base,np.asarray(tri,float)]


def _learn_local_soap_density(environments: list[LocalEnvironment], nrad: int = 6, lmax: int = 4,
                              max_components: int = 6, max_latent: int = 12) -> dict:
    """Learn a smooth multimodal density over accepted first-shell SOAP environments.

    Unlike v20's point-prototype attractors, this model learns *basins with shape*.
    SOAP features are robustly standardized, projected onto a compact PCA subspace,
    deterministically clustered, and each cluster receives a regularized full covariance.
    Stage C therefore minimizes local negative log density rather than distance to an
    ideal/prototype environment.  Correlated distortions present in accepted structures
    remain low-energy directions of the learned basin.
    """
    if not environments:
        raise ValueError("No environments for local SOAP density")
    allr=np.concatenate([e.distances for e in environments])
    rlo=float(np.quantile(allr,0.005)); rhi=float(np.quantile(allr,0.995))
    pad=max(0.15,0.15*(rhi-rlo))
    centers=np.linspace(max(0.15,rlo-pad),rhi+pad,int(nrad))
    spacing=float((centers[-1]-centers[0])/max(len(centers)-1,1))
    width=max(0.10,0.60*spacing)
    arr=np.vstack([_soap_fingerprint(e.vectors,centers,width,lmax) for e in environments])
    med=np.median(arr,axis=0); mad=np.median(np.abs(arr-med[None,:]),axis=0)
    scale=np.maximum(1.4826*mad,0.015)
    z=(arr-med[None,:])/scale[None,:]

    # PCA keeps the density compact and makes full-covariance mixture evaluation cheap.
    z0=z-z.mean(axis=0,keepdims=True)
    _,sv,vt=np.linalg.svd(z0,full_matrices=False)
    var=sv*sv/max(len(z)-1,1); frac=var/max(float(var.sum()),EPS)
    cum=np.cumsum(frac)
    latent=int(np.searchsorted(cum,0.985)+1)
    latent=max(2,min(int(max_latent),latent,vt.shape[0],max(2,len(z)-1)))
    basis=vt[:latent].T
    q=z@basis

    # A small deterministic mixture: enough basins for polymorph/local-environment
    # multimodality, but enough samples per basin to estimate covariance reliably.
    n=len(q); min_per=max(6,latent+2)
    ncomp=max(1,min(int(max_components),n//min_per))
    if ncomp == 1:
        labels=np.zeros(n,dtype=int); means=np.mean(q,axis=0,keepdims=True)
    else:
        # Farthest-point seeds followed by deterministic Lloyd iterations.
        first=int(np.argmin(np.sum((q-np.median(q,axis=0))**2,axis=1)))
        ids=[first]; dmin=np.sum((q-q[first])**2,axis=1)
        for _ in range(1,ncomp):
            j=int(np.argmax(dmin)); ids.append(j); dmin=np.minimum(dmin,np.sum((q-q[j])**2,axis=1))
        means=q[np.asarray(ids)].copy(); labels=np.zeros(n,dtype=int)
        for _ in range(40):
            d2=np.sum((q[:,None,:]-means[None,:,:])**2,axis=2); nl=np.argmin(d2,axis=1)
            if np.array_equal(nl,labels) and _>0: break
            labels=nl
            # Empty/tiny clusters are reassigned to the largest residual points.
            for k in range(ncomp):
                idx=np.where(labels==k)[0]
                if len(idx): means[k]=q[idx].mean(axis=0)
                else:
                    far=int(np.argmax(np.min(d2,axis=1))); means[k]=q[far]; labels[far]=k

    weights=[]; mus=[]; covs=[]; invs=[]; logdets=[]; counts=[]
    global_cov=np.cov(q,rowvar=False) if n>1 else np.eye(latent)
    global_cov=np.atleast_2d(global_cov)
    global_scale=max(float(np.trace(global_cov))/max(latent,1),1e-3)
    floor=max(1e-3,0.03*global_scale)
    eye=np.eye(latent)
    for k in range(ncomp):
        idx=np.where(labels==k)[0]; x=q[idx]; counts.append(int(len(idx)))
        mu=x.mean(axis=0) if len(x) else means[k]
        if len(x)>1: cov=np.cov(x,rowvar=False)
        else: cov=global_cov.copy()
        cov=np.atleast_2d(cov)
        # Shrink poorly sampled basins toward the global covariance, then floor.
        shrink=min(0.55,latent/max(float(len(x)+latent),1.0))
        cov=(1.0-shrink)*cov+shrink*global_cov+floor*eye
        ev,U=np.linalg.eigh(cov); ev=np.maximum(ev,floor); cov=(U*ev[None,:])@U.T
        inv=(U*(1.0/ev)[None,:])@U.T
        weights.append(max(len(x),1)/max(n,1)); mus.append(mu); covs.append(cov); invs.append(inv); logdets.append(float(np.log(ev).sum()))
    weights=np.asarray(weights,float); weights/=weights.sum()
    mus=np.asarray(mus); invs=np.asarray(invs); logdets=np.asarray(logdets)

    # Calibrate a dimension-normalized negative log-density score on training sites.
    diff=q[:,None,:]-mus[None,:,:]
    maha=np.einsum('nkd,kde,nke->nk',diff,invs,diff)
    comp=0.5*(maha+logdets[None,:])-np.log(np.maximum(weights[None,:],EPS))
    m=comp.min(axis=1,keepdims=True)
    nll=(m[:,0]-np.log(np.exp(-(comp-m)).sum(axis=1)))/max(latent,1)
    offset=float(np.quantile(nll,0.05)); score=nll-offset
    return {
        "representation":"first_shell_SOAP_PCA_fullcov_mixture_v20",
        "neighbor_count":int(len(environments[0].vectors)),"n_radial":int(nrad),"lmax":int(lmax),
        "radial_centers_A":centers.tolist(),"radial_width_A":float(width),
        "fingerprint_median":med.tolist(),"fingerprint_scale":scale.tolist(),
        "pca_basis":basis.tolist(),"latent_dim":int(latent),"component_count":int(ncomp),
        "mixture_weights":weights.tolist(),"component_means":mus.tolist(),
        "component_inverse_covariances":invs.tolist(),"component_logdets":logdets.tolist(),
        "score_offset":offset,"score_median":float(np.median(score)),
        "score_q90":float(np.quantile(score,.90)),"score_q95":float(np.quantile(score,.95)),
        "score_q99":float(np.quantile(score,.99)),"component_site_counts":counts,"site_count":int(n),
        "notes":"Multimodal learned first-shell SOAP density. PCA + regularized full-covariance mixture preserves correlated accepted distortions; no ideal geometry is encoded."
    }


def _robust_descriptor_model(arr: np.ndarray, representation: str, meta: dict) -> dict:
    med=np.median(arr,axis=0); mad=np.median(np.abs(arr-med[None,:]),axis=0)
    scale=np.maximum(1.4826*mad,0.02)
    z=(arr-med[None,:])/scale[None,:]; score=np.mean(z*z,axis=1)
    return dict(representation=representation,fingerprint_median=med.tolist(),
                fingerprint_scale=scale.tolist(),score_median=float(np.median(score)),
                score_q90=float(np.quantile(score,.90)),score_q95=float(np.quantile(score,.95)),
                score_q99=float(np.quantile(score,.99)),site_count=int(len(arr)),**meta)

def _learn_same_element_frameworks(structures_by_id: dict[int, object], accepted_rows: list[int],
                                   elements: list[str], r_search: float,
                                   radial_sigma_floor: float = 0.06,
                                   angular_sigma_floor: float = 5.0) -> dict[str, dict]:
    """Learn the local same-element framework including its *joint* 3D shape.

    `neighbor_count` remains the historical retained first-shell size (up to six
    nearest same-element periodic neighbours), so v11 stays backward compatible
    with existing generators.  Unlike v10, however, the directional information
    is no longer reduced to independently sorted angle marginals.

    For each accepted site we retain the ordered Cartesian neighbour vectors and
    derive a rotation-invariant unit-vector Gram matrix G = U U^T.  Given the
    radial rank ordering, G contains the complete relative 3D arrangement of the
    retained neighbour directions (up to global O(3), i.e. rotation/reflection).
    We additionally store eigenvalues of the directional second moment U^T U/k;
    its smallest eigenvalue is a direct planarity/slab diagnostic.
    """
    result = {}
    for element in elements:
        counts = []; site_vectors = []
        for row_id in accepted_rows:
            s = structures_by_id[int(row_id)]; symbols = [str(x.specie.symbol) for x in s]
            for site_id, symbol in enumerate(symbols):
                if symbol == element:
                    counts.append(len(_periodic_neighbors(s, site_id, element, r_search)))
        if not counts:
            continue

        # Historical retained framework shell.  Do NOT interpret k as chemical
        # Ti coordination number: it is the number of nearest same-element
        # neighbours represented by this framework descriptor.
        k = max(1, min(6, int(np.median(counts))))
        for row_id in accepted_rows:
            s = structures_by_id[int(row_id)]; symbols = [str(x.specie.symbol) for x in s]
            for site_id, symbol in enumerate(symbols):
                if symbol != element: continue
                n = _periodic_neighbors(s, site_id, element, r_search)
                if len(n) >= k:
                    # _periodic_neighbors is distance sorted; this rank ordering
                    # gives a deterministic correspondence for radii and Gram rows.
                    site_vectors.append(np.vstack([x["vector"] for x in n[:k]]))
        if not site_vectors: continue

        radial = np.vstack([np.linalg.norm(v, axis=1) for v in site_vectors])
        rmed = np.median(radial, axis=0); rmad = np.median(np.abs(radial-rmed[None]), axis=0)
        rsig = np.maximum(1.4826*rmad, radial_sigma_floor)
        gap_threshold = max(0.18, 2.5*float(np.median(rsig)))
        shell_ids=np.zeros(k,int); sid=0
        for rank in range(1,k):
            if float(rmed[rank]-rmed[rank-1]) > gap_threshold: sid += 1
            shell_ids[rank]=sid
        shell_centres=[float(np.mean(rmed[shell_ids==x])) for x in range(sid+1)]

        pair_groups=defaultdict(list)
        for i in range(k):
            for j in range(i+1,k): pair_groups[tuple(sorted((int(shell_ids[i]),int(shell_ids[j]))))].append((i,j))
        ag=[]
        for sp in sorted(pair_groups):
            obs=[]
            for v in site_vectors:
                u=v/np.maximum(np.linalg.norm(v,axis=1)[:,None],EPS); row=[]
                for i,j in pair_groups[sp]:
                    row.append(float(np.degrees(np.arccos(np.clip(np.dot(u[i],u[j]),-1,1)))))
                obs.append(np.sort(np.asarray(row)))
            arr=np.vstack(obs); am=np.median(arr,axis=0); amad=np.median(np.abs(arr-am[None]),axis=0)
            ag.append({"shell_pair":[int(sp[0]),int(sp[1])],
                       "neighbor_rank_pairs":[[int(i),int(j)] for i,j in pair_groups[sp]],
                       "angular_mean_deg":am.tolist(),
                       "angular_sigma_deg":np.maximum(1.4826*amad,angular_sigma_floor).tolist(),
                       "angle_count":int(len(pair_groups[sp]))})

        # v11 spherical neighbour-cloud model.  The full Gram matrix preserves
        # pair identity instead of sorting angles independently.  Its eigenvalue
        # summary makes dimensional collapse (linear/planar/slab) explicit.
        gram_rows=[]; moment_eigs=[]; planarity=[]; anisotropy=[]
        for v in site_vectors:
            u=v/np.maximum(np.linalg.norm(v,axis=1)[:,None],EPS)
            gram=np.clip(u@u.T,-1.0,1.0)
            gram_rows.append(gram.reshape(-1))
            moment=(u.T@u)/float(k)
            eig=np.sort(np.linalg.eigvalsh(moment))[::-1]  # lambda1 >= lambda2 >= lambda3
            moment_eigs.append(eig)
            planarity.append(float(eig[-1]/max(eig[0],EPS)))
            anisotropy.append(float(eig[0]-eig[-1]))
        gram_arr=np.vstack(gram_rows)
        gram_med=np.median(gram_arr,axis=0)
        gram_mad=np.median(np.abs(gram_arr-gram_med[None]),axis=0)
        gram_sig=np.maximum(1.4826*gram_mad,0.03)
        eig_arr=np.vstack(moment_eigs)
        eig_med=np.median(eig_arr,axis=0)
        eig_mad=np.median(np.abs(eig_arr-eig_med[None]),axis=0)
        eig_sig=np.maximum(1.4826*eig_mad,0.01)
        plan=np.asarray(planarity,float); aniso=np.asarray(anisotropy,float)

        # v12: empirical SO(3) descriptor *distribution*, not a single reference.
        # The radial basis is learned from the observed retained shell, then the
        # accepted site fingerprints define a covariance/PCA-shaped chemistry basin.
        so3_nrad=3; so3_lmax=4
        rlo=float(np.quantile(radial,0.02)); rhi=float(np.quantile(radial,0.98))
        so3_centers=np.linspace(rlo,rhi,so3_nrad)
        so3_width=max(0.18,float((rhi-rlo)/max(so3_nrad-1,1))*0.70)
        so3_arr=np.vstack([_so3_power_fingerprint(v,so3_centers,so3_width,so3_lmax) for v in site_vectors])
        so3_mu=so3_arr.mean(axis=0); centered=so3_arr-so3_mu[None,:]
        cov=(centered.T@centered)/max(len(so3_arr)-1,1)
        evals,evecs=np.linalg.eigh(cov); order=np.argsort(evals)[::-1]; evals=evals[order]; evecs=evecs[:,order]
        # Regularization is deliberately tied to the learned spectrum.  It prevents
        # numerically tiny directions from becoming infinitely rigid while retaining
        # the data-driven distinction between allowed and forbidden distortions.
        positive=evals[evals>1e-10]
        floor=max(1e-6,float(np.median(positive))*0.05 if len(positive) else 1e-4)
        reg_evals=np.maximum(evals,floor)
        z=(centered@evecs)/np.sqrt(reg_evals[None,:])
        md2=np.mean(z*z,axis=1)
        so3={
            "representation":"normalized_radial_gaussian_legendre_power_spectrum_SO3",
            "neighbor_count":int(k),"n_radial":int(so3_nrad),"lmax":int(so3_lmax),
            "radial_centers_A":so3_centers.tolist(),"radial_width_A":float(so3_width),
            "fingerprint_mean":so3_mu.tolist(),"covariance_eigenvalues":evals.tolist(),
            "regularized_eigenvalues":reg_evals.tolist(),"covariance_eigenvectors":evecs.tolist(),
            "mahalanobis_mean_per_dim_median":float(np.median(md2)),
            "mahalanobis_mean_per_dim_q90":float(np.quantile(md2,0.90)),
            "mahalanobis_mean_per_dim_q95":float(np.quantile(md2,0.95)),
            "site_count":int(len(site_vectors)),
            "raw_neighbor_vectors_A":[np.asarray(v,float).tolist() for v in site_vectors],
            "notes":"A smooth SO(3)-invariant empirical distribution. Covariance allows real TiO2 distortion modes without hard switching among references."
        }

        spherical={
            "representation":"radial-rank-ordered_unit-vector_gram_plus_second-moment_spectrum",
            "neighbor_count":int(k),
            "gram_mean":gram_med.reshape(k,k).tolist(),
            "gram_sigma":gram_sig.reshape(k,k).tolist(),
            "second_moment_eigenvalue_mean":eig_med.tolist(),
            "second_moment_eigenvalue_sigma":eig_sig.tolist(),
            "planarity_ratio_lambda3_over_lambda1_median":float(np.median(plan)),
            "planarity_ratio_lambda3_over_lambda1_q05":float(np.quantile(plan,0.05)),
            "planarity_ratio_lambda3_over_lambda1_q95":float(np.quantile(plan,0.95)),
            "anisotropy_lambda1_minus_lambda3_median":float(np.median(aniso)),
            "anisotropy_lambda1_minus_lambda3_q05":float(np.quantile(aniso,0.05)),
            "anisotropy_lambda1_minus_lambda3_q95":float(np.quantile(aniso,0.95)),
            "site_count":int(len(site_vectors)),
            "notes":"Gram matrix preserves the joint spherical arrangement up to global rotation/reflection; eigenvalues expose 3D versus planar collapse."
        }

        # v13: broad spherical-naturalness calibration.  This deliberately learns
        # only length scales and one-sided low-order anisotropy envelopes from a
        # longer same-element neighbourhood; it does NOT learn a reference geometry.
        ext_vectors=[]
        for row_id in accepted_rows:
            s = structures_by_id[int(row_id)]; symbols = [str(x.specie.symbol) for x in s]
            for site_id, symbol in enumerate(symbols):
                if symbol != element: continue
                n = _periodic_neighbors(s, site_id, element, r_search)
                if len(n) >= 12:
                    ext_vectors.append(np.vstack([x["vector"] for x in n[:min(24,len(n))]]))
        natural=None
        if ext_vectors:
            rank_stats={}
            for rank in (6,12,16,20,24):
                vals=[float(np.linalg.norm(v[rank-1])) for v in ext_vectors if len(v)>=rank]
                if vals:
                    rank_stats[f"r{rank}_A"]={"q05":float(np.quantile(vals,.05)),"median":float(np.median(vals)),"q95":float(np.quantile(vals,.95)),"n":len(vals)}
            # First 12 directions: low-l angular powers. For an isotropic cloud the
            # off-diagonal Legendre average tends toward zero. Store absolute-power
            # envelopes so generators can penalize only excessive anisotropy.
            powers={l:[] for l in range(1,5)}; lam3=[]
            for v in ext_vectors:
                vv=v[:12]; u=vv/np.maximum(np.linalg.norm(vv,axis=1)[:,None],EPS); d=np.clip(u@u.T,-1,1)
                mask=~np.eye(12,dtype=bool)
                Pl=_legendre_values(d,4)
                for l in range(1,5): powers[l].append(abs(float(np.mean(Pl[l][mask]))))
                eig=np.linalg.eigvalsh((u.T@u)/12.0); lam3.append(float(eig[0]))
            natural={"representation":"extended_radius_plus_low_l_spherical_anisotropy_v13","reference_neighbor_count":12,
                     "radial_rank_statistics":rank_stats,
                     "low_l_abs_pair_legendre_q95":{str(l):float(np.quantile(powers[l],.95)) for l in powers},
                     "low_l_abs_pair_legendre_median":{str(l):float(np.median(powers[l])) for l in powers},
                     "lambda3_q05":float(np.quantile(lam3,.05)),"lambda3_median":float(np.median(lam3)),
                     "notes":"Broad one-sided naturalness envelope; intended to suppress slab/clump pathologies without matching a TiO2 fingerprint."}

        # v20 retained framework descriptor: SOAP only, used by the v138 Stage-A Ti relaxation.
        # Use up to 24 periodic same-element neighbours, six overlapping radial shells,
        # and l<=4 rotational invariants.  Radial occupancy is NOT normalized away.
        soap=None
        if ext_vectors:
            sky_k=min(24,min(len(v) for v in ext_vectors))
            sky_vec=[v[:sky_k] for v in ext_vectors]
            allr=np.concatenate([np.linalg.norm(v,axis=1) for v in sky_vec])
            centers=np.linspace(float(np.quantile(allr,.03)),float(np.quantile(allr,.97)),6)
            width=max(0.20,float((centers[-1]-centers[0])/5.0)*0.72)
            arr=np.vstack([_soap_fingerprint(v,centers,width,4) for v in sky_vec])
            soap=_robust_descriptor_model(arr,"single_species_SOAP_power_spectrum_v20",
                    {"neighbor_count":int(sky_k),"n_radial":6,"lmax":4,
                     "radial_centers_A":centers.tolist(),"radial_width_A":float(width),
                     "notes":"Retained Ti-framework SOAP descriptor used by v138 Stage-A relaxation."})

        result[element]={"element":element,"neighbor_count":int(k),"radial_mean_A":rmed.tolist(),
                         "radial_sigma_A":rsig.tolist(),"radial_shell_ids":shell_ids.tolist(),
                         "radial_shell_centers_A":shell_centres,"radial_shell_gap_threshold_A":float(gap_threshold),
                         "radial_lower_bound_A":float(max(0.5,np.quantile(radial[:,0],0.01)-0.10)),
                         "connectivity_upper_A":float(np.quantile(radial[:,-1],0.99)+0.20),
                         "angular_shell_pair_groups":ag,"spherical_neighbor_cloud":spherical,"so3_environment_distribution":so3,"spherical_naturalness":natural,"soap_environment":soap,
                         "site_count":int(len(site_vectors)),
                         "score_q90_reference":1.0,"score_q90_max":3.5,
                         "source":"accepted_binary_rows_same_element_periodic_neighbours_SOAP_v20"}
    return result


def _learn_nonbonded_exclusions(structures_by_id: dict[int, object], accepted_rows: list[int],
                                elements: list[str], external_cutoffs: dict[tuple[str,str], float],
                                r_search: float, margin: float) -> tuple[list[dict], dict]:
    """Learn conservative physical nonbonded pair walls from accepted references.

    For cross-element pairs, distances within either directional first-shell cutoff
    are bonded and excluded. Same-element pairs are always nonbonded in the current
    Xn schema. Statistics use one minimum per accepted structure, preventing large
    cells from dominating the wall.
    """
    rows = []
    diagnostics = {}
    for ia, a in enumerate(sorted(elements)):
        for b in sorted(elements)[ia:]:
            per_structure_min=[]
            for row_id in accepted_rows:
                s=structures_by_id[int(row_id)]; symbols=[str(x.specie.symbol) for x in s]
                values=[]
                for i,si in enumerate(symbols):
                    if si != a: continue
                    for n in _periodic_neighbors(s,i,b,r_search):
                        d=float(n["distance"])
                        if a != b:
                            cutoff=max(float(external_cutoffs.get((a,b),0.0)),
                                       float(external_cutoffs.get((b,a),0.0)))
                            if d <= cutoff + 1.0e-8:
                                continue
                        values.append(d)
                if values:
                    per_structure_min.append(float(min(values)))
            if not per_structure_min:
                continue
            arr=np.asarray(per_structure_min,float)
            ref_min=float(np.min(arr)); q05=float(np.quantile(arr,0.05)); med=float(np.median(arr))
            hard=max(0.50,ref_min-float(margin)); soft=max(hard,ref_min-0.5*float(margin))
            record={"element_i":a,"element_j":b,"relation":"physical_nonbonded_exclusion",
                    "hard_min_A":float(hard),"soft_min_A":float(soft),
                    "reference_min_A":ref_min,"reference_q05_A":q05,"reference_median_A":med,
                    "reference_structure_count":int(len(arr)),"margin_A":float(margin),
                    "selection":"minimum_accepted_structure_nonbonded_contact_minus_margin"}
            rows.append(record); diagnostics[f"{a}-{b}"]=record
    return rows, diagnostics


def _load_binary_database(database: str, args) -> tuple[dict, dict, list[dict]]:
    from ase.db import connect
    from pymatgen.io.ase import AseAtomsAdaptor
    rows=list(connect(database,serial=True).select()); structures=[]; species_set=set()
    for row in rows:
        s=AseAtomsAdaptor.get_structure(row.toatoms()); species=sorted({str(x.specie.symbol) for x in s})
        if len(species)!=2: raise ValueError(f"ASE DB row {row.id} is not binary: {species}")
        species_set.update(species); structures.append((int(row.id),s))
    if len(species_set)!=2: raise ValueError(f"Database must contain one binary system; observed={sorted(species_set)}")
    byid={r:s for r,s in structures}; a,b=sorted(species_set)
    broad={(a,b):[],(b,a):[]}
    for _,s in structures:
        symbols=[str(x.specie.symbol) for x in s]
        for i,c in enumerate(symbols):
            p=b if c==a else a; broad[(c,p)].append([x["distance"] for x in _periodic_neighbors(s,i,p,args.r_search)])
    shell={pair:_first_shell_cutoff(v,args.r_search) for pair,v in broad.items()}
    cn_counts={pair:Counter() for pair in broad}; structure_cn=[]
    for row_id,s in structures:
        symbols=[str(x.specie.symbol) for x in s]; local=[]
        for i,c in enumerate(symbols):
            p=b if c==a else a; cutoff=shell[(c,p)]["cutoff_A"]
            cn=sum(x["distance"]<=cutoff+1e-8 for x in _periodic_neighbors(s,i,p,args.r_search))
            cn_counts[(c,p)][int(cn)]+=1; local.append((c,p,i,int(cn)))
        structure_cn.append((row_id,s,local))
    target_cn={pair:int(max(counter.items(),key=lambda x:(x[1],-x[0]))[0]) for pair,counter in cn_counts.items()}
    cn_accepted=[]; rejected=[]; provisional_by_row={}; provisional={(a,b):[],(b,a):[]}; envs={(a,b):[],(b,a):[]}
    exclusion_samples={(a,b):{"bonded_outer":[],"next_shell_inner":[]},(b,a):{"bonded_outer":[],"next_shell_inner":[]}}
    for row_id,s,local in structure_cn:
        directional={}; reasons=[]
        for pair in ((a,b),(b,a)):
            pp=[cn==target_cn[pair] for c,p,_,cn in local if (c,p)==pair]; frac=float(np.mean(pp)) if pp else 0.0
            directional[pair]=frac
            if frac+EPS<args.min_structure_pass_fraction: reasons.append(f"{pair[0]}->{pair[1]}_cn")
        if reasons:
            rejected.append({"row_id":row_id,"directional_pass_fractions":{f"{x}->{y}":directional[(x,y)] for x,y in directional},
                             "rejection_reason":";".join(reasons)}); continue
        cn_accepted.append(row_id); row_env=[]
        for c,p,i,_ in local:
            alln=_periodic_neighbors(s,i,p,args.r_search); cutoff=shell[(c,p)]["cutoff_A"]
            n=[x for x in alln if x["distance"]<=cutoff+1e-8]; expected=target_cn[(c,p)]
            if len(n)!=expected: continue
            e=LocalEnvironment(c,p,np.vstack([x["vector"] for x in n]),f"ase_db:{Path(database).resolve()}:row{row_id}:site{i}")
            row_env.append((c,p,i,e,alln,expected)); provisional[(c,p)].append(e)
        provisional_by_row[row_id]=row_env
    if not cn_accepted: raise RuntimeError("No database rows pass exact directional coordination filtering")
    refs={pair:_medoid_environment(es) for pair,es in provisional.items() if es}
    accepted=[]; geomdiag={}
    for row_id in cn_accepted:
        rowerrs=[]; reasons=[]
        for c,p,i,e,alln,expected in provisional_by_row[row_id]:
            err=_environment_geometry_error(e,refs[(c,p)]); rowerrs.append({"site_id":int(i),"direction":f"{c}->{p}",**err})
            if err["radial_mae_A"]>args.max_site_radial_mae: reasons.append(f"{c}->{p}_radial_mae")
            if err["radial_max_A"]>args.max_site_radial_max: reasons.append(f"{c}->{p}_radial_max")
            if expected>=args.strict_angular_min_cn:
                if err["angular_mae_deg"]>args.max_site_angular_mae: reasons.append(f"{c}->{p}_angular_mae")
                if err["angular_max_deg"]>args.max_site_angular_max: reasons.append(f"{c}->{p}_angular_max")
            else:
                aa=e.angles
                if aa.size and (float(np.min(aa))<args.low_cn_min_angle or float(np.max(aa))>args.low_cn_max_angle):
                    reasons.append(f"{c}->{p}_angular_sanity")
        geomdiag[row_id]=rowerrs
        if reasons:
            rejected.append({"row_id":row_id,"rejection_reason":";".join(sorted(set(reasons)))}); continue
        accepted.append(row_id)
        for c,p,i,e,alln,expected in provisional_by_row[row_id]:
            exclusion_samples[(c,p)]["bonded_outer"].append(float(alln[expected-1]["distance"]))
            if len(alln)>expected: exclusion_samples[(c,p)]["next_shell_inner"].append(float(alln[expected]["distance"]))
            envs[(c,p)].append(e)
    if not accepted: raise RuntimeError("No database rows pass the combined coordination and local-geometry filters")
    exclusion_shell={}
    for pair,v in exclusion_samples.items():
        bonded=v["bonded_outer"]; nxt=v["next_shell_inner"]
        bq=float(np.quantile(bonded,0.99)); nq=float(np.quantile(nxt,0.01)) if nxt else None
        cap=bq+args.max_exclusion_shell_margin
        if nq is not None and nq>bq+1e-6:
            raw=0.5*(bq+nq); cutoff=min(raw,cap); sel="midpoint_between_CN_and_CN_plus_1_shells_on_accepted_rows" if raw<=cap+1e-12 else "midpoint_capped_by_bonded_outer_plus_margin"
        else:
            raw=max(float(shell[pair]["cutoff_A"]),bq+0.05); cutoff=min(raw,cap); sel="initial_valley_or_bonded_outer_plus_margin_fallback" if raw<=cap+1e-12 else "fallback_capped_by_bonded_outer_plus_margin"
        exclusion_shell[pair]={"cutoff_A":float(cutoff),"bonded_outer_q99_A":bq,"next_shell_inner_q01_A":nq,
                               "selection":sel,"max_margin_from_bonded_outer_A":float(args.max_exclusion_shell_margin)}
    max_parent=max(target_cn.values()); parent_elements=sorted({c for (c,_),cn in target_cn.items() if int(cn)==int(max_parent)})
    frameworks=_learn_same_element_frameworks(byid,accepted,parent_elements,args.r_search)
    external_cutoffs={pair:float(exclusion_shell[pair]["cutoff_A"]) for pair in exclusion_shell}
    nonbonded, nbdiag=_learn_nonbonded_exclusions(byid,accepted,[a,b],external_cutoffs,args.r_search,args.nonbonded_exclusion_margin)
    diagnostics={"database":str(Path(database).resolve()),"framework_learning":frameworks,"input_rows":len(rows),
                 "coordination_accepted_rows":cn_accepted,"coordination_accepted_row_count":len(cn_accepted),
                 "accepted_rows":accepted,"accepted_row_count":len(accepted),"rejected_rows":rejected,
                 "rejected_row_count":len(rejected),"shell_learning":{f"{x}->{y}":shell[(x,y)] for x,y in shell},
                 "exclusion_shell_learning":{f"{x}->{y}":exclusion_shell[(x,y)] for x,y in exclusion_shell},
                 "nonbonded_pair_exclusion_learning":nbdiag,
                 "target_cn":{f"{x}->{y}":target_cn[(x,y)] for x,y in target_cn},
                 "cn_distributions":{f"{x}->{y}":dict(sorted(cn_counts[(x,y)].items())) for x,y in cn_counts},
                 "geometry_filter":{"reference_sources":{f"{x}->{y}":refs[(x,y)].source for x,y in refs},
                                    "max_site_radial_mae_A":args.max_site_radial_mae,"max_site_radial_max_A":args.max_site_radial_max,
                                    "max_site_angular_mae_deg":args.max_site_angular_mae,"max_site_angular_max_deg":args.max_site_angular_max,
                                    "strict_angular_min_cn":args.strict_angular_min_cn,"low_cn_min_angle_deg":args.low_cn_min_angle,
                                    "low_cn_max_angle_deg":args.low_cn_max_angle,
                                    "per_row_site_errors":{str(k):v for k,v in geomdiag.items()}}}
    return envs,diagnostics,nonbonded


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Learn Juliette building-block + Xn chemistry with physical nonbonded exclusions")
    p.add_argument("--ase-database",action="append",default=[])
    p.add_argument("--r-search",type=float,default=5.0)
    p.add_argument("--radial-sigma-floor",type=float,default=0.04)
    p.add_argument("--angular-sigma-floor",type=float,default=4.0)
    p.add_argument("--sampling-width-sigma",type=float,default=2.5)
    p.add_argument("--max-radial-reconstruction-error",type=float,default=0.35)
    p.add_argument("--max-angular-reconstruction-error",type=float,default=35.0)
    p.add_argument("--min-structure-pass-fraction",type=float,default=1.0)
    p.add_argument("--max-site-radial-mae",type=float,default=0.12)
    p.add_argument("--max-site-radial-max",type=float,default=0.25)
    p.add_argument("--max-site-angular-mae",type=float,default=12.0)
    p.add_argument("--max-site-angular-max",type=float,default=25.0)
    p.add_argument("--strict-angular-min-cn",type=int,default=4)
    p.add_argument("--low-cn-min-angle",type=float,default=45.0)
    p.add_argument("--low-cn-max-angle",type=float,default=175.0)
    p.add_argument("--max-exclusion-shell-margin",type=float,default=0.40)
    p.add_argument("--nonbonded-exclusion-margin",type=float,default=0.08,
                   help="Safety margin subtracted from the closest nonbonded contact in accepted references")
    p.add_argument("--output",default="data/xn_templates/chemistry_model.json")
    return p.parse_args()


def main() -> None:
    args=parse_args(); _require_runtime_packages()
    if not args.ase_database: raise ValueError("At least one --ase-database source is required")
    if len(args.ase_database)!=1:
        raise ValueError("v11 currently expects one binary ASE database per chemistry model")
    if args.nonbonded_exclusion_margin<=0: raise ValueError("--nonbonded-exclusion-margin must be positive")
    envs,diag,nonbonded=_load_binary_database(args.ase_database[0],args)
    species_records=[]; framework_models={}; label_set=set(); database=args.ase_database[0]
    for center,partner in sorted(envs):
        label=center
        if label in label_set: label=f"{center}_from_{Path(database).stem}"
        if label in label_set: raise ValueError(f"Duplicate construction label {label}")
        label_set.add(label)
        stats=_template_statistics(envs[(center,partner)],args.radial_sigma_floor,args.angular_sigma_floor,
                                   args.sampling_width_sigma,args.max_radial_reconstruction_error,args.max_angular_reconstruction_error)
        stats["first_shell_cutoff_A"]=float(diag["exclusion_shell_learning"][f"{center}->{partner}"]["cutoff_A"])
        species_records.append(_species_record(label,center,partner,stats,f"ase_database:{Path(database).resolve()}"))
        if center in diag.get("framework_learning",{}): framework_models[label]=dict(diag["framework_learning"][center])
    labels_by_element=defaultdict(list)
    for item in species_records: labels_by_element[item["final_formula"]].append(item["label"])
    for item in species_records:
        allowed=[]
        for element in item["external_template"]["allowed_partner_elements"]: allowed.extend(labels_by_element[element])
        item["external_template"]["allowed_partner_labels"]=sorted(set(allowed))
    pair_channels=[]; by_label={x["label"]:x for x in species_records}
    for li in sorted(by_label):
        ii=by_label[li]
        for lj in sorted(ii["external_template"]["allowed_partner_labels"]):
            jj=by_label[lj]; mui=float(ii["external_template"]["radial_mean_A"]); muj=float(jj["external_template"]["radial_mean_A"])
            sig=max(float(ii["external_template"]["radial_sigma_A"]),float(jj["external_template"]["radial_sigma_A"]))
            mu=0.5*(mui+muj) if li!=lj else mui; lo=max(0.0,mu-args.sampling_width_sigma*sig); hi=mu+args.sampling_width_sigma*sig
            shell=max(float(ii["external_template"]["first_shell_cutoff_A"]),float(jj["external_template"]["first_shell_cutoff_A"]),hi+0.05)
            pair_channels.append({"species_i":li,"species_j":lj,"relation":"external_neighbor","distance_mu_A":mu,
                                  "distance_sigma_A":sig,"sampling_min_A":lo,"sampling_max_A":hi,"first_shell_cutoff_A":shell,
                                  "source":"same_template" if li==lj else "symmetric_mean","shell_cutoff_source":"max_directional_first_shell_boundary"})
    model={"version":10,"schema":"juliette_building_block_xn_v1",
           "semantics":{"construction_species":"building_block_plus_external_template","Xn":"n expected external neighbour positions in the block-local frame",
                        "X0":"empty external template; ordinary first-class template","connectivity":"dynamic_reconciliation_not_fixed_graph",
                        "symmetry":"all block poses and templates are expanded from Wyckoff-independent variables",
                        "nonbonded_exclusion":"physical element-pair walls learned from accepted reference structures; external first-shell bonds excluded"},
           "construction_species":species_records,"pair_channels":pair_channels,"nonbonded_pair_exclusions":nonbonded,
           "framework_models":framework_models,"sources":[{"type":"ase_database",**diag,"elements":sorted({p[0] for p in envs})}],
           "parameters":{"r_search_A":args.r_search,"radial_sigma_floor_A":args.radial_sigma_floor,"angular_sigma_floor_deg":args.angular_sigma_floor,
                         "sampling_width_sigma":args.sampling_width_sigma,"max_radial_reconstruction_error_A":args.max_radial_reconstruction_error,
                         "max_angular_reconstruction_error_deg":args.max_angular_reconstruction_error,"min_structure_pass_fraction":args.min_structure_pass_fraction,
                         "max_site_radial_mae_A":args.max_site_radial_mae,"max_site_radial_max_A":args.max_site_radial_max,
                         "max_site_angular_mae_deg":args.max_site_angular_mae,"max_site_angular_max_deg":args.max_site_angular_max,
                         "strict_angular_min_cn":args.strict_angular_min_cn,"low_cn_min_angle_deg":args.low_cn_min_angle,"low_cn_max_angle_deg":args.low_cn_max_angle,
                         "max_exclusion_shell_margin_A":args.max_exclusion_shell_margin,"nonbonded_exclusion_margin_A":args.nonbonded_exclusion_margin}}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(model,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("Juliette Xn chemistry model v10 (learner v18) written")
    print(f"Output: {out}")
    print(f"Accepted database rows: {diag['accepted_row_count']}/{diag['input_rows']}")
    print("Learned physical nonbonded exclusions:")
    for row in nonbonded:
        print(f"  {row['element_i']}-{row['element_j']}: hard>={row['hard_min_A']:.4f} A; reference_min={row['reference_min_A']:.4f} A")
    for item in species_records:
        t=item["external_template"]
        print(f"  {item['label']}: {t['kind']} partners={t['allowed_partner_labels']} r={t['radial_mean_A']:.5f} A")


if __name__ == "__main__":
    main()
