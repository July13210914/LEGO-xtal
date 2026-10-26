#!/usr/bin/env python3
from __future__ import annotations
"""Standalone oxygen Wyckoff benchmark. Learns from reference CIFs each run.
No external learner script, saved model, or ASE database required.
"""

"""Juliette v40: restoration-benchmarked Ti-O / Ti-Ti generation learner.

Reference CIFs are loaded from one directory. StructureMatcher is diagnostic only:
references need not already occur in the ASE database. No interpolation, recursive
admission, target count, or space-group-based database identification is used.
"""

import argparse
import json
import math
import os
import sys
import subprocess
import tempfile
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

EPS = 1.0e-12

def _json_default(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, set): return sorted(obj)
    if hasattr(obj, "item"):
        try: return obj.item()
        except Exception: pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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

def _v30_center_partner_envs(structure, center, partner, cutoff, source="structure"):
    symbols=[str(x.specie.symbol) for x in structure]
    out=[]
    for i,c in enumerate(symbols):
        if c != center: continue
        nn=[x for x in _periodic_neighbors(structure,i,partner,max(5.0,float(cutoff)+1.0)) if x['distance'] <= cutoff+1e-8]
        if nn:
            out.append((i, LocalEnvironment(center,partner,np.vstack([x['vector'] for x in nn]),f"{source}:site{i}")))
    return out

def _v30_ref_shell(reference, center, partner, r_search):
    ds=[]
    for i,site in enumerate(reference):
        if str(site.specie.symbol)!=center: continue
        ds.append([x['distance'] for x in _periodic_neighbors(reference,i,partner,r_search)])
    rec=_first_shell_cutoff(ds,r_search)
    cutoff=float(rec['cutoff_A'])
    envs=_v30_center_partner_envs(reference,center,partner,cutoff,"internal_rutile")
    cns=[len(e.vectors) for _,e in envs]
    if not cns: raise RuntimeError(f"Reference has no {center}<-{partner} environments")
    cn=int(round(float(np.median(cns))))
    return cutoff,cn,envs,rec

def _v30_soap_dim(nmax=6, lmax=6):
    # l=0 radial occupancy + (lmax+1) upper-triangular radial-pair moments.
    return int(nmax + (lmax + 1) * nmax * (nmax + 1) // 2)

def _v30_soap_vec(env, cutoff, sigma):
    # Reuse the normalized whole-neighbourhood SOAP representation already tested in v27-v29.
    v=np.asarray(env.vectors,float).reshape(-1,3)
    # A compact explicit SOAP basis whose smoothing is controlled by sigma.
    # Radial centres are relative to cutoff so the representation is portable.
    nmax,lmax=6,6
    if len(v)==0:
        # Empty cumulative shells must retain the same basis dimension as occupied shells.
        return np.zeros(_v30_soap_dim(nmax,lmax),float)
    centers=np.linspace(max(0.35,0.18*cutoff),cutoff,nmax)
    rr=np.linalg.norm(v,axis=1)
    unit=v/np.maximum(rr[:,None],1e-12)
    fc=0.5*(np.cos(np.pi*np.clip(rr/cutoff,0,1))+1.0)
    g=np.exp(-0.5*((rr[:,None]-centers[None,:])/max(float(sigma),1e-3))**2)*fc[:,None]
    feats=[]
    # l=0 radial occupancy plus rotationally invariant pair angular moments P_l(cos theta).
    feats.extend(np.sum(g,axis=0).tolist())
    cos=np.clip(unit@unit.T,-1,1)
    from numpy.polynomial.legendre import legval
    for l in range(lmax+1):
        coeff=np.zeros(l+1); coeff[-1]=1.0
        Pl=legval(cos,coeff)
        for n in range(nmax):
            for m in range(n,nmax):
                feats.append(float(np.sum((g[:,n,None]*g[:,m][None,:])*Pl)))
    x=np.asarray(feats,float)
    return x/max(float(np.linalg.norm(x)),1e-12)

def _v30r2_bayes_fit(X, exponent=2.0, lam=1e-3):
    X=np.asarray(X,float)
    if X.ndim==1: X=X[None,:]
    X=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    S=np.clip(X@X.T,0.0,1.0)
    K=S**float(exponent)
    return {'X':X,'exponent':float(exponent),'lam':float(lam),
            'Kinv':np.linalg.pinv(K+float(lam)*np.eye(len(K)))}

def _v30r2_bayes_score(X,obj):
    X=np.asarray(X,float)
    if X.ndim==1: X=X[None,:]
    X=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    S=np.clip(X@obj['X'].T,0.0,1.0)
    k=S**obj['exponent']
    # Normalized SOAP has k(x,x)=1. This is GP/Bayesian support variance.
    u=1.0-np.einsum('ni,ij,nj->n',k,obj['Kinv'],k)
    return np.maximum(u,0.0)


def _safe_label(path: Path) -> str:
    s=path.stem.strip()
    s=re.sub(r'[^A-Za-z0-9_.()+-]+','_',s)
    return s or 'reference'


def _load_reference_cifs(reference_dir: str):
    from pymatgen.core import Structure
    d=Path(reference_dir).expanduser().resolve()
    if not d.is_dir():
        raise ValueError(f'v32 reference CIF directory does not exist: {d}')
    files=sorted(list(d.glob('*.cif'))+list(d.glob('*.CIF')))
    # de-duplicate case-insensitive glob overlap
    files=list(dict.fromkeys(files))
    if not files:
        raise ValueError(f'v32 found no *.cif files in {d}')
    refs={}; meta={}
    for p in files:
        label=_safe_label(p)
        base=label; k=2
        while label in refs:
            label=f'{base}_{k}'; k+=1
        try:
            st=Structure.from_file(str(p))
        except Exception as exc:
            raise ValueError(f'Could not read reference CIF {p}: {exc}') from exc
        refs[label]=st
        meta[label]={'file':str(p),'filename':p.name,'formula':st.composition.reduced_formula}
    return refs,meta




def _mutual_complete_linkage_channels(names, pairwise, raw_limit):
    """Deterministic complete-linkage partition: a merge is legal only if every
    cross-reference directed q90U is <= raw_limit in both directions.
    This deliberately forbids transitive A~B~C bridging when A !~ C.
    """
    clusters=[(n,) for n in sorted(names)]
    def compatible(a,b):
        return all(pairwise[x][y]['q90U'] <= raw_limit and
                   pairwise[y][x]['q90U'] <= raw_limit for x in a for y in b)
    def merge_cost(a,b):
        vals=[]
        for x in a:
            for y in b:
                vals += [pairwise[x][y]['q90U'], pairwise[y][x]['q90U']]
        return max(vals) if vals else 0.0
    while True:
        cand=[]
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                if compatible(clusters[i],clusters[j]):
                    cand.append((merge_cost(clusters[i],clusters[j]),
                                 clusters[i]+clusters[j],i,j))
        if not cand: break
        _,merged,i,j=min(cand,key=lambda z:(z[0],z[1]))
        clusters=[c for k,c in enumerate(clusters) if k not in (i,j)]
        clusters.append(tuple(sorted(merged)))
        clusters.sort()
    return [list(c) for c in sorted(clusters,key=lambda c:(-len(c),c))]

def _orbit_perturb_structure(structure, amplitude_A, rng):
    """Perturb crystallographic orbits coherently, plus a small symmetric lattice
    strain.  Equivalent sites receive rotation-related displacements.  This is a
    local Wyckoff-like perturbation diagnostic, not a structure generator.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core import Structure, Lattice
    st=structure.copy()
    amp=float(amplitude_A)
    if amp <= 0: return st
    # modest lattice component; largest Cartesian cell-vector change is O(amp)
    L=np.asarray(st.lattice.matrix,float)
    scale=max(float(np.mean(np.linalg.norm(L,axis=1))),1e-8)
    A=rng.normal(size=(3,3)); E=0.5*(A+A.T)
    E*=0.35*amp/(scale*max(float(np.linalg.norm(E)),1e-12))
    newL=(np.eye(3)+E)@L
    st=Structure(Lattice(newL),st.species,st.frac_coords,coords_are_cartesian=False,
                 site_properties=st.site_properties)
    try:
        sga=SpacegroupAnalyzer(structure,symprec=0.1)
        sym=sga.get_symmetrized_structure(); ops=sga.get_space_group_operations()
        frac=np.asarray(st.frac_coords,float).copy()
        oldfrac=np.asarray(structure.frac_coords,float)
        invL=np.linalg.inv(np.asarray(st.lattice.matrix,float))
        for orbit in sym.equivalent_indices:
            rep=int(orbit[0]); dcart=rng.normal(size=3)
            dcart*=amp/max(float(np.linalg.norm(dcart)),1e-12)
            dfrac=dcart@invL
            repf=oldfrac[rep]
            for j in orbit:
                j=int(j); R=np.eye(3)
                for op in ops:
                    trial=np.asarray(op.operate(repf),float)
                    df=trial-oldfrac[j]; df-=np.round(df)
                    if float(np.linalg.norm(df)) < 2e-3:
                        R=np.asarray(op.rotation_matrix,float); break
                frac[j]=(frac[j] + dfrac@R.T) % 1.0
        st=Structure(st.lattice,st.species,frac,coords_are_cartesian=False,
                     site_properties=st.site_properties)
    except Exception:
        # Conservative fallback: coherent Cartesian translation of each species group.
        frac=np.asarray(st.frac_coords,float).copy(); invL=np.linalg.inv(np.asarray(st.lattice.matrix,float))
        for sp in sorted(set(str(x.specie.symbol) for x in st)):
            d=rng.normal(size=3); d*=amp/max(float(np.linalg.norm(d)),1e-12)
            ids=[i for i,x in enumerate(st) if str(x.specie.symbol)==sp]
            frac[ids]=(frac[ids]+d@invL)%1.0
        st=Structure(st.lattice,st.species,frac,coords_are_cartesian=False,site_properties=st.site_properties)
    return st

def _calibrate_reference_fluctuation(ref, info, center, partner, sigma, exponent, lam,
                                     amplitudes, samples, allowed_amp, seed):
    rng=np.random.default_rng(int(seed))
    curve=[]
    cutoff=info['cutoff']; model=info['model']
    for amp in amplitudes:
        vals=[]
        for _ in range(int(samples)):
            ps=_orbit_perturb_structure(ref,float(amp),rng)
            envs=_v30_center_partner_envs(ps,center,partner,cutoff,f'perturb:{amp}')
            if not envs: continue
            F=np.vstack([_v30_soap_vec(e,cutoff,sigma) for _,e in envs])
            vals.append(float(np.quantile(_v30r2_bayes_score(F,model),.9)))
        curve.append({'amplitude_A':float(amp),'samples':len(vals),
                      'q50_q90U':float(np.quantile(vals,.5)) if vals else float('inf'),
                      'q90_q90U':float(np.quantile(vals,.9)) if vals else float('inf'),
                      'max_q90U':float(np.max(vals)) if vals else float('inf')})
    xs=np.asarray([x['amplitude_A'] for x in curve],float)
    ys=np.asarray([x['q90_q90U'] for x in curve],float)
    finite=np.isfinite(ys)
    if not np.any(finite): threshold=float('inf')
    else:
        xf=xs[finite]; yf=ys[finite]; order=np.argsort(xf); xf=xf[order]; yf=yf[order]
        # monotone envelope avoids Monte-Carlo reversals in the calibration curve
        yf=np.maximum.accumulate(yf)
        threshold=float(np.interp(float(allowed_amp),xf,yf,left=yf[0],right=yf[-1]))
    threshold=max(threshold,1e-8)
    return {'allowed_perturbation_A':float(allowed_amp),'acceptance_q90U':threshold,
            'curve':curve,'samples_per_amplitude':int(samples)}


def _v32_reference_channels(byid, reference_dir, center, partner, r_search,
                               reference_sigma=0.28, reference_max_bayes=0.10,
                               bayes_exponent=2.0, bayes_lambda=1e-3, allowed_perturb_A=0.10,
                               perturb_amplitudes=(0.02,0.05,0.10,0.18), perturb_samples=12, random_seed=3151):
    """Reference-only local channels; DB membership is diagnostic, never required."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    refs,refmeta=_load_reference_cifs(reference_dir)
    matcher=StructureMatcher(ltol=0.20,stol=0.30,angle_tol=5.0,
                             primitive_cell=True,scale=True,
                             attempt_supercell=False,allow_subset=False)
    matched_rows={}
    for name,ref in refs.items():
        hits=[]
        for rid,st in sorted(byid.items()):
            try:
                if matcher.fit(ref,st):
                    rms=matcher.get_rms_dist(ref,st)
                    hits.append({'row':int(rid),
                                 'rms':None if rms is None else float(rms[0]),
                                 'max_dist':None if rms is None else float(rms[1])})
            except Exception:
                continue
        hits.sort(key=lambda x:(float('inf') if x['rms'] is None else x['rms'],x['row']))
        matched_rows[name]=hits
        refmeta[name]['database_status']='matched' if hits else 'external_reference_added'
        refmeta[name]['matched_database_rows']=[x['row'] for x in hits]
        try:
            sga=SpacegroupAnalyzer(ref,symprec=0.1)
            refmeta[name]['space_group_symbol']=sga.get_space_group_symbol()
            refmeta[name]['space_group_number']=int(sga.get_space_group_number())
        except Exception:
            refmeta[name]['space_group_symbol']=None; refmeta[name]['space_group_number']=None

    refinfo={}
    for name,ref in refs.items():
        cutoff,cn,envs,shellrec=_v30_ref_shell(ref,center,partner,r_search)
        F=np.vstack([_v30_soap_vec(e,cutoff,reference_sigma) for _,e in envs])
        refinfo[name]={'cutoff':float(cutoff),'cn':int(cn),'envs':envs,'shellrec':shellrec,
                       'features':F,'model':_v30r2_bayes_fit(F,bayes_exponent,bayes_lambda)}

    names=list(refinfo)
    pairwise={}
    for a in names:
        pairwise[a]={}
        for b in names:
            u=_v30r2_bayes_score(refinfo[a]['features'],refinfo[b]['model'])
            pairwise[a][b]={'q90U':float(np.quantile(u,.9)),
                            'q50U':float(np.quantile(u,.5)),
                            'maxU':float(np.max(u))}

    # Channel identity is strict mutual similarity; no connected-component bridging.
    channels=_mutual_complete_linkage_channels(names,pairwise,reference_max_bayes)

    # Each reference gets its own physically interpretable fluctuation scale.
    calibrations={}
    for ni,name in enumerate(names):
        calibrations[name]=_calibrate_reference_fluctuation(
            refs[name],refinfo[name],center,partner,reference_sigma,bayes_exponent,bayes_lambda,
            perturb_amplitudes,perturb_samples,allowed_perturb_A,random_seed+1009*ni)

    records={}; selected=[]
    for rid,s in sorted(byid.items()):
        byref={}
        for name,info in refinfo.items():
            envs=_v30_center_partner_envs(s,center,partner,info['cutoff'],f'row{rid}:{name}')
            if not envs:
                byref[name]={'q90U':float('inf'),'normalized_D':float('inf'),
                             'acceptance_q90U':float(calibrations[name]['acceptance_q90U']),
                             'q50U':float('inf'),'maxU':float('inf'),'site_count':0,'cn_range':None}
                continue
            F=np.vstack([_v30_soap_vec(e,info['cutoff'],reference_sigma) for _,e in envs])
            u=_v30r2_bayes_score(F,info['model']); cns=[len(e.vectors) for _,e in envs]
            q90=float(np.quantile(u,.9)); thr=calibrations[name]['acceptance_q90U']
            byref[name]={'q90U':q90,'normalized_D':float(q90/max(thr,1e-12)),
                         'acceptance_q90U':float(thr),'q50U':float(np.quantile(u,.5)),
                         'maxU':float(np.max(u)),'site_count':len(envs),
                         'cn_range':[int(min(cns)),int(max(cns))]}
        bych={}
        for ci,members in enumerate(channels,1):
            d,n=min((byref[n]['normalized_D'],n) for n in members)
            bych[f'C{ci}']={'normalized_D':float(d),'q90U':float(byref[n]['q90U']),
                            'nearest_reference':n,'members':members}
        best_ch=min(bych,key=lambda c:bych[c]['normalized_D'])
        best_D=float(bych[best_ch]['normalized_D']); best=float(bych[best_ch]['q90U'])
        best_ref=bych[best_ch]['nearest_reference']; passed=bool(best_D<=1.0)

        symbols=[str(x.specie.symbol) for x in s]
        tio=[]; titi_min=[]; ti_cn330=[]
        for i,z in enumerate(symbols):
            if z!=center: continue
            od=[x['distance'] for x in _periodic_neighbors(s,i,partner,r_search)]
            td=[x['distance'] for x in _periodic_neighbors(s,i,center,max(r_search,5.0)) if x['distance']>1e-6]
            tio.append(sorted(od)[:8])
            if td: titi_min.append(min(td))
            ti_cn330.append(sum(d<=3.30 for d in td))
        try: sg=int(SpacegroupAnalyzer(s,symprec=0.1).get_space_group_number())
        except Exception: sg=None
        topo={'space_group':sg,'volume_per_atom_A3':float(s.volume/len(s)),
              'lattice_abc_A':[float(x) for x in s.lattice.abc],
              'lattice_angles_deg':[float(x) for x in s.lattice.angles],
              'center_count':int(sum(z==center for z in symbols)),
              'center_partner_sorted_first8_A':tio,
              'center_center_min_A':float(min(titi_min)) if titi_min else None,
              'center_center_CN_3p30_median':float(np.median(ti_cn330)) if ti_cn330 else None}
        records[int(rid)]={'pass':passed,'best_channel':best_ch,'best_reference':best_ref,
                           'best_q90U':best,'best_normalized_D':best_D,'by_channel':bych,'by_reference':byref,'topology':topo}
        if passed: selected.append(int(rid))

    refs_out={}
    for n in names:
        refs_out[n]={**refmeta[n],
                     'match_details':matched_rows[n],
                     'first_shell_cutoff_A':refinfo[n]['cutoff'],
                     'reference_CN':refinfo[n]['cn'],
                     'shell_learning':refinfo[n]['shellrec']}
    ch_out=[{'id':f'C{i}','members':m} for i,m in enumerate(channels,1)]
    external=[n for n in names if not matched_rows[n]]
    training_pool={'database_selected_rows':sorted(selected),
                   'external_reference_structures_added':external,
                   'note':'Unmatched user references are retained as authoritative reference/training examples; they are not inserted into the ASE DB file.'}
    return {'method':'v32 directory-loaded fixed Bayesian nearest-shell reference channels; no interpolation',
            'reference_directory':str(Path(reference_dir).expanduser().resolve()),
            'references':refs_out,'pairwise_reference_bayesian_distance':pairwise,
            'reference_channels':ch_out,'reference_fluctuation_calibration':calibrations,
            'bayesian_metric':{'soap_sigma_A':float(reference_sigma),'kernel_exponent':float(bayes_exponent),
                               'lambda':float(bayes_lambda),'channel_merge_raw_q90U_limit':float(reference_max_bayes),
                               'allowed_perturbation_A':float(allowed_perturb_A)},
            'selection_rule':'User CIF references define nearest-shell local chemistry. StructureMatcher only reports DB membership. All references remain valid even when absent from DB. Database rows are admitted only when normalized fluctuation distance D<=1 for at least one reference. Raw q90U is not a universal admission threshold. No interpolation, recursive admission, target count, Ti-Ti, lattice, SG, or global topology enters selection.',
            'all_rows':records,'selected_rows':sorted(selected),'selected_count':len(selected),
            'training_pool':training_pool}



def _titi_spherical_soap_descriptor(structure, site_id, center, rmax, sigma, nmax=12, lmax=10):
    """Single continuous Ti<-Ti spherical SOAP field.

    No learned/cumulative radial shells enter the descriptor.  All Ti neighbours inside
    one smooth outer cutoff contribute simultaneously.  Radial basis functions are
    coupled through rotationally invariant angular (spherical-harmonic/Legendre)
    power-spectrum channels, so the representation retains cross-radius angular
    correlations instead of concatenating independent shell descriptors.
    """
    nn=_periodic_neighbors(structure,site_id,center,float(rmax))
    nn=[x for x in nn if x['distance']>1e-6 and x['distance']<=float(rmax)+1e-8]
    v=np.vstack([x['vector'] for x in nn]) if nn else np.zeros((0,3),float)
    env=LocalEnvironment(center,center,v,f'titi:site{site_id}:rc{rmax:.3f}')
    v=np.asarray(env.vectors,float).reshape(-1,3)
    dim=int(nmax + (lmax+1)*nmax*(nmax+1)//2)
    if len(v)==0:
        return np.zeros(dim,float),0,env
    rr=np.linalg.norm(v,axis=1)
    unit=v/np.maximum(rr[:,None],1e-12)
    # Overlapping radial basis over the full field of view; rmax is only the smooth
    # locality boundary, never an internal shell boundary.
    lo=max(0.35,0.12*float(rmax))
    centers=np.linspace(lo,float(rmax),int(nmax))
    width=max(float(sigma),0.55*float(rmax)/(max(int(nmax)-1,1)))
    fc=0.5*(np.cos(np.pi*np.clip(rr/float(rmax),0,1))+1.0)
    g=np.exp(-0.5*((rr[:,None]-centers[None,:])/width)**2)*fc[:,None]
    feats=list(np.sum(g,axis=0))
    cos=np.clip(unit@unit.T,-1.0,1.0)
    from numpy.polynomial.legendre import legval
    # Addition theorem: sum_m Y_lm(u_i)Y*_lm(u_j) is proportional to P_l(u_i.u_j).
    # Thus these p_nn'l are the rotationally invariant spherical-harmonic SOAP
    # power spectrum, evaluated without explicitly storing every m coefficient.
    for l in range(int(lmax)+1):
        coeff=np.zeros(l+1); coeff[-1]=1.0
        Pl=legval(cos,coeff)
        for n in range(int(nmax)):
            for m in range(n,int(nmax)):
                feats.append(float(np.sum((g[:,n,None]*g[:,m][None,:])*Pl)))
    x=np.asarray(feats,float)
    if len(x)!=dim:
        raise RuntimeError(f'Ti-Ti spherical SOAP dimension {len(x)} != expected {dim}')
    x=x/max(float(np.linalg.norm(x)),1e-12)
    return x,len(nn),env

def _pairwise_cosine_distance(X):
    X=np.asarray(X,float); X=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    return np.maximum(0.0,1.0-np.clip(X@X.T,-1.0,1.0))


def _calibrate_titi_descriptor_radius(structures, center, rmax, sigma, nmax, lmax, amplitude_A, samples, seed, return_records=False):
    """Descriptor-space radius generated by actual symmetry-preserving perturbations."""
    rng=np.random.default_rng(int(seed)); vals=[]; records=[]
    # Structure-balanced cap keeps this diagnostic cheap and prevents large cells dominating.
    for key,st in structures:
        sites=[i for i,x in enumerate(st) if str(x.specie.symbol)==center]
        if not sites: continue
        base={i:_titi_spherical_soap_descriptor(st,i,center,rmax,sigma,nmax,lmax)[0] for i in sites}
        for _ in range(int(samples)):
            sp=_orbit_perturb_structure(st,float(amplitude_A),rng)
            for i in sites:
                xp=_titi_spherical_soap_descriptor(sp,i,center,rmax,sigma,nmax,lmax)[0]
                if base[i].shape != xp.shape:
                    raise RuntimeError(f'Ti-Ti perturbation descriptor dimension changed for {key} site {i}: {base[i].shape} -> {xp.shape}')
                d=max(0.0,1.0-float(np.dot(base[i],xp)))
                vals.append(d); records.append({'source':key,'site':int(i),'distance':float(d)})
    if not vals:
        return (0.02,[],[]) if return_records else (0.02,[])
    # q90 is the supported local fluctuation envelope; never let numerical zero collapse modes.
    radius=max(float(np.quantile(vals,.90)),1e-6)
    return (radius,vals,records) if return_records else (radius,vals)


def _titi_perturbation_training_cycles(structures, center, rmax, sigma, nmax, lmax, amplitudes, samples, seed, base_lookup=None):
    """Build the perturbation training cloud used by the Ti-Ti NN.

    v37 can reuse already-computed unperturbed descriptors through ``base_lookup``.
    This changes only runtime: the numerical descriptors and perturbations are identical.
    """
    rng=np.random.default_rng(int(seed)); base=[]; base_meta=[]
    base_lookup=base_lookup or {}
    for source,st in structures:
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=center: continue
            cached=base_lookup.get((source,int(i)))
            if cached is None:
                x,cn,_=_titi_spherical_soap_descriptor(st,i,center,rmax,sigma,nmax,lmax)
            else:
                x,cn=cached
            base.append(np.asarray(x,float)); base_meta.append((source,int(i),int(cn)))
    if not base: return {'cycles':[],'augmented_X':np.zeros((0,1)),'augmented_meta':[]}
    dim=len(base[0])
    if any(len(x)!=dim for x in base): raise RuntimeError('Base Ti-Ti descriptor dimensions are inconsistent')
    augmented=list(base); augmented_meta=[{'source':s,'site':i,'kind':'base','amplitude_A':0.0} for s,i,_ in base_meta]
    cycles=[]
    for amp in amplitudes:
        ds=[]; added=0
        for source,st in structures:
            sites=[i for i,x in enumerate(st) if str(x.specie.symbol)==center]
            if not sites: continue
            b={}
            for i in sites:
                cached=base_lookup.get((source,int(i)))
                b[i]=np.asarray(cached[0],float) if cached is not None else _titi_spherical_soap_descriptor(st,i,center,rmax,sigma,nmax,lmax)[0]
            for rep in range(int(samples)):
                sp=_orbit_perturb_structure(st,float(amp),rng)
                for i in sites:
                    xp=_titi_spherical_soap_descriptor(sp,i,center,rmax,sigma,nmax,lmax)[0]
                    if len(xp)!=dim or b[i].shape!=xp.shape:
                        raise RuntimeError(f'Perturbation cycle descriptor dimension changed for {source} site {i}')
                    d=max(0.0,1.0-float(np.dot(b[i],xp))); ds.append(d)
                    augmented.append(xp); augmented_meta.append({'source':source,'site':int(i),'kind':'perturbed','amplitude_A':float(amp),'replicate':int(rep),'base_distance':float(d)})
                    added+=1
        cycles.append({'amplitude_A':float(amp),'samples':len(ds),'descriptor_distance_q50':float(np.quantile(ds,.5)) if ds else None,'descriptor_distance_q90':float(np.quantile(ds,.9)) if ds else None,'descriptor_distance_q99':float(np.quantile(ds,.99)) if ds else None,'augmented_sites':int(added)})
    return {'cycles':cycles,'augmented_X':np.vstack(augmented),'augmented_meta':augmented_meta}



def _learn_nn_perturbation_metric(X_base, meta, perturb, seed=3401, epochs=450, lr=1e-3,
                                  latent_dim=16, hidden1=256, hidden2=64,
                                  preserve_weight=0.35, variance_weight=0.08, decorrelation_weight=0.02):
    """Self-supervised nonlinear SOAP metric learned only from perturbation smoothness.

    The network never sees polymorph, SG, shell, or mode labels. Same-site perturbations
    provide the local smoothness signal; random pairs of distinct unperturbed sites retain
    the global SOAP geometry and prevent collapse. Output vectors are unit normalized, so
    downstream distance remains 1-cosine in a learned latent space.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    X=np.asarray(X_base,float); Xa=np.asarray(perturb['augmented_X'],float); am=perturb['augmented_meta']
    if X.ndim!=2 or len(X)==0: return np.zeros((len(X),int(latent_dim))),np.zeros((len(Xa),int(latent_dim))),{}
    lookup={(m['source'],int(m['site'])):i for i,m in enumerate(meta)}
    pos=[]; amps=[]
    for j,m in enumerate(am):
        if m.get('kind')!='perturbed': continue
        i=lookup.get((m['source'],int(m['site'])))
        if i is not None: pos.append((i,j)); amps.append(float(m['amplitude_A']))
    if not pos: return X.copy(),Xa.copy(),{}

    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    class MetricNet(nn.Module):
        def __init__(self,din):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(din,int(hidden1)),nn.LayerNorm(int(hidden1)),nn.GELU(),
                                   nn.Linear(int(hidden1),int(hidden2)),nn.LayerNorm(int(hidden2)),nn.GELU(),
                                   nn.Linear(int(hidden2),int(latent_dim)))
        def forward(self,x): return F.normalize(self.net(x),p=2,dim=-1,eps=1e-8)
    net=MetricNet(X.shape[1]).to(device)
    xb=torch.tensor(X,dtype=torch.float32,device=device); xa=torch.tensor(Xa,dtype=torch.float32,device=device)
    ii=torch.tensor([q[0] for q in pos],dtype=torch.long,device=device)
    jj=torch.tensor([q[1] for q in pos],dtype=torch.long,device=device)
    aa=torch.tensor(amps,dtype=torch.float32,device=device)
    raw=np.maximum(1e-8,1.0-np.sum(X[[q[0] for q in pos]]*Xa[[q[1] for q in pos]],axis=1))
    scale=float(np.median(raw/(np.asarray(amps)**2+1e-8))); scale=max(scale,1e-4)
    opt=torch.optim.AdamW(net.parameters(),lr=float(lr),weight_decay=1e-5)
    rng=np.random.default_rng(int(seed)+17); history=[]
    for ep in range(int(epochs)):
        net.train(); opt.zero_grad()
        zi=net(xb[ii]); zj=net(xa[jj]); dp=1.0-torch.sum(zi*zj,dim=1)
        target=torch.tensor(scale,dtype=torch.float32,device=device)*(aa**2)
        fit=F.smooth_l1_loss(dp,target)
        # Preserve distances/order among real environments. This is geometry, not a class target.
        n=min(4096,max(256,len(X)*8)); ia=rng.integers(0,len(X),n); ib=rng.integers(0,len(X),n)
        ia_t=torch.tensor(ia,dtype=torch.long,device=device); ib_t=torch.tensor(ib,dtype=torch.long,device=device)
        za=net(xb[ia_t]); zb=net(xb[ib_t]); dl=1.0-torch.sum(za*zb,dim=1)
        dr=torch.tensor(np.maximum(0.0,1.0-np.sum(X[ia]*X[ib],axis=1)),dtype=torch.float32,device=device)
        preserve=F.smooth_l1_loss(dl,dr)
        # Explicit anti-collapse / latent-use terms on base sites.
        zbase=net(xb); std=torch.sqrt(zbase.var(dim=0,unbiased=False)+1e-4)
        variance=torch.mean(F.relu(0.12-std))
        zc=zbase-zbase.mean(dim=0,keepdim=True); cov=(zc.T@zc)/max(1,len(X)-1)
        off=cov-torch.diag(torch.diag(cov)); decor=torch.mean(off*off)
        loss=fit+float(preserve_weight)*preserve+float(variance_weight)*variance+float(decorrelation_weight)*decor
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        if ep in (0,int(epochs)//4,int(epochs)//2,3*int(epochs)//4,int(epochs)-1):
            history.append({'epoch':int(ep+1),'loss':float(loss.detach().cpu()),'perturb_fit':float(fit.detach().cpu()),
                            'preserve':float(preserve.detach().cpu()),'variance':float(variance.detach().cpu()),'decorrelation':float(decor.detach().cpu())})
    net.eval()
    with torch.no_grad():
        Z=net(xb).cpu().numpy(); Za=net(xa).cpu().numpy()
    learned=[]
    for amp in sorted(set(amps)):
        ds=[max(0.0,1.0-float(np.dot(Z[i],Za[j]))) for (i,j),a0 in zip(pos,amps) if abs(a0-amp)<1e-12]
        learned.append({'amplitude_A':float(amp),'samples':len(ds),'distance_q50':float(np.quantile(ds,.5)),
                        'distance_q90':float(np.quantile(ds,.9)),'distance_q99':float(np.quantile(ds,.99))})
    # Store the compact network so generation can reproduce the learned metric exactly.
    state={k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()}
    info={'device':device,'epochs':int(epochs),'learning_rate':float(lr),'architecture':[int(X.shape[1]),int(hidden1),int(hidden2),int(latent_dim)],
          'activation':'GELU','normalization':'LayerNorm hidden + L2 latent','distance':'1-cosine(latent)',
          'supervision':'same-site perturbation amplitude^2 + preservation of unperturbed SOAP geometry; no topology/mode/polymorph labels',
          'loss_weights':{'preserve':float(preserve_weight),'variance':float(variance_weight),'decorrelation':float(decorrelation_weight)},
          'target_scale_per_A2':float(scale),'cycles':learned,'history':history,'state_dict':state}
    return Z,Za,info


def _strict_complete_modes(X, threshold):
    """No transitive bridging: complete-linkage clustering in multishell SOAP distance."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    X=np.asarray(X,float)
    if len(X)<=1: return np.ones(len(X),dtype=int)
    D=_pairwise_cosine_distance(X); np.fill_diagonal(D,0.0)
    Z=linkage(squareform(D,checks=False),method='complete')
    lab=fcluster(Z,t=float(threshold),criterion='distance')
    # stable relabel by first occurrence
    mp={}; nxt=1; out=[]
    for x in lab:
        x=int(x)
        if x not in mp: mp[x]=nxt; nxt+=1
        out.append(mp[x])
    return np.asarray(out,int)


def _rank_angle_correlations(envs, min_count=8):
    """Expose high-order dependence: correlations among ranked Ti-Ti-Ti angles for fixed CN."""
    bycn={}
    for e in envs:
        cn=len(e.vectors)
        if cn>=2: bycn.setdefault(cn,[]).append(e.angles)
    out={}
    for cn,aa in sorted(bycn.items()):
        if len(aa)<min_count: continue
        A=np.vstack(aa)
        if A.shape[1]<2: continue
        sd=np.std(A,axis=0); good=np.where(sd>1e-12)[0]; pairs=[]
        if len(good)>=2:
            C=np.corrcoef(A[:,good],rowvar=False)
            for ii in range(C.shape[0]):
                for jj in range(ii+1,C.shape[1]):
                    i=int(good[ii]); j=int(good[jj])
                    if np.isfinite(C[ii,jj]): pairs.append((abs(float(C[ii,jj])),i,j,float(C[ii,jj])))
        pairs.sort(reverse=True)
        out[str(cn)]={'samples':len(aa),'angles_per_site':int(A.shape[1]),
                      'rank_q10_deg':np.quantile(A,.10,axis=0).tolist(),
                      'rank_q50_deg':np.quantile(A,.50,axis=0).tolist(),
                      'rank_q90_deg':np.quantile(A,.90,axis=0).tolist(),
                      'strongest_rank_correlations':[{'rank_i':i+1,'rank_j':j+1,'corr':c} for _,i,j,c in pairs[:12]]}
    return out



def _latent_topology_diagnostic(Z, meta, perturb, Za, learned_radius, tio_report, Xraw=None):
    """Diagnose the geometry/topology of the learned latent Ti manifold without
    imposing a mode count or using topology labels in training.

    Reports PCA effective dimension, kNN connectivity, epsilon-graph component
    persistence, MST bottlenecks, and locations of crystallographic references.
    UMAP/t-SNE are intentionally not used to define topology.
    """
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    Z=np.asarray(Z,float)
    n=len(Z)
    if n==0:
        return {'site_count':0}
    D=_pairwise_cosine_distance(Z); np.fill_diagonal(D,0.0)
    # PCA in latent coordinates: centered Euclidean covariance is only a dimensionality
    # diagnostic; the operational manifold metric remains 1-cosine(latent).
    Y=Z-Z.mean(axis=0,keepdims=True)
    sv=np.linalg.svd(Y,compute_uv=False)
    var=sv*sv
    frac=var/max(float(var.sum()),EPS)
    cum=np.cumsum(frac)
    def ndim_at(q): return int(np.searchsorted(cum,q)+1) if len(cum) else 0
    pca={'explained_variance_fraction':frac.tolist(),
         'cumulative_explained_variance':cum.tolist(),
         'dimensions_90pct':ndim_at(.90),'dimensions_95pct':ndim_at(.95),'dimensions_99pct':ndim_at(.99),
         'participation_ratio':float((var.sum()**2)/max(float(np.sum(var*var)),EPS))}
    # Pairwise distance scale among real sites.
    tri=D[np.triu_indices(n,1)] if n>1 else np.asarray([0.0])
    pair={'q01':float(np.quantile(tri,.01)),'q05':float(np.quantile(tri,.05)),
          'q10':float(np.quantile(tri,.10)),'q25':float(np.quantile(tri,.25)),
          'q50':float(np.quantile(tri,.50)),'q90':float(np.quantile(tri,.90)),
          'q99':float(np.quantile(tri,.99))}
    # Undirected kNN graph using union symmetrization. Components disappearing rapidly
    # with k indicate sampling gaps rather than robust disconnected manifolds.
    knn=[]
    order=np.argsort(D,axis=1)
    for k in (1,2,3,4,5,6,8,10,12,16,24,32):
        if n<=1: break
        kk=min(k,n-1); A=np.zeros((n,n),dtype=np.uint8)
        for i in range(n): A[i,order[i,1:kk+1]]=1
        A=np.maximum(A,A.T)
        nc,lab=connected_components(csr_matrix(A),directed=False)
        sizes=sorted(np.bincount(lab).tolist(),reverse=True)
        knn.append({'k':int(kk),'components':int(nc),'component_sizes':sizes})
    # Epsilon connectivity at physically calibrated and data-derived scales.
    epsvals=set([float(learned_radius)])
    for c in perturb.get('cycles',[]):
        for q in ('distance_q50','distance_q90','distance_q99'):
            if c.get(q) is not None: epsvals.add(float(c[q]))
    for q in (.01,.02,.05,.10,.20,.30,.50): epsvals.add(float(np.quantile(tri,q)))
    eps=[]
    for e in sorted(x for x in epsvals if np.isfinite(x) and x>=0):
        A=(D<=e).astype(np.uint8); np.fill_diagonal(A,0)
        nc,lab=connected_components(csr_matrix(A),directed=False)
        sizes=sorted(np.bincount(lab).tolist(),reverse=True)
        eps.append({'epsilon':float(e),'components':int(nc),'component_sizes':sizes,
                    'largest_fraction':float(sizes[0]/n) if sizes else 0.0})
    # MST is a compact persistence proxy. v34r5 resolves the largest bridges back
    # to actual Ti sites and successively cuts them to expose supported macroscopic basins.
    def site_record(i):
        m=meta[int(i)]; src=m['source']; rec={'index':int(i),'source':src,'site':int(m['site']),
            'CN_within_rmax':int(m.get('CN_within_rmax',0))}
        if src.startswith('db:'):
            rid=int(src.split(':',1)[1]); rr=tio_report.get('all_rows',{}).get(rid,tio_report.get('all_rows',{}).get(str(rid),{}))
            rec['database_row']=rid
            if rr:
                rec['TiO_channel']=rr.get('best_channel'); rec['TiO_reference']=rr.get('best_reference')
                rec['TiO_q90U']=rr.get('best_q90U'); rec['TiO_normalized_D']=rr.get('best_normalized_D')
                rec['topology']=rr.get('topology',{})
        elif src.startswith('ref:'):
            name=src.split(':',1)[1]; rec['external_reference']=name
            # Reference channel follows the strict reference-channel membership.
            for ch,names in tio_report.get('channels',{}).items():
                if name in names: rec['TiO_channel']=ch; break
            rec['TiO_reference']=name
        return rec
    if n>1:
        M=minimum_spanning_tree(csr_matrix(D)).toarray(); edges=[]
        for i,j in zip(*np.nonzero(M)): edges.append((float(M[i,j]),int(i),int(j)))
        edges.sort(reverse=True); evals=np.asarray([e[0] for e in edges],float)
        bott=[]
        Xr=np.asarray(Xraw,float) if Xraw is not None else None
        for rank,(d,i,j) in enumerate(edges[:16],1):
            rawd=None
            if Xr is not None and len(Xr)==n:
                rawd=float(max(0.0,1.0-float(np.dot(Xr[i],Xr[j])/(max(np.linalg.norm(Xr[i])*np.linalg.norm(Xr[j]),EPS)))))
            nxt=float(edges[rank][0]) if rank<len(edges) else 0.0
            bott.append({'rank':rank,'latent_distance':d,'raw_SOAP_distance':rawd,
                         'distance_over_0p10A_q90':float(d/max(learned_radius,EPS)),
                         'gap_ratio_to_next':float(d/max(nxt,EPS)) if nxt>0 else None,
                         'endpoint_i':site_record(i),'endpoint_j':site_record(j)})
        # Successively remove the largest MST edges. This does not admit/interpolate anything;
        # it asks what components are already present when sparse bridges are severed.
        cuts=[]
        for cutn in range(1,min(12,len(edges))+1):
            A=M+M.T
            for _,i,j in edges[:cutn]: A[i,j]=0.0; A[j,i]=0.0
            nc,lab=connected_components(csr_matrix(A),directed=False)
            comps=[]
            for cid in range(nc):
                idx=np.where(lab==cid)[0]; srcs=sorted(set(meta[q]['source'] for q in idx))
                rows=sorted(int(x.split(':',1)[1]) for x in srcs if x.startswith('db:'))
                refs0=sorted(x.split(':',1)[1] for x in srcs if x.startswith('ref:'))
                ch={}
                for q in idx:
                    r=site_record(q); c=r.get('TiO_channel','?'); ch[c]=ch.get(c,0)+1
                comps.append({'size':int(len(idx)),'fraction':float(len(idx)/n),'database_rows':rows,
                              'external_references':refs0,'TiO_channel_counts':ch})
            comps.sort(key=lambda x:x['size'],reverse=True)
            cuts.append({'removed_largest_edges':cutn,'components':int(nc),'component_summaries':comps})
        mst={'max_edge':float(edges[0][0]) if edges else 0.0,
             'edge_q50':float(np.quantile(evals,.5)) if len(evals) else 0.0,
             'edge_q90':float(np.quantile(evals,.9)) if len(evals) else 0.0,
             'largest_bottlenecks':bott,'successive_bottleneck_cuts':cuts}
    else: mst={'max_edge':0.0,'edge_q50':0.0,'edge_q90':0.0,'largest_bottlenecks':[],'successive_bottleneck_cuts':[]}
    # Reference support in latent space. External references are explicit pool sources;
    # database-matched references use all matched rows from the Ti-O report.
    refs={}
    for name,r in tio_report.get('references',{}).items():
        idx=[i for i,m in enumerate(meta) if m['source']==f'ref:{name}']
        rows=set(int(x) for x in r.get('matched_database_rows',[]))
        idx += [i for i,m in enumerate(meta) if m['source'].startswith('db:') and int(m['source'].split(':')[1]) in rows]
        idx=sorted(set(idx))
        if not idx: continue
        # medoid under latent cosine distance is more faithful than averaging unit vectors.
        sub=D[np.ix_(idx,idx)]; med=idx[int(np.argmin(sub.sum(axis=1)))]
        others=[j for j in range(n) if j not in set(idx)]
        near=sorted((float(D[med,j]),j) for j in others)[:8]
        refs[name]={'site_count':len(idx),'sources':sorted(set(meta[i]['source'] for i in idx)),
                    'medoid_index':int(med),'nearest_nonreference_sites':[
                        {'distance':d,'index':int(j),'source':meta[j]['source'],'site':int(meta[j]['site'])} for d,j in near]}
    return {'site_count':int(n),'latent_dimension':int(Z.shape[1]),'metric':'1-cosine(NN latent)',
            'pca':pca,'pairwise_distance_quantiles':pair,'knn_connectivity':knn,
            'epsilon_connectivity':eps,'mst':mst,'reference_support':refs,
            'interpretation_rule':'Use persistent disconnected components / large MST bottlenecks as evidence for separate Ti manifolds. Do not infer modes from a single perturbation-radius complete-linkage cut.'}


def _apply_metric_info_to_X(X, info):
    """Apply an in-memory v34 NN state_dict to arbitrary SOAP descriptors."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    X=np.asarray(X,float)
    arch=info.get('architecture',[]); state=info.get('state_dict',{})
    if len(arch)!=4 or not state or X.ndim!=2:
        raise ValueError('v39 requires a complete NN architecture/state_dict')
    din,h1,h2,ld=[int(x) for x in arch]
    class MetricNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(din,h1),nn.LayerNorm(h1),nn.GELU(),
                                   nn.Linear(h1,h2),nn.LayerNorm(h2),nn.GELU(),nn.Linear(h2,ld))
        def forward(self,x): return F.normalize(self.net(x),p=2,dim=-1,eps=1e-8)
    net=MetricNet()
    sd={k:torch.tensor(v,dtype=torch.float32) for k,v in state.items()}
    net.load_state_dict(sd); net.eval()
    with torch.no_grad(): return net(torch.tensor(X,dtype=torch.float32)).cpu().numpy()


def _nearest_cosine_cross(A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    if len(A)==0 or len(B)==0: return np.full(len(A),np.inf)
    return np.maximum(0.0,1.0-np.max(A@B.T,axis=1))


def _nearest_cosine_self(A):
    A=np.asarray(A,float)
    if len(A)<2: return np.full(len(A),np.inf)
    S=A@A.T; np.fill_diagonal(S,-np.inf)
    return np.maximum(0.0,1.0-np.max(S,axis=1))


def _v34r5_leave_one_structure_out(X, meta, perturb, learned_radius, seeds=(3401,3417,3433)):
    """Structure-level LOO with independent raw/NN attribution.

    A structure is called a standout only when raw SOAP and the retrained NN both put it
    outside ordinary training support. Raw-close/latent-far cases are explicitly MODEL_FAIL.
    Results are also retained per Ti site so one anomalous site cannot label a whole crystal.
    """
    X=np.asarray(X,float); Xa=np.asarray(perturb['augmented_X'],float); am=perturb['augmented_meta']
    db_sources=sorted({m['source'] for m in meta if m['source'].startswith('db:')}, key=lambda z:int(z.split(':')[1]))
    results=[]
    for source in db_sources:
        hold=np.asarray([i for i,m in enumerate(meta) if m['source']==source],int)
        train=np.asarray([i for i,m in enumerate(meta) if m['source']!=source],int)
        if not len(hold) or len(train)<2: continue
        raw_hold=_nearest_cosine_cross(X[hold],X[train]); raw_holes=_nearest_cosine_self(X[train])
        raw_q90=float(np.quantile(raw_holes[np.isfinite(raw_holes)],.90)); raw_q99=float(np.quantile(raw_holes[np.isfinite(raw_holes)],.99))
        seed_site=[]; latent_hole_q90=[]; latent_hole_q99=[]
        # subset perturbation cloud and remap base metadata naturally through source/site keys
        keep_aug=np.asarray([j for j,m in enumerate(am) if m.get('source')!=source],int)
        psub={'augmented_X':Xa[keep_aug], 'augmented_meta':[am[j] for j in keep_aug], 'cycles':perturb.get('cycles',[])}
        mtrain=[meta[i] for i in train]
        for sd in seeds:
            Ztr,_,info=_learn_nn_perturbation_metric(X[train],mtrain,psub,seed=int(sd),epochs=300)
            Zh=_apply_metric_info_to_X(X[hold],info)
            ds=_nearest_cosine_cross(Zh,Ztr); holes=_nearest_cosine_self(Ztr)
            seed_site.append(ds)
            latent_hole_q90.append(float(np.quantile(holes[np.isfinite(holes)],.90)))
            latent_hole_q99.append(float(np.quantile(holes[np.isfinite(holes)],.99)))
        seed_site=np.vstack(seed_site)
        lat_med=np.median(seed_site,axis=0); lat_mean=np.mean(seed_site,axis=0); lat_std=np.std(seed_site,axis=0)
        seed_cv=lat_std/np.maximum(lat_mean,1e-8)
        lq90=float(np.median(latent_hole_q90)); lq99=float(np.median(latent_hole_q99))
        site_rows=[]
        for jj,bi in enumerate(hold):
            raw_far=bool(raw_hold[jj]>raw_q99)
            lat_far=bool(lat_med[jj]>max(lq99,float(learned_radius)))
            stable=bool(seed_cv[jj]<0.25)
            if raw_far and lat_far and stable: verdict='STANDOUT'
            elif (not raw_far) and lat_far: verdict='MODEL_FAIL'
            elif raw_far and not lat_far: verdict='MODEL_COMPRESSES_RAW_NOVELTY'
            elif raw_hold[jj]>raw_q90 or lat_med[jj]>max(lq90,float(learned_radius)): verdict='BORDERLINE_SAMPLING_HOLE'
            else: verdict='WELL_RECONSTRUCTED'
            site_rows.append({'site':int(meta[bi]['site']),'raw_nearest':float(raw_hold[jj]),'raw_over_train_q99':float(raw_hold[jj]/max(raw_q99,1e-12)),
                              'latent_nearest_median':float(lat_med[jj]),'latent_over_0p10A_q90':float(lat_med[jj]/max(float(learned_radius),1e-12)),
                              'seed_cv':float(seed_cv[jj]),'verdict':verdict})
        counts={v:sum(q['verdict']==v for q in site_rows) for v in sorted(set(q['verdict'] for q in site_rows))}
        standout_frac=counts.get('STANDOUT',0)/len(site_rows); fail_frac=counts.get('MODEL_FAIL',0)/len(site_rows)
        if standout_frac>=0.5: overall='STRUCTURE_STANDOUT'
        elif fail_frac>=0.5: overall='MODEL_FAILURE'
        elif counts.get('WELL_RECONSTRUCTED',0)/len(site_rows)>=0.5: overall='WELL_RECONSTRUCTED'
        else: overall='MIXED_OR_SAMPLING_HOLE'
        results.append({'source':source,'row':int(source.split(':')[1]),'Ti_sites':len(site_rows),'overall_verdict':overall,
                        'site_verdict_counts':counts,'raw_training_hole_q90':raw_q90,'raw_training_hole_q99':raw_q99,
                        'latent_training_hole_q90_median':lq90,'latent_training_hole_q99_median':lq99,
                        'latent_0p10A_perturb_q90':float(learned_radius),'seed_count':len(seeds),'sites':site_rows})
    return {'method':'leave-one-database-structure-out; retrain NN from scratch for each holdout and seed',
            'seeds':[int(x) for x in seeds],
            'novelty_rule':'STANDOUT requires raw SOAP novelty + NN latent novelty + cross-seed stability. NN distance alone never establishes novelty.',
            'structure_rule':'A whole structure is called standout/model-failure only when >=50% of its Ti sites support that verdict.',
            'rows':results}

def _learn_titi_multimode_fields(byid, refs, tio_report, center='Ti', rmax=5.5, sigma=0.28, nmax=12, lmax=10,
                                  allowed_perturb_A=0.10, perturb_samples=4, seed=3301,
                                  exponent=2.0, lam=1e-3, min_mode_sites=2, diagnostics=False):
    """Learn the production Ti-Ti latent manifold; diagnostics are on by default.

    Production mode computes exactly what generation needs: base SOAP descriptors, the
    perturbation cloud, one NN metric, the 0.10-A calibration radius, and observed/reference
    latent anchors.  Complete-linkage, MST/PCA topology, mode reports, and LOSO retraining
    run by default and can be disabled explicitly with ``--no-diagnostics``.
    """
    timings={}; t_all=time.perf_counter()
    # v39 split contract: references define chemistry channels/targets; the entire database
    # supplies statistical Ti-Ti population coverage. References are also retained as explicit anchors.
    pool=[(f'db:{rid}',st) for rid,st in sorted(byid.items())] + [(f'ref:{name}',st) for name,st in refs.items()]

    fluct_radius=None; fluct_samples=[]
    if diagnostics:
        t=time.perf_counter()
        fluct_radius,fluct_samples=_calibrate_titi_descriptor_radius(pool,center,rmax,sigma,nmax,lmax,
                                                                     allowed_perturb_A,perturb_samples,seed)
        timings['raw_descriptor_calibration']=time.perf_counter()-t

    t=time.perf_counter(); X=[]; meta=[]; field_envs=[]; base_lookup={}
    for source,st in pool:
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=center: continue
            x,cn,env=_titi_spherical_soap_descriptor(st,i,center,rmax,sigma,nmax,lmax)
            X.append(x); meta.append({'source':source,'site':int(i),'CN_within_rmax':int(cn)})
            field_envs.append(env); base_lookup[(source,int(i))]=(np.asarray(x,float),int(cn))
    X=np.vstack(X) if X else np.zeros((0,1)); timings['base_descriptors']=time.perf_counter()-t

    t=time.perf_counter()
    perturb=_titi_perturbation_training_cycles(pool,center,rmax,sigma,nmax,lmax,(0.02,0.05,allowed_perturb_A),
                                                perturb_samples,seed+7919,base_lookup=base_lookup)
    X_aug=perturb['augmented_X']; timings['perturbation_cloud']=time.perf_counter()-t

    t=time.perf_counter()
    Xnn,Xann,metric_info=_learn_nn_perturbation_metric(X,meta,perturb,seed+12031)
    metric_info['retrained']=True; timings['nn_training']=time.perf_counter()-t
    cyc10=[c for c in metric_info.get('cycles',[]) if abs(c['amplitude_A']-allowed_perturb_A)<1e-8]
    if not cyc10:
        raise RuntimeError(f'v39 Ti-Ti NN did not produce the required {allowed_perturb_A:.3f} A perturbation calibration')
    learned_radius=max(float(cyc10[0]['distance_q90']),1e-6)

    simple_labels=np.asarray([],int); aug_labels=np.asarray([],int)
    learned_labels=np.asarray([],int); learned_aug_labels=np.asarray([],int)
    topology=None; loo=None; clean=[]; site_scores=[]; descendant=[]
    if diagnostics:
        t=time.perf_counter()
        simple_labels=_strict_complete_modes(X,float(fluct_radius)) if len(X) else np.asarray([],int)
        aug_labels=_strict_complete_modes(X_aug,float(fluct_radius)) if len(X_aug) else np.asarray([],int)
        learned_labels=_strict_complete_modes(Xnn,learned_radius) if len(Xnn) else np.asarray([],int)
        learned_aug_labels=_strict_complete_modes(Xann,learned_radius) if len(Xann) else np.asarray([],int)
        timings['complete_linkage']=time.perf_counter()-t

        t=time.perf_counter(); topology=_latent_topology_diagnostic(Xnn,meta,perturb,Xann,learned_radius,tio_report,Xraw=X)
        timings['topology']=time.perf_counter()-t
        t=time.perf_counter(); loo=_v34r5_leave_one_structure_out(X,meta,perturb,learned_radius)
        timings['leave_one_structure_out']=time.perf_counter()-t

        t=time.perf_counter(); modes=[]
        for mid in sorted(set(learned_labels.tolist())):
            idx=np.where(learned_labels==mid)[0]
            if len(idx)<min_mode_sites: continue
            Xm=Xnn[idx]; model=_v30r2_bayes_fit(Xm,exponent,lam); own=_v30r2_bayes_score(Xm,model)
            sources=sorted(set(meta[i]['source'] for i in idx))
            dbrows=sorted(int(x.split(':')[1]) for x in sources if x.startswith('db:'))
            refsupp=sorted(x.split(':',1)[1] for x in sources if x.startswith('ref:'))
            envlist=[field_envs[i] for i in idx]
            cns=np.asarray([len(e.vectors) for e in envlist],int)
            rr=np.concatenate([e.distances for e in envlist if len(e.vectors)]) if any(len(e.vectors) for e in envlist) else np.asarray([])
            aa=np.concatenate([e.angles for e in envlist if len(e.angles)]) if any(len(e.angles) for e in envlist) else np.asarray([])
            cn_summary={'rmax_A':float(rmax),'median':float(np.median(cns)),'q10':float(np.quantile(cns,.1)),'q90':float(np.quantile(cns,.9))}
            radial={'rmax_A':float(rmax),'bond_q10_A':float(np.quantile(rr,.1)) if len(rr) else None,
                    'bond_q50_A':float(np.quantile(rr,.5)) if len(rr) else None,'bond_q90_A':float(np.quantile(rr,.9)) if len(rr) else None,
                    'angle_q10_deg':float(np.quantile(aa,.1)) if len(aa) else None,
                    'angle_q50_deg':float(np.quantile(aa,.5)) if len(aa) else None,'angle_q90_deg':float(np.quantile(aa,.9)) if len(aa) else None}
            angle_corr={'rmax_A':float(rmax),'fixed_CN_rank_correlations':_rank_angle_correlations(envlist)}
            modes.append({'id':f'T{len(modes)+1}','site_count':int(len(idx)),'source_count':len(sources),
                          'database_rows':dbrows,'external_references':refsupp,'CN_within_rmax':cn_summary,
                          'marginal_diagnostics':radial,'high_order_angle_diagnostics':angle_corr,
                          'within_mode_q90_bayesian_uncertainty':float(np.quantile(own,.9)),
                          '_indices':idx.tolist(),'_model':model})
        for i,x in enumerate(Xnn):
            vals=[]
            for m in modes:
                u=float(_v30r2_bayes_score(x[None,:],m['_model'])[0]); vals.append((u,m['id']))
            if vals:
                u,mid=min(vals); site_scores.append({**meta[i],'nearest_mode':mid,'bayesian_U':u})
        clean=[{k:v for k,v in m.items() if not k.startswith('_')} for m in modes]
        if len(learned_aug_labels)>=len(X):
            lookup={(m['source'],int(m['site'])):i for i,m in enumerate(meta)}; byamp={}
            for j,m in enumerate(perturb['augmented_meta']):
                if m.get('kind')!='perturbed': continue
                i=lookup.get((m['source'],int(m['site'])))
                if i is None: continue
                a=float(m['amplitude_A']); byamp.setdefault(a,[]).append(bool(learned_aug_labels[j]==learned_aug_labels[i]))
            for a,v in sorted(byamp.items()): descendant.append({'amplitude_A':a,'samples':len(v),'same_mode_fraction':float(np.mean(v))})
        timings['mode_reports']=time.perf_counter()-t

    perturb_comparison={
        'simple_base_mode_count':int(len(set(simple_labels.tolist()))) if diagnostics and len(simple_labels) else None,
        'perturbation_augmented_mode_count':int(len(set(aug_labels.tolist()))) if diagnostics and len(aug_labels) else None,
        'base_site_count':int(len(X)),'augmented_site_count':int(len(X_aug)),'cycles':perturb['cycles'],
        'learned_nn_metric':metric_info,
        'learned_base_mode_count':int(len(set(learned_labels.tolist()))) if diagnostics and len(learned_labels) else None,
        'learned_augmented_mode_count':int(len(set(learned_aug_labels.tolist()))) if diagnostics and len(learned_aug_labels) else None,
        'learned_complete_linkage_threshold':float(learned_radius) if diagnostics else None,
        'same_site_descendant_retention':descendant,
        'interpretation':'Production metric is the self-supervised NN latent distance. Complete-linkage diagnostics judge the learned geometry by default but never define generation support.'}
    timings['total_titi_learning']=time.perf_counter()-t_all
    return {'method':'continuous spherical-harmonic Ti<-Ti SOAP -> self-supervised NN latent manifold; no interpolation',
            'diagnostics_enabled':bool(diagnostics),'timing_seconds':timings,
            'training_pool_sources':[x for x,_ in pool],
            'descriptor':{'type':'single-cutoff spherical-harmonic SOAP power spectrum','outer_cutoff_A':float(rmax),'radial_basis_nmax':int(nmax),'angular_lmax':int(lmax),'sigma_A':float(sigma),'dimension':int(_v30_soap_dim(nmax,lmax)),'internal_hard_shells':False,'cross_radial_angular_coupling':True},
            'mode_discovery':{'distance':'1-cosine(NN latent SOAP)','complete_linkage_threshold':float(learned_radius) if diagnostics else None,
                              'operational_perturbation_radius':float(learned_radius),
                              'threshold_meaning':f'learned-metric q90 displacement under {allowed_perturb_A:.3f} A symmetry-preserving perturbations',
                              'perturbation_samples_total':len(fluct_samples) if diagnostics else 0,
                              'no_transitive_bridging':True,'no_interpolation':True},
            'perturbation_informed_comparison':perturb_comparison,
            'latent_topology_diagnostic':topology,
            'leave_one_structure_out_generalization':loo,
            'generation_manifold_export':{
                'metric':'1-cosine(NN latent)','latent_anchors':np.asarray(Xnn,float).tolist(),'anchor_meta':meta,
                'allowed_distance_q90_0p10A':float(learned_radius),
                'note':'Exact learned Ti-Ti latent training support for differentiable Wyckoff optimization; anchors are observed/reference environments, never interpolated centers.'},
            'mode_count':len(clean) if diagnostics else None,'modes':clean,'training_site_scores':site_scores,
            'potential_rule':'Generation minimizes excess distance of each Ti environment to the nearest observed/reference latent anchor, with a flat bottom inside the perturbation-calibrated radius. Never interpolate descriptors or mode centers.'}


def _learn_reference_local_force_models(refs, center='Ti', partner='O'):
    """Reference-only local TiO6/OTi3 force targets. No ideal angles or bond lengths."""
    tio=[]; oti=[]
    for name,st in refs.items():
        # Ti-centered: six nearest O vectors
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=center: continue
            nn=_periodic_neighbors(st,i,partner,4.0)
            if len(nn)<6: continue
            v=np.vstack([q['vector'] for q in sorted(nn,key=lambda q:q['distance'])[:6]])
            r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],1e-12)
            cos=np.sort((u@u.T)[np.triu_indices(6,1)])
            tio.append({'source':name,'r':np.sort(r).tolist(),'cos':cos.tolist(),'vectors':v.tolist()})
        # O-centered: three nearest Ti vectors
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=partner: continue
            nn=_periodic_neighbors(st,i,center,4.0)
            if len(nn)<3: continue
            v=np.vstack([q['vector'] for q in sorted(nn,key=lambda q:q['distance'])[:3]])
            r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],1e-12)
            cos=np.sort((u@u.T)[np.triu_indices(3,1)])
            oti.append({'source':name,'r':np.sort(r).tolist(),'cos':cos.tolist(),'vectors':v.tolist()})
    if not tio or not oti: raise RuntimeError('Reference-only local-force learning found no TiO6/OTi3 environments')
    allr=np.array([x for q in tio for x in q['r']],float)
    centers=np.linspace(max(.5,float(allr.min())-.25),float(allr.max())+.25,6)
    width=max(.10,float(np.std(allr))*1.5)
    def so3(v):
        v=np.asarray(v,float)[None,None,:,:]
        # numpy equivalent of the rotational power basis used by generator
        r=np.linalg.norm(v,axis=-1); u=v/np.maximum(r[...,None],1e-12); dots=np.einsum('bnji,bnki->bnjk',u,u)
        rb=np.exp(-.5*((r[...,None]-centers)/width)**2); K=v.shape[2]
        P=[np.ones_like(dots),dots]
        for l in range(2,7): P.append(((2*l-1)*dots*P[-1]-(l-1)*P[-2])/l)
        ps=[]
        for Pl in P:
            for n in range(len(centers)):
                for m in range(n,len(centers)):
                    ps.append(np.einsum('bnj,bnk,bnjk->bn',rb[...,n],rb[...,m],Pl)/(K*K))
        x=np.stack(ps,axis=-1)[0,0]; return x/max(float(np.linalg.norm(x)),1e-12)
    so3_t=np.vstack([so3(q['vectors']) for q in tio]); so3_o=np.vstack([so3(q['vectors']) for q in oti])
    return {
      'source':'authoritative_reference_cifs_only','no_ideal_geometry':True,
      'v140_style':{
        'tio6_anchors':[{'source':q['source'],'r':q['r'],'cos':q['cos']} for q in tio],
        'oti3_anchors':[{'source':q['source'],'r':q['r'],'cos':q['cos']} for q in oti],
        'radial_scale_A':max(.04,float(np.std(allr))), 'angular_scale':.10},
      'so3':{'centers_A':centers.tolist(),'width_A':float(width),'lmax':6,
             'tio6_anchors':so3_t.tolist(),'oti3_anchors':so3_o.tolist(),
             'scale':max(.02,float(np.median(np.std(so3_t,axis=0))))}}

def _build_generation_contract(byid, refs, report, titi, center='Ti', partner='O'):
    """Export the single-file learner->generator contract used by generator v143+.

    Everything here is derived in this learner run: no legacy chemistry model is read or
    embedded.  The contract deliberately separates (1) Ti-O reference support, (2) the
    continuous Ti-Ti latent manifold, and (3) broad physical conditioning scales needed
    to give Wyckoff optimization gradients outside the learned local support.
    """
    from math import gcd
    # Require one consistent binary reduced composition across authoritative references.
    reduced=[]
    for name,st in refs.items():
        comp=st.composition.get_el_amt_dict()
        if center not in comp or partner not in comp or len(comp)!=2:
            raise ValueError(f'Generation contract requires binary {center}/{partner} references; {name} has {comp}')
        a=float(comp[center]); b=float(comp[partner]); ratio=b/max(a,1e-12)
        reduced.append((name,ratio))
    ratios=np.asarray([x[1] for x in reduced],float)
    if np.ptp(ratios)>1e-6:
        raise ValueError(f'Inconsistent reference stoichiometry: {reduced}')
    ratio=float(np.median(ratios))
    # Current generator formula parser is integer-valued.  Keep that contract explicit.
    partner_per_center=int(round(ratio))
    if abs(ratio-partner_per_center)>1e-6:
        raise ValueError(f'Non-integer {partner}/{center} ratio {ratio} is not supported by generator v143')
    formula_unit={center:1, partner:partner_per_center}
    center_cn=int(round(np.median([r['reference_CN'] for r in report['references'].values()])))
    partner_cn=int(round(center_cn/max(partner_per_center,1)))

    # Exact differentiable Ti<-O Bayesian reference fields.  Recompute and serialize
    # the same normalized SOAP reference matrices/K^-1 used by the selector.
    bm=report['bayesian_metric']; ref_fields={}
    tio_bonds=[]
    for name,st in refs.items():
        cutoff=float(report['references'][name]['first_shell_cutoff_A'])
        envs=_v30_center_partner_envs(st,center,partner,cutoff,f'genref:{name}')
        F=np.vstack([_v30_soap_vec(e,cutoff,float(bm['soap_sigma_A'])) for _,e in envs])
        obj=_v30r2_bayes_fit(F,float(bm['kernel_exponent']),float(bm['lambda']))
        thr=float(report['reference_fluctuation_calibration'][name]['acceptance_q90U'])
        ref_fields[name]={
            'cutoff_A':cutoff,'reference_CN':int(report['references'][name]['reference_CN']),
            'acceptance_q90U':thr,'X':np.asarray(obj['X'],float).tolist(),
            'Kinv':np.asarray(obj['Kinv'],float).tolist(),'kernel_exponent':float(obj['exponent']),
            'lambda':float(obj['lam'])}
        for _,e in envs: tio_bonds.extend(np.linalg.norm(np.asarray(e.vectors,float),axis=1).tolist())

    # v39 statistical layer: use the entire database for global Ti-O bond statistics.
    # This does NOT define channels or reference identities.
    for rid,st in sorted(byid.items()):
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=center: continue
            nn=_periodic_neighbors(st,i,partner,4.0)
            if nn:
                tio_bonds.extend([q['distance'] for q in sorted(nn,key=lambda q:q['distance'])[:center_cn]])
    tb=np.asarray(tio_bonds,float)
    if tb.size==0: raise RuntimeError('No Ti-O bonds available for generation contract')
    tio_mu=float(np.median(tb)); tio_sig=max(float(1.4826*np.median(np.abs(tb-tio_mu))),0.03)

    # v39 statistical layer: broad entrance conditioning is estimated from the full DB.
    # Keep references in the pool as authoritative support examples, without letting them redefine statistics alone.
    pool=[(f'db:{rid}',st) for rid,st in sorted(byid.items())] + [(f'ref:{name}',st) for name,st in refs.items()]

    def nearest_same_species(st,species,radius):
        vals=[]
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=species: continue
            ds=[x['distance'] for x in _periodic_neighbors(st,i,species,radius) if x['distance']>1e-6]
            if ds: vals.append(min(ds))
        return vals
    # Broad Ti-framework conditioning statistics used only for fast entrance capture.
    # These are learned from the same accepted/reference pool and never define final
    # Ti-Ti manifold similarity.  Rank radii give a cheap smooth cumulative N(R) field.
    rank_keys=(6,12,16,20,24); rank_samples={k:[] for k in rank_keys}
    titi_nn=[]; oo_nn=[]; volumes=[]
    for _,st in pool:
        titi_nn.extend(nearest_same_species(st,center,8.0))
        oo_nn.extend(nearest_same_species(st,partner,5.0))
        nc=sum(str(x.specie.symbol)==center for x in st)
        if nc: volumes.append(float(st.volume/nc))
        for i,site in enumerate(st):
            if str(site.specie.symbol)!=center: continue
            ds=sorted(x['distance'] for x in _periodic_neighbors(st,i,center,8.0) if x['distance']>1e-6)
            for k in rank_keys:
                if len(ds)>=k: rank_samples[k].append(float(ds[k-1]))
    titi_nn=np.asarray(titi_nn,float); oo_nn=np.asarray(oo_nn,float); volumes=np.asarray(volumes,float)
    if not len(titi_nn) or not len(oo_nn) or not len(volumes):
        raise RuntimeError('Generation contract lacks same-species/volume support')
    titi_refmin=float(np.quantile(titi_nn,0.01)); oo_refmin=float(np.quantile(oo_nn,0.01))
    titi_hard=max(0.5,titi_refmin-0.08); oo_hard=max(0.5,oo_refmin-0.08)
    vmed=float(np.median(volumes)); vq10=float(np.quantile(volumes,.10)); vq90=float(np.quantile(volumes,.90))
    density_profile=[]
    for k in rank_keys:
        vals=np.asarray(rank_samples[k],float)
        if len(vals):
            density_profile.append({'rank':int(k),'radius_A':float(np.median(vals)),
                                    'q10_A':float(np.quantile(vals,.10)),'q90_A':float(np.quantile(vals,.90)),
                                    'samples':int(len(vals))})
    if not density_profile:
        raise RuntimeError('Generation contract could not learn Ti-Ti cumulative density profile')
    coord_cut=float(max(np.quantile(tb,.99)+0.05, min(max(r['first_shell_cutoff_A'] for r in report['references'].values()), tio_mu+0.50)))

    pc=titi['perturbation_informed_comparison']; nn=pc['learned_nn_metric']; exp=titi['generation_manifold_export']
    max_train_cn=max([int(m.get('CN_within_rmax',0)) for m in exp.get('anchor_meta',[])]+[1])
    # The generator can safely truncate the O(K^2) angular tensor to this many nearest
    # periodic Ti candidates.  +8 leaves headroom above every learner/reference environment.
    descriptor_neighbor_cap=int(max_train_cn+8)
    contract={
        'contract_version':2,
        'chemistry':{
            'parent_element':center,'child_element':partner,'formula_unit':formula_unit,
            'parent_coordination':center_cn,'child_coordination':partner_cn,
            'target_example':''.join([center, partner+(str(partner_per_center) if partner_per_center!=1 else '')])},
        'pair_scales':{
            'parent_child_mu_A':tio_mu,'parent_child_sigma_A':tio_sig,
            'coordination_cutoff_A':coord_cut,
            'parent_parent_reference_min_A':titi_refmin,'parent_parent_hard_min_A':titi_hard,
            'child_child_reference_min_A':oo_refmin,'child_child_hard_min_A':oo_hard},
        'conditioning':{
            'volume_per_parent_median_A3':vmed,'volume_per_parent_q10_A3':vq10,'volume_per_parent_q90_A3':vq90,
            'titi_cumulative_density_profile':density_profile,
            'density_shell_width_A':0.18,
            'source_count':len(pool)},
        'tio_reference_field':{
            'descriptor':{'type':'normalized spherical SOAP power spectrum','nmax':6,'lmax':6,
                          'sigma_A':float(bm['soap_sigma_A']),'radial_center_rule':'linspace(max(0.35,0.18*cutoff), cutoff, nmax)',
                          'smooth_outer_cutoff':'0.5*(cos(pi*r/cutoff)+1)'},
            'references':ref_fields,
            'selection_rule':'For each Ti site, evaluate each reference Bayesian support variance and divide by that reference perturbation-calibrated acceptance_q90U; nearest reference wins. D<=1 is inside learned support. No interpolation.'},
        'titi_manifold':{
            'descriptor':titi['descriptor'],
            'nn_metric':nn,
            'latent_anchors':exp['latent_anchors'],
            'anchor_meta':exp['anchor_meta'],
            'allowed_distance_q90_0p10A':float(exp['allowed_distance_q90_0p10A']),
            'max_training_neighbors_within_rmax':int(max_train_cn),
            'descriptor_neighbor_cap':int(descriptor_neighbor_cap),
            'rule':'Use nearest observed/reference latent anchor only; flat-bottom excess outside allowed radius; no interpolated centers.'},
        'local_force_models':_learn_reference_local_force_models(refs,center,partner),
        'provenance':{
            'selected_database_rows':report['selected_rows'],
            'database_role':'full_database_statistical_layer; references_only_define_channels_and_local_targets',
            'reference_only_learning':False,
            'external_references':list(refs.keys()),
            'reference_directory':report['reference_directory']}
    }
    return contract


# -----------------------------------------------------------------------------
# v40 restoration laboratory
# -----------------------------------------------------------------------------
def _v40_torch_cell(structure, device):
    return torch.tensor(np.asarray(structure.lattice.matrix,float),dtype=torch.float64,device=device)


def _v40_reference_state(structure, device):
    cell=_v40_torch_cell(structure,device)
    frac=torch.tensor(np.asarray(structure.frac_coords,float),dtype=torch.float64,device=device)
    species=[str(x.specie.symbol) for x in structure]
    return cell,frac,species


def _v40_pair_vectors(frac, cell, center_ids, neigh_ids):
    """All 27-image vectors center->neighbor; differentiable in fractional coordinates."""
    dev=frac.device
    shifts=torch.tensor([(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)],dtype=frac.dtype,device=dev)
    c=frac[center_ids]
    n=frac[neigh_ids]
    df=n[None,:,None,:]+shifts[None,None,:,:]-c[:,None,None,:]
    return torch.einsum('cnsd,dk->cnsk',df,cell)


def _v40_knn_vectors(frac, cell, species, center_element, neighbor_element, k):
    ci=[i for i,x in enumerate(species) if x==center_element]
    ni=[i for i,x in enumerate(species) if x==neighbor_element]
    if not ci or not ni: return None,None
    v=_v40_pair_vectors(frac,cell,ci,ni)
    d=torch.linalg.norm(v,dim=-1)
    # Remove the zero self-image without mutating an autograd-tracked norm tensor.
    # In-place writes here invalidate LinalgVectorNormBackward0 on PyTorch 2.x.
    if center_element==neighbor_element:
        same=torch.tensor([[ii==jj for jj in ni] for ii in ci],dtype=torch.bool,device=frac.device)
        image13=torch.zeros((27,),dtype=torch.bool,device=frac.device); image13[13]=True
        self_mask=same[:,:,None] & image13[None,None,:]
        d=d.masked_fill(self_mask,1e6)
    flatd=d.reshape(len(ci),-1); flatv=v.reshape(len(ci),-1,3)
    kk=min(int(k),flatd.shape[1])
    nd,ix=torch.topk(flatd,kk,dim=1,largest=False)
    nv=torch.gather(flatv,1,ix[...,None].expand(-1,-1,3))
    return nd,nv


def _v40_power(nv, centers, width, lmax):
    r=torch.linalg.norm(nv,dim=-1).clamp_min(1e-9); u=nv/r[...,None]
    dots=torch.einsum('nki,nli->nkl',u,u).clamp(-1.0,1.0)
    rb=torch.exp(-0.5*((r[...,None]-centers[None,None,:])/max(float(width),1e-8))**2)
    P=[torch.ones_like(dots)]
    if lmax>=1: P.append(dots)
    for l in range(2,int(lmax)+1): P.append(((2*l-1)*dots*P[-1]-(l-1)*P[-2])/float(l))
    vals=[]; nr=int(centers.numel()); K=max(int(nv.shape[1]),1)
    for Pl in P:
        for a in range(nr):
            for b in range(a,nr): vals.append(torch.einsum('nj,nk,njk->n',rb[...,a],rb[...,b],Pl)/float(K*K))
    x=torch.stack(vals,dim=-1)
    return x/torch.linalg.norm(x,dim=-1,keepdim=True).clamp_min(1e-9)


def _v40_anchor_energy(frac,cell,species,contract,parent,child):
    """Reference-only local geometry: nearest empirical anchor, never ideal geometry."""
    q=contract['local_force_models']['so3']; dev=frac.device
    centers=torch.tensor(q['centers_A'],dtype=frac.dtype,device=dev); width=float(q['width_A']); lmax=int(q['lmax'])
    terms=[]
    for ce,ne,k,key in ((parent,child,6,'tio6_anchors'),(child,parent,3,'oti3_anchors')):
        _,nv=_v40_knn_vectors(frac,cell,species,ce,ne,k)
        if nv is None or nv.shape[1]<k: continue
        fp=_v40_power(nv,centers,width,lmax)
        anchors=torch.tensor(q[key],dtype=frac.dtype,device=dev)
        d2=((fp[:,None,:]-anchors[None,:,:])**2).mean(-1)
        # Smooth nearest-anchor energy retains gradients at basin boundaries.
        tau=0.02
        e=-tau*torch.logsumexp(-d2/tau,dim=1)
        terms.append(e.mean())
    return sum(terms) if terms else frac.sum()*0.0


def _v40_coordination_energy(frac,cell,species,contract,parent,child):
    ps=contract['pair_scales']; mu=float(ps['parent_child_mu_A']); sig=max(float(ps['parent_child_sigma_A']),0.04)
    terms=[]
    for ce,ne,k in ((parent,child,int(contract['chemistry']['parent_coordination'])),(child,parent,int(contract['chemistry']['child_coordination']))):
        nd,_=_v40_knn_vectors(frac,cell,species,ce,ne,k)
        if nd is not None: terms.append(torch.nn.functional.smooth_l1_loss((nd-mu)/sig,torch.zeros_like(nd),beta=2.0))
    return sum(terms) if terms else frac.sum()*0.0


def _v40_packing_energy(frac,cell,species,contract,parent):
    """Broad learned Ti crowding/void field from empirical cumulative rank radii."""
    prof=contract['conditioning'].get('titi_cumulative_density_profile',[])
    if not prof: return frac.sum()*0.0
    kmax=max(int(x['rank']) for x in prof); nd,_=_v40_knn_vectors(frac,cell,species,parent,parent,kmax)
    if nd is None: return frac.sum()*0.0
    terms=[]
    for row in prof:
        k=min(int(row['rank']),nd.shape[1]); R=float(row['radius_A'])
        # Smooth cumulative count around the learned rank radius: detects both holes and crowding.
        count=torch.sigmoid((R-nd)/0.18).sum(1)
        z=(count-float(k))/max(1.5,0.20*k); terms.append(torch.nn.functional.smooth_l1_loss(z,torch.zeros_like(z),beta=1.0))
    return torch.stack(terms).mean()


def _v40_exclusion_energy(frac,cell,species,contract,parent,child):
    ps=contract['pair_scales']; total=frac.sum()*0.0
    for el,hard in ((parent,float(ps['parent_parent_hard_min_A'])),(child,float(ps['child_child_hard_min_A']))):
        nd,_=_v40_knn_vectors(frac,cell,species,el,el,1)
        if nd is not None: total=total+torch.relu((hard-nd)/max(hard,0.5)).pow(2).mean()*20.0
    return total


def _v40_energy(method,frac,cell,species,contract,parent,child):
    ex=_v40_exclusion_energy(frac,cell,species,contract,parent,child)
    if method=='coordination': return _v40_coordination_energy(frac,cell,species,contract,parent,child)+ex
    if method=='local_so3': return _v40_anchor_energy(frac,cell,species,contract,parent,child)+ex
    if method=='packing': return _v40_packing_energy(frac,cell,species,contract,parent)+ex
    if method=='local+packing': return _v40_anchor_energy(frac,cell,species,contract,parent,child)+0.35*_v40_packing_energy(frac,cell,species,contract,parent)+ex
    if method=='coord+local+packing': return 0.20*_v40_coordination_energy(frac,cell,species,contract,parent,child)+_v40_anchor_energy(frac,cell,species,contract,parent,child)+0.35*_v40_packing_energy(frac,cell,species,contract,parent)+ex
    raise KeyError(method)


def _v40_cart_error(frac,ref_frac,cell):
    df=frac-ref_frac; df=df-torch.round(df); dc=df@cell
    return torch.linalg.norm(dc,dim=1),dc


def _v40_damage(ref_frac,cell,amplitude,rng):
    n=len(ref_frac); dirs=rng.normal(size=(n,3)); dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
    mags=rng.uniform(float(amplitude[0]),float(amplitude[1]),size=(n,1)); cart=dirs*mags
    inv=np.linalg.inv(cell.detach().cpu().numpy()); df=cart@inv
    return (ref_frac+torch.tensor(df,dtype=ref_frac.dtype,device=ref_frac.device))%1.0


def _v40_force_cosine(method,frac,ref_frac,cell,species,contract,parent,child):
    x=frac.detach().clone().requires_grad_(True); e=_v40_energy(method,x,cell,species,contract,parent,child)
    g=torch.autograd.grad(e,x)[0]; force_cart=(-g)@torch.linalg.inv(cell).T
    _,err=_v40_cart_error(x,ref_frac,cell); target=-err
    fn=torch.linalg.norm(force_cart,dim=1); tn=torch.linalg.norm(target,dim=1)
    valid=(fn>1e-10)&(tn>1e-10); cos=torch.zeros_like(fn)
    cos[valid]=(force_cart[valid]*target[valid]).sum(1)/(fn[valid]*tn[valid])
    return cos.detach().cpu().numpy(),float(e.detach().cpu())


def _v40_restore(method,start,ref_frac,cell,species,contract,parent,child,steps,lr,max_step_A=0.12):
    x=start.detach().clone()
    invT=torch.linalg.inv(cell).T
    for _ in range(int(steps)):
        x=x.detach().requires_grad_(True); e=_v40_energy(method,x,cell,species,contract,parent,child)
        g=torch.autograd.grad(e,x)[0]; fcart=(-g)@invT
        norm=torch.linalg.norm(fcart,dim=1,keepdim=True).clamp_min(1e-12)
        step=float(lr)*fcart; sn=torch.linalg.norm(step,dim=1,keepdim=True)
        step=step*torch.clamp(float(max_step_A)/sn.clamp_min(1e-12),max=1.0)
        x=(x.detach()+step@torch.linalg.inv(cell))%1.0
    err,_=_v40_cart_error(x,ref_frac,cell)
    return x,err.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# v42 topology-local chemistry model selection
# -----------------------------------------------------------------------------
def _v41_numpy_knn(st, i, neighbor_element, nmax, rmax=8.0):
    nn=_periodic_neighbors(st,i,neighbor_element,rmax)
    nn=sorted(nn,key=lambda q:q['distance'])
    # same-species helper can return the central atom/image at ~0 in some backends.
    nn=[q for q in nn if float(q['distance'])>1e-6]
    return nn[:int(nmax)]


def _v41_infer_k(samples, fallback):
    """Infer first-shell population from the most persistent early radial gap."""
    rows=[np.asarray(x,float) for x in samples if len(x)>=2]
    if not rows: return int(fallback)
    m=min(min(len(x) for x in rows),10)
    A=np.vstack([x[:m] for x in rows]); med=np.median(A,axis=0)
    if m<2: return int(fallback)
    # Relative gap avoids preferring a large absolute gap at long range.
    gap=np.diff(med)/np.maximum(med[:-1],1e-8)
    lo=1; hi=min(m-1,max(int(fallback)+2,3))
    k=int(np.argmax(gap[lo-1:hi])+lo)
    # For the chemically established TiO/OTi topology the fallback is also a guard
    # against a noisy single-reference radial gap changing stoichiometric connectivity.
    if fallback in (3,6) and abs(k-int(fallback))>1: k=int(fallback)
    return max(1,k)


def _v41_feature_numpy(v, geometry):
    v=np.asarray(v,float); r=np.linalg.norm(v,axis=1); u=v/np.maximum(r[:,None],1e-12)
    if geometry=='bond': return np.sort(r)
    cos=np.sort((u@u.T)[np.triu_indices(len(v),1)]) if len(v)>1 else np.zeros(0)
    if geometry=='angle': return cos
    # 4-body: signed-volume magnitude for every neighbor triple, normalized by radii.
    if geometry=='four_body':
        vals=[]
        for a in range(len(v)-2):
            for b in range(a+1,len(v)-1):
                for c in range(b+1,len(v)):
                    vals.append(abs(float(np.dot(u[a],np.cross(u[b],u[c])))))
        return np.sort(np.asarray(vals,float))
    # n-body keeps radial + all mutual angular relations jointly.  No ideal target.
    if geometry=='n_body': return np.concatenate([np.sort(r),cos])
    raise KeyError(geometry)


def _v41_learn_local_geometry(refs,parent='Ti',child='O',outside_neighbors=3):
    specs=[('TiO',parent,child,6),('OTi',child,parent,3),('TiTi',parent,parent,4),('OO',child,child,4)]
    out={'source':'authoritative_reference_cifs','principle':'learn topology-local inside geometry plus decaying outside-shell influence','topologies':{}}
    for name,ce,ne,fallback in specs:
        raw=[]; meta=[]
        for refname,st in refs.items():
            for i,site in enumerate(st):
                if str(site.specie.symbol)!=ce: continue
                nn=_v41_numpy_knn(st,i,ne,12)
                if len(nn)>=2:
                    raw.append([float(q['distance']) for q in nn]); meta.append((refname,st,i,nn))
        k=_v41_infer_k(raw,fallback)
        env=[]; kth=[]; kp1=[]
        for refname,st,i,nn in meta:
            if len(nn)<k+1: continue
            vec=np.vstack([np.asarray(q['vector'],float) for q in nn[:k]])
            env.append((refname,vec)); kth.append(float(nn[k-1]['distance'])); kp1.append(float(nn[k]['distance']))
        if not env: raise RuntimeError(f'v41 found no {name} environments with K+1 neighbors')
        km=float(np.median(kth)); op=float(np.median(kp1)); boundary=0.5*(km+op)
        geoms={}
        for g in ('bond','angle','four_body','n_body'):
            feats=[_v41_feature_numpy(v,g) for _,v in env]
            dim=int(feats[0].size) if feats else 0
            # A body order is meaningful only when this topology has enough inside
            # neighbors to define at least one invariant.  E.g. four_body needs K>=3.
            if dim==0:
                geoms[g]={'supported':False,'anchors':[],'scale':[],'dim':0}
                continue
            X=np.vstack(feats)
            scale=np.maximum(np.std(X,axis=0),0.03 if g=='bond' else 0.04)
            geoms[g]={'supported':True,'anchors':X.tolist(),'scale':scale.tolist(),'dim':int(X.shape[1])}
        out['topologies'][name]={'center':ce,'neighbor':ne,'K':int(k),'outside_neighbors':int(outside_neighbors),
            'K_distance_median_A':km,'Kp1_distance_median_A':op,'boundary_A':boundary,
            'boundary_gap_A':float(op-km),'n_environments':len(env),'geometries':geoms}
    return out


def _v41_feature_torch(v,geometry):
    r=torch.linalg.norm(v,dim=1); u=v/r[:,None].clamp_min(1e-10)
    if geometry=='bond': return torch.sort(r).values
    G=u@u.T; iu=torch.triu_indices(len(v),len(v),offset=1,device=v.device); cos=torch.sort(G[iu[0],iu[1]]).values
    if geometry=='angle': return cos
    if geometry=='four_body':
        vals=[]
        for a in range(len(v)-2):
            for b in range(a+1,len(v)-1):
                for c in range(b+1,len(v)):
                    vals.append(torch.abs(torch.dot(u[a],torch.linalg.cross(u[b],u[c]))))
        if not vals:
            return torch.empty(0,dtype=v.dtype,device=v.device)
        return torch.sort(torch.stack(vals)).values
    if geometry=='n_body': return torch.cat([torch.sort(r).values,cos])
    raise KeyError(geometry)


def _v41_site_energy(frac,cell,species,site_i,model,geometry,smear_A):
    ce=model['center']; ne=model['neighbor']; k=int(model['K']); nout=int(model['outside_neighbors'])
    ids=torch.as_tensor([j for j,x in enumerate(species) if x==ne],dtype=torch.long,device=frac.device)
    if len(ids)<k+(1 if ce!=ne else 2): return frac.sum()*0.0
    ci=torch.as_tensor([int(site_i)],dtype=torch.long,device=frac.device)
    dv=_v40_pair_vectors(frac,cell,ci,ids)[0]                 # neighbor,image,xyz
    dd=torch.linalg.norm(dv,dim=-1)
    if ce==ne:
        # Mask all zero self images functionally; never mutate norm outputs in-place.
        zero=dd<1e-8; dd=dd.masked_fill(zero,1e6)
    flat=torch.argsort(dd.reshape(-1))[:k+nout]
    ni=torch.div(flat,dd.shape[1],rounding_mode='floor'); qi=flat%dd.shape[1]
    vec=dv[ni,qi]; dist=dd[ni,qi]
    inside=vec[:k]
    gm=model['geometries'][geometry]
    if not gm.get('supported',int(gm.get('dim',0))>0):
        raise ValueError(f"unsupported geometry {geometry} for {ce}->{ne} with K={k}")
    feat=_v41_feature_torch(inside,geometry)
    anchors=torch.as_tensor(gm['anchors'],dtype=frac.dtype,device=frac.device)
    scale=torch.as_tensor(gm['scale'],dtype=frac.dtype,device=frac.device).clamp_min(1e-5)
    z=(feat[None,:]-anchors)/scale[None,:]
    d2=(z*z).mean(1)
    # Soft nearest empirical geometry: mode preserving, no interpolation target.
    tau=0.12
    local=-tau*torch.logsumexp(-d2/tau,dim=0)
    b=float(model['boundary_A']); sm=max(float(smear_A),1e-3)
    # Inside atoms are fully active: if Kth escapes the learned shell, pull the shell closed.
    inside_escape=torch.nn.functional.softplus((dist[:k]-b)/sm).pow(2).mean()
    # Outside atoms are topology-aware. Intruders are strongly rejected; once outside the
    # boundary their influence decays exponentially instead of disappearing at a cutoff.
    outside=dist[k:k+nout]
    if len(outside):
        intrusion=torch.nn.functional.softplus((b-outside)/sm).pow(2)
        decay=torch.exp(-torch.relu(outside-b)/sm)
        outside_term=(intrusion+0.08*decay).mean()
    else: outside_term=local*0.0
    return local+inside_escape+outside_term


def _v41_damage_one(ref_frac,cell,idx,amp,rng):
    x=ref_frac.clone(); d=rng.normal(size=3); d/=max(np.linalg.norm(d),1e-12); d*=rng.uniform(float(amp[0]),float(amp[1]))
    df=d@np.linalg.inv(cell.detach().cpu().numpy()); delta=torch.as_tensor(df,dtype=x.dtype,device=x.device)
    mask=torch.nn.functional.one_hot(torch.tensor(int(idx),device=x.device),num_classes=len(x)).to(x.dtype)[:,None]
    return (x+mask*delta[None,:])%1.0


def _v41_site_force(model,geometry,smear,x,ref_frac,cell,species,idx):
    q=x.detach().clone().requires_grad_(True); e=_v41_site_energy(q,cell,species,idx,model,geometry,smear)
    g=torch.autograd.grad(e,q)[0]; f=(-g)@torch.linalg.inv(cell).T
    _,err=_v40_cart_error(q,ref_frac,cell); target=-err[idx]
    fn=torch.linalg.norm(f[idx]); tn=torch.linalg.norm(target)
    cos=((f[idx]*target).sum()/(fn*tn).clamp_min(1e-12)).detach()
    return float(cos.cpu()),float(e.detach().cpu())


def _v41_restore_site(model,geometry,smear,start,ref_frac,cell,species,idx,steps,lr):
    x=start.detach().clone(); inv=torch.linalg.inv(cell); invT=inv.T
    for _ in range(int(steps)):
        q=x.detach().requires_grad_(True); e=_v41_site_energy(q,cell,species,idx,model,geometry,smear); g=torch.autograd.grad(e,q)[0]
        fc=(-g)@invT
        # Local experiment: move only the damaged center. Neighbors provide the field.
        step=float(lr)*fc[idx]; sn=torch.linalg.norm(step); step=step*torch.clamp(torch.as_tensor(0.12,dtype=step.dtype,device=step.device)/sn.clamp_min(1e-12),max=1.0)
        mask=torch.nn.functional.one_hot(torch.tensor(int(idx),device=x.device),num_classes=len(x)).to(x.dtype)[:,None]
        x=(x.detach()+mask*(step@inv)[None,:])%1.0
    err,_=_v40_cart_error(x,ref_frac,cell)
    return x,float(err[idx].detach().cpu())


def _v41_boundary_score(frac,cell,species,idx,model):
    ce=model['center']; ne=model['neighbor']; k=int(model['K']); ids=torch.as_tensor([j for j,x in enumerate(species) if x==ne],dtype=torch.long,device=frac.device)
    dv=_v40_pair_vectors(frac,cell,torch.as_tensor([idx],device=frac.device),ids)[0]; dd=torch.linalg.norm(dv,dim=-1)
    if ce==ne: dd=dd.masked_fill(dd<1e-8,1e6)
    ds=torch.sort(dd.reshape(-1)).values; b=float(model['boundary_A'])
    # zero is perfect topological separation; positive is K escape or K+1 intrusion.
    return float((torch.relu(ds[k-1]-b)+torch.relu(b-ds[k])).detach().cpu())


def _v41_geometry_benchmark(refs,learned,parent,child,smears,damage_min,damage_max,trials,steps,lr,seed,device,max_sites):
    rng=np.random.default_rng(int(seed)); dev=torch.device(device); rows=[]
    for refname,st in refs.items():
        cell,rf,species=_v40_reference_state(st,dev)
        for topo,model in learned['topologies'].items():
            centers=[i for i,x in enumerate(species) if x==model['center']][:int(max_sites)]
            for idx in centers:
                for trial in range(int(trials)):
                    damaged=_v41_damage_one(rf,cell,idx,(damage_min,damage_max),rng)
                    ini=float(_v40_cart_error(damaged,rf,cell)[0][idx].detach().cpu()); b0=_v41_boundary_score(damaged,cell,species,idx,model)
                    for geom in ('bond','angle','four_body','n_body'):
                        if not model['geometries'][geom].get('supported',int(model['geometries'][geom].get('dim',0))>0):
                            continue
                        for sm in smears:
                            cos,e0=_v41_site_force(model,geom,sm,damaged,rf,cell,species,idx)
                            final_state,fin=_v41_restore_site(model,geom,sm,damaged,rf,cell,species,idx,steps,lr)
                            b1=_v41_boundary_score(final_state,cell,species,idx,model)
                            rec=1.0-fin/max(ini,1e-12)
                            brec=(1.0-b1/max(b0,1e-12)) if b0>1e-8 else (1.0 if b1<1e-8 else 0.0)
                            rows.append({'reference':refname,'topology':topo,'site':int(idx),'trial':int(trial),'geometry':geom,'outside_smear_A':float(sm),
                                'force_cosine':cos,'energy0':e0,'initial_error_A':ini,'final_error_A':fin,'recovery_fraction':rec,
                                'initial_boundary_violation_A':b0,'final_boundary_violation_A':b1,'boundary_recovery':brec})
    ranking={}; summary={}
    for topo in learned['topologies']:
        cand=[]
        for geom in ('bond','angle','four_body','n_body'):
            if not learned['topologies'][topo]['geometries'][geom].get('supported',int(learned['topologies'][topo]['geometries'][geom].get('dim',0))>0):
                continue
            for sm in smears:
                q=[r for r in rows if r['topology']==topo and r['geometry']==geom and abs(r['outside_smear_A']-sm)<1e-12]
                if not q: continue
                z={'geometry':geom,'outside_smear_A':float(sm),'n':len(q),
                   'force_cosine_mean':float(np.mean([x['force_cosine'] for x in q])),
                   'force_cosine_positive_fraction':float(np.mean([x['force_cosine']>0 for x in q])),
                   'recovery_fraction':float(np.mean([x['recovery_fraction'] for x in q])),
                   'final_error_A':float(np.mean([x['final_error_A'] for x in q])),
                   'boundary_recovery':float(np.mean([x['boundary_recovery'] for x in q]))}
                z['selection_score']=0.50*z['force_cosine_mean']+0.40*z['recovery_fraction']+0.10*z['boundary_recovery']; cand.append(z)
        cand.sort(key=lambda x:x['selection_score'],reverse=True); ranking[topo]=cand; summary[topo]=cand[0] if cand else None
    return {'schema':'juliette_topology_local_force_lab_v42','damage_A':[float(damage_min),float(damage_max)],
            'principle':'K inside neighbors fully active; K+outside neighbors decay beyond learned K/K+1 boundary',
            'geometries':['bond','angle','four_body','n_body'],'outside_smears_A':[float(x) for x in smears],
            'topologies':list(learned['topologies']),'best':summary,'ranking':ranking,'per_trial':rows}



# -----------------------------------------------------------------------------
# v42 routine deployment selection: separate inside repair / outside rejection
# and test a small set of interpretable merged geometry channels.
# -----------------------------------------------------------------------------
def _v42_candidate_parts(model):
    ok=[g for g in ('bond','angle','four_body','n_body') if model['geometries'][g].get('supported',False)]
    cand=[(g,) for g in ok]
    for parts in (('bond','n_body'),('bond','angle'),('angle','n_body'),('bond','angle','n_body')):
        if all(g in ok for g in parts): cand.append(parts)
    return cand


def _v42_candidate_name(parts):
    return '+'.join(parts)


def _v42_local_geometry_energy(inside,model,parts):
    terms=[]
    for geometry in parts:
        gm=model['geometries'][geometry]
        feat=_v41_feature_torch(inside,geometry)
        anchors=torch.as_tensor(gm['anchors'],dtype=inside.dtype,device=inside.device)
        scale=torch.as_tensor(gm['scale'],dtype=inside.dtype,device=inside.device).clamp_min(1e-5)
        z=(feat[None,:]-anchors)/scale[None,:]
        d2=(z*z).mean(1)
        tau=0.12
        # Each component is dimension-normalized before equal-weight merging.
        terms.append(-tau*torch.logsumexp(-d2/tau,dim=0))
    return torch.stack(terms).mean()


def _v42_site_energy(frac,cell,species,site_i,model,parts,smear_A,channel='full'):
    ce=model['center']; ne=model['neighbor']; k=int(model['K']); nout=int(model['outside_neighbors'])
    ids=torch.as_tensor([j for j,x in enumerate(species) if x==ne],dtype=torch.long,device=frac.device)
    ci=torch.as_tensor([int(site_i)],dtype=torch.long,device=frac.device)
    dv=_v40_pair_vectors(frac,cell,ci,ids)[0]
    dd=torch.linalg.norm(dv,dim=-1)
    if ce==ne: dd=dd.masked_fill(dd<1e-8,1e6)
    flat=torch.argsort(dd.reshape(-1))[:k+nout]
    ni=torch.div(flat,dd.shape[1],rounding_mode='floor'); qi=flat%dd.shape[1]
    vec=dv[ni,qi]; dist=dd[ni,qi]; inside=vec[:k]
    zero=frac.sum()*0.0
    local=_v42_local_geometry_energy(inside,model,parts) if channel in ('inside','full') else zero
    b=float(model['boundary_A']); sm=max(float(smear_A),1e-3)
    inside_escape=torch.nn.functional.softplus((dist[:k]-b)/sm).pow(2).mean() if channel in ('inside','full') else zero
    outside=dist[k:k+nout]
    if channel in ('outside','full') and len(outside):
        intrusion=torch.nn.functional.softplus((b-outside)/sm).pow(2)
        decay=torch.exp(-torch.relu(outside-b)/sm)
        outside_term=(intrusion+0.08*decay).mean()
    else: outside_term=zero
    return local+inside_escape+outside_term


def _v42_force_cos(model,parts,smear,x,ref_frac,cell,species,site_i,target_i,channel):
    q=x.detach().clone().requires_grad_(True)
    e=_v42_site_energy(q,cell,species,site_i,model,parts,smear,channel)
    g=torch.autograd.grad(e,q)[0]; f=(-g)@torch.linalg.inv(cell).T
    _,err=_v40_cart_error(q,ref_frac,cell); target=-err[target_i]
    cos=(f[target_i]*target).sum()/(torch.linalg.norm(f[target_i])*torch.linalg.norm(target)).clamp_min(1e-12)
    return float(cos.detach().cpu()),float(e.detach().cpu())


def _v42_restore_target(model,parts,smear,start,ref_frac,cell,species,site_i,target_i,channel,steps,lr):
    x=start.detach().clone(); inv=torch.linalg.inv(cell); invT=inv.T
    for _ in range(int(steps)):
        q=x.detach().requires_grad_(True); e=_v42_site_energy(q,cell,species,site_i,model,parts,smear,channel)
        g=torch.autograd.grad(e,q)[0]; fc=(-g)@invT
        step=float(lr)*fc[target_i]; sn=torch.linalg.norm(step)
        step=step*torch.clamp(torch.as_tensor(0.12,dtype=step.dtype,device=step.device)/sn.clamp_min(1e-12),max=1.0)
        mask=torch.nn.functional.one_hot(torch.tensor(int(target_i),device=x.device),num_classes=len(x)).to(x.dtype)[:,None]
        x=(x.detach()+mask*(step@inv)[None,:])%1.0
    err,_=_v40_cart_error(x,ref_frac,cell)
    return x,float(err[target_i].detach().cpu())


def _v42_damage_outside(ref_frac,cell,species,site_i,model,rng,penetration=(0.10,0.45)):
    ne=model['neighbor']; k=int(model['K'])
    ids=torch.as_tensor([j for j,x in enumerate(species) if x==ne],dtype=torch.long,device=ref_frac.device)
    ci=torch.as_tensor([int(site_i)],dtype=torch.long,device=ref_frac.device)
    dv=_v40_pair_vectors(ref_frac,cell,ci,ids)[0]; dd=torch.linalg.norm(dv,dim=-1)
    if model['center']==ne: dd=dd.masked_fill(dd<1e-8,1e6)
    flat=torch.argsort(dd.reshape(-1)); pick=flat[k]
    ni=torch.div(pick,dd.shape[1],rounding_mode='floor'); qi=pick%dd.shape[1]
    target_i=int(ids[ni].item()); vec=dv[ni,qi]; dist=float(dd[ni,qi].item())
    b=float(model['boundary_A']); desired=max(0.25,b-float(rng.uniform(*penetration)))
    delta_cart=vec*(desired/max(dist,1e-12)-1.0)
    delta_frac=delta_cart@torch.linalg.inv(cell)
    mask=torch.nn.functional.one_hot(torch.tensor(target_i,device=ref_frac.device),num_classes=len(ref_frac)).to(ref_frac.dtype)[:,None]
    return (ref_frac+mask*delta_frac[None,:])%1.0,target_i


def _v42_geometry_benchmark(refs,learned,parent,child,smears,damage_min,damage_max,trials,steps,lr,seed,device,max_sites):
    rng=np.random.default_rng(int(seed)); dev=torch.device(device); rows=[]
    for refname,st in refs.items():
        cell,rf,species=_v40_reference_state(st,dev)
        for topo,model in learned['topologies'].items():
            centers=[i for i,x in enumerate(species) if x==model['center']][:int(max_sites)]
            candidates=_v42_candidate_parts(model)
            for idx in centers:
                for trial in range(int(trials)):
                    # A. damaged center: tests K-neighbor chemistry/capture.
                    damaged=_v41_damage_one(rf,cell,idx,(damage_min,damage_max),rng)
                    ini=float(_v40_cart_error(damaged,rf,cell)[0][idx].detach().cpu())
                    for parts in candidates:
                        for sm in smears:
                            cos,e0=_v42_force_cos(model,parts,sm,damaged,rf,cell,species,idx,idx,'inside')
                            _,fin=_v42_restore_target(model,parts,sm,damaged,rf,cell,species,idx,idx,'inside',steps,lr)
                            rows.append({'reference':refname,'topology':topo,'test':'inside','site':int(idx),'target':int(idx),'trial':int(trial),
                                'candidate':_v42_candidate_name(parts),'parts':list(parts),'outside_smear_A':float(sm),'force_cosine':cos,'energy0':e0,
                                'initial_error_A':ini,'final_error_A':fin,'recovery_fraction':1.0-fin/max(ini,1e-12)})
                    # B. K+1 invasion: move the first outside atom into the learned shell.
                    invaded,target_i=_v42_damage_outside(rf,cell,species,idx,model,rng)
                    ini_o=float(_v40_cart_error(invaded,rf,cell)[0][target_i].detach().cpu())
                    b0=_v41_boundary_score(invaded,cell,species,idx,model)
                    for parts in candidates:
                        for sm in smears:
                            cos,e0=_v42_force_cos(model,parts,sm,invaded,rf,cell,species,idx,target_i,'outside')
                            fs,fin=_v42_restore_target(model,parts,sm,invaded,rf,cell,species,idx,target_i,'outside',steps,lr)
                            b1=_v41_boundary_score(fs,cell,species,idx,model)
                            brec=1.0-b1/max(b0,1e-12) if b0>1e-8 else (1.0 if b1<1e-8 else 0.0)
                            rows.append({'reference':refname,'topology':topo,'test':'outside','site':int(idx),'target':int(target_i),'trial':int(trial),
                                'candidate':_v42_candidate_name(parts),'parts':list(parts),'outside_smear_A':float(sm),'force_cosine':cos,'energy0':e0,
                                'initial_error_A':ini_o,'final_error_A':fin,'recovery_fraction':1.0-fin/max(ini_o,1e-12),'boundary_recovery':brec,
                                'initial_boundary_violation_A':b0,'final_boundary_violation_A':b1})
    deployment={}; ranking={}
    for topo,model in learned['topologies'].items():
        candidates=[]
        names=sorted(set(r['candidate'] for r in rows if r['topology']==topo))
        for name in names:
            for sm in smears:
                qi=[r for r in rows if r['topology']==topo and r['test']=='inside' and r['candidate']==name and abs(r['outside_smear_A']-sm)<1e-12]
                qo=[r for r in rows if r['topology']==topo and r['test']=='outside' and r['candidate']==name and abs(r['outside_smear_A']-sm)<1e-12]
                if not qi or not qo: continue
                z={'candidate':name,'parts':qi[0]['parts'],'outside_smear_A':float(sm),'n_inside':len(qi),'n_outside':len(qo),
                   'inside_force_cosine':float(np.mean([x['force_cosine'] for x in qi])),
                   'inside_positive_fraction':float(np.mean([x['force_cosine']>0 for x in qi])),
                   'inside_recovery':float(np.mean([x['recovery_fraction'] for x in qi])),
                   'outside_force_cosine':float(np.mean([x['force_cosine'] for x in qo])),
                   'outside_positive_fraction':float(np.mean([x['force_cosine']>0 for x in qo])),
                   'outside_recovery':float(np.mean([x['recovery_fraction'] for x in qo])),
                   'boundary_recovery':float(np.mean([x['boundary_recovery'] for x in qo]))}
                z['inside_score']=0.55*z['inside_force_cosine']+0.45*z['inside_recovery']
                z['outside_score']=0.35*z['outside_force_cosine']+0.25*z['outside_recovery']+0.40*z['boundary_recovery']
                z['balanced_score']=0.50*z['inside_score']+0.50*z['outside_score']
                candidates.append(z)
        ri=sorted(candidates,key=lambda z:z['inside_score'],reverse=True)
        ro=sorted(candidates,key=lambda z:z['outside_score'],reverse=True)
        rb=sorted(candidates,key=lambda z:z['balanced_score'],reverse=True)
        ranking[topo]={'inside':ri,'outside':ro,'balanced':rb}
        deployment[topo]={'inside':ri[0] if ri else None,'outside':ro[0] if ro else None,'balanced_diagnostic':rb[0] if rb else None,
                          'policy':'deploy inside and outside as separate force channels; balanced winner is diagnostic only'}
    return {'schema':'juliette_topology_local_force_selection_v42','damage_A':[float(damage_min),float(damage_max)],
            'tests':{'inside':'damage center; restore K-neighbor local chemistry','outside':'push K+1 neighbor inside learned boundary; reject invasion'},
            'candidate_policy':'single learned invariants plus a small set of equal-weight, dimension-normalized merges; no fitted mixture weights',
            'outside_smears_A':[float(x) for x in smears],'deployment':deployment,'ranking':ranking,'per_trial':rows}


def _v42_export_force_contract(learned,benchmark):
    out={'schema':'juliette_local_force_contract_v42','principle':'separate learned inside-restoration and outside-rejection channels','topologies':{}}
    for topo,m in learned['topologies'].items():
        dep=benchmark['deployment'][topo]
        out['topologies'][topo]={'center':m['center'],'neighbor':m['neighbor'],'K':m['K'],'outside_neighbors':m['outside_neighbors'],
            'boundary_A':m['boundary_A'],'K_distance_median_A':m['K_distance_median_A'],'Kp1_distance_median_A':m['Kp1_distance_median_A'],
            'inside':dep['inside'],'outside':dep['outside'],'geometry_models':{}}
        needed=set((dep['inside'] or {}).get('parts',[]))|set((dep['outside'] or {}).get('parts',[]))
        for g in needed: out['topologies'][topo]['geometry_models'][g]=m['geometries'][g]
    return out


# -----------------------------------------------------------------------------
# v43 substantial comparison: network geometry vs SOAP/Brownian score field
# -----------------------------------------------------------------------------
def _v43_soap_descriptor(frac,cell,species,center_element,rmax=5.0,nrad=6,lmax=4,kcap=18):
    """SOAP-like invariant descriptor for every center atom, using BOTH Ti and O surroundings.
    The scalar learned field is differentiated w.r.t. coordinates, so resulting forces are equivariant.
    """
    dev=frac.device; dtype=frac.dtype
    centers=torch.linspace(0.65,float(rmax),int(nrad),dtype=dtype,device=dev)
    width=max(float(rmax)/max(int(nrad),1)*0.55,0.18)
    blocks=[]; center_ids=[i for i,x in enumerate(species) if x==center_element]
    if not center_ids: return None,[]
    for ne in sorted(set(species)):
        nd,nv=_v40_knn_vectors(frac,cell,species,center_element,ne,int(kcap))
        if nv is None:
            blocks.append(torch.zeros((len(center_ids),int(nrad)*(int(nrad)+1)//2*(int(lmax)+1)),dtype=dtype,device=dev)); continue
        # Smooth radial cutoff; far periodic images fade rather than disappear abruptly.
        rr=torch.linalg.norm(nv,dim=-1)
        cut=(0.5*(torch.cos(torch.pi*torch.clamp(rr/max(float(rmax),1e-8),0,1))+1.0))*(rr<float(rmax)).to(dtype)
        nv_eff=nv*cut[...,None]
        blocks.append(_v40_power(nv_eff,centers,width,lmax))
    return torch.cat(blocks,dim=1),center_ids


def _v43_brownian_training(refs,device,seed,steps=8,trajectories=6,sigma_step_A=0.08,rmax=5.0,nrad=6,lmax=4,kcap=18,max_anchors=700):
    """Construct a smooth scalar denoising field from Brownian trajectories around the five references.
    Target energy is per-atom squared displacement from the clean basin. Kernel regression in SOAP space
    yields a differentiable local energy; inference never receives reference coordinates.
    """
    rng=np.random.default_rng(int(seed)); dev=torch.device(device); raw={'Ti':[],'O':[]}
    for refname,st in refs.items():
        cell,rf,species=_v40_reference_state(st,dev); inv=np.linalg.inv(cell.detach().cpu().numpy())
        for tr in range(int(trajectories)):
            x=rf.detach().clone(); cumulative=np.zeros((len(species),3),float)
            for t in range(int(steps)+1):
                for ce in ('Ti','O'):
                    desc,ids=_v43_soap_descriptor(x,cell,species,ce,rmax,nrad,lmax,kcap)
                    if desc is None: continue
                    for q,i in enumerate(ids):
                        y=float(np.dot(cumulative[i],cumulative[i]))
                        raw[ce].append((desc[q].detach().cpu().numpy(),y,refname,tr,t))
                if t==int(steps): break
                dc=rng.normal(size=(len(species),3))*float(sigma_step_A)
                cumulative+=dc
                x=(x+torch.as_tensor(dc@inv,dtype=x.dtype,device=x.device))%1.0
    out={'schema':'juliette_soap_brownian_field_v43','principle':'local SOAP-like environment -> learned scalar denoising energy -> coordinate gradient force',
         'training':'five authoritative references only; Brownian trajectories','descriptor':{'rmax_A':float(rmax),'nrad':int(nrad),'lmax':int(lmax),'kcap_per_species':int(kcap)},'species_models':{}}
    for ce,rows in raw.items():
        if not rows: continue
        if len(rows)>int(max_anchors):
            ix=np.linspace(0,len(rows)-1,int(max_anchors)).astype(int); rows=[rows[i] for i in ix]
        X=np.vstack([r[0] for r in rows]); y=np.asarray([r[1] for r in rows],float)
        # Robust feature scaling then RBF kernel ridge. Median distance sets field smoothness.
        med=np.median(X,axis=0); sc=np.maximum(np.std(X,axis=0),1e-4); Z=(X-med)/sc
        ns=min(len(Z),300); probe=Z[np.linspace(0,len(Z)-1,ns).astype(int)]
        d2=((probe[:,None,:]-probe[None,:,:])**2).mean(2); pos=d2[d2>1e-12]
        bw=float(np.sqrt(np.median(pos))) if len(pos) else 1.0; bw=max(bw,0.15)
        K=np.exp(-0.5*((Z[:,None,:]-Z[None,:,:])**2).mean(2)/(bw*bw))
        lam=2e-3; alpha=np.linalg.solve(K+lam*np.eye(len(K)),y)
        out['species_models'][ce]={'feature_median':med.tolist(),'feature_scale':sc.tolist(),'anchors':X.tolist(),'alpha':alpha.tolist(),'bandwidth':bw,
            'n_anchors':len(rows),'target_energy':'squared Cartesian displacement from clean reference along Brownian trajectory'}
    return out


def _v43_soap_energy(frac,cell,species,soap_model):
    terms=[]; d=soap_model['descriptor']
    for ce,m in soap_model['species_models'].items():
        feat,ids=_v43_soap_descriptor(frac,cell,species,ce,d['rmax_A'],d['nrad'],d['lmax'],d['kcap_per_species'])
        if feat is None: continue
        med=torch.as_tensor(m['feature_median'],dtype=frac.dtype,device=frac.device); sc=torch.as_tensor(m['feature_scale'],dtype=frac.dtype,device=frac.device)
        anc=torch.as_tensor(m['anchors'],dtype=frac.dtype,device=frac.device); a=torch.as_tensor(m['alpha'],dtype=frac.dtype,device=frac.device)
        za=(anc-med[None,:])/sc[None,:]; z=(feat-med[None,:])/sc[None,:]
        d2=((z[:,None,:]-za[None,:,:])**2).mean(2); ker=torch.exp(-0.5*d2/(float(m['bandwidth'])**2))
        terms.append((ker@a).mean())
    return torch.stack(terms).mean() if terms else frac.sum()*0.0


def _v43_geometry_network_energy(frac,cell,species,learned,force_contract):
    """v42 geometry upgraded to a network energy: every valid center contributes and gradients
    are accumulated on every atom participating in overlapping local environments."""
    terms=[]
    for topo,m in learned['topologies'].items():
        fc=force_contract['topologies'][topo]
        for channel in ('inside','outside'):
            sel=fc[channel]
            if not sel: continue
            parts=tuple(sel['parts']); sm=float(sel['outside_smear_A'])
            centers=[i for i,x in enumerate(species) if x==m['center']]
            for i in centers:
                terms.append(_v42_site_energy(frac,cell,species,i,m,parts,sm,channel))
    return torch.stack(terms).mean() if terms else frac.sum()*0.0


def _v43_force(branch,x,cell,species,learned,force_contract,soap_model):
    q=x.detach().clone().requires_grad_(True)
    if branch=='geometry': e=_v43_geometry_network_energy(q,cell,species,learned,force_contract)
    elif branch=='soap': e=_v43_soap_energy(q,cell,species,soap_model)
    else: raise KeyError(branch)
    g=torch.autograd.grad(e,q)[0]; return (-g)@torch.linalg.inv(cell).T,float(e.detach().cpu())


def _v43_combined_force(branch,x,cell,species,learned,force_contract,soap_model):
    if branch!='geometry+soap': return _v43_force(branch,x,cell,species,learned,force_contract,soap_model)
    fg,eg=_v43_force('geometry',x,cell,species,learned,force_contract,soap_model); fs,es=_v43_force('soap',x,cell,species,learned,force_contract,soap_model)
    # Equal information weighting after RMS force normalization; no fitted mixture coefficient.
    ng=torch.sqrt((fg*fg).sum(1).mean()).clamp_min(1e-10); ns=torch.sqrt((fs*fs).sum(1).mean()).clamp_min(1e-10)
    return 0.5*(fg/ng+fs/ns),0.5*(eg+es)


def _v43_restore(branch,start,cell,species,learned,force_contract,soap_model,steps,lr):
    x=start.detach().clone(); inv=torch.linalg.inv(cell)
    for _ in range(int(steps)):
        f,_=_v43_combined_force(branch,x,cell,species,learned,force_contract,soap_model)
        step=float(lr)*f; sn=torch.linalg.norm(step,dim=1,keepdim=True); step=step*torch.clamp(0.08/sn.clamp_min(1e-12),max=1.0)
        x=(x+(step@inv))%1.0
    return x


def _v43_head_to_head(refs,learned,force_contract,soap_model,device,seed,trials=3,steps=50,lr=0.025,damage=(1.0,1.5),max_sites=6):
    rng=np.random.default_rng(int(seed)); dev=torch.device(device); rows=[]; branches=('geometry','soap','geometry+soap')
    for refname,st in refs.items():
        cell,rf,species=_v40_reference_state(st,dev)
        ids=list(range(min(len(species),int(max_sites))))
        # deterministic spread over both species when possible
        ti=[i for i,x in enumerate(species) if x=='Ti']; oo=[i for i,x in enumerate(species) if x=='O']; ids=(ti[:max(1,int(max_sites)//2)]+oo[:max(1,int(max_sites)-int(max_sites)//2)])[:int(max_sites)]
        for idx in ids:
            for tr in range(int(trials)):
                damaged=_v41_damage_one(rf,cell,idx,damage,rng); err0=float(_v40_cart_error(damaged,rf,cell)[0][idx].cpu())
                target=-_v40_cart_error(damaged,rf,cell)[1][idx]
                for branch in branches:
                    f,_=_v43_combined_force(branch,damaged,cell,species,learned,force_contract,soap_model)
                    cos=float(((f[idx]*target).sum()/(torch.linalg.norm(f[idx])*torch.linalg.norm(target)).clamp_min(1e-12)).detach().cpu())
                    fin=_v43_restore(branch,damaged,cell,species,learned,force_contract,soap_model,steps,lr)
                    errs,_=_v40_cart_error(fin,rf,cell); ef=float(errs[idx].cpu()); collateral=float(torch.sqrt(torch.mean(torch.cat([errs[:idx],errs[idx+1:]])**2)).cpu()) if len(errs)>1 else 0.0
                    rows.append({'reference':refname,'species':species[idx],'site':idx,'trial':tr,'branch':branch,'force_cosine':cos,'initial_error_A':err0,'final_error_A':ef,
                                 'recovery_fraction':1-ef/max(err0,1e-12),'collateral_rms_A':collateral})
    summary=[]
    for b in branches:
        q=[r for r in rows if r['branch']==b]
        z={'branch':b,'force_cosine':float(np.mean([x['force_cosine'] for x in q])),'recovery_fraction':float(np.mean([x['recovery_fraction'] for x in q])),
           'collateral_rms_A':float(np.mean([x['collateral_rms_A'] for x in q])),'positive_fraction':float(np.mean([x['force_cosine']>0 for x in q]))}
        z['score']=0.50*z['force_cosine']+0.45*z['recovery_fraction']-0.05*min(z['collateral_rms_A'],2.0); summary.append(z)
    summary=sorted(summary,key=lambda z:z['score'],reverse=True)
    return {'schema':'juliette_force_field_head_to_head_v43','training_population':'five authoritative references only','branches':list(branches),'summary':summary,'winner':summary[0]['branch'],'per_trial':rows}



# -----------------------------------------------------------------------------
# v44: coherent length-scale / optimizer-role / efficiency laboratory
# -----------------------------------------------------------------------------
def _v44_damage(ref,cell,species,task,scale_A,rng):
    """Damage Ti, O, or a mixed pair by a known RMS-like Cartesian scale."""
    x=ref.detach().clone(); inv=torch.linalg.inv(cell)
    ti=[i for i,z in enumerate(species) if z=='Ti']; oo=[i for i,z in enumerate(species) if z=='O']
    if task=='Ti': ids=[int(rng.choice(ti))] if ti else []
    elif task=='O': ids=[int(rng.choice(oo))] if oo else []
    else:
        ids=[]
        if ti: ids.append(int(rng.choice(ti)))
        if oo: ids.append(int(rng.choice(oo)))
    for i in ids:
        v=rng.normal(size=3); v/=max(np.linalg.norm(v),1e-12); dr=v*float(scale_A)
        x[i]=(x[i]+torch.as_tensor(dr,dtype=x.dtype,device=x.device)@inv)%1.0
    return x,ids


def _v44_confidence(x,cell,species,soap_model):
    """Self-assessed local SOAP support. No reference coordinates are used at inference.
    Confidence is RBF support against Brownian training anchors, averaged over centers.
    """
    vals=[]; d=soap_model['descriptor']
    for ce,m in soap_model['species_models'].items():
        feat,ids=_v43_soap_descriptor(x,cell,species,ce,d['rmax_A'],d['nrad'],d['lmax'],d['kcap_per_species'])
        if feat is None: continue
        med=torch.as_tensor(m['feature_median'],dtype=x.dtype,device=x.device); sc=torch.as_tensor(m['feature_scale'],dtype=x.dtype,device=x.device)
        anc=torch.as_tensor(m['anchors'],dtype=x.dtype,device=x.device)
        z=(feat-med)/sc; za=(anc-med)/sc
        d2=((z[:,None,:]-za[None,:,:])**2).mean(2)
        support=torch.exp(-0.5*d2/(float(m['bandwidth'])**2)).max(1).values
        vals.extend([float(v) for v in support.detach().cpu()])
    return float(np.mean(vals)) if vals else 0.0


def _v44_run(branch,optimizer,start,ref,cell,species,learned,force_contract,soap_model,steps,lr,temp_A,rng,damaged_ids,threshold=0.25):
    x=start.detach().clone(); inv=torch.linalg.inv(cell); t0=time.perf_counter(); evals=0; hit=None
    initial=float(torch.sqrt(torch.mean(_v40_cart_error(x,ref,cell)[0][damaged_ids]**2)).cpu())
    # staged = exploratory first half, deterministic second half. Langevin itself has a gentle cooling schedule.
    for it in range(int(steps)):
        f,_=_v43_combined_force(branch,x,cell,species,learned,force_contract,soap_model); evals+=1
        nf=torch.linalg.norm(f,dim=1,keepdim=True).clamp_min(1e-12)
        step=float(lr)*f
        # Cap deterministic moves to keep every optimizer on the same local trial scale.
        step=step*torch.clamp(0.08/torch.linalg.norm(step,dim=1,keepdim=True).clamp_min(1e-12),max=1.0)
        stochastic=(optimizer=='langevin') or (optimizer=='staged' and it<int(steps)//2)
        if stochastic and float(temp_A)>0:
            cool=1.0-(0.75*it/max(int(steps)-1,1)) if optimizer=='langevin' else 1.0
            noise=rng.normal(size=(len(species),3))*float(temp_A)*cool
            # exploration is local: do not inject large unrelated motion into the whole crystal
            mask=np.zeros((len(species),1),float)
            for j in damaged_ids: mask[j]=1.0
            noise*=mask
            step=step+torch.as_tensor(noise,dtype=x.dtype,device=x.device)
        x=(x+(step@inv))%1.0
        errs,_=_v40_cart_error(x,ref,cell)
        cur=float(torch.sqrt(torch.mean(errs[damaged_ids]**2)).cpu())
        if hit is None and cur<=float(threshold): hit=it+1
    elapsed=time.perf_counter()-t0; errs,_=_v40_cart_error(x,ref,cell)
    final=float(torch.sqrt(torch.mean(errs[damaged_ids]**2)).cpu())
    keep=[i for i in range(len(species)) if i not in damaged_ids]
    collateral=float(torch.sqrt(torch.mean(errs[keep]**2)).cpu()) if keep else 0.0
    return {'initial_error_A':initial,'final_error_A':final,'recovery_fraction':1-final/max(initial,1e-12),
            'collateral_rms_A':collateral,'force_evaluations':evals,'wall_seconds':elapsed,
            'ms_per_force_eval':1000.0*elapsed/max(evals,1),'steps_to_0.25A':hit}


def _v44_role_benchmark(refs,learned,force_contract,soap_model,device,seed,scales,branches,trials=1,steps=15,lr=0.025,langevin_A=0.025,tasks=('Ti','O')):
    rng=np.random.default_rng(int(seed)); dev=torch.device(device); rows=[]
    tasks=tuple(tasks)
    for refname,st in refs.items():
        cell,rf,species=_v40_reference_state(st,dev)
        for task in tasks:
            if task=='Ti' and 'Ti' not in species: continue
            if task=='O' and 'O' not in species: continue
            for scale in scales:
                for tr in range(int(trials)):
                    damaged,ids=_v44_damage(rf,cell,species,task,scale,rng)
                    conf=_v44_confidence(damaged,cell,species,soap_model)
                    # Same damaged state is used by every model/optimizer combination.
                    # Near-field 0.10 A only asks whether deterministic refinement works.
                    # Langevin is tested only at the crossover/search scales.
                    opts=('relax',) if float(scale) <= 0.1000001 else ('relax','langevin')
                    for branch in branches:
                        for opt in opts:
                            # deterministic, reproducible noise stream per row while preserving common damage
                            rr=np.random.default_rng(int(seed)+100000*tr+int(round(scale*1000))+17*len(rows))
                            z=_v44_run(branch,opt,damaged,rf,cell,species,learned,force_contract,soap_model,steps,lr,langevin_A,rr,ids)
                            z.update({'reference':refname,'task':task,'scale_A':float(scale),'trial':tr,'branch':branch,'optimizer':opt,
                                      'soap_self_confidence':conf})
                            rows.append(z)
    # Aggregate by task/scale/model/optimizer; keep efficiency explicit instead of hiding it in the power score.
    groups={}
    for r in rows:
        k=(r['task'],r['scale_A'],r['branch'],r['optimizer']); groups.setdefault(k,[]).append(r)
    summary=[]
    for (task,scale,b,opt),q in groups.items():
        hit=[x['steps_to_0.25A'] for x in q if x['steps_to_0.25A'] is not None]
        summary.append({'task':task,'scale_A':scale,'branch':b,'optimizer':opt,'n':len(q),
            'recovery_fraction':float(np.mean([x['recovery_fraction'] for x in q])),
            'collateral_rms_A':float(np.mean([x['collateral_rms_A'] for x in q])),
            'soap_self_confidence':float(np.mean([x['soap_self_confidence'] for x in q])),
            'wall_seconds':float(np.mean([x['wall_seconds'] for x in q])),
            'ms_per_force_eval':float(np.mean([x['ms_per_force_eval'] for x in q])),
            'success_0.25A_fraction':len(hit)/max(len(q),1),
            'mean_steps_to_0.25A':None if not hit else float(np.mean(hit))})
    return {'schema':'juliette_optimizer_role_lengthscale_v44',
            'principle':'role-only test: Ti vs O; deterministic refinement vs Langevin; geometry vs SOAP vs normalized merge',
            'scales_A':[float(x) for x in scales],'tasks':list(tasks),'branches':list(branches),'summary':summary,'per_trial':rows}



def parse_args():
    p=argparse.ArgumentParser(description='Juliette v44 length-scale, optimizer-role and efficiency laboratory; five-reference force learning only')
    p.add_argument('--ase-database',action='append',default=[])
    p.add_argument('--reference-cif-dir',required=True,help='Directory containing all authoritative reference *.cif files')
    p.add_argument('--r-search',type=float,default=5.0)
    p.add_argument('--building-center',default='Ti')
    p.add_argument('--associated-element',default='O')
    p.add_argument('--v31-reference-sigma-A',type=float,default=0.28)
    p.add_argument('--v31-reference-max-bayes',type=float,default=0.10)
    p.add_argument('--v31-bayes-exponent',type=float,default=2.0)
    p.add_argument('--v31-bayes-lambda',type=float,default=1e-3)
    p.add_argument('--v31-allowed-perturb-A',type=float,default=0.10)
    p.add_argument('--v31-perturb-samples',type=int,default=12)
    p.add_argument('--v31-random-seed',type=int,default=3151)
    p.add_argument('--v32-titi-rmax-A',type=float,default=5.5)
    p.add_argument('--v33-titi-nmax',type=int,default=12)
    p.add_argument('--v33-titi-lmax',type=int,default=10)
    p.add_argument('--v32-titi-sigma-A',type=float,default=0.28)
    p.add_argument('--v32-titi-perturb-samples',type=int,default=4)
    p.add_argument('--v32-titi-min-mode-sites',type=int,default=2)
    p.add_argument('--diagnostics',action=argparse.BooleanOptionalAction,default=False,help='Legacy reference diagnostics only; off by default in v42 routine force selection.')
    p.add_argument('--v40-restoration-trials',type=int,default=4)
    p.add_argument('--v41-outside-smears-A',default='0.08,0.16,0.32,0.64',help='Comma-separated outside-shell decay widths')
    p.add_argument('--v41-outside-neighbors',type=int,default=3,help='Number of neighbors beyond K included in the decaying outside field')
    p.add_argument('--v41-max-sites-per-reference',type=int,default=12,help='Limit damaged centers per reference/topology for the geometry sweep')
    p.add_argument('--v40-restoration-steps',type=int,default=75)
    p.add_argument('--v40-restoration-lr',type=float,default=0.035)
    p.add_argument('--v40-damage-min-A',type=float,default=1.0)
    p.add_argument('--v40-damage-max-A',type=float,default=1.5)
    p.add_argument('--v40-device',default='cpu')
    p.add_argument('--no-restoration-benchmark',action='store_true',help='Skip v43 restoration model selection (debug only).')
    p.add_argument('--v43-brownian-trajectories',type=int,default=5)
    p.add_argument('--v43-brownian-steps',type=int,default=7)
    p.add_argument('--v43-brownian-step-A',type=float,default=0.08)
    p.add_argument('--v43-soap-rmax-A',type=float,default=5.0)
    p.add_argument('--v43-soap-nrad',type=int,default=6)
    p.add_argument('--v43-soap-lmax',type=int,default=4)
    p.add_argument('--v43-soap-kcap',type=int,default=18)
    p.add_argument('--v43-soap-max-anchors',type=int,default=600)
    p.add_argument('--v43-compare-trials',type=int,default=1)
    p.add_argument('--v43-compare-steps',type=int,default=30)
    p.add_argument('--v43-compare-lr',type=float,default=0.025)
    p.add_argument('--v44-scales-A',default='0.10,0.50,1.00',help='Coherent imposed-damage scales for the useful first-pass role map')
    p.add_argument('--v44-role-trials',type=int,default=1)
    p.add_argument('--v44-role-steps',type=int,default=15)
    p.add_argument('--v44-langevin-A',type=float,default=0.025,help='Per-step stochastic Cartesian amplitude during Langevin exploration')
    p.add_argument('--output',default='data/xn_templates/chemistry_model.json')
    return p.parse_args()


def main():
    args=parse_args(); _require_runtime_packages()
    # v39 split learning: references define channels/targets; the full ASE database supplies statistical learning.
    byid={}
    if len(args.ase_database)!=1:
        raise ValueError('v39 requires exactly one --ase-database for the full-database statistical layer')
    if args.ase_database:
        from ase.db import connect
        from pymatgen.io.ase import AseAtomsAdaptor
        db=connect(args.ase_database[0]); ad=AseAtomsAdaptor()
        for row in db.select():
            try: byid[int(row.id)]=ad.get_structure(row.toatoms())
            except Exception as exc: print(f'WARNING: skipped DB row {row.id}: {exc}')
    t0=time.perf_counter()
    report=_v32_reference_channels(byid,args.reference_cif_dir,args.building_center,
                                     args.associated_element,args.r_search,
                                     args.v31_reference_sigma_A,args.v31_reference_max_bayes,
                                     args.v31_bayes_exponent,args.v31_bayes_lambda,args.v31_allowed_perturb_A,
                                     (0.02,0.05,0.10,0.18),args.v31_perturb_samples,args.v31_random_seed)
    tio_seconds=time.perf_counter()-t0
    refs,_=_load_reference_cifs(args.reference_cif_dir)
    # v41 deliberately skips the expensive Ti-Ti SOAP/NN production manifold.  This run is a
    # local-chemistry force-discovery experiment: topology + learned geometric invariants only.
    contract={'chemistry':{'parent_element':args.building_center,'child_element':args.associated_element,
                          'parent_coordination':6,'child_coordination':3,'target_example':f'{args.building_center}{args.associated_element}2'}}
    contract_seconds=0.0
    benchmark=None; soap_model=None; comparison=None; learned=None; force_contract=None
    if not args.no_restoration_benchmark:
        print('\n=== v43 SUBSTANTIAL RESTORATION MODEL TEST ===')
        smears=[float(x) for x in str(args.v41_outside_smears_A).split(',') if x.strip()]
        learned=_v41_learn_local_geometry(refs,args.building_center,args.associated_element,args.v41_outside_neighbors)
        # Geometry candidate selection remains reference-only. Defaults are deliberately smaller than v42;
        # v43 spends compute on the final common iterative benchmark rather than exhaustive trajectories.
        benchmark=_v42_geometry_benchmark(refs,learned,args.building_center,args.associated_element,
            smears,args.v40_damage_min_A,args.v40_damage_max_A,max(1,min(args.v40_restoration_trials,2)),
            max(20,min(args.v40_restoration_steps,40)),args.v40_restoration_lr,args.v31_random_seed+889,args.v40_device,
            max(4,min(args.v41_max_sites_per_reference,6)))
        force_contract=_v42_export_force_contract(learned,benchmark)
        print('  Geometry-network winners selected; constructing SOAP/Brownian field from the same five references...')
        soap_model=_v43_brownian_training(refs,args.v40_device,args.v31_random_seed+1441,args.v43_brownian_steps,args.v43_brownian_trajectories,
            args.v43_brownian_step_A,args.v43_soap_rmax_A,args.v43_soap_nrad,args.v43_soap_lmax,args.v43_soap_kcap,args.v43_soap_max_anchors)
        # v44 does NOT repeat the expensive v43 head-to-head.  Its only question is optimizer role.
        comparison=None
        scales=[float(x) for x in str(args.v44_scales_A).split(',') if x.strip()]
        print('  ROLE-ONLY v44: Ti / O; 0.10A relax only; 0.50/1.00A relax vs Langevin...')
        role_benchmark=_v44_role_benchmark(refs,learned,force_contract,soap_model,args.v40_device,args.v31_random_seed+3441,scales,
            ('geometry','soap','geometry+soap'),args.v44_role_trials,args.v44_role_steps,args.v43_compare_lr,args.v44_langevin_A,('Ti','O'))
        for task in ('Ti','O'):
            print(f'  v44 {task}:')
            q=sorted([z for z in role_benchmark['summary'] if z['task']==task],key=lambda z:(z['scale_A'],z['branch'],z['optimizer']))
            for z in q:
                hit='-' if z['mean_steps_to_0.25A'] is None else f"{z['mean_steps_to_0.25A']:.1f}"
                print(f"    {z['scale_A']:.2f}A {z['branch']:<13s} {z['optimizer']:<8s} recovery={z['recovery_fraction']:+.3f} final25={z['success_0.25A_fraction']:.2f} steps25={hit} cost={z['wall_seconds']:.3f}s/eval={z['ms_per_force_eval']:.2f}ms conf={z['soap_self_confidence']:.3f}")
    else:
        role_benchmark=None
    model={'version':'44','schema':'juliette_restoration_force_model_v44',
           'reference_channel_learning':report,
           'local_geometry_learning':learned if benchmark is not None else None,
           'geometry_model_selection':benchmark,
           'local_force_contract':force_contract,
           'soap_brownian_field':soap_model,
           'restoration_head_to_head':comparison,
           'selected_restoration_branch':None,
           'optimizer_role_lengthscale_benchmark':role_benchmark,
           'generation_contract':contract}
    os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    with open(args.output,'w') as f: json.dump(model,f,indent=2,default=_json_default)
    print('Juliette learner v44 optimizer-role / length-scale model written')
    print('Output:',args.output); print('Input database rows:',len(byid))
    print('Reference CIF directory:',report['reference_directory'])
    b=report['bayesian_metric']
    print(f"Channel merge raw limit={b['channel_merge_raw_q90U_limit']:.6f} | allowed fluctuation={b['allowed_perturbation_A']:.3f} A | sigma={b['soap_sigma_A']:.3f} A | exponent={b['kernel_exponent']:.2f} lambda={b['lambda']:.1e}")
    print('NO interpolation / recursive admission / target count.')
    print('v44: role-only test; Ti/O separately, geometry/SOAP/merge, deterministic refinement vs Langevin, with measured cost.')
    print('\n=== v44 REFERENCES / DATABASE MEMBERSHIP ===')
    for n,r in report['references'].items():
        print(f"  {n:<14s} | {r['filename']:<18s} | SG={str(r['space_group_number']):>3s} | shell<={r['first_shell_cutoff_A']:.3f} A | CN={r['reference_CN']} | {r['database_status']} rows={r['matched_database_rows']}")
    print('\n=== v44 REFERENCE-REFERENCE BAYESIAN DISTANCES (directed q90U) ===')
    names=list(report['references'])
    print(' '*16+' '.join(f'{n[:8]:>9s}' for n in names))
    for a in names:
        print(f'{a:<16s}'+' '.join(f"{report['pairwise_reference_bayesian_distance'][a][b]['q90U']:9.4f}" for b in names))
    print('Reference channels (strict mutual / complete linkage):')
    for c in report['reference_channels']: print(f"  {c['id']}: {c['members']}")
    print('Reference fluctuation calibration:')
    for n,c in report['reference_fluctuation_calibration'].items():
        print(f"  {n:<14s} allowed={c['allowed_perturbation_A']:.3f} A -> q90U<={c['acceptance_q90U']:.6f}")
        print('    '+' | '.join(f"{x['amplitude_A']:.2f}A:{x['q90_q90U']:.5f}" for x in c['curve']))
    print('\n=== v43 ALL DATABASE ROWS ===')
    print('row pass channel reference q90U D | SG V/atom TiTi_min TiTi_CN3.30 | per-reference q90U')
    for rid in sorted(report['all_rows']):
        x=report['all_rows'][rid]; t=x['topology']
        pr=' '.join(f"{n[:5]}={x['by_reference'][n]['q90U']:.4f}" for n in names)
        tm='nan' if t['center_center_min_A'] is None else f"{t['center_center_min_A']:.3f}"
        print(f"{rid:3d} {'PASS' if x['pass'] else '----'} {x['best_channel']:>3s} {x['best_reference']:<14s} {x['best_q90U']:.6f} {x['best_normalized_D']:.3f} | SG={str(t['space_group']):>3s} V/a={t['volume_per_atom_A3']:.2f} TiTi={tm} CN3.3={str(t['center_center_CN_3p30_median']):>4s} | {pr}")
    print(f"\nSelected={report['selected_count']}/{len(byid)} rows={report['selected_rows']}")
    print('External references added to training pool:',report['training_pool']['external_reference_structures_added'])
    print('Dataset making completed.')



# === SELF-CONTAINED WYCKOFF OXYGEN BENCHMARK ===

"""v46: exact-Wyckoff oxygen reconstruction, v140 port-inspired vs learned geometry.

Reference oxygen is used ONLY to identify orbit types/stabilizers and for scoring.
No reference oxygen free coordinates enter initialization. Ti positions/cell frozen.

The v140 branch is a *port-inspired differentiable surrogate*, NOT the full v140
Stage-B Ti rotation/Sinkhorn solver: a full port solver would require rebuilding
its Ti-centered learned basis and joint assignment under the selected SG orbits.
This distinction is reported explicitly to avoid mislabeling a control.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def symmetry_orbits(st, tol=2e-4):
    """Orbit/stabilizer decomposition with original (nonstandard) cell operations.

    Fixed affine subspaces come from the stabilizer of each reference orbit,
    but the reference's free-coordinate values are discarded.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga=SpacegroupAnalyzer(st,symprec=1e-3,angle_tolerance=5)
    ops=sga.get_symmetry_operations(cartesian=False)
    rot=np.array([np.rint(x.rotation_matrix).astype(int) for x in ops])
    trans=np.array([x.translation_vector for x in ops])
    o=np.array([site.frac_coords for site in st if site.specie.symbol=='O'])%1
    unused=set(range(len(o))); out=[]
    while unused:
        i=min(unused); p=o[i]
        images=np.einsum('aij,j->ai',rot,p)+trans
        images%=1
        orbit=[]; opids=[]
        for im in images:
            df=o-im; df-=np.rint(df)
            jj=int(np.argmin(np.linalg.norm(df@st.lattice.matrix,axis=1)))
            if np.linalg.norm(df[jj]@st.lattice.matrix)>tol: raise RuntimeError('symmetry mapping failed')
            if jj not in orbit:
                orbit.append(jj); opids.append(len(opids)) # replaced below
        orbit=sorted(orbit)
        if not set(orbit)<=unused: raise RuntimeError('oxygen orbit overlap')
        unused-=set(orbit)
        # Select one operation for each orbit image in stable reference-index order.
        selected=[]
        for j in orbit:
            df=images-o[j]; df-=np.rint(df)
            idx=int(np.argmin(np.linalg.norm(df@st.lattice.matrix,axis=1)))
            selected.append(idx)
        # Stabilizer: R p + t == p (mod 1). Solve (R-I)x = n-t
        stab=[]
        for k in range(len(rot)):
            df=rot[k]@p+trans[k]-p
            if np.linalg.norm(df-np.rint(df))<tol:
                n=np.rint(df).astype(int)
                stab.append((rot[k]-np.eye(3), n-trans[k]))
        mat=np.concatenate([z[0] for z in stab],axis=0)
        rhs=np.concatenate([z[1] for z in stab])
        u,sv,vh=np.linalg.svd(mat,full_matrices=True)
        rank=int((sv>1e-7).sum())
        basis=vh[rank:].T.copy()
        offset=np.linalg.lstsq(mat,rhs,rcond=None)[0] if rank else np.zeros(3)
        if np.linalg.norm(mat@offset-rhs)>1e-5: raise RuntimeError('inconsistent stabilizer')
        out.append(dict(indices=orbit, rotations=rot[selected],translations=trans[selected],
                        basis=basis,offset=offset, multiplicity=len(orbit),dof=basis.shape[1]))
    return out, sga.get_space_group_number()


def expanded(q, orbits, device):
    chunks=[]; k=0
    for ob in orbits:
        d=ob['dof']; basis=torch.as_tensor(ob['basis'],device=device,dtype=q.dtype)
        off=torch.as_tensor(ob['offset'],device=device,dtype=q.dtype)
        gen=off+(basis@q[k:k+d] if d else 0.0); k+=d
        r=torch.as_tensor(ob['rotations'],device=device,dtype=q.dtype)
        t=torch.as_tensor(ob['translations'],device=device,dtype=q.dtype)
        chunks.append((torch.einsum('aij,j->ai',r,gen)+t)%1.0)
    return torch.cat(chunks,0)


def initial_q(orbits,rng,device,dtype):
    # Random coordinates in the stabilizer-allowed subspace; no reference O coordinates.
    return torch.as_tensor(np.concatenate([rng.uniform(-1.5,1.5,ob['dof']) for ob in orbits]),device=device,dtype=dtype)


def image_vectors(a,b,cell):
    # Explicit periodic images, including self images where appropriate.
    shift=torch.tensor([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)],dtype=a.dtype,device=a.device)
    delta=b[None,:,None,:]-a[:,None,None,:]+shift
    vec=delta@cell
    return vec,torch.linalg.norm(vec,dim=-1)


def port_geometry(ti,o,cell,r0,sigma):
    """Soft shared-Ti/O port-like control, no ideal octahedral angular target.

    Each O is encouraged to share three Ti environments; each Ti six O images.
    Gaussian occupation width sigma provides a differentiable capture volume.
    """
    _,d=image_vectors(ti,o,cell)
    sig=max(float(sigma),0.03)
    # Smooth capture of all periodic Ti-O contacts around reference-learned bond scale.
    occ=torch.sigmoid((r0+0.35-d)/sig)
    ct=occ.sum((1,2)); co=occ.sum((0,2))
    ec=((ct-6)**2).mean()+((co-3)**2).mean()
    # Attract occupied Ti-O contacts toward learned median bond radius.
    bond=(occ*(d-r0).square()).sum()/occ.sum().clamp_min(1)
    _,doo=image_vectors(o,o,cell)
    n=len(o); mask=torch.ones_like(doo,dtype=torch.bool)
    eye=torch.arange(n,device=o.device); mask[eye,eye,13]=False
    # Other periodic images of the same O remain physically real.
    rep=torch.nn.functional.softplus((1.70-doo[mask])/0.10).square().sum()/max(n,1)
    return 2*ec+bond/0.15**2+3*rep


def periodic_dist(a,b,cell):
    _,d=image_vectors(a,b,cell); return d.min(-1).values


def score(pred,true,ti,cell,r0):
    D=periodic_dist(pred,true,cell).detach().cpu().numpy()
    i,j=linear_sum_assignment(D); matched=D[i,j]
    # Count all periodic images, not only distinct atom indices.
    _,d=image_vectors(ti,pred,cell)
    mask=(d.detach().cpu().numpy()<r0+0.38)
    ct=mask.sum((1,2)); co=mask.sum((0,2))
    _,doo=image_vectors(pred,pred,cell)
    doo=doo.detach().cpu().numpy(); idx=np.arange(len(pred)); doo[idx,idx,13]=np.inf
    return dict(rms_A=float(np.sqrt(np.mean(matched**2))),within_025=float(np.mean(matched<.25)),
                Ti6=float(np.mean(ct==6)),O3=float(np.mean(co==3)),
                coord_defect=int(np.abs(ct-6).sum()+np.abs(co-3).sum()),OO_min_A=float(doo.min()))


def minimize(q0,orbits,ti,cell,species,branch,sigma,mod,learned,fc,r0,steps,lr):
    q=torch.nn.Parameter(q0.clone()); opt=torch.optim.Adam([q],lr=lr)
    bestE=float('inf'); best=None; trace=[]; start=time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        o=expanded(q,orbits,ti.device)
        if branch=='v140_port_control': E=port_geometry(ti,o,cell,r0,sigma)
        else:
            # Learned inside/outside kernel bandwidths are trained parameters.
            # sigma changes only the shared Ti-O capture/coordination term.
            x=torch.cat([ti,o],0)
            E=_v43_geometry_network_energy(x,cell,species,learned,fc)
            E=E+0.05*port_geometry(ti,o,cell,r0,sigma)
        if not torch.isfinite(E): break
        E.backward()
        opt.step()
        ev=float(E.detach())
        if ev<bestE: bestE=ev; best=q.detach().clone()
        if step in (0,steps-1): trace.append([step+1,ev])
    return (expanded(best,orbits,ti.device).detach() if best is not None else None),dict(
        seconds=time.perf_counter()-start,evals=step+1,best_energy=bestE,trace=trace)


def summarize_trials(rows,widths):
    summary=[]
    for branch in ('v140_port_control','learned_geometry'):
        for sig in widths:
            rr=[r for r in rows if r['branch']==branch and r['width_A']==sig]
            if not rr: continue
            summary.append(dict(branch=branch,width_A=sig,n=len(rr),mean_rms_A=float(np.mean([r['rms_A'] for r in rr])),
                                best_rms_A=float(min(r['rms_A'] for r in rr)),
                                mean_Ti6=float(np.mean([r['Ti6'] for r in rr])),
                                mean_O3=float(np.mean([r['O3'] for r in rr])),
                                wall_s=float(sum(r['seconds'] for r in rr))))
    return summary



def print_ordered_report(rows, widths, starts, verbose=False):
    """One deterministic report, emitted by parent only after workers finish."""
    names=list(dict.fromkeys(r['reference'] for r in rows))
    branches=('v140_port_control','learned_geometry')
    for name in names:
        rr=[r for r in rows if r['reference']==name]
        sg=rr[0]['sg']
        print(f'\n{"="*76}\n{name.upper()} | SG {sg} | {len(rr)} completed trials\n{"="*76}',flush=True)
        print(f'{"Method":<20} {"sigma(A)":>8} {"N":>4} {"RMS mean":>10} {"RMS best":>10} {"w25 mean":>10} {"Ti6":>7} {"O3":>7} {"time(s)":>9}',flush=True)
        for branch in branches:
            for sig in widths:
                sub=[r for r in rr if r['branch']==branch and abs(r['width_A']-sig)<1e-8]
                if not sub: continue
                avg=lambda key: float(np.mean([r[key] for r in sub]))
                print(f'{branch:<20} {sig:8.2f} {len(sub):4d} {avg("rms_A"):10.3f} {min(r["rms_A"] for r in sub):10.3f} {avg("within_025"):10.2f} {avg("Ti6"):7.2f} {avg("O3"):7.2f} {sum(r["seconds"] for r in sub):9.1f}',flush=True)
        if verbose:
            print('  Individual trials (start, sigma, method, RMS, w25, Ti6, O3, seconds):',flush=True)
            for r in sorted(rr,key=lambda r:(r['start'],r['width_A'],branches.index(r['branch']))):
                print(f'    s{r["start"]} {r["width_A"]:.2f} {r["branch"]:<20} {r["rms_A"]:.3f} {r["within_025"]:.2f} {r["Ti6"]:.2f} {r["O3"]:.2f} {r["seconds"]:.1f}',flush=True)
    print(f'\n{"="*76}\nOVERALL SUMMARY\n{"="*76}',flush=True)
    for branch in branches:
        sub=[r for r in rows if r['branch']==branch]
        if sub:
            print(f'{branch:<20} N={len(sub):3d} RMS(mean)={np.mean([r["rms_A"] for r in sub]):.3f} A  RMS(best)={min(r["rms_A"] for r in sub):.3f} A  time(sum)={sum(r["seconds"] for r in sub):.1f}s',flush=True)


def train_geometry(refs,a):
    print('Training local geometry from reference CIFs (no saved model required)...',flush=True)
    learned=_v41_learn_local_geometry(refs,'Ti','O',outside_neighbors=3)
    if a.train_mode=='select':
        selection=_v42_geometry_benchmark(refs,learned,'Ti','O',[0.20,0.50],0.25,0.75,
                    1,a.train_steps,0.025,a.seed+73,'cpu',4)
        fc=_v42_export_force_contract(learned,selection)
    else:
        fc={'schema':'juliette_fast_force_contract','topologies':{}}
        for topo,m in learned['topologies'].items():
            parts=[g for g in ('bond','angle','four_body','n_body') if m['geometries'][g]['supported']]
            fc['topologies'][topo]={
                'inside':{'parts':parts,'outside_smear_A':0.20},
                'outside':{'parts':['bond'],'outside_smear_A':0.20}}
    print('Training complete.',flush=True)
    return learned,fc


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-cif-dir',required=True)
    p.add_argument('--widths',default='0.10,0.20,0.35,0.50,0.75')
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--lr',type=float,default=0.025)
    p.add_argument('--seed',type=int,default=4617)
    p.add_argument('--device',default='auto',help='auto selects CUDA when available')
    p.add_argument('--train-mode',choices=('fast','select'),default='fast',help='fast: learn geometry directly; select: run costly v42 restoration selection')
    p.add_argument('--train-steps',type=int,default=20)
    p.add_argument('--output',default='data/xn_templates/oxygen_wyckoff_results.json')
    p.add_argument('--gpus',type=int,default=0,help='GPU workers (0=auto-detect visible allocated GPUs)')
    p.add_argument('--cpus',type=int,default=0,help='CPU threads in total (0=auto-detect SLURM/affinity)')
    p.add_argument('--verbose-trials',action='store_true',help='Print individual trials in ordered reference blocks')
    p.add_argument('--training-cache',default='',help=argparse.SUPPRESS)
    p.add_argument('--worker-index',type=int,default=-1,help=argparse.SUPPRESS)
    p.add_argument('--workers',type=int,default=1,help=argparse.SUPPRESS)
    a=p.parse_args(); torch.set_default_dtype(torch.float64)
    if a.gpus < 0 or a.cpus < 0 or a.workers < 1:
        p.error('--gpus/--cpus must be >= 0 and --workers >= 1')
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    slurm_cpu=int(os.environ.get('SLURM_CPUS_PER_TASK','0') or 0)
    total_cpus=max(1,min(affinity,slurm_cpu if slurm_cpu else affinity,a.cpus if a.cpus else affinity))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    available=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if a.device == 'auto': a.device='cuda:0' if available else 'cpu'
    if a.gpus == 0: a.gpus=max(1,available)
    if a.gpus > 1 and a.gpus > available:
        p.error(f'Requested {a.gpus} GPUs but PyTorch sees only {available}')
    cpu_per_worker=max(1,total_cpus//(a.gpus if a.worker_index < 0 else a.workers))
    torch.set_num_threads(cpu_per_worker)
    os.environ['OMP_NUM_THREADS']=str(cpu_per_worker)
    os.environ['MKL_NUM_THREADS']=str(cpu_per_worker)
    if a.worker_index < 0:
        print(f'Resources: GPUs visible={available}, GPU workers={a.gpus}, CPU affinity={affinity}, '
              f'SLURM CPUs={slurm_cpu or "unset"}, CPU threads/worker={cpu_per_worker}',flush=True)
    if a.worker_index < 0 and a.gpus > 1:
        if not torch.cuda.is_available():
            p.error('--gpus > 1 requires CUDA')
        with tempfile.TemporaryDirectory(prefix='oxygen_workers_') as tmp:
            # Train once on CPU and share the resulting immutable descriptor model.
            refs,_=_load_reference_cifs(a.reference_cif_dir)
            learned,fc=train_geometry(refs,a)
            cache=Path(tmp)/'trained_geometry.pt'
            torch.save({'learned':learned,'fc':fc},cache)
            print('Training complete once; distributing trial jobs to GPUs.',flush=True)
            processes=[]; outputs=[]; log_handles=[]; log_paths=[]
            for wi in range(a.gpus):
                outfile=str(Path(tmp)/f'worker_{wi}.json'); outputs.append(outfile)
                log_path=Path(tmp)/f'worker_{wi}.log'; log_paths.append(log_path)
                log_handle=log_path.open('w'); log_handles.append(log_handle)
                cmd=[sys.executable,str(Path(__file__).resolve()),
                     '--reference-cif-dir',a.reference_cif_dir,'--widths',a.widths,
                     '--starts',str(a.starts),'--steps',str(a.steps),'--lr',str(a.lr),
                     '--seed',str(a.seed),'--device','cuda:0','--train-mode',a.train_mode,
                     '--train-steps',str(a.train_steps),'--output',outfile,
                     '--worker-index',str(wi),'--workers',str(a.gpus),'--gpus','1',
                     '--training-cache',str(cache),'--cpus',str(cpu_per_worker)]
                env=os.environ.copy()
                if visible:
                    env['CUDA_VISIBLE_DEVICES']=visible.split(',')[wi].strip()
                else:
                    env['CUDA_VISIBLE_DEVICES']=str(wi)
                print(f'Launching GPU worker {wi+1}/{a.gpus}: {env["CUDA_VISIBLE_DEVICES"]}',flush=True)
                processes.append(subprocess.Popen(cmd,env=env,stdout=log_handle,stderr=subprocess.STDOUT))
            errors=[(i,proc.wait()) for i,proc in enumerate(processes)]
            for handle in log_handles: handle.close()
            failures=[(i,code) for i,code in errors if code!=0]
            if failures:
                for i,_ in failures:
                    print(f'\n--- Worker {i} failure log ---\n{log_paths[i].read_text()}',flush=True)
                raise RuntimeError(f'GPU worker failures: {failures}')
            results=[json.loads(Path(out).read_text()) for out in outputs]
            rows=[row for result in results for row in result['trials']]
            summary=summarize_trials(rows,[float(z) for z in a.widths.split(',')])
            result=results[0]
            result['settings']=vars(a); result['trials']=rows; result['summary']=summary
            print_ordered_report(rows,[float(z) for z in a.widths.split(',')],a.starts,a.verbose_trials)
            result['parallel']={'gpu_workers':a.gpus,'partition':'reference/start/width/branch trials',
                                'training':'once in parent, shared via temporary cache'}
            out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(result,indent=2))
            print('Output:',out,flush=True)
        return
    refs,_=_load_reference_cifs(a.reference_cif_dir)
    if a.training_cache:
        cached=torch.load(a.training_cache,map_location='cpu',weights_only=False)
        learned,fc=cached['learned'],cached['fc']
        print(f'Worker {a.worker_index}: loaded shared training.',flush=True)
    else:
        learned,fc=train_geometry(refs,a)
    rng=np.random.default_rng(a.seed); dev=torch.device(a.device)
    widths=[float(z) for z in a.widths.split(',')]; rows=[]
    if a.worker_index < 0: print('=== EXACT WYCKOFF OXYGEN RECONSTRUCTION ===',flush=True)
    for ref_index,(name,st) in enumerate(refs.items()):
        orbits,sg=symmetry_orbits(st)
        cell,rf,sp=_v40_reference_state(st,dev)
        ti=rf[[i for i,x in enumerate(sp) if x=='Ti']].detach()
        true=rf[[i for i,x in enumerate(sp) if x=='O']].detach()
        if sum(x['multiplicity'] for x in orbits)!=len(true): raise RuntimeError('orbit count mismatch')
        species=['Ti']*len(ti)+['O']*len(true)
        # Reference-derived median is a *training* bond scale; not an ideal hardcoded distance.
        # Reported explicitly as reference-informed control; learned branch has same baseline.
        r0=float(learned['topologies']['TiO']['K_distance_median_A'])
        qstarts=[initial_q(orbits,rng,dev,rf.dtype) for _ in range(a.starts)]
        if a.worker_index < 0: print(f'{name}: SG{sg} Ti{len(ti)} O{len(true)} orbits={[x["multiplicity"] for x in orbits]} dof={sum(x["dof"] for x in orbits)}',flush=True)
        for si,q0 in enumerate(qstarts):
            for sig in widths:
                for bi,branch in enumerate(('v140_port_control','learned_geometry')):
                    # Fine-grained jobs balance the expensive learned branch across GPUs.
                    job_index=(((ref_index*a.starts+si)*len(widths)+widths.index(sig))*2+bi)
                    if a.worker_index >= 0 and job_index % a.workers != a.worker_index: continue
                    # Sweep the learned outside/inside boundary smoothness too.
                    trial_fc={'topologies':{k:{**v,'inside':{**v['inside'],'outside_smear_A':sig},
                        'outside':{**v['outside'],'outside_smear_A':sig}} for k,v in fc['topologies'].items()}}
                    pred,cost=minimize(q0,orbits,ti,cell,species,branch,sig,None,learned,trial_fc,r0,a.steps,a.lr)
                    if pred is None: print('nonfinite:',name,si,sig,branch,flush=True); continue
                    sc=score(pred,true,ti,cell,r0)
                    row=dict(reference=name,sg=sg,start=si,width_A=sig,branch=branch,**cost,**sc)
                    rows.append(row)
                    if a.worker_index < 0 and a.verbose_trials:
                        print(f' {branch:19s} s{si} sig={sig:.2f} RMS={sc["rms_A"]:.3f} w25={sc["within_025"]:.2f} Ti6={sc["Ti6"]:.2f} O3={sc["O3"]:.2f} {cost["seconds"]:.1f}s',flush=True)
    summary=summarize_trials(rows,widths)
    if a.worker_index < 0: print_ordered_report(rows,widths,a.starts,False)
    result=dict(scope='exact oxygen orbit types from reference; free oxygen coordinates hidden; frozen Ti',
                caveat='v140_port_control is a port-inspired surrogate, NOT the original v140 Stage-B rotation/Sinkhorn SCF',
                learned_width='sigma affects shared Ti-O capture penalty and learned shell-boundary softness; learned descriptor anchor scales unchanged',
                training={'mode':a.train_mode,'source':'same reference CIFs','model_file_required':False},
                settings=vars(a),summary=summary,trials=rows)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    if a.worker_index < 0: print('Output:',out,flush=True)


# -----------------------------------------------------------------------------
# Standalone generation-oriented, family-conditioned geometry/SOAP benchmark.
# -----------------------------------------------------------------------------
def _family_training(refs, args):
    """Train one independent geometry channel per reference CIF; never cluster.

    Keep the pooled model as the separate pooled_geometry control, but do not
    substitute its parameters into any individual reference channel.
    """
    pooled, fc = train_geometry(refs, args)
    channels = []
    for name, st in refs.items():
        geom = _v41_learn_local_geometry({name: st}, 'Ti', 'O', outside_neighbors=3)
        # Each reference determines its own neighbor-shell cardinality K.
        # The batched kernel explicitly handles per-channel K (see
        # BatchedGeometry.all_energies: largest=max(...), then vec[:,:,:k]).
        # Enforcing pooled K here incorrectly rejects legitimate individual
        # channels, e.g. anatase TiTi K=4 versus pooled TiTi K=3.
        unknown=set(geom['topologies'])-set(pooled['topologies'])
        if unknown:
            raise ValueError(f'Individual channel {name}: unexpected topologies {sorted(unknown)}')
        channels.append(dict(names=[name], geometry=geom))
    print('Geometry channels (individual, no clustering):',
          ' | '.join(ch['names'][0] for ch in channels), flush=True)
    return pooled, fc, channels


def _soap_train(refs,args,channels,device='cpu'):
    if args.no_soap_training: return None
    print('Training SOAP Brownian scalar field once...',flush=True)
    models=[]
    for ci,ch in enumerate(channels):
        subset={n:refs[n] for n in ch['names']}
        print(f'  SOAP family {ci+1}/{len(channels)}: {"+".join(ch["names"])}',flush=True)
        models.append(_v43_brownian_training(subset,device,args.seed+191+ci*71,
            steps=args.soap_train_steps,trajectories=args.soap_trajectories,
            sigma_step_A=args.soap_noise_A,rmax=args.soap_rmax,nrad=args.soap_nrad,
            lmax=args.soap_lmax,kcap=args.soap_kcap,max_anchors=args.soap_anchors))
    return models


def _channel_energy(x,cell,species,channels,fc,tau):
    energies=[]
    for ch in channels:
        energies.append(_v43_geometry_network_energy(x,cell,species,ch['geometry'],fc))
    es=torch.stack(energies)
    # Normalize log-sum-exp by number of channels; no preference for more channels.
    return -tau*(torch.logsumexp(-es/tau,dim=0)-math.log(len(energies)))


def _soap_family_energy(x,cell,species,soap,tau):
    es=torch.stack([_v43_soap_energy(x,cell,species,m) for m in soap])
    return -tau*(torch.logsumexp(-es/tau,dim=0)-math.log(len(soap)))

def _objective(q,orbits,ti,cell,species,branch,sigma,pooled,fc,channels,soap,r0,args):
    o=expanded(q,orbits,ti.device)
    base=port_geometry(ti,o,cell,r0,sigma)
    if branch=='v140_port_control': return base
    x=torch.cat([ti,o],0)
    if branch=='pooled_geometry': g=_v43_geometry_network_energy(x,cell,species,pooled,fc)
    else: g=_channel_energy(x,cell,species,channels,fc,args.channel_temperature)
    if branch=='soap':
        if soap is None: raise RuntimeError('SOAP training disabled')
        return args.constraint_weight*base+args.soap_weight*_soap_family_energy(x,cell,species,soap,args.channel_temperature)
    if branch=='combined':
        if soap is None: raise RuntimeError('SOAP training disabled')
        return args.constraint_weight*base+args.geometry_weight*g+args.soap_weight*_soap_family_energy(x,cell,species,soap,args.channel_temperature)
    return args.constraint_weight*base+args.geometry_weight*g


def _minimize_generation(q0,orbits,ti,cell,species,branch,sigma,pooled,fc,channels,soap,r0,args):
    q=torch.nn.Parameter(q0.clone()); opt=torch.optim.Adam([q],lr=args.lr)
    bestE=float('inf'); best=q0.detach().clone(); start=time.perf_counter(); neval=0
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        E=_objective(q,orbits,ti,cell,species,branch,sigma,pooled,fc,channels,soap,r0,args)
        if not torch.isfinite(E): break
        E.backward(); opt.step(); neval+=1
        ev=float(E.detach())
        if ev<bestE: bestE=ev; best=q.detach().clone()
    return expanded(best,orbits,ti.device).detach(),dict(seconds=time.perf_counter()-start,evals=neval,best_energy=bestE)


def _generation_score(pred,ti,cell,r0):
    _,d=image_vectors(ti,pred,cell); d=d.detach().cpu().numpy()
    ct=(d<r0+.38).sum((1,2)); co=(d<r0+.38).sum((0,2))
    _,oo=image_vectors(pred,pred,cell); oo=oo.detach().cpu().numpy()
    idx=np.arange(len(pred)); oo[idx,idx,13]=np.inf
    mind=float(np.min(oo));
    # Connected bipartite Ti-O network in the periodically repeated unit cell.
    edges=np.any(d<r0+.38,axis=2)
    seen={0}; frontier=[0]; nTi=len(ti)
    while frontier:
        u=frontier.pop()
        if u<nTi: adj=[nTi+j for j in np.flatnonzero(edges[u])]
        else: adj=list(np.flatnonzero(edges[:,u-nTi]))
        for v in adj:
            if v not in seen: seen.add(v); frontier.append(v)
    return dict(Ti6=float(np.mean(ct==6)),O3=float(np.mean(co==3)),
                coord_defect=int(np.abs(ct-6).sum()+np.abs(co-3).sum()),
                OO_min_A=mind,connected=bool(len(seen)==len(ti)+len(pred)),
                valid=bool(np.all(ct==6) and np.all(co==3) and mind>=1.65 and len(seen)==len(ti)+len(pred)))


def _report_generation(rows,branches,widths,verbose=False):
    for name in dict.fromkeys(r['reference'] for r in rows):
        rr=[r for r in rows if r['reference']==name]
        print('\n'+'='*95+'\n'+f'{name.upper()} | SG {rr[0]["sg"]} | {len(rr)} trials'+'\n'+'='*95,flush=True)
        print(f'{"Method":<20} {"sigma":>6} {"N":>4} {"valid":>7} {"Ti6":>7} {"O3":>7} {"OOmin":>8} {"RMS*":>8} {"time(s)":>9}',flush=True)
        for b in branches:
            for w in widths:
                ss=[r for r in rr if r['branch']==b and abs(r['width_A']-w)<1e-8]
                if not ss: continue
                av=lambda k:float(np.mean([r[k] for r in ss]))
                print(f'{b:<20} {w:6.2f} {len(ss):4d} {av("valid"):7.2f} {av("Ti6"):7.2f} {av("O3"):7.2f} {min(r["OO_min_A"] for r in ss):8.3f} {min(r["rms_A"] for r in ss):8.3f} {sum(r["seconds"] for r in ss):9.1f}',flush=True)
        if verbose:
            for r in sorted(rr,key=lambda x:(x['branch'],x['width_A'],x['start'])):
                print('  ',r['branch'],'s'+str(r['start']),'sigma',r['width_A'],'valid',r['valid'],'RMS',round(r['rms_A'],3),flush=True)
    print('\n'+'='*95+'\nOVERALL GENERATION SUMMARY\n'+'='*95,flush=True)
    for b in branches:
        rr=[r for r in rows if r['branch']==b]
        if rr: print(f'{b:<20} N={len(rr):3d} valid={sum(r["valid"] for r in rr):3d} ({np.mean([r["valid"] for r in rr]):.1%}) GPU-trial-seconds={sum(r["seconds"] for r in rr):.1f}',flush=True)
    print('* RMS is diagnostic only, not the generation success criterion.',flush=True)


def main():
    p=argparse.ArgumentParser(description='Standalone Wyckoff oxygen generation: geometry families, SOAP and combined')
    p.add_argument('--reference-cif-dir',required=True)
    p.add_argument('--widths',default='0.20,0.35,0.50')
    p.add_argument('--branches',default='v140_port_control,pooled_geometry,channeled_geometry,soap,combined')
    p.add_argument('--starts',type=int,default=3); p.add_argument('--steps',type=int,default=120)
    p.add_argument('--lr',type=float,default=.025); p.add_argument('--seed',type=int,default=4717)
    p.add_argument('--device',default='auto'); p.add_argument('--gpus',type=int,default=0); p.add_argument('--cpus',type=int,default=0)
    p.add_argument('--output',default='data/xn_templates/oxygen_generation_results.json')
    p.add_argument('--train-mode',choices=('fast','select'),default='fast'); p.add_argument('--train-steps',type=int,default=20)
    p.add_argument('--channel-threshold',type=float,default=0.70)
    p.add_argument('--channel-temperature',type=float,default=0.12)
    p.add_argument('--geometry-weight',type=float,default=1.0)
    p.add_argument('--soap-weight',type=float,default=0.10)
    p.add_argument('--constraint-weight',type=float,default=0.05)
    p.add_argument('--soap-train-steps',type=int,default=3)
    p.add_argument('--soap-trajectories',type=int,default=3)
    p.add_argument('--soap-noise-A',type=float,default=0.08)
    p.add_argument('--soap-rmax',type=float,default=5.0)
    p.add_argument('--soap-nrad',type=int,default=4)
    p.add_argument('--soap-lmax',type=int,default=3)
    p.add_argument('--soap-kcap',type=int,default=12)
    p.add_argument('--soap-anchors',type=int,default=180)
    p.add_argument('--no-soap-training',action='store_true')
    p.add_argument('--verbose-trials',action='store_true')
    p.add_argument('--worker-index',type=int,default=-1,help=argparse.SUPPRESS)
    p.add_argument('--workers',type=int,default=1,help=argparse.SUPPRESS)
    p.add_argument('--training-cache',default='',help=argparse.SUPPRESS)
    a=p.parse_args(); torch.set_default_dtype(torch.float64)
    widths=[float(w) for w in a.widths.split(',')]; branches=a.branches.split(',')
    allowed={'v140_port_control','pooled_geometry','channeled_geometry','soap','combined'}
    if any(b not in allowed for b in branches): p.error('Unknown branch')
    if ('soap' in branches or 'combined' in branches) and a.no_soap_training: p.error('SOAP branches require SOAP training')
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    slurm=int(os.environ.get('SLURM_CPUS_PER_TASK','0') or 0)
    cpus=max(1,min(affinity,slurm or affinity,a.cpus or affinity))
    ngpu=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if a.gpus==0: a.gpus=max(1,ngpu)
    if a.gpus>ngpu and a.gpus>1: p.error('GPU request exceeds visible allocation')
    threads=max(1,cpus//(a.workers if a.worker_index>=0 else a.gpus))
    torch.set_num_threads(threads); os.environ['OMP_NUM_THREADS']=str(threads); os.environ['MKL_NUM_THREADS']=str(threads)
    if a.device=='auto': a.device='cuda:0' if ngpu else 'cpu'
    if a.worker_index<0:
        print(f'Resources: visible GPUs={ngpu}, workers={a.gpus}, CPU affinity={affinity}, SLURM CPUs={slurm or "unset"}, threads/worker={threads}',flush=True)
    if a.worker_index<0 and a.gpus>1:
        with tempfile.TemporaryDirectory(prefix='oxygen_generation_workers_') as tmp:
            refs,_=_load_reference_cifs(a.reference_cif_dir)
            pooled,fc,channels=_family_training(refs,a)
            soap=_soap_train(refs,a,channels)
            cache=Path(tmp)/'training.pt'; torch.save(dict(pooled=pooled,fc=fc,channels=channels,soap=soap),cache)
            print('Training completed once; dispatching independent trials.',flush=True)
            processes=[]; handles=[]; paths=[]
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
            for wi in range(a.gpus):
                out=Path(tmp)/f'worker_{wi}.json'; log=Path(tmp)/f'worker_{wi}.log'
                fh=log.open('w'); handles.append(fh); paths.append(log)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--reference-cif-dir',a.reference_cif_dir,
                     '--widths',a.widths,'--branches',a.branches,'--starts',str(a.starts),'--steps',str(a.steps),
                     '--lr',str(a.lr),'--seed',str(a.seed),'--device','cuda:0','--gpus','1','--cpus',str(threads),
                     '--output',str(out),'--worker-index',str(wi),'--workers',str(a.gpus),'--training-cache',str(cache),
                     '--channel-temperature',str(a.channel_temperature),'--geometry-weight',str(a.geometry_weight),
                     '--soap-weight',str(a.soap_weight),'--constraint-weight',str(a.constraint_weight)]
                env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=visible.split(',')[wi].strip() if visible else str(wi)
                print(f'Launching GPU worker {wi+1}/{a.gpus}: {env["CUDA_VISIBLE_DEVICES"]}',flush=True)
                processes.append(subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT))
            statuses=[pr.wait() for pr in processes]
            for h in handles: h.close()
            for i,code in enumerate(statuses):
                if code: print(f'Worker {i} failed:\n{paths[i].read_text()}',flush=True)
            if any(statuses): raise RuntimeError(f'Worker exit codes: {statuses}')
            results=[json.loads((Path(tmp)/f'worker_{i}.json').read_text()) for i in range(a.gpus)]
            rows=[r for res in results for r in res['trials']]
            rows.sort(key=lambda r:(list(refs).index(r['reference']),branches.index(r['branch']),r['width_A'],r['start']))
            _report_generation(rows,branches,widths,a.verbose_trials)
            result=dict(settings=vars(a),channels=[c['names'] for c in channels],trials=rows,
                summary={b:dict(n=sum(r['branch']==b for r in rows),valid=sum(r['branch']==b and r['valid'] for r in rows)) for b in branches},
                caveat='v140 is a port-inspired surrogate, not full v140 rotation/Sinkhorn SCF',
                scope='Frozen reference Ti, reference-derived oxygen Wyckoff orbit types; independent oxygen coordinates randomized')
            out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
            print('Output:',out,flush=True)
        return
    refs,_=_load_reference_cifs(a.reference_cif_dir)
    if a.training_cache:
        cached=torch.load(a.training_cache,map_location='cpu',weights_only=False)
        pooled,fc,channels,soap=(cached[k] for k in ('pooled','fc','channels','soap'))
    else:
        pooled,fc,channels=_family_training(refs,a)
        soap=_soap_train(refs,a,channels)
    dev=torch.device(a.device); rows=[]
    for ri,(name,st) in enumerate(refs.items()):
        orbits,sg=symmetry_orbits(st)
        cell,rf,sp=_v40_reference_state(st,dev)
        ti=rf[[i for i,x in enumerate(sp) if x=='Ti']].detach()
        true=rf[[i for i,x in enumerate(sp) if x=='O']].detach()
        species=['Ti']*len(ti)+['O']*len(true)
        r0=float(pooled['topologies']['TiO']['K_distance_median_A'])
        # Stable per-reference seeds independent of number of workers.
        rng=np.random.default_rng(a.seed+ri*10007)
        starts=[initial_q(orbits,rng,dev,rf.dtype) for _ in range(a.starts)]
        for si,q0 in enumerate(starts):
            for wi,sigma in enumerate(widths):
                for bi,branch in enumerate(branches):
                    job=(((ri*a.starts+si)*len(widths)+wi)*len(branches)+bi)
                    if a.worker_index>=0 and job%a.workers!=a.worker_index: continue
                    trial_fc={'topologies':{k:{**v,'inside':{**v['inside'],'outside_smear_A':sigma},
                        'outside':{**v['outside'],'outside_smear_A':sigma}} for k,v in fc['topologies'].items()}}
                    pred,cost=_minimize_generation(q0,orbits,ti,cell,species,branch,sigma,pooled,trial_fc,channels,soap,r0,a)
                    gen=_generation_score(pred,ti,cell,r0)
                    diagnostic=score(pred,true,ti,cell,r0)
                    rows.append(dict(reference=name,sg=sg,branch=branch,width_A=sigma,start=si,
                        **cost,**gen,rms_A=diagnostic['rms_A'],within_025=diagnostic['within_025'],
                        oxygen_frac=pred.detach().cpu().numpy().tolist()))
    if a.worker_index<0: _report_generation(rows,branches,widths,a.verbose_trials)
    result=dict(settings=vars(a),channels=[c['names'] for c in channels],trials=rows,
                caveat='v140 is a port-inspired surrogate, not full v140 rotation/Sinkhorn SCF')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    if a.worker_index<0: print('Output:',out,flush=True)



# -----------------------------------------------------------------------------
# Oxygen dynamics attribution experiment. SOAP is never trained or evaluated.
# -----------------------------------------------------------------------------

# Exact port-transport initializer and batched factorial dynamics follow.

import multiprocessing as mp
from scipy.optimize import least_squares, linear_sum_assignment

MODES = ('wyckoff_relax', 'wyckoff_langevin', 'wyckoff_hopping')
INITS = ('random_wyckoff', 'pooled_lcao', 'learned_lcao')
POLICIES = ('v140_port_control', 'pooled_geometry', 'channeled_geometry')
SHIFTS27 = np.array([(i, j, k) for i in (-1, 0, 1)
                     for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


from functools import lru_cache

@lru_cache(maxsize=16)
def _gpu_shift(device, dtype):
    return torch.as_tensor(SHIFTS27,device=device,dtype=dtype)

@lru_cache(maxsize=20)
def _feature_indices(n,device):
    ids=torch.arange(n,device=device)
    return torch.triu_indices(n,n,offset=1,device=device),torch.combinations(ids,r=3) if n>=3 else None


def element_orbits(st, element, tol=2e-4):
    """Symmetry-preserving independent orbit parameters, in native CIF cell."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    analyzer = SpacegroupAnalyzer(st, symprec=1e-3, angle_tolerance=5)
    ops = analyzer.get_symmetry_operations(cartesian=False)
    rotations = np.rint([op.rotation_matrix for op in ops]).astype(int)
    translations = np.asarray([op.translation_vector for op in ops], float)
    points = np.asarray([s.frac_coords for s in st if s.specie.symbol == element]) % 1
    cell = np.asarray(st.lattice.matrix, float)
    unused = set(range(len(points)))
    orbits = []
    while unused:
        idx = min(unused)
        seed = points[idx]
        images = (np.einsum('aij,j->ai', rotations, seed) + translations) % 1
        matching = []
        for target in images:
            delta = points - target
            delta -= np.rint(delta)
            dist = np.linalg.norm(delta @ cell, axis=1)
            j = int(dist.argmin())
            if dist[j] > tol:
                raise ValueError(f'{element} symmetry mapping error {dist[j]:.5g} Å')
            matching.append(j)
        members = sorted(set(matching))
        if not set(members).issubset(unused):
            raise ValueError(f'{element} orbit overlap')
        unused.difference_update(members)
        sel = []
        for j in members:
            diff = images - points[j]
            diff -= np.rint(diff)
            sel.append(int(np.linalg.norm(diff @ cell, axis=1).argmin()))
        matrows, rhsrows = [], []
        for r, t in zip(rotations, translations):
            v = r @ seed + t - seed
            if np.linalg.norm(v - np.rint(v)) < tol:
                matrows.append(r - np.eye(3))
                rhsrows.append(np.rint(v) - t)
        mat = np.concatenate(matrows, axis=0)
        rhs = np.concatenate(rhsrows)
        _, singular, vh = np.linalg.svd(mat, full_matrices=True)
        rank = int((singular > 1e-7).sum())
        basis = vh[rank:].T.copy()
        offset = np.linalg.lstsq(mat, rhs, rcond=None)[0] if rank else np.zeros(3)
        if np.linalg.norm(mat @ offset - rhs) > 1e-5:
            raise ValueError(f'{element} stabilizer inconsistent')
        # Verify that reference site lies on the inferred affine Wyckoff manifold.
        if basis.size:
            residual = seed - offset - basis @ (basis.T @ (seed - offset))
        else:
            residual = seed - offset
        if np.linalg.norm(residual @ cell) > 1e-3:
            raise ValueError(f'{element} Wyckoff fit inconsistent')
        orbits.append(dict(indices=members, rotations=rotations[sel],
            translations=translations[sel], basis=basis, offset=offset,
            multiplicity=len(members), dof=basis.shape[1]))
    return orbits, analyzer.get_space_group_number()


def orbit_design(orbits):
    """O = A @ q + b (unwrapped) for orbit-ordered fractional oxygen."""
    n = sum(len(orb['indices']) for orb in orbits)
    d = sum(int(orb['dof']) for orb in orbits)
    A = np.zeros((n, 3, d), float)
    b = np.empty((n, 3), float)
    k = 0; j = 0
    for ob in orbits:
        nd = int(ob['dof'])
        m = len(ob['indices'])
        if nd:
            A[j:j+m, :, k:k+nd] = np.einsum('nij,jk->nik', ob['rotations'], ob['basis'])
        b[j:j+m] = np.einsum('nij,j->ni', ob['rotations'], ob['offset']) + ob['translations']
        j += m; k += nd
    return A, b


def reference_orbit_parameters(orbits, positions, cell):
    """Recover *symmetry-legal* parameters from orbit representatives.

    Every element_orbits() entry was constructed from its first (lowest-index)
    member, with a verified affine representative ``offset + basis @ q``.
    Fitting the first representative avoids a false global least-squares
    assumption about wrapped fractional coordinates.  Critically, this works
    for any number of independent parameters: 0, 1, 2, 4, ... .
    """
    A, b = orbit_design(orbits)
    p = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if p.ndim != 2 or p.shape[-1] != 3 or cell.shape != (3, 3):
        raise ValueError('Expected fractional sites [N,3] and cell [3,3]')
    components = []
    for ob in orbits:
        basis = np.asarray(ob['basis'], dtype=float)
        if basis.shape[0] != 3 or basis.shape[1] != int(ob['dof']):
            raise ValueError('Invalid Wyckoff basis shape')
        if basis.shape[1]:
            representative = p[int(ob['indices'][0])]
            components.append(np.linalg.lstsq(basis,
                representative - np.asarray(ob['offset'], float), rcond=None)[0])
    q = np.concatenate(components) if components else np.empty(0, dtype=float)
    pred = (np.einsum('nik,k->ni', A, q) + b) % 1
    ref = np.concatenate([p[ob['indices']] for ob in orbits]) % 1
    residual = pred - ref
    # For oblique unit cells, choose the shortest periodic image in Cartesian
    # space; componentwise fractional rounding is not always the minimum.
    shift = SHIFTS27
    dr = (residual[:, None, :] + shift[None, :, :]) @ cell
    maxerr = float(np.sqrt(np.sum(dr * dr, axis=-1).min(axis=-1)).max())
    if maxerr > 1e-3:
        raise ValueError(f'Ti Wyckoff reference reconstruction error {maxerr:.5g} Å; '
                         'check orbit representative indices, basis and ordering')
    return q


def ti_displacements(ti_orbits, original, cell, magnitudes, starts, seed):
    """Perturb only independent Ti orbit parameters; no symmetry breaking."""
    rng = np.random.default_rng(seed)
    qref = reference_orbit_parameters(ti_orbits, original, cell)
    A,b = orbit_design(ti_orbits)
    d = len(qref)
    entries = []
    for magnitude in magnitudes:
        if magnitude < 0: raise ValueError('Ti perturbation must be nonnegative')
        if magnitude and not d:
            entries.append(dict(level=magnitude, applicable=False,
                reason='Ti has no positional Wyckoff freedom at fixed space group and cell'))
            continue
        for start in range(starts):
            q = qref.copy()
            if magnitude:
                direction = rng.standard_normal(d)
                dpos = np.einsum('nij,j->ni',A,direction) @ cell
                norm = np.sqrt(np.mean(np.sum(dpos**2,axis=-1)))
                q += direction * magnitude / max(norm,1e-12)
            pred = (np.einsum('nij,j->ni',A,q)+b)%1
            # ti_orbits indices may not be contiguous by orbit; return original order.
            reordered = np.zeros_like(pred)
            idx = np.concatenate([ob['indices'] for ob in ti_orbits])
            reordered[idx] = pred
            delta = reordered-original
            delta -= np.rint(delta)
            rms = float(np.sqrt(np.mean(np.sum((delta@cell)**2,axis=-1))))
            entries.append(dict(level=float(magnitude),start=start,applicable=True,
                ti=reordered, actual_rms_A=rms))
    return entries


def batched_expand(Q, A, b):
    """Exact symmetry mapping, batch-first, no Python loop over candidates."""
    return (torch.einsum('ndk,bk->bnd', A, Q) + b[None]) % 1.0


def b_geometry(O, Ti, cell, bond_ref, sigma):
    """v140 scalar surrogate evaluated for B structures concurrently."""
    B, no, _ = O.shape
    shift = _gpu_shift(O.device,O.dtype)
    dij = (O[:,None,:,None,:] - Ti[:,:,None,None,:] + shift) @ cell
    dist = torch.linalg.vector_norm(dij, dim=-1).clamp_min(1e-7)
    occ = torch.sigmoid((bond_ref + .35 - dist) / max(sigma, .03))
    ct = occ.sum((2,3)); co = occ.sum((1,3))
    ec = ((ct-6)**2).mean(-1) + ((co-3)**2).mean(-1)
    bond = (occ*(dist-bond_ref).square()).sum((1,2,3))/occ.sum((1,2,3)).clamp_min(1)
    oo = torch.linalg.vector_norm((O[:,:,None,None,:]-O[:,None,:,None,:]+shift)@cell,dim=-1)
    mask = torch.ones((no,no,27),dtype=torch.bool,device=O.device)
    idx = torch.arange(no,device=O.device); mask[idx,idx,13]=False
    rep = torch.nn.functional.softplus((1.70-oo[:,mask])/.10).square().sum(1)/max(no,1)
    return 2*ec + bond/.15**2 + 3*rep


def geometry_features(v, geometry):
    """Features matching _v41_feature_torch but over [B, sites, K, xyz]."""
    r = torch.linalg.vector_norm(v,dim=-1).clamp_min(1e-8)
    u = v/r[...,None]
    if geometry == 'bond': return torch.sort(r,dim=-1).values
    n = v.shape[-2]
    triu, triples = _feature_indices(n,v.device)
    cosine = torch.einsum('...id,...jd->...ij',u,u)
    angles = torch.sort(cosine[...,triu[0],triu[1]],dim=-1).values
    if geometry == 'angle': return angles
    if geometry == 'n_body': return torch.cat((torch.sort(r,dim=-1).values,angles),dim=-1)
    if geometry == 'four_body':
        indices = triples
        ua=u[...,indices[:,0],:]; ub=u[...,indices[:,1],:]; uc=u[...,indices[:,2],:]
        volumes = torch.abs((ua*torch.linalg.cross(ub,uc,dim=-1)).sum(-1))
        return torch.sort(volumes,dim=-1).values
    raise KeyError(geometry)


class BatchedGeometry:
    """Faithful tensorization of v42 inside/outside energies; batch & sites fused."""
    def __init__(self, cell, species, pooled, fc, channels, device, dtype):
        self.cell=cell; self.dtype=dtype; self.device=device
        self.species=species
        self.shift=_gpu_shift(device,dtype)
        self.n_ti=species.count('Ti')
        self.layouts=[]
        for topo,model in pooled['topologies'].items():
            ce=model['center']; ne=model['neighbor']
            ci=torch.tensor([i for i,s in enumerate(species) if s==ce],device=device)
            ni=torch.tensor([i for i,s in enumerate(species) if s==ne],device=device)
            self.layouts.append((topo,ci,ni,ce==ne))
        self.models=[self._prepare(pooled,fc)] + [self._prepare(c['geometry'],fc) for c in channels]

    def _prepare(self, learned,fc):
        entries={}
        for topo,m in learned['topologies'].items():
            fm=fc['topologies'][topo]
            # The force contract is trained on pooled data. An independent
            # reference may have fewer neighbors, so a pooled feature such as
            # OO/angle is mathematically undefined for that reference. Select
            # only the independently learned, supported invariants; never use
            # pooled anchors/scales as a fallback.
            requested=tuple((fm.get('inside') or {}).get('parts',()))
            kernels={}
            for key in requested:
                gm=m['geometries'].get(key)
                if gm is None or not gm.get('supported',False):
                    continue
                anchors=torch.as_tensor(gm['anchors'],dtype=self.dtype,device=self.device)
                scale=torch.as_tensor(gm['scale'],dtype=self.dtype,device=self.device).clamp_min(1e-5)
                if anchors.ndim != 2 or scale.ndim != 1 or anchors.shape[1] != scale.numel() or anchors.shape[0] == 0:
                    raise ValueError(f'Malformed learned feature {topo}/{key}: anchors={tuple(anchors.shape)}, scale={tuple(scale.shape)}')
                kernels[key]=(anchors,scale)
            if requested and not kernels:
                raise ValueError(f'No supported independent features for {topo}; requested={requested}, K={m["K"]}')
            entries[topo]=dict(k=int(m['K']),nout=int(m['outside_neighbors']),
                 boundary=float(m['boundary_A']),sm_in=float((fm.get('inside') or {}).get('outside_smear_A',0.20)),
                 sm_out=float((fm.get('outside') or {}).get('outside_smear_A',0.20)),
                 inside=bool(fm.get('inside')),outside=bool(fm.get('outside')),
                 kernels=kernels)
        return entries

    def all_energies(self, Ti, O, sigma=None):
        """Returns [pooled, channel_0,...] energies each of shape [B]."""
        xyz=torch.cat((Ti,O),dim=1)
        terms=[[] for _ in self.models]
        for topo,ci,ni,same in self.layouts:
            c=xyz[:,ci]; n=xyz[:,ni]
            # Build distances once for every channel / every oxygen candidate.
            v=n[:,None,:,None,:]+self.shift[None,None,None,:,:]-c[:,:,None,None,:]
            # v shape [B, Ncenter, Nneighbor, 27, 3]
            v=v@self.cell
            dd=torch.linalg.vector_norm(v,dim=-1)
            if same:
                dd=dd.masked_fill(dd<1e-8,1e6)
            configs=[mo[topo] for mo in self.models]
            largest=max(m['k']+m['nout'] for m in configs)
            if dd.shape[2]*27 < largest:
                raise ValueError(f'Insufficient periodic images for {topo}')
            idx=dd.flatten(-2).topk(largest,dim=-1,largest=False).indices
            flat=v.flatten(-3,-2)
            index=idx[...,None].expand(*idx.shape,3)
            vec=torch.gather(flat,dim=2,index=index)
            dst=torch.gather(dd.flatten(-2),dim=2,index=idx)
            for mid,m in enumerate(configs):
                k=m['k']; out=m['nout']; inside=vec[:,:,:k]
                zero=dst[:, :, 0].sum(-1)*0.0
                local=zero
                if m['inside']:
                    pieces=[]
                    for geom,(anchors,scale) in m['kernels'].items():
                        feature=geometry_features(inside,geom)
                        d2=(((feature[:,:,None,:]-anchors[None,None,:,:]) / scale[None,None,None,:])**2).mean(-1)
                        pieces.append(-.12*torch.logsumexp(-d2/.12,dim=-1))
                    if pieces:
                        local=torch.stack(pieces).mean(0).mean(-1)
                    sm=max(m['sm_in'] if sigma is None else float(sigma),.001)
                    escape=torch.nn.functional.softplus((dst[:,:,:k]-m['boundary'])/sm).square().mean((-1,-2))
                    terms[mid].append((local+escape,len(ci)))
                if m['outside'] and out:
                    exterior=dst[:,:,k:k+out]
                    sm=max(m['sm_out'] if sigma is None else float(sigma),.001)
                    intrusion=torch.nn.functional.softplus((m['boundary']-exterior)/sm).square()
                    decay=torch.exp(-torch.relu(exterior-m['boundary'])/sm)
                    terms[mid].append(((intrusion+.08*decay).mean((-1,-2)),len(ci)))
        energies=[]
        for t in terms:
            # Match old v43 averaging over all center/channel site contributions:
            # energy = average of center-averaged individual channel values, not
            # average of the chosen topology count.
            # Reweighted below using per-topology center counts.
            energies.append((sum(value*count for value,count in t)/sum(count for _,count in t)) if t else O.sum((1,2))*0)
        return torch.stack(energies,dim=0)

    def energy(self,Ti,O,policy,sigma,bond_ref,geometry_weight,constraint_weight,tau):
        control=b_geometry(O,Ti,self.cell,bond_ref,sigma)
        if policy=='v140_port_control': return control
        all_e=self.all_energies(Ti,O,sigma)
        if policy=='pooled_geometry': geom=all_e[0]
        elif policy=='channeled_geometry':
            # Soft minimum over complete structure-family channel energies;
            # never select a different channel independently at each site.
            geom=-tau*(torch.logsumexp(-all_e[1:]/tau,dim=0)-math.log(all_e.shape[0]-1))
        else: raise KeyError(policy)
        return constraint_weight*control + geometry_weight*geom


def _local_tio_vectors(st):
    samples=[]
    for i,site in enumerate(st):
        if site.specie.symbol != 'Ti': continue
        nn=_v41_numpy_knn(st,i,'O',7)
        if len(nn)<6: raise ValueError('Ti site with fewer than six reference oxygen neighbors')
        samples.append(np.asarray([z['vector'] for z in nn[:6]],float))
    return samples


def _rotation_invariants(v):
    norms=np.linalg.norm(v,axis=1)
    u=v/norms[:,None].clip(1e-9)
    iu=np.triu_indices(6,1)
    return np.r_[np.sort(norms),np.sort((u@u.T)[iu])]


def learn_port_templates(refs,channels):
    """Reference-learned rigid TiO6 medoids; no specific target O used at generation."""
    vectors={name:[np.asarray(v,float) for v in _local_tio_vectors(st)] for name,st in refs.items()}
    output=[]
    groups=[list(refs)] + [list(ch['names']) for ch in channels]
    for names in groups:
        candidate=[v for name in names for v in vectors[name]]
        feats=np.stack([_rotation_invariants(v) for v in candidate])
        scale=np.maximum(feats.std(0),np.r_[np.full(6,.10),np.full(15,.10)])
        z=feats/scale
        # Choose empirical medoid, never a hardcoded ideal octahedron.
        distances=((z[:,None,:]-z[None,:,:])**2).mean(-1)
        medoid=candidate[int(np.argmin(distances.sum(1)))].copy()
        output.append(dict(names=names,base=medoid,
             training_samples=len(candidate),medoid_invariant_spread=float(np.median(distances))))
    return output


def axis_angle_batch(w):
    """Rodrigues rotation in a single batched autograd graph (v140 port rotations)."""
    theta=torch.linalg.vector_norm(w,dim=-1,keepdim=True).clamp_min(1e-9)
    axis=w/theta
    x,y,z=axis.unbind(-1)
    zero=torch.zeros_like(x)
    cross=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*w.shape[:-1],3,3)
    identity=torch.eye(3,dtype=w.dtype,device=w.device)
    return identity+torch.sin(theta)[...,None]*cross+(1-torch.cos(theta))[...,None]*(cross@cross)


def _v140_nearest_triplets(arms_cart,cell):
    """Original v140 nearest-three-port assembly, parameterized by Ti count."""
    A=np.asarray(arms_cart,float).reshape(-1,3)
    C=np.asarray(cell,float); inv=np.linalg.inv(C)
    narm=len(A); nti=narm//6
    af=(A@inv)%1; owner=np.arange(narm)//6
    df=af[None,:,:]-af[:,None,:]; df-=np.rint(df)
    d2=np.sum((df@C)**2,axis=-1); np.fill_diagonal(d2,np.inf)
    choices=[]
    for a in range(narm):
        bs=np.where(owner!=owner[a])[0]
        if len(bs)==0: continue
        for b in bs[np.argsort(d2[a,bs])[:6]]:
            cs=np.where((owner!=owner[a])&(owner!=owner[b]))[0]
            if not len(cs): continue
            db=af[b]-af[a]; db-=np.rint(db)
            pb=A[a]; pb2=pb+db@C
            dc=af[cs]-af[a]; dc-=np.rint(dc)
            pc=pb+dc@C
            center=(pb+pb2+pc)/3
            variance=(np.sum((pb-center)**2,axis=-1)+np.sum((pb2-center)**2,axis=-1)+np.sum((pc-center)**2,axis=-1))/3
            k=int(np.argmin(variance))
            choices.append((float(variance[k]),(int(a),int(b),int(cs[k]))))
    choices.sort(key=lambda x:x[0])
    used=np.zeros(narm,bool); groups=[]
    for _,trio in choices:
        if used[list(trio)].any(): continue
        used[list(trio)]=True; groups.append(trio)
        if len(groups)==narm//3: break
    return groups


def _v140_sinkhorn_batch(arms,cell,centers,tau_A2,niter=30,owner_penalty=2.0):
    """Exact v140 row 1 / column 3 balanced transport with owner correction."""
    B,nti,_,_=arms.shape; no=centers.shape[1]
    af=arms.reshape(B,nti*6,3)@torch.linalg.inv(cell)
    df=af[:,:,None,:]-centers[:,None,:,:]
    df=df-torch.round(df)
    d2=((df@cell)**2).sum(-1)
    logits=-d2/max(float(tau_A2),1e-4)
    for outer in range(4):
        K=torch.exp((logits-logits.amax((-1,-2),keepdim=True)).clamp(min=-60))+1e-12
        for _ in range(max(5,int(niter))):
            K=K/K.sum(-1,keepdim=True).clamp_min(1e-12)
            K=K*(3./K.sum(-2,keepdim=True).clamp_min(1e-12))
        P=K/K.sum(-1,keepdim=True).clamp_min(1e-12)
        P=P*(3./P.sum(-2,keepdim=True).clamp_min(1e-12))
        owner=P.reshape(B,nti,6,no).sum(-2)
        if outer<3:
            logits=logits-owner_penalty*torch.relu(owner-1)[:,:,None,:].expand(B,nti,6,no).reshape_as(logits)
    return P


def _v140_centroids(arms,cell,P,reference):
    B=arms.shape[0]; af=arms.reshape(B,-1,3)@torch.linalg.inv(cell)
    df=af[:,:,None,:]-reference[:,None,:,:]
    df=df-torch.round(df).detach()
    unwrap=reference[:,None,:,:]+df
    center=(P[...,None]*unwrap).sum(1)/P.sum(1)[...,None].clamp_min(1e-9)
    return center


def _v140_hard_flow(P):
    """Original v140 min-cost flow: each arm once; 3 Ti-distinct arms per O."""
    import networkx as nx
    Q=np.asarray(P,float); narm,no=Q.shape; nti=narm//6
    graph=nx.DiGraph(); graph.add_node('s',demand=-narm); graph.add_node('t',demand=narm)
    for a in range(narm):
        an=f'a{a}'; graph.add_node(an,demand=0); graph.add_edge('s',an,capacity=1,weight=0)
        ti=a//6
        for oi in range(no):
            node=f'x{ti}_{oi}'
            if node not in graph: graph.add_node(node,demand=0)
            cost=int(round(-1e5*math.log(max(Q[a,oi],1e-12))))
            graph.add_edge(an,node,capacity=1,weight=cost)
    for ti in range(nti):
        for oi in range(no):
            graph.add_edge(f'x{ti}_{oi}',f'o{oi}',capacity=1,weight=0)
    for oi in range(no):
        graph.add_node(f'o{oi}',demand=0)
        graph.add_edge(f'o{oi}','t',capacity=3,weight=0)
    flow=nx.min_cost_flow(graph)
    groups=[[] for _ in range(no)]
    for a in range(narm):
        for x,amount in flow[f'a{a}'].items():
            if amount: groups[int(x.split('_')[1])].append(a)
    if any(len(g)!=3 or len({a//6 for a in g})!=3 for g in groups):
        raise RuntimeError('v140 flow produced invalid OTi3 ownership')
    return groups


def v140_initialize_batch(Ti,cell,base_templates,sigma,rotation_steps,seed):
    """Real v140 rigid-port rotation + detached Sinkhorn + hard OTi3 assignment.

    Ti is fixed (unlike v140 framework generation); no reference O is supplied.
    Shapes are truly batched on one device throughout port optimization.
    """
    B,nti,_=Ti.shape
    if B==0: return Ti.new_empty((0,2*nti,3)),[]
    no=2*nti
    generator=torch.Generator(device=Ti.device); generator.manual_seed(seed)
    base=torch.as_tensor(np.asarray(base_templates),dtype=Ti.dtype,device=Ti.device)
    if base.shape!=(B,6,3): raise ValueError('Port templates must have shape (batch,6,3)')
    # SO(3) exponential coordinates start with diverse random port orientations.
    rot=torch.nn.Parameter(torch.randn((B,nti,3),device=Ti.device,dtype=Ti.dtype,generator=generator)*1.5)
    stretch=torch.nn.Parameter(torch.zeros((B,nti,6),device=Ti.device,dtype=Ti.dtype))
    def arms():
        R=axis_angle_batch(rot)
        vectors=torch.einsum('bkj,bnij->bnki',base,R)
        return (Ti@cell)[:,:,None,:]+vectors*(1+.12*torch.tanh(stretch))[...,None]
    with torch.no_grad():
        initial=arms().detach().cpu().numpy(); C=cell.detach().cpu().numpy()
        initial_centers=[]
        for b in range(B):
            groups=_v140_nearest_triplets(initial[b],C)
            arm=initial[b].reshape(-1,3); inv=np.linalg.inv(C)
            af=(arm@inv)%1
            centers=[]
            used=set()
            for trio in groups:
                seed_af=af[trio[0]]
                delta=af[list(trio)]-seed_af
                delta-=np.rint(delta)
                centers.append((seed_af+delta.mean(0))%1)
                used.update(trio)
            # v140's greedy local triples need not cover every arm at random
            # rotations. Seed remaining latent centers by farthest-point port
            # sampling; balanced Sinkhorn/flow still constructs every final O.
            while len(centers)<no:
                eligible=np.array([i for i in range(len(af)) if i not in used],dtype=int)
                if not len(eligible):eligible=np.arange(len(af))
                current=np.asarray(centers,float)
                if len(current):
                    dd=af[eligible,None,:]-current[None,:,:]
                    dd-=np.rint(dd)
                    mind=((dd@C)**2).sum(-1).min(-1)
                    select=int(eligible[int(mind.argmax())])
                else:select=int(eligible[0])
                centers.append(af[select]);used.add(select)
            initial_centers.append(centers)
        centers=torch.as_tensor(np.asarray(initial_centers),dtype=Ti.dtype,device=Ti.device)
    optimizer=torch.optim.Adam((rot,stretch),lr=.055)
    lastP=None
    for step in range(max(1,int(rotation_steps))):
        if step%5==0 or lastP is None:
            with torch.no_grad():
                lastP=_v140_sinkhorn_batch(arms(),cell,centers,tau_A2=2*sigma*sigma,niter=20).detach()
                centers=_v140_centroids(arms(),cell,lastP,centers).detach()%1
        optimizer.zero_grad(set_to_none=True)
        a=arms(); af=a.reshape(B,-1,3)@torch.linalg.inv(cell)
        df=af[:,:,None,:]-centers[:,None,:,:]
        df=df-torch.round(df).detach()
        distance=torch.linalg.vector_norm(df@cell,dim=-1)
        width=max(.20*sigma,.015)
        outside=torch.nn.functional.softplus((distance-sigma)/width)*width
        share=(lastP*outside.square()).sum((1,2))/lastP.sum((1,2)).clamp_min(1)
        regularizer=.0025*torch.tanh(stretch).square().mean((1,2))
        loss=(share+regularizer).sum()
        if not torch.isfinite(loss): raise RuntimeError('v140 port rotation diverged')
        loss.backward(); optimizer.step()
    with torch.no_grad():
        a=arms()
        P=_v140_sinkhorn_batch(a,cell,centers,tau_A2=2*sigma*sigma,niter=40)
        hardP=P.detach().cpu().numpy(); af=(a.reshape(B,-1,3)@torch.linalg.inv(cell)).detach().cpu().numpy()
        cf=centers.detach().cpu().numpy(); C=cell.detach().cpu().numpy()
        projected=[]; data=[]
        for b in range(B):
            groups=_v140_hard_flow(hardP[b])
            oxygen=[]; spread=[]
            for oi,trio in enumerate(groups):
                delta=af[b,trio]-cf[b,oi]
                delta-=np.rint(delta)
                local=cf[b,oi]+delta
                cen=local.mean(0)
                spread.append(float(np.sqrt(np.mean(np.sum(((local-cen)@C)**2,axis=-1)))))
                oxygen.append(cen%1)
            projected.append(oxygen)
            entropy=float(-(hardP[b]*np.log(np.maximum(hardP[b],1e-12))).sum()/hardP.shape[1])
            data.append(dict(arm_merge_rms_A=float(np.sqrt(np.mean(np.array(spread)**2))),
                sinkhorn_entropy=entropy,Ti6_port_ownership=True,O3_port_ownership=True))
    return torch.as_tensor(np.asarray(projected),dtype=Ti.dtype,device=Ti.device),data


def _minimum_periodic_sq(target,candidate,cell):
    """Periodic assignment cost [Ncandidate,Ntarget], correct for oblique cells."""
    diff=candidate[:,None,None,:]-target[None,:,None,:]+SHIFTS27[None,None,:,:]
    return np.sum((diff@cell)**2,axis=-1).min(-1)


def _fit_orbits_to_unordered_oxygen(target,orbits,cell,seed,beam=8,maxiter=8):
    """Global unlabeled symmetry-orbit projection; no reference O coordinates.

    Builds orbit candidates from each proposed oxygen position, scores under
    periodic Hungarian matching, and optimizes the joint matched least squares.
    """
    target=np.asarray(target,float)%1
    A,b=orbit_design(orbits)
    nt,d=A.shape[0],A.shape[-1]
    if len(target)!=nt: raise ValueError('Incorrect oxygen count in orbit fit')
    if d==0:
        q=np.empty(0,float); pred=b%1
        cost=_minimum_periodic_sq(target,pred,cell)
        rr,cc=linear_sum_assignment(cost)
        return q,float(np.sqrt(cost[rr,cc].mean()))
    rng=np.random.default_rng(seed)
    # Beam candidates are generated from the oxygen proposals, not the known
    # reference orbit parameters. Each orbit is fitted without arbitrary labels.
    states=[(0.,[],frozenset())]
    offset=0; col=0
    for ob in orbits:
        m=int(ob['multiplicity']); nd=int(ob['dof'])
        rot=np.asarray(ob['rotations'][0],float)
        tr=np.asarray(ob['translations'][0],float)
        inverse=np.linalg.inv(rot)
        choices=[]
        for tid,pos in enumerate(target):
            gen=inverse@(pos-tr)
            coef=ob['basis'].T@(gen-ob['offset']) if nd else np.empty(0)
            if nd:
                subA=A[offset:offset+m,:,col:col+nd]
                pred=(np.einsum('nij,j->ni',subA,coef)+b[offset:offset+m])%1
            else: pred=b[offset:offset+m]%1
            choices.append((coef,pred,tid))
        extended=[]
        for prior,params,used in states:
            available=np.array([j for j in range(nt) if j not in used],dtype=int)
            for coef,pred,_ in choices:
                cost=_minimum_periodic_sq(target[available],pred,cell)
                ii,jj=linear_sum_assignment(cost)
                selected=frozenset(available[jj].tolist())
                total=prior+float(cost[ii,jj].sum())
                extended.append((total,params+[coef],used|selected))
        states=sorted(extended,key=lambda x:x[0])[:beam]
        offset+=m; col+=nd
    best_q=None; best_score=float('inf')
    Aw=np.einsum('nik,ij->njk',A,cell).reshape(-1,d)
    for _,coeffs,_ in states:
        q=np.concatenate(coeffs)
        for _ in range(maxiter):
            predicted=(np.einsum('nij,j->ni',A,q)+b)%1
            cost=_minimum_periodic_sq(target,predicted,cell)
            row,colids=linear_sum_assignment(cost)
            assigned=np.empty_like(predicted); assigned[row]=target[colids]
            raw=np.einsum('nij,j->ni',A,q)+b
            delta=assigned-raw
            # nearest Cartesian periodic image per matched pair
            ds=(delta[:,None,:]+SHIFTS27[None,:,:])@cell
            shift_idx=np.sum(ds**2,axis=-1).argmin(1)
            shift=SHIFTS27[shift_idx]
            rhs=((assigned+shift-b)@cell).reshape(-1)
            qnew=np.linalg.lstsq(Aw,rhs,rcond=None)[0]
            if np.linalg.norm(qnew-q)<1e-7:
                q=qnew; break
            q=qnew
        predicted=(np.einsum('nij,j->ni',A,q)+b)%1
        cost=_minimum_periodic_sq(target,predicted,cell)
        ri,ci=linear_sum_assignment(cost)
        val=float(cost[ri,ci].mean())
        if val<best_score: best_score=val; best_q=q.copy()
    return best_q,float(np.sqrt(best_score))


def batch_quality(Ti,O,cell,bond_ref):
    """Periodic-image coordination metrics; no unique-pair undercounting."""
    d=torch.linalg.vector_norm((O[:,None,:,None,:]-Ti[:,:,None,None,:]
                                +_gpu_shift(O.device,O.dtype))@cell,dim=-1)
    nTi=(d<bond_ref+.38).sum((2,3))
    nO=(d<bond_ref+.38).sum((1,3))
    no=O.shape[1]
    oo=torch.linalg.vector_norm((O[:,:,None,None,:]-O[:,None,:,None,:]
                      +_gpu_shift(O.device,O.dtype))@cell,dim=-1)
    mask=torch.ones((no,no,27),dtype=torch.bool,device=O.device)
    diag=torch.arange(no,device=O.device);mask[diag,diag,13]=False
    smallest=oo[:,mask].amin(-1)
    edges=(d<bond_ref+.38).any(-1).detach().cpu().numpy()
    connected=[]
    for edge in edges:
        nti,nox=edge.shape
        seen={0};pending=[0]
        while pending:
            u=pending.pop()
            adjacent=([nti+j for j in np.flatnonzero(edge[u])]
                      if u<nti else list(np.flatnonzero(edge[:,u-nti])))
            for dest in adjacent:
                if dest not in seen:seen.add(dest);pending.append(dest)
        connected.append(len(seen)==nti+nox)
    conn=torch.tensor(connected,device=O.device,dtype=torch.bool)
    valid=(nTi==6).all(-1)&(nO==3).all(-1)&(smallest>=1.65)&conn
    defect=(nTi-6).abs().sum(-1)+(nO-3).abs().sum(-1)
    return dict(Ti6=(nTi==6).to(O.dtype).mean(-1),O3=(nO==3).to(O.dtype).mean(-1),
                OO_min_A=smallest,coord_defect=defect,connected=conn,local_valid=valid)


def local_orbit_quality(Ti,O,cell,bond_ref,orbit_slices):
    """Local defect score for recombination; never assigns independent family channels."""
    delta=(O[:,None,:,None,:]-Ti[:,:,None,None,:]
           +_gpu_shift(O.device,O.dtype))@cell
    d=torch.linalg.vector_norm(delta,dim=-1).transpose(1,2).flatten(-2)
    closest=torch.topk(d,k=min(3,d.shape[-1]),dim=-1,largest=False).values
    degree=(d<bond_ref+.38).sum(-1)
    bad=(degree-3).abs().to(O.dtype)+2*((closest-bond_ref)**2).mean(-1)
    # O-O close contact penalties are also associated with the offending orbit.
    oo=torch.linalg.vector_norm((O[:,:,None,None,:]-O[:,None,:,None,:]
           +_gpu_shift(O.device,O.dtype))@cell,dim=-1)
    bad+=torch.nn.functional.relu(1.7-oo).square().sum((2,3))
    return torch.stack([bad[:,s].mean(-1) for s in orbit_slices],dim=1)


def optimize_population(q0,Ti,A,b,cell,policy,mode,sigma,geom,bond_ref,args,groups,orbit_q_slices,orbit_o_slices,seed):
    """All dynamics take place entirely in the exact Wyckoff parameter space.

    Mode wyckoff_langevin explicitly cycles local deterministic relaxation,
    stochastic Wyckoff-space exploration and deterministic relaxation, with
    a Metropolis basin-acceptance decision and a separately retained best
    energy configuration.  No independent Cartesian oxygen dynamics.
    """
    if mode not in MODES:
        raise ValueError(f'Unsupported (symmetry-breaking) dynamics: {mode}')
    start=time.perf_counter()
    generator=torch.Generator(device=q0.device);generator.manual_seed(seed)
    B=q0.shape[0]
    x=q0.detach().clone()
    def energy(z):
        return geom.energy(Ti,batched_expand(z,A,b),policy,sigma,bond_ref,
                           args.geometry_weight,args.constraint_weight,
                           args.channel_temperature)
    best=x.clone(); best_E=torch.full((B,),float('inf'),device=x.device,dtype=x.dtype)
    neval=0
    accepted=torch.zeros(B,device=x.device,dtype=torch.long)
    exchanged=torch.zeros(B,device=x.device,dtype=torch.long)
    transitions=torch.zeros(B,device=x.device,dtype=torch.long)
    # A: [N_O,3,d]. Convert its Cartesian index (3), not its variable index (d).
    derivative=torch.einsum('nik,il->nlk',A,cell).reshape(-1,A.shape[-1])
    column_size=torch.sqrt(derivative.square().sum(0)/max(A.shape[0],1)).clamp_min(1e-8)
    # Independent parameter increments are normalized to Cartesian Å RMS.
    def perturb(z,scale):
        if scale<=0:return z.clone()
        rand=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=generator)
        norm=torch.sqrt(((rand*column_size[None])**2).sum(-1,keepdim=True)).clamp_min(1e-8)
        return z+rand*(scale/norm)
    def relax(z,nsteps,langevin=False,noise_scale=0.):
        nonlocal best,best_E,neval
        p=torch.nn.Parameter(z.clone()); optimizer=torch.optim.Adam([p],lr=args.lr)
        for k in range(nsteps):
            optimizer.zero_grad(set_to_none=True)
            values=energy(p)
            if not bool(torch.isfinite(values).all().item()):
                raise RuntimeError(f'Nonfinite energy in {policy}/{mode}')
            with torch.no_grad():
                improve=values<best_E
                best_E=torch.minimum(best_E,values.detach())
                best=torch.where(improve[:,None],p.detach(),best)
            values.sum().backward();optimizer.step();neval+=1
            if langevin and noise_scale>0:
                with torch.no_grad():p.copy_(perturb(p,noise_scale))
        with torch.no_grad():
            values=energy(p)
            if not bool(torch.isfinite(values).all().item()):
                raise RuntimeError(f'Nonfinite terminal energy in {policy}/{mode}')
            improve=values<best_E
            best_E=torch.minimum(best_E,values)
            best=torch.where(improve[:,None],p.detach(),best)
        neval+=1
        return p.detach(),values.detach()

    if mode=='wyckoff_relax':
        relax(x,args.steps)
    elif mode=='wyckoff_langevin':
        cycles=min(max(1,int(args.langevin_cycles)),max(1,args.steps//3))
        first=max(1,args.steps//(2*cycles+1))
        incumbent,incE=relax(x,first)
        remaining=args.steps-first
        for cycle in range(cycles):
            remaining_cycles=cycles-cycle
            allocated=remaining//remaining_cycles
            nthermal=max(1,allocated//3)
            nsettle=allocated-nthermal
            if nsettle<1:nsettle=1;nthermal=max(0,allocated-1)
            # Three phases: relax (from previous cycle) -> thermal exploration
            # -> quench. Thermal noise never breaks any Wyckoff orbit.
            thermal,_=relax(incumbent,nthermal,True,args.noise_A)
            candidate,newE=relax(thermal,nsettle)
            delta=newE-incE
            rnd=torch.rand((B,),device=x.device,generator=generator)
            mask=(delta<=0)|(rnd<torch.exp((-delta/max(args.metropolis_temp,1e-8)).clamp(max=0)))
            disp=torch.linalg.vector_norm(
                (batched_expand(candidate,A,b)-batched_expand(incumbent,A,b)
                 -torch.round(batched_expand(candidate,A,b)-batched_expand(incumbent,A,b)).detach())@cell,
                dim=-1).square().mean(-1).sqrt()
            transitions+=(mask & (disp>args.basin_threshold_A)).long()
            incumbent=torch.where(mask[:,None],candidate,incumbent)
            incE=torch.where(mask,newE,incE)
            accepted+=mask.long()
            remaining-=allocated
    elif mode=='wyckoff_hopping':
        interval=max(2,int(args.hop_interval))
        initial=min(interval,args.steps)
        incumbent,incE=relax(x,initial)
        used=initial
        while used<args.steps:
            oxygen=batched_expand(incumbent,A,b)
            scores=local_orbit_quality(Ti,oxygen,cell,bond_ref,orbit_o_slices)
            for oi,sl in enumerate(orbit_q_slices):
                if sl.start==sl.stop:scores[:,oi]=-1e9
            proposal=incumbent.clone()
            for group in groups.values():
                if len(group)<2:continue
                indices=torch.as_tensor(group,device=x.device,dtype=torch.long)
                local=scores[indices]
                worst=local.argmax(-1)
                bestdonors=local.argmin(0)
                for ridx,global_idx in enumerate(group):
                    j=int(worst[ridx].item())
                    donor=group[int(bestdonors[j].item())]
                    if donor==global_idx or local[ridx,j]<=local[bestdonors[j],j]:continue
                    s=orbit_q_slices[j]
                    proposal[global_idx,s]=incumbent[donor,s]
                    exchanged[global_idx]+=1
            proposal=perturb(proposal,args.hop_A)
            candidate,newE=relax(proposal,min(interval,args.steps-used))
            delta=newE-incE
            rnd=torch.rand((B,),device=x.device,generator=generator)
            mask=(delta<=0)|(rnd<torch.exp((-delta/max(args.metropolis_temp,1e-8)).clamp(max=0)))
            dO=(batched_expand(candidate,A,b)-batched_expand(incumbent,A,b)+.5)%1-.5
            disp=torch.linalg.vector_norm(dO@cell,dim=-1).square().mean(-1).sqrt()
            transitions+=(mask & (disp>args.basin_threshold_A)).long()
            incumbent=torch.where(mask[:,None],candidate,incumbent)
            incE=torch.where(mask,newE,incE)
            accepted+=mask.long()
            used+=interval
    O=batched_expand(best.detach(),A,b)
    if x.device.type=='cuda':torch.cuda.synchronize(x.device)
    elapsed=time.perf_counter()-start
    after=batch_quality(Ti,O,cell,bond_ref)
    return O.detach(),dict(elapsed=elapsed,evals=neval,best_energy=best_E.cpu().tolist(),
         post_valid=after['local_valid'].cpu().tolist(),hop_accepted=accepted.cpu().tolist(),
         orbit_exchanges=exchanged.cpu().tolist(),basin_transitions=transitions.cpu().tolist()),after


def learn_ti_centered_lcao(refs):
    """Empirical Ti-centered radial/angular basis for three-center oxygen sharing.

    Training uses reference Ti/O distances and angles, not target O positions at
    generation. The three Ti-centered amplitudes combine into an O3 site score.
    This is a geometric LCAO surrogate, not a Kohn-Sham orbital calculation.
    """
    models={}
    for name,st in refs.items():
        cell=np.asarray(st.lattice.matrix,float)
        ti=np.asarray([site.frac_coords for site in st if site.specie.symbol=='Ti'],float)
        oxygen=np.asarray([site.frac_coords for site in st if site.specie.symbol=='O'],float)
        vectors=(ti[None,:,None,:]+SHIFTS27[None,None,:,:]-oxygen[:,None,None,:])@cell
        vec=vectors.reshape(len(oxygen),-1,3)
        distance=np.linalg.norm(vec,axis=-1)
        nearest=np.argpartition(distance,3,axis=-1)[:,:3]
        chosen=np.take_along_axis(vec,nearest[:,:,None],axis=1)
        rr=np.linalg.norm(chosen,axis=-1)
        u=chosen/np.maximum(rr[:,:,None],1e-9)
        cos=np.stack([(u[:,i]*u[:,j]).sum(-1) for i,j in ((0,1),(0,2),(1,2))],-1)
        features=np.concatenate([np.sort(rr,axis=-1),np.sort(cos,axis=-1)],axis=-1)
        if not np.isfinite(features).all():raise ValueError(f'Invalid LCAO reference basis for {name}')
        models[name]=features
    return models


def lcao_cartesian_energy(Ti,O,cell,models,sigma,learned=True):
    """Ti-centered 3-body orbital-overlap score evaluated in Cartesian space.

    Every oxygen is *already* a symmetry-expanded Wyckoff coordinate. No
    projection, Cartesian oxygen parameter, or target-reference oxygen lookup.
    """
    shifts=_gpu_shift(O.device,O.dtype)
    v=(Ti[:,None,:,None,:]+shifts[None,None,None,:,:]-O[:,:,None,None,:])@cell
    v=v.flatten(2,3)
    dd=torch.linalg.vector_norm(v,dim=-1)
    if dd.shape[-1]<3:raise ValueError('LCAO needs at least three Ti periodic images')
    idx=dd.topk(3,dim=-1,largest=False).indices
    vec=torch.gather(v,2,idx[...,None].expand(*idx.shape,3))
    r=torch.linalg.vector_norm(vec,dim=-1).clamp_min(1e-8)
    unit=vec/r[...,None]
    cs=torch.stack([(unit[:,:,i]*unit[:,:,j]).sum(-1) for i,j in ((0,1),(0,2),(1,2))],-1)
    features=torch.cat((r.sort(-1).values,cs.sort(-1).values),-1)
    energies=[]
    for anchor in models:
        a=anchor.to(device=O.device,dtype=O.dtype)
        radial=max(float(sigma),0.12)
        scale=torch.tensor([radial]*3+[0.18]*3,device=O.device,dtype=O.dtype)
        diff=(features[:,:,:,None] if False else features[:,:,None,:])-a[None,None,:,:]
        d2=((diff/scale)**2).mean(-1)
        # The overlap of Ti-centered radial/angular functions acts as a
        # smooth learned three-center O3 preference.
        site=-0.20*torch.logsumexp(-d2/0.20,dim=-1)
        energies.append(site.mean(-1))
    if learned and len(energies)>1:
        es=torch.stack(energies,0)
        preference=-0.12*(torch.logsumexp(-es/0.12,dim=0)-math.log(len(energies)))
    else:preference=energies[0]
    # Differentiable occupancy regularizers discourage collapse onto one Ti
    # triplet, using the same periodic Cartesian Ti-O distances.
    soft=torch.sigmoid((2.35-dd)/0.12)
    # Three Ti images per O; six O images per Ti, evaluated with periodic images.
    o3=(soft.sum(-1)-3).square().mean(-1)
    nti=Ti.shape[1]
    soft_per_ti=soft.reshape(O.shape[0],O.shape[1],nti,shifts.shape[0]).sum((1,3))
    ti6=(soft_per_ti-6).square().mean(-1)
    return preference+0.20*o3+0.12*ti6


def symmetry_lcao_initialize(Ti,A,b,cell,anchors,sigma,steps,seed,learned):
    """Optimize *only* independent Wyckoff variables against Cartesian LCAO."""
    B=Ti.shape[0];D=A.shape[-1]
    if D==0:return torch.zeros((B,0),device=Ti.device,dtype=Ti.dtype)
    generator=torch.Generator(device=Ti.device);generator.manual_seed(int(seed))
    # Multiple independent Wyckoff starts, with no unconstrained O proposal.
    Q=torch.rand((B,D),device=Ti.device,dtype=Ti.dtype,generator=generator)*2-0.5
    q=torch.nn.Parameter(Q)
    opt=torch.optim.Adam([q],lr=0.055)
    best=q.detach().clone();best_e=torch.full((B,),float('inf'),device=Ti.device,dtype=Ti.dtype)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        O=batched_expand(q,A,b)
        e=lcao_cartesian_energy(Ti,O,cell,anchors,sigma,learned)
        if not torch.isfinite(e).all():raise RuntimeError('Nonfinite symmetry-first LCAO energy')
        with torch.no_grad():
            mask=e<best_e;best=torch.where(mask[:,None],q.detach(),best);best_e=torch.minimum(best_e,e.detach())
        e.sum().backward();opt.step()
    with torch.no_grad():
        e=lcao_cartesian_energy(Ti,batched_expand(q,A,b),cell,anchors,sigma,learned)
        mask=e<best_e;best=torch.where(mask[:,None],q.detach(),best)
    return best.detach()


def prepare_crystal(st,refs,channels,args,device,crystal_index):
    """Create all paired initial configurations; Sinkhorn proposals are shared across policies."""
    cell,frac,species=_v40_reference_state(st,device)
    originalTi=frac[[i for i,x in enumerate(species) if x=='Ti']].cpu().numpy()
    originalO=frac[[i for i,x in enumerate(species) if x=='O']].cpu().numpy()
    oorbits,sg=element_orbits(st,'O'); tiorbits,_=element_orbits(st,'Ti')
    A_np,b_np=orbit_design(oorbits)
    A=torch.as_tensor(A_np,dtype=frac.dtype,device=device)
    b=torch.as_tensor(b_np,dtype=frac.dtype,device=device)
    levels=[float(x) for x in args.ti_perturbations.split(',')]
    widths=[float(x) for x in args.widths.split(',')]
    frameworks=ti_displacements(tiorbits,originalTi,cell.cpu().numpy(),levels,1,
                                 args.seed+crystal_index*1009)
    failures=[];rows=[]
    rankgeom=BatchedGeometry(cell,['Ti']*len(originalTi)+['O']*len(originalO),
                             args._pooled,args._fc,channels,device,frac.dtype)
    for fw_id,frame in enumerate(frameworks):
        if not frame['applicable']:
            failures.append(dict(crystal=args._crystal,Ti_perturbation_A=frame['level'],
                             status='not_applicable',reason=frame['reason']))
            continue
        ti=torch.as_tensor(frame['ti'],dtype=frac.dtype,device=device)
        for width in widths:
            for startid in range(args.starts):
                for initializer in args.initializers.split(','):
                    entry=dict(crystal=args._crystal,sg=sg,frame_id=fw_id,
                        Ti_perturbation_A=frame['level'],actual_Ti_rms_A=frame['actual_rms_A'],
                        width_A=width,start=startid,initializer=initializer)
                    rows.append((entry,ti))
    if not rows:
        return dict(rows=[],skipped=failures,cell=cell,A=A,b=b,orbits=oorbits,
                    real_O=originalO,sg=sg)
    batchTi=torch.stack([z[1] for z in rows])
    metadata=[z[0] for z in rows]
    outputQ=[None]*len(rows)
    outputInfo=[None]*len(rows)
    rng=np.random.default_rng(args.seed+crystal_index*31)
    for idx,m in enumerate(metadata):
        if m['initializer']=='random_wyckoff':
            outputQ[idx]=rng.uniform(-1.5,1.5,A.shape[-1])
            outputInfo[idx]=dict(initial_projection_rms_A=0.,port_method='none')
    # Symmetry-first LCAO: every Cartesian score is evaluated on oxygen
    # positions expanded from Wyckoff q. No hard-flow oxygen fitting.
    for width in widths:
        for method in ('pooled_lcao','learned_lcao'):
            indices=[i for i,m in enumerate(metadata) if m['width_A']==width and m['initializer']==method]
            if not indices:continue
            # Pooled model is one shared empirical Ti-centered basis. Learned
            # model is a soft minimum across five separate reference channels.
            anchors=(args._lcao_pooled if method=='pooled_lcao' else args._lcao_channels)
            q=symmetry_lcao_initialize(batchTi[indices],A,b,cell,anchors,width,
                  args.port_steps,args.seed+crystal_index*101+int(width*1000)
                  +(0 if method=='pooled_lcao' else 7000),method=='learned_lcao')
            qlist=q.cpu().numpy()
            for ii,index in enumerate(indices):
                outputQ[index]=qlist[ii]
                outputInfo[index]=dict(initial_projection_rms_A=0.0,
                    port_method='symmetry_first_ti_centered_lcao',
                    lcao_basis='individual_reference' if method=='learned_lcao' else 'pooled')
    if any(x is None for x in outputQ):raise RuntimeError('Initializer did not produce every oxygen orbit configuration')
    Q=torch.as_tensor(np.asarray(outputQ),dtype=frac.dtype,device=device)
    return dict(rows=metadata,Ti=batchTi,Q=Q,initinfo=outputInfo,skipped=failures,
        cell=cell,A=A,b=b,orbits=oorbits,real_O=originalO,sg=sg)


def one_worker(worker_id,nworkers,refs,pooled,fc,channels,args,queue):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if args.gpus:
            device=torch.device(f'cuda:{worker_id}')
            torch.cuda.set_device(device)
        else: device=torch.device('cpu')
        names=[name for name in refs if name.lower()!='rutile']
        allrows=[]; skipped=[]
        for ci,name in enumerate(names):
            if ci % nworkers != worker_id:continue
            args._crystal=name
            args._pooled=pooled;args._fc=fc
            args._bond_ref=float(pooled['topologies']['TiO']['K_distance_median_A'])
            begin=time.perf_counter()
            data=prepare_crystal(refs[name],refs,channels,args,device,ci)
            skipped.extend(data['skipped'])
            if not data['rows']:
                queue.put(('block',f'{name.upper()}: no symmetry-compatible Ti frameworks for requested perturbations'))
                continue
            Q=data['Q'];Ti=data['Ti'];A=data['A'];b=data['b'];cell=data['cell']
            metadata=data['rows'];args._oxygen_orbits=data['orbits']
            bondref=args._bond_ref
            species=['Ti']*Ti.shape[1]+['O']*b.shape[0]
            geom=BatchedGeometry(cell,species,pooled,fc,channels,device,Q.dtype)
            initialO=batched_expand(Q,A,b)
            initial=batch_quality(Ti,initialO,cell,bondref)
            inittime=time.perf_counter()-begin
            for policy in args.policies.split(','):
                for width in [float(w) for w in args.widths.split(',')]:
                    indices=[i for i,m in enumerate(metadata) if m['width_A']==width]
                    if not indices:continue
                    ind=torch.tensor(indices,device=device,dtype=torch.long)
                    Qt=Q[ind];Tt=Ti[ind]
                    subgroup={}
                    for j,ix in enumerate(indices):
                        subgroup.setdefault(metadata[ix]['frame_id'],[]).append(j)
                    # Recombine only oxygen orbits attached to the identical Ti framework.
                    orbit_slices=[];pos=0
                    for ob in data['orbits']:
                        orbit_slices.append(slice(pos,pos+ob['multiplicity']))
                        pos+=ob['multiplicity']
                    orbit_q_slices=[];pos=0
                    for ob in data['orbits']:
                        orbit_q_slices.append(slice(pos,pos+ob['dof']))
                        pos+=ob['dof']
                    if 'hopping' in args.optimizers and len(subgroup):
                        if any(len(g)<2 for g in subgroup.values()):
                            raise ValueError('Population hopping requires at least two candidates per Ti framework')
                    with torch.no_grad():
                        energy_initial=geom.energy(Tt,batched_expand(Qt,A,b),policy,width,bondref,
                            args.geometry_weight,args.constraint_weight,args.channel_temperature).cpu().numpy()
                        orbit_initial=local_orbit_quality(Tt,batched_expand(Qt,A,b),cell,bondref,orbit_slices).cpu().numpy()
                    # scores for reporting are evaluated identically for each method
                    for mi,mode in enumerate(args.optimizers.split(',')):
                        method_begin=time.perf_counter()
                        # Hopping transfers parameter blocks, whereas local score
                        # needs oxygen atom slices. Both are provided below.
                        if mode=='wyckoff_hopping':
                            groups={k:g for k,g in subgroup.items()}
                        else:groups={}
                        seed=args.seed+100000*ci+10000*mi+int(width*1000)
                        # A paired mode sees identical Qt and Ti irrespective of policy.
                        oxygen,perf,metrics=optimize_population(
                            Qt,Tt,A,b,cell,policy,mode,width,geom,bondref,args,
                            groups,orbit_q_slices,orbit_slices,seed)
                        after_exact=oxygen.detach().cpu().numpy()
                        baseline=initialO[ind].detach().cpu().numpy()
                        Ti_np=Tt.detach().cpu().numpy()
                        with torch.no_grad():
                            orbit_final=local_orbit_quality(Tt,oxygen,cell,bondref,orbit_slices).cpu().numpy()
                        original=data['real_O'];C=cell.cpu().numpy()
                        # Cross-policy scoring avoids self-energy validation bias.
                        with torch.no_grad():
                            baseline_energies={p:geom.energy(Tt,oxygen,p,width,bondref,
                                args.geometry_weight,args.constraint_weight,
                                args.channel_temperature).cpu().numpy() for p in POLICIES}
                        for j,ix in enumerate(indices):
                            met=metadata[ix].copy(); initinfo=data['initinfo'][ix]
                            tin=torch.as_tensor(Ti_np[j],dtype=Q.dtype,device=device)
                            final_o=torch.as_tensor(after_exact[j],dtype=Q.dtype,device=device)
                            initial_o=torch.as_tensor(baseline[j],dtype=Q.dtype,device=device)
                            first=_generation_score(initial_o,tin,cell,bondref)
                            last=_generation_score(final_o,tin,cell,bondref)
                            rms=score(final_o,torch.as_tensor(original,dtype=Q.dtype,device=device),
                                      tin,cell,bondref)['rms_A']
                            row={**met,**initinfo,**last,'policy':policy,'optimizer':mode,
                                'initial_valid':first['valid'],'initial_Ti6':first['Ti6'],
                                'initial_O3':first['O3'],'initial_coord_defect':first['coord_defect'],
                                'repaired':bool((not first['valid']) and last['valid']),
                                'damaged':bool(first['valid'] and not last['valid']),
                                'rms_A':rms,'best_energy':perf['best_energy'][j],
                                'initial_policy_energy':float(energy_initial[j]),
                                'final_policy_energy':float(baseline_energies[policy][j]),
                                'orbit_quality_initial':orbit_initial[j].tolist(),
                                'orbit_quality_final':orbit_final[j].tolist(),
                                'hops_accepted':perf['hop_accepted'][j],
                                'basin_transitions':perf['basin_transitions'][j],
                                'orbit_exchanges':perf['orbit_exchanges'][j],
                                'evaluations':perf['evals'],
                                'batch_wall_seconds':perf['elapsed'],
                                'amortized_optimization_seconds':perf['elapsed']/len(indices),
                                'cross_policy_energy':{k:float(v[j]) for k,v in baseline_energies.items()},
                                'oxygen_frac':after_exact[j].tolist()}
                            allrows.append(row)
                    local=[r for r in allrows if r['crystal']==name and r['policy']==policy and r['width_A']==width]
                    block=['='*103,
                           f'{name.upper()} | SG {data["sg"]} | GPU {worker_id} | {policy} | sigma={width:.2f} | {len(indices)} starts/mode',
                           '='*103,
                           f'Port initialization: {inittime:.2f}s for {len(metadata)} shared configurations; no SOAP',
                           f'{"Dynamics":22} {"N":>4} {"Valid":>7} {"Repair":>7} {"Damage":>7} {"Ti6":>7} {"O3":>7} {"RMS*":>8} {"BasinMoves":>10}']
                    for mode in args.optimizers.split(','):
                        rr=[r for r in local if r['optimizer']==mode]
                        if not rr:continue
                        block.append(f'{mode:22} {len(rr):4d} {np.mean([r["valid"] for r in rr]):7.2f} '
                          f'{sum(r["repaired"] for r in rr):7d} {sum(r["damaged"] for r in rr):7d} '
                          f'{np.mean([r["Ti6"] for r in rr]):7.2f} {np.mean([r["O3"] for r in rr]):7.2f} '
                          f'{np.mean([r["rms_A"] for r in rr]):8.3f} '
                          f'{sum(r["basin_transitions"] for r in rr):10d}')
                    block.append(f'Batch workload: {len(indices)} candidates/mode, {len(args.optimizers.split(","))} modes; total batch wall={sum(r["batch_wall_seconds"] for r in local[::len(indices)]):.1f}s')
                    block.append('* Reference RMS is diagnostic only. Oxygen remains on exact Wyckoff orbits throughout dynamics.')
                    block.append('='*103)
                    queue.put(('block','\n'.join(block)))
            queue.put(('block',f'{name.upper()} completed | GPU {worker_id} | wall={time.perf_counter()-begin:.2f}s | '
                    f'initializations={len(metadata)} | symmetry-constrained dynamics trials={sum(r["crystal"]==name for r in allrows)}'))
        queue.put(('done',dict(rows=allrows,skipped=skipped)))
    except BaseException:
        import traceback
        queue.put(('error',traceback.format_exc()))


def experiment_main():
    parser=argparse.ArgumentParser(description='Batched symmetry-first Ti-centered LCAO oxygen initializer and Wyckoff-only dynamics')
    parser.add_argument('--reference-cif-dir',required=True)
    parser.add_argument('--initializers',default=','.join(INITS))
    parser.add_argument('--policies',default=','.join(POLICIES))
    parser.add_argument('--optimizers',default=','.join(MODES))
    parser.add_argument('--ti-perturbations',default='0,0.25,0.50')
    parser.add_argument('--widths',default='0.20,0.35')
    parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--port-steps',type=int,default=50,help='Symmetry-first LCAO initialization gradient steps')
    parser.add_argument('--lr',type=float,default=.025)
    parser.add_argument('--noise-A',type=float,default=.08)
    parser.add_argument('--langevin-cycles',type=int,default=3)
    parser.add_argument('--basin-threshold-A',type=float,default=.15)
    parser.add_argument('--hop-A',type=float,default=.15)
    parser.add_argument('--hop-interval',type=int,default=10)
    parser.add_argument('--metropolis-temp',type=float,default=.25)
    parser.add_argument('--channel-temperature',type=float,default=.12)
    parser.add_argument('--channel-threshold',type=float,default=.70)
    parser.add_argument('--geometry-weight',type=float,default=1.0)
    parser.add_argument('--constraint-weight',type=float,default=.05)
    parser.add_argument('--train-mode',choices=('fast','select'),default='fast')
    parser.add_argument('--train-steps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=4907)
    parser.add_argument('--gpus',type=int,default=4)
    parser.add_argument('--cpus',type=int,default=4)
    parser.add_argument('--output',default='data/xn_templates/oxygen_generation_results.json')
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64)
    for field,choices in [('initializers',INITS),('policies',POLICIES),('optimizers',MODES)]:
        selected=getattr(args,field).split(',')
        if len(set(selected))!=len(selected) or set(selected)-set(choices):
            parser.error(f'Invalid {field}; options are {choices}')
    if args.starts<1 or args.steps<1 or args.port_steps<1:parser.error('All starts/steps must be >=1')
    if args.langevin_cycles<1:parser.error('--langevin-cycles must be at least 1')
    widths=[float(x) for x in args.widths.split(',')]
    levels=[float(x) for x in args.ti_perturbations.split(',')]
    if not widths or any(w<=0 for w in widths):parser.error('Smearing widths must be positive')
    if not levels or any(w<0 for w in levels):parser.error('Ti perturbations must be nonnegative')
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    cpus=min(max(1,args.cpus),affinity,int(os.environ.get('SLURM_CPUS_PER_TASK') or affinity))
    gpu_visible=torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.gpus=min(max(0,args.gpus),gpu_visible)
    nworkers=min(4,args.gpus) if args.gpus else 1
    print(f'Resources: GPUs visible={gpu_visible}, requested workers={nworkers}, CPU affinity={affinity}, '
          f'SLURM CPUs={os.environ.get("SLURM_CPUS_PER_TASK","not set")}, CPU target={cpus}',flush=True)
    if nworkers>cpus:print(f'CPU note: {nworkers} GPU processes share {cpus} CPU cores; one BLAS thread each.',flush=True)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1';os.environ['NUMEXPR_NUM_THREADS']='1'
    refs,_=_load_reference_cifs(args.reference_cif_dir)
    if not any(name.lower()=='rutile' for name in refs):
        raise ValueError('Rutile must remain in training (it is only excluded from testing)')
    if len(refs)!=5:raise ValueError(f'Expected five reference CIFs for training; found {len(refs)}')
    print('Training pooled + individual-reference local geometry on five CIFs once (SOAP disabled)...',flush=True)
    pooled,fc,channels=_family_training(refs,args)
    args._lcao_training=learn_ti_centered_lcao(refs)
    all_basis=np.concatenate(list(args._lcao_training.values()),axis=0)
    args._lcao_pooled=[torch.as_tensor(all_basis,dtype=torch.float64)]
    args._lcao_channels=[torch.as_tensor(v,dtype=torch.float64) for v in args._lcao_training.values()]
    print('Learned pooled + five independent Ti-centered LCAO geometry bases; initialization is symmetry-first.',flush=True)
    print('Test crystals:',', '.join(n for n in refs if n.lower()!='rutile'),flush=True)
    print(f'Matrix requested per crystal: {len(INITS)} initializers x {len(POLICIES)} policies x '
          f'{len(MODES)} dynamics x {len(levels)} Ti levels x {len(widths)} smears x '
          f'{args.starts} starts (symmetry-inapplicable Ti cases explicitly skipped).',flush=True)
    if nworkers==1:
        class Local:
            def __init__(self):self.results=[]
            def put(self,item):
                if item[0]=='block':print(item[1],flush=True)
                else:self.results.append(item)
        q=Local();one_worker(0,1,refs,pooled,fc,channels,args,q);messages=q.results
    else:
        ctx=mp.get_context('spawn')
        q=ctx.Queue()
        procs=[ctx.Process(target=one_worker,args=(i,nworkers,refs,pooled,fc,channels,args,q))
               for i in range(nworkers)]
        for p in procs:p.start()
        messages=[];finished=0
        while finished<nworkers:
            try:kind,value=q.get(timeout=10)
            except __import__('queue').Empty:
                crashed=[p for p in procs if p.exitcode is not None and p.exitcode!=0]
                if crashed:raise RuntimeError(f'Workers failed: {[(p.pid,p.exitcode) for p in crashed]}')
                continue
            if kind=='block':print(value,flush=True)
            else:messages.append((kind,value));finished+=1
        for p in procs:p.join()
    errors=[v for k,v in messages if k=='error']
    if errors:raise RuntimeError('\n'.join(errors))
    rows=[r for kind,item in messages if kind=='done' for r in item['rows']]
    skipped=[r for kind,item in messages if kind=='done' for r in item['skipped']]
    summary={}
    init_summary={}
    for initializer in args.initializers.split(','):
        initial=[r for r in rows if r['initializer']==initializer and r['policy']==args.policies.split(',')[0] and r['optimizer']==args.optimizers.split(',')[0]]
        if initial:init_summary[initializer]=dict(n=len(initial),initial_valid_rate=float(np.mean([r['initial_valid'] for r in initial])),
              initial_Ti6=float(np.mean([r['initial_Ti6'] for r in initial])),
              initial_O3=float(np.mean([r['initial_O3'] for r in initial])),
              mean_wyckoff_projection_rms_A=float(np.mean([r['initial_projection_rms_A'] for r in initial])))
    for policy in args.policies.split(','):
        for mode in args.optimizers.split(','):
            sub=[r for r in rows if r['policy']==policy and r['optimizer']==mode]
            if not sub:continue
            failures=[r for r in sub if not r['initial_valid']]
            successes=[r for r in sub if r['initial_valid']]
            summary[f'{policy}/{mode}']=dict(count=len(sub),valid=sum(r['valid'] for r in sub),
                valid_rate=float(np.mean([r['valid'] for r in sub])),
                repair_rate=float(np.mean([r['valid'] for r in failures])) if failures else None,
                preservation_rate=float(np.mean([r['valid'] for r in successes])) if successes else None,
                mean_basin_transitions=float(np.mean([r['basin_transitions'] for r in sub])),
                mean_reference_RMS_A=float(np.mean([r['rms_A'] for r in sub])),
                mean_energy_delta=float(np.mean([r['final_policy_energy']-r['initial_policy_energy'] for r in sub])),
                energy_decreased_but_validity_lost=sum(r['damaged'] and r['final_policy_energy']<r['initial_policy_energy'] for r in sub),
                total_amortized_candidate_seconds=float(sum(r['amortized_optimization_seconds'] for r in sub)))
    result=dict(settings={k:v for k,v in vars(args).items() if not k.startswith('_')},
        channels=[ch['names'] for ch in channels],test_crystals=[n for n in refs if n.lower()!='rutile'],
        skipped_frameworks=skipped,initialization_summary=init_summary,summary=summary,trials=rows,
        scientific_scope={
            'reference_training':'Five CIFs including rutile',
            'target_oxygen':'Never used in generation; only diagnostics',
            'v140':'Genuine rigid TiO6 rotation, balanced Sinkhorn, owner-exclusive min-cost hard assignment; fixed Ti and cell rather than full unrestricted v140 lattice generation',
            'learned_ports':'Family-conditioned empirical TiO6 medoids learned from references; global geometry ranking for channel choice',
            'dynamics':'Every oxygen position is a complete exact Wyckoff orbit throughout optimization; no full-coordinate dynamics',
            'ti_perturbation':'Only permitted Ti Wyckoff DOF at fixed cell; unachievable levels skipped',
            'runtime':'No intermediate files/checkpoints; completed diagnostic blocks printed to SLURM stdout',
            'budget':'All modes receive specified gradient-step budget; alternating Langevin spends steps on thermal exploration and local relaxation',
            'RMS':'Diagnostic, not validity criterion'})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,default=_json_default))
    print('='*103,flush=True)
    print('OVERALL DIAGNOSTICS',flush=True)
    print('Initialization attribution:',json.dumps(init_summary,indent=2),flush=True)
    for key,val in summary.items():
        print(f'{key:<43} N={val["count"]:4d} valid={val["valid_rate"]:.3f} '
              f'repair={val["repair_rate"]} preserve={val["preservation_rate"]} '
              f'basin-transitions={val["mean_basin_transitions"]:.2f}',flush=True)
    print(f'Output: {output} | trials={len(rows)} | skipped Ti perturbations={len(skipped)}',flush=True)


if __name__=='__main__':experiment_main()
