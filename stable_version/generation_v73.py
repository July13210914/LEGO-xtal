#!/usr/bin/env python3
"""JULIETTE v73 — cleaned generalized single-center Wyckoff-polyhedron generator.

Public reference input is exclusively --ref-folder, with automatic composition
inference and multi-CIF learning. The original v64 GPU search retains its own
internal --ref argument, which is mapped from --ref-folder by the router.
GPU worker errors/tracebacks are relayed to the parent console. Unreferenced
historic CLI entry points were removed; active v64 kernels are preserved.

User-selected --center/--block with CIF-inferred composition and multi-CIF BuildingBlock policy.
For TiO6/TiO2 --engine auto dispatches the UNCHANGED original v64 GPU
engine; generic blocks use a separate CPU search sharing the original v64
WyckoffPolyhedron symmetry/affine machinery. Anions are derived only by
periodic sharing of whole-polyhedron vertices, never independently optimized.

Original v64 baseline (preserved):

Includes the previous v63 multi-GPU progressive WP search and TiO6/LCAO
chemistry kernels. The v64 finalist stage retains a complete polyhedron-state
object (Wyckoff u, SG maps, lattice, constrained full-octahedron F, and arm-to-O
sharing policy). Ti/lattice proposals, octahedral fitting, and optional stalled
LCAO topology updates are evaluated against freshly reconstructed oxygen.

The archive retains a full independent WP-state snapshot and exports only
final per-SG CIFs, regenerated from that state. O atoms have no independent
positions to optimize. Failed Ti/O coordination remains diagnostic and never
becomes a hidden acceptance gate. v64 refinement flags are displayed with
--v64-help. --self-test runs a database-free structural smoke test.

Requires the normal JULIETTE Python dependencies for production searches,
including PyTorch, SciPy, PyXtal, pymatgen and spglib.
"""
from __future__ import annotations
import argparse
import collections
import copy
import contextlib
import csv
import hashlib
import itertools
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import shutil
import tempfile
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DTYPE=torch.float64

_TI_CIF_CACHE = None

def _v46_known_missing_symop_warning():
    """Suppress only the specific harmless CIF warning, not other problems."""
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message=r'^Issues encountered while parsing CIF: No _symmetry_equiv_pos_as_xyz type key found\..*')

def log(*items):
    print(*items, flush=True)

def str_seed(seed, *parts):
    message='|'.join(map(str, (seed, *parts))).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(message,digest_size=8).digest(),'little') % (2**31-1)

def parse_sgs(spec):
    if str(spec).lower()=='all': return list(range(1,231))
    values=[]
    for tok in str(spec).split(','):
        tok=tok.strip()
        if not tok: continue
        if '-' in tok:
            lo,hi=map(int,tok.split('-',1)); values.extend(range(lo,hi+1))
        else: values.append(int(tok))
    if not values or any(v<1 or v>230 for v in values):
        raise ValueError('Space groups must be in 1..230, e.g. 136,87 or 1-230')
    return list(dict.fromkeys(values))

def reference_ti_cifs(folder):
    """Read actual CIF asymmetric sites and expand symmetry through pymatgen."""
    try:
        from pymatgen.core import Structure
    except ImportError as e:
        raise RuntimeError('pymatgen is required to read reference CIFs') from e
    paths=sorted(Path(folder).glob('*.cif'))
    if not paths: raise FileNotFoundError(f'No reference CIFs in {folder}/*.cif')
    rows=[]
    for path in paths:
        with warnings.catch_warnings():
            _v46_known_missing_symop_warning()
            s=Structure.from_file(str(path))
        symbols=[x.specie.symbol for x in s]
        if set(symbols)!={'Ti','O'}:
            raise ValueError(f'{path}: expected TiO2 reference, found {set(symbols)}')
        ni=symbols.count('Ti'); no=symbols.count('O')
        if no!=2*ni: raise ValueError(f'{path}: Ti={ni} O={no}; not TiO2')
        frac=np.asarray([site.frac_coords%1 for site in s if site.specie.symbol=='Ti'],float)
        cell=np.asarray(s.lattice.matrix,float)
        rows.append({'path':str(path), 'name':path.stem, 'ti':frac,'cell':cell,
                     'volume_ti':abs(float(np.linalg.det(cell)))/ni, 'n_ti':ni})
        log(f'  {path.name}: Ti={ni} O={no} volume/Ti={rows[-1]["volume_ti"]:.3f} A^3')
    return rows

def shifts_np(radius):
    return np.asarray(list(itertools.product(range(-radius,radius+1),repeat=3)),float)

def periodic_vectors_numpy(frac,cell,k,radius=3):
    """v22r1 pair-centered periodic Ti->Ti neighbors, excluding ONLY true self-zero."""
    f=np.asarray(frac,float)%1.0; cell=np.asarray(cell,float)
    df=f[None,:,:]-f[:,None,:]
    df-=np.rint(df)
    v=(df[:,:,None,:]+shifts_np(radius)[None,None,:,:])@cell
    v=v.reshape(len(f),-1,3)
    d=np.linalg.norm(v,axis=-1)
    # Mask only the atom's OWN zero image; distinct coincident atoms remain collisions.
    zz=np.flatnonzero(np.all(shifts_np(radius)==0,axis=1))
    for i in range(len(f)):
        d[i,i*len(shifts_np(radius))+int(zz[0])]=1e9
    ix=np.argsort(d,axis=-1,kind='stable')[:,:k]
    return np.take_along_axis(v,ix[:,:,None],axis=1),np.take_along_axis(d,ix,axis=1)

def shell_complete_sizes(dist,minimum_gap,max_k,min_k):
    if dist.shape[1]<max_k+1:raise ValueError('Insufficient image box for learning')
    chosen=[]
    for site,d in enumerate(dist):
        gap=d[min_k:max_k+1]-d[min_k-1:max_k]
        ii=np.where(gap>=minimum_gap)[0]
        if not len(ii):
            raise ValueError(f'Ti site {site}: no complete shell between {min_k} and {max_k}; '
                             'increase --shell-max-k')
        chosen.append(min_k+int(ii[0]))
    return np.asarray(chosen,int)

def radial_pair_descriptor(v):
    radial=np.sort(np.linalg.norm(v,axis=-1))
    p=np.linalg.norm(v[:,None,:]-v[None,:,:],axis=-1)
    return radial,np.sort(p[np.triu_indices(len(v),1)])

def reference_angular_geometry(v):
    """Rotation-invariant Ti-centered angular and 3-D shape targets.

    Neighbor order follows ascending Ti--Ti distance. Angle histograms give a
    permutation-independent distribution; the full Gram graph is retained
    separately for correspondence-aware comparison during optimization.
    """
    v=np.asarray(v,float)
    r=np.linalg.norm(v,axis=-1)
    u=v/np.maximum(r[:,None],1.e-12)
    gram=np.clip(u@u.T,-1.,1.)
    pair=np.triu_indices(len(v),1)
    angles=np.sort(np.arccos(np.clip(gram[pair],-1.,1.)))
    eig=np.linalg.eigvalsh(u.T@u/len(u))
    return gram,angles,eig

def gram_permutation_match(gram, ref_gram, radius, ref_radius, max_swaps=8,
                           cycle_passes=3, ambiguous_angle_deg=5.0):
    """One-to-one, permutation-invariant Ti neighbor Gram alignment.

    Fast, batched GPU path for generation: signature-informed 2-opt matching,
    followed by exact evaluation of all 3-cycles *only* for near-matching
    angular histograms with an inconsistent correspondence. This fixes many
    symmetry-induced 2-opt local minima with modest optimization overhead.

    The resulting matched Gram tensor is DIFFERENTIABLE with respect to the
    actual Ti coordinates (the discrete permutation has no gradient).

    For reference validation, an independent bounded graph-isomorphism check
    below *proves* zero-cost correspondence when appropriate. Global optimality
    is not claimed for a strongly distorted generated environment.
    """
    shape=gram.shape[:-2]
    k=gram.shape[-1]
    gg=gram.detach().reshape(-1,k,k)
    hh=ref_gram.detach().reshape(-1,k,k)
    rr=radius.detach().reshape(-1,k)
    ss=ref_radius.detach().reshape(-1,k)
    q=len(gg)
    device=gg.device
    if k<2:
        perm=torch.zeros((q,k),dtype=torch.long,device=device)
        return gram,perm.reshape(*shape,k)
    with torch.no_grad():
        # A vertex fingerprint is invariant to the OTHER neighbors' labels.
        # Distances and sorted angular rows narrow the assignment ambiguities.
        row_g=torch.sort(gg,dim=-1).values
        row_h=torch.sort(hh,dim=-1).values
        weights=torch.linspace(.7,1.3,k,dtype=gg.dtype,device=device)
        sig_g=(row_g*weights).mean(-1)
        sig_h=(row_h*weights).mean(-1)
        cand_order=torch.argsort(rr+.10*sig_g,dim=-1,stable=True)
        ref_order=torch.argsort(ss+.10*sig_h,dim=-1,stable=True)
        perm=torch.empty((q,k),dtype=torch.long,device=device)
        perm.scatter_(1,ref_order,cand_order)

        def swap_refine(perm0,sub_gg,sub_hh,sub_rr,sub_ss):
            nq=len(perm0)
            if k<3 or max_swaps==0 or nq==0:return perm0
            ia,ib=torch.triu_indices(k,k,offset=1,device=device)
            mask=torch.ones((len(ia),k),dtype=gg.dtype,device=device)
            mask.scatter_(1,ia[:,None],0.)
            mask.scatter_(1,ib[:,None],0.)
            row=torch.arange(nq,device=device)
            for _ in range(max_swaps):
                gp=sub_gg.gather(1,perm0[:,:,None].expand(nq,k,k))
                gp=gp.gather(2,perm0[:,None,:].expand(nq,k,k))
                rp=sub_rr.gather(1,perm0)
                edge_delta=(gp[:,ia,:]-gp[:,ib,:])*(sub_hh[:,ib,:]-sub_hh[:,ia,:])
                delta_corr=2.*(edge_delta*mask[None,:,:]).sum(-1)
                delta_gram=-2.*delta_corr/(k*(k-1))
                delta_rad=2.*(rp[:,ia]-rp[:,ib])*(sub_ss[:,ia]-sub_ss[:,ib])
                delta_cost=delta_gram+.35*delta_rad/(k*3.0**2)
                dmin,which=delta_cost.min(-1)
                active=dmin < -1.e-12
                if not bool(active.any()):break
                a=ia[which[active]];b=ib[which[active]]
                i=row[active]
                old_a=perm0[i,a].clone();old_b=perm0[i,b].clone()
                perm0[i,a]=old_b;perm0[i,b]=old_a
            return perm0

        perm=swap_refine(perm,gg,hh,rr,ss)
        if cycle_passes>0 and k>=3:
            idx=torch.arange(k,device=device)
            current=gg.gather(1,perm[:,:,None].expand(q,k,k))
            current=current.gather(2,perm[:,None,:].expand(q,k,k))
            off=torch.triu_indices(k,k,offset=1,device=device)
            gram_err=((current[:,off[0],off[1]]-
                       hh[:,off[0],off[1]])**2).mean(-1)
            ga=torch.sort(gg[:,off[0],off[1]],dim=-1).values
            ha=torch.sort(hh[:,off[0],off[1]],dim=-1).values
            # Low unordered-angle error + high matched-graph error is the
            # signature of a correspondence failure, not geometric mismatch.
            angle_hist=torch.sqrt(((torch.acos(ga.clamp(-1.+1e-7,1.-1e-7))-
                                    torch.acos(ha.clamp(-1.+1e-7,1.-1e-7)))**2).mean(-1))
            need=(gram_err>.0025**2)&(angle_hist < math.radians(ambiguous_angle_deg))
            affected=torch.nonzero(need,as_tuple=True)[0]
            if len(affected):
                gg2=gg[affected];hh2=hh[affected];rr2=rr[affected];ss2=ss[affected]
                pp=perm[affected].clone()
                # Every oriented 3-cycle is an explicit one-to-one assignment.
                # Evaluate their REAL graph+radial cost, not a heuristic score.
                triples=list(itertools.combinations(range(k),3))
                moves=[]
                for a,b,c in triples:
                    for turn in (0,1):
                        mapping=list(range(k))
                        mapping[a],mapping[b],mapping[c]=((b,c,a) if turn==0 else (c,a,b))
                        moves.append(mapping)
                mapping=torch.tensor(moves,dtype=torch.long,device=device)
                # Chunk to keep GPU memory modest for many trapped environments.
                for batch_start in range(0,len(pp),96):
                    sl=slice(batch_start,min(batch_start+96,len(pp)))
                    p=pp[sl].clone();g=gg2[sl];h=hh2[sl];r=rr2[sl];s=ss2[sl]
                    nq=len(p)
                    for _ in range(cycle_passes):
                        mapped=p[:,mapping]  # [batch, moves, k]
                        # The graph is indexed via the ORIGINAL (unpermuted)
                        # coordinates; explicit row+column gather preserves
                        # the full coherent 3-D angular correspondence.
                        expanded=g[:,None,:,:].expand(nq,len(mapping),k,k)
                        vg=expanded.gather(2,mapped[:,:,:,None].expand(nq,len(mapping),k,k))
                        vg=vg.gather(3,mapped[:,:,None,:].expand(nq,len(mapping),k,k))
                        er=((vg-h[:,None,:,:])**2).sum((-1,-2))/(k*(k-1))
                        vr=r[:,None,:].expand(nq,len(mapping),k).gather(2,mapped)
                        er=er+.35*((vr-s[:,None,:])**2).sum(-1)/(k*3.**2)
                        gp=g.gather(1,p[:,:,None].expand(nq,k,k))
                        gp=gp.gather(2,p[:,None,:].expand(nq,k,k))
                        old=((gp-h)**2).sum((-1,-2))/(k*(k-1))
                        old=old+.35*((r.gather(1,p)-s)**2).sum(-1)/(k*3.**2)
                        cost,j=er.min(-1)
                        better=cost<old-1e-12
                        if not bool(better.any()):break
                        rows=torch.nonzero(better,as_tuple=True)[0]
                        p[rows]=mapped[rows,j[rows]]
                        p=swap_refine(p,g,h,r,s)
                    pp[sl]=p
                perm[affected]=pp

            # Rare hard case: a perfect/near-perfect symmetric shell can sit
            # behind a four-vertex permutation barrier. This rescue is
            # restricted to virtually identical unordered angles and radii;
            # it never performs 4-cycle enumeration for generic bad Ti sites.
            aligned_g=gg.gather(1,perm[:,:,None].expand(q,k,k))
            aligned_g=aligned_g.gather(2,perm[:,None,:].expand(q,k,k))
            err=((aligned_g[:,off[0],off[1]]-hh[:,off[0],off[1]])**2).mean(-1)
            same_r=torch.max(torch.abs(torch.sort(rr,dim=-1).values-
                                       torch.sort(ss,dim=-1).values),dim=-1).values
            unresolved=(err>1e-8)&(angle_hist<math.radians(.4))&(same_r<.002)
            affected=torch.nonzero(unresolved,as_tuple=True)[0]
            if k>=4 and len(affected):
                # All directed 4-cycles. No 2-opt or 3-cycle neighborhood
                # can necessarily escape their local minima in symmetric graphs.
                four_moves=[]
                for combo in itertools.combinations(range(k),4):
                    a,b,c,d=combo
                    for q3 in itertools.permutations((b,c,d)):
                        cyc=(a,)+q3
                        mapping=list(range(k))
                        for x,y in zip(cyc,cyc[1:]+cyc[:1]):mapping[x]=y
                        four_moves.append(mapping)
                mapping=torch.tensor(four_moves,dtype=torch.long,device=device)
                for start in range(0,len(affected),24):
                    ids=affected[start:start+24]
                    g=gg[ids];h=hh[ids];r=rr[ids];s=ss[ids]
                    p=perm[ids].clone();nq=len(p)
                    for _ in range(2):
                        mapped=p[:,mapping]
                        vg=g[:,None,:,:].expand(nq,len(mapping),k,k)
                        vg=vg.gather(2,mapped[:,:,:,None].expand(nq,len(mapping),k,k))
                        vg=vg.gather(3,mapped[:,:,None,:].expand(nq,len(mapping),k,k))
                        cost=((vg-h[:,None,:,:])**2).sum((-1,-2))/(k*(k-1))
                        vr=r[:,None,:].expand(nq,len(mapping),k).gather(2,mapped)
                        cost=cost+.35*((vr-s[:,None,:])**2).sum(-1)/(k*3.**2)
                        gp=g.gather(1,p[:,:,None].expand(nq,k,k))
                        gp=gp.gather(2,p[:,None,:].expand(nq,k,k))
                        old=((gp-h)**2).sum((-1,-2))/(k*(k-1))
                        old=old+.35*((r.gather(1,p)-s)**2).sum(-1)/(k*3.**2)
                        better_cost,best_idx=cost.min(-1)
                        improved=better_cost<old-1e-12
                        if not bool(improved.any()):break
                        row_ids=torch.nonzero(improved,as_tuple=True)[0]
                        p[row_ids]=mapped[row_ids,best_idx[row_ids]]
                        p=swap_refine(p,g,h,r,s)
                    perm[ids]=p

            # Certified-identity rescue. In nearly identical environments,
            # a symmetry-related ordering can require an arbitrarily long
            # permutation cycle. Instead of pretending finite swap depth is
            # exact, solve the small reference graph-isomorphism problem.
            # The strict trigger prevents CPU matching for generic generated
            # structures during gradient optimization.
            aligned_g=gg.gather(1,perm[:,:,None].expand(q,k,k))
            aligned_g=aligned_g.gather(2,perm[:,None,:].expand(q,k,k))
            err=((aligned_g[:,off[0],off[1]]-hh[:,off[0],off[1]])**2).mean(-1)
            unresolved=(err>1e-8)&(angle_hist<math.radians(.12))&(same_r<.001)
            retry_ids=torch.nonzero(unresolved,as_tuple=True)[0]
            if len(retry_ids):
                g_cpu=gg[retry_ids].cpu().numpy()
                h_cpu=hh[retry_ids].cpu().numpy()
                r_cpu=rr[retry_ids].cpu().numpy()
                s_cpu=ss[retry_ids].cpu().numpy()
                for a in range(len(retry_ids)):
                    exact=exact_reference_gram_match(g_cpu[a],h_cpu[a],r_cpu[a],s_cpu[a],
                                                     radial_tol=.002,angular_cos_tol=.003)
                    if exact is not None:
                        ridx=int(retry_ids[a]);newperm=torch.tensor(exact,device=device,dtype=torch.long)
                        # Use certified match only if it improves the full
                        # physically motivated Gram+radius assignment score.
                        old_perm=perm[ridx]
                        def _assignment_cost(p):
                            g2=gg[ridx].index_select(0,p).index_select(1,p)
                            return (((g2-hh[ridx])**2).sum()/(k*(k-1))+
                                    .35*((rr[ridx,p]-ss[ridx])**2).sum()/(k*3.**2))
                        if float(_assignment_cost(newperm))<float(_assignment_cost(old_perm))-1e-12:
                            perm[ridx]=newperm

    # Keep gradient through the actual candidate Gram elements.
    original=gram.reshape(-1,k,k)
    aligned=original.gather(1,perm[:,:,None].expand(q,k,k))
    aligned=aligned.gather(2,perm[:,None,:].expand(q,k,k))
    return aligned.reshape(*shape,k,k),perm.reshape(*shape,k)

def exact_reference_gram_match(gram,ref_gram,radius,ref_radius,
                               radial_tol=2e-4,angular_cos_tol=2e-4,
                               max_nodes=200000):
    """Certified *zero-correspondence* match for validation/learning.

    Backtracking is exact within its explicitly stated radial and cosine
    tolerances. Unlike generic quadratic assignment, near-equal radial shell
    groups do not force an arbitrary ordering. Nonexistence is certified only
    if the node budget is not exhausted (else RuntimeError).

    Works on k<=10 reference shells; NEVER called for every generated site
    inside the GPU optimization loop.
    """
    g=np.asarray(gram,dtype=float);h=np.asarray(ref_gram,dtype=float)
    r=np.asarray(radius,dtype=float);s=np.asarray(ref_radius,dtype=float)
    k=len(r)
    if g.shape!=(k,k) or h.shape!=(k,k) or s.shape!=(k,):
        raise ValueError('Invalid graph/radius shape in exact Gram matcher')
    possible=[]
    for j in range(k):
        cand=[i for i in range(k) if abs(float(r[i]-s[j]))<=radial_tol]
        if not cand:return None
        # Strong, permutation-invariant per-vertex degree-signature filter:
        # sorted cosines of other neighbors and their radial distances.
        mask=np.arange(k)!=j
        close=[]
        for i in cand:
            mask_i=np.arange(k)!=i
            aa=np.sort(g[i,mask_i]);bb=np.sort(h[j,mask])
            if np.max(abs(aa-bb))<=angular_cos_tol:
                close.append(i)
        if not close:return None
        possible.append(close)
    order=sorted(range(k),key=lambda j:(len(possible[j]),-len(set(np.round(h[j],3)))))
    perm=[-1]*k;used=[False]*k;nodes=0
    def rec(depth):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise RuntimeError(f'Exact Gram graph matcher exhausted node limit {max_nodes}')
        if depth==k:return True
        j=order[depth]
        for i in possible[j]:
            if used[i]:continue
            if any(abs(g[i,perm[order[t]]]-h[j,order[t]])>angular_cos_tol for t in range(depth)):
                continue
            perm[j]=i;used[i]=True
            if rec(depth+1):return True
            used[i]=False;perm[j]=-1
        return False
    if rec(0):return np.asarray(perm,dtype=int)
    return None

def soap_torch(v,radial_centers,width,lmax=4):
    """SO(3)-invariant radial density power spectrum, with occupancy retained."""
    d=torch.linalg.vector_norm(v,dim=-1).clamp_min(1e-8)
    u=v/d[...,None]
    dot=torch.einsum('...kd,...ld->...kl',u,u).clamp(-1.,1.)
    radial=torch.exp(-.5*((d[...,None]-radial_centers)/width)**2)
    K=v.shape[-2]
    occ=radial.mean(dim=-2)
    powers=[]
    p0=torch.ones_like(dot); p1=dot
    for ell in range(lmax+1):
        if ell==0: p=p0
        elif ell==1: p=p1
        else:
            p=((2*ell-1)*dot*p1-(ell-1)*p0)/ell
            p0,p1=p1,p
        mat=torch.einsum('...kr,...ls,...kl->...rs',radial,radial,p)/(K*K)
        ir=torch.triu_indices(mat.shape[-1],mat.shape[-1],device=mat.device)
        powers.append(mat[...,ir[0],ir[1]])
    return torch.cat([occ]+powers,dim=-1)

def learn_ti_reference(refs,min_k=8,max_k=30,gap=.015,merge=.040,
                       soap_neighbors=24,soap_radials=5,soap_lmax=4):
    """Direct from ref CIFs.  No oxygen descriptors or external serialized learner."""
    modes=[]; radials=[]; volumes=[]; all_clouds=[]; all_neighbor_d=[]; min_contact=[]
    required_radius=1; kmax_all=0
    per_reference=[]
    for row in refs:
        f=row['ti']; C=row['cell']
        V,D=periodic_vectors_numpy(f,C,max(max_k+1,soap_neighbors),radius=3)
        counts=shell_complete_sizes(D[:,:max_k+1],gap,max_k,min_k)
        all_neighbor_d.extend(D[:,:max(24,soap_neighbors)].tolist())
        min_contact.append(float(D[:,0].min()))
        volumes.append(float(row['volume_ti']))
        kmax_all=max(kmax_all,int(counts.max()))
        for i,k in enumerate(counts):
            d,p=radial_pair_descriptor(V[i,:k])
            g,theta,eig=reference_angular_geometry(V[i,:k])
            # Same-species part of v22r1 descriptor: sorted radii and all pair distances.
            match=None
            for j,m in enumerate(modes):
                if m['k']!=k:continue
                rms=math.sqrt((np.mean((d-m['r'])**2)+np.mean((p-m['p'])**2)))
                if rms>merge:continue
                # Do not merge local modes on radial/pair histograms alone:
                # different assignments of the same distances can encode
                # genuinely different Ti--Ti--Ti angular graphs.
                angle_difference=float(np.sqrt(np.mean((theta-m['angles_rad'])**2)))*180/math.pi
                if angle_difference>2.0:continue
                candidate=torch.tensor(g[None],dtype=DTYPE)
                refg=torch.tensor(m['G'][None],dtype=DTYPE)
                matched,_=gram_permutation_match(candidate,refg,
                   torch.tensor(d[None],dtype=DTYPE),torch.tensor(m['r'][None],dtype=DTYPE),
                   max_swaps=12)
                graph_discrepancy=float(torch.sqrt(((matched-refg)**2).mean()))
                if graph_discrepancy>.08:continue
                match=j;break
            if match is not None:
                modes[match]['multiplicity']+=1
            else:
                modes.append({'k':int(k),'r':d.copy(),'p':p.copy(),'G':g.copy(),
                              'angles_rad':theta.copy(),'eig':eig.copy(),'source':row['name'],
                              'reference_ti_index':int(i),'multiplicity':1})
            radials.append(D[i])
            all_clouds.append(V[i,:soap_neighbors].copy())
        per_reference.append((row,counts,V,D))
        # Determine the minimal centered image radius that reproduces each site's learned shell.
        for rad in (1,2,3):
            _,Dsmall=periodic_vectors_numpy(f,C,int(counts.max()),radius=rad)
            if np.max(np.abs(Dsmall[:,:int(counts.max())]-D[:,:int(counts.max())]))<1e-7:
                required_radius=max(required_radius,rad)
                break
        else:required_radius=3
        log(f'  Ti-only complete shells in {row["name"]}: '
            +str(dict(zip(*[x.tolist() for x in np.unique(counts,return_counts=True)]))))
    # Smooth reference N(R), calibrated directly from reference titanium positions.
    neighbor_dist=np.asarray(all_neighbor_d)
    available=neighbor_dist.shape[1]
    ranks=[k for k in (6,12,16,20,24) if k<=available]
    density_profile=[(float(np.median(neighbor_dist[:,k-1])),int(k)) for k in ranks]
    v_median=float(np.median(volumes)); v_min=float(min(volumes)); v_max=float(max(volumes))
    hard=max(.6,float(min(min_contact))-.12)
    # Smooth SOAP reference distribution on the identical complete set of reference Ti lattices.
    clouds=np.asarray(all_clouds)
    allr=np.linalg.norm(clouds,axis=-1).reshape(-1)
    centers=np.linspace(float(np.quantile(allr,.03)),float(np.quantile(allr,.97)),soap_radials)
    width=max(.20,float((centers[-1]-centers[0])/max(len(centers)-1,1))*.72)
    with torch.no_grad():
        rcent=torch.tensor(centers,dtype=DTYPE)
        features=soap_torch(torch.tensor(clouds,dtype=DTYPE),rcent,width,soap_lmax).numpy()
    med=np.median(features,axis=0)
    scale=np.maximum(1.4826*np.median(abs(features-med),axis=0),.015)
    # Train an optional, invariant SOAP feature compressor. This is NOT PCA on
    # Ti neighbor coordinates: complete 8/9/10 shells and Gram matching remain.
    standardized=(features-med)/scale
    pca_center=standardized.mean(axis=0)
    centered=standardized-pca_center
    _,singular,vt=np.linalg.svd(centered,full_matrices=False)
    # One sample-mean direction is null: exclude numerical zero modes.
    energy=singular**2
    nonzero=energy>max(float(energy.max()) if len(energy) else 0.,1e-30)*1e-12
    basis=np.asarray(vt[nonzero].T,float)
    powers=energy[nonzero]
    ratios=powers/max(float(powers.sum()),1e-30)
    # Preserve a one-sided anti-collapse bound from the original reference 3-D neighbor clouds.
    lam3=[]
    for V in clouds:
        vv=V[:min(len(V),12)];norm=np.linalg.norm(vv,axis=-1,keepdims=True)
        u=vv/np.maximum(norm,1e-12)
        lam3.append(float(np.linalg.eigvalsh(u.T@u/len(u))[0]))
    lambda_floor=max(0.,float(np.quantile(lam3,.05))*.80)
    library={'modes':modes,'density_profile':density_profile,'volume_median':v_median,
             'volume_min':v_min,'volume_max':v_max,'hard_titi':hard,
             'soap_centers':centers,'soap_width':width,'soap_med':med,'soap_scale':scale,
             'soap_neighbors':int(soap_neighbors),'soap_lmax':int(soap_lmax),
             'lambda3_floor':lambda_floor,'image_radius':required_radius,
             'shell_gap':gap,'shell_min_k':min_k,'shell_max_k':max_k,'max_mode_k':kmax_all,
             'reference_count':len(refs),'reference_ti_site_count':len(radials),
             'soap_pca_mean':pca_center,'soap_pca_basis':basis,
             'soap_pca_variance_ratio':ratios}
    log(f'LEARNED: Ti-only modes={len(modes)}; shell k={sorted(set(m["k"] for m in modes))}; '
        f'density ranks={ranks}; min Ti-Ti={min(min_contact):.3f} A; hard={hard:.3f} A; '
        f'volume/Ti median={v_median:.3f} A^3; validated images=+/-{required_radius}')
    return library

_TI_LEGACY_RULES = {
    'repulsion_width_A': .10,
    'repulsion_normalizer_A': .15,
    'density_width_A': .18,
    'volume_log_width': .35,
    'span_neighbors': 12,
}

_TI_EMPIRICAL_DEFAULTS = {
    'contact_margin_A': 0.12,
    'screen_margin_A': 0.26,
    'density_width_A': 0.18,
    'repulsion_width_A': 0.10,
    'repulsion_normalizer_A': 0.15,
    'volume_log_width_floor': 0.35,
    'span_neighbors': 12,
    'density_ranks': (6, 12, 16, 20, 24),
}

def ti_empirical_settings(args=None):
    result=dict(_TI_EMPIRICAL_DEFAULTS)
    if args is not None:
        for key in result:
            attr='ti_empirical_'+key
            if hasattr(args,attr):result[key]=getattr(args,attr)
    raw=result['density_ranks']
    if isinstance(raw,str):
        try: ranks=tuple(int(x.strip()) for x in raw.split(',') if x.strip())
        except ValueError as exc:raise ValueError('Invalid --ti-empirical-density-ranks') from exc
    else:ranks=tuple(raw)
    if not ranks or sorted(set(ranks))!=list(ranks) or any(k<1 or k>64 for k in ranks):
        raise ValueError('--ti-empirical-density-ranks must be unique ascending positive ranks <=64')
    result['density_ranks']=ranks
    positive=('density_width_A','repulsion_width_A','repulsion_normalizer_A','volume_log_width_floor')
    if any(not np.isfinite(float(result[k])) or float(result[k])<=0 for k in positive):
        raise ValueError('Empirical smoothing widths and volume floor must be positive')
    if any(not np.isfinite(float(result[k])) or float(result[k])<0 for k in ('contact_margin_A','screen_margin_A')):
        raise ValueError('Empirical contact and screening margins must be nonnegative')
    if int(result['span_neighbors'])<3 or int(result['span_neighbors'])>64:
        raise ValueError('Empirical span neighbors must be in 3..64')
    return result

def recalibrate_ti_span_floor(refs,k):
    eigmins=[]
    for ref in refs:
        v,_=periodic_vectors_numpy(ref['ti'],ref['cell'],k,radius=3)
        u=v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
        eigmins.extend(np.linalg.eigvalsh(np.einsum('nki,nkj->nij',u,u)/k)[:,0].tolist())
    return max(0.,float(np.quantile(eigmins,.05))*.80)

def hybrid_ti_rules(refs,library,learned,empirical):
    # Reference observations determine targets; human-specified margins determine
    # recoverable search range. Do not estimate optimization noise from 22 Ti sites.
    min_contact=float(min(periodic_vectors_numpy(r['ti'],r['cell'],1,radius=3)[1][:,0].min()
                          for r in refs))
    hard=max(.6,min_contact-float(empirical['contact_margin_A']))
    screen=max(.6,min_contact-float(empirical['screen_margin_A']))
    ranks=empirical['density_ranks']
    max_rank=max(ranks)
    d=np.concatenate([periodic_vectors_numpy(r['ti'],r['cell'],max_rank,radius=3)[1]
                      for r in refs],axis=0)
    density=[(float(np.median(d[:,k-1])),int(k)) for k in ranks]
    sk=int(empirical['span_neighbors'])
    # Legacy k=12 is exactly the same library training calculation.
    floor=float(library['lambda3_floor']) if sk==12 else recalibrate_ti_span_floor(refs,sk)
    return {
        'lambda3_floor':floor,
        'hard_titi_A':hard,
        'screen_titi_A':screen,
        'density_profile':density,
        'density_width_A':float(empirical['density_width_A']),
        'repulsion_width_A':float(empirical['repulsion_width_A']),
        'repulsion_normalizer_A':float(empirical['repulsion_normalizer_A']),
        'volume_log_width':max(float(learned['volume_log_width']),float(empirical['volume_log_width_floor'])),
        'span_neighbors':sk,
    }

def add_ti_empirical_cli(ap):
    # Shared by the standalone Ti-stage CLI and outer Juliette controller.
    ap.add_argument('--ti-empirical-contact-margin-A',dest='ti_empirical_contact_margin_A',type=float,default=.12,
                    help='Exploration offset below smallest reference Ti--Ti separation (A)')
    ap.add_argument('--ti-empirical-screen-margin-A',dest='ti_empirical_screen_margin_A',type=float,default=.26,
                    help='Outer Ti-screen offset below smallest reference Ti--Ti separation (A)')
    ap.add_argument('--ti-empirical-density-width-A',dest='ti_empirical_density_width_A',type=float,default=.18)
    ap.add_argument('--ti-empirical-repulsion-width-A',dest='ti_empirical_repulsion_width_A',type=float,default=.10)
    ap.add_argument('--ti-empirical-repulsion-normalizer-A',dest='ti_empirical_repulsion_normalizer_A',type=float,default=.15)
    ap.add_argument('--ti-empirical-volume-log-width-floor',dest='ti_empirical_volume_log_width_floor',type=float,default=.35)
    ap.add_argument('--ti-empirical-span-neighbors',dest='ti_empirical_span_neighbors',type=int,default=12)
    ap.add_argument('--ti-empirical-density-ranks',dest='ti_empirical_density_ranks',default='6,12,16,20,24')
    ap.add_argument('--ti-soap-pca',choices=('off','audit','on'),default='audit',
                    help='audit: train PCA and report variance but preserve original SOAP objective; on: experimental compressed SOAP loss')
    ap.add_argument('--ti-soap-pca-variance',type=float,default=.995,
                    help='Reference variance fraction used to select PCA dimension (0..1)')
    ap.add_argument('--ti-soap-pca-max-components',type=int,default=0,
                    help='Optional upper bound on retained PCA components; 0 means no artificial cap')

def ti_pca_settings(args):
    if not (0.0<float(args.ti_soap_pca_variance)<=1.0):
        raise ValueError('--ti-soap-pca-variance must lie in (0,1]')
    if int(args.ti_soap_pca_max_components)<0:
        raise ValueError('--ti-soap-pca-max-components must be nonnegative')
    return args.ti_soap_pca,float(args.ti_soap_pca_variance),int(args.ti_soap_pca_max_components)

def report_ti_soap_pca(lib,args):
    mode,frac,cap=ti_pca_settings(args)
    ratios=np.asarray(lib['soap_pca_variance_ratio'],float)
    rank=int(len(ratios))
    cumulative=np.cumsum(ratios)
    chosen=min(rank,max(1,int(np.searchsorted(cumulative,frac)+1))) if rank else 0
    if cap:chosen=min(chosen,cap)
    kept=float(cumulative[chosen-1]) if chosen else 0.
    n=int(lib['reference_ti_site_count'])
    original=int(len(lib['soap_med']))
    print(f' TI SOAP PCA: mode={mode} ref_sites={n} original_dim={original} '
          f'nonzero_rank={rank} selected_dim={chosen} variance={kept:.6f} '
          f'residual_fraction={max(0.,1.-kept):.6f}',flush=True)
    if mode=='on':
        if chosen<=0:raise ValueError('PCA compression needs nondegenerate training reference data')
        if n<max(10,2*chosen):
            print(' TI SOAP PCA WARNING: reference sample is small relative to retained PCA dimension; '
                  'compression is experimental, not a calibrated prediction of new polymorphs.',flush=True)
        if kept<frac-1e-8:
            print(' TI SOAP PCA WARNING: component cap reduced retained variance below requested threshold.',flush=True)
    return chosen

def infer_ti_rules(refs, library, empirical=None):
    """Estimate physically meaningful Ti-stage scales from reference Ti sites.

    These are candidate rules, NOT forced replacements.  Reference positions
    are used for learning only.  We never read generated or reference oxygen.
    Geometry constraints (periodicity, symmetry, distinct images, Gram/modes)
    stay fixed; this routine learns the values used by those constraints.
    """
    largest=max(25,int(library['soap_neighbors'])+1)
    dlist=[]
    for ref in refs:
        _,di=periodic_vectors_numpy(ref['ti'],ref['cell'],largest,radius=3)
        dlist.extend(np.asarray(di,float))
    distances=np.asarray(dlist,float)
    smallest=distances[:,0]
    median_nn=float(np.median(smallest))
    nn_spread=float(1.4826*np.median(np.abs(smallest-median_nn)))
    # Avoid using a fixed Ti radius.  Lower tail minus observed dispersion is
    # an empirical conservative exclusion bound, never larger than the 5% tail.
    contact_low=float(np.quantile(smallest,.05))
    contact_rule=max(0.05*median_nn,contact_low-nn_spread)
    # Find complete-shell ranks supported by actual gaps in the reference Ti
    # distribution.  Equal-distance image shells are retained intact.
    rank_max=min(24,distances.shape[1]-1)
    gaps=distances[:,1:rank_max+1]-distances[:,:rank_max]
    relevant=gaps[:,1:]  # rank >= 2 (avoid single-image degeneracy)
    signal=float(np.quantile(relevant[relevant>1e-9],.70)) if np.any(relevant>1e-9) else 0.
    gap_floor=max(float(library['shell_gap']),.025*signal)
    support=np.mean(gaps[:,1:]>=gap_floor,axis=0)
    # A rank should correspond to a complete shell in at least some references.
    # Bin the range and select the most supported closure per band, no fixed ranks.
    selected=[]
    ranges=np.array_split(np.arange(2,rank_max+1,dtype=int),5)
    for band in ranges:
        if len(band)==0:continue
        valid=[int(k) for k in band if support[k-2]>0]
        if valid:
            selected.append(max(valid,key=lambda k:(support[k-2],k)))
        else:
            selected.append(int(band[-1]))
    ranks=tuple(sorted(set(selected)))
    density_profile=[(float(np.median(distances[:,k-1])),int(k)) for k in ranks]
    # Empirical bandwidths reflect variation across genuine reference centers.
    sd_by_rank=[]
    for radius,k in density_profile:
        rad=distances[:,k-1]
        sd_by_rank.append(1.4826*np.median(np.abs(rad-np.median(rad))))
    density_width=max(.02*median_nn,float(np.median(sd_by_rank)))
    repulsion_width=max(.015*median_nn,min(nn_spread,.10*median_nn))
    repulsion_scale=max(.02*median_nn,nn_spread)
    vols=np.asarray([float(r['volume_ti']) for r in refs],float)
    lnvol=np.log(vols)
    volume_width=max(.025,1.4826*float(np.median(np.abs(lnvol-np.median(lnvol)))))
    counts=np.asarray([int(m['k']) for m in library['modes']],int)
    span_neighbors=max(3,int(np.median(counts)))
    # Calibrate the matching 3-D span lower bound with the SAME number of
    # neighbors used by the learned planarity descriptor.
    eigmins=[]
    for ref in refs:
        v,_=periodic_vectors_numpy(ref['ti'],ref['cell'],span_neighbors,radius=3)
        norms=np.linalg.norm(v,axis=-1,keepdims=True)
        unit=v/np.maximum(norms,1e-12)
        for cloud in unit:
            eigmins.append(float(np.linalg.eigvalsh(cloud.T@cloud/len(cloud))[0]))
    learned_span_floor=max(0.,float(np.quantile(eigmins,.05))*.80)
    learned={
        'lambda3_floor':learned_span_floor,
        'hard_titi_A':float(contact_rule),
        'screen_titi_A':float(contact_rule),
        'density_profile':density_profile,
        'density_width_A':float(density_width),
        'repulsion_width_A':float(repulsion_width),
        'repulsion_normalizer_A':float(repulsion_scale),
        'volume_log_width':float(volume_width),
        'span_neighbors':int(span_neighbors),
    }
    legacy={
        **_TI_LEGACY_RULES,
        'lambda3_floor':float(library['lambda3_floor']),
        'hard_titi_A':float(library['hard_titi']),
        'screen_titi_A':2.70,
        'density_profile':list(library['density_profile']),
    }
    empirical=ti_empirical_settings(empirical)
    hybrid=hybrid_ti_rules(refs,library,learned,empirical)
    evidence={'reference_sites':len(distances),'reference_structures':len(refs),
              'nearest_TiTi_median_A':median_nn,
              'nearest_TiTi_robust_spread_A':nn_spread,
              'nearest_TiTi_5pct_A':contact_low,
              'shell_gap_reference_support':[(int(k),round(float(support[k-2]),3)) for k in ranks]}
    return {'legacy':legacy,'learned':learned,'hybrid':hybrid,'empirical':empirical,'evidence':evidence}

def apply_ti_rule_mode(library,model,mode):
    """Return a view of the library, with learned Ti parameters if requested."""
    if mode not in ('legacy','compare','learned','hybrid'):
        raise ValueError('Ti rule mode must be legacy, compare, hybrid or learned')
    updated=dict(library)
    values=model['learned' if mode=='learned' else ('hybrid' if mode=='hybrid' else 'legacy')]
    updated['lambda3_floor']=float(values['lambda3_floor'])
    updated['hard_titi']=float(values['hard_titi_A'])
    updated['density_profile']=[tuple(z) for z in values['density_profile']]
    updated['_ti_rule_values']=values
    updated['_ti_rule_mode']=mode
    return updated

@dataclass
class WyckoffTemplate:
    spg:int
    token:tuple
    n_ti:int
    lattice_type:str
    lattice_keys:tuple
    blocks:list
    ncoord:int

def lattice_parameters(ltype):
    ltype=str(ltype).lower()
    if ltype=='cubic':return ('a',)
    if ltype in ('tetragonal','hexagonal','trigonal'):return ('a','c')
    if ltype=='orthorhombic':return ('a','b','c')
    if ltype=='monoclinic':return ('a','b','c','beta')
    return ('a','b','c','alpha','beta','gamma')

def affine_map(wp,dof):
    if dof==0:
        return np.zeros((3,0)),np.asarray(wp.get_position_from_free_xyzs(np.zeros(0)),float)%1.0
    zero=np.asarray(wp.get_position_from_free_xyzs(np.zeros(dof)),float)%1.0
    A=np.zeros((3,dof))
    for k in range(dof):
        x=np.zeros(dof);x[k]=.137
        y=np.asarray(wp.get_position_from_free_xyzs(x),float)%1.0
        A[:,k]=((y-zero+.5)%1.-.5)/.137
    # Verify that the old v140 affine construction actually reconstructs PyXtal WP mapping.
    for z in (.17,.41,.79):
        u=np.asarray([((z+.071*k)%.9)+.03 for k in range(dof)])
        actual=np.asarray(wp.get_position_from_free_xyzs(u),float)%1.0
        approx=(A@u+zero)%1.0
        delta=(actual-approx+.5)%1.-.5
        if float(np.max(abs(delta)))>2e-5:
            raise ValueError('Unexpected nonlinear Wyckoff free-position mapping; '
                             f'WP={wp} max_error={max(abs(delta)):.3e}')
    return A,zero

def template_from_token(spg,token,device):
    from pyxtal.symmetry import Group
    g=Group(int(spg));blocks=[];dof_count=0;n_ti=0
    for i in token:
        wp=g[int(i)];d=int(wp.get_dof())
        A,b=affine_map(wp,d)
        rot=np.asarray([op.rotation_matrix for op in wp.ops],float)
        tra=np.asarray([op.translation_vector for op in wp.ops],float)
        if len(rot)!=int(wp.multiplicity):
            raise RuntimeError(f'WP {i}: number of symmetry ops != multiplicity')
        blocks.append({'index':int(i),'label':str(getattr(wp,'letter',i)), 'dof':d,
                       'A':torch.tensor(A,dtype=DTYPE,device=device),
                       'b':torch.tensor(b,dtype=DTYPE,device=device),
                       'rot':torch.tensor(rot,dtype=DTYPE,device=device),
                       'tra':torch.tensor(tra,dtype=DTYPE,device=device)})
        dof_count+=d;n_ti+=len(rot)
    return WyckoffTemplate(spg,tuple(token),n_ti,str(g.lattice_type).lower(),
                           lattice_parameters(g.lattice_type),blocks,dof_count)

def make_lattice(t,raw):
    """Conventional cell, same lattice constraints as v140. Raw lengths=log(length)."""
    B=len(raw);nlat=len(t.lattice_keys);lt=t.lattice_type
    x=raw[:,:nlat]
    if lt=='cubic':
        a=torch.exp(x[:,0].clamp(-1.5,5.)); lengths=torch.stack([a,a,a],-1)
        alpha=beta=gamma=torch.ones(B,dtype=DTYPE,device=raw.device)*(math.pi/2)
    elif lt in ('tetragonal','hexagonal','trigonal'):
        a=torch.exp(x[:,0].clamp(-1.5,5.));c=torch.exp(x[:,1].clamp(-1.5,5.))
        lengths=torch.stack([a,a,c],-1)
        alpha=beta=torch.ones(B,dtype=DTYPE,device=raw.device)*(math.pi/2)
        gamma=torch.ones(B,dtype=DTYPE,device=raw.device)*(2*math.pi/3 if lt in ('hexagonal','trigonal') else math.pi/2)
    elif lt=='orthorhombic':
        lengths=torch.exp(x[:,:3].clamp(-1.5,5.))
        alpha=beta=gamma=torch.ones(B,dtype=DTYPE,device=raw.device)*(math.pi/2)
    elif lt=='monoclinic':
        lengths=torch.exp(x[:,:3].clamp(-1.5,5.))
        beta=math.pi/3+torch.sigmoid(x[:,3])*math.pi/3
        alpha=gamma=torch.ones_like(beta)*(math.pi/2)
    else:
        lengths=torch.exp(x[:,:3].clamp(-1.5,5.))
        alpha,beta,gamma=(math.pi/4+torch.sigmoid(x[:,i])*math.pi/2 for i in (3,4,5))
    a,b,c=lengths.unbind(-1)
    ca,cb,cg=torch.cos(alpha),torch.cos(beta),torch.cos(gamma)
    sg=torch.sin(gamma).clamp_min(1e-6)
    y3=c*(ca-cb*cg)/sg
    zsq=c*c-(c*cb)**2-y3*y3
    zz=torch.sqrt(zsq.clamp_min(1e-10))
    zero=torch.zeros_like(a)
    cell=torch.stack([torch.stack([a,zero,zero],-1),
                      torch.stack([b*cg,b*sg,zero],-1),
                      torch.stack([c*cb,y3,zz],-1)],dim=1)
    ang=torch.stack([alpha,beta,gamma],-1)*180./math.pi
    return cell,lengths,ang,zsq

def expand_ti(t,raw):
    cur=len(t.lattice_keys);coords=[];indep=[]
    for wp in t.blocks:
        d=wp['dof'];u=torch.sigmoid(raw[:,cur:cur+d]);cur+=d
        representative=(u@wp['A'].T+wp['b'])%1.
        orb=(torch.einsum('oij,bj->boi',wp['rot'],representative)+wp['tra'][None,:,:])%1.
        coords.append(orb);indep.append(u)
    return torch.cat(coords,dim=1),indep

def geometry(t,raw,cell_override=None):
    cell,abc,ang,zsq=make_lattice(t,raw)
    if cell_override is not None:
        cell=torch.as_tensor(cell_override,dtype=raw.dtype,device=raw.device)
        if cell.ndim==2:cell=cell[None].expand(len(raw),-1,-1)
        zsq=torch.ones_like(zsq)
    frac,indep=expand_ti(t,raw)
    return cell,frac,abc,ang,zsq,indep

def neighbor_torch(frac,cell,k,shifts):
    """Same pair-centered image kernel as v22r1; batched and autograd-safe."""
    B,N,_=frac.shape
    df=frac[:,:,None,:]-frac[:,None,:,:] # [B, center, base, xyz]
    df=df-torch.round(df).detach()
    rel=df[:,:,:,None,:]+shifts[None,None,None,:,:]
    vv=torch.einsum('bnmsd,bdk->bnmsk',rel,cell).reshape(B,N,-1,3)
    dist=torch.linalg.vector_norm(vv,dim=-1)
    # Self-exclude the central image only (v22r1-style physical neighbors).
    own=torch.eye(N,dtype=torch.bool,device=frac.device)[None,:,:,None]
    shift0=(torch.abs(shifts).sum(-1)<1e-10)[None,None,None,:]
    dist=dist.masked_fill((own & shift0).reshape(1,N,-1),1e8)
    if k>dist.shape[-1]:raise RuntimeError('Not enough periodic image neighbors')
    nd,indices=torch.topk(dist,k,dim=-1,largest=False)
    nv=torch.gather(vv,2,indices[...,None].expand(-1,-1,-1,3))
    return nv,nd,vv,dist

class TiObjective:
    def __init__(self,lib,args,device):
        self.device=device;self.lib=lib;self.args=args
        r=max(lib['image_radius'],args.image_radius)
        self.image_radius=r
        self.shifts=torch.tensor(shifts_np(r),dtype=DTYPE,device=device)
        self.mode_groups={}
        for k in sorted(set(x['k'] for x in lib['modes'])):
            im=[(i,m) for i,m in enumerate(lib['modes']) if m['k']==k]
            mm=[m for _,m in im]
            self.mode_groups[k]={
                'r':torch.tensor(np.stack([m['r'] for m in mm]),dtype=DTYPE,device=device),
                'p':torch.tensor(np.stack([m['p'] for m in mm]),dtype=DTYPE,device=device),
                'G':torch.tensor(np.stack([m['G'] for m in mm]),dtype=DTYPE,device=device),
                'angles':torch.tensor(np.stack([m['angles_rad'] for m in mm]),dtype=DTYPE,device=device),
                'eig':torch.tensor(np.stack([m['eig'] for m in mm]),dtype=DTYPE,device=device),
                'indices':[i for i,_ in im]}
        self.centers=torch.tensor(lib['soap_centers'],dtype=DTYPE,device=device)
        self.soap_med=torch.tensor(lib['soap_med'],dtype=DTYPE,device=device)
        self.soap_scale=torch.tensor(lib['soap_scale'],dtype=DTYPE,device=device)
        self.soap_pca_mode=getattr(args,'ti_soap_pca','audit')
        self.soap_pca_basis=None
        self.soap_pca_center=None
        if self.soap_pca_mode=='on':
            count=report_ti_soap_pca(lib,args)
            self.soap_pca_basis=torch.tensor(lib['soap_pca_basis'][:,:count],dtype=DTYPE,device=device)
            self.soap_pca_center=torch.tensor(lib['soap_pca_mean'],dtype=DTYPE,device=device)
        self.maxk=max(lib['max_mode_k'],lib['soap_neighbors'],24, max(x[1] for x in lib['density_profile']))
        self.rules=lib.get('_ti_rule_values',_TI_LEGACY_RULES)

    def evaluate(self,t,raw,stage,details=False,cell_override=None):
        cell,frac,abc,angles,zsq,_=geometry(t,raw,cell_override)
        B,N,_=frac.shape
        vectors,nd,allv,alld=neighbor_torch(frac,cell,self.maxk,self.shifts)
        valid=torch.isfinite(nd).all(-1).all(-1) & (zsq>1e-6)
        hard=self.lib['hard_titi'];width=float(self.rules['repulsion_width_A'])
        near=torch.nn.functional.softplus((hard-alld)/width)*width
        # All contact images, rather than only the closest global contact.
        repulsion=(near*near).sum((1,2))/N / (float(self.rules['repulsion_normalizer_A'])**2)
        # Smooth radius-count density profile. Target counts extracted directly from ref Ti CIFs.
        dens=[];observed=[]
        for radius,ncnt in self.lib['density_profile']:
            count=torch.sigmoid((radius-alld)/float(self.rules['density_width_A'])).sum(-1)
            z=(count-ncnt)/max(1.5,.20*ncnt)
            dens.append(torch.nn.functional.smooth_l1_loss(z,torch.zeros_like(z),beta=1.,reduction='none'))
            observed.append(count)
        density=torch.stack(dens,-1).mean((1,2))
        vol=torch.linalg.det(cell).abs()/N
        vref=self.lib['volume_median']
        zv=torch.log((vol/vref).clamp_min(1e-9))/float(self.rules['volume_log_width'])
        volume=torch.nn.functional.smooth_l1_loss(zv,torch.zeros_like(zv),beta=1.,reduction='none')
        # Preserve learned low-end 3-D span, but never force isotropy.
        span_k=int(self.rules['span_neighbors'])
        uv=vectors[:,:,:span_k,:]/nd[:,:,:span_k,None].clamp_min(1e-8)
        moment=torch.einsum('bnkd,bnke->bnde',uv,uv)/uv.shape[2]
        lam3=torch.linalg.eigvalsh(moment)[...,0]
        floor=self.lib['lambda3_floor']
        planar=torch.relu((floor-lam3)/max(floor,.05)).pow(2).mean(1)
        if stage=='natural':
            loss=(1.0*density+.30*volume+2.0*repulsion+.12*planar)
            geom_resid=torch.zeros(B,device=self.device,dtype=DTYPE)
            soft_resid=geom_resid; soap_resid=geom_resid
            site_best=None;mode_ids=None;site_score=None
            site_radial=site_pair=site_angular=site_gram=site_shape=None
        else:
            groups=[];identities=[];radial_parts=[];pair_parts=[]
            angular_parts=[];gram_parts=[];shape_parts=[]
            for k,m in self.mode_groups.items():
                nm=len(m['indices']); vec=vectors[:,:,:k,:];d=nd[:,:,:k]
                ij=torch.triu_indices(k,k,1,device=self.device)
                sep=torch.linalg.vector_norm(vec[:,:,:,None,:]-vec[:,:,None,:,:],dim=-1)
                pair=torch.sort(sep[:,:,ij[0],ij[1]],dim=-1).values
                er=((d[:,:,None,:]-m['r'][None,None,:,:])**2).mean(-1)
                ep=((pair[:,:,None,:]-m['p'][None,None,:,:])**2).mean(-1)
                # Explicit Ti--Ti--Ti angular distribution (radians).
                unit=vec/d[...,None].clamp_min(1e-9)
                gram=torch.einsum('bnkd,bnld->bnkl',unit,unit).clamp(-1.,1.)
                theta=torch.sort(torch.acos(gram[:,:,ij[0],ij[1]].clamp(-1.+1e-7,1.-1e-7)),dim=-1).values
                angle_m=m['angles'][None,None,:,:]
                ea=((theta[:,:,None,:]-angle_m)**2).mean(-1)
                # Retain the pairwise CORRESPONDENCE of angles via Gram graphs.
                # 2-opt correspondence search is performed on detached tensors;
                # aligned Gram residuals remain fully differentiable.
                gram_exp=gram[:,:,None,:,:].expand(B,N,nm,k,k)
                ref_exp=m['G'][None,None,:,:,:].expand(B,N,nm,k,k)
                rad_exp=d[:,:,None,:].expand(B,N,nm,k)
                ref_rad=m['r'][None,None,:,:].expand(B,N,nm,k)
                aligned,_=gram_permutation_match(gram_exp,ref_exp,rad_exp,ref_rad,
                                                  max_swaps=self.args.gram_swaps)
                # Off-diagonal cosine RMS is smooth, without acos singularities.
                eg=((aligned[...,ij[0],ij[1]]-ref_exp[...,ij[0],ij[1]])**2).mean(-1)
                # Three-dimensional span/anisotropy. Not an isotropy target:
                # match the eigenvalue spectrum of the actual reference mode.
                moment=torch.einsum('bnkd,bnke->bnde',unit,unit)/k
                eig=torch.linalg.eigvalsh(moment)
                es=((eig[:,:,None,:]-m['eig'][None,None,:,:])**2).mean(-1)
                # Convert angular residual to equivalent bond-vector distance.
                # Weights balance radial, unordered pairs, angle histogram,
                # matched Gram graph, and shape, all in squared-A units.
                score=torch.sqrt(.25*er+.30*ep+.20*(2.7**2)*ea+
                                 .20*(2.7**2)*eg+.05*(3.0**2)*es+1e-12)
                groups.append(score)
                if details:
                    radial_parts.append(torch.sqrt(er+1e-15))
                    pair_parts.append(torch.sqrt(ep+1e-15))
                    angular_parts.append(torch.sqrt(ea+1e-15)*(180./math.pi))
                    # Report matched physical angles rather than just cosine loss.
                    grad_safe_aligned=aligned[...,ij[0],ij[1]].clamp(-1.+1e-7,1.-1e-7)
                    grad_safe_ref=ref_exp[...,ij[0],ij[1]].clamp(-1.+1e-7,1.-1e-7)
                    gram_angle=torch.sqrt(((torch.acos(grad_safe_aligned)-
                                            torch.acos(grad_safe_ref))**2).mean(-1)+1e-15)
                    gram_parts.append(gram_angle*(180./math.pi))
                    shape_parts.append(torch.sqrt(es+1e-15))
                identities.extend(m['indices'])
            matrix=torch.cat(groups,dim=-1)
            site_best,where=matrix.min(-1)
            ids=torch.tensor(identities,dtype=torch.long,device=self.device)
            mode_ids=ids[where]
            def choose(parts):
                return torch.cat(parts,dim=-1).gather(-1,where[...,None]).squeeze(-1)
            if details:
                site_radial=choose(radial_parts);site_pair=choose(pair_parts)
                site_angular=choose(angular_parts);site_gram=choose(gram_parts)
                site_shape=choose(shape_parts)
            temperature=max(float(self.args.mode_temperature),1e-4)
            msoft=-temperature*torch.logsumexp(-matrix/temperature,dim=-1)
            msoft=msoft+temperature*math.log(matrix.shape[-1])
            site_score=.70*site_best+.30*msoft
            worst=max(1,math.ceil(N*.25))
            geom_resid=.70*site_score.mean(1)+.30*torch.topk(site_score,worst,dim=1).values.mean(1)
            soft_resid=site_best.mean(1)
            vv=vectors[:,:,:self.lib['soap_neighbors'],:]
            fp=soap_torch(vv,self.centers,self.lib['soap_width'],self.lib['soap_lmax'])
            z=(fp-self.soap_med)/self.soap_scale
            if self.soap_pca_basis is None:
                soap_local=torch.nn.functional.smooth_l1_loss(z,torch.zeros_like(z),beta=2.,reduction='none').mean(-1)
            else:
                # Experimental compression, explicit residual penalty. The
                # full reference SOAP descriptor is still computed; PCA reduces
                # objective feature dimension, not neighbor count or LCAO physics.
                zc=z-self.soap_pca_center
                coeff=zc@self.soap_pca_basis
                residual_sq=(zc.square().sum(-1)-coeff.square().sum(-1)).clamp_min(0.)
                soap_local=(torch.nn.functional.smooth_l1_loss(coeff,torch.zeros_like(coeff),
                            beta=2.,reduction='none').sum(-1)+.25*residual_sq)/z.shape[-1]
            soap_resid=.70*soap_local.mean(1)+.30*torch.topk(soap_local,worst,dim=1).values.mean(1)
            loss=(geom_resid+.12*density+.08*volume+2.0*repulsion+.12*planar
                  +float(self.args.soap_weight)*soap_resid)
        # If metric is degenerate, penalize it without losing differentiability.
        metric=torch.relu(1e-5-zsq)*1e5
        loss=torch.nan_to_num(loss+metric,nan=1e7,posinf=1e7,neginf=1e7)
        if not details:return loss
        ans={'total':loss,'geometry':geom_resid,'mode_hard_mean':soft_resid,
             'soap':soap_resid,'density':density,'volume_penalty':volume,
             'repulsion':repulsion,'planarity':planar,
             'minimum_titi_A':nd[:,:,0].min(1).values,
             'volume_per_ti_A3':vol,'lambda3_min':lam3.min(1).values,
             'valid':valid,'cell':cell,'ti_frac':frac,
             'lattice_lengths':abc,'lattice_angles':angles,'density_counts':torch.stack(observed,-1).mean(1),
             'per_site_mode_A':site_best,'per_site_mode_id':mode_ids,
             'per_site_objective_A':site_score,
             'per_site_radial_A':site_radial,'per_site_pair_A':site_pair,
             'per_site_angular_deg':site_angular,'per_site_gram_deg':site_gram,
             'per_site_shape_rms':site_shape}
        return loss,ans

def initial_raw(t,lib,starts,seed,fixed=False):
    rng=np.random.default_rng(seed);nlat=len(t.lattice_keys)
    n=nlat+t.ncoord
    x=np.zeros((starts,n),float)
    mean_len=(lib['volume_median']*t.n_ti)**(1/3)
    nlength={'cubic':1,'tetragonal':2,'hexagonal':2,'trigonal':2,'orthorhombic':3,
             'monoclinic':3}.get(t.lattice_type,3)
    for i in range(starts):
        # Start at a mildly expanded but condensed Ti framework, not an empty huge cell.
        perturb=np.exp(rng.normal(0.,.17,nlength))
        x[i,:nlength]=np.log(mean_len*1.08*perturb)
        if nlat>nlength:x[i,nlength:nlat]=rng.normal(0.,.20,nlat-nlength)
        if t.ncoord:
            u=rng.uniform(.025,.975,size=t.ncoord)
            x[i,nlat:]=np.log(u/(1.-u))
    return x

_V49_ACTIVE=None  # Retained legacy experiment, disabled in v50 production.

_V50_CONFIG=None    # Independent runtime policy: initialized in each worker process.

_V50_COUNTERS=collections.Counter()

def _v50_enabled():
    return bool(_V50_CONFIG is not None and getattr(_V50_CONFIG,'v50_fast',False))

def _v50_checkpoint_complete(candidate,threshold=None):
    """Existing full calibrated chemical gate, with no new physics or tolerances."""
    cfg=_V50_CONFIG
    if threshold is None:
        threshold=min(.12,float(getattr(cfg,'early_accept_sharing_A',.10))) if cfg else .10
    detail=candidate['details']
    # Complete parent-SG symmetry cannot be certified by local chemistry alone.
    if cfg is not None and getattr(cfg,'early_require_parent_sg',False):
        return False
    return bool(candidate.get('bad_triplets',1)==0
                and candidate.get('rms',float('inf'))<=threshold
                and float(detail['collision'])<=float(getattr(cfg,'early_accept_collision',1e-8))
                and int(detail['badTi'])==0 and int(detail['badO'])==0
                and np.all(detail['hardTi']==6) and np.all(detail['hardO']==3))

def _v49_motion_policy():
    return _V49_ACTIVE

def _v49_ti_motion(t,x,override,active,step,stage):
    control=_v49_motion_policy()
    if control is None:return None
    return control.ti(t,x,override,active,step,stage)

def _v49_oxygen_motion(kind,oxygen,cell,step,topology=None,rank=None):
    control=_v49_motion_policy()
    if control is None:return False
    return control.oxygen(kind,oxygen,cell,step,topology,rank)

def run_optimizer(objective,t,initial,stage,steps,lr,freeze_cell=False,override=None,
                  progress=0,patience=30,min_steps=50,early_tol=1e-4):
    """Batched Adam with per-start stagnation-based stopping.

    A slow-converging start is not terminated merely because another start has
    already converged. Only an individual start with no meaningful improvement
    for `patience` iterations is stopped, and its BEST structure is retained.
    """
    if _v50_enabled():
        return _v50_run_optimizer(objective,t,initial,stage,steps,lr,freeze_cell,
                                  override,progress,patience,min_steps,early_tol)
    if steps<=0:
        return initial.detach().clone(),[],{'steps_executed':0,'early_stopped':0}
    x=initial.detach().clone().requires_grad_(True)
    nlat=len(t.lattice_keys)
    optim=torch.optim.Adam([x],lr=lr)
    count=len(x)
    best=x.detach().clone()
    bestE=torch.full((count,),float('inf'),dtype=DTYPE,device=x.device)
    anchor=torch.full_like(bestE,float('inf'))
    stalled=torch.zeros(count,dtype=torch.int,device=x.device)
    active=torch.ones(count,dtype=torch.bool,device=x.device)
    finished=torch.zeros(count,dtype=torch.int,device=x.device)
    trace=[];iters=0
    for it in range(steps):
        iters=it+1
        fac=.15+.85*.5*(1.+math.cos(math.pi*it/max(steps-1,1)))
        for pg in optim.param_groups:pg['lr']=float(lr)*fac
        optim.zero_grad(set_to_none=True)
        y=objective.evaluate(t,x,stage,cell_override=override)
        if not bool(torch.isfinite(y).all()):
            log(f'  WARNING {stage}: non-finite objective at step {it+1}; retaining best states')
            break
        with torch.no_grad():
            better=y<bestE
            bestE=torch.where(better,y.detach(),bestE)
            best[better]=x.detach()[better]
            significant=(anchor==float('inf')) | (y.detach()<anchor-early_tol*(1.+anchor.abs()))
            anchor=torch.where(significant,y.detach(),anchor)
            stalled=torch.where(significant,torch.zeros_like(stalled),stalled+active.int())
            newly=active & (stalled>=patience) & (it+1>=min_steps)
            finished[newly]=it+1
            active=active & ~newly
        if not bool(active.any()):break
        y[active].sum().backward()
        if x.grad is None:break
        if freeze_cell:x.grad[:,:nlat]=0.
        with torch.no_grad():x.grad[~active]=0.
        torch.nn.utils.clip_grad_norm_([x],max_norm=10.)
        optim.step()
        # Adam momentum moves entries even with zero gradient: explicitly pin
        # finished starts to their best point to prevent hidden drift.
        with torch.no_grad():x[~active]=best[~active]
        motion_done=_v49_ti_motion(t,x,override,active,it+1,stage)
        if motion_done is not None:
            newly=active & torch.as_tensor(motion_done,dtype=torch.bool,device=x.device)
            finished[newly]=it+1
            active=active & ~newly
            with torch.no_grad():x[~active]=best[~active]
            if not bool(active.any()):break
        if progress and ((it+1)%progress==0 or it==0 or it+1==steps):
            with torch.no_grad():
                E,D=objective.evaluate(t,x,stage,details=True,cell_override=override)
                best_i=int(E.argmin())
                record={'step':it+1,'stage':stage,'best_objective':float(E[best_i]),
                        'median_objective':float(E.median()),
                        'best_geometry_A':float(D['geometry'][best_i]),
                        'best_density':float(D['density'][best_i]),
                        'best_soap':float(D['soap'][best_i]),
                        'min_titi_A':float(D['minimum_titi_A'][best_i]),
                        'best_volume_per_ti_A3':float(D['volume_per_ti_A3'][best_i]),
                        'active_starts':int(active.sum())}
                trace.append(record)
                log('  '+stage+f' step {it+1:4d}/{steps}: '
                    +f'best={record["best_objective"]:.4f} median={record["median_objective"]:.4f} '
                    +f'mode={record["best_geometry_A"]:.3f} A '
                    +f'active={record["active_starts"]}/{count}')
    with torch.no_grad():
        y=objective.evaluate(t,x,stage,cell_override=override)
        better=y<bestE
        best[better]=x.detach()[better]
        finished[finished==0]=iters
        ybest,scores=objective.evaluate(t,best,stage,details=True,cell_override=override)
        result={'steps_executed':iters,
                'early_stopped':int((finished<steps).sum()),
                'finish_step_per_start':finished.cpu().numpy(),
                'median_final_objective':float(torch.median(ybest)),
                'best_final_objective':float(ybest.min()),
                'min_titi_A':float(scores['minimum_titi_A'].min())}
    log(f'  {stage}: steps={iters}/{steps}, early-stopped={result["early_stopped"]}/{count}, '
        f'best={result["best_final_objective"]:.4f}')
    return best.detach().clone(),trace,result

def periodic_match_rms(ref_frac,probe_frac,cell):
    """Match two Ti lattices under identical setting, atom permutation and origin shift."""
    if len(ref_frac)!=len(probe_frac):return float('inf')
    ref=np.asarray(ref_frac,float); probe=np.asarray(probe_frac,float);cell=np.asarray(cell,float)
    if len(ref)==0:return 0.
    # An arbitrary origin is a gauge degree of freedom. Try aligning each probe site to ref[0].
    best=float('inf')
    shifts=shifts_np(1)
    for j in range(len(probe)):
        trial=(probe+ref[0]-probe[j])%1.
        df=ref[:,None,:]-trial[None,:,:]
        df-=np.rint(df)
        dv=(df[:,:,None,:]+shifts[None,None,:,:])@cell
        dist=np.linalg.norm(dv,axis=-1).min(-1)
        rows,cols=linear_sum_assignment(dist**2)
        rm=float(np.sqrt(np.mean(dist[rows,cols]**2)))
        best=min(best,rm)
    return best

def cell_metric_discrepancy(A,B):
    G1=np.asarray(A)@np.asarray(A).T
    G2=np.asarray(B)@np.asarray(B).T
    return float(np.linalg.norm(G1-G2)/max(np.linalg.norm(G1),1e-9))

def group_basins(states,objective,t,args,override):
    """Classify converged starts: basis/matching invariant to atom permutations and origin.

    For a fixed Wyckoff combination the orbit order/setting is fixed. The metric
    prefilter and origin-shift matching avoid falsely counting a coordinate gauge as
    a new lattice. Deliberately conservative: ambiguous results remain distinct.
    """
    with torch.no_grad():
        losses,D=objective.evaluate(t,states,'learned',details=True,cell_override=override)
    los=losses.cpu().numpy()
    cc=D['cell'].cpu().numpy();ff=D['ti_frac'].cpu().numpy()%1.
    clusters=[]
    for i in np.argsort(los):
        i=int(i);assigned=False
        for cluster in clusters:
            j=cluster['representative']
            if cell_metric_discrepancy(cc[j],cc[i])>args.basin_metric_tolerance:continue
            if periodic_match_rms(ff[j],ff[i],cc[j])<=args.basin_rms_tolerance:
                cluster['members'].append(i);assigned=True;break
        if not assigned:
            clusters.append({'representative':i,'members':[i]})
    return clusters,los,D

def parse_ti_args(argv=None):
    """Parse private Ti-backend arguments, independently from the public v59 CLI."""
    ap=argparse.ArgumentParser(description='Juliette internal Ti-only Wyckoff optimizer')
    ap.add_argument('--ref',default='ref',help='CIF training directory, ref/*.cif')
    ap.add_argument('--output',default='ti_generation_v25',help='New Ti-only results directory')
    ap.add_argument('--space-groups',default='136',help='Single SG, comma list, range 1-230, or all')
    ap.add_argument('--ti-rule-mode',choices=('legacy','compare','hybrid','learned'),default='compare',
                    help='compare (default): legacy generation + hybrid/learned audit; hybrid: learned targets with empirical flexibility; learned: unconstrained inferred widths; legacy: original')
    add_ti_empirical_cli(ap)
    ap.add_argument('--ti-count',type=int,default=12,help='Ti atoms in generated conventional cell')
    ap.add_argument('--wyckoff',default=None,help='Specific zero-based PyXtal WP indices, comma-separated; one space group only')
    ap.add_argument('--max-combinations',type=int,default=8,help='Number of enumerated Wyckoff combinations in the deterministic selection pool')
    ap.add_argument('--max-wyckoff',type=int,default=0,help='Seeded distinct Wyckoff combinations to generate (0=all in pool)')
    ap.add_argument('--max-orbits',type=int,default=12,help='Max inequivalent Ti Wyckoff orbits per combination')
    ap.add_argument('--starts',type=int,default=12,help='Independent random Ti initializations per combination')
    ap.add_argument('--adaptive-starts',action=argparse.BooleanOptionalAction,default=False,help='Stop sampling when no new Ti basin appears in consecutive batches')
    ap.add_argument('--start-batch',type=int,default=16)
    ap.add_argument('--no-new-batches',type=int,default=2)
    ap.add_argument('--keep-basins',type=int,default=3,help='Best distinct basins saved per Wyckoff combination; 0 = all')
    ap.add_argument('--natural-steps',type=int,default=120)
    ap.add_argument('--learned-steps',type=int,default=160)
    ap.add_argument('--natural-lr',type=float,default=.040)
    ap.add_argument('--learned-lr',type=float,default=.020)
    ap.add_argument('--mode-temperature',type=float,default=.075,help='Soft matching temperature in Angstrom')
    ap.add_argument('--soap-weight',type=float,default=.05,help='Weak direct-reference Ti SOAP conditioning')
    ap.add_argument('--gram-swaps',type=int,default=8,help='2-opt passes; ambiguous correspondences also try 3- and 4-cycles and exact reference-identity rescue')
    ap.add_argument('--stop-patience',type=int,default=30,help='Stagnant steps before stopping individual starts (0 disables)')
    ap.add_argument('--stop-tolerance',type=float,default=.00010,help='Meaningful objective reduction for early stopping')
    ap.add_argument('--minimum-steps',type=int,default=55,help='Minimum optimizer steps before early stopping')
    ap.add_argument('--no-global-dedup',action='store_true',help='Keep all basin CIFs in oxygen entrance; skip crystallographic dedup')
    ap.add_argument('--image-radius',type=int,default=2,help='Periodic image box, at least learning validated radius')
    ap.add_argument('--shell-min-k',type=int,default=8,help='v22r1 Ti->Ti minimum extended shell')
    ap.add_argument('--shell-max-k',type=int,default=30)
    ap.add_argument('--shell-gap',type=float,default=.015)
    ap.add_argument('--mode-merge',type=float,default=.040)
    ap.add_argument('--soap-neighbors',type=int,default=24)
    ap.add_argument('--fixed-cell',action='store_true',help='Freeze lattice; requires --lattice')
    ap.add_argument('--lattice',default=None,help='a,b,c,alpha,beta,gamma, in A and degrees')
    ap.add_argument('--basin-rms-tolerance',type=float,default=.12,help='Ti matching RMS tolerance, Angstrom')
    ap.add_argument('--basin-metric-tolerance',type=float,default=.025,help='Relative metric discrepancy')
    ap.add_argument('--progress-every',type=int,default=0,help='Iteration diagnostics (0 = print only stage summaries)')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='auto',help='auto, cpu or cuda:0')
    return ap.parse_args(argv)

CHANNELS={'Ti':[('O',6,10),('Ti',0,8)],'O':[('Ti',3,7),('O',0,10)]}

SHIFTS=np.asarray(list(itertools.product([-1.,0.,1.],repeat=3)),float)

LOCAL_SHIFTS=SHIFTS.copy()

def structure(path):
    from pymatgen.core import Structure
    s=Structure.from_file(str(path))
    species={x.specie.symbol for x in s}
    if species!={'Ti','O'}:raise ValueError(f'{path}: expected Ti and O, got {species}')
    return s

def sites(s,sp):
    return np.asarray([x.frac_coords%1 for x in s if x.specie.symbol==sp],float)

def periodic_neighbors(center,base,n,cell,shifts=None):
    """Periodic shell-complete vectors, invariant to fractional-cell wrapping.

    Unlike v12, images are centered around each center/base pair's nearest
    lattice translation BEFORE extending by the learned image-box radius.
    This prevents a small oxygen shift through a cell boundary from silently
    deleting a distant yet required neighbor image in a higher-order shell.
    """
    if shifts is None:
        shifts=torch.as_tensor(SHIFTS,dtype=center.dtype,device=center.device)
    # Each center/base pair gets its own central image, then full surrounding
    # shifts; coordinate wrap of ANY atom does not change neighbor signatures.
    relative=base[None,:,:]-center[:,None,:]
    relative=relative-torch.round(relative)
    delta=(relative[:,:,None,:]+shifts[None,None,:,:])@cell
    delta=delta.reshape(len(center),-1,3)
    dist=torch.linalg.vector_norm(delta,dim=-1)
    dist=torch.where(dist<1e-7,torch.full_like(dist,1e5),dist)
    if dist.shape[1]<n:
        raise RuntimeError(f'Not enough periodic replicas for {n} neighbors')
    ix=torch.topk(dist,n,dim=1,largest=False).indices
    vec=torch.gather(delta,1,ix[:,:,None].expand(-1,-1,3))
    return vec,torch.gather(dist,1,ix)

def all_periodic_distances(center,base,cell,shifts):
    """All contacts in the pair-centered image box, base-major then shift."""
    rel=base[None,:,:]-center[:,None,:]
    rel=rel-torch.round(rel)
    coords=(rel[:,:,None,:]+shifts[None,None,:,:])@cell
    return torch.linalg.vector_norm(coords,dim=-1).reshape(len(center),-1)

def reference_neighbors(s,i,sp,k):
    """Rigid arms / learned shell gaps use EXACTLY the scoring neighbor kernel."""
    center=torch.tensor(np.asarray(s[i].frac_coords,float).reshape(1,3)%1,dtype=DTYPE)
    base=torch.tensor(sites(s,sp),dtype=DTYPE)
    cell=torch.tensor(s.lattice.matrix,dtype=DTYPE)
    with torch.no_grad():
        v,d=periodic_neighbors(center,base,k,cell)
    return v[0].detach().cpu().numpy(),d[0].tolist()

def complete_shell_channel_sizes(structures,minimum_gap=.015,max_k=30):
    """Learn a *shell-complete size for EACH atom and channel*.

    A common k across anatase, brookite, hollandite and rutile is generally
    impossible. Each learned mode owns its reference shell counts; prediction
    is matched against these same counts. No tied periodic shell is truncated
    in the training data and no environment information is silently dropped.
    """
    global SHIFTS
    print('\nPER-ATOM PERIODIC SHELL-COMPLETENESS CHECK',flush=True)
    # Determine shell ends with the full 7x7x7 image box. Runtime uses the
    # smallest image box proven equivalent on every selected reference shell.
    wide=torch.tensor(np.asarray(list(itertools.product(range(-3,4),repeat=3)),float),dtype=DTYPE)
    shell_config=[]
    for st in structures:
        counts={}
        for sp in ('Ti','O'):
            per_atom=[]
            for target,first,n0 in CHANNELS[sp]:
                center=torch.tensor(sites(st,sp),dtype=DTYPE)
                base=torch.tensor(sites(st,target),dtype=DTYPE)
                cell=torch.tensor(st.lattice.matrix,dtype=DTYPE)
                with torch.no_grad():
                    _,dist=periodic_neighbors(center,base,max_k+1,cell,wide)
                d=np.sort(dist.detach().cpu().numpy(),axis=1)
                selected=[];gaps=[]
                for index,di in enumerate(d):
                    good=np.flatnonzero((di[n0:max_k+1]-di[n0-1:max_k])>=minimum_gap)
                    if not len(good):
                        raise RuntimeError(f'{sp}->{target} atom {index+1}: no complete shell '
                                           f'from k={n0} to k={max_k}; inspect actual '
                                           'distance degeneracies or increase --shell-max-k.')
                    k=int(n0+good[0])
                    selected.append(k);gaps.append(float(di[k]-di[k-1]))
                per_atom.append(np.array(selected,dtype=int))
                hist=dict(zip(*np.unique(selected,return_counts=True)))
                h='{'+', '.join(f'{int(k)}:{int(v)}' for k,v in hist.items())+'}'
                print(f'  {sp}->{target}: requested {n0}; reference shell sizes {h}; '
                      f'min boundary gap={min(gaps):.4f} A',flush=True)
            counts[sp]=np.stack(per_atom,axis=1)
        shell_config.append(counts)
    # Determine whether periodic replicas beyond [-1,1] are required.
    # The wide reference must contain all selected neighbors. If not, fail
    # explicitly rather than report spurious symmetry-invariant descriptors.
    sizes=(1,2,3)
    image_boxes={r:torch.tensor(np.asarray(list(itertools.product(range(-r,r+1),repeat=3)),float),dtype=DTYPE)
                 for r in sizes}
    selected_radius=1
    for st,cfg in zip(structures,shell_config):
        cell=torch.tensor(st.lattice.matrix,dtype=DTYPE)
        for sp in ('Ti','O'):
            centers=torch.tensor(sites(st,sp),dtype=DTYPE)
            for j,(target,_,_) in enumerate(CHANNELS[sp]):
                base=torch.tensor(sites(st,target),dtype=DTYPE)
                # Compare each selected k against the larger image box.
                k=int(cfg[sp][:,j].max())
                with torch.no_grad():
                    _,ref=periodic_neighbors(centers,base,k,cell,image_boxes[3])
                    ref=np.sort(ref.detach().cpu().numpy(),axis=1)
                    for radius in sizes[:-1]:
                        _,test=periodic_neighbors(centers,base,k,cell,image_boxes[radius])
                        if np.max(np.abs(np.sort(test.detach().cpu().numpy(),axis=1)-ref))<1e-6:
                            break
                    else:
                        radius=3
                selected_radius=max(selected_radius,radius)
    SHIFTS=np.asarray(list(itertools.product(range(-selected_radius,selected_radius+1),repeat=3)),float)
    print(f'  Periodic images: [ {-selected_radius}, {selected_radius} ]^3, '
          f'{len(SHIFTS)} image translations (validated against [-3,3]^3).',flush=True)
    print('  All reference modes retain their OWN complete outer shells.',flush=True)
    print('  No globally shared k is imposed on incompatible polymorphs.',flush=True)
    return shell_config

def reference_descriptor_matrices(s,counts):
    """Return per-atom four-channel descriptors; shell sizes may differ by atom.

    By grouping equal (primary-total, secondary-total) counts, this remains
    cheap even when symmetry-equivalent atoms comprise most of a structure.
    """
    ti=torch.tensor(sites(s,'Ti'),dtype=DTYPE)
    o=torch.tensor(sites(s,'O'),dtype=DTYPE)
    cell=torch.tensor(s.lattice.matrix,dtype=DTYPE)
    coord={'Ti':ti,'O':o}
    result={}
    with torch.no_grad():
        for sp,first in [('Ti',6),('O',3)]:
            rows=counts[sp]
            entries=[None]*len(rows)
            for k0,k1 in sorted(set(map(tuple,rows.tolist()))):
                idx=np.flatnonzero(np.all(rows==[k0,k1],axis=1))
                center=coord[sp][idx]
                primary_sp='O' if sp=='Ti' else 'Ti'
                primary,_=periodic_neighbors(center,coord[primary_sp],k0,cell)
                other,_=periodic_neighbors(center,coord[sp],k1,cell)
                mats=torch_signature(primary,other,first)
                for a,i in enumerate(idx):
                    entries[i]=[section[a].detach().cpu().numpy().copy() for section in mats]
            result[sp]=entries
    return result

def learn(paths,mode_merge=.040,accept_floor=.075,shell_max_k=30,reference_structures=None):
    training={'Ti':[],'O':[]}; first=[];nexts=[]; oo_min=[]; reference=[]
    print('\n'+'='*80+'\nLEARNING: COHERENT ATOM-CENTERED LOCAL CHEMISTRY',flush=True)
    reference=[structure(p) for p in paths] if reference_structures is None else list(reference_structures)
    if len(reference)!=len(paths):raise ValueError("Reference structures and path names have different sizes")
    shell_config=complete_shell_channel_sizes(reference,max_k=shell_max_k)
    for path,s,counts in zip(paths,reference,shell_config):
        print(f'\nTRAIN {path.name}: Ti={len(sites(s,"Ti"))} O={len(sites(s,"O"))}',flush=True)
        sigs=reference_descriptor_matrices(s,counts)
        per_species={'Ti':0,'O':0}
        for i,at in enumerate(s):
            sp=at.specie.symbol
            idx=per_species[sp];per_species[sp]+=1
            sections=[section.copy() for section in sigs[sp][idx]]
            selected_counts=tuple(map(int,counts[sp][idx]))
            training[sp].append(dict(parts=sections,counts=selected_counts,source=path.stem,index=i))
            if sp in ('Ti','O'):
                main='O' if sp=='Ti' else 'Ti'
                k=6 if sp=='Ti' else 3
                _,dd=reference_neighbors(s,i,main,k+1)
                first.append((sp,dd[k-1]));nexts.append((sp,dd[k]))
                if sp=='O':
                    _,oo=reference_neighbors(s,i,'O',1);oo_min.append(oo[0])
    library={}
    for sp,records in training.items():
        modes=[]
        for item in records:
            ds=[sum(float(np.mean((a-b)**2)) for a,b in zip(item['parts'],mode['parts']))
                if item['counts']==mode['counts'] else float('inf')
                for mode in modes]
            if ds and min(ds)**.5 <= mode_merge:
                modes[int(np.argmin(ds))]['count']+=1
                continue
            modes.append({**item,'count':1})
        library[sp]=modes
        print(f'\n{sp}: {len(records)} reference atoms -> {len(modes)} coherent full-environment modes',flush=True)
        for j,m in enumerate(modes):
            print(f'  mode {j:2d} source={m["source"]:11s} atom={m["index"]:3d} '
                  f'multiplicity={m["count"]:2d} shell_counts={m["counts"]} '
                  f'dimensions={[len(z) for z in m["parts"]]}',flush=True)
    bounds={}
    for sp,k in [('Ti',6),('O',3)]:
        f=[v for s,v in first if s==sp];n=[v for s,v in nexts if s==sp]
        high=max(f);low=min(n)
        gap=low-high
        if gap<=0:
            raise ValueError(f'{sp} coordination shells overlap across training structures: '
                             f'max primary={high:.4f} min next={low:.4f}; a universal cutoff is unsafe.')
        threshold=(high+low)/2
        bounds[sp]=(threshold,high,low)
        print(f'{sp}-center first-shell {k}: furthest bonded={high:.4f} A, nearest extra={low:.4f} A, '
              f'gap={gap:.4f} A, LEARNED cutoff={threshold:.4f} A',flush=True)
    # Strict regression: reference descriptors must be reproducible by the
    # exact same neighbor machinery used at evaluation time.
    worst_roundtrip=0.
    for path,st,counts in zip(paths,reference,shell_config):
        batch=reference_descriptor_matrices(st,counts)
        for sp in ('Ti','O'):
            records=[r for r in training[sp] if r['source']==path.stem]
            if len(records)!=len(batch[sp]):
                raise RuntimeError(f'{path.name}: reference descriptor atom count changed')
            for j,record in enumerate(records):
                if record['counts']!=tuple(map(int,counts[sp][j])):
                    raise RuntimeError(f'{path.name}: reference descriptor shell sizes changed')
                for part,mat in zip(record['parts'],batch[sp][j]):
                    worst_roundtrip=max(worst_roundtrip,float(np.max(np.abs(part-mat))))
    print(f'  Reference descriptor round-trip: max difference={worst_roundtrip:.3e} A',flush=True)
    if worst_roundtrip>1e-8:
        raise RuntimeError('Learning/evaluation descriptor mismatch: STOP before generation')
    # Physically informed O--O short-distance guard, learned from actual references.
    oo_guard=max(0.,min(oo_min)-.12)
    print(f'O--O shortest training distance={min(oo_min):.4f} A; collision guard={oo_guard:.4f} A',flush=True)
    print('DESCRIPTORS: primary sorted bond + full primary pair geometry; outer same-species '
          'radii; opposite-species radial/pair geometry; cross-species pair distances.',flush=True)
    print('LIMITATION: sorting each distance family does not preserve all neighbor correspondences.',flush=True)
    print('REFERENCE and generated structures use the SAME differentiable periodic neighbor kernel.',flush=True)
    print('Each mode uses its own shell-complete neighbor counts; no global-count constraint.',flush=True)
    print('REFERENCE O coordinates are discarded after template and cutoff extraction.',flush=True)
    return library,bounds,oo_guard,reference

def torch_signature(v,w,k):
    d=torch.linalg.vector_norm(v,dim=-1)
    near=v[:,:k,:]
    pairs=torch.linalg.vector_norm(near[:,:,None,:]-near[:,None,:,:],dim=-1)
    ij=torch.triu_indices(k,k,1)
    primary=torch.cat((torch.sort(d[:,:k],dim=1).values,
                       torch.sort(pairs[:,ij[0],ij[1]],dim=1).values),dim=1)
    extra=torch.sort(d[:,k:],dim=1).values
    other_d=torch.linalg.vector_norm(w,dim=-1)
    otherpair=torch.linalg.vector_norm(w[:,:,None,:]-w[:,None,:,:],dim=-1)
    ij2=torch.triu_indices(w.shape[1],w.shape[1],1)
    other=torch.cat((torch.sort(other_d,dim=1).values,
                     torch.sort(otherpair[:,ij2[0],ij2[1]],dim=1).values),dim=1)
    cross=torch.sort(torch.linalg.vector_norm(near[:,:,None,:]-w[:,None,:,:],dim=-1).reshape(len(v),-1),dim=1).values
    return [primary,extra,other,cross]

class Chemistry:
    def __init__(self,ti,cell,lib,bounds,oo_guard,accept=.10,hierarchy="ti-first",calibration=None,target_o_cn=True):
        self.ti=torch.tensor(ti,dtype=DTYPE)
        self.cell=torch.tensor(cell,dtype=DTYPE)
        self.shifts=torch.tensor(SHIFTS,dtype=DTYPE)
        self.lib={}
        self.groups={}
        for sp in ('Ti','O'):
            groups=defaultdict(list)
            for index,mode in enumerate(lib[sp]):
                groups[tuple(mode['counts'])].append(index)
            self.groups[sp]={}
            for counts,indices in sorted(groups.items()):
                refs=[torch.tensor(np.stack([lib[sp][i]['parts'][j] for i in indices]),dtype=DTYPE)
                      for j in range(4)]
                self.groups[sp][counts]=(torch.tensor(indices,dtype=torch.long),refs)
            print(f'  chemistry {sp} comparison groups: '
                  + ', '.join(f'k={key}: {len(ids)} coherent modes'
                              for key,(ids,_) in self.groups[sp].items()),flush=True)
        self.cutTi=bounds['Ti'][0];self.cutO=bounds['O'][0];self.guard=oo_guard
        if hierarchy not in ('v20','ti-first'):
            raise ValueError(f'Unknown hierarchy {hierarchy!r}')
        self.hierarchy=hierarchy
        self.accept=accept;self.nTi=len(ti)
        self.calibration=calibration
        self.target_o_cn=bool(target_o_cn)
        # Ti-centered complete-mode geometry is primary. In ti-first mode,
        # O's primary Ti3 triangle is a gentle construction hint while
        # extended O-centered shell topology is weak: it is not a whitelist.
        # All four channel values still come strictly from the four CIFs.
        self.weightsTi=torch.tensor([1.0,.65,.45,.40],dtype=DTYPE)
        self.weightsO=(torch.tensor([1.0,.65,.45,.40],dtype=DTYPE)
                       if hierarchy=='v20' else
                       torch.tensor([.18,.030,.020,.020],dtype=DTYPE))
        self.sharing_weight=1.5 if hierarchy=='v20' else 4.0
        self.collision_weight=.1 if hierarchy=='v20' else .75
        self.source=lib
        self.ablation_oxygen_weight=1.0
        self.minimum_learned_TiO_bond=min(float(m['parts'][0][0]) for m in lib['Ti'])
    def neighboring(self,center,base,n):
        return periodic_neighbors(center,base,n,self.cell,self.shifts)
    def score(self,oxygen,details=False,components=False):
        o=oxygen.reshape(-1,3)
        ti=self.ti
        # Each reference mode is evaluated against ITS OWN complete-shell
        # neighbor counts. Matching never combines channels from different modes.
        coord={'Ti':ti,'O':o}
        residual={};modes={};score={}
        diagnostic_residual={};diagnostic_mode={}
        cache={}
        for sp,first in [('Ti',6),('O',3)]:
            cost_list=[];resid_list=[];id_list=[]
            old_cost_list=[];old_resid_list=[];old_id_list=[]
            for (k0,k1),(ids,refs) in self.groups[sp].items():
                center=coord[sp]
                primary_sp='O' if sp=='Ti' else 'Ti'
                ck0=(sp,primary_sp,k0)
                ck1=(sp,sp,k1)
                if ck0 not in cache:cache[ck0]=self.neighboring(center,coord[primary_sp],k0)[0]
                if ck1 not in cache:cache[ck1]=self.neighboring(center,coord[sp],k1)[0]
                des=torch_signature(cache[ck0],cache[ck1],first)
                per=torch.stack([
                    torch.sqrt(torch.mean((section[:,None,:]-ref[None,:,:])**2,dim=-1)+1e-12)
                    for section,ref in zip(des,refs)],dim=-1)
                weights=self.weightsTi if sp=='Ti' else self.weightsO
                mode_cost=torch.sum(weights[None,None,:]*per**2,dim=-1)
                best,local_ids=mode_cost.min(dim=1)
                selected=per[torch.arange(len(local_ids)),local_ids]
                cost_list.append(best)
                resid_list.append(selected)
                id_list.append(ids[local_ids])
                # Unmodified v20 full-mode selector for OUTCOME DIAGNOSTICS,
                # independently of the weak O template selector driving v21.
                # This prevents reweighting from artificially hiding O failures.
                if sp=='O' and self.hierarchy=='ti-first':
                    legacy_cost=torch.sum(self.weightsTi[None,None,:]*per**2,dim=-1)
                    old_best,old_ids=legacy_cost.min(dim=1)
                    old_cost_list.append(old_best)
                    old_resid_list.append(per[torch.arange(len(old_ids)),old_ids])
                    old_id_list.append(ids[old_ids])
            candidate_cost=torch.stack(cost_list,dim=1)
            bestgroup=candidate_cost.argmin(dim=1)
            raw_mode_cost=candidate_cost.gather(1,bestgroup[:,None]).squeeze(1)
            residual[sp]=torch.stack(resid_list,dim=1)[torch.arange(len(bestgroup)),bestgroup]
            modes[sp]=torch.stack(id_list,dim=1)[torch.arange(len(bestgroup)),bestgroup]
            if self.calibration is None:
                score[sp]=raw_mode_cost
            else:
                # A learned residual inside the observed range is NOT a defect.
                # Retain a small continuous reference pull so the objective
                # remains informative inside the calibrated acceptance region.
                lim=self.calibration.limit_tensor(sp,modes[sp],self.source)
                excess=torch.relu(residual[sp]-lim)
                w=self.weightsTi if sp=='Ti' else self.weightsO
                score[sp]=torch.sum(w[None,:]*(excess.square()+
                         self.calibration.reference_pull*residual[sp].square()),dim=1)
            if old_cost_list:
                dgroup=torch.stack(old_cost_list,dim=1).argmin(dim=1)
                diagnostic_residual[sp]=torch.stack(old_resid_list,dim=1)[
                    torch.arange(len(dgroup)),dgroup]
                diagnostic_mode[sp]=torch.stack(old_id_list,dim=1)[
                    torch.arange(len(dgroup)),dgroup]
            else:
                diagnostic_residual[sp]=residual[sp]
                diagnostic_mode[sp]=modes[sp]
        # Ablation: erase *learned* O-centered descriptor forces only.
        # Keep shared O-site construction, O CN3 counting, and O-O collisions.
        score["O"]=score["O"]*self.ablation_oxygen_weight
        # Translation-invariant all-contact counting. Recenter the image box
        # per atomic pair: fixed-image replicas from v12 become incorrect when
        # a moving oxygen crosses a fractional-cell boundary.
        dto=all_periodic_distances(ti,o,self.cell,self.shifts)
        dot=all_periodic_distances(o,ti,self.cell,self.shifts)
        soft=.055
        cnTi=torch.sigmoid((self.cutTi-dto)/soft).sum(dim=1)
        cnO=torch.sigmoid((self.cutO-dot)/soft).sum(dim=1)
        # Same oxygen's central zero image is excluded, but its nonzero
        # periodic self-images are physical O--O neighbors.
        doo=all_periodic_distances(o,o,self.cell,self.shifts)
        shiftzero=torch.all(self.shifts==0,dim=1)
        image_base=torch.arange(len(o),device=o.device).repeat_interleave(len(self.shifts))
        zero_images=shiftzero.repeat(len(o))
        selfmask=(image_base[None,:]==torch.arange(len(o),device=o.device)[:,None]) & zero_images[None,:]
        doo=torch.where(selfmask,torch.full_like(doo,1e4),doo)
        collision=(torch.relu(self.guard-doo)**2).sum(dim=1)
        # Ti--O hard short-distance guard derived conservatively from references.
        toshort=(torch.relu(self.minimum_learned_TiO_bond-.12-dto)**2).sum(dim=1)
        if self.hierarchy=='v20':
            # Exact original v20 chemical energy for controlled A/B tests.
            localTi=score['Ti']+.65*(cnTi-6)**2+.35*toshort
            localO=score['O']+(.65*(cnO-3)**2 if self.target_o_cn else 0.)+.7*collision
        else:
            # Keep connection RULES strong (soft CN 6/3, contact guards),
            # independently of weak, *learned* O-side topology preference.
            # Coordination remains a smooth optimizer signal, never a hard
            # candidate-rejection/SCF-stopping condition.
            localTi=score['Ti']+1.20*(cnTi-6)**2+.65*toshort
            localO=score['O']+(1.20*(cnO-3)**2 if self.target_o_cn else 0.)+1.40*collision
        everything=torch.cat((localTi,localO))
        # Scalar aggregation ONLY for optimizers; retain every local vector.
        # Normalize smooth-max so a perfectly matched structure scores ~0,
        # rather than carrying an artificial ln(number of atoms) offset.
        worst=(torch.logsumexp(10*everything,dim=0)-math.log(len(everything)))/10
        total=localTi.mean()+localO.mean()+.3*worst
        if components:
            # Return differentiable FULL-channel residuals for coherent
            # octahedral-force feedback. The independent O positions are NOT
            # variables: oxygen centers are projected from octahedra.
            if self.calibration is None:
                ti_lim=torch.tensor([self.accept,2*self.accept,2*self.accept,2*self.accept],dtype=DTYPE)
                o_lim=ti_lim
                limitsTi=ti_lim[None,:].expand_as(residual['Ti'])
                limitsO=o_lim[None,:].expand_as(residual['O'])
            else:
                limitsTi=self.calibration.limit_tensor('Ti',modes['Ti'],self.source)
                limitsO=self.calibration.limit_tensor('O',modes['O'],self.source)
            return total,localTi,localO,residual['Ti'],residual['O'],limitsTi,limitsO
        if not details:return total
        with torch.no_grad():
            # Final O-mode diagnostics always use v20 full-channel matching.
            # Differentiable optimizer components above still use the new
            # topology-guidance selector; they must not share a strict O gate.
            residual=diagnostic_residual
            modes=diagnostic_mode
            # hard coordination tally independent of sigmoid threshold.
            hardTi=(dto<self.cutTi).sum(dim=1)
            hardO=(dot<self.cutO).sum(dim=1)
            # Keep exact coordination and learned geometry as INDEPENDENT
            # final measurements. Topology is NOT a generation gate.
            old_lim=torch.tensor([self.accept,2*self.accept,2*self.accept,2*self.accept],dtype=DTYPE)
            old_geoTi=torch.all(residual['Ti']<=old_lim[None,:],dim=1)
            old_geoO=torch.all(residual['O']<=old_lim[None,:],dim=1)
            limTi=(self.calibration.limit_tensor('Ti',modes['Ti'],self.source)
                   if self.calibration is not None else old_lim[None,:].expand_as(residual['Ti']))
            limO=(self.calibration.limit_tensor('O',modes['O'],self.source)
                  if self.calibration is not None else old_lim[None,:].expand_as(residual['O']))
            geoPrimaryTi=(residual['Ti'][:,0]<=limTi[:,0])
            geoPrimaryO=(residual['O'][:,0]<=limO[:,0])
            geoTi=torch.all(residual['Ti']<=limTi,dim=1)
            geoO=torch.all(residual['O']<=limO,dim=1)
            topologyTi=(hardTi==6)
            topologyO=(hardO==3)
            # Retain combined legacy counters for historical comparisons only.
            acceptTi=geoPrimaryTi&topologyTi
            acceptO=geoPrimaryO&topologyO
            allTi=geoTi&topologyTi
            allO=geoO&topologyO
            return {'energy':float(total),'hierarchy':self.hierarchy,
                    'geometryLossTi':float(score['Ti'].mean()),
                    'geometryLossO':float(score['O'].mean()),
                    'connectionCNlossTi':float(((cnTi-6)**2).mean()),
                    'connectionCNlossO':float(((cnO-3)**2).mean()),
                    'target_o_cn':self.target_o_cn,
                    'localTi':localTi.detach().cpu().numpy(),
                    'localO':localO.detach().cpu().numpy(),'rTi':residual['Ti'].detach().cpu().numpy(),
                    'rO':residual['O'].detach().cpu().numpy(),'modeTi':modes['Ti'].detach().cpu().numpy(),
                    'modeO':modes['O'].detach().cpu().numpy(),'hardTi':hardTi.detach().cpu().numpy(),
                    'hardO':hardO.detach().cpu().numpy(),'softTi':cnTi.detach().cpu().numpy(),
                    'softO':cnO.detach().cpu().numpy(),'badTi':int((~allTi).sum()),
                    'badO':int((~allO).sum()),'primaryBadTi':int((~acceptTi).sum()),
                    'primaryBadO':int((~acceptO).sum()),'worst':float(everything.max()),
                    'collision':float(collision.sum()),'goodTi':allTi.detach().cpu().numpy(),
                    'goodO':allO.detach().cpu().numpy(),
                    'geometryTi':geoTi.detach().cpu().numpy(),
                    'geometryO':geoO.detach().cpu().numpy(),
                    'topologyTi':topologyTi.detach().cpu().numpy(),
                    'topologyO':topologyO.detach().cpu().numpy(),
                    'geometryBadTi':int((~geoTi).sum()),
                    'geometryBadO':int((~geoO).sum()),
                    'geometryTi_v21':old_geoTi.detach().cpu().numpy(),
                    'geometryO_v21':old_geoO.detach().cpu().numpy(),
                    'geometryBadTi_v21':int((~old_geoTi).sum()),
                    'geometryBadO_v21':int((~old_geoO).sum()),
                    'geometry_limitsTi':limTi.detach().cpu().numpy(),
                    'geometry_limitsO':limO.detach().cpu().numpy(),
                    'geometry_policy':(self.calibration.name if self.calibration is not None else 'v21_strict'),
                    'topologyBadTi':int((~topologyTi).sum()),
                    'topologyBadO':int((~topologyO).sum())}

def continuous_candidate_objective(candidate):
    """Learned chemistry, coherent sharing and O-O repulsion; no hard topology gate.

    Chemistry.energy contains *soft* coordination guidance. Neither hard CN,
    topology-bad counts, local PASS flags nor the original database O enter
    candidate selection. A candidate with imperfect topology remains viable.
    """
    d=candidate['details']
    # All SCF/global controllers use the *same* hierarchy as chemistry
    # feedback. No separate hard O-geometry pass flag enters the ranking.
    hierarchy=d.get('hierarchy','v20')
    sharing_weight=4.0 if hierarchy=='ti-first' else 1.5
    collision_weight=.75 if hierarchy=='ti-first' else .1
    # Two crystallographically different oxygen sites must not silently be
    # assigned an identical periodic-Ti triplet. This discrete defect has no
    # differentiable O-coordinate force, but LCAO can repair it after a move.
    duplicate_cost=(2.0*float(candidate['bad_triplets'])/
                    max(1,len(candidate['oxygen']))) if hierarchy=='ti-first' else 0.
    return (float(d['energy']) + sharing_weight*float(candidate['rms'])**2
            + collision_weight*float(d['collision']) + duplicate_cost)

def trilaterate(p,rr):
    """Construct O from 3 fixed Ti positions and a learned OTi3 bond triple."""
    a,b,c=np.asarray(p,float);ra,rb,rc=rr
    ab=b-a; d=np.linalg.norm(ab)
    if d<1e-7:return []
    ex=ab/d;v=c-a;i=np.dot(ex,v);tmp=v-i*ex;j=np.linalg.norm(tmp)
    if j<1e-6:return []
    ey=tmp/j;ez=np.cross(ex,ey)
    x=(ra*ra-rb*rb+d*d)/(2*d)
    y=(ra*ra-rc*rc+i*i+j*j-2*i*x)/(2*j)
    zz=ra*ra-x*x-y*y
    if zz<-.035:return []
    z=np.sqrt(max(0.,zz))
    return [a+x*ex+y*ey+sign*z*ez for sign in ((-1,1) if z>1e-8 else (1,))]

def frac_candidates(ti,cell,lib,seed,cap=2500):
    """Generate oxygen candidate 3D points exclusively from Ti lattice + learned
    OTi3 distances. No target O positions are read. Candidate count bounded."""
    rng=np.random.default_rng(seed)
    inv=np.linalg.inv(cell)
    car=np.asarray(ti)@cell
    # Candidate Ti triplets: central Ti plus pairs of nearby distinct periodic Ti images.
    Tiimg=(np.asarray(ti)[None,:,:]+LOCAL_SHIFTS[:,None,:]).reshape(-1,3)@cell
    tree=cKDTree(Tiimg)
    allcand=[];triangles=0
    # fixed Ti--Ti separations in learned OTi3 triangles
    d3=[]
    for t in lib['O']:
        # O-centered OTi3 primary length-3 and pair-distance-3 signature
        signature=t['parts'][0]
        d3.append(np.sort(signature[3:6]))
    largest=max(float(max(z)) for z in d3)+.24
    for a,p in enumerate(car):
        ix=tree.query_ball_point(p,r=largest)
        ix=[i for i in ix if np.linalg.norm(Tiimg[i]-p)>1e-6]
        if len(ix)>45:
            ix=sorted(ix,key=lambda i:np.linalg.norm(Tiimg[i]-p))[:45]
        for ii,jj in itertools.combinations(ix,2):
            pb,pc=Tiimg[ii],Tiimg[jj]
            if np.linalg.norm(pb-pc)>largest:continue
            trip=np.stack((p,pb,pc))
            sides=np.sort([np.linalg.norm(p-pb),np.linalg.norm(p-pc),np.linalg.norm(pb-pc)])
            # Candidate ranking, never a hard chemistry gate. Even a strongly
            # distorted Ti3 triangle is permitted to propose a provisional O;
            # only SCF and final learned chemistry decide its compatibility.
            distances=np.array([np.sqrt(np.mean((sides-tri)**2)) for tri in d3])
            valid=[lib['O'][z] for z in np.argsort(distances)[:2]]
            triangles+=1
            for mode in valid:
                rad=np.asarray(mode['parts'][0][:3])
                for order in set(itertools.permutations(rad.tolist(),3)):
                    for xyz in trilaterate(trip,order):
                        allcand.append((xyz@inv)%1.)
    if not allcand:
        print('  OTi3 Ti-only triangle field: EMPTY; continue with TiO6-first coherent LCAO proposals',flush=True)
        return np.empty((0,3),dtype=float)
    arr=np.asarray(allcand)
    # periodic deduplication by cheap rounding, followed by near-neighbor merging
    rounded=np.round(arr/.012).astype(int)
    _,unique=np.unique(rounded,axis=0,return_index=True)
    arr=arr[np.sort(unique)]
    rng.shuffle(arr)
    if len(arr)>cap:arr=arr[:cap]
    print(f'  OTi3 candidate construction: Ti triangles={triangles}; raw={len(allcand)}; '
          f'unique/capped={len(arr)} (NO target O used)',flush=True)
    return arr

def wrap_cart(point, cell, inv=None):
    if inv is None:
        inv = np.linalg.inv(cell)
    return ((np.asarray(point) @ inv) % 1.0) @ cell

def minimum_image_delta(a, b, cell, inv=None):
    """Return Cartesian vectors a - nearest periodic image of b.

    Both arguments have final dimension 3, with broadcast-compatible leading
    dimensions. Explicit replica comparison is essential for skewed cells.
    """
    if inv is None:
        inv = np.linalg.inv(cell)
    da = (np.asarray(a) - np.asarray(b)) @ inv
    base = da - np.rint(da)
    shifts = np.asarray(list(itertools.product((-1., 0., 1.), repeat=3)))
    d = (base[..., None, :] + shifts) @ cell
    best = np.argmin(np.sum(d * d, axis=-1), axis=-1)
    return np.take_along_axis(d, np.expand_dims(best, (-1, -2)), axis=-2)[..., 0, :]

def periodic_point_tree(frac, cell):
    points = (np.asarray(frac)[None, :, :] + LOCAL_SHIFTS[:, None, :]).reshape(-1, 3) @ cell
    return cKDTree(points), points

def mode_shortlist(ti, cell, library, k=6):
    """Rank TiO6 templates by Ti-only neighbors. Never inspect test oxygen."""
    result=[]
    ti_t=torch.tensor(ti,dtype=DTYPE)
    c=torch.tensor(cell,dtype=DTYPE)
    for i in range(len(ti)):
        vals=[]
        for m_id,m in enumerate(library['Ti']):
            n=int(m['counts'][1])
            v,_=periodic_neighbors(ti_t[i:i+1],ti_t,n,c)
            v=v[0].detach().cpu().numpy()
            dist=np.linalg.norm(v,axis=1)
            pairs=np.linalg.norm(v[:,None,:]-v[None,:,:],axis=-1)
            r=np.r_[np.sort(dist),np.sort(pairs[np.triu_indices(n,1)])]
            ref=m['parts'][2]
            vals.append((float(np.sqrt(np.mean((r-ref)**2))),m_id))
        vals.sort()
        result.append([(index,err) for err,index in vals[:k]])
    return result

def triangle_orientation_seeds(ti,cell,peaks,library,mode_choices,seed,
                               proposals_per_ti=8,max_local_peaks=26,max_pairs=22):
    """Fit whole learned TiO6 octahedra to the Ti-only OTi3 proposal field.

    Each returned trial is a complete six-arm octahedron, expressed by a
    learned mode and a proper rotation; no six independent O positions exist.
    """
    rng=np.random.default_rng(seed)
    if len(peaks)==0:
        return [[] for _ in range(len(ti))]
    tree,images=periodic_point_tree(peaks,cell)
    refs=np.asarray(list(itertools.combinations(range(6),3)),dtype=int)
    answer=[]
    maxbond=max(float(np.max(np.linalg.norm(m['rigid_arms'],axis=1))) for m in library['Ti'])
    for i,frac in enumerate(ti):
        center=frac@cell
        local=tree.query_ball_point(center,r=maxbond+.24)
        if not local:
            answer.append([])
            continue
        local_pts=images[local]-center
        # Retain the peaks whose bond distances most nearly match learned modes.
        radial=np.linalg.norm(local_pts,axis=1)
        learned=np.concatenate([np.linalg.norm(library['Ti'][j]['rigid_arms'],axis=1)
                                for j,_ in mode_choices[i]])
        delta=np.min(np.abs(radial[:,None]-learned[None,:]),axis=1)
        order=np.argsort(delta)
        local_pts=local_pts[order[:max_local_peaks]]
        if len(local_pts)<3:
            answer.append([])
            continue
        triples=np.asarray(list(itertools.combinations(range(len(local_pts)),3)),dtype=int)
        c3=local_pts[triples]
        edges=np.sort(np.column_stack((
            np.linalg.norm(c3[:,0]-c3[:,1],axis=1),
            np.linalg.norm(c3[:,0]-c3[:,2],axis=1),
            np.linalg.norm(c3[:,1]-c3[:,2],axis=1))),axis=1)
        trials=[]
        for mode_id,framework_error in mode_choices[i]:
            arms=library['Ti'][mode_id]['rigid_arms']
            t3=arms[refs]
            sides=np.sort(np.column_stack((
                np.linalg.norm(t3[:,0]-t3[:,1],axis=1),
                np.linalg.norm(t3[:,0]-t3[:,2],axis=1),
                np.linalg.norm(t3[:,1]-t3[:,2],axis=1))),axis=1)
            err=np.sqrt(np.mean((edges[:,None,:]-sides[None,:,:])**2,axis=-1))
            nsel=min(max_pairs,err.size)
            flat=np.argpartition(err.ravel(),nsel-1)[:nsel]
            for idx in flat:
                ci,ai=np.unravel_index(int(idx),err.shape)
                # Ranking only: an initially distorted TiO6 MUST NOT be
                # discarded because a pre-SCF triangle residual is large.
                targets=c3[ci]
                source=t3[ai]
                for perm in itertools.permutations((0,1,2)):
                    try:
                        rot,fit=Rotation.align_vectors(targets[list(perm)],source)
                    except ValueError:
                        continue
                    pts=center+rot.apply(arms)
                    near=tree.query(pts)[0]
                    cost=float(np.mean(np.minimum(near,1.5)**2))
                    cost+=.08*float(fit)/3+.05*framework_error**2
                    trials.append((cost,int(mode_id),rot))
        if not trials:
            answer.append([])
            continue
        trials.sort(key=lambda x:x[0])
        kept=[]
        for cost,mode_id,rot in trials:
            if any(j==mode_id and (r.inv()*rot).magnitude()<.17 for _,j,r in kept):
                continue
            kept.append((cost,mode_id,rot))
            if len(kept)>=proposals_per_ti:
                break
        answer.append(kept)
    return answer

def framework_orientation_seeds(ti,cell,library,mode_choices,peaks,
                                proposals_per_ti=8):
    """Initialize COMPLETE learned TiO6 units from the Ti-only framework.

    This is independent of whether a Ti3 triangle already matches a learned
    OTi3 triangle. Three Ti-neighbor vectors provide a proper-rotation frame;
    learned TiO6 geometry is NOT distorted at initialization. The OTi3 field,
    when present, ranks orientations but never determines admissibility.
    No target oxygen coordinates or pre-SCF geometric rejection thresholds.
    """
    n=len(ti);inv=np.linalg.inv(cell)
    t=torch.tensor(ti,dtype=DTYPE)
    c=torch.tensor(cell,dtype=DTYPE)
    frame=periodic_neighbors(t,t,8,c)[0].detach().cpu().numpy()
    triplets=np.asarray(list(itertools.combinations(range(6),3)),dtype=int)
    # Optional LCAO peak field only ranks geometrical hypotheses.
    peak_tree,peak_images=(periodic_point_tree(peaks,cell)
                           if len(peaks) else (None,None))
    answer=[]
    for i in range(n):
        target=frame[i];trial=[]
        for mode_id,framework_error in mode_choices[i]:
            m=library['Ti'][int(mode_id)]
            source=np.asarray(m['framework_arms'],float)
            arms=np.asarray(m['rigid_arms'],float)
            src=source[triplets]
            dst=target[triplets]
            def sig(x):
                radii=np.sort(np.linalg.norm(x,axis=-1),axis=-1)
                pairs=np.sort(np.stack((np.linalg.norm(x[:,0]-x[:,1],axis=-1),
                                        np.linalg.norm(x[:,0]-x[:,2],axis=-1),
                                        np.linalg.norm(x[:,1]-x[:,2],axis=-1)),axis=-1),axis=-1)
                return np.concatenate((radii,pairs),axis=-1)
            sr=sig(src);tr=sig(dst)
            mismatch=np.sqrt(np.mean((tr[:,None,:]-sr[None,:,:])**2,axis=-1))
            top=min(9,mismatch.size)
            order=np.argpartition(mismatch.ravel(),top-1)[:top]
            for flat in order:
                tid,sid=np.unravel_index(int(flat),mismatch.shape)
                for perm in itertools.permutations(range(3)):
                    try:
                        rot,fit=Rotation.align_vectors(dst[tid][list(perm)],src[sid])
                    except ValueError:
                        continue
                    if not np.isfinite(fit):
                        continue
                    # Secondary score on full Ti-only environment, not O.
                    predicted=rot.apply(source)
                    delta=np.linalg.norm(predicted[:,None,:]-target[None,:,:],axis=-1)
                    frame_cost=float(np.mean(np.min(delta,axis=1)**2))
                    cost=.20*float(mismatch[tid,sid]**2)+.13*frame_cost
                    cost+=.03*float(framework_error**2)
                    if peak_tree is not None:
                        guess=ti[i]@cell+rot.apply(arms)
                        near=peak_tree.query(guess,k=1)[0]
                        cost+=float(np.mean(np.minimum(near,1.8)**2))
                    trial.append((cost,int(mode_id),rot))
            # No failure of entry simply because a framework lacks a
            # suitable triangle. Every learned template remains a candidate.
            trial.append((3.+.03*float(framework_error**2),int(mode_id),Rotation.identity()))
        trial.sort(key=lambda z:z[0])
        kept=[]
        for item in trial:
            _,mode,rot=item
            if any(mid==mode and (previous.inv()*rot).magnitude()<.18
                   for _,mid,previous in kept):
                continue
            kept.append(item)
            if len(kept)>=proposals_per_ti:
                break
        if not kept:
            raise RuntimeError(f'Ti{i}: missing learned TiO6 templates (model error)')
        answer.append(kept)
    return answer

def coherent_lcao_proposal_field(octas,original_peaks):
    """Propose provisional O sites from whole TiO6 vertices, no O dynamics.

    This guarantees a TiO6-first entrance independent of whether the Ti-only
    network already admits exact learned OTi3 trilateration. For each octahedral
    vertex, identify nearby vertices of OTHER periodic Ti images, and propose
    a triplet centroid. Unresolved proposals remain provisional and must
    are measured against learned chemistry at the end without rejection.
    """
    n=len(octas.ti)
    arms=octas.positions().reshape(-1,3)
    cell=octas.cell
    frac=(arms@octas.inv)%1
    tree,images=periodic_point_tree(frac,cell)
    centers=[]
    for base_id,point in enumerate(arms):
        ds,inds=tree.query(point,k=min(len(images),28))
        seen={base_id};points=[point]
        for idx in np.atleast_1d(inds):
            armid=int(idx)%(6*n)
            if armid in seen:continue
            seen.add(armid)
            points.append(images[int(idx)])
            if len(points)==3:break
        if len(points)==3:
            centers.append((np.mean(points,axis=0)@octas.inv)%1)
    if len(original_peaks):
        centers.extend(np.asarray(original_peaks,float))
    # Include the actual six-vertex locations as additional provisional sites.
    # These are not free coordinates and never become optimization variables.
    centers.extend(frac)
    arr=np.asarray(centers,dtype=float).reshape(-1,3)
    if not len(arr):raise RuntimeError('Learned TiO6 has no vertices')
    return arr

class Octahedra:
    """ONLY Ti-centered whole octahedral mode, orientation, and distortion DOFs."""
    def __init__(self,ti,cell,lib,proposals,seed=0):
        self.ti=np.array(ti,dtype=float)
        self.cell=np.array(cell,dtype=float)
        self.inv=np.linalg.inv(cell)
        self.lib=lib
        self.mode=[];self.rot=[];self.stretch=[]
        rng=np.random.default_rng(seed)
        for trials in proposals:
            # A common best hypothesis is essential for robust convergence;
            # other entrances sample distinct *whole-octahedron* alternatives.
            if seed==0:
                which=0
            else:
                weights=np.exp(-np.clip(np.asarray([z[0] for z in trials])-
                                         trials[0][0],0,2)/.07)
                weights=weights/weights.sum()
                which=int(rng.choice(len(trials),p=weights))
            _,mode,rot=trials[which]
            self.mode.append(mode)
            self.rot.append(rot)
            self.stretch.append(np.eye(3))
        self.mode=np.array(self.mode,dtype=int)
        self.stretch=np.array(self.stretch)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def positions(self):
        result=[]
        for i in range(len(self.ti)):
            arm=self.lib['Ti'][int(self.mode[i])]['rigid_arms']
            result.append(self.ti[i]@self.cell+self.rot[i].apply(arm@self.stretch[i]))
        return np.asarray(result)

    def distortion(self):
        return float(np.sqrt(np.mean(np.sum((self.stretch-np.eye(3))**2,axis=(1,2)))))

    def fit(self,targets,mode_options=None,mix=.55,distortion_max=.11,
            distortion_ridge=1.5):
        """Fit TiO6 rotations and bounded symmetric local strain.

        All targets are six provisional bond vectors per Ti. NO oxygen
        positions are optimized here. Each TiO6 moves as a coherent unit.
        """
        distortion_max=min(float(distortion_max),float(getattr(self,"_ablation_max_strain",distortion_max)))
        for i in range(len(self.ti)):
            opts=[int(self.mode[i])]
            if mode_options is not None:
                opts+= [int(k) for k,_ in mode_options[i] if k!=self.mode[i]]
            fitlist=[]
            for mode in opts:
                src=self.lib['Ti'][mode]['rigid_arms']
                try:
                    rot,rss=Rotation.align_vectors(targets[i],src)
                except ValueError:
                    continue
                # Local-frame symmetric distortion (not six Cartesian shifts).
                tgt_local=rot.inv().apply(targets[i])
                reg=distortion_ridge*np.eye(3)
                D=np.linalg.solve(src.T@src+reg,src.T@tgt_local+reg)
                D=.5*(D+D.T)
                eigen,Q=np.linalg.eigh(D)
                D=(Q*np.clip(eigen,1-distortion_max,1+distortion_max))@Q.T
                fit_error=np.mean(np.sum((rot.apply(src@D)-targets[i])**2,axis=1))
                fit_error+=.20*np.sum((D-np.eye(3))**2)
                fitlist.append((fit_error,mode,rot,D))
            if not fitlist:
                continue
            fitlist.sort(key=lambda z:z[0])
            _,mode,rot,D=fitlist[0]
            if mode!=self.mode[i] and np.linalg.norm(self.stretch[i]-np.eye(3))<.20:
                self.mode[i]=mode
                self.stretch[i]=np.eye(3)
            diff=rot*self.rot[i].inv()
            self.rot[i]=Rotation.from_rotvec(mix*diff.as_rotvec())*self.rot[i]
            self.stretch[i]=(1-mix)*self.stretch[i]+mix*D
            # Keep the six-vertex octahedral object near a learned template.
            val,Q=np.linalg.eigh(self.stretch[i])
            self.stretch[i]=(Q*np.clip(val,1-distortion_max,1+distortion_max))@Q.T

def _select_provisional_centers(octas,peaks,oo_guard):
    """Select provisional oxygen sites for one LCAO proposal field."""
    n=len(octas.ti);count=2*n
    if len(peaks)<count:
        raise RuntimeError(f'Only {len(peaks)} octahedral LCAO proposals for {count} necessary centers')
    xyz=octas.positions().reshape(-1,3)
    tree,images=periodic_point_tree((xyz@octas.inv)%1,octas.cell)
    dist,idx=tree.query(peaks@octas.cell,k=min(len(images),18))
    if dist.ndim==1:
        dist=dist[:,None];idx=idx[:,None]
    labels=(idx%(6*n))//6
    armidx=idx%(6*n)
    rankings=[]
    for j in range(len(peaks)):
        seen=set();chosen=[]
        for d,ti_id,arm_id in zip(dist[j],labels[j],armidx[j]):
            # Same crystallographic Ti basis site can contribute different
            # periodic Ti images to the OTi3 triangle (rutile has only 2 Ti).
            # We only forbid reusing the identical octahedral *vertex*.
            if int(arm_id) in seen:
                continue
            chosen.append((float(d),int(arm_id)))
            seen.add(int(arm_id))
            if len(chosen)==3:
                break
        if len(chosen)<3:
            continue
        # Allow initially imperfect octahedra; progressively reduce their
        # coincidence distortion through coherent SCF rotations/distortions.
        score=sum(min(v[0],1.8)**2 for v in chosen)
        rankings.append((score,j,tuple(v[1] for v in chosen)))
    rankings.sort()
    selected=[];used=np.zeros(6*n,dtype=float)
    for relax in (1.0,.7,.4,0.0):
        for base,j,arms in rankings:
            if j in selected:
                continue
            if selected:
                d=minimum_image_delta(peaks[j]@octas.cell,peaks[selected]@octas.cell,
                                      octas.cell,octas.inv)
                if np.min(np.linalg.norm(d,axis=1))<max(.70,oo_guard*.72)*relax:
                    continue
            penalty=sum(used[a] for a in arms)
            if relax>0 and penalty>2.1:
                continue
            selected.append(j)
            for a in arms:
                used[a]+=1
            if len(selected)==count:
                break
        if len(selected)==count:
            break
    if len(selected)!=count:
        raise RuntimeError(f'LCAO found {len(selected)} separated OTi3 centers, require {count}')
    return peaks[selected]@octas.cell

def initialize_centers(octas,peaks,oo_guard):
    """Initialize the OTi3 sharing field without constraining TiO6 entrances.

    Compare (a) the learned OTi3 Ti-triangle proposal field, when available,
    and (b) the coherent TiO6-derived LCAO proposal field. Their centers are
    NOT dynamic oxygen parameters: only complete octahedra are updated in SCF.
    """
    choices=[]
    if len(peaks):choices.append(('Ti-only OTi3',np.asarray(peaks)))
    choices.append(('TiO6-derived LCAO',coherent_lcao_proposal_field(octas,peaks)))
    ranked=[];errors=[]
    for kind,candidates in choices:
        try:
            centers=_select_provisional_centers(octas,candidates,oo_guard)
            _,derived,rms,bad=assign_provisional_network(octas,centers)
            ranked.append((bad,rms,centers,kind))
        except RuntimeError as exc:
            errors.append(f'{kind}: {exc}')
    if not ranked:
        raise RuntimeError('No LCAO provisional-center initialization: '+'; '.join(errors))
    ranked.sort(key=lambda x:(x[0],x[1]))
    return ranked[0][2]

def assign_provisional_network(octas,centers,max_repairs=8):
    """Polynomial-capacity assignment, NOT exhaustive 3-vertex handshaking.

    There are 2N candidate O sites with three slots each. One matching of
    6N octahedral vertices to these slots is computed by a cost-minimizing
    bipartite assignment. Duplicate Ti at a site is discouraged iteratively;
    nonunique-sharing violations are counted for final diagnostics, not rejected.
    """
    arms=octas.positions().reshape(-1,3)
    n=len(octas.ti);nO=len(centers)
    dif=minimum_image_delta(arms[:,None,:],centers[None,:,:],octas.cell,octas.inv)
    dist2=np.einsum('ijk,ijk->ij',dif,dif)
    cost=np.repeat(dist2,3,axis=1)
    ar,seat=linear_sum_assignment(cost)
    trip=np.empty(len(arms),dtype=int)
    trip[ar]=seat//3
    # Distinct periodic Ti *images*, not distinct primitive-cell Ti indices.
    # In rutile with only 2 basis Ti sites, oxygen still has three Ti neighbors.
    # (Two can be periodic images of the same basis Ti.)
    unwrapped=centers[trip]+dif[np.arange(len(arms)),trip]
    period_offset=np.rint((arms-unwrapped)@octas.inv).astype(int)
    bad_groups=0
    for g in range(nO):
        ids=np.flatnonzero(trip==g)
        keys=[(int(a//6),*map(int,period_offset[a])) for a in ids]
        if len(set(keys))!=3:
            bad_groups+=1
    duplicate_groups=bad_groups
    # Reconstruct centers exclusively from whole-octahedron vertex predictions.
    new_unwrapped=np.stack([np.mean(unwrapped[trip==g],axis=0) for g in range(nO)])
    mismatch=float(np.sqrt(np.mean(np.sum((unwrapped-new_unwrapped[trip])**2,axis=1))))
    new=wrap_cart(new_unwrapped,octas.cell,octas.inv)
    return trip,new,mismatch,duplicate_groups

def symmetry_mappings(ti,cell,symprec=.055):
    """Symmetry from Ti and lattice alone; O never enters symmetry inference."""
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError('spglib is needed for symmetry-adapted octahedra') from exc
    data=spglib.get_symmetry((np.asarray(cell),np.asarray(ti),[22]*len(ti)),
                             symprec=symprec)
    if data is None:
        raise RuntimeError('spglib cannot identify Ti-only framework symmetry')
    inv=np.linalg.inv(cell)
    result=[]
    for R,t in zip(data['rotations'],data['translations']):
        ti2=(np.asarray(ti)@R.T+t)%1
        delta=minimum_image_delta(ti2[:,None,:]@cell,ti[None,:,:]@cell,cell,inv)
        distances=np.linalg.norm(delta,axis=-1)
        mapping=distances.argmin(axis=1)
        if np.max(distances[np.arange(len(ti)),mapping])>symprec or len(set(mapping))!=len(ti):
            continue
        mat=inv@R.T@cell
        # row-vector Cartesian action x' = x @ mat
        result.append((np.asarray(mapping,dtype=int),mat,R,t))
    if not result:
        raise RuntimeError('No Ti-preserving symmetry operation (including identity)')
    return result

def octahedral_symmetry_mismatch(octas,operation):
    mapping,mat,_,_=operation
    points=octas.positions()-octas.ti[:,None,:]@octas.cell
    per=[]
    for i,j in enumerate(mapping):
        transformed=points[i]@mat
        d=np.linalg.norm(transformed[:,None]-points[j][None,:,:],axis=-1)
        a,b=linear_sum_assignment(d*d)
        per.append(np.mean(d[a,b]**2))
    return float(np.sqrt(np.mean(per)))

def active_symmetry(octas,operations,tolerance=.19):
    """Select Ti-only symmetry operations already compatible with whole TiO6.

    This deliberately permits oxygen symmetry to be lower than Ti-only SG.
    Incompatible operations are NEVER imposed as spurious O Wyckoff orbits.
    """
    return [op for op in operations if octahedral_symmetry_mismatch(octas,op)<tolerance]

def symmetry_feedback(octas,operations,targets,weight=.12):
    if not operations or weight<=0:
        return targets
    proposed=octas.positions()-octas.ti[:,None,:]@octas.cell
    accum=np.zeros_like(proposed)
    counts=np.zeros((len(proposed),6,1))
    for mapping,mat,_,_ in operations:
        for i,j in enumerate(mapping):
            transformed=proposed[i]@mat
            d=np.linalg.norm(transformed[:,None]-proposed[j][None,:,:],axis=-1)
            aa,bb=linear_sum_assignment(d*d)
            if np.mean(d[aa,bb]**2)>.10:
                continue
            accum[j,bb]+=transformed[aa]
            counts[j,bb]+=1
    mean=np.divide(accum,counts,out=proposed.copy(),where=counts>0)
    return (1-weight)*targets+weight*mean

def adapt_complete_octahedra(octas,ops,tolerance=.17,iterations=3):
    """Symmetry-adapt the *six-vertex objects* before any OTi3 SCF update.

    Only whole-TiO6 compatible Ti operations are projected. Imposing the
    full Ti-only SG could force wrong oxygen chemistry, as SG225 demonstrates.
    This changes octahedral rotation/strain, not individual O coordinates.
    """
    compatible=active_symmetry(octas,ops,tolerance)
    if len(compatible)<=1:
        return compatible
    for _ in range(iterations):
        cart=octas.positions()-octas.ti[:,None,:]@octas.cell
        targets=symmetry_feedback(octas,compatible,cart,weight=.70)
        octas.fit(targets,mode_options=None,mix=.65,distortion_max=.11)
        compatible=active_symmetry(octas,ops,tolerance)
        if len(compatible)<=1:
            break
    return compatible

def scf(octas,peaks,chem,mode_choices,ops,steps=35,inner_refit=2,
        symmetry_tol=.17,anneal_every=0,seed=0,verbose=False):
    """Alternating LCAO proposals and coherent octahedron-only SCF updates.

    No Cartesian O gradients, O Wyckoff variables, or independently evolving
    oxygen positions. Oxygen centers are *derived* from octahedron vertices.
    """
    rng=np.random.default_rng(seed)
    centers=initialize_centers(octas,peaks,chem.guard)
    scores=[];best=None
    v50_scf=_V50SCFConvergence(_V50_CONFIG) if _v50_enabled() else None
    v50_success=None
    v50_scf_reason='budget'
    symmetry=adapt_complete_octahedra(octas,ops,symmetry_tol)
    for step in range(steps):
        trip,derived,rms,duplicate=assign_provisional_network(octas,centers)
        # LCAO: provisional sites are the least-square coincidence positions of
        # three octahedral vertices. They are never independent optimization DOFs.
        change=minimum_image_delta(derived,centers,octas.cell,octas.inv)
        centers=wrap_cart(centers+.78*change,octas.cell,octas.inv)
        arm=octas.positions().reshape(-1,3)
        delta=minimum_image_delta(centers[trip],arm,octas.cell,octas.inv)
        # Target bond vectors are obtained by OTi3 coincidence projection;
        # only octahedral rotations and bounded shape modes respond to them.
        ti_center=np.repeat(octas.ti@octas.cell,6,axis=0)
        goals=(arm+delta-ti_center).reshape(len(octas.ti),6,3)
        sym_targets=symmetry_feedback(octas,symmetry,goals)
        octas.fit(sym_targets,mode_options=mode_choices if step%5==0 else None,
                  mix=.56,distortion_max=.11,distortion_ridge=1.5)
        if anneal_every and (step+1)%anneal_every==0 and step+1<steps:
            # Collective perturbation of entire TiO6 orientations, never O.
            angle=.075*(1-(step+1)/steps)
            for i in range(len(octas.ti)):
                octas.rot[i]=(Rotation.from_rotvec(rng.normal(size=3)*angle)*octas.rot[i])
        if step%6==5:
            symmetry=active_symmetry(octas,ops,symmetry_tol)
        trip,centers2,rms,duplicate=assign_provisional_network(octas,centers)
        scores.append((rms,duplicate,octas.distortion(),len(symmetry)))
        if best is None or (duplicate,rms)<(best[0],best[1]):
            best=(duplicate,rms,octas.copy(),centers2.copy(),trip.copy(),step)
        if v50_scf is not None:
            should_stop,reason=v50_scf.check(octas,centers2,trip,rms,duplicate,chem,step+1)
            if should_stop:
                v50_scf_reason=reason
                if reason=='complete':
                    v50_success=(octas.copy(),centers2.copy(),trip.copy(),step)
                break
        if verbose:
            print(f'    SCF cycle {step+1:03d} rms={rms:.4f} A '
                  f'bad_O_triplets={duplicate} oct_strain={octas.distortion():.4f} '
                  f'compatible_Ti_symops={len(symmetry)}/{len(ops)}',flush=True)
        if _v49_oxygen_motion('SCF',(centers2@octas.inv)%1.,octas.cell,
                              step+1,topology=trip,rank=(int(duplicate),float(rms))):
            break
        # Small coincidence error alone does NOT certify learned O geometry.
        # Continue SCF: only the final full-channel evaluator can certify OTi3.
    # Score both current and best snapshots via learned chemistry metrics.
    final=[]
    for key,oo,cen,tr,last_step in [
        ('best_coincidence',best[2],best[3],best[4],best[5]),
        ('last_cycle',octas,centers2,trip,step)]:
        current_trip,rebuilt,rms,bad=assign_provisional_network(oo,cen)
        o_frac=(rebuilt@oo.inv)%1
        with torch.no_grad():
            detail=chem.score(torch.tensor(o_frac,dtype=DTYPE),details=True)
        no_overlap=(detail['collision']<1e-8)
        # Old combined strict criterion retained as a diagnostic only.
        # It is NOT used to rank, stop or reject SCF trajectories.
        valid=(bad==0 and rms<.12 and no_overlap and
               detail['badTi']==0 and detail['badO']==0 and
               np.all(detail['hardTi']==6) and np.all(detail['hardO']==3))
        final.append(dict(name=key,oct=oo,oxygen=o_frac,trip=current_trip,
                          rms=rms,bad_triplets=bad,details=detail,
                          pass_chemistry=valid,cycle=last_step+1))
    if v50_success is not None:
        oo,cen,tr,cycle=v50_success
        new_trip,rebuilt,rms,bad=assign_provisional_network(oo,cen)
        oxygen=(rebuilt@oo.inv)%1.
        with torch.no_grad():
            detail=chem.score(torch.tensor(oxygen,dtype=DTYPE),details=True)
        complete_candidate=dict(name='v50_early_complete',oct=oo,oxygen=oxygen,
                                trip=new_trip,rms=rms,bad_triplets=bad,
                                details=detail,pass_chemistry=False,cycle=cycle+1)
        complete_candidate['pass_chemistry']=_v50_checkpoint_complete(complete_candidate)
        final.append(complete_candidate)
    if v50_scf is not None:
        # A chemically complete coherent candidate always outranks an incomplete one.
        final.sort(key=lambda q:(not _v50_checkpoint_complete(q),
                                 continuous_candidate_objective(q)))
    else:
        final.sort(key=continuous_candidate_objective)
    winner=final[0]
    if v50_scf is not None:
        winner['v50_scf_stop_reason']=v50_scf_reason
        winner['v50_scf_checks']=v50_scf.chem_checks
        _V50_COUNTERS['scf_cycles']+=len(scores)
        _V50_COUNTERS['scf_stops_'+v50_scf_reason]+=1
    winner['cycles']=len(scores)
    winner['entrance_rms']=scores[0][0]
    winner['best_rms']=best[1]
    winner['compatible_symops']=len(active_symmetry(winner['oct'],ops,symmetry_tol))
    winner['ti_symops']=len(ops)
    final_compatible=active_symmetry(winner['oct'],ops,symmetry_tol)
    winner['symmetry_max_rms']=max(
        (octahedral_symmetry_mismatch(winner['oct'],op) for op in final_compatible),
        default=0.0)
    winner['history']=scores
    return winner

def _rotation_perturbation(omega):
    """Differentiable SO(3) rotations: one collective orientation per TiO6."""
    x,y,z=omega.unbind(dim=-1)
    zero=torch.zeros_like(x)
    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(-1,3,3)
    norm2=(omega**2).sum(dim=1,keepdim=True)+1e-12
    theta=torch.sqrt(norm2)
    a=(torch.sin(theta)/theta).reshape(-1,1,1)
    b=((1-torch.cos(theta))/norm2).reshape(-1,1,1)
    return torch.eye(3,dtype=DTYPE).unsqueeze(0)+a*K+b*(K@K)

def coherent_chemistry_feedback(result,chem,ops,steps=64,rounds=2,
                                strain_limit=.11,accept=.10):
    """Correct full-channel OTi3 geometry through *only whole TiO6 DOFs*.

    Unlike v13, O-centered outer and cross-neighbor geometry generates SCF
    feedback. The optimizer knows NO reference/test oxygen positions.
    Each O position is the mean of its three CURRENT octahedral arms under
    an LCAO sharing assignment, and is never an optimizable tensor.
    The assignment is recomputed between SCF rounds, rather than globally
    enumerating an exponentially large handshaking network.
    """
    if steps<=0 or rounds<=0:
        result['chem_feedback_steps']=0
        return result
    v50_feedback=_V50FeedbackConvergence(_V50_CONFIG) if _v50_enabled() else None
    if v50_feedback is not None and _v50_checkpoint_complete(result):
        result['chem_feedback_steps']=0
        result['v50_feedback_stop_reason']='already_complete'
        _V50_COUNTERS['feedback_stops_already_complete']+=1
        return result
    working=result['oct'].copy()
    strain_limit=min(float(strain_limit),float(getattr(working,'_ablation_max_strain',strain_limit)))
    cell=working.cell;inv=working.inv;n=len(working.ti)
    one=torch.eye(3,dtype=DTYPE)
    ci=torch.tensor(cell,dtype=DTYPE)
    ci_inv=torch.tensor(inv,dtype=DTYPE)
    ti_cart=torch.tensor(working.ti@cell,dtype=DTYPE)
    best=result.copy()
    current_centers=result['oxygen']@cell
    total_done=0
    motion_stopped=False
    v50_feedback_reason='budget'
    # No categorical topology gates or binary local PASS flags in ranking.
    def rank(d):
        return continuous_candidate_objective(d)
    def evaluate(oo,centers):
        tr,o_cart,rms,bad=assign_provisional_network(oo,centers)
        oxygen=(o_cart@oo.inv)%1.0
        with torch.no_grad():
            detail=chem.score(torch.tensor(oxygen,dtype=DTYPE),details=True)
        valid=(bad==0 and rms<.12 and detail['collision']<1e-8 and
               detail['badTi']==0 and detail['badO']==0 and
               np.all(detail['hardTi']==6) and np.all(detail['hardO']==3))
        return dict(oct=oo.copy(),oxygen=oxygen,trip=tr.copy(),
                    rms=rms,bad_triplets=bad,details=detail,
                    pass_chemistry=bool(valid))
    for round_id in range(rounds):
        trip,cen,rms,bad=assign_provisional_network(working,current_centers)
        # Keep optimizing imperfect sharing assignments; final diagnostics
        # will quantify nonunique periodic-Ti triplets and CN deviations.
        current_centers=cen
        arms0=working.positions().reshape(-1,3)
        # Frozen periodic unwrapping labels within each differentiable round.
        # Changing the sharing pattern is handled in the next SCF round.
        nearby=minimum_image_delta(arms0,cen[trip],cell,inv)
        offsets=np.rint((arms0-nearby-cen[trip])@inv).astype(int)@cell
        offs=torch.tensor(offsets,dtype=DTYPE)
        group=torch.tensor(trip,dtype=torch.long)
        nO=len(cen)
        baserot=torch.tensor(np.stack([r.as_matrix().T for r in working.rot]),dtype=DTYPE)
        source=torch.tensor(np.stack([working.lib['Ti'][int(m)]['rigid_arms']
                                      for m in working.mode]),dtype=DTYPE)
        stretch_base=torch.tensor(working.stretch,dtype=DTYPE)
        angle=torch.nn.Parameter(torch.zeros(n,3,dtype=DTYPE))
        strains=torch.nn.Parameter(torch.zeros(n,6,dtype=DTYPE))
        optimizer=torch.optim.Adam((angle,strains),lr=.025)
        round_best=None
        round_best_rank=None
        for step in range(steps):
            optimizer.zero_grad()
            # Bounded six-parameter symmetric strain: all six vertices move
            # together, preserving the learned octahedron mode.
            sv=(min(.065,strain_limit) if strain_limit>0 else 0.0)*torch.tanh(strains)
            E=torch.zeros((n,3,3),dtype=DTYPE)
            E[:,0,0]=sv[:,0];E[:,1,1]=sv[:,1];E[:,2,2]=sv[:,2]
            E[:,0,1]=E[:,1,0]=sv[:,3]
            E[:,0,2]=E[:,2,0]=sv[:,4]
            E[:,1,2]=E[:,2,1]=sv[:,5]
            D=stretch_base+E
            rot=_rotation_perturbation(angle)
            arm=ti_cart[:,None,:]+(source@D)@baserot@rot.transpose(1,2)
            arm_flat=arm.reshape(-1,3)
            # Oxygen arises ONLY from coherent TiO6 geometry, never from
            # an independent coordinate or a Wyckoff orbit parameter.
            sums=torch.zeros((nO,3),dtype=DTYPE).index_add(0,group,arm_flat-offs)
            oxygen_cart=sums/3.
            shared_cart=oxygen_cart[group]+offs
            co_error=((arm_flat-shared_cart)**2).sum(dim=1).mean()
            oxygen_frac=torch.remainder(oxygen_cart@ci_inv,1.)
            chemical,local_ti,local_o,res_ti,res_o,ti_lim,o_lim=chem.score(oxygen_frac,components=True)
            # Learned geometric tolerance produces a smooth, differentiable
            # corrective force *before* final validation; not a relaxation
            # of either TiO6 or OTi3 final criteria.
            exceed_ti=torch.relu(res_ti-ti_lim)
            exceed_o=torch.relu(res_o-o_lim)*chem.ablation_oxygen_weight
            if chem.hierarchy=='v20':
                worst_geo=(exceed_ti.square().sum(dim=1).mean()+
                           2.*exceed_o.square().sum(dim=1).mean())
                conn_weight=1.5
            else:
                # Full Ti geometry is strict. O primary Ti3 shape is weak,
                # extended O shell geometry weaker still. O CN3 and physical
                # sharing remain strongly represented in `chemical` and co_error.
                worst_geo=(exceed_ti.square().sum(dim=1).mean()+
                           .18*exceed_o[:,0].square().mean()+
                           .025*exceed_o[:,1:].square().sum(dim=1).mean())
                conn_weight=4.0
            strain_reg=(D-one).square().sum(dim=(1,2)).mean()
            angle_reg=angle.square().sum(dim=1).mean()
            loss=chemical+5.*worst_geo+conn_weight*co_error+.035*strain_reg+.002*angle_reg
            if not torch.isfinite(loss):break
            loss.backward()
            torch.nn.utils.clip_grad_norm_((angle,strains),max_norm=4.)
            optimizer.step()
            total_done+=1
            # Unconditionally check final chemistry; optimizing a scalar
            # alone does not determine topology/geometry quality. Save in memory.
            if (step+1)%8==0 or step+1==steps:
                aa=angle.detach().cpu().numpy()
                dd=D.detach().cpu().numpy()
                trial=working.copy()
                for j in range(n):
                    trial.rot[j]=Rotation.from_rotvec(aa[j])*working.rot[j]
                    val,Q=np.linalg.eigh(dd[j])
                    trial.stretch[j]=(Q*np.clip(val,1-strain_limit,1+strain_limit))@Q.T
                trial_cen=oxygen_cart.detach().cpu().numpy()
                candidate=evaluate(trial,trial_cen)
                ordering=rank(candidate)
                if round_best is None or ordering<round_best_rank:
                    round_best=candidate
                    round_best_rank=ordering
                if ordering<rank(best):
                    best={**result,**candidate}
                if v50_feedback is not None:
                    finish,reason=v50_feedback.check(candidate,total_done)
                    if finish:
                        v50_feedback_reason=reason
                        motion_stopped=True
                        if reason=='complete':
                            # A completed checkpoint must be preserved even when
                            # its smooth scalar is marginally higher than the old best.
                            best={**result,**candidate}
                        break
                if _v49_oxygen_motion('feedback',candidate['oxygen'],cell,total_done,
                                      topology=candidate['trip'],rank=ordering):
                    motion_stopped=True
                    break
        if round_best is None:
            break
        # Rebuild the LCAO network for the next collective SCF update.
        working=round_best['oct'].copy()
        current_centers=round_best['oxygen']@cell
        if motion_stopped:break
    best['chem_feedback_steps']=total_done
    if v50_feedback is not None:
        best['v50_feedback_stop_reason']=v50_feedback_reason
        best['v50_feedback_checkpoints']=v50_feedback.checkpoints
        _V50_COUNTERS['feedback_steps']+=total_done
        _V50_COUNTERS['feedback_stops_'+v50_feedback_reason]+=1
    best['compatible_symops']=len(active_symmetry(best['oct'],ops,.17))
    best['symmetry_max_rms']=max(
        (octahedral_symmetry_mismatch(best['oct'],op)
         for op in active_symmetry(best['oct'],ops,.17)),default=0.)
    return best

def write_final_cif(ti,cell,o,target):
    from pymatgen.core import Structure
    crystal=Structure(cell,['Ti']*len(ti)+['O']*len(o),
                      np.vstack((ti,o)),coords_are_cartesian=False)
    crystal.to(filename=str(target),fmt='cif')

def screen_ti_geometry(ti,cell,b):
    cart=np.asarray(ti)@np.asarray(cell)
    images=(np.asarray(ti)[None,:,:]+SHIFTS[:,None,:]).reshape(-1,3)@cell
    d=np.linalg.norm(cart[:,None,:]-images[None,:,:],axis=-1)
    # Mask ONLY each Ti atom's own zero-translation image.  Blanket masking
    # d < 1e-6 would incorrectly accept two distinct coincident Ti atoms.
    zero_shift=np.flatnonzero(np.all(SHIFTS==0,axis=1))
    if len(zero_shift)!=1:raise RuntimeError('Periodic Ti image grid lacks unique origin')
    idx=np.arange(len(ti))
    d[idx,int(zero_shift[0])*len(ti)+idx]=np.inf
    m=float(d.min())
    if not np.isfinite(m) or m<b.bf_physical_min_TiTi_A:
        return False,f'Ti--Ti short contact {m:.3f} A'
    if np.linalg.det(cell)<=1e-8:return False,'bad cell volume'
    return True,f'Ti--Ti min {m:.3f} A'

class GeometryPolicy:
    """Frozen tolerances for ONE target, built without that target's O sites."""
    def __init__(self,name,base,global_limits,group_limits=None,group_support=None,
                 eligible_structures=0,eligible_atoms=None,reference_pull=.06):
        self.name=name
        self.base=np.asarray(base,float)
        self.global_limits={sp:np.asarray(global_limits[sp],float) for sp in ('Ti','O')}
        self.group_limits=dict(group_limits or {})
        self.group_support=dict(group_support or {})
        self.eligible_structures=eligible_structures
        self.eligible_atoms=eligible_atoms or {}
        self.reference_pull=float(reference_pull)
        self._cache={}

    def limits_np(self,species,mode_indices,library):
        idx=np.asarray(mode_indices,dtype=int).reshape(-1)
        if self.name=='v21_strict':
            return np.broadcast_to(self.base,(len(idx),4)).copy()
        if self.name=='group_channel_q95':
            return np.asarray([self.group_limits.get((species,tuple(library[species][int(m)]['counts'])),
                                        self.global_limits[species]) for m in idx],dtype=float)
        return np.broadcast_to(self.global_limits[species],(len(idx),4)).copy()

    def limit_tensor(self,species,mode_indices,library):
        # Mode indices have no differentiable dependency; threshold tensors
        # are constants. They never encode the held-out target's oxygen.
        ids=mode_indices.detach().cpu().numpy()
        return torch.as_tensor(self.limits_np(species,ids,library),dtype=DTYPE,
                               device=mode_indices.device)

def _balanced_quantile(values,weights,p):
    order=np.argsort(values)
    a=np.asarray(values)[order]; w=np.asarray(weights)[order]
    if len(a)==0:raise ValueError('no calibration observations')
    if len(a)==1:return float(a[0])
    cw=np.cumsum(w)/np.sum(w)
    # Weighted midpoint interpolation avoids overweighting large unit cells.
    positions=(np.cumsum(w)-.5*w)/np.sum(w)
    return float(np.interp(p,positions,a,left=a[0],right=a[-1]))

def collect_calibration_data(dataset,lib,bounds,guard,args):
    """Compute v21 four-channel residuals for original DB structures only.

    Exclude wrong-coordination environments from the respective species fit;
    the OTi2 failure modes remain in original topology table, not relabeled
    as valid OTi3. References are training positive controls, not calibrators.
    """
    env=[];counts={}
    for entry in dataset:
        if entry['is_reference']:continue
        st=entry['original']; label=entry['structure_id']
        t=sites(st,'Ti');o=sites(st,'O');c=np.asarray(st.lattice.matrix,float)
        chem=Chemistry(t,c,lib,bounds,guard,accept=args.accept_A,hierarchy=args.hierarchy)
        with torch.no_grad(): d=chem.score(torch.tensor(o,dtype=DTYPE),details=True)
        counts[label]={}
        for sp,expected,rkey,mkey,ckey in (
            ('Ti',6,'rTi','modeTi','hardTi'),('O',3,'rO','modeO','hardO')):
            residual=np.asarray(d[rkey]); mids=np.asarray(d[mkey]);cn=np.asarray(d[ckey])
            counts[label][sp]=int(np.sum(cn==expected))
            for j,(rr,mid,cc) in enumerate(zip(residual,mids,cn)):
                env.append(dict(structure_id=label,species=sp,site=j,
                    cn=int(cc),expected=expected,cn_ok=bool(cc==expected),
                    mode_id=int(mid),group=tuple(lib[sp][int(mid)]['counts']),
                    residuals=np.asarray(rr,float)))
    return env,counts

def make_geometry_policy(calibration_env,lib,exclude,args):
    """Leave-one-TiO2-structure-out fit; no target-specific oxygen labels.

    v22 keeps the four-reference lower floor [a,2a,2a,2a], takes the
    structure-balanced per-channel 95th percentiles of other database CIFs,
    and optionally conditions on learned mode shell-count groups. Sparse
    groups inherit global thresholds. Fit cohort is explicit supervision.
    """
    base=np.asarray([args.accept_A,2*args.accept_A,2*args.accept_A,2*args.accept_A],float)
    limits={}; group_limits={}; support={}; atoms={}
    excluded=[x for x in calibration_env if x['structure_id']!=exclude and x['cn_ok']]
    structures=sorted({x['structure_id'] for x in excluded})
    for sp in ('Ti','O'):
        sub=[x for x in excluded if x['species']==sp]
        atoms[sp]=len(sub)
        def fit(observations):
            if len({x['structure_id'] for x in observations}) < args.calibration_min_structures:
                return base.copy()
            multiplicities=defaultdict(int)
            for item in observations:multiplicities[item['structure_id']]+=1
            weights=np.asarray([1./multiplicities[x['structure_id']] for x in observations])
            res=np.asarray([x['residuals'] for x in observations])
            return np.maximum(base,[ _balanced_quantile(res[:,k],weights,args.calibration_quantile)
                                     for k in range(4)])
        limits[sp]=fit(sub)
        groups={tuple(m['counts']) for m in lib[sp]}
        for grp in groups:
            gsub=[x for x in sub if x['group']==grp]
            ng=len({x['structure_id'] for x in gsub})
            eligible=(len(gsub)>=args.calibration_group_min_atoms and
                      ng>=args.calibration_group_min_structures)
            group_limits[(sp,grp)]=fit(gsub) if eligible else limits[sp]
            support[(sp,grp)]=(len(gsub),ng,'group' if eligible else 'fallback_global')
    return GeometryPolicy(args.geometry_policy,base,limits,group_limits,support,
            eligible_structures=len(structures),eligible_atoms=atoms,
            reference_pull=args.geometry_reference_pull)

def global_basin_signature(candidate):
    """Canonical discrete sharing topology and learned modes, independent of O labels."""
    trip=np.asarray(candidate['trip'],int)
    nO=len(candidate['oxygen'])
    sharing=tuple(sorted(tuple(sorted(np.flatnonzero(trip==j).tolist())) for j in range(nO)))
    return (tuple(np.asarray(candidate['oct'].mode,int).tolist()),sharing)

def global_early_quality(candidate,threshold,primary_limit):
    """v22 calibrated EARLY STOP; never use target oxygen or uncalibrated Ti limits.

    Must satisfy all Ti6/O3 coordination, triplet uniqueness, no collisions and
    strong sharing, in addition to calibrated Ti environment acceptance. It
    remains a heuristic, not a statement of physical validity.
    """
    d=candidate['details']
    if d.get('geometry_policy')=='v21_strict':
        if d.get('hierarchy')=='v20':
            return bool(candidate['rms']<threshold and d['collision']<1e-8 and
                        np.all(np.asarray(d['rTi'])[:,0]<=primary_limit) and
                        np.all(np.asarray(d['rO'])[:,0]<=primary_limit))
        rr=np.asarray(d['rTi'])
        return bool(candidate['rms']<threshold and d['collision']<1e-8 and
                    candidate['bad_triplets']==0 and
                    np.all(rr[:,0]<=primary_limit) and
                    np.all(rr<=np.array([primary_limit,2*primary_limit,
                                          2*primary_limit,2*primary_limit])))
    return bool(candidate['rms']<threshold and candidate['bad_triplets']==0
          and d['collision']<1e-8 and np.all(d['topologyTi'])
          and np.all(d['topologyO']) and np.all(d['geometryTi']))

def _global_orbit(octas,ops,index):
    """Use only Ti symmetries already compatible with the current oxygen entrance.

    Operations constrain the proposed rotational moves. The inherited v19 SCF
    and chemistry-feedback stages are not exact group-projected solvers.
    """
    active=active_symmetry(octas,ops,.15)
    orbit={index:np.eye(3)}
    for mapping,mat,_,_ in active:
        j=int(mapping[index])
        if j not in orbit:orbit[j]=np.asarray(mat,float)
    return orbit

def global_octahedral_move(octas,ops,proposals,mode_options,rng,amplitude,
                            force_escape=False):
    """Propose coordinated whole TiO6 updates, never individual oxygen moves.

    Four move types: orbit, neighboring-cluster, mode jump, fresh-orientation
    restart. For symmetry-equivalent sites rotation conjugation includes mirrors.
    The local optimizer may later relax/break that approximate symmetry.
    """
    trial=octas.copy();n=len(trial.ti)
    if n==0:raise RuntimeError('No Ti sites')
    kind=int(rng.integers(0,4))
    if force_escape:kind=int(rng.choice([1,2,3]))
    i=int(rng.integers(n))
    moved=set()
    if kind==0:    # symmetry orbit
        orbit=_global_orbit(trial,ops,i)
    elif kind==1:  # local cluster plus its symmetry partners
        cart=trial.ti@trial.cell
        delta=minimum_image_delta(cart,cart[i],trial.cell,trial.inv)
        ds=np.linalg.norm(delta,axis=1)
        k=min(n,max(2,int(np.ceil(np.sqrt(n)))))
        orbit={}
        for j in np.argsort(ds)[:k]:
            orbit.update(_global_orbit(trial,ops,int(j)))
    else:
        orbit=_global_orbit(trial,ops,i)
    # A world-frame rotation is transported to symmetry-related octahedra.
    # Reflections conjugate proper rotations to proper rotations.
    theta=rng.normal(size=3)
    theta*=amplitude/max(float(np.linalg.norm(theta)),1e-12)
    base=Rotation.from_rotvec(theta)
    for j,mat in orbit.items():
        transformed=mat.T@base.as_matrix()@mat
        trial.rot[j]=Rotation.from_matrix(transformed)*trial.rot[j]
        moved.add(j)
    if kind in (1,2) and rng.random()<.6:
        # Small coherent symmetric strain perturbation. No vertex-level DOFs.
        h=rng.normal(size=(3,3));h=.5*(h+h.T)
        h*=min(.07,amplitude*.09)/max(np.linalg.norm(h),1e-12)
        for j in moved:
            D=trial.stretch[j]+h
            vals,Q=np.linalg.eigh(D)
            trial.stretch[j]=(Q*np.clip(vals,.89,1.11))@Q.T
    if kind==2:  # complete learned geometry mode switch, not independent bonds
        for j in moved:
            possibilities=[int(k) for k,_ in mode_options[j]
                           if int(k)!=int(trial.mode[j])]
            if possibilities:
                trial.mode[j]=int(rng.choice(possibilities))
                trial.stretch[j]=np.eye(3)
    if kind==3:  # large orientation entrance jump for an entire orbit
        for j in moved:
            candidates=proposals[j]
            if candidates:
                _,mode,rot=candidates[int(rng.integers(len(candidates)))]
                trial.mode[j]=int(mode)
                trial.rot[j]=rot
                trial.stretch[j]=np.eye(3)
    return trial,('orbit','cluster','mode','reseed')[kind]

def _global_temperature(initial_energy,step,limit,kind,replica=0,replicas=1):
    """Algorithmic temperatures in units of the common continuous score."""
    base=max(.015,.12*max(abs(float(initial_energy)),.1))
    if kind=='anneal':
        frac=min(1.,step/max(1,limit-1))
        return base*(.025**frac)
    if kind=='hopping':
        return base*.18
    if kind=='replica':
        return base*(.06*(35.**(replica/max(1,replicas-1))))
    raise ValueError(kind)

def _global_scored_run(octas,peaks,chem,options,ops,args,seed,cycles):
    """Fresh LCAO update AFTER EVERY global proposal, before scoring.

    Calls original v19 SCF, including initialize_centers and reassignment.
    """
    return scf(octas,peaks,chem,options,ops,steps=cycles,
               symmetry_tol=args.symmetry_tol,anneal_every=0,
               seed=seed,verbose=False)

def _global_checkpoint(history,step,best,started,basins,escapes,attempts):
    history.append(dict(proposals=step,elapsed=time.perf_counter()-started,
                        objective=continuous_candidate_objective(best),
                        rms=float(best['rms']),geometry_bad=int(best['details']['geometryBadTi']+
                                          best['details']['geometryBadO']),
                        basins=len(basins),escapes=escapes,attempts=attempts))

def run_global_method(kind,ti,cell,lib,proposals,options,peaks,chem,ops,args,seed):
    """Annealing, basin hopping, or replica-exchange basin hopping.

    `--global-steps` counts total attempted proposals ACROSS replicas.
    All methods use the identical learned model, global moves, and SCF.
    No per-proposal gradient chemistry feedback (only final best refinement).
    """
    started=time.perf_counter()
    rng=np.random.default_rng(seed)
    replicas=args.replicas if kind=='replica' else 1
    ninitial=replicas
    search_cycles=args.anneal_scf_cycles if kind=='anneal' else args.global_scf_cycles
    current=[];energies=[];basins=set();best=None
    counters=dict(attempts=0,accepted=0,improvements=0,escapes=0,
                  swaps_attempted=0,swaps_accepted=0,stalled_intervals=0,
                  fresh_lcao_calls=0,scf_cycles=0,basin_revisits=0,
                  move_attempts=defaultdict(int),move_accepted=defaultdict(int))
    for ri in range(ninitial):
        octa=Octahedra(ti,cell,lib,proposals,
                       seed=0 if ri==0 else seed+101*(ri+1))
        ans=_global_scored_run(octa,peaks,chem,options,ops,args,seed+ri,
                               search_cycles)
        counters['fresh_lcao_calls']+=1
        counters['scf_cycles']+=int(ans['cycles'])
        current.append(ans);energies.append(continuous_candidate_objective(ans))
        basins.add(global_basin_signature(ans))
        if best is None or energies[-1]<continuous_candidate_objective(best):
            best=ans
    reference_energy=max(energies)
    initial_candidate=best
    initial_seconds=time.perf_counter()-started
    initial_best=continuous_candidate_objective(best)
    last_new_best=0
    last_new_basin=0
    good_at=0 if global_early_quality(best,args.early_sharing,args.early_primary_A) else None
    history=[]
    _global_checkpoint(history,0,best,started,basins,0,0)
    stop_reason='max_global_proposals'
    adaptive_scale=1.
    for step in range(1,args.global_steps+1):
        elapsed=time.perf_counter()-started
        if args.global_max_seconds>0 and elapsed>=args.global_max_seconds:
            stop_reason='time_budget';break
        if (good_at is not None and step-good_at>=args.early_confirm and
            global_early_quality(best,args.early_sharing,args.early_primary_A)):
            stop_reason='quality_converged';break
        # One proposal globally; RE total budget is not multiplied by replica count.
        rid=(step-1)%replicas if kind=='replica' else 0
        lag=step-last_new_best
        stuck=(lag>=args.stagnation_window)
        # Increase step scale following stagnation; allow whole-octahedron jumps.
        scale=adaptive_scale*(1.+.35*max(0,lag//max(args.stagnation_window,1)))
        scale=min(scale,3.)
        amp=args.global_rotation*scale
        if kind=='anneal':
            amp*=max(.35,1.-.65*step/max(args.global_steps,1))
        if stuck and step%args.stagnation_window==0:
            counters['stalled_intervals']+=1
        force_escape=stuck and (step%max(2,args.stagnation_window//3)==0)
        octa,move=global_octahedral_move(current[rid]['oct'],ops,proposals,
                           options,rng,amp,force_escape=force_escape)
        counters['attempts']+=1
        counters['move_attempts'][move]+=1
        before_sig=global_basin_signature(current[rid])
        counters['fresh_lcao_calls']+=1
        try:
            cand=_global_scored_run(octa,peaks,chem,options,ops,args,
                                    seed+step*79+rid,search_cycles)
            counters['scf_cycles']+=int(cand['cycles'])
            e=continuous_candidate_objective(cand)
            if not np.isfinite(e):
                continue
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            # A difficult global entrance is not a database runtime failure.
            continue
        signature=global_basin_signature(cand)
        if signature not in basins:
            basins.add(signature)
            # Ignore new but extremely poor basins for exhaustion detection:
            # only discovery of a competitive basin resets the patience clock.
            if e <= continuous_candidate_objective(best) + max(.025, .08*abs(continuous_candidate_objective(best))):
                last_new_basin=step
        else:
            counters['basin_revisits']+=1
        # Global best is independent of Metropolis acceptance; keep rejected improvements.
        if e+args.improvement_eps<continuous_candidate_objective(best):
            best=cand
            last_new_best=step
            counters['improvements']+=1
            if global_early_quality(best,args.early_sharing,args.early_primary_A):good_at=step
            else:good_at=None
        temperature=_global_temperature(reference_energy,step,args.global_steps,
                                         kind,rid,replicas)
        de=e-energies[rid]
        accept=(de<=0 or rng.random()<math.exp(-min(700.,max(0.,de)/temperature)))
        if accept:
            counters['accepted']+=1
            counters['move_accepted'][move]+=1
            if stuck and signature!=before_sig:counters['escapes']+=1
            current[rid]=cand;energies[rid]=e
        if kind=='replica' and replicas>1 and step%args.exchange_every==0:
            # Adjacent replica exchange with correct sign for temperature swap.
            # States themselves are already locally quenched (basin sampling).
            parity=(step//args.exchange_every)%2
            for a in range(parity,replicas-1,2):
                b=a+1
                ta=_global_temperature(reference_energy,step,args.global_steps,kind,a,replicas)
                tb=_global_temperature(reference_energy,step,args.global_steps,kind,b,replicas)
                log_alpha=(1./ta-1./tb)*(energies[a]-energies[b])
                counters['swaps_attempted']+=1
                if log_alpha>=0 or rng.random()<math.exp(max(-700.,log_alpha)):
                    current[a],current[b]=current[b],current[a]
                    energies[a],energies[b]=energies[b],energies[a]
                    counters['swaps_accepted']+=1
        if step%args.checkpoint_every==0 or step==args.global_steps:
            _global_checkpoint(history,step,best,started,basins,counters['escapes'],counters['attempts'])
            print(f'    {kind} global={step:4d} best={continuous_candidate_objective(best):.5g} '
                  f'sharing={best["rms"]:.4f} A distinct_basins={len(basins)} '
                  f'accept={counters["accepted"]}/{counters["attempts"]} '
                  f'elapsed={time.perf_counter()-started:.1f}s',flush=True)
        # Stop only after repeated windows without improvement AND without
        # encountering a new basin, never assert mathematical impossibility.
        if (step>=args.min_global_steps and step-last_new_best>=args.exhaustion_window
            and step-last_new_basin>=args.exhaustion_window):
            stop_reason='empirically_exhausted';break
    else:
        step=args.global_steps
    if not history or history[-1]['proposals']!=counters['attempts']:
        _global_checkpoint(history,counters['attempts'],best,started,basins,
                           counters['escapes'],counters['attempts'])
    # Publication-report checkpoints are the SAME run, not three reruns.
    global_candidate=best
    global_seconds=time.perf_counter()-started
    # Only the winning basin receives expensive differentiable refinement.
    feedback=coherent_chemistry_feedback(best,chem,ops,
                  steps=args.chemistry_feedback_steps,
                  rounds=args.chemistry_feedback_rounds,accept=args.accept_A)
    if continuous_candidate_objective(feedback)<continuous_candidate_objective(best):
        best=feedback
    counters['feedback_steps']=int(feedback.get('chem_feedback_steps',0))
    counters['distinct_basins']=len(basins)
    counters['stop_reason']=stop_reason
    counters['initial_objective']=float(initial_best)
    counters['objective']=float(continuous_candidate_objective(best))
    counters['total_seconds']=float(time.perf_counter()-started)
    counters['history']=history
    counters['entrances']=ninitial
    counters['phase_initial']=(initial_candidate,initial_seconds)
    counters['phase_global']=(global_candidate,global_seconds)
    counters['phase_final']=(best,counters['total_seconds'])
    return best,counters

def search_target(ti,cell,lib,bounds,guard,args,seed,calibration=None):
    """Prepare once, benchmark all selected global methods and v19 baseline."""
    preparation_started=time.perf_counter()
    # Strict-policy option retains the original v21 continuous geometry score.
    # Only calibrated policies activate channel-excess optimizer forces.
    active_calibration=(None if calibration is None or calibration.name=='v21_strict'
                        else calibration)
    chem=Chemistry(ti,cell,lib,bounds,guard,accept=args.accept_A,
                   hierarchy=getattr(args,"hierarchy","ti-first"),calibration=active_calibration,
                   target_o_cn=getattr(args,"target_o_cn",True))
    ops=symmetry_mappings(ti,cell,args.symprec)
    peaks=frac_candidates(ti,cell,lib,seed,cap=args.candidate_cap)
    options=mode_shortlist(ti,cell,lib,args.modes_per_ti)
    framework_seeds=framework_orientation_seeds(ti,cell,lib,options,peaks,
                                          proposals_per_ti=args.octahedral_proposals)
    triangle_seeds=triangle_orientation_seeds(ti,cell,peaks,lib,options,seed,
                                          proposals_per_ti=args.octahedral_proposals)
    proposals=[]
    for ti_id,(ff,tt) in enumerate(zip(framework_seeds,triangle_seeds)):
        combined=sorted(ff+tt,key=lambda x:x[0]);kept=[]
        for candidate in combined:
            _,mid,rot=candidate
            if any(m==mid and (old.inv()*rot).magnitude()<.18 for _,m,old in kept):
                continue
            kept.append(candidate)
            if len(kept)>=args.octahedral_proposals:break
        if not kept:raise RuntimeError(f'no learned TiO6 template for Ti{ti_id}')
        proposals.append(kept)
    print('  TiO6-FIRST entrance: '+str(len(proposals))+' complete octahedra; '
          +str(sum(len(z) for z in triangle_seeds))+' OTi3-field alignments; '
          +str(sum(len(z) for z in framework_seeds))+' Ti-only framework alignments',flush=True)
    preparation_seconds=time.perf_counter()-preparation_started
    outcomes={}
    if 'baseline' in args.methods:
        started=time.perf_counter();trials=[];raw_trials=[];updates=0
        raw_scf_seconds=0.; local_feedback_seconds=0.
        v50_scf_s=0.0;v50_feedback_s=0.0
        for k in range(args.starts):
            octa=Octahedra(ti,cell,lib,proposals,seed=0 if k==0 else seed+491*k)
            scf_started=time.perf_counter()
            ans=scf(octa,peaks,chem,options,ops,steps=args.cycles,
                    symmetry_tol=args.symmetry_tol,anneal_every=args.anneal_every,
                    seed=seed+k,verbose=args.verbose_cycles)
            v50_scf_s+=time.perf_counter()-scf_started
            raw_trials.append(ans)
            before_local=time.perf_counter()
            raw_scf_seconds+=before_local-started-raw_scf_seconds-local_feedback_seconds
            ans=coherent_chemistry_feedback(ans,chem,ops,
                    steps=args.chemistry_feedback_steps,
                    rounds=args.chemistry_feedback_rounds,accept=args.accept_A)
            trials.append(ans)
            feedback_seconds=time.perf_counter()-before_local
            local_feedback_seconds+=feedback_seconds
            v50_feedback_s+=feedback_seconds
            updates+=int(ans.get('chem_feedback_steps',0))
            if _v50_enabled():
                print(f'    v50 stages: SCF={ans.get("cycles","?")} '
                      f'({ans.get("v50_scf_stop_reason","budget")}) feedback='
                      f'{ans.get("chem_feedback_steps",0)} '
                      f'({ans.get("v50_feedback_stop_reason","budget")})',flush=True)
            print(f'    baseline entrance {k+1}/{args.starts}: objective='
                  f'{continuous_candidate_objective(ans):.6g} '
                  f'sharing={ans["rms"]:.5f} A',flush=True)
        winner=min(trials,key=continuous_candidate_objective)
        metadata=dict(attempts=0,accepted=0,improvements=0,escapes=0,swaps_attempted=0,
                      swaps_accepted=0,stalled_intervals=0,fresh_lcao_calls=args.starts,
                      scf_cycles=sum(int(x['cycles']) for x in trials),
                      feedback_steps=updates,entrances=args.starts,distinct_basins=len(
                          {global_basin_signature(x) for x in trials}),
                      initial_objective=continuous_candidate_objective(trials[0]),
                      objective=continuous_candidate_objective(winner),
                      stop_reason='fixed_v19_starts',total_seconds=time.perf_counter()-started,
                      history=[])
        initial_baseline=min(raw_trials,key=continuous_candidate_objective)
        metadata['phase_initial']=(initial_baseline,raw_scf_seconds)
        metadata['phase_global']=(initial_baseline,raw_scf_seconds)
        metadata['phase_final']=(winner,metadata['total_seconds'])
        metadata['baseline_local_seconds']=local_feedback_seconds
        if _v50_enabled():
            metadata['v50_stage_time_s']={'preparation':preparation_seconds,
                                         'scf':v50_scf_s,'feedback':v50_feedback_s}
            metadata['v50_stage_steps']={'scf':metadata['scf_cycles'],
                                         'feedback':metadata['feedback_steps']}
        outcomes['baseline']=(winner,metadata)
    for method in ('anneal','hopping','replica'):
        if method in args.methods:
            # Baseline budget matching uses an explicit cap without modifying
            # local SCF or introducing target oxygen into the search.
            previous_time=args.global_max_seconds
            if args.match_baseline_time and 'baseline' in outcomes:
                baseline_time=outcomes['baseline'][1]['total_seconds']
                args.global_max_seconds=min(previous_time,baseline_time) if previous_time>0 else baseline_time
            try:
                winner,metadata=run_global_method(method,ti,cell,lib,proposals,
                                      options,peaks,chem,ops,args,
                                      seed+{'anneal':1049,'hopping':2089,'replica':4099}[method])
            finally:
                args.global_max_seconds=previous_time
            outcomes[method]=(winner,metadata)
            print(f'    {method} FINISHED: global={metadata["attempts"]} '
                  f'basins={metadata["distinct_basins"]} '
                  f'escapes={metadata["escapes"]} '
                  f'objective={metadata["objective"]:.6g} '
                  f'runtime={metadata["total_seconds"]:.1f}s '
                  f'stop={metadata["stop_reason"]}',flush=True)
    for _,metadata in outcomes.values():
        metadata["shared_preparation_seconds"]=float(preparation_seconds)
    chosen_method=min(outcomes,key=lambda k:continuous_candidate_objective(outcomes[k][0]))
    winner=outcomes[chosen_method][0]
    winner['chosen_method']=chosen_method
    # Only method winners are kept. Avoid serializing full global trajectories.
    trials=[outcomes[m][0] for m in args.methods]
    return winner,trials,len(peaks),len(ops),outcomes

REF_NAMES=('anatase','brookite','hollandite','rutile')  # original v22r1 ordering

def load_local_chemistry(ref_dir,shell_max_k,accept_A=.10,hierarchy="ti-first"):
    """Exact v22 reference learning, reference order, and positive-mode checks."""
    paths=[Path(ref_dir)/f'{name}.cif' for name in REF_NAMES]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError('Missing original v22r1 reference files: '+', '.join(missing))
    with contextlib.redirect_stdout(io.StringIO()):
        library,bounds,guard,references=learn(paths,shell_max_k=shell_max_k)
    from torch import no_grad
    for name,st in zip(REF_NAMES,references):
        with contextlib.redirect_stdout(io.StringIO()):
            chem=Chemistry(sites(st,'Ti'),np.asarray(st.lattice.matrix,float),
                           library,bounds,guard,accept=accept_A,hierarchy=hierarchy)
            with no_grad():
                d=chem.score(torch.tensor(sites(st,'O'),dtype=DTYPE),details=True)
        if d['badTi'] or d['badO'] or d['collision']>1e-8:
            raise RuntimeError(f'Original v22 reference-mode calibration failed on {name}')
    for mode in library['Ti']:
        ref=references[REF_NAMES.index(mode['source'])]
        mode['rigid_arms']=reference_neighbors(ref,mode['index'],'O',6)[0]
        mode['framework_arms']=reference_neighbors(ref,mode['index'],'Ti',8)[0]
    summary={'Ti':len(library['Ti']),'O':len(library['O']),
             'cutTi':float(bounds['Ti'][0]),'cutO':float(bounds['O'][0]),
             'guard':float(guard),'names':list(REF_NAMES)}
    return library,bounds,guard,summary

def prepare_v22_calibration(cfg,library,bounds,guard,reference_summary):
    """Build original v22r1 geometry-policy cohort, once for generated targets.

    Copies original v22 database eligibility/structure matcher deduplication and
    uses UNCHANGED collect_calibration_data and make_geometry_policy formulas.
    Generated targets have no known O and are excluded by construction.
    No database oxygen ever enters the learned templates or LCAO proposals.
    """
    if cfg.geometry_policy=='v21_strict':
        cfg.v22_calibration=None
        print(' v22 geometry policy: v21_strict (explicitly selected; original raw objective)',flush=True)
        return {'policy':'v21_strict','calibration_structures':0,
                'Ti_eligible_atoms':0,'O_eligible_atoms':0}
    db=Path(cfg.db)
    if not db.is_file():
        raise FileNotFoundError('Original v22r1 channel_q95 calibration requires '
                                f'{db}; supply --db path or explicitly choose '
                                '--v22-geometry-policy v21_strict. '
                                'No silent calibration substitution is allowed.')
    from ase.db import connect
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    matcher=StructureMatcher(ltol=.01,stol=.03,angle_tol=1.,primitive_cell=True,
                             scale=False,attempt_supercell=True)
    seen=[];dataset=[]
    for name in REF_NAMES:
        ref=structure(Path(cfg.ref)/f'{name}.cif')
        seen.append((f'ref_{name}',ref))
        dataset.append(dict(structure_id=f'ref_{name}',is_reference=True,original=ref))
    n_screened=0;n_duplicate=0;n_invalid=0;n_errors=0
    with connect(str(db)) as conn:
        for row in conn.select():
            n_screened+=1
            try:
                atoms=row.toatoms();symbols=atoms.get_chemical_symbols()
                nt=symbols.count('Ti');no=symbols.count('O')
                if (not bool(np.all(atoms.pbc)) or nt==0 or no!=2*nt
                        or set(symbols)!={'Ti','O'}):
                    n_invalid+=1;continue
                frac=np.asarray(atoms.get_scaled_positions(wrap=True))
                cell=np.asarray(atoms.cell,float)
                ti=frac[np.asarray(symbols)=='Ti']
                oxygen=frac[np.asarray(symbols)=='O']
                crystal=Structure(cell,['Ti']*len(ti)+['O']*len(oxygen),
                                  np.vstack((ti,oxygen)),coords_are_cartesian=False)
                if any(matcher.fit(crystal,old) for _,old in seen):
                    n_duplicate+=1;continue
                label=f'db_{int(row.id):06d}'
                seen.append((label,crystal))
                dataset.append(dict(structure_id=label,is_reference=False,original=crystal))
            except Exception as e:
                n_errors+=1
                if n_errors<=3:
                    print(f' v22 DB screening record {row.id}: {type(e).__name__}: {e}',flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        env,_=collect_calibration_data(dataset,library,bounds,guard,cfg)
    # This equals v22's leave-one-target-out for newly generated Ti structures,
    # since none of these Ti-only candidates is in the DB cohort by identity.
    policy=make_geometry_policy(env,library,'__generated_Ti_without_target_O__',cfg)
    cfg.v22_calibration=policy
    print(f' v22 calibrated geometry: {policy.name} DB screened={n_screened} '
          f'eligible={len(dataset)-4}, duplicates={n_duplicate}, invalid={n_invalid} '
          f'errors={n_errors}, observations={len(env)}; '
          f'Ti_limits={policy.global_limits["Ti"]} O_limits={policy.global_limits["O"]}',flush=True)
    return {'policy':policy.name,'calibration_structures':len(dataset)-4,
            'Ti_eligible_atoms':policy.eligible_atoms.get('Ti',0),
            'O_eligible_atoms':policy.eligible_atoms.get('O',0),
            'screened':n_screened,'duplicates':n_duplicate,'invalid':n_invalid,
            'errors':n_errors,'calibration_observations':len(env)}

def full_symmetry_audit(oxygen,cell,operations,threshold=.055):
    f=np.asarray(oxygen,float)%1.;cc=np.asarray(cell,float)
    if not len(f):return 0,float('inf'),len(operations)
    passed=0;worst=0.
    for R,t in operations:
        transformed=(f@np.asarray(R).T+t)%1.
        delta=minimum_image_delta(transformed[:,None,:]@cc,f[None,:,:]@cc,cc)
        d=np.linalg.norm(delta,axis=-1)
        rr,ii=linear_sum_assignment(d*d)
        err=float(np.max(d[rr,ii]));worst=max(worst,err)
        passed+=err<threshold
    return passed,worst,len(operations)

def chemistry_quality(result,near_factor=1.5):
    d=result['details'];q={}
    for sym,required in [('Ti',6),('O',3)]:
        good=np.asarray(d['good'+sym],bool)
        resid=np.asarray(d['r'+sym],float)
        limits=np.asarray(d['geometry_limits'+sym],float)
        cn=np.asarray(d['hard'+sym],int)
        q['good'+sym]=int(good.sum())
        q['cn'+sym]=int((cn==required).sum())
        q['cnLoss'+sym]=float(d['connectionCNloss'+sym])
        q['n'+sym]=len(good)
        q['primary'+sym]=int(((resid[:,0]<=limits[:,0]) & (cn==required)).sum())
        q['near'+sym]=int((np.all(resid<=near_factor*limits,axis=1)&(cn==required)).sum())
        local=np.asarray(d['local'+sym],float)
        # Original v22r1 per-atom loss: learned geometry plus smooth
        # coordination/contact penalties. These are objective values,
        # not hard good/bad site classifications and not distances in A.
        q['min'+sym]=float(np.min(local))
        q['mean'+sym]=float(np.mean(local))
        q['max'+sym]=float(np.max(local))
        q['rms'+sym]=float(np.sqrt(np.mean(resid**2)))
    q.update(sharing=float(result['rms']),bad_triplets=int(result['bad_triplets']),
             collision=float(d['collision']),inner_score=float(d['energy']),
             full_compatible=int(result['compatible_symops']),
             total_symops=int(result['ti_symops']))
    return q

def local_loss_summary(q, sym):
    """Continuous per-atom v22 loss statistics; no learned thresholds applied."""
    return (f'{sym} loss(min/avg/max)='
            f'{q["min"+sym]:.6g}/{q["mean"+sym]:.6g}/{q["max"+sym]:.6g}')

def local_loss_pair(q):
    return local_loss_summary(q,'Ti')+'; '+local_loss_summary(q,'O')

def chemical_rank(q):
    """Count fully learned-correct Ti sites, then O sites, then near-correct.

    The continuous chemistry variables guide the search ONLY after site counts.
    Neither a global Hamiltonian nor v22's scalar candidate objective chooses
    the population winner. v22 itself retains its ORIGINAL internal objective.
    """
    return (-q['goodTi'],-q['goodO'],-q['nearTi'],-q['nearO'],
            -q['primaryTi'],-q['primaryO'],
            round(q['rmsTi'],7),round(q['rmsO'],7),
            round(q['sharing'],7),q['bad_triplets'],round(q['collision'],7),
            round(q['maxTi'],7),round(q['maxO'],7))

def _v22_search_options(cfg, starts):
    """Translate only option names. No changes to the original oxygen algorithms."""
    return argparse.Namespace(
        accept_A=cfg.accept_A, hierarchy=cfg.hierarchy,
        target_o_cn=cfg.target_o_cn,
        symprec=cfg.symprec, symmetry_tol=cfg.symmetry_tol,
        candidate_cap=cfg.candidate_cap, modes_per_ti=cfg.modes_per_ti,
        octahedral_proposals=cfg.octahedral_proposals,
        methods=list(cfg.v22_inner_methods), starts=int(starts), cycles=cfg.scf_cycles,
        anneal_every=cfg.octa_anneal_every, verbose_cycles=False,
        chemistry_feedback_steps=cfg.feedback_steps,
        chemistry_feedback_rounds=cfg.feedback_rounds,
        global_steps=cfg.v22_global_steps, global_scf_cycles=cfg.v22_global_scf_cycles,
        anneal_scf_cycles=cfg.v22_anneal_scf_cycles,
        global_max_seconds=cfg.v22_global_max_seconds,
        match_baseline_time=False, replicas=cfg.v22_replicas,
        exchange_every=cfg.v22_exchange_every,
        global_rotation=cfg.v22_global_rotation,
        stagnation_window=cfg.v22_stagnation_window,
        exhaustion_window=cfg.v22_exhaustion_window,
        min_global_steps=cfg.v22_min_global_steps,
        checkpoint_every=cfg.v22_checkpoint_every,
        early_sharing=cfg.v22_early_sharing,
        early_primary_A=cfg.v22_early_primary_A,
        early_confirm=cfg.v22_early_confirm,
        improvement_eps=cfg.v22_improvement_eps,
    )

def propose_v22_oxygen(ti,cell,library,bounds,guard,cfg,seed,starts=None,diagnostics=None):
    """Run the COMPLETE original v22r1 search_target on one fixed Ti framework.

    search_target, run_global_method, global_octahedral_move, SCF, and chemistry
    feedback have been copied verbatim. Only outer population bookkeeping is new.
    Never pass the input oxygen coordinates to any of the physics functions.
    """
    if diagnostics is None:diagnostics=collections.Counter()
    nstarts=cfg.bf_fresh if starts is None else starts
    global _V50_CONFIG
    _V50_CONFIG=cfg
    opts=_v22_search_options(cfg,nstarts)
    policy=getattr(cfg,'v22_calibration',None)
    if cfg.geometry_policy!='v21_strict' and policy is None:
        raise RuntimeError('Calibrated v22 geometry requested but original calibration cohort is unavailable')
    stream=io.StringIO()
    # torch.device context chooses devices for the UNMODIFIED factory calls in
    # v22r1 Chemistry, descriptor kernels, and differentiable feedback.
    # NumPy/SciPy octahedral geometry and Hungarian assignments remain CPU.
    # Keep this LOCAL to one task: never mutate torch's global default device.
    oxygen_device=getattr(cfg,'v44_worker_device',
                          getattr(cfg,'oxygen_device_resolved','cpu'))
    prior_gpu_fits=dict(_V44_FIT_STATS)
    try:
        if str(oxygen_device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA requested for oxygen but unavailable: {oxygen_device}')
        with torch.device(oxygen_device):
            with contextlib.redirect_stdout(stream):
                _,_,npeaks,nops,outcomes=search_target(ti,cell,library,bounds,guard,opts,
                                                       seed,calibration=policy)
    except Exception as exc:
        # Save the last 5 diagnostic lines without turning an implementation
        # failure into a false statement about incompatible Ti geometry.
        tail=' | '.join(stream.getvalue().splitlines()[-5:])
        diagnostics['v22_search_failures']+=1
        raise RuntimeError(f'Original v22r1 search_target failed: {type(exc).__name__}: {exc}; last output: {tail}') from exc
    diagnostics['v44_gpu_octa_fit_batches']+=_V44_FIT_STATS['batch_calls']-prior_gpu_fits.get('batch_calls',0)
    diagnostics['v44_gpu_octa_mode_fits']+=_V44_FIT_STATS['mode_fits']-prior_gpu_fits.get('mode_fits',0)
    diagnostics['v22_search_targets']+=1
    diagnostics['Ti_only_OTi3_peaks']+=int(npeaks)
    diagnostics['v22_Ti_symmetry_operations']+=int(nops)
    outputs=[]
    for method,(answer,meta) in outcomes.items():
        diagnostics['v22_successful_methods']+=1
        diagnostics['v22_global_moves']+=int(meta['attempts'])
        diagnostics['v22_global_accepted']+=int(meta['accepted'])
        diagnostics['v22_global_improvements']+=int(meta['improvements'])
        diagnostics['v22_global_escapes']+=int(meta['escapes'])
        diagnostics['v22_distinct_basins_reported']+=int(meta['distinct_basins'])
        diagnostics['SCF_cycles']+=int(meta['scf_cycles'])
        diagnostics['feedback_steps']+=int(meta['feedback_steps'])
        diagnostics['v22_fresh_lcao_calls']+=int(meta['fresh_lcao_calls'])
        q=chemistry_quality(answer,cfg.near_correct_factor)
        outputs.append(dict(ti=np.asarray(ti,float).copy(),cell=np.asarray(cell,float).copy(),
                            answer=answer,q=q,method=method,v22_meta=meta,
                            seed=seed))
    if outputs:
        strongest=min(outputs,key=lambda row:chemical_rank(row['q']))
        q=strongest['q']
        print(f'      original v22 oxygen: methods={len(outputs)} global_moves='
              f'{sum(int(v[1]["attempts"]) for v in outcomes.values())} '
              f'{local_loss_pair(q)} winning_method={strongest["method"]}',flush=True)
    return outputs,[]

def oxygen_quality_satisfied(row,cfg,operations=None):
    """External scheduling gate; never enters original v22 equations/objective.

    An optional full-parent-SG requirement is *only* a quality test. It never
    projects or changes the oxygen coordinates or the original SCF dynamics.
    """
    q=row['q']
    if not (q['goodTi']==q['nTi'] and q['goodO']==q['nO']
            and q['bad_triplets']==0 and q['collision']<=cfg.early_accept_collision
            and q['sharing']<=cfg.early_accept_sharing_A):
        return False
    if getattr(cfg,'early_require_parent_sg',False):
        if operations is None:raise ValueError('Full-parent-SG early gate requires parent symmetry operations')
        n,_,all_ops=full_symmetry_audit(row['answer']['oxygen'],row['cell'],operations)
        return bool(n==all_ops)
    return True

_V22_OCTA_FIT=Octahedra.fit

_V44_FIT_DEVICE='cpu'

_V44_FIT_STATS=collections.Counter()

def _v44_torch_octa_fit(self,targets,mode_options=None,mix=.55,distortion_max=.11,
                        distortion_ridge=1.5):
    """Same v22r1 Kabsch rotation, ridge solve, symmetric eigen-clip and SO3 mixing.

    A single float64 SVD/solve/eigh batch covers every (Ti,mode) proposal;
    the *discrete* mode decision follows the original scoring and tie order.
    Per-octahedron final SO(3) updates use original SciPy Rotation primitives.
    """
    device=torch.device(_V44_FIT_DEVICE)
    pairs=[];starts=[]
    targets=np.asarray(targets,dtype=np.float64)
    for i in range(len(self.ti)):
        opts=[int(self.mode[i])]
        if mode_options is not None:
            opts += [int(k) for k,_ in mode_options[i] if k!=self.mode[i]]
        starts.append(len(pairs))
        for mid in opts:
            pairs.append((i,mid))
    if not pairs:return
    source=np.stack([self.lib['Ti'][m]['rigid_arms'] for _,m in pairs])
    dest=targets[np.asarray([i for i,_ in pairs])]
    X=torch.as_tensor(source,dtype=torch.float64,device=device)
    Y=torch.as_tensor(dest,dtype=torch.float64,device=device)
    # R maps row vectors X to Y: X @ R. Right-handed Kabsch covariance.
    cov=X.transpose(1,2)@Y
    U,_,Vh=torch.linalg.svd(cov,full_matrices=False)
    det=torch.linalg.det(U@Vh)
    eye=torch.eye(3,dtype=X.dtype,device=device).expand(len(pairs),3,3).clone()
    eye[:,2,2]=torch.where(det<0,-1.,1.)
    R=(U@eye)@Vh
    target_local=Y@R.transpose(1,2)
    reg=distortion_ridge*torch.eye(3,dtype=X.dtype,device=device)
    D=torch.linalg.solve(X.transpose(1,2)@X+reg,
                         X.transpose(1,2)@target_local+reg)
    D=.5*(D+D.transpose(1,2))
    eig,Q=torch.linalg.eigh(D)
    D=(Q*torch.clamp(eig,1-distortion_max,1+distortion_max).unsqueeze(-2))@Q.transpose(1,2)
    fit=((X@D)@R-Y).square().sum(dim=-1).mean(dim=-1)
    fit=fit+.20*(D-torch.eye(3,dtype=X.dtype,device=device)).square().sum(dim=(1,2))
    error=fit.detach().cpu().numpy()
    Rm=R.detach().cpu().numpy()
    Dm=D.detach().cpu().numpy()
    for i in range(len(self.ti)):
        stop=starts[i+1] if i+1<len(starts) else len(pairs)
        best=starts[i]+int(np.argmin(error[starts[i]:stop]))
        if not np.isfinite(error[best]):continue
        mode=pairs[best][1]
        rot=Rotation.from_matrix(Rm[best].T)
        stretch=Dm[best]
        if mode!=self.mode[i] and np.linalg.norm(self.stretch[i]-np.eye(3))<.20:
            self.mode[i]=mode
            self.stretch[i]=np.eye(3)
        diff=rot*self.rot[i].inv()
        self.rot[i]=Rotation.from_rotvec(mix*diff.as_rotvec())*self.rot[i]
        self.stretch[i]=(1-mix)*self.stretch[i]+mix*stretch
        val,Q=np.linalg.eigh(self.stretch[i])
        self.stretch[i]=(Q*np.clip(val,1-distortion_max,1+distortion_max))@Q.T
    _V44_FIT_STATS['batch_calls']+=1
    _V44_FIT_STATS['mode_fits']+=len(pairs)

def _v44_fit_parity(device,seed=714):
    """Compare real Octahedra.fit against the torch implementation on perturbed TiO6.

    Validates both mode switches and bounded octahedral Cartesian vertices,
    with the exact same input modes, rotations, strains and targets.
    """
    rng=np.random.default_rng(seed)
    # Non-degenerate physical octahedra, randomized orientations and modes.
    arms=np.asarray([[1.94,0,0],[-1.94,0,0],[0,2.02,0],[0,-2.02,0],
                     [0,0,1.99],[0,0,-1.99]],float)
    lib={'Ti':[{'rigid_arms':arms.copy()},
               {'rigid_arms':arms*np.array([1.02,.98,1.01])},
               {'rigid_arms':arms*np.array([.97,1.01,1.02])}]}
    n=12
    ti=rng.uniform(.05,.95,(n,3));cell=np.diag([12.,12.,12.])
    proposals=[]
    for _ in range(n):
        proposals.append([(0.,m,Rotation.from_rotvec(rng.normal(size=3)*.3))
                          for m in range(3)])
    obj=Octahedra(ti,cell,lib,proposals,seed=0)
    targets=np.asarray([Rotation.from_rotvec(rng.normal(size=3)*.25).apply(
               arms*np.array([1.01,.99,1.02]))+rng.normal(0,.012,(6,3))
                        for _ in range(n)])
    choices=[[(m,float(m)) for m in range(3)] for _ in range(n)]
    max_delta=0.;mode_equal=True
    for opts in [None,choices]:
        for mix in [.30,.56]:
            expected=obj.copy();actual=obj.copy()
            _V22_OCTA_FIT(expected,targets,mode_options=opts,mix=mix)
            saved=_V44_FIT_DEVICE
            try:
                globals()['_V44_FIT_DEVICE']=str(device)
                _v44_torch_octa_fit(actual,targets,mode_options=opts,mix=mix)
            finally:
                globals()['_V44_FIT_DEVICE']=saved
            max_delta=max(max_delta,float(np.max(np.linalg.norm(
                            expected.positions()-actual.positions(),axis=-1))))
            mode_equal &= bool(np.array_equal(expected.mode,actual.mode))
    return max_delta,mode_equal

def _v44_fit_speed_ratio(device,rounds=3):
    """Return elapsed CUDA/CPU on representative 12-site, three-mode fits.

    Enabling GPU only when it is faster is an *execution* decision. The
    accuracy decision remains the preceding vertex/mode parity test.
    """
    rng=np.random.default_rng(441)
    arms=np.asarray([[1.94,0,0],[-1.94,0,0],[0,2.02,0],[0,-2.02,0],
                     [0,0,1.99],[0,0,-1.99]],float)
    lib={'Ti':[{'rigid_arms':arms*(1.+.01*k)} for k in range(3)]}
    seed_rot=Rotation.from_rotvec(rng.normal(size=3)*.2)
    proposals=[[(float(k),k,seed_rot) for k in range(3)] for _ in range(12)]
    src=Octahedra(rng.random((12,3)),np.diag([12.,12.,12.]),lib,proposals)
    targets=rng.normal(size=(12,6,3))*.025+arms[None,:,:]
    choices=[[(m,float(m)) for m in range(3)] for _ in range(12)]
    def measure(fn,dev):
        old=_V44_FIT_DEVICE
        globals()['_V44_FIT_DEVICE']=str(dev)
        try:
            # CUDA warm-up removes one-time context and cuSOLVER loading cost.
            fn(src.copy(),targets,mode_options=choices)
            if str(dev).startswith('cuda'):torch.cuda.synchronize(device=dev)
            t0=time.perf_counter()
            for _ in range(rounds):fn(src.copy(),targets,mode_options=choices)
            if str(dev).startswith('cuda'):torch.cuda.synchronize(device=dev)
            return time.perf_counter()-t0
        finally:globals()['_V44_FIT_DEVICE']=old
    cpu=measure(_V22_OCTA_FIT,'cpu')
    gpu=measure(_v44_torch_octa_fit,device)
    return gpu/max(cpu,1e-9),cpu,gpu

def configure_v44_fit(cfg,device):
    """Enable only after parity certification, never silently alter v22r1 physics."""
    global _V44_FIT_DEVICE
    Octahedra.fit=_V22_OCTA_FIT
    _V44_FIT_DEVICE=str(device)
    if not str(device).startswith('cuda') and cfg.gpu_octa_fit=='on':
        raise RuntimeError('--gpu-octa-fit on requires CUDA oxygen device; use --oxygen-device cuda')
    if cfg.gpu_octa_fit=='off' or not str(device).startswith('cuda'):
        return 'original_scipy'
    try:
        discrepancy,modes_equal=_v44_fit_parity(device)
        if not modes_equal or discrepancy>cfg.gpu_fit_parity_A:
            raise RuntimeError(f'fit-parity: mode_equal={modes_equal}; '
                               f'max_vertex_error={discrepancy:.4g} A; '
                               f'tolerance={cfg.gpu_fit_parity_A:.4g} A')
        ratio,cpu,gpu=_v44_fit_speed_ratio(device)
        print(f' GPU octa-fit 12Ti benchmark: original CPU={cpu:.3f}s '
              f'batched CUDA={gpu:.3f}s ratio={ratio:.2f} '
              f'parity={discrepancy:.2g} A',flush=True)
        if cfg.gpu_octa_fit=='auto' and ratio>=.95:
            print(' GPU octa-fit: CUDA not faster; retaining original CPU fit '
                  '(chemistry remains on CUDA)',flush=True)
            return 'original_scipy'
        Octahedra.fit=_v44_torch_octa_fit
        print(f' GPU octa-fit enabled {device}; parity max vertex '
              f'error={discrepancy:.3g} A; modes identical',flush=True)
        return 'cuda_batched_fit'
    except Exception as exc:
        Octahedra.fit=_V22_OCTA_FIT
        if cfg.gpu_octa_fit=='on':raise
        print(f' GPU octa-fit parity unavailable/failed: {exc}; '
              'using original SciPy octa-fit',flush=True)
        return 'original_scipy'

_V44_WORKER_CONTEXT=None

def resolve_v44_compute(cfg):
    """Auto detect visible CUDA GPUs and constrain CPU parallelism."""
    try:affinity=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):affinity=os.cpu_count() or 1
    slurm=os.environ.get('SLURM_CPUS_PER_TASK','')
    allocated=min(affinity,int(slurm)) if slurm.isdigit() else affinity
    ngpu=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cfg.oxygen_device=='cuda' and not ngpu:
        raise RuntimeError('--oxygen-device cuda requires a CUDA-visible GPU; '
                           'check your Slurm allocation and CUDA_VISIBLE_DEVICES')
    device='cuda' if ngpu and cfg.oxygen_device!='cpu' else 'cpu'
    workers=(min(4,ngpu) if cfg.gpu_workers=='auto' else int(cfg.gpu_workers)) if device=='cuda' else 0
    if device=='cuda' and workers>ngpu:
        raise ValueError(f'--gpu-workers={workers} but only {ngpu} CUDA GPUs visible')
    if device=='cpu':
        workers=min(allocated,max(1,allocated//2)) if cfg.oxygen_workers=='auto' else min(int(cfg.oxygen_workers),allocated)
    cfg.oxygen_device_resolved='cuda:0' if device=='cuda' else 'cpu'
    cfg.v44_cuda_devices=tuple(f'cuda:{j}' for j in range(workers))
    cfg.v44_worker_device=cfg.oxygen_device_resolved
    return workers,max(1,allocated),ngpu

def _v44_atomic_cif(row,path):
    """CIF-only durable checkpoint; replace best file atomically."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.partial.cif')
    try:
        write_final_cif(row['ti'],row['cell'],row['answer']['oxygen'],tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def argc_screen(cfg):
    return argparse.Namespace(bf_physical_min_TiTi_A=cfg.min_TiTi_A)

def configured_ti_backend(cfg,ti_flags):
    """Select Ti-only PyTorch backend independently from CUDA oxygen workers."""
    flags=list(ti_flags)
    has_explicit=any(x=='--device' or x.startswith('--device=') for x in flags)
    if not has_explicit:flags+=['--device',cfg.ti_device]
    if not any(x=='--ti-rule-mode' or x.startswith('--ti-rule-mode=') for x in flags):
        flags+=['--ti-rule-mode',cfg.ti_rule_mode]
    for attr,suffix in (('ti_empirical_contact_margin_A','ti-empirical-contact-margin-A'),
                        ('ti_empirical_screen_margin_A','ti-empirical-screen-margin-A'),
                        ('ti_empirical_density_width_A','ti-empirical-density-width-A'),
                        ('ti_empirical_repulsion_width_A','ti-empirical-repulsion-width-A'),
                        ('ti_empirical_repulsion_normalizer_A','ti-empirical-repulsion-normalizer-A'),
                        ('ti_empirical_volume_log_width_floor','ti-empirical-volume-log-width-floor'),
                        ('ti_empirical_span_neighbors','ti-empirical-span-neighbors'),
                        ('ti_empirical_density_ranks','ti-empirical-density-ranks'),
                        ('ti_soap_pca','ti-soap-pca'),
                        ('ti_soap_pca_variance','ti-soap-pca-variance'),
                        ('ti_soap_pca_max_components','ti-soap-pca-max-components')):
        key='--'+suffix
        if not any(x==key or x.startswith(key+'=') for x in flags):
            flags.extend([key,str(getattr(cfg,attr))])
    return flags

def v48_iter_wyckoff(sg, n_ti, max_orbits=0):
    """Yield ALL multisets of PyXtal Wyckoff orbits summing to n_ti.

    A fixed-coordinate zero-DOF orbit may appear at most once. A variable
    orbit may occur repeatedly, representing independent inequivalent sites.
    max_orbits=0 imposes no artificial length cap; enumeration streams.
    """
    from pyxtal.symmetry import Group
    g=Group(int(sg))
    allowed=[(i,int(g[i].multiplicity),int(g[i].get_dof()))
             for i in range(len(g)) if 0<int(g[i].multiplicity)<=n_ti]
    def rec(first, remaining, prefix):
        if remaining==0:
            yield tuple(prefix)
            return
        if max_orbits and len(prefix)>=max_orbits:return
        for j in range(first,len(allowed)):
            index,m,dof=allowed[j]
            if m>remaining:continue
            if not dof and index in prefix:continue
            yield from rec(j,remaining-m,prefix+[index])
    yield from rec(0,int(n_ti),[])

def _v48_ti_stage(cfg,ti_flags):
    """Learn Ti rules ONCE for the sweep, not once per SG/WP task."""
    args=['--ref',cfg.ref,'--starts',str(cfg.full_ti_starts),
          '--keep-basins',str(cfg.full_ti_basins),
          '--natural-steps',str(cfg.full_natural_steps),
          '--learned-steps',str(cfg.full_learned_steps),
          '--device',cfg.ti_device,
          '--ti-rule-mode','hybrid',
          '--ti-soap-pca','audit']
    args=configured_ti_backend(cfg,args)
    # Do not allow study-specific or redundant original Ti arguments in sweep.
    if ti_flags:
        unknown=' '.join(ti_flags)
        raise ValueError(f'Unknown options for --full-search: {unknown}; '
                         'use --full-ti-starts, --full-ti-basins and sweep flags')
    ts=parse_ti_args(args)
    dev=torch.device('cuda:0' if ts.device=='auto' and torch.cuda.is_available()
                     else ('cpu' if ts.device=='auto' else ts.device))
    refs=reference_ti_cifs(cfg.ref)
    l=learn_ti_reference(refs,ts.shell_min_k,ts.shell_max_k,ts.shell_gap,
                         ts.mode_merge,ts.soap_neighbors)
    model=infer_ti_rules(refs,l,ts)
    l=apply_ti_rule_mode(l,model,'hybrid')
    report_ti_soap_pca(l,ts)
    if ts.ti_soap_pca=='on':
        raise RuntimeError('Production exhaustive search requires full SOAP (PCA audit/off only)')
    objective=TiObjective(l,ts,dev)
    print(f' Ti stage initialized once: device={dev} modes={len(l["modes"])} '
          f'hybrid margins=on full-SOAP=on',flush=True)
    return ts,dev,l,objective

def _v48_nearest(frac,cell):
    _,dist=periodic_vectors_numpy(np.asarray(frac,float),np.asarray(cell,float),1,radius=2)
    return float(np.min(dist[:,0]))

class _V48r5MovingOrigin:
    """Moving material-frame displacement observer, never an optimizer control.

    Fractional changes are measured in the midpoint lattice, separating ionic
    motion from homogeneous cell strain. The minimum-image distance is found
    over 27 periodic images, also for skew cells. For O only, the next snapshot
    is matched to the previous snapshot using periodic Hungarian assignment.
    Both net motion and accumulated individual-ion travel are tracked, so a
    hop followed by a return cannot masquerade as immobility.
    """
    _shifts=np.asarray(list(itertools.product((-1.,0.,1.),repeat=3)),float)

    def __init__(self,origin_reset_A=.25,stop_A=.05,windows=(5,10,20),consecutive=3,
                 match_oxygen=False):
        self.reset_A=float(origin_reset_A)
        self.stop_A=float(stop_A)
        self.windows=tuple(sorted(set(int(w) for w in windows)))
        self.consecutive=int(consecutive)
        self.match=bool(match_oxygen)
        self.history=collections.deque(maxlen=max(self.windows,default=1)+1)
        self.path_history=collections.deque(maxlen=max(self.windows,default=1)+1)
        self.since_reset=0
        self.resets=0
        self.still=collections.Counter()
        self.potential_stop_first={}
        self.stable_samples=collections.Counter()
        self.latest={}
        self.best=None
        self.last=None
        self.last_step=None
        self.sample_count=0
        self.total_path_A=None
        self.topology=None
        self.topology_change_steps=[]

    @classmethod
    def _motion(cls,old_frac,old_cell,new_frac,new_cell,match=False):
        fa=np.asarray(old_frac,float).reshape(-1,3)
        fb=np.asarray(new_frac,float).reshape(-1,3)
        ca=np.asarray(old_cell,float).reshape(3,3)
        cb=np.asarray(new_cell,float).reshape(3,3)
        if fa.shape!=fb.shape or not np.isfinite(fa).all() or not np.isfinite(fb).all():
            raise ValueError('Ionic coordinates changed shape or became nonfinite')
        mid=.5*(ca+cb)
        if match:
            d=fb[None,:,:]-fa[:,None,:]
            cart=np.einsum('nmsk,kj->nmsj',d[:,:,None,:]+cls._shifts[None,None,:,:],mid)
            sq=np.einsum('nmsj,nmsj->nms',cart,cart)
            bestsq=sq.min(axis=2)
            rows,cols=linear_sum_assignment(bestsq)
            order=np.empty(len(fa),int);order[rows]=cols
            fb=fb[order]
        d=fb-fa
        cart=(d[:,None,:]+cls._shifts[None,:,:])@mid
        distances=np.linalg.norm(cart,axis=-1).min(axis=1)
        cell_rel=float(np.linalg.norm(cb-ca)/max(np.linalg.norm(ca),1e-12))
        return distances,fb%1.,cell_rel

    @staticmethod
    def _triplet_network(trip,nO):
        tr=np.asarray(trip,int).reshape(-1)
        if len(tr)%6 or np.any(tr<0) or np.any(tr>=nO):
            return None
        neighbors=[[] for _ in range(nO)]
        for arm,oxygen_index in enumerate(tr):
            neighbors[int(oxygen_index)].append(arm//6)
        return tuple(sorted(tuple(sorted(v)) for v in neighbors))

    def observe(self,frac,cell,step,score=None,prefix='O',topology=None):
        frac=np.asarray(frac,float).reshape(-1,3)%1.
        cell=np.asarray(cell,float).reshape(3,3)
        if not np.isfinite(frac).all() or not np.isfinite(cell).all():
            raise ValueError('Nonfinite displacement snapshot')
        self.sample_count+=1
        snap=(int(step),frac.copy(),cell.copy())
        topo_changed=0
        if topology is not None:
            network=self._triplet_network(topology,len(frac))
            if network is not None:
                if self.topology is not None and network!=self.topology:
                    topo_changed=1
                    self.topology_change_steps.append(int(step))
                self.topology=network
        row={f'{prefix}_origin_resets':self.resets,
             f'{prefix}_origin_reset':0,
             f'{prefix}_origin_age_steps':self.since_reset,
             f'{prefix}_topology_change':topo_changed,
             f'{prefix}_topology_events':len(self.topology_change_steps)}
        if self.last is None:
            self.last=snap
            self.anchor=snap
            self.total_path_A=np.zeros(len(frac),float)
            self.history.append(snap)
            self.path_history.append(self.total_path_A.copy())
            self.last_step=int(step)
            row.update({f'{prefix}_step_max_A':0.,f'{prefix}_step_RMS_A':0.,
                        f'{prefix}_anchor_max_A':0.,f'{prefix}_anchor_RMS_A':0.,
                        f'{prefix}_cell_rel_step':0.,f'{prefix}_path_total_max_A':0.})
        else:
            disp,aligned,cell_rel=self._motion(self.last[1],self.last[2],frac,cell,match=self.match)
            snap=(int(step),aligned.copy(),cell.copy())
            self.total_path_A+=disp
            anchor_disp,_,_=self._motion(self.anchor[1],self.anchor[2],aligned,cell)
            did_reset=bool(float(anchor_disp.max())>=self.reset_A)
            if did_reset:
                self.anchor=snap
                self.resets+=1
                self.since_reset=0
            else:
                self.since_reset+=1
            row.update({f'{prefix}_step_max_A':float(disp.max()),
                        f'{prefix}_step_RMS_A':float(np.sqrt(np.mean(disp**2))),
                        f'{prefix}_anchor_max_A':float(anchor_disp.max()),
                        f'{prefix}_anchor_RMS_A':float(np.sqrt(np.mean(anchor_disp**2))),
                        f'{prefix}_origin_reset':int(did_reset),
                        f'{prefix}_origin_resets':self.resets,
                        f'{prefix}_origin_age_steps':self.since_reset,
                        f'{prefix}_cell_rel_step':cell_rel,
                        f'{prefix}_path_total_max_A':float(self.total_path_A.max())})
            self.last=snap
            self.history.append(snap)
            self.path_history.append(self.total_path_A.copy())
            self.last_step=int(step)
        for w in self.windows:
            # The window must include w *actual successive* steps. A trace
            # interrupted by an optimizer restart is not silently bridged.
            old=next(((j,q) for j,q in enumerate(self.history)
                      if q[0]==int(step)-w),None)
            if old is None:
                row[f'{prefix}_net_w{w}_max_A']=float('nan')
                row[f'{prefix}_path_w{w}_max_A']=float('nan')
                row[f'{prefix}_stop_w{w}']=0
                self.still[w]=0
                continue
            idx,then=old
            d,_,_=self._motion(then[1],then[2],snap[1],snap[2])
            p=self.total_path_A-self.path_history[idx]
            net=float(d.max())
            path=float(p.max())
            row[f'{prefix}_net_w{w}_max_A']=net
            row[f'{prefix}_path_w{w}_max_A']=path
            # A candidate is immobile only if BOTH net and traveled path
            # are small. A rapid hop followed by return fails this check.
            # Geometrically stationary but rewired networks must NOT qualify.
            network_stable=not any(int(step)-w < x <= int(step)
                                   for x in self.topology_change_steps)
            stationary=(net<self.stop_A and path<self.stop_A and network_stable)
            self.still[w]=self.still[w]+1 if stationary else 0
            row[f'{prefix}_stop_w{w}']=int(self.still[w]>=self.consecutive)
            if row[f'{prefix}_stop_w{w}']:
                self.stable_samples[w]+=1
                self.potential_stop_first.setdefault(w,int(step))
        self.latest=row.copy()
        if score is not None and np.isfinite(float(score)):
            if self.best is None or float(score)<self.best[0]:
                self.best=(float(score),snap)
        return row

    def best_final_difference(self):
        if self.best is None or self.last is None:return None
        _,snap=self.best
        d,_,cell_rel=self._motion(snap[1],snap[2],self.last[1],self.last[2],match=self.match)
        return {'best_step':snap[0], 'final_step':self.last[0],
                'max_A':float(d.max()),'RMS_A':float(np.sqrt(np.mean(d**2))),
                'cell_rel':cell_rel,'resets':self.resets}

class _V50TiConvergence:
    """Per-start moving frame, cumulative path, lattice stability and objective trend."""
    def __init__(self,cfg):
        self.cfg=cfg
        self.window=int(cfg.v50_ti_window)
        self.trackers={}
        self.loss_history={}
        self.reasons=collections.Counter()

    def check(self,t,x,override,indices,losses,step,stage):
        if not len(indices):return []
        # Compute positions only for live starts; already converged starts are
        # never copied or passed through the expensive Ti objective again.
        with torch.no_grad():
            cells,frac,*_=geometry(t,x.index_select(0,indices).detach(),override)
            cells=cells.detach().cpu().numpy()
            frac=frac.detach().cpu().numpy()
            ids=indices.detach().cpu().tolist()
            scores=losses.detach().cpu().tolist()
        completed=[]
        lower=int(self.cfg.v50_ti_min_natural if stage=='natural'
                  else self.cfg.v50_ti_min_learned)
        for j,i in enumerate(ids):
            key=(stage,i)
            if key not in self.trackers:
                self.trackers[key]=_V48r5MovingOrigin(
                    origin_reset_A=float(self.cfg.audit_origin_reset_A),
                    stop_A=float(self.cfg.v50_ti_stop_A),
                    windows=(self.window,),consecutive=int(self.cfg.audit_movement_consecutive),
                    match_oxygen=False)
                self.loss_history[key]=collections.deque(maxlen=self.window+1)
            tracker=self.trackers[key]
            info=tracker.observe(frac[j],cells[j],step,prefix='Ti')
            history=self.loss_history[key]
            history.append((step,float(scores[j])))
            if step<max(lower,self.window+int(self.cfg.audit_movement_consecutive)):
                continue
            if not info.get(f'Ti_stop_w{self.window}',0) or len(history)<self.window+1:
                continue
            first=history[0][1]
            last=min(y for _,y in history)
            delta=max(0.,first-last)
            tolerance=max(float(self.cfg.v50_ti_objective_abs),
                          float(self.cfg.v50_ti_objective_rel)*abs(first))
            if delta>tolerance:
                continue
            former=tracker.history[0][2]
            cell_rel=float(np.linalg.norm(cells[j]-former)/max(np.linalg.norm(former),1e-12))
            if cell_rel>float(self.cfg.v49_cell_rel_tolerance):
                continue
            self.reasons['movement_and_objective']+=1
            completed.append(i)
        return completed

def _v50_run_optimizer(objective,t,initial,stage,steps,lr,freeze_cell=False,
                       override=None,progress=0,patience=30,min_steps=50,early_tol=1e-4):
    """Compact expensive forward/backward to active starts; keep full Adam state.

    Adam still owns the original full parameter tensor, so a stopping decision
    does not disturb live starts' momentum or their initial/best-state identity.
    Stopped rows are pinned to best and never presented to TiObjective again
    until the final reporting pass. The learning rate is unchanged.
    """
    if steps<=0:
        return initial.detach().clone(),[],{'steps_executed':0,'early_stopped':0}
    started=time.perf_counter()
    x=initial.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([x],lr=lr)
    n=len(x)
    best=x.detach().clone()
    bestE=torch.full((n,),float('inf'),dtype=DTYPE,device=x.device)
    anchor=torch.full_like(bestE,float('inf'))
    stalls=torch.zeros(n,dtype=torch.int,device=x.device)
    active=torch.ones(n,dtype=torch.bool,device=x.device)
    finish=torch.zeros(n,dtype=torch.int,device=x.device)
    controller=_V50TiConvergence(_V50_CONFIG)
    reasons=collections.Counter()
    trace=[];executed=0
    for it in range(steps):
        executed=it+1
        idx=torch.nonzero(active,as_tuple=True)[0]
        if not len(idx):break
        fac=.15+.85*.5*(1.+math.cos(math.pi*it/max(steps-1,1)))
        opt.param_groups[0]['lr']=float(lr)*fac
        opt.zero_grad(set_to_none=True)
        y=objective.evaluate(t,x.index_select(0,idx),stage,cell_override=override)
        if not bool(torch.isfinite(y).all()):
            log(f'  v50 WARNING {stage}: nonfinite objective at step {it+1}; preserving best')
            break
        with torch.no_grad():
            new=(y<bestE[idx])
            bestE[idx]=torch.minimum(bestE[idx],y.detach())
            if bool(new.any()):best[idx[new]]=x.detach()[idx[new]]
            anchor_old=anchor[idx]
            significant=(anchor_old==float('inf')) | (y.detach()<anchor_old-early_tol*(1.+anchor_old.abs()))
            anchor[idx]=torch.where(significant,y.detach(),anchor_old)
            stalls[idx]=torch.where(significant,torch.zeros_like(stalls[idx]),stalls[idx]+1)
            patience_stop=stalls[idx]>=patience
            patience_stop=patience_stop & (it+1>=min_steps)
            if bool(patience_stop.any()):
                stopped=idx[patience_stop]
                finish[stopped]=it+1
                active[stopped]=False
                reasons['objective_patience']+=len(stopped)
            keep=~patience_stop
        if not bool(active.any()):break
        y[keep].sum().backward()
        if x.grad is None:break
        if freeze_cell:x.grad[:,:len(t.lattice_keys)]=0.
        with torch.no_grad():x.grad[~active]=0.
        torch.nn.utils.clip_grad_norm_([x],max_norm=10.)
        opt.step()
        with torch.no_grad():x[~active]=best[~active]
        # Check actual post-step movement, including the full ion path per step.
        idx2=torch.nonzero(active,as_tuple=True)[0]
        if len(idx2):
            # The current evaluation y was made immediately before Adam.step.
            # Use stored best objective up to that point for the trend gate.
            stop_ids=controller.check(t,x,override,idx2,bestE[idx2],it+1,stage)
            if stop_ids:
                with torch.no_grad():
                    active[stop_ids]=False
                    finish[stop_ids]=it+1
                    x[stop_ids]=best[stop_ids]
                reasons['stationary_and_objective']+=len(stop_ids)
        if progress and ((it+1)%progress==0 or it==0 or it+1==steps):
            with torch.no_grad():
                ids=torch.nonzero(active,as_tuple=True)[0]
                probe=x.index_select(0,ids) if len(ids) else best
                ee=objective.evaluate(t,probe,stage)
                record={'step':it+1,'stage':stage,'best_objective':float(ee.min()),
                        'median_objective':float(ee.median()),'active_starts':int(active.sum())}
                trace.append(record)
                log(f'  v50 {stage} {it+1}/{steps}: best={record["best_objective"]:.5f} '
                    f'live={record["active_starts"]}/{n}')
        if not bool(active.any()):break
    with torch.no_grad():
        # Evaluate the exact original per-start best state, not the last Adam step.
        last=objective.evaluate(t,x,stage,cell_override=override)
        better=last<bestE
        best[better]=x.detach()[better]
        finish[finish==0]=executed
        res,detail=objective.evaluate(t,best,stage,details=True,cell_override=override)
        ans={'steps_executed':executed,'early_stopped':int((finish<steps).sum()),
             'finish_step_per_start':finish.cpu().numpy(),
             'median_final_objective':float(torch.median(res)),
             'best_final_objective':float(res.min()),
             'min_titi_A':float(detail['minimum_titi_A'].min()),
             'v50_start_stops':dict(reasons),
             'v50_forward_start_steps':int(executed*n - sum(
                 max(0,executed-int(s)) for s in finish.cpu().tolist())),
             'v50_elapsed_s':time.perf_counter()-started}
    _V50_COUNTERS['ti_'+stage+'_wall_s']+=ans['v50_elapsed_s']
    _V50_COUNTERS['ti_'+stage+'_steps']+=executed
    _V50_COUNTERS['ti_'+stage+'_forward_start_steps']+=ans['v50_forward_start_steps']
    log(f'  v50 {stage}: optimizer iterations={executed}/{steps}; '
        f'early={ans["early_stopped"]}/{n}; stops={dict(reasons)}; '
        f'live-forward-start-steps={ans["v50_forward_start_steps"]}/{executed*n}; '
        f'elapsed={ans["v50_elapsed_s"]:.2f}s; best={ans["best_final_objective"]:.5f}')
    return best.detach().clone(),trace,ans

class _V50SCFConvergence:
    """Conservative SCF convergence; only exact coherent network checkpoints."""
    def __init__(self,cfg):
        self.cfg=cfg
        self.best_rms=float('inf');self.best_dup=float('inf')
        self.best_step=0;self.chem_checks=0
        self.movement=_V48r5MovingOrigin(
            origin_reset_A=float(cfg.audit_origin_reset_A),
            stop_A=float(cfg.audit_movement_stop_A),windows=(min(12,max(5,int(cfg.v50_scf_stall_cycles))),),
            consecutive=2,match_oxygen=True)
        self.last_topology=None

    def check(self,octas,centers,trip,rms,duplicate,chem,step):
        oxygen=np.asarray(centers,float)@octas.inv
        info=self.movement.observe(oxygen,octas.cell,step,topology=trip,prefix='O')
        improved=(duplicate<self.best_dup or
                  (duplicate==self.best_dup and rms<self.best_rms-0.001))
        if improved:
            self.best_dup=int(duplicate);self.best_rms=float(rms);self.best_step=int(step)
        if (step>=int(self.cfg.v50_scf_min_cycles) and
                step%int(self.cfg.v50_scf_check_every)==0 and
                duplicate==0 and rms<=float(self.cfg.early_accept_sharing_A)):
            with torch.no_grad():
                detail=chem.score(torch.tensor(oxygen%1.,dtype=DTYPE),details=True)
            self.chem_checks+=1
            candidate={'rms':float(rms),'bad_triplets':int(duplicate),'details':detail}
            if _v50_checkpoint_complete(candidate):
                return True,'complete'
        window=int(self.movement.windows[0])
        # Give near-perfect sharing at least one full chemistry checkpoint
        # before classifying it as an unproductive stationary network.
        if (duplicate==0 and rms<=float(self.cfg.early_accept_sharing_A)
                and step%int(self.cfg.v50_scf_check_every)!=0):
            return False,None
        if (step>=int(self.cfg.v50_scf_min_cycles)
            and step-self.best_step>=int(self.cfg.v50_scf_stall_cycles)
            and info.get(f'O_stop_w{window}',0)==1):
            return True,'stationary_no_sharing_progress'
        return False,None

class _V50FeedbackConvergence:
    """Observe repaired chemistry, triplet connectivity and scalar at full checkpoints."""
    def __init__(self,cfg):
        self.cfg=cfg
        self.best=None
        self.last_network=None
        self.still=0
        self.checkpoints=0

    def check(self,candidate,step):
        self.checkpoints+=1
        if _v50_checkpoint_complete(candidate):
            return True,'complete'
        if not np.isfinite(float(candidate['rms'])):
            return True,'nonfinite'
        d=candidate['details']
        net=_V48r5MovingOrigin._triplet_network(candidate['trip'],len(candidate['oxygen']))
        changed=self.last_network is not None and net!=self.last_network
        self.last_network=net
        metric=(int(candidate['bad_triplets']),
                int(d['badTi'])+int(d['badO']),
                float(candidate['rms']),float(d['collision']),
                continuous_candidate_objective(candidate))
        improved=False
        if self.best is None:
            improved=True
        else:
            old=self.best
            improved=(metric[0]<old[0] or metric[1]<old[1] or
                      metric[2]<old[2]-.01 or metric[3]<old[3]-.01 or
                      metric[4]<old[4]-float(self.cfg.v50_feedback_rank_abs))
        if improved:
            self.best=metric
        self.still=0 if (improved or changed) else self.still+1
        if (step>=int(self.cfg.v50_feedback_min_steps) and
                self.still>=int(self.cfg.v50_feedback_stall_checkpoints)):
            return True,'chemical_plateau'
        return False,None

def _v50_fresh_output(cfg):
    if cfg.ti_rule_audit_only or cfg.v50_self_test:return
    if cfg.v50_allow_resume:return
    output=Path(cfg.output)
    if output.is_file():
        raise FileExistsError(f'v50 output must be a new directory, not a file: {output}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'v50 refuses to reuse an existing nonempty output directory: '
                              f'{output}. Choose a new --output. '
                              'Use --v50-allow-resume only for your own v50 run.')

def _v52_require(cfg):
    if cfg.v52_ti_count < 2 or cfg.v52_ti_count > cfg.full_max_ti:
        raise ValueError('Invalid Ti count; increase --full-max-ti explicitly if necessary')
    if cfg.v52_ti_raw_batch < 1 or cfg.v52_ti_opt_batch < 0 or cfg.v52_ti_opt_batch > cfg.v52_ti_raw_batch:
        raise ValueError('Raw Ti batch must be positive; optimized batch must be 0 (auto) or within raw batch')
    for name in ('v52_base_batches','v52_max_batches','v52_basins_per_batch',
                 'v52_oxygen_entrances','v52_max_oxygen_entrances',
                 'v52_global_proposals','v52_scf_cycles','v52_feedback_steps',
                 'v52_global_steps','v52_enumeration_cap'):
        if int(getattr(cfg,name)) < 1:raise ValueError(name+' must be positive')
    if cfg.v52_base_batches > cfg.v52_max_batches:
        raise ValueError('Initial Ti batch allowance cannot exceed the maximum')
    if cfg.v52_oxygen_entrances > cfg.v52_max_oxygen_entrances:
        raise ValueError('Oxygen entrances exceeds its cap')
    if cfg.v52_global_method not in ('anneal','hopping','replica'):
        raise ValueError('Unknown nonlocal oxygen method')
    if cfg.v52_max_scale < 1 or cfg.v52_max_wp_per_sg < 0 or cfg.v52_max_orbits < 0:
        raise ValueError('Invalid complexity/wyckoff cap')
    if cfg.v52_feedback_rounds != 1:
        raise ValueError('v52 requires exactly one brief coherent feedback round')
    if cfg.v50_allow_resume:
        raise ValueError('v52 never loads or resumes generated candidates; use a fresh --output')
    if cfg.early_require_parent_sg:
        raise ValueError('v52 parent-SG oxygen compatibility gate is not supported yet; '
                         'Ti parent orbit symmetry is preserved, O chemistry is validated')
    if cfg.v51_ti_dynamics or cfg.full_search or cfg.v50_ab or cfg.movement_ab or cfg.convergence_audit:
        raise ValueError('--v52-dfs is exclusive with the older search/experiment modes')
    return parse_sgs(cfg.v52_sgs)

def _v52_oxygen_cfg(cfg, global_search=False):
    c=copy.copy(cfg)
    c.v22_inner_methods=(['baseline',cfg.v52_global_method] if global_search
                         else ['baseline'])
    c.scf_cycles=cfg.v52_scf_cycles
    c.feedback_steps=cfg.v52_feedback_steps
    c.feedback_rounds=cfg.v52_feedback_rounds
    # Preserve physical tests, but permit these explicitly smaller evaluation
    # budgets to finish without inheriting v50's larger minimum cycle counts.
    c.v50_scf_min_cycles=min(c.scf_cycles,cfg.v50_scf_min_cycles)
    c.v50_feedback_min_steps=min(c.feedback_steps,cfg.v50_feedback_min_steps)
    if global_search:
        c.v22_global_steps=cfg.v52_global_steps
        c.v22_min_global_steps=min(cfg.v22_min_global_steps,c.v22_global_steps)
        c.v22_global_scf_cycles=min(cfg.v22_global_scf_cycles,c.scf_cycles)
        c.v22_anneal_scf_cycles=min(cfg.v22_anneal_scf_cycles,c.scf_cycles)
        c.v22_global_max_seconds=0.0  # No wall-time stop; proposal count bounds it.
        c.v22_early_sharing=-1.0  # Do not end exploration just because one complete basin appears.
        c.v22_early_primary_A=-1.0
    return c

def _v52_raw_filter(t,lib,raw,batch_size):
    """Cheap geometry ranking; preserve an exploration minority by stratification.

    No raw initial is ever declared physically impossible based on this proxy.
    Real Ti geometry and chemistry remain the authoritative downstream tests.
    """
    with torch.no_grad():
        param=torch.as_tensor(raw,dtype=DTYPE,device=t.blocks[0]['A'].device)
        cells=make_lattice(t,param)[0].detach().cpu().numpy()
        frac=expand_ti(t,param)[0].detach().cpu().numpy()%1.
    vol=np.abs(np.linalg.det(cells))/t.n_ti
    median=float(lib['volume_median'])
    scores=[]
    for i in range(len(raw)):
        try:
            minimum=_v48_nearest(frac[i],cells[i])
            score=max(0.,2.85-minimum)*2. + .6*abs(math.log(max(1e-8,vol[i])/median))
        except Exception:
            score=float('inf')
        scores.append(score)
    ordered=sorted(range(len(raw)),key=lambda j:(scores[j],j))
    # Reserve one selection per batch for an alternative raw geometry, avoiding
    # complete collapse to a single proxy basin.
    n=min(len(raw),batch_size)
    selected=ordered[:max(0,n-1)]
    for j in np.linspace(len(ordered)-1,0,len(ordered),dtype=int):
        candidate=ordered[int(j)]
        if candidate not in selected and np.isfinite(scores[candidate]):
            selected.append(candidate);break
    if len(selected)<n:
        selected.extend(j for j in ordered if j not in selected and len(selected)<n)
    return np.asarray(raw[selected[:n]],float),{
        'raw_count':len(raw),'opt_count':len(selected[:n]),
        'proxy_best':float(min(scores)),
        'proxy_selected_max':float(max(scores[i] for i in selected[:n]))}

def _v52_ti_duplicate(ti,cell,already):
    # Pre-screen and symmetry/periodic matching within a Wyckoff combination.
    # Across WP labels, the completed TiO2 matcher below is authoritative.
    for old in already:
        if cell_metric_discrepancy(old['cell'],cell)<.025:
            if periodic_match_rms(old['ti'],ti,old['cell'])<.12:
                return True
    return False

def _v52_structure(row):
    from pymatgen.core import Structure,Lattice
    ti=np.asarray(row['ti'],float);oxygen=np.asarray(row['answer']['oxygen'],float)
    coords=np.concatenate([ti,oxygen],axis=0)
    return Structure(Lattice(np.asarray(row['cell'],float)),
        ['Ti']*len(ti)+['O']*len(oxygen),coords,coords_are_cartesian=False)

def _v52_autotune_opt_batch(t,lib,objective,cfg):
    """Benchmark the actual Ti natural/learned forward+backward at GPU launch.

    Use start-steps per second, not smallest latency per optimizer call.  This
    only selects a batch width; it never changes the Ti loss or force equations.
    CPU development mode defaults to 8 and reports no claimed GPU optimum.
    """
    if cfg.v52_ti_opt_batch:
        return cfg.v52_ti_opt_batch,{'mode':'explicit','selected':cfg.v52_ti_opt_batch}
    if objective.device.type!='cuda':
        n=min(8,cfg.v52_ti_raw_batch)
        return n,{'mode':'cpu_default_no_gpu_benchmark','selected':n}
    available=[n for n in (4,8,12,16,24,32,48) if n<=cfg.v52_ti_raw_batch]
    if cfg.v52_ti_raw_batch not in available:available.append(cfg.v52_ti_raw_batch)
    available=sorted(set(available))
    raw=initial_raw(t,lib,max(available),str_seed(cfg.seed,'v52_gpu_batch_calibration'))
    raw=torch.as_tensor(raw,dtype=DTYPE,device=objective.device)
    observations=[]
    for n in available:
        measures=[]
        try:
            # Warm each shape before timing. CUDA synchronizations exclude
            # asynchronous timing artifacts and ensure actual GPU completion.
            for attempt in range(3):
                x=raw[:n].detach().clone().requires_grad_(True)
                torch.cuda.synchronize(objective.device)
                start=time.perf_counter()
                for stage in ('natural','learned'):
                    if x.grad is not None:x.grad=None
                    objective.evaluate(t,x,stage).sum().backward()
                torch.cuda.synchronize(objective.device)
                if attempt:measures.append(time.perf_counter()-start)
                del x
            seconds=float(np.median(measures))
            observations.append({'batch':n,'seconds_per_both_objectives':seconds,
                'combined_start_steps_per_second':2*n/max(seconds,1e-12)})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            observations.append({'batch':n,'error':'out_of_memory'})
    measured=[d for d in observations if 'combined_start_steps_per_second' in d]
    if not measured:raise RuntimeError('No usable GPU Ti objective batch for v52')
    best=max(measured,key=lambda r:r['combined_start_steps_per_second'])
    return int(best['batch']),{'mode':'CUDA_forward_backward_autotune',
        'selected':best['batch'],'samples':observations}

def _v53_flag_present(flags, name):
    return any(f==name or f.startswith(name+'=') for f in flags)

def _v53_defaults(cfg, flags):
    # v53 specific recommended capacities. Explicit v52 options remain honored;
    # using the same CLI names keeps this script fully standalone.
    for name,value in (('v52_ti_raw_batch',96),('v52_max_batches',5),
                       ('v52_max_oxygen_entrances',4),('v52_max_scale',4.)):
        flag='--'+name.replace('_','-')
        if not _v53_flag_present(flags,flag):setattr(cfg,name,value)
    if (cfg.v53_autotune_max_batch<1 or cfg.v53_raw_factor<=1 or
        cfg.v53_novelty_pool<2 or cfg.v53_novelty_loss_abs<0 or
        cfg.v53_novelty_loss_rel<0):
        raise ValueError('Invalid v53 batch/novelty settings')
    if cfg.v53_plan_only and cfg.v53_dfs:
        # This is harmless: plan-only takes precedence.
        pass
    if cfg.v52_dfs or cfg.v51_ti_dynamics or cfg.full_search or cfg.v50_ab:
        raise ValueError('v53 DFS is exclusive with other generation modes')
    if cfg.v50_allow_resume:
        raise ValueError('v53 always starts fresh; existing outputs cannot be reused')
    return _v52_require(cfg)

def _v53_capacity(cfg,free_ti,ti_orbits,cell_dof):
    """Complexity-scaled *evaluation* capacity, rounded rather than ceiling-clipped.

    Preserve distinct work allocations for closely spaced Wyckoff complexities.
    No time limit or chemically-complete target affects the allowance.
    """
    scale=min(float(cfg.v52_max_scale),1+.22*free_ti+.30*max(ti_orbits-1,0)
              +.12*max(cell_dof-1,0))
    raw_batches=max(1,int(math.floor(cfg.v52_base_batches*scale+.5)))
    raw_oxygen=max(1,int(math.floor(cfg.v52_oxygen_entrances*math.sqrt(scale)+.5)))
    ti_batches=min(cfg.v52_max_batches,raw_batches)
    oxygen=min(cfg.v52_max_oxygen_entrances,raw_oxygen)
    # Promotions are a per-basin subset of incomplete oxygen entrances.
    desired_global=max(1,int(math.floor(cfg.v52_global_proposals*math.sqrt(scale)+.5)))
    glob=min(oxygen,desired_global)
    return {'complexity_scale':round(scale,3),'free_ti':int(free_ti),
            'ti_orbits':int(ti_orbits),'cell_dof':int(cell_dof),
            'ti_batches':ti_batches,'oxygen_entrances':oxygen,
            'global_promotions':glob,
            'ti_batch_cap_hit':raw_batches>ti_batches,
            'oxygen_cap_hit':raw_oxygen>oxygen,
            'promotions_limited_by_entrances':desired_global>glob}

def _v53_plan(cfg):
    from pyxtal.symmetry import Group
    plan=[];counts=[]
    for sg in parse_sgs(cfg.v52_sgs):
        g=Group(sg);num=0
        limit=cfg.v52_max_wp_per_sg or cfg.v52_enumeration_cap
        for tok in itertools.islice(v48_iter_wyckoff(sg,cfg.v52_ti_count,cfg.v52_max_orbits),limit+1):
            if num>=limit:
                if not cfg.v52_max_wp_per_sg:
                    raise RuntimeError(f'SG{sg}: enumeration exceeded {limit}; raise '
                                       '--v52-enumeration-cap or limit --v52-max-orbits')
                break
            cap=_v53_capacity(cfg,sum(int(g[i].get_dof()) for i in tok),len(tok),
                              len(lattice_parameters(g.lattice_type)))
            plan.append({'sg':int(sg),'wp':tuple(tok),'capacity':cap})
            num+=1
        counts.append((int(sg),num))
    return plan,counts

def _v53_opt_width(t,lib,objective,cfg):
    """Tune GPU throughput independently of the number of raw Ti entrances."""
    if cfg.v52_ti_opt_batch:
        width=int(cfg.v52_ti_opt_batch)
        return width,{'mode':'explicit','selected':width,'samples':[]}
    clone=copy.copy(cfg)
    clone.v52_ti_raw_batch=min(cfg.v52_ti_raw_batch,cfg.v53_autotune_max_batch)
    return _v52_autotune_opt_batch(t,lib,objective,clone)

def _v53_distance(a,b):
    if len(a['ti'])!=len(b['ti']):return float('inf')
    cell_delta=cell_metric_discrepancy(a['cell'],b['cell'])
    # Structures can differ by both lattice metric and Ti positions.
    if cell_delta>.12:return cell_delta/.025
    rms=periodic_match_rms(a['ti'],b['ti'],a['cell'])
    return max(cell_delta/.025,rms/.12)

def _v53_suppress_inner_logs(enable):
    """Suppress only repetitive legacy search diagnostics, never summaries."""
    if not enable:return lambda:None
    global log
    original=log
    def quiet(*items):
        first=str(items[0]).strip() if items else ''
        if first.startswith(('original v22 oxygen:','chemistry Ti comparison groups:',
                             'chemistry O comparison groups:')):
            return
        original(*items)
    log=quiet
    def restore():
        global log
        log=original
    return restore

def _v54_matcher(*,prototype=False,ti_only=False):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    if prototype:
        # Loose comparison is an ANNOTATION ONLY; never used to discard results.
        return StructureMatcher(ltol=.20,stol=.32,angle_tol=7.,primitive_cell=True,
                                scale=True,attempt_supercell=True)
    if ti_only:
        return StructureMatcher(ltol=.045,stol=.10,angle_tol=2.5,primitive_cell=True,
                                scale=False,attempt_supercell=True)
    return StructureMatcher(ltol=.05,stol=.12,angle_tol=2.5,primitive_cell=True,
                            scale=False,attempt_supercell=True)

def _v54_ti_structure(basin):
    from pymatgen.core import Structure,Lattice
    return Structure(Lattice(np.asarray(basin['cell'],float)),
        ['Ti']*len(basin['ti']),np.asarray(basin['ti'],float)%1.,
        coords_are_cartesian=False)

def _v54_fit(matcher,a,b):
    try:return bool(matcher.fit(a,b))
    except (ValueError,RuntimeError,TypeError,IndexError):return False

def _v54_bucket_vol(cell,n):
    return abs(float(np.linalg.det(cell)))/max(1,n)

def _v54_pair_profile(basin):
    """Rotation, permutation and origin insensitive PBC pair-distance sketch.

    Only a very loose dissimilarity prefilter for *strict* Ti matching; final
    identity is always determined by crystallographic matching.
    """
    cache=basin.get('_v54_pair_profile_cache')
    if cache is not None:return cache
    xyz=np.asarray(basin['ti'],float)%1.
    if len(xyz)<2:return np.array([],float)
    delta=xyz[:,None,:]-xyz[None,:,:]
    delta-=np.rint(delta)
    dv=(delta[:,:,None,:]+shifts_np(1)[None,None,:,:])@np.asarray(basin['cell'],float)
    d=np.linalg.norm(dv,axis=-1).min(axis=-1)
    profile=np.sort(d[np.triu_indices(len(xyz),1)])
    basin['_v54_pair_profile_cache']=profile
    return profile

def _v54_raw_unique(t,raw):
    """Cheap, same-Wyckoff raw geometry dedup, before Ti GPU optimization.

    Do not identify different cells just because raw parameter vectors happen
    to coincide. Match periodic Ti sites and metric to discard *actual* repeats.
    This is deliberately conservative and does not stand in for global matching.
    """
    with torch.no_grad():
        x=torch.as_tensor(raw,dtype=DTYPE,device=t.blocks[0]['A'].device)
        cells=make_lattice(t,x)[0].detach().cpu().numpy()
        fracs=expand_ti(t,x)[0].detach().cpu().numpy()%1.
    uniq=[]
    for i in range(len(raw)):
        dup=False
        for j in uniq:
            if cell_metric_discrepancy(cells[i],cells[j])>.008:continue
            if periodic_match_rms(fracs[j],fracs[i],cells[j])<.055:
                dup=True;break
        if not dup:uniq.append(i)
    return np.asarray(uniq,dtype=int)

def _v54_ti_batch(t,lib,objective,args,cfg,seed):
    """v53 Ti physics with a measured raw-geometry dedup before GPU work."""
    raw=initial_raw(t,lib,cfg.v52_ti_raw_batch,seed)
    indices=_v54_raw_unique(t,raw)
    distinct=np.asarray(raw[indices],float)
    chosen,report=_v52_raw_filter(t,lib,distinct,
                                  min(cfg.v52_ti_opt_batch,len(distinct)))
    gpu=torch.as_tensor(chosen,dtype=DTYPE,device=objective.device)
    a=time.perf_counter()
    natural,_,na=run_optimizer(objective,t,gpu,'natural',cfg.full_natural_steps,
        args.natural_lr,progress=0,
        patience=args.stop_patience or cfg.full_natural_steps+1,
        min_steps=args.minimum_steps,early_tol=args.stop_tolerance)
    b=time.perf_counter()
    learned,_,le=run_optimizer(objective,t,natural,'learned',cfg.full_learned_steps,
        args.learned_lr,progress=0,
        patience=args.stop_patience or cfg.full_learned_steps+1,
        min_steps=args.minimum_steps,early_tol=args.stop_tolerance)
    c=time.perf_counter()
    groups,losses,details=group_basins(learned,objective,t,args,None)
    basins=[]
    for rank,g in enumerate(groups,1):
        j=int(g['representative'])
        basins.append({'rank':rank,
            'ti':details['ti_frac'][j].detach().cpu().numpy().copy()%1.,
            'cell':details['cell'][j].detach().cpu().numpy().copy(),
            'loss':float(losses[j]),'members':len(g['members']),
            'min_TiTi_A':float(details['minimum_titi_A'][j]),
            'volume_per_Ti_A3':float(details['volume_per_ti_A3'][j])})
    report['raw_count']=len(raw)
    return basins,{'raw':report,'raw_unique':len(indices),
        'raw_duplicates':len(raw)-len(indices),'found_basins':len(groups),
        'natural_iterations':na['steps_executed'],
        'learned_iterations':le['steps_executed'],
        'natural_seconds':b-a,'learned_seconds':c-b,
        'natural_work':na.get('v50_forward_start_steps'),
        'learned_work':le.get('v50_forward_start_steps')}

class _V54TiBank:
    """Cross-SG/WP Ti identity audit; no Ti-only match is a full TiO2 proof.

    Visit the entire optimized basin population (not only retained parents).
    Cross-SG Ti matches are reported, but are not automatically pruned because
    the SAME Ti framework can admit DIFFERENT valid O-sharing networks.
    """
    def __init__(self):
        self.matcher=None;self.known=[];self.selected=[];self.groups=0

    def _fit_to(self,basin,old):
        if len(basin['ti'])!=len(old['ti']):return False
        v=_v54_bucket_vol(basin['cell'],len(basin['ti']))
        w=_v54_bucket_vol(old['cell'],len(old['ti']))
        if abs(v-w)/max(v,w,1.e-10)>.13:return False
        # Conservative necessary condition for near-identical Ti geometry.
        # Large changes of the shortest Ti pair cannot be strict duplicates.
        amin=basin.get('min_TiTi_A');bmin=old.get('min_TiTi_A')
        if amin is not None and bmin is not None:
            if abs(float(amin)-float(bmin))>.40:return False
        a_pair=_v54_pair_profile(basin);b_pair=_v54_pair_profile(old)
        if len(a_pair)==len(b_pair) and len(a_pair):
            if float(np.median(np.abs(a_pair-b_pair)))>.65:return False
        # Prefer the inexpensive same-coordinate-system test first.
        if cell_metric_discrepancy(basin['cell'],old['cell'])<.025:
            if periodic_match_rms(basin['ti'],old['ti'],basin['cell'])<.10:
                return True
        if self.matcher is None:self.matcher=_v54_matcher(ti_only=True)
        if '_v54_structure_cache' not in basin:
            basin['_v54_structure_cache']=_v54_ti_structure(basin)
        return _v54_fit(self.matcher,basin['_v54_structure_cache'],old['_structure'])

    def examine_batch(self,basins,sg,wp,batch):
        """Register each optimized basin and return novelty accounting.
        A match is classified according to where it first appeared.
        """
        stats=collections.Counter(); annotated=[]
        for basin in basins:
            matched=None
            for old in self.known:
                if self._fit_to(basin,old):matched=old;break
            status='new'
            if matched is not None:
                status=('same_WP' if matched['sg']==sg and matched['wp']==tuple(wp)
                        else 'cross_WP' if matched['sg']==sg else 'cross_SG')
            else:
                self.known.append({**basin,'sg':sg,'wp':tuple(wp),'batch':batch,
                                   '_structure':basin.get('_v54_structure_cache') or _v54_ti_structure(basin)})
            stats[status]+=1
            annotated.append((basin,status,matched))
        return stats,annotated

    def match_selected(self,basin,sg,wp):
        for old in self.selected:
            if self._fit_to(basin,old):
                return old
        self.selected.append({**basin,'sg':sg,'wp':tuple(wp),
                              '_structure':basin.get('_v54_structure_cache') or _v54_ti_structure(basin)})
        return None

def _v59_quality_key(q):
    return (float(q['maxTi']),float(q['meanTi']),float(q['minTi']),
            -int(q['goodTi']),-int(q['goodO']),float(q['sharing']),
            float(q['maxO']),float(q['meanO']))

class _V59TopArchive:
    """Globally unique Ti_nO_2n final candidates, top K per initial SG.

    No chemical coordination acceptance gate. All rows must have finite
    coordinates, finite per-Ti metrics and exact TiO2 composition to qualify
    as structures. The matcher determines physical uniqueness. A candidate
    is never written to disk until write_final() after all searches finish.
    A cross-SG duplicate retains the first encountered initial-SG provenance;
    improved same-SG representatives replace their earlier version.
    """
    def __init__(self, capacity, expected_ti, symprec=.055, matcher=None,
                 structure_factory=None):
        if int(capacity)<1:raise ValueError('max-unique-per-sg must be >=1')
        self.capacity=int(capacity)
        self.expected_ti=int(expected_ti)
        self.symprec=float(symprec)
        self.records=[]
        self.matcher=matcher
        self.structure_factory=structure_factory or _v52_structure
        self.stats=collections.Counter()
        # A completed initial SG can never gain/change another representative.
        # Later SGs may be globally deduplicated against it, but cannot replace it.
        self.finalized_sgs=set()
        self.written_paths=set()

    @staticmethod
    def key(record):
        q=record['q']
        return _v59_quality_key(q)+(str(record['name']),)

    def count(self,sg):
        return sum(1 for r in self.records if r['sg']==int(sg))

    def by_sg(self,sg):
        return sorted((r for r in self.records if r['sg']==int(sg)),key=self.key)

    def consider(self,row,name,sg,wp,stage):
        if int(sg) in self.finalized_sgs:
            raise RuntimeError(f'Cannot modify finalized SG{int(sg):03d}')
        ti=np.asarray(row['ti'],float)
        cell=np.asarray(row['cell'],float)
        oxygen=np.asarray(row['answer']['oxygen'],float)
        q=dict(row['q'])
        # Stoichiometry and finite coordinates are representational validity,
        # not Ti-CN6 or O-CN3 chemical acceptance thresholds.
        if (len(ti)!=self.expected_ti or len(oxygen)!=2*self.expected_ti or
            ti.shape!=(self.expected_ti,3) or oxygen.shape!=(2*self.expected_ti,3) or
            cell.shape!=(3,3) or not np.isfinite(ti).all() or
            not np.isfinite(oxygen).all() or not np.isfinite(cell).all() or
            abs(float(np.linalg.det(cell)))<1e-8 or
            any(not np.isfinite(float(q[k])) for k in
                ('maxTi','meanTi','minTi','maxO','meanO','sharing'))):
            self.stats['invalid_geometry_or_composition']+=1
            return 'invalid',None
        record={'sg':int(sg),'wp':tuple(wp),'name':str(name),'stage':stage,
                'q':q,'cell':cell.copy(), 'ti':ti.copy(),
                'oxygen':oxygen.copy()}
        # Only build a structure if there is any possibility of saving it;
        # a worse-than-worst result cannot enter a full SG archive.
        own=self.by_sg(sg)
        if len(own)>=self.capacity and self.key(record)>=self.key(own[-1]):
            self.stats['below_cutoff']+=1
            return 'below_cutoff',None
        structure=self.structure_factory(row)
        record['structure']=structure
        if self.matcher is None:self.matcher=_v54_matcher()
        for old in self.records:
            if _v54_fit(self.matcher,structure,old['structure']):
                if old['sg']!=int(sg):
                    self.stats['cross_sg_duplicate']+=1
                    return 'cross_sg_duplicate',old
                if self.key(record)<self.key(old):
                    self.records[next(i for i,x in enumerate(self.records) if x is old)]=record
                    self.stats['improved_duplicate']+=1
                    return 'improved_duplicate',record
                self.stats['duplicate']+=1
                return 'duplicate',old
        if len(own)>=self.capacity:
            self.records=[x for x in self.records if x is not own[-1]]
            self.stats['evicted_lower_rank']+=1
        self.records.append(record)
        self.stats['inserted']+=1
        return 'ranked',record

    @staticmethod
    def symmetry_details(record,symprec):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        a=SpacegroupAnalyzer(record['structure'],symprec=symprec)
        return int(a.get_space_group_number()),len(a.get_primitive_standard_structure())

    def progress(self,sg,label='PROGRESS'):
        """Concise scientific quality report, not a chemical acceptance gate."""
        rows=self.by_sg(sg)
        if not rows:
            log(f'  SG{sg:03d} {label}: ranked=0/{self.capacity} '
                'Ti_O_site_quality_full=0 best_Ti_loss_max=NA')
            return
        best=rows[0]['q'];worst=rows[-1]['q']
        full=sum(int(r['q']['goodTi']==self.expected_ti and
                     r['q']['goodO']==2*self.expected_ti) for r in rows)
        log(f'  SG{sg:03d} {label}: ranked={len(rows)}/{self.capacity} '
            f'Ti_O_site_quality_full={full}/{len(rows)} '
            f'best_Ti_loss_max={best["maxTi"]:.6g} '
            f'worst_retained_Ti_loss_max={worst["maxTi"]:.6g} '
            f'best_rank_Ti={best["goodTi"]}/{self.expected_ti} '
            f'best_rank_O={best["goodO"]}/{2*self.expected_ti} '
            '(coordination diagnostic only; 0 complete is permitted)')

    def _cif_path(self,output,record,rank):
        name=f'SG{record["sg"]:03d}_R{rank:02d}_{record["name"]}.cif'
        return Path(output)/'cif'/name

    def write_sg_final(self,output,sg):
        """Export only AFTER the given SG has exhausted its capacity or reached K.

        These SG-final entries never change later: subsequent SGs are allowed
        to deduplicate against them but never replace their initial-SG owner.
        No local/global intermediate structure CIF is written.
        """
        sg=int(sg)
        if sg in self.finalized_sgs:
            return 0
        rows=self.by_sg(sg)
        self.progress(sg,'FINAL QUALITY')
        for rank,r in enumerate(rows,1):
            path=self._cif_path(output,r,rank)
            row={'ti':r['ti'],'cell':r['cell'],
                 'answer':{'oxygen':r['oxygen']}}
            _v44_atomic_cif(row,path)
            self.written_paths.add(str(path))
        self.finalized_sgs.add(sg)
        log(f'  SG{sg:03d} FINAL CIFs WRITTEN: {len(rows)} to '
            f'{Path(output)/"cif"} (SG finalized; no intermediate CIFs)')
        return len(rows)

    def write_final(self,output):
        """All-SG stdout ranking, and a final flush for any unexported SGs."""
        rows=sorted(self.records,key=lambda r:
                    (self.key(r)[0],self.key(r)[1],self.key(r)[2],r['sg'],r['name']))
        for sg in sorted({r['sg'] for r in rows}):
            if sg not in self.finalized_sgs:
                self.write_sg_final(output,sg)
        log('FINAL STRUCTURES (ascending Ti_loss_max; final SG determined from full Ti+O)')
        log(f'FINAL EXPECTED COMPOSITION: Ti={self.expected_ti} O={2*self.expected_ti} per candidate; N_good fields are counts')
        log('initial_spgr final_spgr N_atom_primitive Ti_loss_min Ti_loss_max '
            'Ti_loss_avg N_good_Ti N_good_O')
        if not rows:log('  (no finite TiO2 candidates)')
        for r in rows:
            try:
                final_sg,n_prim=self.symmetry_details(r,self.symprec)
            except Exception as exc:
                final_sg='unknown';n_prim='unknown'
                log(f'  SYMMETRY WARNING {r["name"]}: {exc!r}')
            q=r['q']
            log(f'{r["sg"]:03d} {final_sg} {n_prim} '
                f'{q["minTi"]:.8g} {q["maxTi"]:.8g} {q["meanTi"]:.8g} '
                f'{q["goodTi"]} {q["goodO"]}')
        log(f'FINAL CIFs WRITTEN: {len(self.written_paths)} to {Path(output)/"cif"} '
            '(written once per finalized SG; no intermediate CIFs)')
        return rows

def _v54_summary(tag,c):
    log(f'{tag} | raw={int(c["raw"])} opt={int(c["opt"])} '
        f'Ti_basins={int(c["basins"])} pooled={int(c["pool_unique"])} '
        f'Ti_selected={int(c["selected"])} O_local={int(c["local_calls"])} '
        f'O_global={int(c["global_calls"])} '
        f'chemistry_full_local={int(c["local_complete"])} '
        f'archive_new={int(c["archive_new"])} archive_improved={int(c["archive_improved"])} '
        f'archive_duplicates={int(c["archive_duplicate"])} '
        f'Ti_s={c["ti_seconds"]:.1f} '
        f'Olocal_s={c["oxygen_local_seconds"]:.1f} '
        f'Oglobal_s={c["oxygen_global_seconds"]:.1f}')

def _v54_plan(cfg):
    plan,counts=_v53_plan(cfg)
    return plan,counts

def _v54_defaults(cfg,flags):
    _v53_defaults(cfg,flags)
    if (cfg.v54_wp_per_sg<0 or
        cfg.v54_o_repeat_policy not in ('observe','skip-global')):
        raise ValueError('Invalid v54 widths or oxygen repeat policy')
    # v54 broadening is explicitly across WP, not per-start or per-oxygen search.
    if not _v53_flag_present(flags,'--v52-max-wp-per-sg'):
        cfg.v52_max_wp_per_sg=cfg.v54_wp_per_sg
    return _v52_require(cfg)

class _V55TiBank(_V54TiBank):
    """One representative per strict Ti equivalence class, indexed by volume.

    The index rejects comparisons only if the unchanged v54 _fit_to would
    *necessarily* reject them. Final identity still uses the original exact
    periodic test + pymatgen matcher; a bucket match alone is never a dedup.
    """
    def __init__(self):
        super().__init__()
        self._by_count={}
        self.prefilter_comparisons=0
        self.exact_comparisons=0
        self.index_seconds=0.0

    def examine_batch(self,basins,sg,wp,batch):
        import bisect
        tally=collections.Counter();tagged=[]
        for basin in basins:
            n=len(basin['ti'])
            vol=_v54_bucket_vol(basin['cell'],n)
            rec=self._by_count.setdefault(n,{'vol':[],'data':[]})
            t0=time.perf_counter()
            # The original exclusion test is abs(v-w)/max(v,w) > .13.
            # Its exact necessary interval is [.87*v, v/.87].
            i0=bisect.bisect_left(rec['vol'],.87*vol)
            i1=bisect.bisect_right(rec['vol'],vol/.87)
            self.index_seconds+=time.perf_counter()-t0
            matched=None
            self.prefilter_comparisons+=i1-i0
            for j in range(i0,i1):
                self.exact_comparisons+=1
                old=rec['data'][j]
                if self._fit_to(basin,old):
                    matched=old
                    break
            status='new'
            if matched is None:
                representative={**basin,'sg':sg,'wp':tuple(wp),'batch':batch,
                   '_structure':basin.get('_v54_structure_cache') or _v54_ti_structure(basin)}
                self.known.append(representative)
                at=bisect.bisect_right(rec['vol'],vol)
                rec['vol'].insert(at,vol)
                rec['data'].insert(at,representative)
            else:
                status=('same_WP' if matched['sg']==sg and matched['wp']==tuple(wp)
                        else 'cross_WP' if matched['sg']==sg else 'cross_SG')
            tally[status]+=1
            tagged.append((basin,status,matched))
        return tally,tagged

def _v56_defaults(cfg,flags):
    # Keep the wider WP catalog, existing chemical parameters, and strict
    # resume prohibition. v56 deliberately makes Ti *depth* a fixed pilot
    # capacity; WP catalog still carries its symmetry-complexity annotations.
    sgs=_v54_defaults(cfg,flags)
    for name,value in (('v52_ti_raw_batch',cfg.v56_raw_per_batch),
                       ('v52_ti_opt_batch',cfg.v56_opt_per_batch)):
        if not _v53_flag_present(flags,'--'+name.replace('_','-')):
            setattr(cfg,name,value)
    if (cfg.v56_ti_batches<1 or cfg.v56_ti_quality<1 or cfg.v56_ti_reserve<0
        or cfg.v56_pool_window<cfg.v56_ti_quality+cfg.v56_ti_reserve
        or cfg.v56_global_match_cap<cfg.v56_ti_quality+cfg.v56_ti_reserve
        or cfg.v56_repeat_ti_o_reserve not in (0,1)
        or cfg.v56_o_global_per_wp<0 or cfg.v56_raw_per_batch<1
        or cfg.v56_opt_per_batch<0):
        raise ValueError('Invalid v56 shared Ti pool, oxygen width or matching allowance')
    cfg.v52_base_batches=1
    cfg.v52_max_batches=max(cfg.v52_max_batches,cfg.v56_ti_batches)
    cfg.v52_oxygen_entrances=1
    cfg.v52_max_oxygen_entrances=1
    cfg.v52_global_proposals=1  # The v52 chemistry config requires >=1; v56 controls promotions.
    _v52_require(cfg)
    return sgs

def _v56_plan(cfg):
    plan,counts=_v54_plan(cfg)
    for p in plan:
        c=p['capacity']
        c['ti_batches']=cfg.v56_ti_batches
        c['oxygen_entrances']=1
        c['global_promotions']=0 if cfg.v55_local_only else cfg.v56_o_global_per_wp
        c['ti_batch_cap_hit']=False
        c['oxygen_cap_hit']=False
    return plan,counts

def _v56_merge_pool(batch_basins):
    """Cheap same-WP cross-batch Ti coalescence; keep best-loss representative.

    A loose/global crystallographic matcher is NOT applied here. Only the
    original same-basis tight PBC match is used; any missed equivalences are
    verified in the selected-candidate global matcher downstream.
    """
    merged=[];dup=0
    for basin in sorted(batch_basins,key=lambda x:(float(x['loss']),x['_v56_batch'],x['rank'])):
        if _v52_ti_duplicate(basin['ti'],basin['cell'],merged):
            dup+=1
        else:
            merged.append(basin)
    return merged,dup

def _v56_global_priority(local):
    """Prefer meaningful local chemistry progress; tie-break on Ti loss.

    This is ranking only; never a hard sharing/Ti-loss rejection threshold.
    In particular, the v55 SG15 rescue (7/12 Ti, 19/24 O, 0.456 A) remains
    eligible for promotion despite relatively poor initial sharing.
    """
    row,basin,_,_,_=local
    return (chemical_rank(row['q']),float(basin['loss']))

def _v57_defaults(cfg,flags):
    cfg.v56_ti_batches=1
    cfg.v56_raw_per_batch=cfg.v57_raw_per_batch
    cfg.v56_opt_per_batch=cfg.v57_opt_per_batch
    cfg.v56_ti_quality=max(1,cfg.v57_ti_quality_total)
    cfg.v56_ti_reserve=cfg.v57_ti_reserve
    cfg.v56_o_global_per_wp=cfg.v57_o_global_per_wp
    cfg.v56_pool_window=max(cfg.v56_pool_window,cfg.v57_ti_quality_total+cfg.v57_ti_reserve)
    cfg.v56_global_match_cap=max(cfg.v56_global_match_cap,cfg.v57_ti_quality_total+cfg.v57_ti_reserve)
    if (cfg.v57_opt_per_batch<1 or cfg.v57_raw_per_batch<cfg.v57_opt_per_batch
        or cfg.v57_ti_quality_total<1 or cfg.v57_ti_reserve<0 or
        cfg.v57_o_global_per_wp<0 or cfg.v57_reserve_loss_abs<0 or
        cfg.v57_reserve_loss_rel<0):
        raise ValueError('Invalid v57 Ti/oxygen width, raw-to-opt ratio or reserve loss allowance')
    sgs=_v56_defaults(cfg,flags)
    # v57 flags supersede legacy v52/v56 widths; this makes A and B
    # genuinely comparable on identical Ti batch and raw entrance settings.
    cfg.v52_ti_raw_batch=cfg.v57_raw_per_batch
    cfg.v52_ti_opt_batch=cfg.v57_opt_per_batch
    return sgs

def _v57_plan(cfg):
    return _v56_plan(cfg)

def _v57_select_shared(pool,cfg,probe,geometry_ok=None,distance=None):
    """Quality-first, balanced by optimization batch; global Ti repeats allowed.

    Select up to four quality basins (A: four from single batch; B: two
    from each independent batch) and one distance-maximizing novelty reserve.
    Tight local basin duplicates were already collapsed by _v56_merge_pool.
    Cross-WP/SG duplicates are annotated, but sent to O to permit genuinely
    different three-Ti sharing networks. No fixed absolute Ti-loss gate.
    """
    if geometry_ok is None:
        bounds=argc_screen(cfg)
        geometry_ok=lambda b:screen_ti_geometry(b['ti'],b['cell'],bounds)[0]
    if distance is None:distance=_v53_distance
    stat=collections.Counter();valid=[]
    for b in sorted(pool,key=lambda b:(float(b['loss']),b['_v56_batch'],b['rank'])):
        if geometry_ok(b):valid.append(b)
        else:stat['geometry_rejected']+=1
    stat['geometry_valid']=len(valid)
    batches=cfg.v56_ti_batches
    quality_total=cfg.v57_ti_quality_total
    quotas=[quality_total//batches+int(bi<quality_total%batches) for bi in range(batches)]
    chosen=[];ids=set()
    # Explicitly preserve top two Ti minima from each batch in B; this
    # restores a route for the v54 SG15_B00_T02 successes.
    for bi,quota in enumerate(quotas):
        local=[b for b in valid if b['_v56_batch']==bi]
        for b in local[:quota]:
            if id(b) not in ids:
                chosen.append((b,'quality'));ids.add(id(b))
    # Sparse batch: backfill its missing quality allowance with the best
    # remaining Ti minima across the full shared pool.
    for b in valid:
        if len(chosen)>=quality_total:break
        if id(b) not in ids:
            chosen.append((b,'quality'));ids.add(id(b))
    # One novelty reserve, bounded by a relative/additive loss window.
    # Never remove dictionaries using equality (their numpy array elements
    # are not scalar-comparable).
    if chosen and cfg.v57_ti_reserve:
        anchor=max(float(b['loss']) for b,_ in chosen)
        threshold=anchor+max(cfg.v57_reserve_loss_abs,
             cfg.v57_reserve_loss_rel*max(abs(anchor),1.e-9))
        remaining=[b for b in valid if id(b) not in ids and float(b['loss'])<=threshold]
        for _ in range(cfg.v57_ti_reserve):
            if not remaining:break
            b=max(remaining,key=lambda x:(
                min(float(distance(x,z)) for z,_ in chosen),-float(x['loss'])))
            remaining=[x for x in remaining if x is not b]
            chosen.append((b,'reserve'));ids.add(id(b))
    # Match just the O-bound shortlist, for DIAGNOSTICS ONLY. Repeated
    # Ti frameworks remain oxygen parents; full Ti+O dedup runs after SCF.
    for b,kind in chosen:
        status=probe(b)
        b['_v57_global_status']=status
        stat['global_probes']+=1
        stat['global_'+status]+=1
        if status!='new':stat['repeat_O_parents']+=1
    stat['quality_selected']=sum(k=='quality' for _,k in chosen)
    stat['reserve_selected']=sum(k=='reserve' for _,k in chosen)
    stat['selected']=len(chosen)
    stat['not_sent_to_global']=len(pool)-stat['global_probes']
    stat['novelty_window_size']=sum(1 for b in valid if chosen and
        float(b['loss'])<=max(float(z['loss']) for z,_ in chosen))
    return chosen,stat

def _v57_print_diagnostics(label,c):
    _v54_summary(label,c)
    log(f'{label} POOL: cross_batch_collapsed={int(c["pool_duplicates"])} '
        f'pooled_unique={int(c["pool_unique"])} geometry_valid={int(c["pool_valid"])} '
        f'global_Ti_compared={int(c["global_probes"])} '
        f'global_Ti_repeated_BUT_EVALUATED={int(c["repeat_O_parents"])} '
        f'pool_not_globally_compared={int(c["global_unchecked"])} '
        f'quality_parents={int(c["quality_selected"])} '
        f'novelty_parents={int(c["reserve_selected"])} '
        f'global_O_promotions={int(c["global_calls"])} '
        f'global_O_budget={int(c["global_budget"])} '
        f'promotions_Ti_first={int(c["promotion_Ti_first"])} '
        f'promotions_O_first={int(c["promotion_O_first"])} '
        f'promotions_Ti_fallback={int(c["promotion_Ti_fallback"])}')

def _v57ap_oxygen_priority(candidate):
    """Rank local oxygen-network readiness, not local Ti-site readiness.

    The number of chemically correct O sites is the primary channel; sharing
    and O loss resolve ties.  Critically, 0/12 Ti with 24/24 good O sites
    must remain ahead of 1/12 Ti with 13/24 O for the oxygen-focused slot.
    This is a PROMOTION rank only: the strict complete-chemistry validator
    and global complete Ti+O dedup matcher remain unchanged.
    """
    row,basin,_,_,_=candidate
    q=row['q']
    return (-int(q['goodO']),-int(q.get('nearO',0)),
            round(float(q['sharing']),7),round(float(q['meanO']),7),
            -int(q['goodTi']),round(float(q['meanTi']),7),float(basin['loss']))

def _v57ap_select_promotions(incomplete,budget):
    """Select DISTINCT candidates: first Ti-ready; second O-ready.

    A single local candidate is never promoted twice.  Extra optional slots
    (if explicitly requested) revert to standard Ti-first global priority.
    At a two-slot budget this performs exactly 1 Ti-first + 1 O-first.
    """
    if budget<=0 or not incomplete:return []
    remaining=list(incomplete)
    result=[]
    priorities=(('Ti_first',_v56_global_priority),
                ('O_first',_v57ap_oxygen_priority))
    for name,priority in priorities:
        if len(result)>=budget or not remaining:break
        winner=min(remaining,key=priority)
        result.append((name,winner))
        remaining=[row for row in remaining if row is not winner]
    while len(result)<budget and remaining:
        winner=min(remaining,key=_v56_global_priority)
        result.append(('Ti_fallback',winner))
        remaining=[row for row in remaining if row is not winner]
    return result

def _v59_run_dfs(cfg,ti_flags,flags):
    if ti_flags:raise ValueError('Unrecognized Ti arguments: '+' '.join(ti_flags))
    sgs=_v57_defaults(cfg,flags)
    plan,sg_plan=_v57_plan(cfg)
    log('JULIETTE v59.2 | RANKED PER-SG TOP-K TiO2 | A+ 4 QUALITY + 1 NOVELTY | 1 Ti-first + 1 O-first global/WP')
    if cfg.v57_profile=='APLUS':
        log('A+ OXYGEN POLICY: one Ti-first global promotion plus one oxygen-first global promotion (distinct parents); no Ti/O chemistry gate changes')
    log('SGs=',sgs,'Ti=',cfg.v52_ti_count,'planned_WPs=',len(plan),
        'global_O_enabled=',not cfg.v55_local_only,
        'global_Ti_repeats_for_O=ENABLED',
        'oxygen_CN3_optimizer_target=',bool(cfg.target_o_cn),
        'oxygen_CN3_final_diagnostic=ENABLED',
        'max_unique_per_initial_SG=',cfg.max_unique_per_sg,
        'stop_SG_at_cap=',cfg.stop_sg_at_cap,
        'time_stop=NONE; FINAL SG CIF ONLY')
    for sg,n in sg_plan:log(f'  PLAN SG{sg:03d}: Wyckoff combinations={n}')
    for p in plan:
        c=p['capacity']
        log(f'  PLAN SG{p["sg"]:03d} WP={"-".join(map(str,p["wp"]))} '
            f'complexity={c["complexity_scale"]:.2f} '
            f'Ti_batches={c["ti_batches"]} shared_Ti_slots={cfg.v57_ti_quality_total}+{cfg.v57_ti_reserve} '
            f'O_local_per_parent=1 O_global_per_WP={c["global_promotions"]}')
    if cfg.v56_plan_only:return
    _v50_fresh_output(cfg)
    out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    cfg.gpu_workers=1
    try:torch.set_num_threads(max(1,min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))))
    except (RuntimeError,ValueError):torch.set_num_threads(1)
    if not plan:log('No eligible Wyckoff configurations.');return
    start=time.perf_counter()
    ti_args,device,ti_lib,objective=_v48_ti_stage(cfg,[])
    template=template_from_token(plan[0]['sg'],plan[0]['wp'],device)
    width,tune=_v53_opt_width(template,ti_lib,objective,cfg)
    cfg.v52_ti_opt_batch=int(width)
    cfg.v52_ti_raw_batch=max(cfg.v52_ti_raw_batch,
        int(math.ceil(width*cfg.v53_raw_factor)))
    log(f'GPU optimized Ti batch={width} random entrances={cfg.v52_ti_raw_batch} '
        f'Ti batches per WP={cfg.v56_ti_batches} shared selected maximum='
        f'{cfg.v57_ti_quality_total+cfg.v57_ti_reserve}')
    for obs in tune.get('samples',[]):
        if 'combined_start_steps_per_second' in obs:
            log(f'  GPU TUNE batch={obs["batch"]} rate='
                f'{obs["combined_start_steps_per_second"]:.1f} start-steps/s')
    o_lib,bounds,guard,refs=load_local_chemistry(cfg.ref,cfg.oxygen_shell_max_k,
        cfg.accept_A,cfg.hierarchy)
    prepare_v22_calibration(cfg,o_lib,bounds,guard,refs)
    resolve_v44_compute(cfg)
    cfg.v44_fit_backend=configure_v44_fit(cfg,cfg.v44_worker_device)
    localcfg=_v52_oxygen_cfg(cfg,False)
    globalcfg=_v52_oxygen_cfg(cfg,True)
    final=_V59TopArchive(cfg.max_unique_per_sg,cfg.v52_ti_count,cfg.symprec)
    ti_bank=_V55TiBank()
    by_sg=collections.defaultdict(list)
    for p in plan:by_sg[p['sg']].append(p)
    total=collections.Counter();errors=[];done=0
    restore=_v53_suppress_inner_logs(cfg.v53_quiet_inner and not cfg.v53_verbose_inner)
    try:
        for sg in sgs:
            sg_start=time.perf_counter();sgstats=collections.Counter()
            log('\n'+'='*70)
            log(f'SG{sg:03d} START | {len(by_sg[sg])} Wyckoff combinations')
            if not by_sg[sg]:
                log(f'SG{sg:03d} SKIP | no 24-Ti Wyckoff combinations under current orbit cap')
                continue
            processed_sg=0
            for p in by_sg[sg]:
                if cfg.stop_sg_at_cap and final.count(sg)>=cfg.max_unique_per_sg:
                    log(f'SG{sg:03d} CAP REACHED: {final.count(sg)}/{cfg.max_unique_per_sg}; '
                        'remaining Wyckoff combinations skipped')
                    break
                processed_sg+=1
                wp=p['wp'];cap=p['capacity']
                key=f'SG{sg:03d}_WP'+'-'.join(map(str,wp))
                count=collections.Counter();count['global_budget']=cap['global_promotions']
                all_basins=[];incomplete=[]
                wp_start=time.perf_counter()
                log(f'  {key} SHARED POOL | {cap["ti_batches"]} Ti batches first, '
                    f'then <= {cfg.v57_ti_quality_total+cfg.v57_ti_reserve} local O '
                    f'and <= {cap["global_promotions"]} global O across WP')
                try:
                    tpl=template_from_token(sg,wp,device)
                    if tpl.n_ti!=cfg.v52_ti_count:raise RuntimeError('Incorrect Ti count')
                    # Phase I: finish ALL GPU Ti starts before any O.
                    for bi in range(cap['ti_batches']):
                        seed=str_seed(cfg.seed,'v54_Ti',sg,*wp,bi)
                        basins,rep=_v54_ti_batch(tpl,ti_lib,objective,ti_args,cfg,seed)
                        for b in basins:b['_v56_batch']=bi
                        all_basins.extend(basins)
                        count.update({'raw':rep['raw']['raw_count'],
                            'raw_unique':rep['raw_unique'],
                            'raw_duplicates':rep['raw_duplicates'],
                            'screened':rep['raw']['opt_count'],
                            'opt':rep['raw']['opt_count'],
                            'basins':len(basins),'batches':1})
                        ti_s=rep['natural_seconds']+rep['learned_seconds']
                        count['ti_seconds']+=ti_s
                        log(f'    Ti BATCH {bi+1}/{cap["ti_batches"]}: '
                            f'raw={rep["raw"]["raw_count"]} screened={rep["raw"]["opt_count"]} '
                            f'within_batch_basins={len(basins)} '
                            f'best_learned_Ti_loss={min((b["loss"] for b in basins),default=float("nan")):.6g} '
                            f'Ti_s={ti_s:.2f}')
                    # Phase II: cheap WP-level collapse and quality/novelty selection.
                    t_dedup=time.perf_counter()
                    pool,collapsed=_v56_merge_pool(all_basins)
                    count['pool_duplicates']=collapsed
                    count['pool_unique']=len(pool)
                    count['global_unchecked']+=0
                    log(f'    SHARED POOL {key}: input_basins={len(all_basins)} '
                        f'cross_batch_collapse={collapsed} distinct={len(pool)} '
                        f'best_loss={min((b["loss"] for b in pool),default=float("nan")):.6g}')
                    def probe(b):
                        tally,_=ti_bank.examine_batch([b],sg,wp,b['_v56_batch'])
                        return next((k for k,v in tally.items() if v), 'new')
                    selected,selection=_v57_select_shared(pool,cfg,probe)
                    count['ti_dedup_seconds']+=time.perf_counter()-t_dedup
                    count['pool_valid']+=selection['geometry_valid']
                    count['bad_ti_geometry']+=selection['geometry_rejected']
                    count['global_probes']+=selection['global_probes']
                    count['global_unchecked']+=selection['not_sent_to_global']
                    count['ti_global_new']+=selection['global_new']
                    count['quality_selected']+=selection['quality_selected']
                    count['reserve_selected']+=selection['reserve_selected']
                    count['repeat_O_parents']+=selection['repeat_O_parents']
                    for x in ('same_WP','cross_WP','cross_SG'):
                        count['ti_dup_'+x]+=selection['global_'+x]
                        count['global_repeats']+=selection['global_'+x]
                    count['selected']+=len(selected)
                    log(f'    SHARED SELECT {key}: global_probes={selection["global_probes"]}/'
                        f'{cfg.v56_global_match_cap} quality={selection["quality_selected"]} '
                        f'reserve={selection["reserve_selected"]} '
                        f'repeat_Ti_O={selection["repeat_O_parents"]} '
                        f'repeated_global={count["global_repeats"]} '
                        f'not_globally_checked={selection["not_sent_to_global"]}')
                    for b,kind in selected:
                        log(f'      Ti PARENT batch={b["_v56_batch"]+1} '
                            f'within_rank={b["rank"]} Ti_loss={b["loss"]:.7g} '
                            f'selection={kind} Ti_global={b.get("_v57_global_status","unknown")}')
                    # Phase III: exactly one short local oxygen construction per
                    # selected Ti basin; collect ALL candidates before global O.
                    for basin,kind in selected:
                        bi=basin['_v56_batch']
                        ose=str_seed(cfg.seed,'v54_O',sg,*wp,bi,basin['rank'],0)
                        name=f'{key}_B{bi:02d}_T{basin["rank"]:02d}_O00'
                        t0=time.perf_counter();count['local_calls']+=1
                        try:
                            rr,_=propose_v22_oxygen(basin['ti'],basin['cell'],
                                o_lib,bounds,guard,localcfg,ose,starts=1)
                            if not rr:count['empty_local']+=1;continue
                            best=min(rr,key=lambda r:_v59_quality_key(r['q']))

                            # No chemical pass/fail gate: chemistry counters are diagnostics only.
                            dt=time.perf_counter()
                            status,info=final.consider(best,name,sg,wp,'local')
                            count['oxygen_dedup_seconds']+=time.perf_counter()-dt
                            if status=='ranked':count['archive_new']+=1
                            if status=='improved_duplicate':count['archive_improved']+=1
                            if status in ('duplicate','cross_sg_duplicate'):count['archive_duplicate']+=1
                            count['local_O_CN3_sites']+=best['q']['cnO']
                            count['local_O_total_sites']+=best['q']['nO']
                            complete=oxygen_quality_satisfied(best,cfg)
                            count['local_complete']+=int(complete)
                            log(f'      O LOCAL {name} selection={kind} Ti_loss={basin["loss"]:.6g} '
                                f'Ti={best["q"]["goodTi"]}/{len(basin["ti"])} '
                                f'O={best["q"]["goodO"]}/{2*len(basin["ti"])} '
                                f'O_CN3={best["q"]["cnO"]}/{best["q"]["nO"]} '
                                f'Ti_loss_min/avg/max={best["q"]["minTi"]:.6g}/'
                                f'{best["q"]["meanTi"]:.6g}/{best["q"]["maxTi"]:.6g} '
                                f'sharing={best["q"]["sharing"]:.4f}A '
                                f'chemistry_complete_diagnostic={int(complete)} rank_status={status}')
                            # All finite local oxygen structures are eligible for complementary
                            # global refinement, regardless of Ti/O chemical coordination counts.
                            if status!='invalid':
                                incomplete.append((best,basin,ose,name,True))
                        except Exception as exc:
                            count['local_error']+=1;errors.append((name,'local',repr(exc)))
                            log(f'      O LOCAL ERROR {name}: {exc}')
                            if cfg.v52_fail_fast:raise
                        finally:
                            count['oxygen_local_seconds']+=time.perf_counter()-t0
                    # Phase IV: BOUNDED shared promotion allowance per Wyckoff WP,
                    # rather than one per failed Ti framework.
                    ranked=sorted(incomplete,key=_v56_global_priority)
                    count['global_eligible']+=len(ranked)
                    promotions=_v57ap_select_promotions(incomplete,cap['global_promotions'])
                    for strategy,(local,basin,seed,name,unique_o) in promotions:
                        count['global_calls']+=1
                        count['promotion_'+strategy]+=1
                        log(f'      O PROMOTE {strategy} {name} localTi={local["q"]["goodTi"]}/{len(basin["ti"])} '
                            f'localO={local["q"]["goodO"]}/{2*len(basin["ti"])} '
                            f'localSharing={local["q"]["sharing"]:.4f}A')
                        t0=time.perf_counter()
                        try:
                            rr,_=propose_v22_oxygen(basin['ti'],basin['cell'],
                                o_lib,bounds,guard,globalcfg,seed,starts=1)
                            if not rr:count['empty_global']+=1;continue
                            winner=min([local]+rr,key=lambda r:_v59_quality_key(r['q']))

                            dt=time.perf_counter()
                            status,info=final.consider(winner,name+'_G',sg,wp,'global')
                            count['oxygen_dedup_seconds']+=time.perf_counter()-dt
                            count['global_O_CN3_sites']+=winner['q']['cnO']
                            count['global_O_total_sites']+=winner['q']['nO']
                            if status=='ranked':count['archive_new']+=1
                            if status=='improved_duplicate':count['archive_improved']+=1
                            if status in ('duplicate','cross_sg_duplicate'):count['archive_duplicate']+=1
                            log(f'      O GLOBAL {name} promoted_by={strategy} Ti_loss={basin["loss"]:.6g} '
                                f'Ti={winner["q"]["goodTi"]}/{len(basin["ti"])} '
                                f'O={winner["q"]["goodO"]}/{2*len(basin["ti"])} '
                                f'O_CN3={winner["q"]["cnO"]}/{winner["q"]["nO"]} '
                                f'Ti_loss_min/avg/max={winner["q"]["minTi"]:.6g}/'
                                f'{winner["q"]["meanTi"]:.6g}/{winner["q"]["maxTi"]:.6g} '
                                f'sharing={winner["q"]["sharing"]:.4f}A '
                                f'rank_status={status}')
                        except Exception as exc:
                            count['global_error']+=1;errors.append((name,'global',repr(exc)))
                            log(f'      O GLOBAL ERROR {name}: {exc}')
                            if cfg.v52_fail_fast:raise
                        finally:
                            count['oxygen_global_seconds']+=time.perf_counter()-t0
                    done+=1
                except Exception as exc:
                    errors.append((key,'Wyckoff',repr(exc)))
                    log(f'  WP ERROR {key}: {exc}')
                    if cfg.v52_fail_fast:raise
                count['wp_wall_seconds']+=time.perf_counter()-wp_start
                _v57_print_diagnostics('  WP SUMMARY '+key,count)
                final.progress(sg,'LIVE QUALITY')
                sgstats.update(count);total.update(count)
            log(f'SG{sg:03d} SUMMARY: searched_WP={processed_sg}/{len(by_sg[sg])} '
                f'ranked_unique_this_SG={final.count(sg)}/{cfg.max_unique_per_sg} '
                f'global_ranked_total={len(final.records)} '
                f'sg_wall_s={time.perf_counter()-sg_start:.1f}')
            _v57_print_diagnostics('  SG STAGES',sgstats)
            if cfg.write_final_per_sg:
                final.write_sg_final(out,sg)
    finally:
        restore()
    elapsed=time.perf_counter()-start
    log('\n'+'='*70)

    log('JULIETTE v59.2 FINAL SUMMARY — STDOUT ONLY')
    _v57_print_diagnostics('TOTAL WORK',total)
    log(f'FINAL ARCHIVE IN MEMORY: globally_distinct={len(final.records)} '
        f'per_initial_SG_limit={cfg.max_unique_per_sg} '
        f'completed_WP={done}/{len(plan)} errors={len(errors)}')
    log(f'CONTINUOUS CHEMISTRY DIAGNOSTIC: local_O_CN3='
        f'{total["local_O_CN3_sites"]}/{total["local_O_total_sites"]} '
        f'global_O_CN3={total["global_O_CN3_sites"]}/{total["global_O_total_sites"]}')
    log(f'TIME: wall={elapsed:.1f}s Ti_s={total["ti_seconds"]:.1f} '
        f'O_local_s={total["oxygen_local_seconds"]:.1f} '
        f'O_global_s={total["oxygen_global_seconds"]:.1f}')
    for name,stage,error in errors:log(f'  ERROR {name} {stage}: {error}')
    log('FINAL RANKING: lower Ti_loss_max is better; then Ti_loss_avg and Ti_loss_min. '
        'No Ti/O coordination acceptance threshold.')
    final.write_final(out)

# Defaults captured from the successful v57 A+ configuration; only the
# present search mode has CLI controls. Other internal numerical constants
# are retained to keep all chemical and Ti loss functions unchanged.
_CLEAN_DEFAULTS = {'accept_A': 0.1,
 'active_movement_stop': False,
 'adaptive_ti_widths': (2, 4, 8),
 'anneal_cooling': 0.75,
 'anneal_temperature': 0.04,
 'audit_basin_rank': 1,
 'audit_basins': 2,
 'audit_hopping': False,
 'audit_movement_consecutive': 3,
 'audit_movement_stop_A': 0.05,
 'audit_movement_windows': '5,10,20',
 'audit_moving_origin': True,
 'audit_origin_reset_A': 0.25,
 'audit_oxygen': True,
 'audit_oxygen_chem_every': 1,
 'audit_oxygen_starts': 1,
 'audit_starts': 8,
 'audit_sweep_output': None,
 'audit_targets': '74:1,5;15:0,1;42:1,4',
 'audit_ti_count': 12,
 'audit_ti_details_every': 1,
 'basins_per_wyckoff': 2,
 'bf_elites': 2,
 'bf_fresh': 4,
 'bf_generations': 2,
 'bf_improvement_A': 0.01,
 'bf_min_generations': 2,
 'bf_patience': 0,
 'bf_radius_large_A': 0.24,
 'bf_radius_small_A': 0.085,
 'bf_trials': 5,
 'bf_uniform_share': 0.15,
 'bf_warm': 2,
 'calibration_group_min_atoms': 12,
 'calibration_group_min_structures': 3,
 'calibration_min_structures': 4,
 'calibration_quantile': 0.95,
 'candidate_cap': 1500,
 'convergence_audit': False,
 'db': 'tio2.db',
 'early_accept_collision': 1e-08,
 'early_accept_sharing_A': 0.1,
 'early_require_parent_sg': False,
 'feedback_rounds': 2,
 'feedback_steps': 64,
 'full_dedup': True,
 'full_learned_steps': 160,
 'full_max_orbits': 0,
 'full_max_seconds': 0.0,
 'full_max_ti': 96,
 'full_min_titi_A': 2.7,
 'full_natural_steps': 120,
 'full_oxygen_per_job': 12,
 'full_oxygen_strategy': 'adaptive',
 'full_phase': 'auto',
 'full_plan_limit': 2000000,
 'full_plan_only': False,
 'full_refine_per_cycle': 1,
 'full_retry_failed': False,
 'full_scheduling': 'breadth',
 'full_search': False,
 'full_tasks_per_job': 12,
 'full_ti_basins': 2,
 'full_ti_counts': '12',
 'full_ti_policies': 'hop',
 'full_ti_starts': 8,
 'geometry_policy': 'channel_q95',
 'geometry_reference_pull': 0.06,
 'gpu_fit_parity_A': 2e-05,
 'gpu_octa_fit': 'auto',
 'gpu_workers': 'auto',
 'hierarchy': 'ti-first',
 'max_initials': 0,
 'max_wyckoff': 2,
 'min_TiTi_A': 2.7,
 'modes_per_ti': 5,
 'movement_ab': False,
 'near_correct_factor': 1.5,
 'octa_anneal_every': 0,
 'octahedral_proposals': 8,
 'output': 'tio2_v58_clean',
 'oxygen_budget_steps': (50, 150, 300),
 'oxygen_cpu_threads': 1,
 'oxygen_device': 'auto',
 'oxygen_shell_max_k': 30,
 'oxygen_workers': 'auto',
 'policies': ['bf', 'anneal', 'hop'],
 'probe_only': False,
 'probe_ti_cif': None,
 'ref': 'ref',
 'refine_inner_methods': ['hopping', 'anneal', 'replica'],
 'refine_methods': 'hopping,anneal,replica',
 'refine_per_gen': 1,
 'report_all_atoms': True,
 'rutile_control': False,
 'save_survivors': True,
 'scf_cycles': 40,
 'screen_baseline_starts': 1,
 'search_policies': 'bf,anneal,hop',
 'seed': 37,
 'space_groups': '136',
 'stop_framework_on_complete': True,
 'stop_refinement_on_complete': True,
 'stop_wyckoff_on_complete': True,
 'symmetry_tol': 0.17,
 'symprec': 0.055,
 'ti_de_crossover': 0.85,
 'ti_de_scale': 0.7,
 'ti_device': 'auto',
 'ti_dynamics_basins': 1,
 'ti_dynamics_matched': True,
 'ti_dynamics_policies': ['bf', 'hop', 'anneal', 'de', 'hybrid'],
 'ti_dynamics_policies_raw': 'bf,hop,anneal,de,hybrid',
 'ti_dynamics_seeds': (37, 38),
 'ti_dynamics_skip_complete_origin': True,
 'ti_dynamics_starts': 8,
 'ti_dynamics_study': False,
 'ti_dynamics_targets': [(136, (1, 6)), (136, (2, 4)), (141, (4, 8))],
 'ti_dynamics_targets_raw': '136:1,6;136:2,4;141:4,8',
 'ti_empirical_contact_margin_A': 0.12,
 'ti_empirical_density_ranks': '6,12,16,20,24',
 'ti_empirical_density_width_A': 0.18,
 'ti_empirical_repulsion_normalizer_A': 0.15,
 'ti_empirical_repulsion_width_A': 0.1,
 'ti_empirical_screen_margin_A': 0.26,
 'ti_empirical_span_neighbors': 12,
 'ti_empirical_volume_log_width_floor': 0.35,
 'ti_pca_distortions': '0.04,0.15,0.40',
 'ti_pca_opt_steps': 45,
 'ti_pca_oxygen_pairs': 0,
 'ti_pca_prep_learned_steps': 30,
 'ti_pca_prep_natural_steps': 30,
 'ti_pca_rank_cap': 0,
 'ti_pca_starts': 3,
 'ti_pca_targets': '136:1,5;136:2,4',
 'ti_pca_validation': False,
 'ti_rule_audit_only': False,
 'ti_rule_mode': 'compare',
 'ti_soap_pca': 'audit',
 'ti_soap_pca_max_components': 0,
 'ti_soap_pca_variance': 0.995,
 'ti_study': False,
 'ti_study_local_opt': 'both',
 'ti_study_oxygen_methods': ['baseline'],
 'ti_study_oxygen_pairs': 0,
 'ti_study_save_top': 8,
 'ti_study_seeds': (37, 38, 39),
 'ti_width_stall': 1,
 'v22_anneal_scf_cycles': 1,
 'v22_checkpoint_every': 25,
 'v22_early_confirm': 4,
 'v22_early_primary_A': 0.12,
 'v22_early_sharing': 0.12,
 'v22_exchange_every': 12,
 'v22_exhaustion_window': 96,
 'v22_global_max_seconds': 45.0,
 'v22_global_rotation': 0.35,
 'v22_global_scf_cycles': 5,
 'v22_global_steps': 300,
 'v22_improvement_eps': 1e-05,
 'v22_inner_methods': ['baseline', 'anneal', 'hopping', 'replica'],
 'v22_methods': 'baseline,anneal,hopping,replica',
 'v22_min_global_steps': 48,
 'v22_replicas': 4,
 'v22_stagnation_window': 24,
 'v49_cell_rel_tolerance': 0.002,
 'v50_ab': False,
 'v50_allow_resume': False,
 'v50_fast': True,
 'v50_feedback_min_steps': 48,
 'v50_feedback_rank_abs': 0.002,
 'v50_feedback_stall_checkpoints': 3,
 'v50_scf_check_every': 5,
 'v50_scf_min_cycles': 18,
 'v50_scf_stall_cycles': 12,
 'v50_self_test': False,
 'v50_ti_min_learned': 45,
 'v50_ti_min_natural': 45,
 'v50_ti_objective_abs': 0.0004,
 'v50_ti_objective_rel': 0.002,
 'v50_ti_stop_A': 0.05,
 'v50_ti_window': 20,
 'v51_2_self_test': False,
 'v51_anneal_cooling': 0.84,
 'v51_hop_temperature': 0.085,
 'v51_include_complete': False,
 'v51_large_A': 0.24,
 'v51_max_move_A': 0.65,
 'v51_methods': 'bf,anneal,hop',
 'v51_oxygen_starts': 1,
 'v51_promote_per_policy': 1,
 'v51_quench_anchor': 0.8,
 'v51_quench_lr': 0.008,
 'v51_quench_max_A': 0.2,
 'v51_quench_steps': 8,
 'v51_reheat_after': 3,
 'v51_reheat_temperature': 0.4,
 'v51_screen_feedback_rounds': 1,
 'v51_screen_feedback_steps': 24,
 'v51_screen_scf_cycles': 20,
 'v51_seeds': '41,42',
 'v51_self_test': False,
 'v51_small_A': 0.08,
 'v51_temp_floor': 0.025,
 'v51_temperature': 0.25,
 'v51_ti_dynamics': False,
 'v51_trials': 8,
 'v51_wall_budget_s': 1500.0,
 'v51_wall_guard_s': 190.0,
 'v52_base_batches': 1,
 'v52_basins_per_batch': 2,
 'v52_dfs': False,
 'v52_enumeration_cap': 10000,
 'v52_fail_fast': False,
 'v52_feedback_rounds': 1,
 'v52_feedback_steps': 20,
 'v52_global_method': 'anneal',
 'v52_global_proposals': 1,
 'v52_global_steps': 12,
 'v52_max_batches': 4,
 'v52_max_orbits': 4,
 'v52_max_oxygen_entrances': 3,
 'v52_max_scale': 3.0,
 'v52_max_wp_per_sg': 0,
 'v52_oxygen_entrances': 1,
 'v52_plan_only': False,
 'v52_scf_cycles': 20,
 'v52_self_test': False,
 'v52_sgs': '74,15,60',
 'v52_ti_count': 12,
 'v52_ti_opt_batch': 0,
 'v52_ti_raw_batch': 24,
 'v53_autotune_max_batch': 48,
 'v53_dfs': False,
 'v53_novelty_loss_abs': 0.05,
 'v53_novelty_loss_rel': 0.5,
 'v53_novelty_pool': 16,
 'v53_plan_only': False,
 'v53_quiet_inner': True,
 'v53_raw_factor': 2.0,
 'v53_self_test': False,
 'v53_verbose_inner': False,
 'v54_dfs': False,
 'v54_o_repeat_policy': 'observe',
 'v54_plan_only': False,
 'v54_self_test': False,
 'v54_wp_per_sg': 3,
 'v55_dfs': False,
 'v55_local_only': False,
 'v55_plan_only': False,
 'v55_self_test': False,
 'v56_dfs': False,
 'v56_global_match_cap': 12,
 'v56_o_global_per_wp': 1,
 'v56_opt_per_batch': 48,
 'v56_plan_only': False,
 'v56_pool_window': 32,
 'v56_raw_per_batch': 96,
 'v56_repeat_ti_o_reserve': 1,
 'v56_self_test': False,
 'v56_ti_batches': 2,
 'v56_ti_quality': 2,
 'v56_ti_reserve': 1,
 'v57_dfs': True,
 'v57_o_global_per_wp': 2,
 'v57_opt_per_batch': 48,
 'v57_plan_only': False,
 'v57_profile': 'APLUS',
 'v57_raw_per_batch': 96,
 'v57_reserve_loss_abs': 0.05,
 'v57_reserve_loss_rel': 0.5,
 'v57_self_test': False,
 'v57_ti_quality_total': 4,
 'v57_ti_reserve': 1,
 'v57ap_self_test': False,
 'warm_fresh': 0,
 'wyckoff': None,
 'wyckoff_pool': 8}

def parse_args(argv=None):
    """Standard unversioned CLI. Numerical kernels retain internal legacy keys."""
    p=argparse.ArgumentParser(
        description='Juliette v59.2: ranked top-K TiO2, 24 formula units, broad space-group search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--space-groups',default='1-230',
                   help='Initial space groups: comma list, range, or all; all 230 by default')
    p.add_argument('--formula-units',type=int,default=24,
                   help='TiO2 formula units in the generated conventional/search cell')
    p.add_argument('--max-unique-per-sg',type=int,default=10,
                   help='Maximum globally unique final structures attributed to each initial SG')
    p.add_argument('--stop-sg-at-cap',action=argparse.BooleanOptionalAction,default=True,
                   help='Stop future WP generation when SG has 10 distinct ranked candidates; '
                        'disable to guarantee best-K over every configured WP')
    p.add_argument('--write-final-per-sg',action=argparse.BooleanOptionalAction,default=True,
                   help='Export only the FINAL top-K CIFs for each initial SG as soon as that SG finishes; '
                        'never save intermediate oxygen candidates')
    p.add_argument('--wp-per-sg',type=int,default=3,
                   help='Maximum viable Wyckoff patterns explored per initial SG (0 for all with enumeration cap)')
    p.add_argument('--max-ti-orbits',type=int,default=6,
                   help='Maximum inequivalent Ti Wyckoff orbits (0 unlimited)')
    p.add_argument('--enumeration-cap',type=int,default=10000)
    p.add_argument('--raw-starts',type=int,default=96)
    p.add_argument('--optimized-starts',type=int,default=48)
    p.add_argument('--quality-basins',type=int,default=4)
    p.add_argument('--novelty-basins',type=int,default=1)
    p.add_argument('--reserve-loss-abs',type=float,default=.05)
    p.add_argument('--reserve-loss-rel',type=float,default=.5)
    p.add_argument('--global-oxygen-per-wp',type=int,default=2)
    p.add_argument('--scf-cycles',type=int,default=20)
    p.add_argument('--feedback-steps',type=int,default=20)
    p.add_argument('--global-steps',type=int,default=12)
    p.add_argument('--global-method',choices=('anneal','hopping','replica'),default='anneal')
    p.add_argument('--o-cn-target',action=argparse.BooleanOptionalAction,default=True,
                   help='Include O-CN3 in oxygen optimization; O-CN3 is always reported')
    p.add_argument('--ref',default='ref')
    p.add_argument('--db',default='tio2.db')
    p.add_argument('--output',default='tio2_v59_24fu')
    p.add_argument('--seed',type=int,default=37)
    p.add_argument('--ti-device',default='auto')
    p.add_argument('--oxygen-device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--gpu-octa-fit',choices=('auto','on','off'),default='auto')
    p.add_argument('--symprec',type=float,default=.055)
    p.add_argument('--min-ti-ti-A',type=float,default=2.7)
    p.add_argument('--max-tuned-batch',type=int,default=48)
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--verbose-inner',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(argv)
    if a.formula_units<2 or a.formula_units>96:p.error('--formula-units must be 2..96')
    if a.max_unique_per_sg<1:p.error('--max-unique-per-sg must be >=1')
    if a.wp_per_sg<0 or a.max_ti_orbits<0 or a.enumeration_cap<1:p.error('Invalid Wyckoff enumeration bounds')
    if a.raw_starts<a.optimized_starts or a.optimized_starts<1:p.error('Optimized starts must be 1..raw starts')
    if a.quality_basins<1 or a.novelty_basins<0 or a.global_oxygen_per_wp<0:
        p.error('Invalid basin/oxygen capacities')
    if a.scf_cycles<1 or a.feedback_steps<1 or a.global_steps<1:
        p.error('Oxygen evaluation budgets must be positive')
    try:parse_sgs(a.space_groups)
    except ValueError as e:p.error(str(e))
    cfg=argparse.Namespace(**_CLEAN_DEFAULTS.copy())
    values={
      'v52_sgs':a.space_groups,'v52_ti_count':a.formula_units,
      'v54_wp_per_sg':a.wp_per_sg,'v52_max_orbits':a.max_ti_orbits,
      'v52_enumeration_cap':a.enumeration_cap,
      'v57_raw_per_batch':a.raw_starts,'v57_opt_per_batch':a.optimized_starts,
      'v57_ti_quality_total':a.quality_basins,'v57_ti_reserve':a.novelty_basins,
      'v57_reserve_loss_abs':a.reserve_loss_abs,
      'v57_reserve_loss_rel':a.reserve_loss_rel,
      'v57_o_global_per_wp':a.global_oxygen_per_wp,
      'v52_scf_cycles':a.scf_cycles,'v52_feedback_steps':a.feedback_steps,
      'v52_global_steps':a.global_steps,'v52_global_method':a.global_method,
      'target_o_cn':a.o_cn_target,'ref':a.ref,'db':a.db,'output':a.output,
      'seed':a.seed,'ti_device':a.ti_device,'oxygen_device':a.oxygen_device,
      'gpu_octa_fit':a.gpu_octa_fit,'symprec':a.symprec,
      'min_TiTi_A':a.min_ti_ti_A,'v53_autotune_max_batch':a.max_tuned_batch,
      'v52_fail_fast':a.fail_fast,'v53_verbose_inner':a.verbose_inner,
      'v53_quiet_inner':not a.verbose_inner,'v56_plan_only':a.plan_only,
      'max_unique_per_sg':a.max_unique_per_sg,'stop_sg_at_cap':a.stop_sg_at_cap,
      'write_final_per_sg':a.write_final_per_sg,
      'self_test':a.self_test,'v57_profile':'APLUS',
    }
    for k,v in values.items():setattr(cfg,k,v)
    cfg.v56_pool_window=max(cfg.v56_pool_window,cfg.v57_ti_quality_total+cfg.v57_ti_reserve)
    cfg.v56_global_match_cap=max(cfg.v56_global_match_cap,cfg.v57_ti_quality_total+cfg.v57_ti_reserve)
    cfg.v22_inner_methods=['baseline',a.global_method]
    cfg.v53_raw_factor=2.0
    cfg.v50_allow_resume=False
    return cfg

def v59_self_test():
    from dataclasses import is_dataclass
    assert is_dataclass(WyckoffTemplate)
    test=WyckoffTemplate(74,(1,5),12,'tetragonal',('a','c'),[],8)
    assert test.n_ti==12
    cfg=parse_args(['--space-groups','15,60,74','--formula-units','24',
                    '--max-unique-per-sg','10','--no-o-cn-target',
                    '--no-stop-sg-at-cap'])
    assert cfg.v52_ti_count==24 and cfg.max_unique_per_sg==10
    assert not cfg.target_o_cn and not cfg.stop_sg_at_cap
    assert _v22_search_options(cfg,1).target_o_cn is False
    # Regression: public v59 parser must not shadow the Ti-backend parser.
    ti_flags=configured_ti_backend(cfg,[
        '--ref',cfg.ref,'--starts',str(cfg.full_ti_starts),
        '--keep-basins',str(cfg.full_ti_basins),
        '--natural-steps',str(cfg.full_natural_steps),
        '--learned-steps',str(cfg.full_learned_steps),
        '--device',cfg.ti_device,'--ti-rule-mode','hybrid',
        '--ti-soap-pca','audit'])
    ti_args=parse_ti_args(ti_flags)
    assert ti_args.starts==cfg.full_ti_starts
    assert ti_args.keep_basins==cfg.full_ti_basins
    assert ti_args.natural_steps==cfg.full_natural_steps
    assert ti_args.learned_steps==cfg.full_learned_steps
    assert ti_args.ti_rule_mode=='hybrid' and ti_args.ti_soap_pca=='audit'
    # Exercise the production startup Ti-stage entrypoint, not just parser unit tests.
    # It should reach reference loading; the v59 crash occurred before this point.
    class _ReachedReferenceLoading(Exception):
        pass
    original_reference_loader=globals()['reference_ti_cifs']
    try:
        def _reached_loader(*_args,**_kwargs):
            raise _ReachedReferenceLoading()
        globals()['reference_ti_cifs']=_reached_loader
        try:
            _v48_ti_stage(cfg,[])
        except _ReachedReferenceLoading:
            pass
        else:
            raise AssertionError('Ti-stage startup did not reach reference loader')
    finally:
        globals()['reference_ti_cifs']=original_reference_loader
    # Test archive ranking, dedup, eviction, memory-only policy without pymatgen.
    class FakeMatcher:
        def fit(self,a,b):return a==b
    fake=lambda row:row['signature']
    bank=_V59TopArchive(2,2,matcher=FakeMatcher(),structure_factory=fake)
    def row(sig,maxi,goodTi=0):
        return {'ti':np.zeros((2,3)), 'cell':np.eye(3)*4,
                'answer':{'oxygen':np.zeros((4,3))},'signature':sig,
                'q':{'minTi':maxi/4,'maxTi':maxi,'meanTi':maxi/2,
                     'goodTi':goodTi,'goodO':0,'maxO':0.5,'meanO':0.1,
                     'sharing':0.3}}
    assert bank.consider(row('A',.3),'A',60,(1,),'local')[0]=='ranked'
    assert bank.consider(row('B',.2),'B',60,(1,),'local')[0]=='ranked'
    assert bank.consider(row('C',.4),'C',60,(1,),'local')[0]=='below_cutoff'
    assert bank.consider(row('C',.1),'C',60,(1,),'local')[0]=='ranked'
    assert [r['name'] for r in bank.by_sg(60)]==['C','B']
    assert bank.consider(row('B',.05),'B2',60,(1,),'global')[0]=='improved_duplicate'
    assert bank.consider(row('B',.01),'X',15,(1,),'global')[0]=='cross_sg_duplicate'
    assert bank.count(15)==0 and bank.count(60)==2
    assert bank.consider(row('D',.001),'D',60,(1,),'global')[0]=='ranked'
    assert [r['name'] for r in bank.by_sg(60)]==['D','B2']
    assert bank.consider(row('B',.0005),'B3',60,(1,),'global')[0]=='improved_duplicate'
    assert [r['name'] for r in bank.by_sg(60)]==['B3','D']
    assert cfg.write_final_per_sg
    # Writing one SG finalizes it; a later SG is still free to match its records.
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        original_cif=globals()['_v44_atomic_cif']
        saved=[]
        try:
            def _test_cif(record,path):
                Path(path).parent.mkdir(parents=True,exist_ok=True)
                Path(path).write_text('TEST CIF')
                saved.append(str(path))
            globals()['_v44_atomic_cif']=_test_cif
            assert bank.write_sg_final(scratch,60)==2
            assert bank.write_sg_final(scratch,60)==0
            assert len(saved)==2
            assert bank.consider(row('B',.0001),'X',15,(1,),'global')[0]=='cross_sg_duplicate'
            try:bank.consider(row('A',.01),'A',60,(1,),'local')
            except RuntimeError:pass
            else:raise AssertionError('Finalized SG unexpectedly mutable')
            assert len(bank.write_final(scratch))==2 and len(saved)==2
        finally:
            globals()['_v44_atomic_cif']=original_cif
    q=lambda gt,go:{'goodTi':gt,'goodO':go,'nearTi':0,'nearO':0,
                     'primaryTi':0,'primaryO':0,'rmsTi':0.,'rmsO':0.,
                     'sharing':0.1,'bad_triplets':0,'collision':0.,
                     'maxTi':1.,'maxO':1.,'meanTi':.5,'meanO':.5}
    candidates=[({'q':q(3,15)},{'loss':.3},0,'Ti',True),
                ({'q':q(0,24)},{'loss':.1},0,'O',True)]
    assert [t for t,_ in _v57ap_select_promotions(candidates,2)]==['Ti_first','O_first']
    log('v59.2 SELF-TEST PASSED: constructor, separated CLI/Ti parsers, O-CN switch, '
        'top-K ranking, per-SG final CIF output, global duplicates, promotion policy')

def main(argv=None):
    cfg=parse_args(argv)
    global _V50_CONFIG
    _V50_CONFIG=cfg
    if cfg.self_test:return v59_self_test()
    return _v59_run_dfs(cfg,[],[])

# -----------------------------------------------------------------------------
# JULIETTE v59.2 paired, blind GENERATION ablation (not reference reconstruction)
# The original v59.2 global method is left intact above and never used as an
# alternate source of oxygen coordinates. The same v59.2 kernels are reused.
# -----------------------------------------------------------------------------

def ab59_parse(argv=None):
    p=argparse.ArgumentParser(description='Paired v59.2 TiO2 generation ablation: '
        'A rigid Ti-only, B collective strain Ti-only, C collective strain full Ti/O chemistry',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,allow_abbrev=False)
    p.add_argument('--ref',default='ref_tio2')
    p.add_argument('--space-groups',default='61,87,136,141')
    p.add_argument('--formula-units',type=int,default=24)
    p.add_argument('--wp-per-sg',type=int,default=2)
    p.add_argument('--wp-scan-limit',type=int,default=400)
    p.add_argument('--max-ti-orbits',type=int,default=24)
    p.add_argument('--metal-cycles',type=int,default=1)
    p.add_argument('--raw-starts',type=int,default=8)
    p.add_argument('--optimized-starts',type=int,default=4)
    p.add_argument('--basins-per-wp',type=int,default=2)
    p.add_argument('--natural-steps',type=int,default=30)
    p.add_argument('--learned-steps',type=int,default=50)
    p.add_argument('--natural-lr',type=float,default=.04)
    p.add_argument('--learned-lr',type=float,default=.02)
    p.add_argument('--entrances',type=int,default=3)
    p.add_argument('--modes-per-ti',type=int,default=4)
    p.add_argument('--octahedral-proposals',type=int,default=6)
    p.add_argument('--scf-cycles',type=int,default=30)
    p.add_argument('--feedback-steps',type=int,default=40)
    p.add_argument('--feedback-rounds',type=int,default=2)
    p.add_argument('--strain-max',type=float,default=.11)
    p.add_argument('--max-o-move',type=float,default=.10)
    p.add_argument('--device',default='auto')
    p.add_argument('--gpu-workers',default='auto',
                   help='Automatic number of local GPU workers (default: auto), or 1,2,...; '
                        'one search process per GPU, distinct space-group assignments')
    p.add_argument('--gpu-plan-only',action='store_true',
                   help='Print local GPU/space-group allocation and exit without searching')
    p.add_argument('--gpu-stream-logs',action='store_true',
                   help='Stream each GPU worker log to the parent stdout (default: save only)')
    p.add_argument('--ti-rule-mode',choices=('hybrid','learned','legacy'),default='hybrid')
    p.add_argument('--shell-max-k',type=int,default=30)
    p.add_argument('--seed',type=int,default=37)
    p.add_argument('--output',default='tio2_v59_paired_ablation')
    p.add_argument('--variants',default='A,B,C',help='Subset of A,B,C for debugging; full comparison uses all')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--write-qualified-cif',action='store_true',help='Only combined qualified final structures')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(argv)
    a.variants=[v.strip().upper() for v in a.variants.split(',') if v.strip()]
    if not a.variants or len(set(a.variants))!=len(a.variants) or any(v not in 'ABC' for v in a.variants):
        p.error('--variants must be a nonempty unique subset of A,B,C')
    if a.formula_units<2 or a.formula_units>96:p.error('formula units must be 2..96')
    if a.max_o_move<=0 or a.max_o_move>.10:p.error('0 < --max-o-move <= 0.10')
    if a.strain_max<0 or a.strain_max>.20:p.error('0 <= --strain-max <= .20')
    if min(a.raw_starts,a.optimized_starts,a.basins_per_wp,a.entrances,a.metal_cycles)<1:
        p.error('Search sizes must be positive')
    if a.raw_starts<a.optimized_starts:p.error('raw-starts must be >= optimized-starts')
    if a.wp_per_sg<0 or a.max_ti_orbits<0 or a.wp_scan_limit<1:p.error('Invalid WP scan limits')
    if a.feedback_steps<0 or a.feedback_rounds<0 or a.scf_cycles<1:p.error('Invalid SCF budgets')
    parse_sgs(a.space_groups)
    return a

def ab59_plan(a):
    from pyxtal.symmetry import Group
    plan=[];by_sg={}
    for sg in parse_sgs(a.space_groups):
        patterns=list(itertools.islice(v48_iter_wyckoff(sg,a.formula_units,a.max_ti_orbits),
                                      a.wp_scan_limit))
        if a.wp_per_sg and len(patterns)>a.wp_per_sg:
            # Simple deterministic stratification over independent orbit counts.
            lengths=sorted(set(map(len,patterns)))
            selected=[]
            for count in lengths:
                selected.append(next(z for z in patterns if len(z)==count))
                if len(selected)==a.wp_per_sg:break
            remaining=[z for z in patterns if z not in selected]
            remaining=sorted(remaining,key=lambda z:(len(z),sum(Group(sg)[int(x)].get_dof() for x in z),z))
            while len(selected)<a.wp_per_sg and remaining:
                selected.append(remaining.pop(len(remaining)//2))
            patterns=selected
        by_sg[sg]=len(patterns)
        plan.extend((sg,tuple(z)) for z in patterns)
    return plan,by_sg

def ab59_metal_basins(template,statistics,objective,a,seed,device):
    # Preserves v59.2's reference-statistics Ti objective and its batched
    # natural -> learned Adam kernels, with one shared Ti bank for A/B/C.
    raw=initial_raw(template,statistics,a.raw_starts,seed)
    with torch.no_grad():
        vals=[]
        for chunk in torch.as_tensor(raw,dtype=DTYPE,device=device).split(8):
            vals.extend(objective.evaluate(template,chunk,'natural').detach().cpu().tolist())
    order=np.argsort(vals,kind='stable')
    elite=order[:max(1,a.optimized_starts//2)].tolist()
    rng=np.random.default_rng(str_seed(seed,'shared-lattice-diversity'))
    leftovers=[j for j in range(len(raw)) if j not in elite]
    rng.shuffle(leftovers)
    indices=elite+leftovers[:a.optimized_starts-len(elite)]
    x=torch.as_tensor(raw[indices],dtype=DTYPE,device=device)
    natural,_,_=run_optimizer(objective,template,x,'natural',a.natural_steps,a.natural_lr,
                              patience=a.natural_steps+1,min_steps=a.natural_steps+1)
    learned,_,_=run_optimizer(objective,template,natural,'learned',a.learned_steps,a.learned_lr,
                              patience=a.learned_steps+1,min_steps=a.learned_steps+1)
    with torch.no_grad():
        loss,detail=objective.evaluate(template,learned,'learned',details=True)
        values=loss.cpu().numpy()
        cells=detail['cell'].cpu().numpy()
        metals=detail['ti_frac'].cpu().numpy()%1
    basins=[]
    for ix in np.argsort(values,kind='stable'):
        if any(cell_metric_discrepancy(cells[ix],x['cell'])<.012 and
               periodic_match_rms(metals[ix],x['ti'],cells[ix])<.08 for x in basins):
            continue
        basins.append(dict(cell=cells[ix].copy(),ti=metals[ix].copy(),
                           metal_loss=float(values[ix]),raw=learned[ix].detach().cpu().numpy().copy()))
        if len(basins)>=a.basins_per_wp:break
    return basins


def ab59_movement(octa,oxygen,assignment):
    arm=octa.positions().reshape(-1,3)
    shared=(np.asarray(oxygen,float)[np.asarray(assignment,dtype=int)])@octa.cell
    delta=minimum_image_delta(arm,shared,octa.cell,octa.inv)
    r=np.linalg.norm(delta,axis=1)
    return float(np.max(r)),float(np.sqrt(np.mean(r*r))),float(np.mean(r))

def ab59_record(result,chem_full,variant,initial_SG,wp,basin_i,seed_i,metal_loss,a):
    oc=result['oct']
    oxygen=np.asarray(result['oxygen'],float)%1.
    with torch.no_grad():
        full=chem_full.score(torch.as_tensor(oxygen,dtype=DTYPE),details=True)
    move_max,move_rms,move_mean=ab59_movement(oc,oxygen,result['trip'])
    count=len(oc.ti);no=2*count
    ti_cn=np.asarray(full['hardTi'],int)
    o_cn=np.asarray(full['hardO'],int)
    # All candidates are scored against the SAME full learned Ti/O chemistry;
    # ablated A/B channels are restored for evaluation only.
    n_good_ti=int(np.sum(full['goodTi']))
    n_good_o=int(np.sum(full['goodO']))
    coord_ok=bool(np.all(ti_cn==6) and np.all(o_cn==3))
    chemistry_ok=bool(n_good_ti==count and n_good_o==no and coord_ok and
                      full['collision']<=1e-8)
    sharing_ok=bool(result['bad_triplets']==0 and move_max<=a.max_o_move)
    full_pass=bool(chemistry_ok and sharing_ok)
    full_ref_arms=np.stack([oc.lib['Ti'][int(m)]['rigid_arms'] for m in oc.mode])
    strain=np.asarray(oc.stretch)
    # Max movement of a mode vertex caused by coherent deformation alone.
    strained=np.einsum('nvi,nij->nvj',full_ref_arms,strain)
    deformation=np.linalg.norm(strained-full_ref_arms,axis=-1)
    actual_ops=active_symmetry(oc,symmetry_mappings(oc.ti,oc.cell,a._symprec),.17)
    # Measure the *actual full crystal* SG. Ideal-arm symmetry is NOT a proxy.
    final_sg=0;metal_sg=0;final_inversion=False;final_mirrors=0
    try:
        import spglib
        full_set=spglib.get_symmetry_dataset((oc.cell,np.vstack((oc.ti,oxygen)),
                                             [22]*count+[8]*no),symprec=.05)
        ti_set=spglib.get_symmetry_dataset((oc.cell,oc.ti,[22]*count),symprec=.05)
        if full_set is not None:
            final_sg=int(full_set['number'])
            rotations=np.asarray(full_set['rotations'],int)
            final_inversion=bool(any(np.array_equal(z,-np.eye(3,dtype=int)) for z in rotations))
            final_mirrors=int(sum(np.linalg.det(z)<0 and np.trace(z)==1 for z in rotations))
        if ti_set is not None:metal_sg=int(ti_set['number'])
    except Exception:
        pass
    return dict(variant=variant,sg_initial=initial_SG,wyckoff=str(tuple(wp)),
        basin=basin_i,entrance=seed_i,metal_loss=metal_loss,
        max_o_move=move_max,rms_o_move=move_rms,mean_o_move=move_mean,
        max_coherent_deformation=float(np.max(deformation)),
        rms_coherent_deformation=float(np.sqrt(np.mean(deformation**2))),
        strain_matrix_rms=float(oc.distortion()),
        Ti_loss_min=float(np.min(full['localTi'])),
        Ti_loss_max=float(np.max(full['localTi'])),
        Ti_loss_avg=float(np.mean(full['localTi'])),
        O_loss_min=float(np.min(full['localO'])),
        O_loss_max=float(np.max(full['localO'])),
        O_loss_avg=float(np.mean(full['localO'])),
        overall_full_chem_energy=float(full['energy']),
        n_good_ti=n_good_ti,n_good_o=n_good_o,
        n_ti=count,n_o=no,hard_ti6=int(np.sum(ti_cn==6)),
        hard_o3=int(np.sum(o_cn==3)),
        bad_triplets=int(result['bad_triplets']),
        collision=float(full['collision']),
        selected_symops=int(result.get('compatible_symops',len(actual_ops))),
        audited_ideal_symops=len(actual_ops),
        measured_metal_sg=metal_sg,measured_full_sg=final_sg,
        full_has_inversion=final_inversion,full_mirror_ops=final_mirrors,
        chemistry_ok=chemistry_ok,
        sharing_ok=sharing_ok,
        qualified=full_pass,
        gate005=bool(chemistry_ok and result['bad_triplets']==0 and move_max<=.05),
        gate010=bool(chemistry_ok and result['bad_triplets']==0 and move_max<=.10),
        mode_counts=dict(sorted(collections.Counter(int(m) for m in oc.mode).items())),
        scf_cycles=int(result.get('cycles',0)),
        feedback_steps=int(result.get('chem_feedback_steps',0)))


def ab59_output(out,rows,a,qualified=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    flat=[]
    for row in rows:
        item={k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v)
              for k,v in row.items()}
        flat.append(item)
    columns=list(flat[0]) if flat else [
        'variant','sg_initial','wyckoff','basin','entrance','max_o_move','Ti_loss_max',
        'O_loss_max','chemistry_ok','sharing_ok','qualified']
    with (out/'per_candidate.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader();writer.writerows(flat)
    # Paired comparisons use the identical framework and seed key, not
    # unrelated variant-level minima from different structures.
    groups=defaultdict(dict)
    for row in rows:
        key=(row['sg_initial'],row.get('wyckoff',''),row.get('basin',0),row.get('entrance',0))
        groups[key][row['variant']]=row
    paired=[]
    for key,d in groups.items():
        comp=dict(sg=key[0],wyckoff=key[1],basin=key[2],entrance=key[3])
        for variant in a.variants:
            candidate=d.get(variant)
            comp[f'{variant}_Omax']=candidate['max_o_move'] if candidate else None
            comp[f'{variant}_Ti_loss_max']=candidate['Ti_loss_max'] if candidate else None
            comp[f'{variant}_O_loss_max']=candidate['O_loss_max'] if candidate else None
            comp[f'{variant}_qualified']=candidate['qualified'] if candidate else None
        for left,right in (('A','B'),('B','C')):
            if left in d and right in d:
                comp[f'{right}_minus_{left}_Omax']=d[right]['max_o_move']-d[left]['max_o_move']
                comp[f'{right}_minus_{left}_Ti_loss_max']=d[right]['Ti_loss_max']-d[left]['Ti_loss_max']
        paired.append(comp)
    if paired:
        with (out/'paired_comparisons.csv').open('w',newline='') as handle:
            keys=list(dict.fromkeys(k for item in paired for k in item))
            writer=csv.DictWriter(handle,fieldnames=keys)
            writer.writeheader();writer.writerows(paired)
    statistics=[]
    for sg in parse_sgs(a.space_groups):
        for variant in a.variants:
            subset=[x for x in rows if x['variant']==variant and x['sg_initial']==sg]
            best=min(subset,key=lambda d:(not d['qualified'],d['Ti_loss_max'],
                                          d['O_loss_max'],d['max_o_move']),default=None)
            stats=dict(sg=sg,variant=variant,attempts=len(subset),
              chemical_pass=sum(x['chemistry_ok'] for x in subset),
              movement005=sum(x['max_o_move']<=.05 for x in subset),
              movement010=sum(x['max_o_move']<=.10 for x in subset),
              gate005=sum(x['gate005'] for x in subset),
              gate010=sum(x['gate010'] for x in subset),
              qualified=sum(x['qualified'] for x in subset),
              best_Omax=min((x['max_o_move'] for x in subset),default=None),
              best_Ti_loss_max=min((x['Ti_loss_max'] for x in subset),default=None),
              best_joint_chemistry=min((x['overall_full_chem_energy'] for x in subset),default=None),
              best_qualified_Ti_loss=(best['Ti_loss_max'] if best and best['qualified'] else None))
            statistics.append(stats)
    with (out/'per_sg.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(statistics[0]) if statistics else [])
        writer.writeheader();writer.writerows(statistics)
    totals=[]
    for v in a.variants:
        subset=[x for x in rows if x['variant']==v]
        total=dict(variant=v,attempts=len(subset),
          chemical_pass=sum(x['chemistry_ok'] for x in subset),
          movement010=sum(x['max_o_move']<=.10 for x in subset),
          qualified=sum(x['qualified'] for x in subset),
          best_Omax=min((x['max_o_move'] for x in subset),default=None),
          best_Ti_loss_max=min((x['Ti_loss_max'] for x in subset),default=None))
        totals.append(total)
    with (out/'summary.json').open('w') as handle:
        json.dump(dict(config={k:v for k,v in vars(a).items() if not k.startswith('_')},
                       totals=totals,per_sg=statistics,paired=paired,candidates=rows),handle,indent=2)
    log('========== PAIRED v59.2 ABLATION SUMMARY ==========')
    for s in totals:
        log(f'  {s["variant"]}: trials={s["attempts"]} '
            f'chem_pass={s["chemical_pass"]} Omove<=.10={s["movement010"]} '
            f'full_qualified={s["qualified"]} '
            f'best_Omax={s["best_Omax"]} best_Ti_loss_max={s["best_Ti_loss_max"]}')
    log('  Detailed candidate metrics: per_candidate.csv; matched contrasts: paired_comparisons.csv; per-SG metrics: per_sg.csv')
    return totals

def ab59_self_test():
    # No crystallographic library required. Check that the strain ablation
    # genuinely changes *both* the v59.2 Kabsch fitter and torch feedback path.
    base=2*np.asarray([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1.]])
    src={'Ti':[dict(rigid_arms=base)]}
    proposals=[[(0.,0,Rotation.identity())]]
    octa=Octahedra(np.array([[.33,.33,.33]]),np.eye(3)*8,src,proposals,seed=0)
    targets=(1.09*base)[None,:,:]
    for lim in (0.,.11):
        p=octa.copy();p._ablation_max_strain=lim
        p.fit(targets,mode_options=None,mix=1.,distortion_max=.11)
        deviation=np.max(np.abs(np.linalg.eigvalsh(p.stretch[0])-1.))
        assert deviation<=lim+1e-10,(lim,deviation)
        if lim==0.:assert deviation<1e-10
        else:assert deviation>.01,(lim,deviation)
    # Verify original chemistry O-geometry path multiplies by explicit weight
    # while Ti geometry / O CN / collision controls stay present in source.
    src_code=Path(__file__).read_text()
    assert 'score["O"]=score["O"]*self.ablation_oxygen_weight' in src_code
    assert 'exceed_o=torch.relu(res_o-o_lim)*chem.ablation_oxygen_weight' in src_code
    assert 'sv=(min(.065,strain_limit) if strain_limit>0 else 0.0)' in src_code
    assert 'frac_candidates(ti,cell,lib' in src_code  # baseline retained, NOT called by ab59
    assert 'np.empty((0,3))' in src_code
    synthetic=[dict(variant='A',sg_initial=1,max_o_move=1.5,Ti_loss_max=.9,
       O_loss_max=.8,overall_full_chem_energy=2.,chemistry_ok=False,
       gate005=False,gate010=False,qualified=False),
       dict(variant='B',sg_initial=1,max_o_move=.08,Ti_loss_max=.4,
       O_loss_max=.5,overall_full_chem_energy=.9,chemistry_ok=True,
       gate005=False,gate010=True,qualified=True)]
    with tempfile.TemporaryDirectory() as d:
        a=ab59_parse(['--space-groups','1','--variants','A,B'])
        total=ab59_output(d,synthetic,a)
        assert [s['qualified'] for s in total]==[0,1]
        assert (Path(d)/'per_candidate.csv').is_file()
        assert (Path(d)/'per_sg.csv').is_file()
    log('AB59 SELF-TEST PASSED: full-rigid vs strained fitter, coupled flags, paired summaries')


# =============================================================================
# JULIETTE v59.3 — EXPLORATION FIRST (v59.2 LCAO + coherent TiO6 strain)
# =============================================================================
# This runner intentionally retains the v59.2 implementation above. It changes
# orchestration, NOT the definition of the learned chemistry or the polyhedra.
# Finite Ti_n O_2n structures are ranked even if every hard chemistry flag is 0.

def e593_parse(argv=None):
    p=argparse.ArgumentParser(
        description='JULIETTE v60: Ti Wyckoff-orbit building units, short LCAO/SCF probes, '
                    'selective coherent TiO6 refinement, final top-N CIFs regardless of chemistry flags',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    # v60 additional options are parsed by v60_parse; advertise them also
    # in the inherited parser's --help for an entirely self-documenting CLI.
    p.add_argument('--search-mode',choices=('pilot','full'),default='pilot',
                   help='Full: all SGs, wider basin search, per-SG top 10')
    p.add_argument('--sg-shard-count',type=int,default=1,
                   help='Partition SG search among independent jobs')
    p.add_argument('--sg-shard-index',type=int,default=0,
                   help='Zero-based shard index')
    p.add_argument('--site-tries',type=int,default=24,
                   help='Full-SG TiO6 site-compatible orientation tries (v60)')
    p.add_argument('--center-scf-mix',type=float,default=.04,
                   help='Wyckoff-free Ti-center SCF update (v60)')
    p.add_argument('--oo-probe-weight',type=float,default=4.,
                   help='Continuous O-O repulsion weight in cheap probe selection')
    p.add_argument('--oo-archive-weight',type=float,default=8.,
                   help='Continuous O-O repulsion weight in final archive')
    p.add_argument('--sharing-archive-weight',type=float,default=.05,
                   help='Sharing mismatch weight in final archive')
    p.add_argument('--full-sg-audit-tolerance',type=float,default=.055)
    p.add_argument('--require-full-o-symmetry',action='store_true',
                   help='Optional full Ti/O symmetry gate, normally off')
    p.add_argument('--print-probe-errors',type=int,default=3)
    p.add_argument('--ref-folder',dest='ref',default='ref_tio2',
                   help='Reference CIF folder (only public reference-input flag)')
    p.add_argument('--ref',dest='ref',help=argparse.SUPPRESS)

    p.add_argument('--space-groups',default='61,87,136,141')
    p.add_argument('--formula-units',type=int,default=24)
    p.add_argument('--wp-per-sg',type=int,default=2)
    p.add_argument('--wp-scan-limit',type=int,default=500)
    p.add_argument('--max-ti-orbits',type=int,default=24)
    p.add_argument('--metal-cycles',type=int,default=1)
    p.add_argument('--raw-starts',type=int,default=16)
    p.add_argument('--optimized-starts',type=int,default=6)
    p.add_argument('--basins-per-wp',type=int,default=3)
    p.add_argument('--natural-steps',type=int,default=12)
    p.add_argument('--learned-steps',type=int,default=20)
    p.add_argument('--natural-lr',type=float,default=.04)
    p.add_argument('--learned-lr',type=float,default=.02)
    p.add_argument('--entrances',type=int,default=8,
                   help='Independent whole-TiO6 LCAO starts for EACH distinct Ti framework')
    p.add_argument('--modes-per-ti',type=int,default=4)
    p.add_argument('--octahedral-proposals',type=int,default=8)
    p.add_argument('--probe-scf-cycles',type=int,default=6,
                   help='Cheap LCAO/SCF cycles for each fresh basin proposal')
    p.add_argument('--beam-width',type=int,default=4)
    p.add_argument('--beam-generations',type=int,default=2)
    p.add_argument('--mutations-per-parent',type=int,default=1)
    p.add_argument('--rotation-amplitude',type=float,default=.60)
    p.add_argument('--pool-per-sg',type=int,default=32,
                   help='Maximum retained distinct, cheap-probed candidates per SG in memory')
    p.add_argument('--refine-top',type=int,default=8,
                   help='Distinct finalists per SG to receive longer SCF and coherent feedback')
    p.add_argument('--refine-scf-cycles',type=int,default=12)
    p.add_argument('--feedback-steps',type=int,default=12)
    p.add_argument('--feedback-rounds',type=int,default=1)
    p.add_argument('--feedback-oxygen-weight',type=float,default=0.,
                   help='0: Ti-first soft chemistry only; 1: also learned O-centered geometry')
    p.add_argument('--strain-max',type=float,default=.11)
    p.add_argument('--max-unique-per-sg',type=int,default=5,
                   help='Ranked, structurally unique, FINAL CIFs per initial SG (up to 10)')
    p.add_argument('--max-o-move',type=float,default=.10,
                   help='DIAGNOSTIC threshold ONLY; does not reject ranked structures')
    p.add_argument('--device',default='auto')
    p.add_argument('--gpu-workers',default='auto',
                   help='auto: one worker per visible GPU; or specify 1,2,4,...')
    p.add_argument('--gpu-plan-only',action='store_true',
                   help='Print GPU/space-group assignments, without running')
    p.add_argument('--gpu-stream-logs',action='store_true',
                   help='Also show full worker logs in central stdout (default: compact progress)')
    p.add_argument('--ti-rule-mode',choices=('hybrid','learned','legacy'),default='hybrid')
    p.add_argument('--shell-max-k',type=int,default=30)
    p.add_argument('--seed',type=int,default=37)
    p.add_argument('--output',default='tio2_v60_ti_wyckoff')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(argv)
    if not 2<=a.formula_units<=96:p.error('formula-units must be 2..96')
    if not 1<=a.max_unique_per_sg<=10:p.error('max-unique-per-sg must be between 1 and 10')
    if not 0<a.max_o_move<=.1:p.error('0 < max-o-move <= .10')
    if not 0<=a.strain_max<=.11:p.error('0 <= strain-max <= .11')
    for name in ('raw_starts','optimized_starts','basins_per_wp','entrances',
                 'probe_scf_cycles','beam_width','pool_per_sg','refine_top',
                 'refine_scf_cycles','metal_cycles'):
        if getattr(a,name)<1:p.error(f'{name} must be >=1')
    if a.raw_starts<a.optimized_starts:p.error('raw-starts must be >= optimized-starts')
    if a.pool_per_sg<a.refine_top:p.error('pool-per-sg must be >= refine-top')
    if a.beam_generations<0 or a.mutations_per_parent<1:p.error('invalid mutation budget')
    if a.feedback_rounds<0 or a.feedback_steps<0:p.error('invalid feedback budget')
    if a.feedback_oxygen_weight<0 or a.feedback_oxygen_weight>1:p.error('feedback-oxygen-weight must be 0..1')
    if a.rotation_amplitude<=0:p.error('rotation-amplitude must be positive')
    parse_sgs(a.space_groups)
    return a

def e593_probe_key(item):
    """Continuous priority, never a hard chemistry/coordination pass/fail."""
    r=item['ans']
    return float(continuous_candidate_objective(r))

def e593_topology(item):
    """Diversity fingerprint: framework, sharing topology, coarse rigid pose.

    Two entrances with the same oxygen-sharing graph but different whole-unit
    orientations must be allowed to compete: this was a central failure mode
    seen in the paired v59.2 control. Coarse pose bins are only for search
    diversity, never for scientific scoring or structural deduplication.
    """
    discrete=global_basin_signature(item['ans'])
    oc=item['ans']['oct']
    poses=getattr(oc,'rot',None)
    orientation=()
    if poses is not None:
        orientation=tuple(tuple(np.rint(np.asarray(r.as_rotvec())/.35).astype(int))
                          for r in poses[:min(8,len(poses))])
    framework=(item.get('sg'),tuple(item.get('wp',())),
               item.get('metal_cycle'),item.get('basin'))
    return (framework,discrete,orientation)

def e593_quality(item,kind):
    ans=item['ans'];d=ans['details']
    score=e593_probe_key(item)
    if kind=='score':return (score,float(ans['rms']))
    if kind=='sharing':return (float(ans['rms']),score)
    if kind=='Ti':return (-int(np.sum(d.get('goodTi',[]))),float(np.max(d.get('localTi',[np.inf]))),score)
    if kind=='O':return (-int(np.sum(d.get('goodO',[]))),float(np.max(d.get('localO',[np.inf]))),score)
    if kind=='topology':return (int(ans['bad_triplets']),score)
    raise ValueError(kind)

def e593_select_diverse(items, capacity):
    """Reserve different scientific perspectives; do not only keep low RMS.

    Identical mode/sharing topology retains the lowest continuous energy,
    while genuinely distinct sharing basins remain eligible even when the
    harsh 'goodTi/goodO' binary flags are all zero.
    """
    if not items or capacity<=0:return []
    by_sig={}
    for item in items:
        if not np.isfinite(e593_probe_key(item)):continue
        sig=e593_topology(item)
        old=by_sig.get(sig)
        if old is None or e593_probe_key(item)<e593_probe_key(old):
            by_sig[sig]=item
    remaining=list(by_sig.values())
    chosen=[];selected=set()
    # Reserve slots before quality filling. A topologically distinct candidate
    # with weak binary diagnostics is NOT automatically discarded.
    for kind in ('score','Ti','O','sharing','topology'):
        if len(chosen)>=capacity:break
        for item in sorted(remaining,key=lambda x:e593_quality(x,kind)):
            sig=e593_topology(item)
            if sig not in selected:
                chosen.append(item);selected.add(sig);break
    for item in sorted(remaining,key=e593_probe_key):
        if len(chosen)>=capacity:break
        sig=e593_topology(item)
        if sig not in selected:
            chosen.append(item);selected.add(sig)
    return chosen


def e593_new_item(ans,info,source,move):
    out=dict(info)
    out.update(ans=ans,source=source,move=move)
    return out

def e593_search_framework(ti,cell,library,chem_probe,ops,a,seed,info):
    """Build LCAO proposals once; move entire TiO6s between short probes."""
    mode_options=mode_shortlist(ti,cell,library,min(a.modes_per_ti,len(library['Ti'])))
    proposals=framework_orientation_seeds(ti,cell,library,mode_options,
                                           np.empty((0,3),float),
                                           proposals_per_ti=a.octahedral_proposals)
    def probe(octa,code,source,move):
        octa._ablation_max_strain=a.strain_max
        ans=scf(octa,np.empty((0,3),float),chem_probe,mode_options,ops,
                steps=a.probe_scf_cycles,seed=code,symmetry_tol=.17)
        item=e593_new_item(ans,info,source,move)
        item['mode_options']=mode_options
        return item
    pool=[];errors=0
    for n in range(a.entrances):
        s=0 if n==0 else str_seed(seed,'entrance',n)
        try:
            octa=Octahedra(ti,cell,library,proposals,seed=s)
            pool.append(probe(octa,str_seed(seed,'probe',n),'entrance',f'fresh{n}'))
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            errors+=1
    beam=e593_select_diverse(pool,a.beam_width)
    seen=set(global_basin_signature(z['ans']) for z in pool)
    rng=np.random.default_rng(str_seed(seed,'whole-polyhedron-basin-mutations'))
    new_basin_count=0
    for generation in range(a.beam_generations):
        next_items=[]
        if not beam:break
        # Rotate parent ordering to include weaker but topologically distinct
        # basins. This is not a gradient minimization or greedy-only descent.
        order=np.roll(np.arange(len(beam)),generation % len(beam))
        for parent_idx in order:
            parent=beam[int(parent_idx)]
            for repeat in range(a.mutations_per_parent):
                try:
                    mutation,move=global_octahedral_move(
                        parent['ans']['oct'],ops,proposals,mode_options,rng,
                        amplitude=a.rotation_amplitude,
                        force_escape=bool(generation%2))
                    # Respect user-selected deformation cap after the coherent move.
                    for i,D in enumerate(mutation.stretch):
                        vals,U=np.linalg.eigh(D)
                        mutation.stretch[i]=(U@np.diag(np.clip(vals,1-a.strain_max,1+a.strain_max))@U.T)
                    mutation._ablation_max_strain=a.strain_max
                    item=probe(mutation,str_seed(seed,'mutation',generation,int(parent_idx),repeat),
                               'mutation',move)
                    sig=global_basin_signature(item['ans'])
                    if sig not in seen:
                        new_basin_count+=1;seen.add(sig)
                    next_items.append(item)
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError):
                    errors+=1
        pool.extend(next_items)
        beam=e593_select_diverse(beam+next_items,a.beam_width)
    return pool,dict(initial=len(pool)-sum(1 for z in pool if z['source']=='mutation'),
                     total=len(pool),new_topologies=new_basin_count,
                     distinct_topologies=len(seen),errors=errors)

def e593_final_candidates(item,a):
    """Return BOTH initial probe and final refinement: refinement may worsen."""
    source=item['ans']; chemistry=item['chem_probe']; ops=item['ops']
    choices=item['mode_options']
    results=[('probe',source)]
    try:
        o=source['oct'].copy();o._ablation_max_strain=a.strain_max
        improved=scf(o,np.empty((0,3)),chemistry,choices,ops,
                     steps=a.refine_scf_cycles,
                     seed=str_seed(a.seed,'final-scf',item['sg'],item['wp'],
                                   item['metal_cycle'],item['basin'],item['entrance'],item['move']),
                     symmetry_tol=.17)
        results.append(('refine_scf',improved))
        if a.feedback_steps>0 and a.feedback_rounds>0:
            further=coherent_chemistry_feedback(improved,chemistry,ops,
                       steps=a.feedback_steps,rounds=a.feedback_rounds,
                       strain_limit=a.strain_max)
            results.append(('coherent_feedback',further))
    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
        log(f'    FINAL REFINE warning: {type(exc).__name__}: {exc}')
    return results

def e593_result_as_archive_row(result,record):
    oc=result['oct']
    return dict(ti=np.asarray(oc.ti,float).copy(),
                cell=np.asarray(oc.cell,float).copy(),
                answer=dict(oxygen=np.asarray(result['oxygen'],float).copy()),
                q=dict(maxTi=record['Ti_loss_max'],meanTi=record['Ti_loss_avg'],
                       minTi=record['Ti_loss_min'],
                       goodTi=record['n_good_ti'],goodO=record['n_good_o'],
                       sharing=record['rms_o_move'],maxO=record['O_loss_max'],
                       meanO=record['O_loss_avg']))

def e593_write_table(path,records,columns=None):
    path=Path(path)
    if not records and columns is None:columns=['sg_initial','status']
    if columns is None:columns=list(dict.fromkeys(k for x in records for k in x))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader()
        for r in records:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in r.items()})

def e593_self_test():
    ab59_self_test()
    a=e593_parse(['--space-groups','1','--max-unique-per-sg','5',
                  '--beam-width','4','--refine-top','2'])
    # Strict binary flags must NOT affect basin selection.
    def fixture(score,good,signature):
        return dict(ans=dict(details=dict(energy=score,collision=0.,hierarchy='ti-first',
                                          goodTi=np.zeros(2,bool)+bool(good),
                                          goodO=np.zeros(4,bool)+bool(good),
                                          localTi=np.ones(2)*score,
                                          localO=np.ones(4)*score),
                             rms=.2,bad_triplets=0,oxygen=np.zeros((4,3)),
                             trip=np.asarray(signature,dtype=int),
                             oct=argparse.Namespace(mode=np.array([0,0]))))
    pool=[fixture(1.,False,[0,0,1,1]),
          fixture(.2,False,[0,1,0,1]),
          fixture(.3,True,[0,1,1,0]),
          fixture(.5,False,[0,0,1,1])]
    selected=e593_select_diverse(pool,3)
    assert len(selected)==3,len(selected)
    assert all(e593_topology(s) is not None for s in selected)
    assert len(set(e593_topology(x) for x in selected))==3
    assert any(e593_probe_key(x)>.5 for x in selected), 'zero-good basins must remain eligible'
    assert a.refine_top<=a.pool_per_sg
    with tempfile.TemporaryDirectory() as d:
        e593_write_table(Path(d)/'probes.csv',[dict(sg_initial=1,status='screened')])
        assert (Path(d)/'probes.csv').is_file()
    log('V59.3 SELF-TEST PASSED: no PASS gate, diverse topology pool, LCAO/strain inheritance')


# ============================================================================
# v59.4: WYCKOFF-ORBIT POLYHEDRA + SYMMETRIC LCAO
# The v59.2 chemistry, Ti-only optimizer, CIF archive, and exploration helpers
# above remain intact. Only the post-metal structural DOFs are replaced.
# ============================================================================

from scipy.linalg import null_space as _w594_null_space

class WyckoffIncompatible(ValueError):
    """A learned shape cannot realize the required site symmetry at allowed strain."""

def w594_group_operations(group):
    """Explicit requested SG, NEVER symmetry inferred from the Ti-only lattice."""
    # The general Wyckoff position lists the FULL space-group operations.
    # PyXtal Group itself has no public `.ops` attribute in recent versions.
    ops=group[0].ops
    return [(np.asarray(op.rotation_matrix,dtype=float),
             np.asarray(op.translation_vector,dtype=float)) for op in ops]

def w594_cart_action(cell,frac_rotation):
    return np.linalg.inv(cell) @ np.asarray(frac_rotation,dtype=float).T @ cell

def w594_wrap_delta(x,y):
    return (np.asarray(x)-np.asarray(y)+.5)%1.-.5

def w594_full_group_orbit_maps(cell,representative,wp_sites,group_ops,tol=2e-5):
    """Choose invertible *space-group* operations mapping the representative
    onto each Wyckoff image.

    `wp['rot']` describes a WP coordinate parametrization.  For special sites
    its matrices may be singular projectors, and MUST NOT be inverted or used
    to rotate an attached TiO6.  The full general-WP operations are genuine
    lattice isometries, so their Cartesian rotations are invertible.
    """
    center=np.asarray(representative,dtype=float)%1.
    sites=np.asarray(wp_sites,dtype=float)%1.
    full=[]
    for R,t in group_ops:
        R=np.asarray(R,dtype=float)
        t=np.asarray(t,dtype=float)
        M=w594_cart_action(cell,R)
        if abs(abs(np.linalg.det(M))-1.)>1e-5:
            raise ValueError('Non-isometric full space-group rotation')
        full.append((R,t,M))
    if not full:raise WyckoffIncompatible('Space group has no operations')
    maps=[]
    for i,pos in enumerate(sites):
        matches=[]
        for R,t,M in full:
            delta=w594_wrap_delta(center@R.T+t,pos)
            err=float(np.linalg.norm(delta@cell))
            if err<tol:
                # When a WP member is the representative, prefer identity;
                # otherwise any coset representative is allowed.  The site
                # stabilizer ensures all such choices yield the same unlabelled
                # complete polyhedron.
                identity=float(np.linalg.norm(M-np.eye(3)))
                matches.append((identity,err,M))
        if not matches:
            nearest=min(float(np.linalg.norm(w594_wrap_delta(center@R.T+t,pos)@cell))
                        for R,t,_ in full)
            raise WyckoffIncompatible(
                f'No full-SG operation maps TiO6 representative to WP image {i}; '
                f'closest={nearest:.6g} A')
        matches.sort(key=lambda row:(row[0],row[1]))
        maps.append(matches[0][2])
    return np.asarray(maps,dtype=float)

def w594_site_ops(group_ops,rep,cell,tol=3e-6):
    site=[]
    for rot,trans in group_ops:
        if np.max(np.abs(w594_wrap_delta(rep@rot.T+trans,rep)))<tol:
            site.append(w594_cart_action(cell,rot))
    return site

def w594_polardecompose(F):
    """Row-vector deformation F=D @ R.T, D symmetric, R proper."""
    U,s,VT=np.linalg.svd(F)
    if np.linalg.det(U@VT)<0:
        return None,None
    D=(U*s)@U.T
    R=U@VT
    return D,R

def w594_bound_F(F,max_strain):
    D,R=w594_polardecompose(F)
    if D is None:return None
    eig,U=np.linalg.eigh(D)
    eig=np.clip(eig,1-max_strain,1+max_strain)
    return (U@np.diag(eig)@U.T)@R

def w594_vertex_permutation(verts,mat):
    transformed=verts@mat
    distances=np.linalg.norm(transformed[:,None,:]-verts[None,:,:],axis=-1)
    i,j=linear_sum_assignment(distances**2)
    permutation=np.empty(len(verts),dtype=int)
    permutation[i]=j
    error=float(np.max(distances[i,j]))
    return permutation,error

def w594_fixed_vertex_basis(src,site_mats,F_hint):
    """Null-space of exact vertex-invariance constraints for one set of labels.

    For a point-site operation g with vertex permutation p, the 3x3 coherent
    transform F must satisfy (src @ F) @ g = (src @ F)[p].  The equations are
    LINEAR in the nine entries of F. This gives an exact, differentiable-in-
    the-subspace representation, not a soft symmetry penalty.
    """
    rows=[];permutations=[]
    for mat in site_mats:
        p,_=w594_vertex_permutation(src@F_hint,mat)
        permutations.append(p)
        for i in range(len(src)):
            for j in range(3):
                coefficients=np.zeros((3,3))
                for k in range(3):
                    coefficients[k,:]+=src[i,k]*mat[:,j]
                    coefficients[k,j]-=src[p[i],k]
                rows.append(coefficients.reshape(9))
    if not rows:
        return np.eye(9),permutations
    A=np.asarray(rows,dtype=float)
    A[np.abs(A)<1e-9]=0.
    basis=np.eye(9) if not np.any(A) else _w594_null_space(A,rcond=1e-10)
    return basis,permutations

def w594_project_F(src,site_mats,F_hint,max_strain,permutations=None):
    """Find coherent, site-symmetric F close to the requested mode/pose.

    Reject if exact site symmetry cannot coexist with the bounded deformation;
    never silently lower the SG or deform six vertices independently.
    """
    if not site_mats:
        return w594_bound_F(F_hint,max_strain),np.eye(9),[]
    if permutations is None:
        B,permutations=w594_fixed_vertex_basis(src,site_mats,F_hint)
    else:
        rows=[]
        for mat,p in zip(site_mats,permutations):
            for i in range(len(src)):
                for j in range(3):
                    cof=np.zeros((3,3))
                    for k in range(3):
                        cof[k,:]+=src[i,k]*mat[:,j]
                        cof[k,j]-=src[p[i],k]
                    rows.append(cof.reshape(-1))
        A=np.asarray(rows,dtype=float) if rows else np.zeros((0,9))
        A[np.abs(A)<1e-9]=0.
        B=_w594_null_space(A,rcond=1e-10) if np.any(A) else np.eye(9)
    if B.shape[1]<1:return None,B,permutations
    projected=(B@(B.T@F_hint.reshape(9))).reshape(3,3)
    D,R=w594_polardecompose(projected)
    if D is None:return None,B,permutations
    eig=np.linalg.eigvalsh(D)
    if np.min(eig)<1-max_strain-1e-7 or np.max(eig)>1+max_strain+1e-7:
        return None,B,permutations
    error=max((w594_vertex_permutation(src@projected,mat)[1] for mat in site_mats),default=0)
    if error>1e-5:return None,B,permutations
    return projected,B,permutations

class WyckoffPolyhedron:
    """ONE independent TiO6 Wyckoff orbit, including all symmetry images.

    Degree-of-freedom inventory: free Wyckoff coordinates u; discrete rigid
    shape mode; three-component spatial orientation (direction + roll); and
    one coherent 3x3 deformation. On special sites the 9 affine coefficients
    of F=D@R.T are restricted to the exact site-symmetry null-space.
    """
    def __init__(self,wp,cell,group_ops,library,mode,rotation,max_strain=.11,
                 u=None,global_offset=0,site_attempts=12,seed=0,
                 building_center_type=None,building_block=None):
        self.building_center_type=building_center_type or 'Ti'
        self.building_block=building_block
        self.wp=wp;self.cell=np.asarray(cell,float);self.lib=library
        self.global_offset=int(global_offset)
        self.count=len(wp['rot'])
        self.A=np.asarray(wp['A'].detach().cpu(),float)
        self.b=np.asarray(wp['b'].detach().cpu(),float)
        self.dof=int(wp['dof'])
        self.u=(np.full(self.dof,.5) if u is None else np.asarray(u,float).copy())
        self.rotfrac=np.asarray(wp['rot'].detach().cpu(),float)
        self.trans=np.asarray(wp['tra'].detach().cpu(),float)
        # WP coordinate operations can be rank-deficient at special sites.
        # Attached rigid/deformed polyhedra must propagate through actual SG
        # isometries, never through WP projectors.
        self.maps=w594_full_group_orbit_maps(self.cell,
                        (self.A@self.u+self.b)%1.,self.frac_sites(),group_ops)
        self.max_strain=float(max_strain)
        self.group_ops=group_ops
        self.mode=int(mode)
        self.site_ops=w594_site_ops(group_ops,self.representative(),cell)
        self.F=None;self.B=None;self.permutations=None
        self.set_pose(rotation,site_attempts=site_attempts,seed=seed)

    def representative(self):
        return (self.A@self.u+self.b)%1.

    def frac_sites(self):
        rep=self.representative()
        return (np.einsum('oij,j->oi',self.rotfrac,rep)+self.trans)%1.

    def src(self,mode=None):
        # v71 typed source geometry; original TiO2/v64 path is unchanged.
        if self.building_block is not None:
            modes=self.building_block.modes
            idx=self.mode if mode is None else int(mode)
            return np.asarray(modes[idx],float)
        return np.asarray(self.lib['Ti'][self.mode if mode is None else int(mode)]['rigid_arms'],float)

    def _candidate(self,mode,F):
        return w594_project_F(self.src(mode),self.site_ops,F,self.max_strain)

    def set_pose(self,rotation,site_attempts=12,seed=0):
        # General-position Ti centers have the identity site stabilizer. The
        # very first learned orientation is already site compatible with zero
        # coherent strain; evaluating dozens of random alternative rotations
        # for each independent orbit is both redundant and extremely costly
        # in low-symmetry space groups. This is equivalent to selecting the
        # first zero-projection-error candidate from the original search.
        if len(self.site_ops)==1:
            candidate,B,p=self._candidate(self.mode,rotation.as_matrix().T)
            if candidate is not None and np.linalg.norm(
                    candidate-rotation.as_matrix().T)<1.e-8:
                self.F,self.B,self.permutations=candidate,B,p
                return
        rng=np.random.default_rng(seed)
        rotations=[rotation,Rotation.identity()]
        if len(self.site_ops)>1:
            # Discrete, crystallographically adapted SO(3) entrances. Align
            # the learned template's principal axes with invariants of the
            # local site point group, including the 24 handed axis mappings.
            # This avoids relying entirely on random rotations at special sites.
            _,source_axes=np.linalg.eigh(self.src().T@self.src())
            probe_tensor=np.diag([1.,2.,3.])
            point_metric=np.sum([M.T@probe_tensor@M for M in self.site_ops],axis=0)
            _,site_axes=np.linalg.eigh(point_metric)
            for perm in itertools.permutations(range(3)):
                for signs in itertools.product((-1.,1.),repeat=3):
                    P=np.eye(3)[:,perm]*np.asarray(signs)[None,:]
                    F=source_axes@P@site_axes.T
                    if np.linalg.det(F)>0.:
                        rotations.append(Rotation.from_matrix(F.T))
        for _ in range(max(0,site_attempts-1)):
            rotations.append(Rotation.random(random_state=rng))
        best=None
        for rr in rotations:
            guess=rr.as_matrix().T
            candidate,B,p=self._candidate(self.mode,guess)
            if candidate is None:continue
            error=np.linalg.norm(candidate-guess)
            if best is None or error<best[0]:best=(error,candidate,B,p)
        if best is None:
            raise WyckoffIncompatible(
                f'WP {self.wp["label"]} mode {self.mode}: no bounded-strain pose '
                f'compatible with site stabilizer of order {len(self.site_ops)}')
        _,self.F,self.B,self.permutations=best

    def try_mode_pose(self,mode,rotation):
        mode=int(mode)
        F,B,p=self._candidate(mode,rotation.as_matrix().T)
        if F is None:return False
        self.mode=mode;self.F=F;self.B=B;self.permutations=p
        return True

    @property
    def direction_vector(self):
        """Three-component axis-angle vector: direction INCLUDING roll (SO(3))."""
        D,rotation=w594_polardecompose(self.F)
        if rotation is None:raise RuntimeError('Improper coherent transform')
        return Rotation.from_matrix(rotation.T).as_rotvec()

    @property
    def deformation_matrix(self):
        """One 3x3 symmetric coherent strain/stretch; no per-vertex shifts."""
        D,rotation=w594_polardecompose(self.F)
        if D is None:raise RuntimeError('Improper coherent transform')
        return D

    def source_vectors(self):
        return self.src()@self.F

    def site_error(self):
        src=self.source_vectors()
        return max((w594_vertex_permutation(src,g)[1] for g in self.site_ops),default=0.)

    def positions(self):
        local=self.source_vectors()
        return (self.frac_sites()@self.cell)[:,None,:] + np.einsum('vk,okj->ovj',local,self.maps)

    def fit(self,targets,mix=.55,mode_candidates=None,distortion_ridge=1.5):
        # Gather every symmetry-related target into the representative frame.
        pulled=[]
        for i,mat in enumerate(self.maps):
            pulled.append(np.asarray(targets[i])@np.linalg.inv(mat))
        target=np.mean(pulled,axis=0)
        candidate_modes=[self.mode] if mode_candidates is None else list(dict.fromkeys(
            [self.mode]+[int(x) for x in mode_candidates]))
        variants=[]
        for mode in candidate_modes:
            src=self.src(mode)
            # Fully coherent affine fitting, constrained to site invariants.
            # Corresponding arm labels are inherited from the LCAO triplets.
            try:
                rr,_=Rotation.align_vectors(target,src)
            except (ValueError,np.linalg.LinAlgError):
                continue
            hint=rr.as_matrix().T
            if mode==self.mode:
                B=self.B;p=self.permutations
            else:
                _,B,p=w594_project_F(src,self.site_ops,hint,self.max_strain)
            if B is None or B.shape[1]<1:continue
            # Fit F in the invariant basis, with ridge near current pose.
            # Both src@F and the desired target have shape [6,3].
            ridge=.10*float(distortion_ridge)
            fit_mat=np.kron(src,np.eye(3))@B
            rhs=np.concatenate((target.reshape(-1),np.sqrt(ridge)*hint.reshape(-1)))
            lhs=np.vstack((fit_mat,np.sqrt(ridge)*B))
            q=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
            F=(B@q).reshape(3,3)
            D,R=w594_polardecompose(F)
            if D is None:continue
            eig=np.linalg.eigvalsh(D)
            if np.min(eig)<1-self.max_strain-1e-7 or np.max(eig)>1+self.max_strain+1e-7:
                # Soft candidate can be unsuitable; do not distort to fit.
                continue
            e=float(np.mean(np.sum((src@F-target)**2,axis=1)))
            variants.append((e,mode,F,B,p))
        if not variants:return False
        variants.sort(key=lambda x:x[0])
        _,mode,F,B,p=variants[0]
        if mode!=self.mode:
            self.mode=mode;self.F=F;self.B=B;self.permutations=p
        else:
            mixed=(1-mix)*self.F+mix*F
            mixed=(B@(B.T@mixed.reshape(9))).reshape(3,3)
            # CRITICAL: direct interpolation between two allowed rotations can
            # collapse axes. Both endpoints can satisfy 11% eigenstrain while
            # their midpoint violates it by 80% or more. Never store an
            # unbounded mixed F. Using the allowed fitted endpoint retains
            # both exact site symmetry and the strain bound.
            dm,rm=w594_polardecompose(mixed)
            if dm is not None and np.max(np.abs(np.linalg.eigvalsh(dm)-1.))<=self.max_strain+1e-7:
                self.F=mixed
            else:
                self.F=F
        if self.site_error()>2e-5:raise RuntimeError('Special-site polyhedron symmetry drift')
        if np.max(np.abs(np.linalg.svd(self.F,compute_uv=False)-1))>self.max_strain+1e-5:
            raise RuntimeError('Wyckoff orbit fit exceeded coherent strain bound')
        return True

    def move(self,kind,rng,amplitude,proposal=None,mode_options=None):
        old=(self.mode,self.F.copy(),self.B.copy(),list(self.permutations))
        if kind=='mode' and mode_options:
            candidates=[int(x) for x in mode_options if int(x)!=self.mode]
            rng.shuffle(candidates)
            for m in candidates:
                if self.try_mode_pose(m,Rotation.random(random_state=rng)):
                    return True
            return False
        if kind=='reseed' and proposal:
            choices=list(proposal);rng.shuffle(choices)
            for _,mode,rot in choices:
                if self.try_mode_pose(int(mode),rot):return True
            return False
        delta=rng.normal(size=3)
        delta*=amplitude/max(np.linalg.norm(delta),1e-12)
        Fhint=self.F@Rotation.from_rotvec(delta).as_matrix().T
        F,B,p=w594_project_F(self.src(),self.site_ops,Fhint,self.max_strain,self.permutations)
        if F is not None:
            self.F=F;self.B=B;self.permutations=p;return True
        self.mode,self.F,self.B,self.permutations=old
        return False

class WyckoffOctahedra:
    """SCF-compatible view; the ONLY mutable TiO6 parameters belong to WPs."""
    def __init__(self,template,raw,cell,lib,proposals,group_ops,seed=0,
                 max_strain=.11,site_attempts=12):
        self.template=template;self.lib=lib
        self.cell=np.asarray(cell,float).copy();self.inv=np.linalg.inv(cell)
        self.max_strain=float(max_strain)
        self.group_ops=group_ops
        self.orbits=[]
        rng=np.random.default_rng(seed)
        raw=np.asarray(raw,float).reshape(-1)
        offset=len(template.lattice_keys)
        index=0
        for bi,wp in enumerate(template.blocks):
            d=int(wp['dof'])
            u=1/(1+np.exp(-np.clip(raw[offset:offset+d],-25,25)))
            offset+=d
            prop=proposals[index]
            if seed==0:choice=0
            else:
                weights=np.exp(-np.clip(np.asarray([p[0] for p in prop])-prop[0][0],0,2)/.07)
                choice=int(rng.choice(len(prop),p=weights/weights.sum()))
            _,mode,rot=prop[choice]
            # A special-site orbit may require a different orientation/mode.
            # Try alternative learned poses; NEVER silently discard its stabilizer.
            alternatives=[(int(mode),rot)]+[(int(m),r) for _,m,r in prop if int(m)!=int(mode) or r is not rot]
            last=None;obj=None
            for m,r in alternatives:
                try:
                    obj=WyckoffPolyhedron(wp,cell,group_ops,lib,m,r,
                                        max_strain=max_strain,u=u,global_offset=index,
                                        site_attempts=site_attempts,seed=str_seed(seed,bi,m))
                    break
                except WyckoffIncompatible as exc:
                    last=exc
            if obj is None:raise WyckoffIncompatible(str(last))
            self.orbits.append(obj)
            index+=len(wp['rot'])
        self._rebuild()
        if len(self.ti)!=template.n_ti:raise AssertionError('Incorrect expanded orbit size')

    def _rebuild(self):
        self.ti=np.concatenate([o.frac_sites() for o in self.orbits])
        self.mode=np.concatenate([np.repeat(o.mode,o.count) for o in self.orbits])
        # This read-only analysis view is not used to generate vertex positions.
        # Actual Cartesian arms are computed by equivariant F @ group maps.
        self.stretch=np.concatenate([
            np.repeat(w594_polardecompose(o.F)[0][None],o.count,axis=0)
            for o in self.orbits])

    def copy(self):return copy.deepcopy(self)

    def positions(self):
        return np.concatenate([o.positions() for o in self.orbits],axis=0)

    def distortion(self):
        return float(np.sqrt(np.mean(np.sum((self.stretch-np.eye(3))**2,axis=(1,2)))))

    def fit(self,targets,mode_options=None,mix=.55,distortion_max=.11,distortion_ridge=1.5):
        if distortion_max<self.max_strain-1e-9:
            self.max_strain=float(distortion_max)
            for o in self.orbits:o.max_strain=self.max_strain
        index=0
        for orb in self.orbits:
            pool=[]
            if mode_options is not None:
                for opts in mode_options[index:index+orb.count]:
                    pool.extend([int(k) for k,_ in opts])
            orb.fit(targets[index:index+orb.count],mix=mix,
                    mode_candidates=pool or None,distortion_ridge=distortion_ridge)
            index+=orb.count
        self._rebuild()

    def mutate(self,rng,amplitude,proposals,mode_options):
        trial=self.copy()
        oi=int(rng.integers(len(trial.orbits)))
        kind=str(rng.choice(['rotate','rotate','mode','reseed']))
        orbit=trial.orbits[oi]
        off=orbit.global_offset
        options=[int(k) for k,_ in mode_options[off]]
        if kind=='mode' and len(set(options))<=1:
            kind='rotate'  # KCaH3-style single-mode orbit remains explorable.
        changed=orbit.move(kind,rng,amplitude,proposal=proposals[off],mode_options=options)
        if not changed:raise WyckoffIncompatible('No site-compatible orbit mutation')
        trial._rebuild()
        return trial,kind

    def symmetry_error(self):
        return max([o.site_error() for o in self.orbits] or [0.])

    def n_independent(self):return len(self.orbits)

class WyckoffOxygenOrbit:
    """An O Wyckoff orbit: coordinates are fitted FROM LCAO centroids only."""
    def __init__(self,wp,cell,u,global_offset=0):
        self.wp=wp
        self.cell=np.asarray(cell,float)
        self.A=np.asarray(wp['A'].detach().cpu(),float)
        self.b=np.asarray(wp['b'].detach().cpu(),float)
        self.rotfrac=np.asarray(wp['rot'].detach().cpu(),float)
        self.trans=np.asarray(wp['tra'].detach().cpu(),float)
        self.count=len(self.rotfrac)
        self.dof=int(wp['dof'])
        self.u=np.asarray(u,float).copy()
        self.global_offset=int(global_offset)

    def frac(self):
        rep=(self.A@self.u+self.b)%1.
        return (np.einsum('oij,j->oi',self.rotfrac,rep)+self.trans)%1.

    def fit_centroids(self,centroids,step=.85):
        if self.dof==0:return
        prev=self.frac()
        target=np.asarray(centroids,float)%1.
        change=w594_wrap_delta(target,prev)
        M=np.concatenate([R@self.A for R in self.rotfrac],axis=0)
        adjustment=np.linalg.lstsq(M,change.reshape(-1),rcond=None)[0]
        adjustment=np.clip(adjustment,-.08,.08)
        self.u=(self.u+step*adjustment)%1.

def w594_oxygen_representative(wp,cell,centers,seed_index=0):
    """Project LCAO oxygen proposals onto a candidate O Wyckoff orbit.

    Solve each member's *affine* free-coordinate map directly.  The rotation
    matrices stored by PyXtal for special Wyckoff coordinates need not be
    invertible: even zero-DOF O sites are legitimate LCAO hypotheses.
    """
    A=np.asarray(wp['A'].detach().cpu(),float)
    b=np.asarray(wp['b'].detach().cpu(),float)
    rot=np.asarray(wp['rot'].detach().cpu(),float)
    trans=np.asarray(wp['tra'].detach().cpu(),float)
    centers=np.asarray(centers,float)%1.
    if A.shape[1]==0:
        # Fixed O sites MUST be scored against the generated LCAO oxygen
        # proposals. Returning zero here (old v59.4/v59.5 behavior) made an
        # unrelated special-site pattern look like a perfect projection.
        fixed=(np.einsum('oij,j->oi',rot,b)+trans)%1.
        ds=np.linalg.norm(w594_wrap_delta(fixed[:,None,:],centers[None,:,:])@cell,axis=-1)
        ir,ic=linear_sum_assignment(ds**2)
        return np.empty(0),float(np.sum(ds[ir,ic]**2))
    rng=np.random.default_rng(seed_index)
    probes=rng.choice(len(centers),size=min(4,len(centers)),replace=False)
    best=None
    shifts=[(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    for index in probes:
        seed=centers[int(index)]
        for R,t in zip(rot,trans):
            # image(u)=R@(A@u+b)+t: use the full affine Jacobian.
            J=R@A
            base=R@b+t
            for shift in shifts:
                y=seed+np.asarray(shift,dtype=float)-base
                u=np.linalg.lstsq(J,y,rcond=None)[0]%1.
                if not np.isfinite(u).all():continue
                orb=WyckoffOxygenOrbit(wp,cell,u).frac()
                delta=w594_wrap_delta(orb[:,None,:],centers[None,:,:])
                d=np.linalg.norm(delta@cell,axis=-1)
                rr,cc=linear_sum_assignment(d**2)
                total=float(np.sum(d[rr,cc]**2))
                if best is None or total<best[0]:best=(total,u)
    if best is None:raise RuntimeError('O WP projection failed')
    return best[1],best[0]

class WyckoffOxygenNetwork:
    """Symmetric physical O sites, always reconstructed from shared TiO6 arms.

    The WP pattern is a discrete LCAO hypothesis. Its representatives are
    NOT independent oxygen optimization parameters; at every SCF iteration they
    are least-square projections of the currently assigned arm centroids.
    """
    def __init__(self,oxygen_template,cell,lcao_centers,seed=0):
        self.template=oxygen_template;self.cell=np.asarray(cell,float)
        self.orbits=[]
        unused=np.arange(len(lcao_centers))
        cost=0.;n=0
        for bi,wp in enumerate(oxygen_template.blocks):
            if len(unused)<len(wp['rot']):
                raise RuntimeError('Not enough oxygen center suggestions')
            choices=np.asarray(lcao_centers)[unused]
            u,_=w594_oxygen_representative(wp,cell,choices,seed_index=str_seed(seed,bi))
            orb=WyckoffOxygenOrbit(wp,cell,u,global_offset=n)
            locations=orb.frac()
            ds=np.linalg.norm(w594_wrap_delta(locations[:,None,:],choices[None,:,:])@cell,axis=-1)
            aa,bb=linear_sum_assignment(ds**2)
            cost+=float(np.sum(ds[aa,bb]**2))
            used=unused[bb]
            unused=unused[~np.isin(unused,used)]
            self.orbits.append(orb)
            n+=orb.count
        if n!=len(lcao_centers):
            raise RuntimeError(f'O Wyckoff multiplicity {n} != {len(lcao_centers)}')
        self.initial_cost=cost

    def copy(self):return copy.deepcopy(self)

    def positions(self):
        return np.concatenate([o.frac() for o in self.orbits],axis=0)

    def fit_centroids(self,cart_centroids,step=.85):
        frac=(np.asarray(cart_centroids)@np.linalg.inv(self.cell))%1.
        index=0
        for o in self.orbits:
            o.fit_centroids(frac[index:index+o.count],step)
            index+=o.count

def w594_group_symmetry_error(coords,cell,group_ops):
    coords=np.asarray(coords,float)%1.
    err=0.
    for R,t in group_ops:
        moved=(coords@R.T+t)%1.
        delta=w594_wrap_delta(moved[:,None,:],coords[None,:,:])@cell
        dist=np.linalg.norm(delta,axis=-1)
        i,j=linear_sum_assignment(dist**2)
        err=max(err,float(np.max(dist[i,j])))
    return err

def w594_lcao_site_candidates(ti_octas,sg,cell,seed,limit=8,max_orbits=0,
                               generated_centers=None,forced_tokens=None):
    """Choose O Wyckoff orbits from a GENERATED oxygen network.

    ``generated_centers`` is an optional oxygen network already produced by
    v59's metal-only LCAO optimizer. It is never the reference database O.
    This gives the Wyckoff branch a matched initial basin for a controlled
    experiment. Without it, use the original unrelaxed LCAO proposal field.
    """
    if generated_centers is None:
        candidates=coherent_lcao_proposal_field(ti_octas,np.empty((0,3),float))
        init=_select_provisional_centers(ti_octas,candidates,1.4)
        centers=(init@np.linalg.inv(cell))%1.
    else:
        centers=np.asarray(generated_centers,dtype=float).reshape(-1,3)%1.
        if len(centers)!=2*len(ti_octas.ti):
            raise ValueError('Seeded oxygen count does not match 2*Ti count')
    candidates_wp=[]
    # Database recovery: use ONLY the exact original O Wyckoff decomposition.
    # Token metadata is read in the separate original-structure phase; no
    # reference O coordinates or optimized free parameters are supplied here.
    if forced_tokens is not None:
        token_iterator=(tuple(int(v) for v in t) for t in forced_tokens)
    else:
        token_iterator=itertools.islice(
            v48_iter_wyckoff(sg,len(centers),max_orbits=max_orbits),limit)
    for token in token_iterator:
        try:
            tmpl=template_from_token(sg,token,torch.device('cpu'))
            oxygen=WyckoffOxygenNetwork(tmpl,cell,centers,seed=str_seed(seed,token))
            candidates_wp.append((oxygen.initial_cost,oxygen,tuple(token)))
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            continue
    if not candidates_wp:
        raise RuntimeError('No compatible O Wyckoff decomposition for this LCAO field')
    candidates_wp.sort(key=lambda v:v[0])
    return candidates_wp

def w594_coherent_scf(octas,oxygen_network,chem,mode_options,steps=6,
                      move_center=.0,seed=0):
    """Fit only independent TiO6 Wyckoff polyhedra; O always SG-equivariant.

    Oxygen sharing is reassigned by original Hungarian LCAO; oxygen sites are
    generated from Wyckoff-projected arm centroids. `move_center` is an optional
    symmetry-independent metal Wyckoff coordinate update (default 0 in probe).
    """
    if steps<1:raise ValueError('steps must be >=1')
    working=octas.copy();net=oxygen_network.copy();best=None
    snapshots=[]
    for cyc in range(steps):
        oxygen_centers=net.positions()@working.cell
        trip,rebuilt,rms,bad=assign_provisional_network(working,oxygen_centers)
        # O WP representatives updated from arm-derived centroids, not O DOFs.
        net.fit_centroids(rebuilt,step=.7)
        oxygen_centers=net.positions()@working.cell
        arms=working.positions().reshape(-1,3)
        delta=minimum_image_delta(oxygen_centers[trip],arms,working.cell,working.inv)
        ti_ctr=np.repeat(working.ti@working.cell,6,axis=0)
        targets=(arms+delta-ti_ctr).reshape(len(working.ti),6,3)
        working.fit(targets,mode_options=mode_options,mix=.56,
                    distortion_max=working.max_strain)
        if move_center>0 and cyc%2==1:
            # Shift ONLY free Ti representative coordinates, via orbit Jacobians.
            # Preserve exact WP maps; the corresponding metal chemistry object
            # must be refreshed after the move.
            w594_update_metal_centers(working,net,trip,mix=move_center)
            chem.ti=torch.tensor(working.ti,dtype=DTYPE)
        O=net.positions()
        tr,_,rms,bad=assign_provisional_network(working,O@working.cell)
        # The metal WP representative may move; chemistry must see NEW Ti sites.
        chem.ti=torch.tensor(working.ti,dtype=DTYPE)
        with torch.no_grad():
            details=chem.score(torch.tensor(O,dtype=DTYPE),details=True)
        geometry_err=max(w594_group_symmetry_error(working.ti,working.cell,working.group_ops),
                         w594_group_symmetry_error(O,working.cell,working.group_ops),
                         working.symmetry_error())
        if geometry_err>1e-4:
            raise RuntimeError(f'Wyckoff symmetry violation: {geometry_err:.6g} A')
        cand=dict(oct=working.copy(),oxygen=O.copy(),trip=tr.copy(),
                  rms=float(rms),bad_triplets=int(bad),details=details,
                  oxygen_wyckoff=net.copy(),
                  wyckoff_symmetry_error=float(geometry_err),
                  independent_ti_orbits=len(working.orbits),
                  independent_o_orbits=len(net.orbits),
                  pass_chemistry=False,cycle=cyc+1,cycles=cyc+1,
                  compatible_symops=len(working.group_ops),
                  ti_symops=len(working.group_ops),symmetry_max_rms=0.)
        cand['pass_chemistry']=bool(bad==0 and rms<.12 and
                np.all(details['hardTi']==6) and np.all(details['hardO']==3)
                and np.all(details['goodTi']) and np.all(details['goodO']))
        snapshots.append((continuous_candidate_objective(cand),cand))
        if best is None or snapshots[-1][0]<best[0]:best=snapshots[-1]
    answer=best[1]
    answer['history']=[(float(z[1]['rms']),int(z[1]['bad_triplets'])) for z in snapshots]
    return answer

def w594_update_metal_centers(octas,o_net,trip,mix=.06):
    arms=octas.positions().reshape(-1,3)
    oxygen=o_net.positions()@octas.cell
    delta=minimum_image_delta(oxygen[np.asarray(trip,dtype=int)],arms,
                              octas.cell,octas.inv)
    shifts=np.mean(delta.reshape(len(octas.ti),6,3),axis=1)
    start=0
    for orb in octas.orbits:
        if orb.dof:
            transported=[shifts[start+i]@np.linalg.inv(orb.maps[i])
                         for i in range(orb.count)]
            desired=np.mean(transported,axis=0)@octas.inv
            change=np.linalg.lstsq(orb.A,desired,rcond=None)[0]
            orb.u=(orb.u+mix*np.clip(change,-.03,.03))%1.
        start+=orb.count
    octas._rebuild()

def w594_orbit_feedback(result,chem,steps=12,rounds=1,strain_limit=.11):
    """Autograd on ONE coherent transform per independent Ti Wyckoff orbit.

    The O orbit coordinates are projected means of assigned arms, with no
    independent O Parameter tensors.  All Ti and O symmetry images are created
    by the chosen Wyckoff operators on EVERY forward evaluation.
    """
    if steps<=0 or rounds<=0:return result
    working=result['oct'].copy()
    oxygen_net=result['oxygen_wyckoff'].copy()
    cell=torch.tensor(working.cell,dtype=DTYPE)
    inv=torch.tensor(working.inv,dtype=DTYPE)
    one=torch.eye(3,dtype=DTYPE)
    best=result.copy()
    best_value=continuous_candidate_objective(result)
    iterations=0
    for rd in range(rounds):
        initial_arms=working.positions().reshape(-1,3)
        initial_centers=oxygen_net.positions()@working.cell
        trip,_,_,_=assign_provisional_network(working,initial_centers)
        chem.ti=torch.tensor(working.ti,dtype=DTYPE)
        group=torch.tensor(trip,dtype=torch.long)
        close=minimum_image_delta(initial_arms,initial_centers[trip],working.cell,working.inv)
        offsets=np.rint((initial_arms-close-initial_centers[trip])@working.inv).astype(int)@working.cell
        offs=torch.tensor(offsets,dtype=DTYPE)
        bases=[];params=[];basis=[];q0=[];maps=[];centers=[];counts=[]
        for o in working.orbits:
            bases.append(torch.tensor(o.src(),dtype=DTYPE))
            basis.append(torch.tensor(o.B,dtype=DTYPE))
            q0.append(torch.tensor(o.B.T@o.F.reshape(-1),dtype=DTYPE))
            maps.append(torch.tensor(o.maps,dtype=DTYPE))
            centers.append(torch.tensor(o.frac_sites()@working.cell,dtype=DTYPE))
            counts.append(o.count)
            params.append(torch.nn.Parameter(torch.zeros(o.B.shape[1],dtype=DTYPE)))
        optimizer=torch.optim.Adam(params,lr=.045)
        nO=len(oxygen_net.positions())
        o_projection=[]
        start=0
        for o in oxygen_net.orbits:
            o_loc=torch.tensor(o.frac(),dtype=DTYPE)
            J=np.concatenate([R@o.A for R in o.rotfrac],axis=0)
            pinv=torch.tensor(np.linalg.pinv(J),dtype=DTYPE) if o.dof else None
            o_projection.append((o_loc,pinv,torch.tensor(J,dtype=DTYPE),o.count))
            start+=o.count
        round_best=None
        for step in range(steps):
            optimizer.zero_grad()
            arms=[];F_list=[];strain_penalty=0.
            for i in range(len(bases)):
                # B spans EXACT site-compatible coherent transformations.
                vec=q0[i]+.042*torch.tanh(params[i])
                F=(basis[i]@vec).reshape(3,3)
                F_list.append(F)
                local=bases[i]@F
                transformed=torch.einsum('vk,okj->ovj',local,maps[i])+centers[i][:,None,:]
                arms.append(transformed)
                sv=torch.linalg.svdvals(F)
                strain_penalty=strain_penalty+(sv-1).square().sum()+(
                    torch.relu(sv-(1+strain_limit)).square().sum()+
                    torch.relu((1-strain_limit)-sv).square().sum())*100
            allarms=torch.cat(arms,dim=0).reshape(-1,3)
            # Frozen periodic images within one gradient block; sharing
            # reassignment occurs at the next SCF/feedback round.
            unwrapped=allarms-offs
            totals=torch.zeros((nO,3),dtype=DTYPE).index_add(0,group,unwrapped)
            predicted=totals/3.
            raw_fraction=predicted@inv
            projected=[];idx=0;new_u=[]
            for base,pinv,J,count in o_projection:
                q=raw_fraction[idx:idx+count]
                closest=base+q-base-torch.round(q-base).detach()
                du=pinv@(closest-base).reshape(-1) if pinv is not None else None
                if du is None:
                    point=base
                else:
                    # The Wyckoff Jacobian imposes the exact O site symmetry.
                    # Clamping only prevents an SCF step jumping a cell image.
                    du=torch.clamp(du,-.10,.10)
                    point=base+(J@du).reshape(count,3)
                projected.append(point)
                new_u.append(du)
                idx+=count
            oxygen_frac=torch.remainder(torch.cat(projected,dim=0),1.)
            oxygen_cart=oxygen_frac@cell
            shared=oxygen_cart[group]+offs
            co=((allarms-shared)**2).sum(dim=-1).mean()
            chemical,*_=chem.score(oxygen_frac,components=True)
            loss=chemical+4.*co+.025*strain_penalty
            if not torch.isfinite(loss):break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,max_norm=4.)
            optimizer.step()
            iterations+=1
            if (step+1)%6==0 or step==steps-1:
                trial=working.copy()
                valid=True
                for orb,F in zip(trial.orbits,F_list):
                    M=F.detach().numpy()
                    D,R=w594_polardecompose(M)
                    if D is None or np.max(np.abs(np.linalg.eigvalsh(D)-1))>strain_limit+1e-5:
                        valid=False;break
                    orb.F=M
                if not valid:continue
                trial._rebuild()
                newnet=oxygen_net.copy()
                for o,du in zip(newnet.orbits,new_u):
                    if du is not None:o.u=(o.u+du.detach().numpy())%1.
                trialO=newnet.positions()
                if trial.symmetry_error()>2e-5:continue
                symerr=max(w594_group_symmetry_error(trial.ti,trial.cell,trial.group_ops),
                           w594_group_symmetry_error(trialO,trial.cell,trial.group_ops))
                if symerr>1e-4:continue
                newtrip,_,rms,bad=assign_provisional_network(trial,trialO@trial.cell)
                chem.ti=torch.tensor(trial.ti,dtype=DTYPE)
                with torch.no_grad():details=chem.score(torch.tensor(trialO,dtype=DTYPE),details=True)
                rec=dict(result,oct=trial,oxygen=trialO,trip=newtrip,rms=float(rms),
                         bad_triplets=int(bad),details=details,
                         oxygen_wyckoff=newnet,
                         wyckoff_symmetry_error=float(symerr),
                         chem_feedback_steps=iterations)
                value=continuous_candidate_objective(rec)
                if round_best is None or value<round_best[0]:round_best=(value,rec)
                if value<best_value:best_value=value;best=rec
        if round_best is None:break
        working=round_best[1]['oct'].copy()
        oxygen_net=round_best[1]['oxygen_wyckoff'].copy()
    best['chem_feedback_steps']=iterations
    return best

def w594_unoptimized_probe_snapshot(octas,oxygen_net,chem,stage='initial-LCAO'):
    """Accept a finite, target-SG-exact LCAO proposal even if SCF fails.

    This is NOT a chemical pass: the quality audit and top-N rank remain
    diagnostic. No atomic coordinate is changed or symmetry relaxed.
    """
    o=octas.copy()
    net=oxygen_net.copy()
    oxy=net.positions()
    if not np.all(np.isfinite(oxy)):
        raise FloatingPointError('Nonfinite O-Wyckoff coordinates in fallback')
    error=max(w594_group_symmetry_error(o.ti,o.cell,o.group_ops),
              w594_group_symmetry_error(oxy,o.cell,o.group_ops),
              o.symmetry_error())
    if not np.isfinite(error) or error>1e-4:
        raise RuntimeError(f'Fallback rejected: full SG error={error:.6g} A')
    trip,_,rms,bad=assign_provisional_network(o,oxy@o.cell)
    chem.ti=torch.tensor(o.ti,dtype=DTYPE)
    with torch.no_grad():
        details=chem.score(torch.tensor(oxy,dtype=DTYPE),details=True)
    if not np.isfinite(rms):
        raise FloatingPointError('Nonfinite sharing RMS in LCAO snapshot')
    return dict(oct=o,oxygen=oxy.copy(),trip=trip.copy(),rms=float(rms),
                bad_triplets=int(bad),details=details,
                oxygen_wyckoff=net,
                wyckoff_symmetry_error=float(error),
                independent_ti_orbits=len(o.orbits),
                independent_o_orbits=len(net.orbits),
                pass_chemistry=False,cycle=0,cycles=0,
                compatible_symops=len(o.group_ops),
                ti_symops=len(o.group_ops),symmetry_max_rms=0.,
                probe_stage=stage,history=[])



def w594_parse(argv=None):
    a=e593_parse(argv)
    # e593_parse cannot accept extra flags; all new options below are carried
    # in a second preparse and stripped before invoking the inherited parser.
    return a

_w594_previous_topology=e593_topology

def e593_topology(item):
    ans=item['ans'];oc=ans['oct']
    if not isinstance(oc,WyckoffOctahedra):
        return _w594_previous_topology(item)
    framework=(item.get('sg'),tuple(item.get('wp',())),
               item.get('metal_cycle'),item.get('basin'))
    pose=tuple((int(o.mode),tuple(np.rint(o.F.reshape(-1)/.12).astype(int)))
               for o in oc.orbits)
    oxygen_pattern=tuple(item.get('oxygen_token',()))
    return (framework,oxygen_pattern,global_basin_signature(ans),pose)

def w594_parse(argv=None):
    extra=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    extra.add_argument('--oxygen-wp-scan',type=int,default=8,
                       help='Number of symmetry-exact O Wyckoff patterns tried for each LCAO entrance')
    extra.add_argument('--o-hypotheses',type=int,default=2,
                       help='Different oxygen Wyckoff patterns retained per TiO6 entrance')
    extra.add_argument('--max-oxygen-orbits',type=int,default=0,
                       help='0: unlimited oxygen WP orbits (supports P1)')
    extra.add_argument('--site-tries',type=int,default=12,
                       help='Orientation hypotheses per Ti Wyckoff site stabilizer')
    extra.add_argument('--center-scf-mix',type=float,default=.02,
                       help='Bounded symmetry-independent Ti Wyckoff-center update in SCF (0 disables)')
    options,remaining=extra.parse_known_args(argv)
    a=e593_parse(remaining)
    for name,value in vars(options).items():setattr(a,name,value)
    if a.oxygen_wp_scan<1 or a.o_hypotheses<1 or a.max_oxygen_orbits<0:
        raise ValueError('oxygen WP search bounds must be positive')
    if a.site_tries<1:raise ValueError('--site-tries must be positive')
    if not 0<=a.center_scf_mix<=.2:raise ValueError('--center-scf-mix must be in [0,.2]')
    if not 0<=a.center_scf_mix<=.1:raise ValueError('--center-scf-mix must be in [0,.1]')
    return a



def w594_sharing_graph_audit(octas,oxygen,trip):
    """Diagnostic ONLY: is discrete TiO6-arm -> O sharing group-equivariant?

    Exact Ti/O Wyckoff coordinates do not alone imply the LCAO assignment is
    equivariant. This audit detects that distinct issue without discarding a
    chemically interesting, symmetry-exact final structure.
    """
    n=len(octas.ti)
    arms=octas.positions()
    local=arms-octas.ti[:,None,:]@octas.cell
    oxy=np.asarray(oxygen,float)%1.
    trip=np.asarray(trip,int)
    total=0;bad=0;max_arm_err=0.
    for R,t in octas.group_ops:
        warped=(octas.ti@R.T+t)%1.
        d=np.linalg.norm(w594_wrap_delta(warped[:,None,:],octas.ti[None,:,:])@octas.cell,axis=-1)
        aa,bb=linear_sum_assignment(d**2)
        ti_perm=np.empty(n,dtype=int);ti_perm[aa]=bb
        movedO=(oxy@R.T+t)%1.
        do=np.linalg.norm(w594_wrap_delta(movedO[:,None,:],oxy[None,:,:])@octas.cell,axis=-1)
        ao,bo=linear_sum_assignment(do**2)
        o_perm=np.empty(len(oxy),dtype=int);o_perm[ao]=bo
        M=w594_cart_action(octas.cell,R)
        for i in range(n):
            j=ti_perm[i]
            dist=np.linalg.norm((local[i]@M)[:,None,:]-local[j][None,:,:],axis=-1)
            ia,ib=linear_sum_assignment(dist**2)
            max_arm_err=max(max_arm_err,float(np.max(dist[ia,ib])))
            for k,l in zip(ia,ib):
                total+=1
                if trip[j*6+l]!=o_perm[trip[i*6+k]]:bad+=1
    return int(bad),int(total),float(max_arm_err)

# ============================================================================
# JULIETTE v59.5: all-in-one metal-only oxygen recovery from ASE TiO2 database
# Embedded v59.4b: lattice, LCAO, coherent WP polyhedra, and chemistry kernels.
# Embedded independent periodic geometry audit: reference-vs-regenerated.
# The only external data are tio2.db and the learned chemistry CIF directory.
# ============================================================================

class AuditError(RuntimeError):
    pass

def quantile(values, q):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float('nan')


def image_shifts(cell, reach=4.5):
    """Conservative triclinic periodic image enumeration for distances < reach.

    Includes a margin for the fractional displacement between primitive sites.
    The cutoff does not depend on the visual orientation of the cell.
    """
    inv = np.linalg.inv(cell)
    heights = 1.0 / np.linalg.norm(inv, axis=0)
    if np.min(heights) < 0.5:
        raise AuditError(f'Nearly degenerate cell: minimum reciprocal height={np.min(heights):.3g} A')
    widths = np.ceil(reach / heights).astype(int) + 1
    if np.max(widths) > 8:
        raise AuditError(f'Cell too thin for bounded image enumeration: widths={widths.tolist()}')
    return np.asarray(list(itertools.product(*(range(-k, k+1) for k in widths))), dtype=float)

def _validate(ti, oxygen, cell):
    ti = np.asarray(ti, float).reshape(-1,3)
    oxygen = np.asarray(oxygen, float).reshape(-1,3)
    cell = np.asarray(cell, float).reshape(3,3)
    if len(ti) < 1 or len(oxygen) != 2*len(ti):
        raise AuditError(f'Expected Ti_n O_2n, got Ti={len(ti)} O={len(oxygen)}')
    if not (np.isfinite(ti).all() and np.isfinite(oxygen).all() and np.isfinite(cell).all()):
        raise AuditError('Nonfinite coordinates or cell')
    if abs(np.linalg.det(cell)) < 1.e-6:
        raise AuditError('Singular unit cell')
    inv = np.linalg.inv(cell)
    ft = (ti@inv)%1.
    fo = (oxygen@inv)%1.
    return ti,oxygen,cell,ft,fo

def periodic_geometry(ti, oxygen, cell, source='', name='', reach=4.5):
    """Return site-resolved geometry; no binary chemical judgement.

    O--O includes periodic self-images but excludes each atom's own zero image.
    Ti--O shells count distinct periodic images, as crystallography requires.
    """
    ti,oxygen,cell,ft,fo = _validate(ti,oxygen,cell)
    shifts=image_shifts(cell,reach=reach)
    zero = np.flatnonzero(np.all(shifts==0.,axis=1))[0]
    nt=len(ti); no=len(oxygen)
    # Cross-species nearest physical images, not just unique atom IDs.
    ti_o_shells=[];ti_o_bonds=[];ti_o_seventh=[]
    for i in range(nt):
        vec=((fo[:,None,:]-ft[i]+shifts[None,:,:])@cell).reshape(-1,3)
        ds=np.linalg.norm(vec,axis=-1)
        ix=np.argpartition(ds,6)[:7]
        ix=ix[np.argsort(ds[ix])]
        ti_o_shells.append(vec[ix[:6]])
        ti_o_bonds.append(ds[ix[:6]])
        ti_o_seventh.append(float(ds[ix[6]]))
    oxygen_ti_third=[];oxygen_ti_fourth=[]
    for i in range(no):
        ds=np.linalg.norm(((ft[:,None,:]-fo[i]+shifts[None,:,:])@cell).reshape(-1,3),axis=-1)
        ix=np.partition(ds,3)[:4]
        ix.sort()
        oxygen_ti_third.append(float(ix[2]))
        oxygen_ti_fourth.append(float(ix[3]))
    # Enumerate physically unique O--O contacts within the supplied radius.
    # Distinct periodic images are distinct contacts; unordered i-j pairs are
    # counted once, and self images use only one of s,-s.
    oo_dist=[];oo_nearest=np.full(no,np.inf)
    positive_shift=((shifts[:,0]>0)|((shifts[:,0]==0)&(shifts[:,1]>0))|
                    ((shifts[:,0]==0)&(shifts[:,1]==0)&(shifts[:,2]>0)))
    for i in range(no):
        dx=((fo[:,None,:]-fo[i]+shifts[None,:,:])@cell)
        dd=np.linalg.norm(dx,axis=-1)
        dd[i,zero]=np.inf
        oo_nearest[i]=float(dd.min())
        # Non-self i-j contacts only once, but every periodic image counts.
        if i+1<no:
            d=dd[i+1:]
            oo_dist.extend(d[d<reach].ravel())
        self_contact=dd[i,positive_shift]
        oo_dist.extend(self_contact[self_contact<reach].ravel())
    oo_dist=np.asarray(oo_dist,float)
    shells=np.asarray(ti_o_shells,float)
    bonds=np.asarray(ti_o_bonds,float)
    seventh=np.asarray(ti_o_seventh,float)
    three=np.asarray(oxygen_ti_third,float)
    fourth=np.asarray(oxygen_ti_fourth,float)
    # Rotation- and permutation-invariant descriptor of actual TiO6 shapes.
    jj,kk=np.triu_indices(6,1)
    pair=np.linalg.norm(shells[:,:,None,:]-shells[:,None,:,:],axis=-1)
    signatures=np.concatenate([np.sort(bonds,axis=1),np.sort(pair[:,jj,kk],axis=1)],axis=1)
    row={'source':source,'name':name,'n_ti':nt,'n_o':no,
         'volume_per_ti_A3':float(abs(np.linalg.det(cell))/nt),
         'min_oo_A':float(np.min(oo_nearest)),
         'oo_nn_p05_A':quantile(oo_nearest,.05),'oo_nn_median_A':quantile(oo_nearest,.5),
         'min_tio_A':float(bonds.min()),'tio6_max_A':float(bonds.max()),
         'tio6_range_max_A':float(np.max(bonds[:,-1]-bonds[:,0])),
         'tio6_mean_A':float(bonds.mean()),
         'tio_seventh_min_A':float(seventh.min()),
         'oti3_max_A':float(three.max()),'oti4_min_A':float(fourth.min())}
    return {'row':row,'_ti_o_shells':shells,'_signatures':signatures,
            '_ti_o_bonds':bonds,'_seventh':seventh,'_oti3':three,'_oti4':fourth,
            '_oo_nearest':oo_nearest,'_oo_contacts':oo_dist,
            '_ft':ft,'_fo':fo,'_cell':cell}

_VERTEX_PERMUTATIONS = np.asarray(list(itertools.permutations(range(6))), dtype=int)

def matched_affine_strain(reference, target):
    """Find best proper coherent affine fit, minimizing strain and residual.

    All six-vertex assignments are tested in a vectorized 720-way batch. This
    avoids spuriously large 'strain' for near-symmetric octahedra when a
    simple nearest-neighbor matching chooses another equivalent permutation.
    The estimate is descriptive; without the generator's saved F it is not a
    direct certificate that its internal 11% bound was respected.
    """
    src=np.asarray(reference,float).reshape(6,3)
    dst=np.asarray(target,float).reshape(6,3)
    aligned=dst[_VERTEX_PERMUTATIONS]  # (720,6,3)
    pinv=np.linalg.pinv(src)
    Fs=np.einsum('ij,pjk->pik',pinv,aligned)
    determinants=np.linalg.det(Fs)
    acceptable=determinants>1e-8
    if not acceptable.any():return float('nan'),float('nan')
    Fs=Fs[acceptable]
    aligned=aligned[acceptable]
    singular=np.linalg.svd(Fs,compute_uv=False)
    strains=np.max(np.abs(singular-1),axis=1)
    residuals=np.sqrt(np.mean(np.sum((src[None,:,:]@Fs-aligned)**2,axis=-1),axis=1))
    scores=residuals+.20*strains
    ix=int(np.argmin(scores))
    return float(strains[ix]),float(residuals[ix])

def pick_motifs(origins, max_motifs=128, distinct_rms=.015):
    """Select diverse actual original TiO6 geometries as reference motifs."""
    records=[]
    for r in origins:
        for sig,vec in zip(r['_signatures'],r['_ti_o_shells']):
            records.append((sig,vec,r['row']['name']))
    if not records:return []
    # Greedy diversity sampling, independent of the oxygen coordinates in the
    # generated crystal. At most max_motifs learned geometries are retained.
    rng=np.random.default_rng(441)
    rng.shuffle(records)
    selected=[]
    for sig,vec,name in records:
        if selected and min(np.sqrt(np.mean((sig-s)**2)) for s,_,_ in selected)<distinct_rms:
            continue
        selected.append((sig,vec,name))
        if len(selected)>=max_motifs:break
    return selected

def evaluate_geometry(record, oo_guard, tio_lower, tio_cn_upper, oti_cn_upper, motifs, origin=False, model_oo_guard=None):
    row=dict(record['row'])
    contacts=record['_oo_contacts']
    dif=np.maximum(0.,oo_guard-contacts)
    row['reference_oo_guard_A']=float(oo_guard)
    row['oo_pairs_below_guard']=int((contacts<oo_guard).sum())
    row['oo_repulsion_sum_A2']=float(np.sum(dif**2))
    if model_oo_guard is not None:
        model_dif=np.maximum(0.,model_oo_guard-contacts)
        row['generator_model_oo_guard_A']=float(model_oo_guard)
        row['generator_model_oo_pairs_below']=int((contacts<model_oo_guard).sum())
        row['generator_model_oo_repulsion_A2']=float(np.sum(model_dif**2))
    row['oo_nn_below_guard_sites']=int((record['_oo_nearest']<oo_guard).sum())
    row['ti_o_bonds_too_short']=int((record['_ti_o_bonds']<tio_lower).sum())
    row['ti_sites_cn6_within_reference_cutoff']=int(((record['_ti_o_bonds'][:,-1]<=tio_cn_upper) & (record['_seventh']>tio_cn_upper)).sum())
    row['ti_sites_total']=row['n_ti']
    row['o_sites_cn3_within_reference_cutoff']=int(((record['_oti3']<=oti_cn_upper)&(record['_oti4']>oti_cn_upper)).sum())
    row['reference_oti_cn_upper_A']=float(oti_cn_upper)
    row['reference_tio_lower_A']=float(tio_lower)
    row['reference_tio_cn_upper_A']=float(tio_cn_upper)
    deviations=[];strains=[];affine=[]
    # Origin self-matches are not evidence of exceptional chemical accuracy:
    # exclude motifs from the same source when measuring original shape drift.
    for sig,shell in zip(record['_signatures'],record['_ti_o_shells']):
        pool=[x for x in motifs if not (origin and x[2]==row['name'])]
        if not pool:pool=motifs if not origin else []
        if not pool:
            deviations.append(float('nan'));strains.append(float('nan'));affine.append(float('nan'))
            continue
        best=sorted(pool,key=lambda x:np.mean((sig-x[0])**2))[:3]
        shape_rms=float(np.sqrt(np.mean((sig-best[0][0])**2)))
        estimates=[matched_affine_strain(v,shell) for _,v,_ in best]
        estimates=[v for v in estimates if np.isfinite(v[0])]
        if estimates:
            strain,rms=min(estimates,key=lambda x:x[1]+0.2*x[0])
        else:strain=rms=float('nan')
        deviations.append(shape_rms);strains.append(strain);affine.append(rms)
    row['shape_distortion_rms_max_A']=quantile(deviations,1)
    row['shape_distortion_rms_med_A']=quantile(deviations,.5)
    row['estimated_coherent_eigenstrain_max']=quantile(strains,1)
    row['estimated_coherent_eigenstrain_med']=quantile(strains,.5)
    row['affine_fit_residual_max_A']=quantile(affine,1)
    row['apparent_eigenstrain_over_11pct']=int(np.isfinite(row['estimated_coherent_eigenstrain_max']) and row['estimated_coherent_eigenstrain_max']>.11)
    return row,deviations,strains,affine

def measured_full_sg_from_geometry(record, symprec=.05):
    """spglib label from final Ti+O structure, never used in generation."""
    try:
        import spglib
    except ImportError:
        return None
    cell=record['_cell']
    f=np.concatenate((record['_ft'],record['_fo']),axis=0)
    z=np.concatenate((np.full(len(record['_ft']),22),np.full(len(record['_fo']),8)))
    ds=spglib.get_symmetry_dataset((cell,f,z),symprec=symprec)
    return int(ds.number) if ds is not None else None

def synthetic_self_test():
    """True crystallographic rutile-style example; deliberately degraded copy."""
    a=4.6;c=2.96;cell=np.diag([a,a,c]);u=.305
    ti_f=np.array([[0,0,0],[.5,.5,.5]],float)
    o_f=np.array([[u,u,0],[-u,-u,0],[.5+u,.5-u,.5],[.5-u,.5+u,.5]])%1.
    t=ti_f@cell;o=o_f@cell
    reference=periodic_geometry(t,o,cell,source='origin',name='synthetic_rutile')
    assert len(reference['_ti_o_shells'])==2
    assert abs(reference['row']['min_oo_A'])>1.7
    exact=periodic_geometry(t,o,cell,source='generated',name='unaltered')
    # Shift an oxygen site near another independent site, creates true OO clash.
    bad=o.copy();bad[1]=bad[0]+np.array([.03,0.,0.])
    collided=periodic_geometry(t,bad,cell,source='generated',name='artificial_overlap')
    assert collided['row']['min_oo_A']<.031
    guard=reference['row']['min_oo_A']-.12
    assert np.count_nonzero(collided['_oo_contacts']<guard)>0
    # This is the precise F-mixing defect in v59.4b (0.11 bound not preserved).
    from scipy.spatial.transform import Rotation
    F0=np.eye(3);F1=Rotation.from_euler('z',180,degrees=True).as_matrix()
    Fmix=.44*F0+.56*F1
    assert np.min(np.linalg.svd(Fmix,compute_uv=False))<.89
    # Reference motif must return a zero-distance match to exact copy.
    motifs=pick_motifs([reference],max_motifs=10)
    exactrow,_,_,_=evaluate_geometry(exact,guard,1.7,2.3,2.3,motifs)
    badrow,_,_,_=evaluate_geometry(collided,guard,1.7,2.3,2.3,motifs)
    assert exactrow['oo_pairs_below_guard']==0
    assert badrow['oo_pairs_below_guard']>0
    assert exactrow['shape_distortion_rms_max_A']<1e-10
    assert badrow['shape_distortion_rms_max_A']>0.05
    print('PASS synthetic TiO2 self-test: periodic O--O collision detected, intact structure clear')
    print('PASS independent diagnosis: direct F mixing violates 11% eigenstrain cap')
    print('These synthetic controls are NOT measurements on tio2.db or generated CIFs.')

@dataclass(frozen=True)
class MetalOnly:
    """The *entire* permissible source object passed to generation."""
    cell: np.ndarray
    ti_frac: np.ndarray
    row_id: int

def strip_oxygen(atoms, row_id):
    """Return only the original cell + Ti coordinates, never an O array."""
    symbols=np.asarray(atoms.get_chemical_symbols())
    if set(symbols)!={'Ti','O'}:
        raise ValueError('Not a binary Ti/O structure')
    n_ti=int(np.sum(symbols=='Ti'));n_o=int(np.sum(symbols=='O'))
    if n_o!=2*n_ti:
        raise ValueError(f'Expected TiO2 stoichiometry, got Ti={n_ti} O={n_o}')
    if not np.all(atoms.get_pbc()):
        raise ValueError('Expected fully periodic cell')
    cell=np.asarray(atoms.cell.array,dtype=float)
    if abs(np.linalg.det(cell))<1e-6:raise ValueError('Degenerate cell')
    ti=np.asarray(atoms.get_scaled_positions(wrap=True),float)[symbols=='Ti']%1.
    return MetalOnly(cell.copy(),ti.copy(),int(row_id))

def chemistry_metrics(j,lib,bounds,guard,ti,cell,oxygen):
    import torch
    ti=np.asarray(ti,float)%1.;o=np.asarray(oxygen,float)%1.
    with contextlib.redirect_stdout(io.StringIO()):
        chem=j.Chemistry(ti,cell,lib,bounds,guard,accept=.10,
                         hierarchy='ti-first',calibration=None,target_o_cn=True)
    with torch.no_grad():
        d=chem.score(torch.tensor(o,dtype=j.DTYPE),details=True)
    n=len(ti)
    return dict(n_ti=n,n_o=len(o),Ti_loss_min=float(np.min(d['localTi'])),
                Ti_loss_avg=float(np.mean(d['localTi'])),
                Ti_loss_max=float(np.max(d['localTi'])),
                O_loss_min=float(np.min(d['localO'])),
                O_loss_avg=float(np.mean(d['localO'])),
                O_loss_max=float(np.max(d['localO'])),
                good_ti=int(np.sum(d['goodTi'])),good_o=int(np.sum(d['goodO'])),
                Ti_CN6=int(np.sum(np.asarray(d['hardTi'])==6)),
                O_CN3=int(np.sum(np.asarray(d['hardO'])==3)),
                collision=float(d['collision']),
                chemistry_ok=int(np.all(d['goodTi']) and np.all(d['goodO'])
                                 and np.all(d['hardTi']==6)
                                 and np.all(d['hardO']==3)
                                 and d['collision']<=1e-8),
                loss_total=float(d['energy']))

def get_audit_record(audit,metal,oxygen,name,source):
    return audit.periodic_geometry(metal.ti_frac@metal.cell,
                                   np.asarray(oxygen,float)@metal.cell,
                                   metal.cell,name=name,source=source)

def _wrap_difference(a,b):
    return (np.asarray(a)-np.asarray(b)+.5)%1.-.5

def fit_wp_orbit(j,wp,observed,cell):
    """Fit a PyXtal Wyckoff orbit to metal-only observed sites.

    Works with free coordinates and special sites. Never reads O geometry.
    """
    observed=np.asarray(observed,float)%1.
    d=int(wp.get_dof())
    A,b=j.affine_map(wp,d)
    rotations=np.asarray([op.rotation_matrix for op in wp.ops],float)
    trans=np.asarray([op.translation_vector for op in wp.ops],float)
    if len(observed)!=len(rotations):return None
    shifts=np.array(list(__import__('itertools').product((-1,0,1),repeat=3)),float)
    trial=[]
    if d==0:
        trial=[np.zeros(0)]
    else:
        # Any WP member may be the anchor, and its coordinate map may be
        # singular. Solve the full affine map by least squares, not inversion.
        for x in observed:
            for R,t in zip(rotations,trans):
                J=R@A;base=R@b+t
                for shift in shifts:
                    u=np.linalg.lstsq(J,x+shift-base,rcond=None)[0]%1.
                    if np.all(np.isfinite(u)):
                        trial.append(u)
    best=None
    for u in trial:
        rep=(A@u+b)%1.
        positions=(np.einsum('oij,j->oi',rotations,rep)+trans)%1.
        delta=_wrap_difference(positions[:,None,:],observed[None,:,:])@cell
        dist=np.linalg.norm(delta,axis=-1)
        rr,cc=linear_sum_assignment(dist**2)
        err=float(np.max(dist[rr,cc]))
        if best is None or err<best[0]:
            best=(err,u.copy(),positions.copy())
            if err<1e-7:break
    return best

def identify_metal_wyckoffs(j,metal,sg,max_tol=.03):
    """Discover Ti Wyckoff token + free WP params *from Ti only*.

    Requires a cell in PyXtal's conventional setting. If no exact mapping is
    possible, report the mismatch instead of inventing a compatible structure.
    """
    from pyxtal.symmetry import Group
    group=Group(int(sg));ops=j.w594_group_operations(group)
    ti=np.asarray(metal.ti_frac,float)%1.
    unused=set(range(len(ti)))
    blocks=[];worst=0.
    while unused:
        i=min(unused);p=ti[i]
        image=np.array([(p@R.T+t)%1. for R,t in ops])
        ds=np.linalg.norm(_wrap_difference(image[:,None,:],ti[None,:,:])@metal.cell,axis=-1)
        orbit_indices=sorted(set(int(np.argmin(row)) for row in ds))
        if np.max(np.min(ds,axis=1))>max_tol:
            raise ValueError('Ti-only framework is inconsistent with target SG operations')
        if not set(orbit_indices)<=unused:
            raise ValueError('Overlapping Ti Wyckoff orbit partition')
        observed=ti[orbit_indices]
        choices=[]
        for wi in range(len(group)):
            wp=group[wi]
            if int(wp.multiplicity)!=len(orbit_indices):continue
            fit=fit_wp_orbit(j,wp,observed,metal.cell)
            if fit is not None and fit[0]<=max_tol:
                choices.append((fit[0],int(wp.get_dof()),wi,fit[1],fit[2]))
        if not choices:
            raise ValueError(f'No PyXtal WP describes Ti-only orbit of size {len(observed)}')
        choices.sort(key=lambda z:(z[0],z[1],z[2]))
        err,dof,wi,u,sites=choices[0]
        worst=max(worst,err)
        blocks.append((wi,u,sites))
        unused-=set(orbit_indices)
    token=tuple(int(wi) for wi,_,_ in blocks)
    templ=j.template_from_token(int(sg),token,__import__('torch').device('cpu'))
    perm_ti=np.concatenate([sites for _,_,sites in blocks],axis=0)%1.
    delta=np.linalg.norm(_wrap_difference(perm_ti[:,None,:],ti[None,:,:])@metal.cell,axis=-1)
    rr,cc=linear_sum_assignment(delta**2)
    if np.max(delta[rr,cc])>max_tol:
        raise ValueError('Expanded Ti Wyckoff order does not match observed Ti')
    n_lattice=len(templ.lattice_keys)
    raw=np.zeros(n_lattice+sum(int(b['dof']) for b in templ.blocks))
    start=n_lattice
    for _,u,_ in blocks:
        u=np.clip(np.asarray(u,float),1e-10,1-1e-10)
        raw[start:start+len(u)]=np.log(u/(1-u))
        start+=len(u)
    return templ,raw,perm_ti,ops,dict(sg=int(sg),wp_token=token,
                                      max_wp_ti_error_A=float(worst),
                                      independent_ti_orbits=len(blocks))

def audit_original_oxygen_wyckoff(j,atoms,metal,symprec=.04,scan_limit=12):
    """ORIGINAL-only diagnostic: can original O occupy the scanned O WP token?

    This function is called only in the baseline audit phase, before oxygen
    deletion, and its output is never used to initialize generation.
    """
    import spglib
    z=np.asarray(atoms.get_atomic_numbers(),dtype=int)
    f=np.asarray(atoms.get_scaled_positions(wrap=True),float)%1.
    c=np.asarray(atoms.cell.array,float)
    full=spglib.standardize_cell((c,f,z),to_primitive=False,
                                 no_idealize=True,symprec=symprec)
    if full is None:raise ValueError('Full TiO2 standardization failed')
    cell,frac,number=full
    cell=np.asarray(cell,float)
    frac=np.asarray(frac,float)%1.
    number=np.asarray(number)
    ti=frac[number==22]
    oxy=frac[number==8]
    standardized,sg=ti_only_standardized(j,metal,symprec=symprec)
    # Full-structure and metal-only conventionalization may choose different
    # settings. Refuse to compare their WP labels if they are not aligned.
    cell_err=float(np.max(np.abs(cell-standardized.cell)))
    dist=np.linalg.norm(_wrap_difference(ti[:,None,:],standardized.ti_frac[None,:,:])@cell,axis=-1)
    ir,ic=linear_sum_assignment(dist**2)
    ti_err=float(np.max(dist[ir,ic])) if len(ir)==len(ti)==len(standardized.ti_frac) else float('inf')
    if cell_err>.002 or ti_err>.025:
        return dict(original_O_wp_audit='not-comparable-conventional-settings',
                    original_O_wp_cell_error_A=cell_err,
                    original_O_wp_ti_match_error_A=ti_err)
    otemplate,raw,osites,ops,meta=identify_metal_wyckoffs(j,
                MetalOnly(cell,oxy,metal.row_id),sg,max_tol=.03)
    true_token=tuple(sorted(int(k) for k in meta['wp_token']))
    scanned=[tuple(k) for k in __import__('itertools').islice(
               j.v48_iter_wyckoff(sg,len(oxy)),int(scan_limit))]
    return dict(original_O_wp_audit='matched',
                original_O_wp_token=str(true_token),
                original_O_wp_multiplicities=str(tuple(len(x['rot']) for x in otemplate.blocks)),
                original_O_wp_in_first_scan=int(true_token in scanned),
                original_O_wp_scan_limit=int(scan_limit),
                original_O_wp_ti_match_error_A=ti_err)

def exact_database_wyckoff_recipe(j,atoms,row_id,symprec=.04,match_tol=.035):
    """Read original full SG and EXACT Ti Wyckoff orbits from an ASE row.

    Original oxygen contributes to the baseline/full-SG determination, NOT
    an O Wyckoff generator: its letters are optional audit metadata only.
    Generation receives Ti coordinates and site types, never original O
    positions, O Wyckoff free parameters or O Wyckoff tokens.
    """
    import spglib
    from pyxtal.symmetry import Group
    from collections import defaultdict
    atoms_metal=strip_oxygen(atoms,row_id)
    positions=np.asarray(atoms.get_scaled_positions(wrap=True),dtype=float)%1.
    cell=np.asarray(atoms.cell.array,float)
    z=np.asarray(atoms.get_atomic_numbers(),dtype=int)
    standardized=spglib.standardize_cell((cell,positions,z),to_primitive=False,
                                        no_idealize=True,symprec=symprec)
    if standardized is None:
        raise ValueError(f'db_id={row_id}: full TiO2 conventionalization failed')
    conventional_cell,conventional_frac,conventional_z=standardized
    conventional_cell=np.asarray(conventional_cell,float)
    conventional_frac=np.asarray(conventional_frac,float)%1.
    conventional_z=np.asarray(conventional_z,int)
    if set(conventional_z)!={8,22} or 2*sum(conventional_z==22)!=sum(conventional_z==8):
        raise ValueError(f'db_id={row_id}: conventional cell lost TiO2 stoichiometry')
    ds=spglib.get_symmetry_dataset(
        (conventional_cell,conventional_frac,conventional_z),symprec=symprec)
    if ds is None:
        raise ValueError(f'db_id={row_id}: full TiO2 symmetry dataset unavailable')
    def field(name):
        return getattr(ds,name) if hasattr(ds,name) else ds[name]
    sg=int(field('number'))
    original_ds=spglib.get_symmetry_dataset((cell,positions,z),symprec=symprec)
    if original_ds is None:
        raise ValueError(f'db_id={row_id}: original full symmetry not detected')
    original_sg=int(getattr(original_ds,'number') if hasattr(original_ds,'number')
                    else original_ds['number'])
    if sg!=original_sg:
        raise ValueError(f'db_id={row_id}: SG changed during conventionalization: '
                         f'{original_sg} -> {sg}; refuse incorrect database Wyckoff')
    equiv=np.asarray(field('equivalent_atoms'),int)
    letters=np.asarray(field('wyckoffs'),str)
    group=Group(sg)
    data={}
    for element,number in (('Ti',22),):
        inds=np.flatnonzero(conventional_z==number)
        by_equiv=defaultdict(list)
        for k in inds:by_equiv[int(equiv[k])].append(int(k))
        # Preserve original orbit order; repeated free WPs remain separate.
        blocks=[]
        for eq,sites in sorted(by_equiv.items(),key=lambda kv:min(kv[1])):
            letter=str(letters[sites[0]]).lower()
            if any(str(letters[k]).lower()!=letter for k in sites):
                raise ValueError(f'db_id={row_id}: inconsistent WP letters within {element} orbit')
            observed=conventional_frac[sites]
            choices=[]
            for wi in range(len(group)):
                wp=group[wi]
                if int(wp.multiplicity)!=len(sites):continue
                if str(getattr(wp,'letter','')).lower()!=letter:continue
                fit=fit_wp_orbit(j,wp,observed,conventional_cell)
                if fit is not None:choices.append((fit[0],wi,fit[1],fit[2]))
            if not choices:
                raise ValueError(f'db_id={row_id}: PyXtal SG{sg} has no matching '
                                 f'{element} {len(sites)}{letter} orbit')
            choices.sort(key=lambda c:c[0])
            error,wi,u,wp_sites=choices[0]
            if error>match_tol:
                raise ValueError(f'db_id={row_id}: {element} {len(sites)}{letter} '
                                 f'Wyckoff fit error {error:.4f} A > {match_tol:.4f} A; '
                                 'possible nonstandard space-group setting')
            blocks.append((int(wi),np.asarray(u,float),
                           np.asarray(wp_sites,float),letter,len(sites),float(error)))
        data[element]=blocks
    # Original oxygen site types are AUDIT metadata only. No O Wyckoff
    # mapping is fitted, returned as a generation constraint, or required.
    oxy_by_equiv=defaultdict(list)
    for ix in np.flatnonzero(conventional_z==8):
        oxy_by_equiv[int(equiv[ix])].append(int(ix))
    oxygen_labels=tuple(f'{len(sites)}{letters[sites[0]]}' for _,sites in
                  sorted(oxy_by_equiv.items(),key=lambda kv:min(kv[1])))
    oxygen_sizes=tuple(len(sites) for _,sites in
                  sorted(oxy_by_equiv.items(),key=lambda kv:min(kv[1])))
    ti_token=tuple(item[0] for item in data['Ti'])
    o_token=()  # NO oxygen Wyckoff inputs in Ti-WP-only generation
    import torch
    template=j.template_from_token(sg,ti_token,torch.device('cpu'))
    # Preserve exactly the same fitted Ti Wyckoff parameters and actual Ti
    # coordinates; only the original oxygen *labels* survive.
    ti_ordered=np.concatenate([b[2] for b in data['Ti']],axis=0)%1.
    actual_ti=conventional_frac[conventional_z==22]
    distance=np.linalg.norm(_wrap_difference(ti_ordered[:,None,:],actual_ti[None,:,:])
                            @conventional_cell,axis=-1)
    r,c=linear_sum_assignment(distance**2)
    ti_error=float(np.max(distance[r,c]))
    if ti_error>match_tol:
        raise ValueError(f'db_id={row_id}: regenerated Ti Wyckoff sites mismatch '
                         f'full database Ti positions by {ti_error:.4f} A')
    n_lat=len(template.lattice_keys)
    raw=np.zeros(n_lat+sum(int(b['dof']) for b in template.blocks),float)
    k=n_lat
    for wi,u,_,_,_,_ in data['Ti']:
        u=np.clip(u,1e-10,1.-1e-10)
        raw[k:k+len(u)]=np.log(u/(1.-u));k+=len(u)
    # Intentionally do not include original oxygen positions or their fitted
    # fractional/free coordinates in the returned object.
    recipe=dict(db_id=int(row_id),sg=sg,cell=conventional_cell.copy(),
                ti_frac=ti_ordered.copy(),ti_raw=raw,
                ti_wp_token=ti_token,oxygen_wp_token=o_token,
                ti_wp_labels=tuple(f'{b[4]}{b[3]}' for b in data['Ti']),
                oxygen_wp_labels=oxygen_labels,
                ti_orbit_sizes=tuple(b[4] for b in data['Ti']),
                oxygen_orbit_sizes=oxygen_sizes,
                ti_wp_fit_error_A=ti_error,
                oxygen_wp_fit_error_A=0.0,  # no O-WP fit is performed
                raw_Ti_count=len(atoms_metal.ti_frac),
                conventional_Ti_count=len(ti_ordered))
    return recipe

def ti_only_standardized(j,metal,symprec=.04):
    """Conventional setting and SG inferred without looking at oxygen."""
    import spglib
    t=np.asarray(metal.ti_frac,float)%1.; c=np.asarray(metal.cell,float)
    ds=spglib.get_symmetry_dataset((c,t,[22]*len(t)),symprec=symprec)
    if ds is None:raise ValueError('Ti-only SG inference failed')
    res=spglib.standardize_cell((c,t,[22]*len(t)),to_primitive=False,
                                no_idealize=True,symprec=symprec)
    if res is None:raise ValueError('Ti-only conventionalization failed')
    cell,ti,_=res
    standardized=MetalOnly(np.asarray(cell,float),np.asarray(ti,float)%1.,metal.row_id)
    return standardized,int(ds.number)

def _make_args(j,arguments):
    # Reuse the actual v59.4b defaults, including LCAO/SCF search controls.
    a=j.w594_parse([])
    a.entrances=int(arguments.entrances)
    a.oxygen_wp_scan=int(arguments.oxygen_wp_scan)
    a.o_hypotheses=int(arguments.o_hypotheses)
    a.probe_scf_cycles=int(arguments.probe_scf)
    a.beam_width=int(arguments.beam_width)
    a.beam_generations=int(arguments.beam_generations)
    a.mutations_per_parent=int(arguments.mutations_per_parent)
    a.refine_top=int(arguments.refine_top)
    a.refine_scf_cycles=int(arguments.refine_scf)
    a.feedback_steps=int(arguments.feedback_steps)
    a.feedback_rounds=1 if a.feedback_steps>0 else 0
    a.center_scf_mix=0.  # DO NOT move the original Ti metal framework.
    a.strain_max=float(arguments.strain_max)
    a.modes_per_ti=int(arguments.modes_per_ti)
    a.octahedral_proposals=int(arguments.octahedral_proposals)
    a.seed=int(arguments.seed)
    a.device=getattr(arguments,'device','auto')
    a.site_tries=int(arguments.site_tries)
    a.rotation_amplitude=float(arguments.rotation_amplitude)
    a.max_oxygen_orbits=int(arguments.max_oxygen_orbits)
    return a


# ============================================================================
# JULIETTE v59.6: Ti-WYCKOFF-ONLY polyhedra and arm-derived oxygen network.
# NO O Wyckoff objects, representatives, tokens, free variables, or projection.
# Every oxygen coordinate is recomputed from the six-arm sets after LCAO
# assignment.  The O-triangle combinatorics are the only oxygen state.
# ============================================================================

def w596_oxygen_from_arms(octas,seed_centers):
    """Select a discrete 3-arm sharing graph and derive 2N O centroids.

    seed_centers are ONLY hypotheses to choose graph connectivity, never
    optimized physical O coordinates. The returned oxygen positions are exact
    periodic least-squares means of the assigned ideal TiO6 arms.
    """
    centers=np.asarray(seed_centers,float).reshape(-1,3)
    if len(centers)!=2*len(octas.ti):
        raise ValueError('Arm-derived TiO2 requires 2*NTi oxygen hypotheses')
    trip,oxygen_cart,rms,bad=assign_provisional_network(octas,centers)
    if len(trip)!=6*len(octas.ti) or len(oxygen_cart)!=2*len(octas.ti):
        raise RuntimeError('LCAO 6-arm/2-oxygen stoichiometry mismatch')
    oxy=(oxygen_cart@octas.inv)%1.
    return oxy,np.asarray(trip,int),float(rms),int(bad)

def w596_oo_penalty(oxygen,cell,guard):
    """Pair repulsion diagnostic on distinct periodic physical O sites."""
    positions=np.asarray(oxygen,float).reshape(-1,3)@cell
    if len(positions)<2:return 0.,float('inf')
    delta=minimum_image_delta(positions[:,None,:],positions[None,:,:],cell)
    ds=np.linalg.norm(delta,axis=-1)
    iu=np.triu_indices(len(ds),1)
    d=ds[iu]
    return float(np.square(np.maximum(0.,float(guard)-d)).sum()),float(d.min())

def w596_snapshot(octas,seed_centers,chem,stage='arm-derived-LCAO'):
    """No oxygen WP: all O positions are rebuilt from TiO6 vertices."""
    o=octas.copy()
    oxygen,trip,rms,bad=w596_oxygen_from_arms(o,seed_centers)
    chem.ti=torch.tensor(o.ti,dtype=DTYPE)
    with torch.no_grad():
        details=chem.score(torch.tensor(oxygen,dtype=DTYPE),details=True)
    sym_ti=max(w594_group_symmetry_error(o.ti,o.cell,o.group_ops),o.symmetry_error())
    if sym_ti>1e-4:raise RuntimeError('TiO6 Wyckoff symmetry drift')
    oxygen_sym=w594_group_symmetry_error(oxygen,o.cell,o.group_ops)
    rep,min_oo=w596_oo_penalty(oxygen,o.cell,chem.guard)
    residual=minimum_image_delta(o.positions().reshape(-1,3),
                                  (oxygen@o.cell)[trip],o.cell,o.inv)
    arm_errors=np.linalg.norm(residual,axis=-1)
    # Physical O symmetry is diagnosed, not imposed as an independent O WP.
    return dict(oct=o,oxygen=oxygen,trip=trip,rms=rms,bad_triplets=bad,
                sharing_max_A=float(np.max(arm_errors)),
                sharing_p95_A=float(np.quantile(arm_errors,.95)),
                details=details,oxygen_wyckoff=None,oxygen_source='TiO6-arm-centroids',
                independent_ti_orbits=len(o.orbits),independent_o_orbits=0,
                wyckoff_symmetry_error=float(sym_ti),
                oxygen_symmetry_error_A=float(oxygen_sym),
                oo_overlap_penalty=float(rep),min_oo_A=float(min_oo),
                pass_chemistry=bool(bad==0 and rms<.12 and
                    np.all(details['goodTi']) and np.all(details['goodO']) and
                    rep<1e-8),
                cycle=0,cycles=0,compatible_symops=len(o.group_ops),
                ti_symops=len(o.group_ops),symmetry_max_rms=0.,
                probe_stage=stage,history=[])

def w596_scf(octas,oxygen_seed,chem,mode_options,steps=5,center_mix=0.):
    """Fit only independent TiO6 WP objects; O is never an optimization DOF.

    The provisional 2N center list selects which arms share. At every cycle
    those physical sites are *rederived* from the current arm coordinates.
    """
    work=octas.copy()
    seeds=np.asarray(oxygen_seed,float).reshape(-1,3)@work.cell
    best=w596_snapshot(work,seeds,chem,stage='initial')
    best_loss=continuous_candidate_objective(best)+2*best['oo_overlap_penalty']
    history=[]
    for k in range(max(0,int(steps))):
        oxygen,trip,rms,bad=w596_oxygen_from_arms(work,seeds)
        # These targets are a deterministic function of the rigid-arm field
        # and its discrete assignments. No independently placed O orbit.
        arms=work.positions().reshape(-1,3)
        cart=oxygen@work.cell
        delta=minimum_image_delta(cart[trip],arms,work.cell,work.inv)
        metal_cart=np.repeat(work.ti@work.cell,6,axis=0)
        goals=(arms+delta-metal_cart).reshape(-1,6,3)
        work.fit(goals,mode_options=mode_options if k%4==0 else None,
                 mix=.55,distortion_max=work.max_strain)
        if center_mix>0:
            # Project all Ti-center shifts onto the independent Wyckoff
            # coordinate DOFs; never optimize one Ti atom independently.
            v60_update_wyckoff_centers(work,oxygen,trip,mix=center_mix)
        # The assignments are discrete; carry the derived sites only as
        # suggestions for the next matching. They are NOT free parameters.
        seeds=cart
        snap=w596_snapshot(work,seeds,chem,stage='orbit-scf')
        snap['cycle']=k+1;snap['cycles']=k+1
        priority=continuous_candidate_objective(snap)+2*snap['oo_overlap_penalty']
        history.append((snap['rms'],snap['bad_triplets'],snap['min_oo_A']))
        if priority<best_loss:
            best=snap;best_loss=priority
    best['history']=history
    return best

def w596_feedback(result,chem,steps=12,rounds=1,strain_limit=.11):
    """Differentiable Ti-orbit-only chemistry + LCAO coincidence refinement.

    Trainable tensors: one SITE-CONSTRAINED coherent 3x3 F per Ti WP orbit.
    Physical oxygen is `mean(assigned, unwrapped arms)` inside the torch
    graph, not an oxygen parameter. Connectivity fixed inside each gradient
    block and recomputed between blocks. Ti centers fixed in recovery test.
    """
    if steps<=0 or rounds<=0:return result
    winner=result
    best_score=continuous_candidate_objective(winner)+2*winner['oo_overlap_penalty']
    work=result['oct'].copy()
    for rd in range(int(rounds)):
        center_cart=winner['oxygen']@work.cell
        oxygen0,trip,_,_=w596_oxygen_from_arms(work,center_cart)
        centers=oxygen0@work.cell
        arms0=work.positions().reshape(-1,3)
        nO=len(oxygen0)
        # Unwrap each arm into the same physical image as the matched O.
        nearest=minimum_image_delta(arms0,centers[trip],work.cell,work.inv)
        image_offset=(arms0-centers[trip])-nearest
        cell=torch.tensor(work.cell,dtype=DTYPE)
        inv=torch.tensor(work.inv,dtype=DTYPE)
        offsets=torch.tensor(image_offset,dtype=DTYPE)
        assignment=torch.tensor(trip,dtype=torch.long)
        ncounts=torch.bincount(assignment,minlength=nO).to(dtype=DTYPE)[:,None]
        params=[];configs=[]
        for orbit in work.orbits:
            B=torch.tensor(orbit.B,dtype=DTYPE)
            base_coeff=torch.tensor(orbit.B.T@orbit.F.reshape(-1),dtype=DTYPE)
            conf=(torch.tensor(orbit.src(),dtype=DTYPE),B,base_coeff,
                  torch.tensor(orbit.maps,dtype=DTYPE),
                  torch.tensor(orbit.frac_sites()@work.cell,dtype=DTYPE))
            configs.append(conf)
            params.append(torch.nn.Parameter(torch.zeros(B.shape[1],dtype=DTYPE)))
        optim=torch.optim.Adam(params,lr=.04)
        round_best=None
        chem.ti=torch.tensor(work.ti,dtype=DTYPE)
        for step in range(int(steps)):
            optim.zero_grad()
            cart_arms=[];fs=[];strain_loss=0.
            for p,(src,B,coef,maps,sites) in zip(params,configs):
                F=(B@(coef+.035*torch.tanh(p))).reshape(3,3)
                fs.append(F)
                local=src@F
                cart_arms.append(torch.einsum('vk,okj->ovj',local,maps)+sites[:,None,:])
                singular=torch.linalg.svdvals(F)
                strain_loss=strain_loss+torch.relu(singular-(1+strain_limit)).square().sum()
                strain_loss=strain_loss+torch.relu((1-strain_limit)-singular).square().sum()
                strain_loss=strain_loss+.05*(singular-1).square().sum()
            all_arms=torch.cat(cart_arms).reshape(-1,3)
            unwrapped=all_arms-offsets
            sums=torch.zeros((nO,3),dtype=DTYPE).index_add(0,assignment,unwrapped)
            oxygen_cart=sums/ncounts
            oxygen_frac=torch.remainder(oxygen_cart@inv,1.)
            ideal_centers=oxygen_cart[assignment]+offsets
            share_loss=torch.mean(torch.sum((all_arms-ideal_centers)**2,dim=-1))
            chem_loss,*_=chem.score(oxygen_frac,components=True)
            diff=oxygen_cart[:,None,:]-oxygen_cart[None,:,:]
            frac=diff@inv
            diff=(frac-torch.round(frac).detach())@cell
            d=torch.sqrt(torch.sum(diff**2,dim=-1)+1e-16)
            keep=torch.triu(torch.ones((nO,nO),dtype=torch.bool),diagonal=1)
            oo_loss=torch.relu(float(chem.guard)-d[keep]).square().sum()
            loss=chem_loss+4.*share_loss+4.*oo_loss+20.*strain_loss
            if not torch.isfinite(loss):break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,4.)
            optim.step()
            if step%5!=4 and step!=steps-1:continue
            trial=work.copy();valid=True
            for orbit,F in zip(trial.orbits,fs):
                mat=F.detach().numpy()
                D,R=w594_polardecompose(mat)
                if D is None or max(abs(np.linalg.eigvalsh(D)-1))>strain_limit+1e-7:
                    valid=False;break
                orbit.F=mat
            if not valid:continue
            trial._rebuild()
            try:
                evaluated=w596_snapshot(trial,oxygen_frac.detach().numpy()@trial.cell,
                                        chem,stage='orbit-autograd')
            except (ValueError,RuntimeError,FloatingPointError):
                continue
            new_score=continuous_candidate_objective(evaluated)+2*evaluated['oo_overlap_penalty']
            if round_best is None or new_score<round_best[0]:round_best=(new_score,evaluated)
            if new_score<best_score:
                best_score=new_score;winner=evaluated
        if round_best is None:break
        work=round_best[1]['oct'].copy()
    winner['chem_feedback_steps']=int(steps)*int(rounds)
    return winner

def w596_fit_arm_correspondence(octas,control_octas):
    """Initialize each independent TiO6 from v59 arm poses, with permutation.

    The old code averaged six arm arrays across WP images by ARRAY INDEX,
    despite the space-group operation generally permuting octahedral arms.
    We solve the 6x6 correspondence before pulling every target to the
    representative frame.  No reference database oxygen appears here.
    """
    from scipy.optimize import linear_sum_assignment
    source=np.asarray(control_octas.positions(),float)
    offset=0;aligned=0
    for orbit in octas.orbits:
        count=orbit.count
        s=source[offset:offset+count]
        old_mode=int(control_octas.mode[offset])
        if orbit.try_mode_pose(old_mode,control_octas.rot[offset]):aligned+=1
        # Avoid wrong vertex correspondence across symmetry-related TiO6.
        current=orbit.positions()
        targets=np.zeros((count,6,3))
        for i in range(count):
            d=np.linalg.norm(current[i,:,None,:]-s[i,None,:,:],axis=-1)
            ii,jj=linear_sum_assignment(d*d)
            targets[i,ii]=s[i,jj]-(orbit.frac_sites()[i]@octas.cell)
        orbit.fit(targets,mix=1.,mode_candidates=[old_mode,orbit.mode])
        offset+=count
    octas._rebuild()
    return aligned

def w596_seeds_from_v59(j,lib,chem,ordered,cell,template,raw,ops,a,seed,info):
    control_ops=j.symmetry_mappings(ordered,cell,.04)
    controls,_=j.e593_search_framework(ordered,cell,lib,chem,control_ops,a,
                         j.str_seed(seed,'matched-arm-basin'),dict(info,ops=control_ops))
    seeds=j.e593_select_diverse(controls,max(a.entrances,a.refine_top))
    modes=j.mode_shortlist(ordered,cell,lib,min(a.modes_per_ti,len(lib['Ti'])))
    proposals=j.framework_orientation_seeds(ordered,cell,lib,modes,
                                np.empty((0,3)),proposals_per_ti=a.octahedral_proposals)
    candidates=[];errors=[];records=[]
    for rank,c in enumerate(seeds):
        original=c['ans']
        try:
            octas=j.WyckoffOctahedra(template,raw,cell,lib,proposals,ops,
                    seed=j.str_seed(seed,'arm-WP',rank),max_strain=a.strain_max,
                    site_attempts=a.site_tries)
            fitted=w596_fit_arm_correspondence(octas,original['oct'])
            # Independently generated v59 oxygen centroids only select a
            # possible sharing graph; final O is rederived from WP TiO6 arms.
            seeds_cart=np.asarray(original['oxygen'],float)@cell
            probe=w596_scf(octas,seeds_cart,chem,modes,steps=a.probe_scf_cycles)
            probe['matched_v59_baseline_Ti_loss_max']=float(np.max(original['details']['localTi']))
            probe['matched_v59_orbits_poses_aligned']=fitted
            item=j.e593_new_item(probe,info,'v59-arm-basin',f'WP-Ti-arms-seed{rank}')
            item['mode_options']=modes
            candidates.append(item)
            records.append(dict(seed=rank,
                v59_Ti_loss_max=probe['matched_v59_baseline_Ti_loss_max'],
                WP_Ti_loss_max=float(np.max(probe['details']['localTi'])),
                matched_orbits=fitted,
                oxygen_source='arm-centroid (NO O Wyckoff)'))
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            errors.append(f'seed{rank}: {type(exc).__name__}: {exc}')
    return j.e593_select_diverse(candidates,max(a.entrances,a.refine_top)),errors,records

def w596_independent_search(j,lib,chem,ordered,cell,template,raw,ops,a,seed,info):
    modes=j.mode_shortlist(ordered,cell,lib,min(a.modes_per_ti,len(lib['Ti'])))
    proposals=j.framework_orientation_seeds(ordered,cell,lib,modes,
                             np.empty((0,3)),proposals_per_ti=a.octahedral_proposals)
    candidates=[];errors=[]
    def one(octas,origin,key):
        peaks=j.coherent_lcao_proposal_field(octas,np.empty((0,3)))
        start=j._select_provisional_centers(octas,peaks,max(.5,chem.guard))
        snap=w596_scf(octas,start,chem,modes,steps=a.probe_scf_cycles)
        item=j.e593_new_item(snap,info,origin,key)
        item['mode_options']=modes
        candidates.append(item)
        return item
    for k in range(a.entrances):
        try:
            octas=j.WyckoffOctahedra(template,raw,cell,lib,proposals,ops,
                seed=0 if k==0 else j.str_seed(seed,'ind',k),
                max_strain=a.strain_max,site_attempts=a.site_tries)
            one(octas,'independent',f'fresh{k}')
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            errors.append(f'entrance{k}: {type(exc).__name__}: {exc}')
    rng=np.random.default_rng(j.str_seed(seed,'whole-orbit-moves'))
    parents=j.e593_select_diverse(candidates,a.beam_width)
    for generation in range(a.beam_generations):
        children=[]
        for pi,parent in enumerate(parents):
            for k in range(a.mutations_per_parent):
                try:
                    octas,move=parent['ans']['oct'].mutate(rng,a.rotation_amplitude,
                                                           proposals,modes)
                    item=one(octas,'mutation',f'{generation}-{pi}-{move}')
                    children.append(item)
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    errors.append(f'mutation{generation}.{pi}.{k}: {type(exc).__name__}: {exc}')
        parents=j.e593_select_diverse(parents+children,a.beam_width)
    return j.e593_select_diverse(candidates,max(a.entrances,a.refine_top)),errors

def w596_final_candidates(item,a):
    result=item['ans'];steps=max(0,int(a.refine_scf_cycles))
    answer=[('probe',result)]
    if steps:
        try:
            refined=w596_scf(result['oct'],result['oxygen']@result['oct'].cell,
                             item['chem_probe'],item['mode_options'],steps=steps)
            answer.append(('refine_scf',refined))
            if a.feedback_steps>0 and a.feedback_rounds>0:
                f=w596_feedback(refined,item['chem_probe'],steps=a.feedback_steps,
                       rounds=a.feedback_rounds,strain_limit=a.strain_max)
                answer.append(('coherent_feedback',f))
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            log('    Ti-WP-only refinement warning: '+type(exc).__name__+': '+str(exc))
    return answer

def recovery_candidates(j,lib,bounds,guard,metal,method,args,seed,recipe=None):
    """Regenerate O without access to source O. Metal/cell remain fixed."""
    import torch
    # Chem evaluator receives only Ti and LIBRARY reference chemistry.
    a=_make_args(j,args)
    with contextlib.redirect_stdout(io.StringIO()):
        chem=j.Chemistry(metal.ti_frac,metal.cell,lib,bounds,guard,accept=.10,
                         hierarchy='ti-first',calibration=None,target_o_cn=True)
    chem.ablation_oxygen_weight=0.
    info=dict(sg=0,wp=(),metal_cycle=0,basin=0,entrance=0,metal_loss=0.,
              ops=[],chem_probe=chem,chem_full=chem)
    data=[];failures=[];wp_info={}
    if method=='v59-lcao':
        # This is the original, non-Wyckoff LCAO control: fixed metal lattice.
        ops=j.symmetry_mappings(metal.ti_frac,metal.cell,args.symprec)
        info['ops']=ops
        try:
            pool,diag=j.e593_search_framework(metal.ti_frac,metal.cell,lib,chem,
                                    ops,a,seed,info)
        except Exception as exc:
            return [],[f'control exploration: {type(exc).__name__}: {exc}'],wp_info
        failures.extend([f'{diag.get("errors",0)} LCAO entrance/mutation errors'])
        for item in j.e593_select_diverse(pool,a.refine_top):
            for phase,result in j.e593_final_candidates(item,a):
                data.append((phase,result,method))
    elif method=='wyckoff':
        # CONTROLLED DATABASE TEST: original FULL SG and Ti Wyckoff token.
        # NO oxygen Wyckoff constraint, free coordinate, or template.
        if recipe is None:
            raise ValueError('Exact-database Wyckoff method requires original WP recipe')
        sg=int(recipe['sg'])
        if not np.allclose(metal.cell,recipe['cell'],rtol=0,atol=1e-8):
            raise ValueError('Wyckoff generator received different cell from database recipe')
        if periodic_oxygen_match(metal.ti_frac,recipe['ti_frac'],metal.cell)>1e-5:
            raise ValueError('Wyckoff generator received different Ti sites from database recipe')
        standardized=metal
        templ=j.template_from_token(sg,recipe['ti_wp_token'],torch.device('cpu'))
        raw=np.asarray(recipe['ti_raw'],float).copy()
        ordered=np.asarray(recipe['ti_frac'],float).copy()
        ops=j.w594_group_operations(__import__('pyxtal.symmetry',fromlist=['Group']).Group(sg))
        wp_info=dict(original_full_sg=sg,original_ti_wp_token=str(recipe['ti_wp_token']),
                     original_oxygen_wp_token=str(recipe['oxygen_wp_token']),
                     original_ti_wp_labels=str(recipe['ti_wp_labels']),
                     original_oxygen_wp_labels=str(recipe['oxygen_wp_labels']),
                     ti_wp_match_error_A=float(recipe['ti_wp_fit_error_A']),
                     oxygen_wp_match_error_A=float(recipe['oxygen_wp_fit_error_A']),
                     exact_database_wp=1)
        # v59.6: O is exclusively derived from learned TiO6 arms, not from
        # original oxygen WP labels or an independent O Wyckoff template.
        wp_info['ti_only_sg']=sg
        wp_info['target_sg_from']='original-full-structure-ASE-row'
        wp_info['standardized_Ti_count']=len(ordered)
        wp_info['input_Ti_count']=len(metal.ti_frac)
        wp_info['oxygen_parameterization']='none: TiO6 arm sharing centroid'
        with contextlib.redirect_stdout(io.StringIO()):
            chem=j.Chemistry(ordered,standardized.cell,lib,bounds,guard,accept=.10,
                             hierarchy='ti-first',calibration=None,target_o_cn=True)
        chem.ablation_oxygen_weight=0.
        info.update(sg=sg,wp=tuple(templ.token),chem_probe=chem,chem_full=chem,ops=ops)
        if getattr(args,'wyckoff_seed_source','independent')=='matched-v59':
            pool,failures,records=w596_seeds_from_v59(
                       j,lib,chem,ordered,standardized.cell,templ,raw,ops,a,seed,info)
            wp_info['v59_to_wp_seed_records']=records
            wp_info['v59_to_wp_seed_attempts']=len(records)
            if records:
                print(f'    MATCHED Ti-WP ARMS: {len(records)} v59 basins; '
                      f'best_v59_Ti_loss={min(x["v59_Ti_loss_max"] for x in records):.5g} '
                      f'best_TiWP_loss={min(x["WP_Ti_loss_max"] for x in records):.5g}; '
                      f'O is generated ONLY by shared TiO6 vertices',flush=True)
        else:
            pool,failures=w596_independent_search(
                   j,lib,chem,ordered,standardized.cell,templ,raw,ops,a,seed,info)
        for item in j.e593_select_diverse(pool,a.refine_top):
            for phase,result in w596_final_candidates(item,a):
                ti_out=np.asarray(result['oct'].ti,float)%1.
                if ti_out.shape!=ordered.shape:
                    failures.append(f'{phase}: changed Ti multiplicity')
                    continue
                mismatch=np.linalg.norm(_wrap_difference(ti_out[:,None,:],
                                ordered[None,:,:])@standardized.cell,axis=-1)
                ir,ic=linear_sum_assignment(mismatch**2)
                if np.max(mismatch[ir,ic])>1e-3:
                    failures.append(f'{phase}: changed fixed database Ti positions')
                    continue
                data.append((phase,result,method))
    else:
        raise ValueError(method)
    return data,failures,wp_info

def coherent_strain_extrema(octas):
    """Actual post-SCF eigenstrain, including any illegal F interpolation.

    It is important not to infer a bound from the user's `--strain-max` flag.
    """
    if hasattr(octas,'orbits'):
        ds=[]
        for orb in octas.orbits:
            # F is row-vector deformation; singular values give stretches.
            eig=np.linalg.svd(orb.F,compute_uv=False)
            ds.extend(np.abs(eig-1.).tolist())
    else:
        ds=[]
        for D in octas.stretch:
            ds.extend(np.abs(np.linalg.eigvalsh(D)-1.).tolist())
    return float(max(ds)) if ds else 0.

def chemistry_row(metrics, rec, extra, reference_guard):
    row={**extra,**metrics}
    row.update(min_oo_A=float(rec['row']['min_oo_A']),
               oo_contacts_below_guard=int(np.sum(rec['_oo_contacts']<reference_guard)),
               oo_repulsion_A2=float(np.sum(np.maximum(0.,reference_guard-rec['_oo_contacts'])**2)),
               ti_o_min_A=float(rec['row']['min_tio_A']),
               ti_o_sixth_max_A=float(rec['row']['tio6_max_A']),
               ti_o_seventh_min_A=float(rec['row']['tio_seventh_min_A']),
               o_ti_third_max_A=float(rec['row']['oti3_max_A']),
               o_ti_fourth_min_A=float(rec['row']['oti4_min_A']))
    return row

def periodic_oxygen_match(a,b,cell):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape:return float('inf')
    d=np.linalg.norm(_wrap_difference(a[:,None,:],b[None,:,:])@cell,axis=-1)
    ii,jj=linear_sum_assignment(d**2)
    return float(np.max(d[ii,jj]))

def _write_csv(out,rows):
    keys=sorted(set().union(*(r.keys() for r in rows))) if rows else ['db_id','status']
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v
                                    for k,v in row.items()})

def audit_oxygen_under_ti_symmetry(ti_frac, o_frac, cell, operations, tolerance_A=.04):
    """Independent *audit*, never generation input.

    Evaluate whether EVERY symmetry operation of the Ti-only structure also
    maps the oxygen set onto itself. All positions must be in the SAME
    fractional coordinate frame as the operations. Correct oxygen-WP symmetry
    requires this condition; symmetric generated O alone is insufficient to
    establish that the imposed Ti-only SG was chemically appropriate.
    """
    ti=np.asarray(ti_frac,float)%1.
    oxygen=np.asarray(o_frac,float)%1.
    cell=np.asarray(cell,float)
    errs_o=[];errs_ti=[];worst_operation=None
    for k,(rot,shift) in enumerate(operations):
        rot=np.asarray(rot,float);shift=np.asarray(shift,float)
        def match(x):
            if not len(x):return float('nan')
            moved=(x@rot.T+shift)%1.
            dd=np.linalg.norm(_wrap_difference(moved[:,None,:],x[None,:,:])@cell,axis=-1)
            rr,cc=linear_sum_assignment(dd**2)
            return float(np.max(dd[rr,cc]))
        eO=match(oxygen);eTi=match(ti)
        errs_o.append(eO);errs_ti.append(eTi)
        if worst_operation is None or eO>errs_o[worst_operation]:worst_operation=k
    a=np.asarray(errs_o,float);t=np.asarray(errs_ti,float)
    return dict(ti_operations_tested=len(operations),
                ti_max_operation_error_A=float(max(t)) if len(t) else None,
                original_O_max_error_under_Ti_SG_A=float(max(a)) if len(a) else None,
                original_O_median_error_under_Ti_SG_A=float(np.median(a)) if len(a) else None,
                ti_ops_not_preserving_original_O=int(np.count_nonzero(a>tolerance_A)),
                original_O_compatible_with_full_Ti_SG=bool(len(a) and np.all(a<=tolerance_A)),
                ti_ops_oxygen_tolerance_A=float(tolerance_A),
                worst_ti_only_op_index=worst_operation)

def audit_original_ti_vs_full_symmetry(atoms,symprec=.04):
    """Audit original oxygen only. Never use this result in recovery_candidates.

    This deliberately analyzes Ti-only and complete TiO2 using the SAME
    original cell/setting, avoiding false differences from conventionalization.
    """
    try:
        import spglib
    except ImportError:
        # Preserve the independent chemistry-recovery benchmark when symmetry
        # analysis is unavailable; do not silently claim a symmetry result.
        return dict(ti_only_sg_original_setting=None,original_full_sg=None,
                    ti_operations_tested=0,ti_ops_not_preserving_original_O=None,
                    original_O_max_error_under_Ti_SG_A=None,
                    original_O_median_error_under_Ti_SG_A=None,
                    original_O_compatible_with_full_Ti_SG=None,
                    ti_only_group_exceeds_original_group=None,
                    ti_symmetry_audit_status='spglib unavailable')
    xyz=np.asarray(atoms.get_scaled_positions(wrap=True),float)%1.
    syms=np.asarray(atoms.get_chemical_symbols())
    cell=np.asarray(atoms.cell.array,float)
    ti=xyz[syms=='Ti'];o=xyz[syms=='O']
    ds_ti=spglib.get_symmetry_dataset((cell,ti,[22]*len(ti)),symprec=symprec)
    ds_full=spglib.get_symmetry_dataset((cell,xyz,[22 if s=='Ti' else 8 for s in syms]),symprec=symprec)
    operations=spglib.get_symmetry((cell,ti,[22]*len(ti)),symprec=symprec)
    if ds_ti is None or ds_full is None or operations is None:
        raise ValueError('spglib could not audit Ti-only and full symmetry')
    ops=list(zip(operations['rotations'],operations['translations']))
    a=audit_oxygen_under_ti_symmetry(ti,o,cell,ops,tolerance_A=symprec)
    a.update(ti_only_sg_original_setting=int(ds_ti.number),
             original_full_sg=int(ds_full.number),
             ti_only_group_exceeds_original_group=bool(int(ds_ti.number)!=int(ds_full.number)
                or a['ti_ops_not_preserving_original_O']>0),
             ti_symmetry_audit_status='audited')
    return a

def independent_original_eval(atoms,j,audit,lib,bounds,guard,row_id):
    """O is available ONLY in audit, before any MetalOnly object is constructed."""
    metal=strip_oxygen(atoms,row_id)
    symbols=np.asarray(atoms.get_chemical_symbols())
    fullfrac=np.asarray(atoms.get_scaled_positions(wrap=True),float)%1.
    true_oxygen=fullfrac[symbols=='O']
    metrics=chemistry_metrics(j,lib,bounds,guard,metal.ti_frac,metal.cell,true_oxygen)
    rec=get_audit_record(audit,metal,true_oxygen,f'db_{row_id}','origin')
    return metal,metrics,rec


def self_test():
    from scipy.spatial.transform import Rotation
    class FakeAtoms:
        def get_chemical_symbols(self):return ['Ti','O','O']
        def get_pbc(self):return [True]*3
        class cell:array=np.eye(3)*4.5
        def get_scaled_positions(self,wrap=True):
            return np.array([[0,0,0],[.27,.28,0],[.73,.72,0]])
    metal=strip_oxygen(FakeAtoms(),3)
    assert not hasattr(metal,'oxygen')
    assert metal.row_id==3 and len(metal.ti_frac)==1
    j=sys.modules[__name__]
    class Op:
        rotation_matrix=np.eye(3);translation_vector=np.zeros(3)
    class WP:
        ops=[Op()]
        multiplicity=1
        def get_dof(self):return 3
        def get_position_from_free_xyzs(self,u):return np.asarray(u)
    cell=np.eye(3)*4.6
    result=fit_wp_orbit(j,WP(),np.array([[.27,.28,.31]]),cell)
    assert result and result[0]<1e-7
    assert abs(result[1][0]-.27)<1e-6
    # Test geometric deduplication honors atom permutation and PBC.
    o=np.array([[.1,.1,.1],[.3,.4,.5]])
    other=o[[1,0]].copy();other[1,0]+=1
    assert periodic_oxygen_match(o,other,cell)<1e-9
    # Ti-only symmetry is NOT generally a valid oxygen symmetry.
    # Rutile-like Ti positions admit a pure body-centred translation, while
    # the proper rutile oxygen framework requires a screw component instead.
    rutile_ti=np.array([[0.,0.,0.],[.5,.5,.5]])
    u=.305
    rutile_o=np.array([[u,u,0.],[-u,-u,0.],
                       [.5+u,.5-u,.5],[.5-u,.5+u,.5]])%1.
    transops=[(np.eye(3),np.zeros(3)),(np.eye(3),np.array([.5,.5,.5]))]
    check=audit_oxygen_under_ti_symmetry(rutile_ti,rutile_o,
                   np.diag([4.594,4.594,2.959]),transops,tolerance_A=.04)
    assert check['ti_max_operation_error_A']<1e-9
    assert check['ti_ops_not_preserving_original_O']==1
    assert check['original_O_max_error_under_Ti_SG_A']>1.0
    print('PASS symmetry counterexample: Ti-only translation exact, oxygen mismatch >1 A')
    # Test the actual deformation bound checker catches direct F blending.
    class FakeOctas:
        stretch=np.array([np.diag([1,.12,.12])])
    assert coherent_strain_extrema(FakeOctas())>.8
    # Real O repulsion measure remains purely geometric and symmetric.
    synthetic_self_test()
    print('PASS metal-only LCAO input and free-WP fit; exact database O-WP labels allowed, original O coordinates excluded')
    print('IMPORTANT: no actual tio2.db structures were available or measured in this environment')


# V60 generation replaces the recovery-only CLI. See v60_main below.

# ============================================================================
# JULIETTE v60 — NORMAL TiO2 GENERATION WITH v59.6 Ti-WYCKOFF BUILDING UNITS
# ============================================================================
# The standalone source retains the validated v59.6 classes and v59.2 metal
# optimization, but this entry point does NOT load an ASE database, does NOT
# use original oxygen positions or original Ti frameworks, and does NOT
# initialize from an independently fitted v59 oxygen network.
#
# Generate Ti Wyckoff coordinates -> one (mode, SO(3), coherent strain) for
# each Ti Wyckoff orbit -> expand all six arms under proper SG isometries ->
# LCAO 3-arm physical oxygen sharing -> short Wyckoff SCF -> selectively refine.
# Only actual Ti-orbit degrees of freedom are optimized. No O Wyckoff variables.

def v60_parse(argv=None):
    # Full search is a budget preset, not a different oxygen parameterization.
    # A shard always owns complete SGs so final top-N CIFs are never split.
    requested=list(sys.argv[1:] if argv is None else argv)
    aux=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    aux.add_argument('--search-mode',choices=('pilot','full'),default='pilot',
                     help='Full uses all 230 SGs and a wider exploration budget')
    aux.add_argument('--sg-shard-count',type=int,default=1,
                     help='Number of independent disjoint SG shards')
    aux.add_argument('--sg-shard-index',type=int,default=0,
                     help='Zero-based index of the SG shard to execute')
    aux.add_argument('--center-scf-mix',type=float,default=0.,
                     help='Wyckoff-free Ti-center update each SCF iteration; 0 disables')
    aux.add_argument('--strict-v596-archive',action='store_true',
                     help='Archive only space-group-equivariant, physically coherent v59.6 arm-derived oxygen networks')
    aux.add_argument('--allow-broken-sharing',action='store_true',
                     help='Diagnostic only: override full-search strict v59.6 archive (not recommended)')
    aux.add_argument('--max-physical-arm-residual',type=float,default=.35,
                     help='Maximum TiO6-arm to shared-O displacement for structurally valid full-search CIFs, A')
    aux.add_argument('--ti-o-bond-slack',type=float,default=.20,
                     help='Allowed Ti--O deviation beyond learned-mode strain envelope (A)')
    aux.add_argument('--site-tries',type=int,default=24,
                     help='Site-compatible independent TiO6 orientation tries')
    aux.add_argument('--oo-probe-weight',type=float,default=4.,
                     help='O--O overlap importance in cheap basin selection; soft, not a gate')
    aux.add_argument('--oo-archive-weight',type=float,default=8.,
                     help='O--O overlap importance in final top-N rank; soft, not a gate')
    aux.add_argument('--sharing-archive-weight',type=float,default=.05,
                     help='Additional sharing mismatch weight in final top-N rank')
    aux.add_argument('--full-sg-audit-tolerance',type=float,default=.055)
    aux.add_argument('--require-full-o-symmetry',action='store_true',
                     help='Optional symmetry filter on physical shared O (off by default)')
    aux.add_argument('--print-probe-errors',type=int,default=3,
                     help='How many failed basin examples to print per Ti framework')
    extras,remaining=aux.parse_known_args(argv)
    a=e593_parse(remaining)
    for k,v in vars(extras).items():setattr(a,k,v)
    if a.search_mode=='full':
        # Preserve explicit CLI settings; only replace parser defaults.
        # v61.6: a 3-orbit cap can exclude complete SGs at 24 FU
        # (e.g. SG75-78 need more independent Ti sites). Zero means
        # unlimited orbit count; the explicit scan cap still limits work.
        preset=dict(space_groups='all',wp_per_sg=8,wp_scan_limit=1600,
                    max_ti_orbits=0,
                    metal_cycles=2,raw_starts=24,optimized_starts=8,
                    basins_per_wp=3,natural_steps=12,learned_steps=20,
                    entrances=12,site_tries=32,probe_scf_cycles=4,
                    beam_width=6,beam_generations=3,mutations_per_parent=2,
                    pool_per_sg=40,refine_top=10,refine_scf_cycles=10,
                    feedback_steps=8,max_unique_per_sg=10)
        for name,value in preset.items():
            flag='--'+name.replace('_','-')
            if not any(z==flag or z.startswith(flag+'=') for z in requested):
                setattr(a,name,value)
    # STRICT SG is the default in every search mode: never refine bad LCAO
    # topologies. A diagnostic opt-out must be explicitly requested.
    if not a.allow_broken_sharing:
        a.strict_v596_archive=True
    if a.max_physical_arm_residual<=0:
        raise ValueError('--max-physical-arm-residual must be positive')
    if a.ti_o_bond_slack<0:
        raise ValueError('--ti-o-bond-slack must be nonnegative')
    if a.sg_shard_count<1 or not 0<=a.sg_shard_index<a.sg_shard_count:
        raise ValueError('Require 0 <= --sg-shard-index < --sg-shard-count')
    all_sgs=parse_sgs(a.space_groups)
    chosen=[sg for i,sg in enumerate(all_sgs)
            if i%a.sg_shard_count==a.sg_shard_index]
    if not chosen:
        raise ValueError('SG shard is empty; reduce --sg-shard-count')
    a._full_sg_selection=','.join(map(str,all_sgs))
    a.space_groups=','.join(map(str,chosen))
    # Validate the final effective budget after applying the full preset.
    if not 1<=a.max_unique_per_sg<=10:
        raise ValueError('max-unique-per-sg must be between 1 and 10')
    if a.raw_starts<a.optimized_starts or a.pool_per_sg<a.refine_top:
        raise ValueError('Invalid search budget: raw < optimized or pool < refine')
    if a.site_tries<1:raise ValueError('--site-tries must be positive')
    if not 0<=a.center_scf_mix<=.2:raise ValueError('--center-scf-mix must be in [0,.2]')
    if a.oo_probe_weight<0 or a.oo_archive_weight<0 or a.sharing_archive_weight<0:
        raise ValueError('Penalty weights must be >= 0')
    if a.full_sg_audit_tolerance<=0:raise ValueError('Positive symmetry tolerance needed')
    if a.print_probe_errors<0:raise ValueError('Invalid diagnostic count')
    return a

def v60_probe_score(item,oo_weight=4.):
    """Soft *search* preference for symmetry-consistent v59.6 LCAO basins.

    Crucially this does not fit, symmetrize, or move an oxygen independently.
    It prevents the beam from favoring inexpensive but SG1-collapsed sharing
    graphs while the physical archive certificate remains strict.
    """
    r=item['ans']
    raw=float(continuous_candidate_objective(r))
    oo=float(oo_weight)*float(r.get('oo_overlap_penalty',0.))
    sym=float(r.get('oxygen_symmetry_error_A',0.))
    sharing=float(r.get('sharing_max_A',0.))
    # Distinguish exact whole-unit symmetry from discrete oxygen-network
    # symmetry. The latter is not automatic after Hungarian assignment.
    sg_penalty=(20.+40.*sym) if sym>1.e-4 else 0.
    arm_penalty=30.*max(sharing-.50,0.)**2
    return raw+oo+sg_penalty+arm_penalty

def v60_diverse(items,capacity,oo_weight=4.):
    """Distinct modes, topology and orbit poses; no binary goodTi/goodO gate.

    The strict chemical flags are diagnostic only. O--O repulsion is a strong
    continuous *search* penalty, not a cutoff, so imperfect structures can
    still be retained if they are the best available in their SG.
    """
    if not items or capacity<1:return []
    by_sig={}
    for item in items:
        score=v60_probe_score(item,oo_weight)
        if not np.isfinite(score):continue
        sig=e593_topology(item)  # includes orbit mode/F and sharing graph
        old=by_sig.get(sig)
        if old is None or score<v60_probe_score(old,oo_weight):by_sig[sig]=item
    available=list(by_sig.values())
    if not available:return []
    # Take distinct representatives of five views, then rank remaining with
    # the same continuous, O--O-aware objective. Structural diversity is not
    # confused with stringent chemical pass/fail flags.
    views=[
      lambda x:v60_probe_score(x,oo_weight),
      lambda x:float(np.max(x['ans']['details']['localTi'])),
      lambda x:float(np.max(x['ans']['details']['localO'])),
      lambda x:float(x['ans'].get('oo_overlap_penalty',0.)),
      lambda x:float(x['ans'].get('sharing_max_A',float('inf'))),
    ]
    picked=[];used=set()
    for key in views:
        if len(picked)>=capacity:break
        for x in sorted(available,key=key):
            sig=e593_topology(x)
            if sig not in used:
                picked.append(x);used.add(sig);break
    for x in sorted(available,key=lambda z:v60_probe_score(z,oo_weight)):
        if len(picked)>=capacity:break
        sig=e593_topology(x)
        if sig not in used:picked.append(x);used.add(sig)
    return picked

def v60_update_wyckoff_centers(octas,oxygen,trip,mix=0.):
    """SCF translation in Wyckoff-free coordinates, not per Ti atom.

    Each orbit collects periodic oxygen--arm residuals from *all* its TiO6
    images, pulls them through the proper SG cartesian maps, and projects the
    mean shift onto its Wyckoff coordinate basis. No O coordinates are trained.
    """
    if mix<=0:return 0.
    arms=octas.positions().reshape(-1,3)
    target=np.asarray(oxygen,float)[np.asarray(trip,int)]@octas.cell
    shift=minimum_image_delta(target,arms,octas.cell,octas.inv).reshape(-1,6,3).mean(axis=1)
    index=0;max_change=0.
    for orbit in octas.orbits:
        if orbit.dof:
            all_shift=np.stack([shift[index+i]@np.linalg.inv(orbit.maps[i])
                                for i in range(orbit.count)])
            frac_shift=np.mean(all_shift,axis=0)@octas.inv
            free_step=np.linalg.lstsq(orbit.A,frac_shift,rcond=None)[0]
            free_step=np.clip(free_step,-.03,.03)*float(mix)
            orbit.u=(orbit.u+free_step)%1.
            max_change=max(max_change,float(np.linalg.norm((orbit.A@free_step)@octas.cell)))
        index+=orbit.count
    octas._rebuild()
    return max_change

def v60_orbit_scf(octas,oxygen_seed,chem,mode_options,steps=5,center_mix=0.):
    """One SCF loop, one transform and free-coordinate fit per Ti orbit."""
    return w596_scf(octas,oxygen_seed,chem,mode_options,
                    steps=steps,center_mix=center_mix)

class V603InvalidTopology(ValueError):
    """Expected failure of a discrete LCAO topology; do not refine or archive."""

def v603_certify(result,ops,chem,a):
    """Check v59.6 arm-sharing construction BEFORE costly polyhedral SCF.

    The atomic oxygen positions remain *derived* from the TiO6 arms. The
    certificate never adjusts a single arm, oxygen position, or WP.
    """
    certificate=v602_v596_certificate(
        result,ops,chem.guard,
        max_arm_residual=a.max_physical_arm_residual,sg_tolerance=1.e-4)
    # Even when the discrete sharing is correct, reject impossible Ti--O
    # distances relative to this orbit's learned six-arm mode. This compares
    # the actual physical shared oxygen positions to the rigid mode after
    # bounded coherent strain; it is not a learned-chemistry pass/fail flag.
    if certificate['ok']:
        octas=result['oct']
        oxygen=np.asarray(result['oxygen'],float)
        trip=np.asarray(result['trip'],int)
        ti_cart=np.repeat(octas.ti@octas.cell,6,axis=0)
        phys=(oxygen@octas.cell)[trip]
        bonds=np.linalg.norm(minimum_image_delta(phys,ti_cart,
                                     octas.cell,octas.inv),axis=1)
        baseline=np.concatenate([
            np.tile(np.linalg.norm(orb.src(),axis=1),orb.count)
            for orb in octas.orbits])
        allowed_low=baseline*(1-a.strain_max)-a.ti_o_bond_slack
        allowed_high=baseline*(1+a.strain_max)+a.ti_o_bond_slack
        if np.any(bonds<allowed_low) or np.any(bonds>allowed_high):
            certificate=dict(certificate,ok=False,
                reason='physical Ti--O bond outside learned-mode strain envelope',
                min_TiO_A=float(bonds.min()),max_TiO_A=float(bonds.max()))
    return certificate


class V60Archive(_V59TopArchive):
    """Original no-gate archive with physically penalized final ranking."""
    @staticmethod
    def key(record):
        q=record['q']
        return (float(q.get('rank_score',q['maxTi'])),float(q['maxTi']),
                float(q['meanTi']),float(q['maxO']),float(q['sharing']),str(record['name']))

    def write_final(self,output):
        for sg in sorted({r['sg'] for r in self.records}):
            if sg not in self.finalized_sgs:self.write_sg_final(output,sg)
        rows=sorted(self.records,key=lambda r:
                    (float(r['q']['maxTi']),r['sg'],r['name']))
        log('FINAL STRUCTURES (sorted by actual Ti_loss_max, NOT archive score)')
        log('initial_spgr final_spgr N_atom_primitive Ti_loss_min Ti_loss_max Ti_loss_avg '
            'N_good_Ti N_good_O min_OO_A rank_score sharing_graph_violations')
        if not rows:log('  (no finite TiO2 candidates)')
        for r in rows:
            try:final_sg,n_prim=self.symmetry_details(r,self.symprec)
            except Exception as exc:
                final_sg='unknown';n_prim='unknown'
                log(f'  SYMMETRY WARNING {r["name"]}: {exc!r}')
            q=r['q']
            log(f'{r["sg"]:03d} {final_sg} {n_prim} '
                f'{q["minTi"]:.8g} {q["maxTi"]:.8g} {q["meanTi"]:.8g} '
                f'{q["goodTi"]} {q["goodO"]} '
                f'{q.get("minOO",float("nan")):.5g} '
                f'{q["rank_score"]:.5g} {q.get("sharing_violations",0)}')
        log(f'FINAL CIFs WRITTEN: {len(self.written_paths)} to {Path(output)/"cif"}')
        return rows

def v60_archive_row(result,record,a,graph_violations=0):
    row=e593_result_as_archive_row(result,record)
    rep=float(result.get('oo_overlap_penalty',0.))
    sharing=float(record['max_o_move'])
    rank=(float(record['Ti_loss_max'])+float(a.oo_archive_weight)*rep+
          float(a.sharing_archive_weight)*sharing)
    row['q'].update(rank_score=rank,oo_penalty=rep,
                    minOO=float(result.get('min_oo_A',float('nan'))),
                    sharing_violations=int(graph_violations))
    return row


def v602_v596_certificate(result,group_ops,guard,max_arm_residual=.50,
                           sg_tolerance=1.e-4):
    """Audit invariant v59.6 structure: only TiO6 arms define physical O.

    No oxygen Wyckoff coordinates, oxygen relaxation, or per-vertex motion.
    The discrete LCAO triplet partition must map onto itself under every
    space-group operation. A failed certificate is NOT a chemical-score gate:
    it signals that the saved physical atoms do not represent the SG/TiO6
    building units the generator claims to have constructed.
    """
    o=result['oct']
    oxy=np.asarray(result['oxygen'],float).reshape(-1,3)
    trip=np.asarray(result['trip'],int).reshape(-1)
    nt=len(o.ti);no=len(oxy)
    problems=[]
    if no!=2*nt or len(trip)!=6*nt or np.any(trip<0) or np.any(trip>=no):
        return dict(ok=False,reason='invalid arm-to-oxygen stoichiometry',
                    graph_bad=-1,oxygen_sg_err=float('inf'),
                    arm_max_A=float('inf'),centroid_error_A=float('inf'))
    counts=np.bincount(trip,minlength=no)
    if np.any(counts!=3):problems.append('not exactly 3 arms per physical O')
    if result.get('bad_triplets',0):
        problems.append(f'duplicate Ti image within O triplet: {result["bad_triplets"]}')
    # Check that oxygen coordinates are genuinely the centroids of the
    # assigned ideal TiO6 arms, not a second independently optimized field.
    arms=np.asarray(o.positions(),float).reshape(-1,3)
    anchored=oxy[trip]@o.cell
    diff=minimum_image_delta(arms,anchored,o.cell,o.inv)
    unwrapped=anchored+diff
    means=np.stack([np.mean(unwrapped[trip==i],axis=0)
                    if counts[i] else oxy[i]@o.cell for i in range(no)])
    err=np.linalg.norm(minimum_image_delta(means,oxy@o.cell,o.cell,o.inv),axis=-1)
    centroid_error=float(np.max(err))
    if centroid_error>1.e-5:
        problems.append(f'O is not an LCAO arm centroid: {centroid_error:.4g} A')
    arm_max=float(np.max(np.linalg.norm(diff,axis=-1)))
    if arm_max>max_arm_residual:
        problems.append(f'physical TiO6 arm mismatch {arm_max:.4g} > {max_arm_residual:.4g} A')
    if float(result.get('min_oo_A',0.))<float(guard)-1.e-4:
        problems.append(f'O--O contact {result.get("min_oo_A",0.):.4g} < guard {guard:.4g} A')
    oxygen_sym=float(w594_group_symmetry_error(oxy,o.cell,group_ops))
    if oxygen_sym>sg_tolerance:
        problems.append(f'physical O breaks group: {oxygen_sym:.4g} A')
    graph_bad,total,arm_err=w594_sharing_graph_audit(o,oxy,trip)
    if graph_bad:
        problems.append(f'LCAO graph breaks group: {graph_bad}/{total} arms')
    if arm_err>sg_tolerance:
        problems.append(f'ideal TiO6 arms break group: {arm_err:.4g} A')
    return dict(ok=not problems,reason='; '.join(problems),
                graph_bad=int(graph_bad),graph_total=int(total),
                oxygen_sg_err=oxygen_sym,arm_max_A=arm_max,
                arm_sg_err=arm_err,centroid_error_A=centroid_error)




# ============================================================================
# JULIETTE v61 — TiO6-WYCKOFF OBJECTS WITH ATTACHED SHARING POLICY
#
# ONLY the full six-arm TiO6 objects are constrained by the parent SG.
# Shared oxygen is the periodic centroid of 3 selected arms; it may lower SG.
# No independent oxygen coordinates, oxygen Wyckoff variables, or global O-SG
# symmetry constraints are used by the optimizer.  Cartesian arms / oxygen
# centroids are transient geometry views; only final CIFs persist atom arrays.
# ============================================================================

class V61InvalidSharing(ValueError):
    """An incomplete/non-threefold LCAO topology cannot be optimized."""

class V61SharingPolicy:
    """Discrete immutable TiO6-arm -> physical-O incidence, with orbit links.

    A physical O is an integer key shared by exactly three arms.  The policy
    has NO oxygen positions, independent O Wyckoff sites or O optimizers.
    Each WyckoffPolyhedron carries its own `sharing_policy` subsection; the
    global triplet registry is required only to link different TiO6 orbits.
    """
    def __init__(self,octas,trip):
        trip=np.asarray(trip,dtype=np.int64).reshape(-1)
        n_ti=len(octas.ti)
        n_o=2*n_ti
        if len(trip)!=6*n_ti or np.any(trip<0) or np.any(trip>=n_o):
            raise V61InvalidSharing('Wrong TiO6 arm or physical-O count')
        counts=np.bincount(trip,minlength=n_o)
        if np.any(counts!=3):
            raise V61InvalidSharing('Each physical O must link exactly three TiO6 arms')
        self.trip=trip.copy()
        self.trip.flags.writeable=False
        self.n_oxygen=n_o
        self.groups=tuple(tuple(int(x) for x in np.flatnonzero(trip==i))
                          for i in range(n_o))
        self.orbit_links=[]
        start=0
        for orbit in octas.orbits:
            end=start+orbit.count*6
            mapping=tuple(tuple(int(x) for x in row)
                          for row in trip[start:end].reshape(orbit.count,6))
            self.orbit_links.append(mapping)
            start=end
        assert start==len(trip)

    def attach(self,octas):
        if len(self.orbit_links)!=len(octas.orbits):
            raise V61InvalidSharing('Sharing policy does not match Ti Wyckoff orbit count')
        octas.sharing_registry=self
        for orbit,links in zip(octas.orbits,self.orbit_links):
            if len(links)!=orbit.count:raise V61InvalidSharing('Orbit multiplicity changed')
            orbit.sharing_policy=links
        return octas

def v61_derive_oxygen(octas):
    """Evaluate physical O centroids lazily from an ATTACHED fixed sharing graph.

    The returned fractional array is a transient read-only result for scoring;
    there are no trainable independent oxygen degrees of freedom.
    """
    policy=getattr(octas,'sharing_registry',None)
    if policy is None:raise V61InvalidSharing('TiO6 orbits have no sharing policy')
    cell=octas.cell;inv=octas.inv
    arms=octas.positions().reshape(-1,3)
    oxygen_cart=np.empty((policy.n_oxygen,3),float)
    for site,arm_ids in enumerate(policy.groups):
        first=arms[arm_ids[0]]
        # Each member is unwrapped to the same periodic image as the anchor.
        local=first+minimum_image_delta(arms[list(arm_ids)],first,cell,inv)
        oxygen_cart[site]=local.mean(axis=0)
    oxygen=np.mod(oxygen_cart@inv,1.)
    residual=minimum_image_delta(arms,oxygen[policy.trip]@cell,cell,inv)
    shift=np.linalg.norm(residual,axis=1)
    bad_groups=0
    # Different periodic images of the same crystallographic Ti index are
    # allowed, but the *same* Ti image cannot appear twice at one oxygen.
    for site,arm_ids in enumerate(policy.groups):
        keys=[]
        for ia in arm_ids:
            image=np.rint((arms[ia]-oxygen[site]@cell-residual[ia])@inv).astype(int)
            keys.append((int(ia//6),*image.tolist()))
        if len(set(keys))!=3:bad_groups+=1
    return oxygen,float(np.sqrt(np.mean(shift**2))),int(bad_groups),shift

def v61_attach_lcao_policy(octas,centers):
    """Use classic v59.6 LCAO *once* to sample a discrete sharing policy.

    The Hungarian match is used for topology proposal, not repeated between
    steps.  Changing topology is a separate exploration branch.
    """
    _,trip,_,_=w596_oxygen_from_arms(octas,centers)
    V61SharingPolicy(octas,trip).attach(octas)
    oxygen,rms,bad,shift=v61_derive_oxygen(octas)
    if bad:raise V61InvalidSharing(f'{bad} repeated Ti images in O triplets')
    return oxygen,rms

def v61_ideal_symmetry_error(octas):
    """Exact Ti-WP and six-arm equivariance, WITHOUT testing oxygen symmetry."""
    return max(float(octas.symmetry_error()),
               float(w594_group_symmetry_error(octas.ti,octas.cell,octas.group_ops)))

def v61_snapshot(octas,chem,stage='polyhedron-sharing'):
    work=octas.copy()
    oxygen,rms,bad,shift=v61_derive_oxygen(work)
    sym=v61_ideal_symmetry_error(work)
    if sym>1e-4:raise RuntimeError(f'TiO6 WP symmetry drift {sym:.4g} A')
    if coherent_strain_extrema(work)>work.max_strain+1e-5:
        raise RuntimeError('TiO6 mode/rotation/coherent strain exceeds limit')
    chem.ti=torch.as_tensor(work.ti,dtype=DTYPE)
    chem.cell=torch.as_tensor(work.cell,dtype=DTYPE)
    with torch.no_grad():
        details=chem.score(torch.as_tensor(oxygen,dtype=DTYPE),details=True)
    oo_rep,oo_min=w596_oo_penalty(oxygen,work.cell,chem.guard)
    return dict(oct=work,oxygen=oxygen,trip=work.sharing_registry.trip,
                rms=float(rms),bad_triplets=int(bad),
                sharing_max_A=float(np.max(shift)),
                sharing_p95_A=float(np.quantile(shift,.95)),
                details=details,oxygen_wyckoff=None,
                oxygen_source='TiO6-WP-fixed-sharing-centroids',
                independent_ti_orbits=len(work.orbits),independent_o_orbits=0,
                wyckoff_symmetry_error=float(sym),
                oxygen_symmetry_error_A=float('nan'),
                oo_overlap_penalty=float(oo_rep),min_oo_A=float(oo_min),
                pass_chemistry=bool(bad==0 and np.all(details['goodTi'])
                                    and np.all(details['goodO']) and oo_rep<1e-8),
                cycle=0,cycles=0,compatible_symops=len(work.group_ops),
                ti_symops=len(work.group_ops),symmetry_max_rms=0.,
                probe_stage=stage,history=[])

def v61_scf(octas,chem,mode_options,steps=4,center_mix=0.):
    """Coherent orbit-local SCF with FIXED linked sharing policy.

    Only `orbit.fit()` changes a mode and its shared rotation/strain; optional
    center updates act on Wyckoff free coordinates, never on individual Ti/O.
    """
    work=octas.copy()
    best=v61_snapshot(work,chem,'initial-fixed-sharing')
    best_score=continuous_candidate_objective(best)+2*best['oo_overlap_penalty']
    history=[]
    for k in range(max(0,int(steps))):
        oxygen,_,bad,_=v61_derive_oxygen(work)
        if bad:break
        arms=work.positions().reshape(-1,3)
        cart=oxygen@work.cell
        correction=minimum_image_delta(cart[work.sharing_registry.trip],arms,
                                       work.cell,work.inv)
        Ti_centers=np.repeat(work.ti@work.cell,6,axis=0)
        goals=(arms+correction-Ti_centers).reshape(-1,6,3)
        work.fit(goals,mode_options=mode_options if k%4==0 else None,
                 mix=.55,distortion_max=work.max_strain)
        if center_mix>0:
            # A free-Wyckoff shift invalidates cached coset maps. Recompute
            # them immediately; restore the complete prior object if invalid.
            before_centers=work.copy()
            v60_update_wyckoff_centers(work,oxygen,work.sharing_registry.trip,
                                       mix=center_mix)
            try:
                v64_refresh_state(work)
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                work=before_centers
        snap=v61_snapshot(work,chem,'orbit-fixed-sharing-SCF')
        snap['cycle']=k+1;snap['cycles']=k+1
        history.append((snap['rms'],snap['bad_triplets'],snap['min_oo_A']))
        score=continuous_candidate_objective(snap)+2*snap['oo_overlap_penalty']
        if score<best_score:
            best,best_score=snap,score
    best['history']=history
    return best

def v61_feedback(result,chem,steps=8,rounds=1,strain_limit=.11):
    """v59.6 differentiable coherent-F optimizer, but NEVER rematch oxygen.

    Training variables are site-constrained affine coefficients per Ti orbit.
    Oxygen centroids are torch reductions over the immutable triplet registry.
    """
    if steps<=0 or rounds<=0:return result
    winner=result
    winner_score=continuous_candidate_objective(winner)+2*winner['oo_overlap_penalty']
    work=result['oct'].copy()
    for rd in range(int(rounds)):
        oxygen0,_,bad,_=v61_derive_oxygen(work)
        if bad:break
        trip=work.sharing_registry.trip
        centers=oxygen0@work.cell
        arms0=work.positions().reshape(-1,3)
        n_o=len(oxygen0)
        nearest=minimum_image_delta(arms0,centers[trip],work.cell,work.inv)
        offset=(arms0-centers[trip])-nearest
        cell=torch.as_tensor(work.cell,dtype=DTYPE)
        inv=torch.as_tensor(work.inv,dtype=DTYPE)
        offsets=torch.as_tensor(offset,dtype=DTYPE)
        assignment=torch.as_tensor(trip.copy(),dtype=torch.long)
        counts=torch.bincount(assignment,minlength=n_o).to(dtype=DTYPE)[:,None]
        params=[];configs=[]
        for orbit in work.orbits:
            B=torch.as_tensor(orbit.B,dtype=DTYPE)
            coeff=torch.as_tensor(orbit.B.T@orbit.F.reshape(-1),dtype=DTYPE)
            configs.append((torch.as_tensor(orbit.src(),dtype=DTYPE),B,coeff,
                            torch.as_tensor(orbit.maps,dtype=DTYPE),
                            torch.as_tensor(orbit.frac_sites()@work.cell,dtype=DTYPE)))
            params.append(torch.nn.Parameter(torch.zeros(B.shape[1],dtype=DTYPE)))
        optimizer=torch.optim.Adam(params,lr=.04)
        round_best=None
        chem.ti=torch.as_tensor(work.ti,dtype=DTYPE)
        for step in range(int(steps)):
            optimizer.zero_grad()
            all_arms=[];updated=[];strain_loss=0.
            for p,(src,B,base,maps,sites) in zip(params,configs):
                F=(B@(base+.035*torch.tanh(p))).reshape(3,3)
                updated.append(F)
                local=src@F
                all_arms.append(torch.einsum('vk,okj->ovj',local,maps)+sites[:,None,:])
                singular=torch.linalg.svdvals(F)
                strain_loss=strain_loss+torch.relu(singular-(1+strain_limit)).square().sum()
                strain_loss=strain_loss+torch.relu((1-strain_limit)-singular).square().sum()
            flat=torch.cat(all_arms).reshape(-1,3)
            unwrapped=flat-offsets
            sums=torch.zeros((n_o,3),dtype=DTYPE).index_add(0,assignment,unwrapped)
            center_cart=sums/counts
            center_frac=torch.remainder(center_cart@inv,1.)
            ideals=center_cart[assignment]+offsets
            coincidence=torch.mean(torch.sum((flat-ideals)**2,dim=-1))
            chem_loss,*_=chem.score(center_frac,components=True)
            pair=center_cart[:,None,:]-center_cart[None,:,:]
            pair=(pair@inv-torch.round((pair@inv)).detach())@cell
            d=torch.linalg.vector_norm(pair,dim=-1)
            upper=torch.triu(torch.ones((n_o,n_o),dtype=torch.bool),diagonal=1)
            oo=torch.relu(float(chem.guard)-d[upper]).square().sum()
            loss=chem_loss+4.*coincidence+4.*oo+20.*strain_loss
            if not torch.isfinite(loss):break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,4.)
            optimizer.step()
            if step%5!=4 and step!=steps-1:continue
            trial=work.copy()
            valid=True
            for orbit,F in zip(trial.orbits,updated):
                mat=F.detach().cpu().numpy()
                D,R=w594_polardecompose(mat)
                if D is None or max(abs(np.linalg.eigvalsh(D)-1))>strain_limit+1e-7:
                    valid=False;break
                orbit.F=mat
            if not valid:continue
            trial._rebuild()
            try:sample=v61_snapshot(trial,chem,'fixed-sharing-autograd')
            except (RuntimeError,ValueError,FloatingPointError):continue
            score=continuous_candidate_objective(sample)+2*sample['oo_overlap_penalty']
            if round_best is None or score<round_best[0]:round_best=(score,sample)
            if score<winner_score:winner,winner_score=sample,score
        if round_best is None:break
        work=round_best[1]['oct'].copy()
    winner['chem_feedback_steps']=int(steps)*int(rounds)
    return winner

def v61_probe_score(item,oo_weight=4.):
    """Do not reward/punish physical oxygen SG or the sharing graph SG."""
    r=item['ans']
    return (float(continuous_candidate_objective(r))+
            float(oo_weight)*float(r.get('oo_overlap_penalty',0.))+
            8.*max(0.,float(r.get('sharing_max_A',0.))-1.)**2)

def v61_certificate(result,a):
    """Only TiO6 symmetry, coherent bound, triplets and final O shift are hard."""
    octas=result['oct']
    problems=[]
    registry=getattr(octas,'sharing_registry',None)
    if registry is None:problems.append('missing TiO6 sharing policy')
    else:
        trip=np.asarray(registry.trip,dtype=int)
        if not np.array_equal(trip,np.asarray(result['trip'],dtype=int)):
            problems.append('result graph differs from polyhedron policy')
        for orbit,links in zip(octas.orbits,registry.orbit_links):
            if getattr(orbit,'sharing_policy',None)!=links:
                problems.append('orbit sharing policy drift')
        oxygen,_,bad,shift=v61_derive_oxygen(octas)
        if bad:problems.append('duplicate Ti image in physical oxygen triplet')
        delta=minimum_image_delta(oxygen,result['oxygen'],octas.cell,octas.inv)
        if np.max(np.linalg.norm(delta,axis=1))>1e-5:
            problems.append('physical O not derived from TiO6 arm centroids')
        if np.max(shift)>a.max_o_move+1e-7:
            problems.append(f'max O shift {np.max(shift):.4f} > {a.max_o_move:.4f} A')
    symmetry=v61_ideal_symmetry_error(octas)
    if symmetry>1e-4:problems.append(f'TiO6 Wyckoff symmetry drift {symmetry:.4g} A')
    strain=coherent_strain_extrema(octas)
    if strain>a.strain_max+1e-5:
        problems.append(f'TiO6 coherent strain {strain:.4g} > {a.strain_max:.4g}')
    return dict(ok=not problems,reason='; '.join(problems),
                polyhedron_symmetry_error_A=float(symmetry),
                strain_max=float(strain),max_shift_A=float(result.get('sharing_max_A',float('inf'))))

def v61_final_candidates(item,a):
    answers=[('probe',item['ans'])]
    if a.refine_scf_cycles>0:
        try:
            refined=v61_scf(item['ans']['oct'],item['chem_probe'],
                            item['mode_options'],steps=a.refine_scf_cycles,
                            center_mix=a.center_scf_mix)
            answers.append(('refine_scf',refined))
            if a.feedback_steps>0 and a.feedback_rounds>0:
                answers.append(('coherent_feedback',v61_feedback(
                    refined,item['chem_probe'],steps=a.feedback_steps,
                    rounds=a.feedback_rounds,strain_limit=a.strain_max)))
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            log('    v61 refinement warning: '+type(exc).__name__+': '+str(exc))
    return answers

def v61_audit_item(result,item,a,archive,full_ops,phase,finalist):
    """Archive only if TiO6 is symmetry-correct and max arm shift <= 1 A.

    Full crystal SG is OBSERVED, not required to match the Ti-WP SG.
    """
    octas=result['oct']
    if len(octas.ti)!=a.formula_units or len(result['oxygen'])!=2*a.formula_units:
        raise ValueError('Wrong Ti:O composition')
    cert=v61_certificate(result,a)
    if not cert['ok']:
        return None,'invalid_ti_polyhedron_or_shift',cert
    chem=item['chem_full']
    chem.ti=torch.as_tensor(octas.ti,dtype=DTYPE)
    chem.cell=torch.as_tensor(octas.cell,dtype=DTYPE)
    record=ab59_record(result,chem,'v61',item['sg'],item['wp'],
                       item['basin'],finalist,item['metal_loss'],a)
    # Graph and physical-O SG are diagnostic only, NOT acceptance conditions.
    graph_bad,graph_total,arm_sym=w594_sharing_graph_audit(
        octas,result['oxygen'],result['trip'])
    if arm_sym>1e-4:
        return None,'broken_ideal_tio6_group_symmetry',dict(
            ok=False,reason=f'ideal TiO6 arm group error {arm_sym:.4g} A')
    oxygen_sym=w594_group_symmetry_error(result['oxygen'],octas.cell,full_ops)
    record.update(v61_representation_ok=True,v61_representation_issue='',
                  stage=phase,source=item['source'],move=item['move'],
                  target_sg=int(item['sg']),Ti_WP_token=str(tuple(item['wp'])),
                  independent_TiO6_orbits=octas.n_independent(),
                  Ti_WP_symmetry_error_A=float(cert['polyhedron_symmetry_error_A']),
                  TiO6_arm_group_error_A=float(arm_sym),
                  O_full_symmetry_error_A=float(oxygen_sym),
                  sharing_graph_violations=int(graph_bad),
                  sharing_graph_tested=int(graph_total),
                  post_scf_eigenstrain_max=float(cert['strain_max']),
                  min_OO_A=float(result['min_oo_A']),
                  OO_overlap_penalty=float(result['oo_overlap_penalty']),
                  sharing_max_A=float(result['sharing_max_A']),
                  sharing_p95_A=float(result['sharing_p95_A']),
                  sharing_policy_type='orbit-linked-fixed-topology',
                  source_is_reference_reconstruction=False,
                  O_Wyckoff_variables=0)
    converted=v60_archive_row(result,record,a,graph_bad)
    converted["_v64_state"]=octas  # Archive stores the full authoritative WP state.
    name=f'WP{abs(str_seed(0,item["wp"]))%100000:05d}_C{item["metal_cycle"]}' \
         f'_B{item["basin"]:02d}_F{finalist:02d}_{phase}'
    status,_=archive.consider(converted,name,item['sg'],item['wp'],phase)
    record['archive_result']=status
    record['rank_score']=converted['q']['rank_score']
    return record,status,None

def v61_parse(argv=None):
    """Preserve v60 CLI and sharding, but apply the v61 physical invariant."""
    supplied=list(sys.argv[1:] if argv is None else argv)
    pre=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    pre.add_argument('--sharing-trials',type=int,default=None,
                     help='Distinct LCAO sharing policies sampled per TiO6 entrance')
    pre.add_argument('--topology-jitter',type=float,default=.20,
                     help='Cartesian jitter on LCAO center hypotheses (A)')
    pre.add_argument('--max-o-shift',type=float,default=1.0,
                     help='Max displacement from any ideal TiO6 arm to shared O (A)')
    pre.add_argument('--probe-shortlist',type=int,default=3,
                     help='How many distinct sharing policies per entrance receive probe SCF (before optional quality rescue)')
    pre.add_argument('--quality-rescue',action=argparse.BooleanOptionalAction,default=True,
                     help='Probe a second independent policy if a one-policy shortlist would discard it')
    pre.add_argument('--preprobe-shift-limit',type=float,default=None,
                     help='Cheap pre-SCF maximum arm shift, with buffer above final 1 A')
    pre.add_argument('--site-preflight',action=argparse.BooleanOptionalAction,default=True,
                     help='Pretest site-compatible TiO6 modes once per Ti framework')
    pre.add_argument('--preflight-tries',type=int,default=8)
    pre.add_argument('--novelty-patience',type=int,default=6,
                     help='Stop after this many entrances without a new physical family, once a promising family exists; 0 disables')
    pre.add_argument('--novelty-min-entrances',type=int,default=8)
    pre.add_argument('--novelty-ti-loss',type=float,default=.15)
    pre.add_argument('--novelty-quality-delta',type=float,default=.005,
                     help='Reset novelty patience if an existing physical family improves this much in probe score')
    pre.add_argument('--family-volume-fraction',type=float,default=.05)
    pre.add_argument('--family-rms',type=float,default=.12,
                     help='Radial-fingerprint RMS tolerance (A) for structural-family merging')
    pre.add_argument('--family-max',type=float,default=.32,
                     help='99th-percentile fingerprint tolerance (A) for family merging')
    ours,legacy=pre.parse_known_args(supplied)
    if '--require-full-o-symmetry' in legacy or '--strict-v596-archive' in legacy:
        raise ValueError('v61 enforces TiO6 symmetry only; oxygen SG is a measured output')
    a=v60_parse(legacy)
    a.strict_v596_archive=False
    a.require_full_o_symmetry=False
    if a.search_mode=='full':
        if '--probe-scf-cycles' not in legacy:a.probe_scf_cycles=2
        if '--beam-generations' not in legacy:a.beam_generations=1
        if '--refine-scf-cycles' not in legacy:a.refine_scf_cycles=5
    a.sharing_trials=(ours.sharing_trials if ours.sharing_trials is not None
                      else (6 if a.search_mode=='full' else 3))
    a.topology_jitter=float(ours.topology_jitter)
    a.max_o_move=float(ours.max_o_shift)
    a.probe_shortlist=int(ours.probe_shortlist)
    a.quality_rescue=bool(ours.quality_rescue)
    a.preprobe_shift_limit=(float(ours.preprobe_shift_limit) if ours.preprobe_shift_limit is not None
                            else 1.40*a.max_o_move)
    a.family_rms=float(ours.family_rms)
    a.family_max=float(ours.family_max)
    a.family_volume_fraction=float(ours.family_volume_fraction)
    a.site_preflight=bool(ours.site_preflight)
    a.preflight_tries=int(ours.preflight_tries)
    a.novelty_patience=int(ours.novelty_patience)
    a.novelty_min_entrances=int(ours.novelty_min_entrances)
    a.novelty_ti_loss=float(ours.novelty_ti_loss)
    a.novelty_quality_delta=float(ours.novelty_quality_delta)
    a.max_physical_arm_residual=a.max_o_move
    if a.sharing_trials<1 or a.topology_jitter<0 or a.max_o_move<=0:
        raise ValueError('sharing-trials>=1, topology-jitter>=0, max-o-shift>0 required')
    if (a.probe_shortlist<1 or a.preprobe_shift_limit<a.max_o_move or
        a.family_rms<=0 or a.family_max<=0 or a.family_volume_fraction<=0 or
        a.preflight_tries<1 or a.novelty_patience<0 or
        a.novelty_min_entrances<1 or a.novelty_ti_loss<=0 or
        a.novelty_quality_delta<0):
        raise ValueError('Require probe-shortlist>=1, preprobe-shift-limit>=max-o-shift, and positive family tolerances')
    return a

def v61_self_test():
    a=v61_parse(['--space-groups','87','--max-o-shift','1.0'])
    assert a.max_o_move==1.0 and not a.require_full_o_symmetry
    assert not a.strict_v596_archive and a.sharing_trials>=1
    assert a.probe_shortlist==3 and a.preprobe_shift_limit>=a.max_o_move
    log('V61.3 SELF-TEST PASSED: only TiO6 WP symmetry constrained; O-SG free; '
        'linked fixed sharing policy; 1.0 A max O shift')


# ==========================================================================
# JULIETTE v61.1: inexpensive physical-family screening and search-funnel audit
# No change to the v59.6/v61 TiO6 orbit geometry, symmetry, SCF, or chemistry.
# ==========================================================================

def v611_fingerprint(ti,oxygen,cell,neighbors=8):
    """Origin/ordering/basis-insensitive local radial fingerprint, Ti and O.

    Species-resolved periodic neighbor shells; oxygen positions are computed
    from the fixed TiO6 sharing registry and are NEVER trainable coordinates.
    This is a conservative *family* prefilter, not a proof of isomorphism.
    """
    cell=np.asarray(cell,float)
    coords=(np.asarray(ti,float)%1,np.asarray(oxygen,float)%1)
    if any(len(x)==0 for x in coords):raise ValueError('Empty species in fingerprint')
    translations=np.array(list(itertools.product((-2.,-1.,0.,1.,2.),repeat=3)),float)
    views=[(x[None,:,:]+translations[:,None,:]).reshape(-1,3)@cell for x in coords]
    trees=[cKDTree(v) for v in views]
    shells=[]
    for si,orig in enumerate(coords):
        cart=orig@cell
        for tj,target in enumerate(coords):
            # Include enough periodic images to handle a nearest same-atom
            # image and any repeated boundary neighbors. Ignore only exact self.
            count=min(int(neighbors)+8,len(views[tj]))
            distances=trees[tj].query(cart,k=count)[0]
            if distances.ndim==1:distances=distances[:,None]
            block=[]
            for row in distances:
                usable=row[row>1.e-5]
                if len(usable)<neighbors:
                    usable=np.pad(usable,(0,neighbors-len(usable)),constant_values=1.e3)
                block.append(usable[:neighbors])
            # Sorting separately for each shell rank tolerates Ti/O relabeling.
            shell=np.sort(np.asarray(block),axis=0).T.reshape(-1)
            shells.append(shell)
    volume=float(abs(np.linalg.det(cell))/len(coords[0]))
    return volume,np.concatenate(shells)

def v611_same_family(a,b,rms_tol=.055,max_tol=.16,volume_fraction=.025):
    """Conservative family equivalence; no orientation or O-label dependence."""
    av,ad=a;bv,bd=b
    if len(ad)!=len(bd) or abs(av-bv)>volume_fraction*max(av,bv):return False
    delta=np.abs(ad-bd)
    return bool(np.sqrt(np.mean(delta*delta))<=rms_tol and
                np.quantile(delta,.99)<=max_tol)

def v611_item_fingerprint(item):
    fp=item.get('_v611_family')
    if fp is None:
        x=item['ans'];oc=x['oct']
        fp=v611_fingerprint(oc.ti,x['oxygen'],oc.cell)
        item['_v611_family']=fp
    return fp

def v611_diverse(items,capacity,oo_weight=4.,rms_tol=.055,max_tol=.16,volume_fraction=.025):
    """Pre-refinement structural diversity, not mutable arm-label diversity."""
    if capacity<1:return [],0
    ranked=sorted((x for x in items if np.isfinite(v61_probe_score(x,oo_weight))),
                  key=lambda x:v61_probe_score(x,oo_weight))
    kept=[];fingerprints=[];duplicates=0
    for item in ranked:
        fp=v611_item_fingerprint(item)
        if any(v611_same_family(fp,old,rms_tol,max_tol,volume_fraction) for old in fingerprints):
            duplicates+=1;continue
        kept.append(item);fingerprints.append(fp)
        if len(kept)>=capacity:break
    return kept,duplicates

class V611Archive(V60Archive):
    """Final archive: collapse physically equivalent families before matcher.

    pymatgen's strict structural matcher remains the second, authoritative
    comparison for candidates not merged by this conservative radial screen.
    """
    def __init__(self,*args,family_rms=.055,family_max=.16,
                 family_volume_fraction=.025,**kwargs):
        super().__init__(*args,**kwargs)
        self.family_rms=float(family_rms)
        self.family_max=float(family_max)
        self.family_volume_fraction=float(family_volume_fraction)

    def consider(self,row,name,sg,wp,stage):
        ti=np.asarray(row['ti'],float)
        oxygen=np.asarray(row['answer']['oxygen'],float)
        cell=np.asarray(row['cell'],float)
        if ti.shape!=(self.expected_ti,3) or oxygen.shape!=(2*self.expected_ti,3) or \
           not np.isfinite(ti).all() or not np.isfinite(oxygen).all() or \
           not np.isfinite(cell).all():
            return super().consider(row,name,sg,wp,stage)
        newfp=v611_fingerprint(ti,oxygen,cell)
        for i,old in enumerate(self.records):
            oldfp=old.get('_v611_family')
            if oldfp is None:
                oldfp=v611_fingerprint(old['ti'],old['oxygen'],old['cell'])
                old['_v611_family']=oldfp
            if not v611_same_family(newfp,oldfp,self.family_rms,self.family_max,
                                    self.family_volume_fraction):
                continue
            if old['sg']!=int(sg):
                self.stats['cross_sg_family_duplicate']+=1
                return 'cross_sg_family_duplicate',old
            replacement={'sg':int(sg),'wp':tuple(wp),'name':str(name),
                         'stage':stage,'q':dict(row['q']),
                         'cell':cell.copy(),'ti':ti.copy(),
                         'oxygen':oxygen.copy(),'_v611_family':newfp}
            if self.key(replacement)<self.key(old):
                replacement['structure']=self.structure_factory(row)
                self.records[i]=replacement
                self.stats['improved_family_duplicate']+=1
                return 'improved_family_duplicate',replacement
            self.stats['family_duplicate']+=1
            return 'family_duplicate',old
        return super().consider(row,name,sg,wp,stage)

def v613_preprobe_select(proposals,limit,quality_rescue=True):
    """Keep the best cheap geometry, with one independent backup when budget=1.

    The LCAO-proxy score does NOT measure Ti/O chemistry. A single-policy
    shortlist can therefore remove the best chemical basin irrevocably. If
    the budget is one, rescue the unjittered baseline if it is distinct;
    otherwise rescue the next-best distinct policy. All candidates are still
    fixed TiO6 sharing registries with no oxygen-coordinate variables.

    Returns selected proposals and number of additional SCF evaluations.
    """
    if limit<1:raise ValueError('probe-shortlist must be >= 1')
    if not proposals:return [],0
    ranked=sorted(proposals,key=lambda x:x[0])
    selected=list(ranked[:limit])
    extra=0
    if quality_rescue and limit==1 and len(ranked)>1:
        baseline=next((p for p in proposals if str(p[4]).endswith('_trial0')),None)
        backup=(baseline if baseline is not None and baseline[2]!=selected[0][2]
                else next((p for p in ranked[1:] if p[2]!=selected[0][2]),None))
        if backup is not None:
            selected.append(backup)
            extra=1
    return selected,extra

def v612_site_mode_preflight(template, raw, cell, lib, proposals, full_ops,
                             strain_max, tries=8, seed=0):
    """Precompute viable modes for each Ti Wyckoff site, ONCE per metal basin.

    No new geometrical model: this constructs the existing WyckoffPolyhedron.
    A failed mode is not retried in every entrance.  A mode that passed once is
    safe under the same cell, Ti Wyckoff coordinates, and SG stabilizer.
    The returned proposals retain only whole-octahedron-compatible modes.
    """
    raw=np.asarray(raw,float).reshape(-1)
    offset=len(template.lattice_keys)
    site_offset=0
    usable=[list(x) for x in proposals]
    diagnostics=collections.Counter()
    for bi,wp in enumerate(template.blocks):
        dof=int(wp['dof'])
        u=1./(1.+np.exp(-np.clip(raw[offset:offset+dof],-25.,25.)))
        offset+=dof
        old=list(proposals[site_offset])
        if dof==3:
            # General positions have no nontrivial stabilizer to prefilter.
            # Their first pose is valid, so skip mode-by-mode construction.
            diagnostics['general_orbits_skipped']+=1
            site_offset+=len(wp['rot'])
            continue
        viable=[]
        for mode in sorted(set(int(v[1]) for v in old)):
            poses=[p for p in old if int(p[1])==mode]
            # The existing set_pose searches 24 site-axis alignments plus a
            # few random rotations, so a single deterministic test is useful.
            try:
                WyckoffPolyhedron(wp,cell,full_ops,lib,mode,poses[0][2],
                    max_strain=strain_max,u=u,global_offset=site_offset,
                    site_attempts=tries,seed=str_seed(seed,'site',bi,mode))
                viable.append(mode)
                diagnostics['compatible_modes']+=1
            except WyckoffIncompatible:
                diagnostics['incompatible_modes']+=1
        if not viable:
            diagnostics['infeasible_orbits']+=1
            return None,dict(diagnostics)
        filtered=[p for p in old if int(p[1]) in viable]
        for i in range(site_offset,site_offset+len(wp['rot'])):
            # Every orbit member must use a viable independent whole-TiO6
            # mode. Preserve each member's own orientation hints and scores.
            options=[p for p in proposals[i] if int(p[1]) in viable]
            usable[i]=options or filtered
        site_offset+=len(wp['rot'])
    return usable,dict(diagnostics)

def v61_ti_orbit_search(ti,cell,lib,chem,template,raw,full_ops,a,seed,info):
    """v61.1: broad TiO6/policy discovery, CHEAP shortlist, then probe SCF.

    The v61 fixed-topology and local Wyckoff-TiO6 SCF kernels are unchanged.
    Oxygen is derived from arms on demand and its full SG is not constrained.
    """
    wall_start=time.perf_counter()
    clock=collections.Counter()
    counts=collections.Counter()
    t0=time.perf_counter()
    modes=mode_shortlist(ti,cell,lib,min(a.modes_per_ti,len(lib['Ti'])))
    proposals=framework_orientation_seeds(
        ti,cell,lib,modes,np.empty((0,3),float),
        proposals_per_ti=a.octahedral_proposals)
    clock['orientations_s']+=time.perf_counter()-t0
    if a.site_preflight:
        t0=time.perf_counter()
        proposals,mode_diag=v612_site_mode_preflight(
            template,raw,cell,lib,proposals,full_ops,
            a.strain_max,tries=a.preflight_tries,seed=seed)
        clock['site_preflight_s']+=time.perf_counter()-t0
        if proposals is None:
            return [],dict(fresh=0,mutations=0,valid=0,diverse=0,errors=[],
                topology_rejected=0,rejection_reasons={},
                timings=dict(clock),funnel=dict(mode_diag,
                preflight_infeasible_basins=1))
    else:
        mode_diag={}
    candidates=[];errors=[];entrances=0;mutations=0
    rejected=collections.Counter()
    rng=np.random.default_rng(str_seed(seed,'v61-sharing-policies'))
    for key,n in mode_diag.items():counts[key]+=n
    # Track both physical-family novelty AND improvements inside an existing
    # family. A score-improving basin is not a plateau simply because its
    # physical structure matches a previously explored family.
    novelty=[];novelty_entrance=0;good_novelty=False
    entrance_curve=[];sharing_stage=collections.defaultdict(collections.Counter)

    def finish_entrance(k,started,prior,valid):
        entrance_curve.append(dict(entrance_index=k+1,
            wall_s=round(time.perf_counter()-started,4),
            probes=counts['probe_scf_runs']-prior[0],
            new_families=counts['novel_physical_families']-prior[1],
            improved_families=counts['improved_existing_family']-prior[2],
            valid=int(valid)))

    def prepare(octas,source,label,retain_existing=False,centers=None):
        t0=time.perf_counter()
        try:
            work=octas.copy()
            if not retain_existing or not hasattr(work,'sharing_registry'):
                if centers is None:
                    peaks=coherent_lcao_proposal_field(octas,np.empty((0,3),float))
                    centers=_select_provisional_centers(octas,peaks,max(.5,chem.guard))
                    counts['lcao_center_builds']+=1
                # LCAO bases are shared between trials. The first proposal is
                # the ORIGINAL, unjittered v59.6 proposal, not a random shift.
                v61_attach_lcao_policy(work,centers)
            counts['sharing_proposed']+=1
            if v61_ideal_symmetry_error(work)>1e-4:
                raise V61InvalidSharing('TiO6 orbit violates parent SG')
            oxygen,_,bad,shift=v61_derive_oxygen(work)
            if bad:
                raise V61InvalidSharing('Repeated Ti image in shared oxygen')
            max_shift=float(np.max(shift))
            if max_shift>a.preprobe_shift_limit:
                raise V61InvalidSharing('Preprobe max arm displacement too large')
            rep,doo=w596_oo_penalty(oxygen,work.cell,chem.guard)
            # Only geometry is used here: no Chemistry.score and no SCF.
            proxy=2.*(max_shift/a.max_o_move)**2+4.*rep
            key=tuple(sorted(work.sharing_registry.groups))
            return proxy,work,key,source,label,max_shift,doo
        finally:
            clock['lcao_s']+=time.perf_counter()-t0

    def probe(prepared):
        nonlocal mutations
        _,work,_,source,label,_,_=prepared
        t0=time.perf_counter()
        counts['probe_scf_runs']+=1
        try:
            snap=v61_scf(work,chem,modes,steps=a.probe_scf_cycles,
                         center_mix=a.center_scf_mix)
        finally:
            clock['probe_scf_s']+=time.perf_counter()-t0
        if not np.isfinite(v61_probe_score(dict(ans=snap),a.oo_probe_weight)):
            raise V61InvalidSharing('Nonfinite chemistry or sharing score')
        item=e593_new_item(snap,info,source,label)
        item['mode_options']=modes
        candidates.append(item)
        return item

    for k in range(a.entrances):
        ent_t0=time.perf_counter()
        ent_prior=(counts['probe_scf_runs'],counts['novel_physical_families'],
                   counts['improved_existing_family'])
        counts['entrance_attempts']+=1
        t0=time.perf_counter()
        try:
            octas=WyckoffOctahedra(
                template,raw,cell,lib,proposals,full_ops,
                seed=0 if k==0 else str_seed(seed,'TiWP',k),
                max_strain=a.strain_max,site_attempts=a.site_tries)
            entrances+=1
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            counts['entrance_failed']+=1
            if 'WyckoffIncompatible' in type(exc).__name__:
                counts['site_incompatible_entrances']+=1
            errors.append(f'entrance{k}: {type(exc).__name__}: {exc}')
            clock['entrances_s']+=time.perf_counter()-t0
            finish_entrance(k,ent_t0,ent_prior,False)
            continue
        clock['entrances_s']+=time.perf_counter()-t0
        proposals_local=[];seen_policies=set()
        # Build the expensive TiO6 vertex field and LCAO base centers just
        # ONCE per entrance, not once for each jittered sharing trial.
        t0=time.perf_counter()
        try:
            peaks=coherent_lcao_proposal_field(octas,np.empty((0,3),float))
            base=_select_provisional_centers(octas,peaks,max(.5,chem.guard))
            counts['lcao_center_builds']+=1
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
            counts['lcao_center_failures']+=1
            errors.append(f'entrance{k}.lcao: {type(exc).__name__}: {exc}')
            clock['lcao_s']+=time.perf_counter()-t0
            finish_entrance(k,ent_t0,ent_prior,False)
            continue
        clock['lcao_s']+=time.perf_counter()-t0
        for trial in range(a.sharing_trials):
            trial_stats=sharing_stage[trial+1]
            trial_stats['attempts']+=1
            try:
                counts['sharing_trials']+=1
                start=(base if trial==0 else base+rng.normal(
                    0.,a.topology_jitter,base.shape))
                result=prepare(octas,'independent-polyhedral-LCAO' if trial==0
                    else 'independent-polyhedral-alternate',
                    f'entrance{k}_trial{trial}',centers=start)
                if result[2] in seen_policies:
                    trial_stats['duplicates']+=1
                    counts['duplicate_sharing_partition']+=1
                    continue
                seen_policies.add(result[2])
                trial_stats['unique']+=1
                proposals_local.append(result)
            except V61InvalidSharing as exc:
                trial_stats['invalid']+=1
                rejected[str(exc)]+=1
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                trial_stats['invalid']+=1
                errors.append(f'entrance{k}.trial{trial}: {type(exc).__name__}: {exc}')
        selected,rescued=v613_preprobe_select(
            proposals_local,a.probe_shortlist,a.quality_rescue)
        counts['quality_rescue_probes']+=rescued
        counts['preprobe_shortlisted_out']+=len(proposals_local)-len(selected)
        counts['preprobe_selected']+=len(selected)
        for proposal in selected:
            try:
                idx=int(str(proposal[4]).rsplit('_trial',1)[1])+1
                sharing_stage[idx]['selected']+=1
            except (IndexError,ValueError):pass
        first_policy_quality=None
        for ci,candidate in enumerate(selected):
            try:
                item=probe(candidate)
                if ci==0:
                    first_policy_quality=float(v61_probe_score(item,a.oo_probe_weight))
                elif rescued and ci==len(selected)-1 and first_policy_quality is not None:
                    if v61_probe_score(item,a.oo_probe_weight)<first_policy_quality:
                        counts['quality_rescue_wins']+=1
                # Novelty is measured on physical Ti/O geometry, not discrete
                # mode/arm identifiers. This is only for early stopping, NOT
                # for imposing oxygen Wyckoff symmetry.
                fp=v611_item_fingerprint(item)
                quality=float(v61_probe_score(item,a.oo_probe_weight))
                existing=next((i for i,(prior,_) in enumerate(novelty)
                    if v611_same_family(fp,prior,a.family_rms,
                        a.family_max,a.family_volume_fraction)),None)
                if existing is not None:
                    counts['probe_family_duplicates']+=1
                    previous=novelty[existing][1]
                    if quality<previous:
                        novelty[existing]=(fp,quality)
                        counts['improved_existing_family']+=1
                        if previous-quality>=a.novelty_quality_delta:
                            novelty_entrance=k+1
                            counts['plateau_resets_on_quality']+=1
                else:
                    novelty.append((fp,quality))
                    novelty_entrance=k+1
                    counts['novel_physical_families']+=1
                    snap=item['ans']
                    if (float(np.max(snap['details']['localTi']))<=a.novelty_ti_loss
                        and snap['min_oo_A']>=chem.guard
                        and snap['sharing_max_A']<=a.max_o_move):
                        good_novelty=True
                        counts['promising_families']+=1
            except V61InvalidSharing as exc:rejected[str(exc)]+=1
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                errors.append(f'entrance{k}.probe: {type(exc).__name__}: {exc}')
        finish_entrance(k,ent_t0,ent_prior,True)
        if (a.novelty_patience>0 and good_novelty
                and k+1>=a.novelty_min_entrances
                and k+1-novelty_entrance>=a.novelty_patience):
            counts['novelty_plateau_stops']+=1
            counts['entrances_avoided']+=a.entrances-k-1
            break

    # Mutation is a separate *discrete* proposal branch. It cannot change the
    # linked sharing policy inside one optimization trajectory.
    screen_t0=time.perf_counter()
    beam,_=v611_diverse(candidates,a.beam_width,a.oo_probe_weight,
                        a.family_rms,a.family_max,a.family_volume_fraction)
    clock['family_screen_s']+=time.perf_counter()-screen_t0
    counts['initial_beam']=len(beam)
    for gen in range(a.beam_generations):
        children=[]
        for pi,parent in enumerate(beam):
            for mi in range(a.mutations_per_parent):
                try:
                    t0=time.perf_counter()
                    moved,kind=parent['ans']['oct'].mutate(
                        rng,a.rotation_amplitude,proposals,modes)
                    clock['mutate_s']+=time.perf_counter()-t0
                    keep_policy=(mi%2==0)
                    pref=prepare(moved,'TiWP-coherent-pose-mutation' if keep_policy
                                 else 'TiWP-link-alternate',
                                 f'g{gen}_p{pi}_{mi}_{kind}',retain_existing=keep_policy)
                    child=probe(pref)
                    children.append(child);mutations+=1
                except V61InvalidSharing as exc:rejected[str(exc)]+=1
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    errors.append(f'mutation{gen}.{pi}.{mi}: {type(exc).__name__}: {exc}')
        screen_t0=time.perf_counter()
        beam,dups=v611_diverse(beam+children,a.beam_width,a.oo_probe_weight,
                              a.family_rms,a.family_max,a.family_volume_fraction)
        clock['family_screen_s']+=time.perf_counter()-screen_t0
        counts['mutant_family_collapses']+=dups
    unique=len(set(e593_topology(x) for x in candidates))
    clock['search_total_s']=time.perf_counter()-wall_start
    return candidates,dict(fresh=entrances,mutations=mutations,
                           valid=len(candidates),diverse=unique,errors=errors,
                           topology_rejected=int(sum(rejected.values())),
                           rejection_reasons=dict(rejected),
                           timings=dict(clock),funnel=dict(counts),
                           entrance_curve=entrance_curve,
                           sharing_curve=[dict(trial_index=k,**dict(sharing_stage[k]))
                                          for k in sorted(sharing_stage)])


# v61.5: machine-readable progress events contain final structures, never
# provisional chemistry or mutable worker-local candidate indexes.
V615_EVENT_PREFIX='JULIETTE_PROGRESS_JSON '

def v615_sg_event(sg, archive, wp_count, counts, clocks, errors, out):
    """Final, stable per-SG structure list; no independent oxygen variables."""
    structures=[]
    for rank,rec in enumerate(archive.by_sg(sg),1):
        try:
            final_sg,n_primitive=archive.symmetry_details(rec,archive.symprec)
        except Exception:
            final_sg,n_primitive=None,None
        q=rec['q']
        structures.append(dict(initial_spgr=int(sg),
            final_spgr=final_sg,N_atom_primitive=n_primitive,
            Ti_loss_min=float(q['minTi']),Ti_loss_max=float(q['maxTi']),
            Ti_loss_avg=float(q['meanTi']),N_good_Ti=int(q['goodTi']),
            N_good_O=int(q['goodO']),min_OO_A=(float(q['minOO']) if np.isfinite(float(q.get('minOO',float('nan')))) else None),
            rank_score=float(q['rank_score']),
            sharing_graph_violations=int(q.get('sharing_violations',0)),
            max_O_shift_A=float(q['sharing']),
            cif_name=archive._cif_path(out,rec,rank).name))
    return dict(event='sg_complete',sg=int(sg),wp_tasks=int(wp_count),
                probes=int(counts['probes']),wall_s=float(clocks['sg_wall_s']),
                n_cifs=len(structures),n_errors=sum(int(e.get('sg',-1))==int(sg) for e in errors),
                probe_families=int(counts.get('v62_total_families',0)),
                promising_probe_families=int(counts.get('v62_promising_families',0)),
                expanded_wps=int(counts.get('v63_expanded_wps',0)),
                structures=structures)

def v615_emit_sg_event(event):
    """Send to parent only; do not clutter direct synthetic/debug executions."""
    if os.environ.get('JULIETTE_GPU_CHILD')=='1':
        log(V615_EVENT_PREFIX+json.dumps(event,allow_nan=False,separators=(',',':')))

# ==========================================================================
# JULIETTE v62 — INTEGRATED, SINGLE-PASS BROADENING SATURATION STUDY
# No alternative generator, no repeated test configurations, no oxygen WPs.
# All diagnostic curves come from the same trajectories that generate CIFs.
# ==========================================================================

def v62_sample_sgs(candidates, n, seed):
    """Reproducible crystal-system-stratified random SG sample.

    Selects at least one SG per represented crystal system if n permits;
    the rest are sampled randomly. This is not a claim of uniform physical
    structure coverage; it is a useful diverse cost/feasibility panel.
    """
    values=sorted(set(map(int,candidates)))
    if n<1 or n>=len(values):return values
    groups=((1,2),(3,15),(16,74),(75,142),(143,167),(168,194),(195,230))
    bins=[[v for v in values if lo<=v<=hi] for lo,hi in groups]
    bins=[x for x in bins if x]
    rng=np.random.default_rng(int(seed))
    picked=[]
    if n>=len(bins):
        for b in bins:picked.append(int(rng.choice(b)))
    # Equal weighting among represented crystal systems prevents very large
    # low-symmetry ranges from hiding high-symmetry constraints.
    while len(picked)<n:
        pool=[b for b in bins if any(v not in picked for v in b)]
        if not pool:break
        b=pool[int(rng.integers(len(pool)))]
        remaining=[v for v in b if v not in picked]
        picked.append(int(rng.choice(remaining)))
    return sorted(picked)

def v62_parse(argv=None):
    """A normal JULIETTE run with optional internal 12–16-SG sampling."""
    supplied=list(sys.argv[1:] if argv is None else argv)
    aux=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    aux.add_argument('--study-sgs',type=int,default=0,
        help='Sample N (recommended 12–16) SGs in one normal search; 0 searches supplied SGs')
    aux.add_argument('--study-seed',type=int,default=62026,
        help='Reproducible stratified SG selection, independent of structure seed')
    options,others=aux.parse_known_args(supplied)
    if options.study_sgs<0 or options.study_sgs>230:
        raise ValueError('--study-sgs must be between 0 and 230')
    a=v61_parse(others)
    a.study_sgs=int(options.study_sgs)
    a.study_seed=int(options.study_seed)
    if a.study_sgs:
        original=parse_sgs(a.space_groups)
        selected=v62_sample_sgs(original,a.study_sgs,a.study_seed)
        a.space_groups=','.join(map(str,selected))
        a.study_selected=selected
    else:a.study_selected=parse_sgs(a.space_groups)
    return a

# ===========================================================================
# JULIETTE v63 -- broad WP coverage, progressive search costs.
# The TiO6 geometry, LCAO topology, site constraints and final audits are
# deliberately UNCHANGED. The only additions are candidate scheduling.
# ===========================================================================

def v63_has_flag(argv, name):
    return any(z==name or z.startswith(name+'=') for z in argv)

def v63_parse(argv=None):
    """Select broad, inexpensive screen + small, quality-guided expansion."""
    supplied=list(sys.argv[1:] if argv is None else argv)
    aux=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    aux.add_argument('--progressive-wp',action=argparse.BooleanOptionalAction,default=True,
        help='Screen all selected Ti Wyckoff patterns, then expand a small subset')
    aux.add_argument('--screen-entrances',type=int,default=6)
    aux.add_argument('--screen-sharing-trials',type=int,default=2)
    aux.add_argument('--screen-basins',type=int,default=1)
    aux.add_argument('--screen-metal-cycles',type=int,default=1)
    aux.add_argument('--expand-wps',type=int,default=4,
        help='Total WPs per SG receiving additional cycles, including exploratory failures')
    aux.add_argument('--expand-zero-probe-wps',type=int,default=1,
        help='Reserve up to this many expansion slots for zero-probe WPs')
    aux.add_argument('--wp-diversity-pool',type=int,default=None,
        help='Candidates evaluated by WP diversity selector; defaults to --wp-scan-limit')
    options, rest=aux.parse_known_args(supplied)
    a=v62_parse(rest)
    if a.search_mode=='full':
        # v63 full-mode defaults are bounded for campaign use. Every budget
        # remains an explicit CLI override; previous generations' scientific
        # kernels and acceptance criteria are not changed.
        defaults=dict(wp_per_sg=16,wp_scan_limit=400,
                      basins_per_wp=2,entrances=12,
                      sharing_trials=3,probe_shortlist=1,
                      beam_generations=0,refine_top=3,
                      refine_scf_cycles=4,feedback_steps=0)
        for key,value in defaults.items():
            if not v63_has_flag(supplied,'--'+key.replace('_','-')):
                setattr(a,key,value)
    for name,value in vars(options).items():setattr(a,name,value)
    if a.wp_diversity_pool is None:a.wp_diversity_pool=a.wp_scan_limit
    for field in ('screen_entrances','screen_sharing_trials','screen_basins',
                  'screen_metal_cycles','wp_diversity_pool'):
        if getattr(a,field)<1:raise ValueError(f'--{field.replace("_","-")} must be >= 1')
    if a.expand_wps<0 or a.expand_zero_probe_wps<0:
        raise ValueError('Expansion counts must be nonnegative')
    # Explicit --metal-cycles, --entrances, --basins-per-wp and --sharing-trials
    # continue to cap the screen and expansion sizes.
    a.screen_entrances=min(a.screen_entrances,a.entrances)
    a.screen_sharing_trials=min(a.screen_sharing_trials,a.sharing_trials)
    a.screen_basins=min(a.screen_basins,a.basins_per_wp)
    a.screen_metal_cycles=min(a.screen_metal_cycles,a.metal_cycles)
    return a

def v63_wp_candidate_pool(sg, formula_units, max_orbits, sample_count, seed):
    """Short deterministic prefix + uniform sampled multiset completions.

    Avoid relying only on the first 400 DFS patterns (severe order bias in
    low-symmetry SGs). Dynamic programming counts admissible combinations;
    no oxygen Wyckoff positions or physical site compatibility is introduced.
    The requested pool is a candidate *selection* budget, not a hard cap on
    what can exist crystallographically.
    """
    from pyxtal.symmetry import Group
    from functools import lru_cache
    from random import Random
    group=Group(int(sg))
    allowed=[(i,int(group[i].multiplicity),int(group[i].get_dof()))
             for i in range(len(group))
             if 0<int(group[i].multiplicity)<=formula_units]
    cap=int(sample_count)
    if not allowed:return []
    first=[];seen=set()
    # Reserve one quarter for the traditional deterministic traversal. This
    # preserves SG87's simple familiar WP basins without making all other
    # selected WPs descendants of the same DFS branch.
    deterministic=max(1,min(cap,cap//4))
    for token in itertools.islice(v48_iter_wyckoff(sg,formula_units,max_orbits),deterministic):
        if token not in seen:first.append(token);seen.add(token)
    M=len(allowed)
    orbit_cap=int(max_orbits) if max_orbits else int(formula_units)

    @lru_cache(maxsize=None)
    def count(i, left, room):
        if left==0:return 1
        if i==M or room==0:return 0
        _,m,d=allowed[i]
        bound=min(left//m,room,1 if d==0 else room)
        return sum(count(i+1,left-k*m,room-k) for k in range(bound+1))

    total=count(0,int(formula_units),orbit_cap)
    if not total:return first
    rng=Random(int(seed))
    # Uniform unranking over all valid multisets. This does not enumerate the
    # exponentially many combinations just to obtain a diverse candidate pool.
    def unrank(rank):
        selected=[];left=int(formula_units);room=orbit_cap
        for i,(index,m,d) in enumerate(allowed):
            if left==0:break
            bound=min(left//m,room,1 if d==0 else room)
            for k in range(bound+1):
                ways=count(i+1,left-k*m,room-k)
                if rank<ways:
                    if k:selected.extend([index]*k)
                    left-=k*m;room-=k
                    break
                rank-=ways
            else:raise AssertionError('WP unranking/count mismatch')
        if left:raise AssertionError('Incomplete WP unranking')
        return tuple(selected)

    retries=0
    while len(first)<min(cap,total) and retries<max(200,cap*15):
        retries+=1
        token=unrank(rng.randrange(total))
        if token not in seen:
            seen.add(token);first.append(token)
    # If a tiny combinatorial space and repeated random picks prevented
    # completion, use a bounded DFS fallback. It cannot exceed the pool cap.
    if len(first)<min(cap,total):
        for token in v48_iter_wyckoff(sg,formula_units,max_orbits):
            if token not in seen:seen.add(token);first.append(token)
            if len(first)>=min(cap,total):break
    return first

def v63_select_patterns(tokens, group, limit):
    """Greedy coverage of orbit counts, multiplicities, free DOF and site types.

    Choosing many nearly identical WP tokens is undesirable. Selecting a
    balanced family of *Ti* Wyckoff combinations does not modify chemistry.
    """
    if not tokens:return []
    if not limit or len(tokens)<=limit:return list(tokens)
    # Use true multiplicities and dof, not the index values themselves.
    features=[]
    for token in tokens:
        mult=tuple(sorted(int(group[int(i)].multiplicity) for i in token))
        dofs=tuple(sorted(int(group[int(i)].get_dof()) for i in token))
        features.append((len(token),sum(dofs),mult,dofs,len(set(token))))
    selected=[min(range(len(tokens)),key=lambda j:(features[j][0],features[j][1],tokens[j]))]
    remaining=set(range(len(tokens)))-set(selected)
    def distance(x,y):
        nx,dx,mx,wx,ux=x;ny,dy,my,wy,uy=y
        # Primary preference: different orbit counts and multiplicity makeup.
        return (3.0*abs(nx-ny) + 1.2*abs(dx-dy) +
                2.0*len(set(mx).symmetric_difference(my)) +
                0.5*len(set(wx).symmetric_difference(wy)) +
                0.5*abs(ux-uy))
    while remaining and len(selected)<limit:
        j=max(remaining,key=lambda k:(min(distance(features[k],features[i]) for i in selected),
                                       -features[k][0],-features[k][1],-k))
        selected.append(j);remaining.remove(j)
    return [tokens[i] for i in selected]

def v63_plan(a):
    # Retain the historical pilot pathway and its reproducible known controls.
    # v63 changes the full-search exploration strategy, not pilot physics.
    if a.search_mode!='full':return ab59_plan(a)
    from pyxtal.symmetry import Group
    plan=[];counts={}
    for sg in parse_sgs(a.space_groups):
        group=Group(int(sg))
        candidates=v63_wp_candidate_pool(sg,a.formula_units,a.max_ti_orbits,
                     a.wp_diversity_pool,str_seed(a.seed,'v63-WP-pool',sg))
        chosen=v63_select_patterns(candidates,group,a.wp_per_sg)
        counts[sg]=len(chosen)
        plan.extend((sg,tuple(pattern)) for pattern in chosen)
    return plan,counts

def v63_pick_expansions(rows, capacity, zero_reserve):
    """Quality-guided WP promotion with explicitly reserved exploration.

    No-probe WPs are never described as impossible. If no productive WP
    exists, one zero-probe WP may still receive another metal cycle.
    """
    if capacity<=0 or not rows:return []
    live=[r for r in rows if int(r.get('probes',0))>0]
    missing=[r for r in rows if int(r.get('probes',0))==0]
    def real(v):
        try:return float(v) if np.isfinite(float(v)) else float('inf')
        except (TypeError,ValueError):return float('inf')
    live.sort(key=lambda r:(-int(r.get('promising',0)),
                             -int(r.get('new_families',0)),
                             real(r.get('best_score')), -int(r.get('probes',0)),
                             int(r['index'])))
    missing.sort(key=lambda r:(int(r.get('site_incompatible',0)),
                                real(r.get('metal_loss')),int(r['index'])))
    nreserve=min(int(zero_reserve),len(missing),int(capacity))
    chosen=live[:max(0,capacity-nreserve)]
    chosen+=missing[:nreserve]
    if len(chosen)<capacity:
        # Fill remaining slots only from actually probed patterns. Expanding
        # every zero-probe WP would reproduce the old expensive failure mode.
        used={r['index'] for r in chosen}
        chosen += [r for r in live if r['index'] not in used][:capacity-len(chosen)]
    return chosen

def v62_candidate_family_update(families,item,a,guard):
    """Count physical probe families before pool truncation; best replaces old.

    A new family is *not* automatically chemically usable. Oxygen O-SG and
    sharing-graph SG are not filters; only derived oxygen geometry is scored.
    """
    fp=v611_item_fingerprint(item)
    score=v61_probe_score(item,a.oo_probe_weight)
    snap=item['ans']
    bestti=float(np.max(snap['details']['localTi']))
    usable=bool(bestti<=a.novelty_ti_loss and
                snap['min_oo_A']>=guard and snap['sharing_max_A']<=a.max_o_move)
    for entry in families:
        if v611_same_family(fp,entry['fingerprint'],a.family_rms,
                           a.family_max,a.family_volume_fraction):
            improved=bool(score < entry['score']-1.e-9)
            if improved:
                entry.update(score=score,fingerprint=fp,usable=usable,ti_loss_max=bestti)
            return 'improved' if improved else 'duplicate'
    families.append(dict(fingerprint=fp,score=score,usable=usable,ti_loss_max=bestti))
    return 'new'

def v62_recommendations(wps,cycles,basins,entrances,sharing,sgs):
    """Observational marginal yield, distinguishing novelty from quality wins.

    Recommendations are NOT scientific selection rules. No reduction is
    recommended if the sampling lacks chemically promising probe families.
    """
    def integer(v):return int(v) if v not in (None,'') else 0
    def real(v):return float(v) if v not in (None,'') else 0.0
    positive_sgs={int(r['initial_spgr']) for r in wps
                  if integer(r.get('promising_probe_families'))>0}
    results=[]
    for name,rows,index,quality_key in (
        ('wp-per-sg',wps,'wp_index','improved_best_probe'),
        ('metal-cycles',cycles,'cycle_index','improved_best_probe'),
        ('basins-per-wp',basins,'basin_index','improved_families'),
        ('entrances',entrances,'entrance_index','improved_families')):
        if not rows:continue
        by=collections.defaultdict(collections.Counter)
        for row in rows:
            key=integer(row[index]);bucket=by[key]
            bucket['attempts']+=1
            bucket['probes']+=integer(row.get('probes'))
            bucket['families']+=integer(row.get('new_families'))
            bucket['improvements']+=integer(row.get(quality_key))
            bucket['seconds']+=real(row.get('wall_s'))
        mx=max(by);total=sum(z['probes'] for z in by.values())
        tail=[k for k in by if k>max(1,mx//2)] or [mx]
        f=sum(by[k]['families'] for k in tail)
        imp=sum(by[k]['improvements'] for k in tail)
        p=sum(by[k]['probes'] for k in tail)
        t=sum(by[k]['seconds'] for k in tail)
        if mx<2 or total<8:
            verdict='INCONCLUSIVE: insufficient sampled breadth/probes'
        elif p<6:
            verdict='INCONCLUSIVE: late stages produced too few probes'
        elif f or imp:
            verdict=f'KEEP: late stages yielded {f} novel families and {imp} quality improvements ({t:.1f}s)'
        elif len(positive_sgs)<2:
            verdict='INCONCLUSIVE: fewer than two SGs yielded promising probe families'
        else:
            suggested=max(1,mx//2)
            verdict=(f'TRY --{name} {suggested} in a confirmation run; '
                     f'late stages: 0 new families/quality wins across {p} probes ({t:.1f}s)')
        results.append(dict(parameter=name,max_observed=mx,probes=total,
            promising_sgs=len(positive_sgs),verdict=verdict,
            stages=[dict(index=k,**dict(by[k])) for k in sorted(by)]))
    if sharing:
        by=collections.defaultdict(collections.Counter)
        for row in sharing:
            k=integer(row['trial_index']);z=by[k]
            for field in ('attempts','unique','selected','duplicates','invalid'):
                z[field]+=integer(row.get(field))
        late=[k for k in by if k>1]
        lateunique=sum(by[k]['unique'] for k in late)
        lateattempts=sum(by[k]['attempts'] for k in late)
        results.append(dict(parameter='sharing-trials',max_observed=max(by),
            probes=sum(z['selected'] for z in by.values()),
            promising_sgs=len(positive_sgs),
            verdict=(f'Late policies (>1): {lateunique}/{lateattempts} different partitions; '
                     'not enough evidence to claim new chemical families without probing'),
            stages=[dict(index=k,**dict(by[k])) for k in sorted(by)]))
    return results

def v62_aggregate_study(out):
    """Use worker CSVs consolidated by multi-GPU launcher, or direct output."""
    def load(name):
        path=Path(out)/name
        if not path.is_file():return []
        with path.open(newline='') as f:return list(csv.DictReader(f))
    w,c,b,e,s=(load(f'breadth_{key}.csv') for key in
             ('wp','cycle','basin','entrance','sharing'))
    groups={int(r['initial_spgr']) for r in w}
    report=v62_recommendations(w,c,b,e,s,groups)
    Path(out,'breadth_recommendations.json').write_text(json.dumps(dict(
        space_groups=sorted(groups),wp_samples=len(w),cycle_samples=len(c),basin_samples=len(b),
        entrance_samples=len(e),sharing_samples=len(s),recommendations=report,
        warning=('Single-pass observational yield; no matched-budget reruns. '
                 'Probe families use the generator radial fingerprint, not validated CIF matches. '
                 'Zero-probe patterns cannot establish saturation.')),indent=2))
    log('=== V63 BROADENING SATURATION (observational, all GPUs) ===')
    log(f'  SGs sampled={len(groups)}, WPs={len(w)}, cycles={len(c)}, basins={len(b)}, '
        f'entrances={len(e)}, sharing trial records={len(s)}')
    for r in report:log(f'  {r["parameter"]}: {r["verdict"]}')
    log(f'  Detailed curves: {Path(out)/"breadth_wp.csv"}; '
        f'{Path(out)/"breadth_cycle.csv"}; {Path(out)/"breadth_basin.csv"}; '
        f'{Path(out)/"breadth_entrance.csv"}; '
        f'{Path(out)/"breadth_sharing.csv"}')
    return report

def v61_main(argv=None):
    a=v63_parse(argv)
    if a.self_test:return v64_self_test()
    plan,by_sg=v63_plan(a)  # Diverse Ti Wyckoff patterns
    log('JULIETTE v64 | persistent WP crystal state + coupled Ti/lattice/arm refinement')
    log(f'  search_mode={a.search_mode} SG_shard={a.sg_shard_index}/{a.sg_shard_count} '
        f'SGs_in_shard={len(parse_sgs(a.space_groups))}')
    log('  O = periodic LCAO centroids of TiO6 arms; NO oxygen Wyckoff parameters')
    log('  Database oxygen, database Ti frameworks and matched-v59 seeds: NOT USED')
    log(f'  SGs={len(by_sg)} tasks={len(plan)} FU={a.formula_units} '
        f'patterns={by_sg} output_top={a.max_unique_per_sg}')
    log(f'  METAL raw={a.raw_starts} opt={a.optimized_starts} '
        f'basins={a.basins_per_wp} cycles={a.metal_cycles} '
        f'Adam={a.natural_steps}+{a.learned_steps}')
    log(f'  LCAO entrances={a.entrances} SCF_probe={a.probe_scf_cycles} '
        f'beam={a.beam_width} generations={a.beam_generations} '
        f'mutations/parent={a.mutations_per_parent} '
        f'finalists={a.refine_top} SCF_refine={a.refine_scf_cycles} '
        f'autograd={a.feedback_steps}x{a.feedback_rounds}')
    log(f'  Coherent strain <= {a.strain_max:.3f}; centerSCF={a.center_scf_mix:.3f}; '
        f'O-O probe weight={a.oo_probe_weight}; '
        f'archive weight={a.oo_archive_weight}; '
        f'physical-O-symmetry-filter=OFF; '
        f'sharing_trials={a.sharing_trials}; '
        f'max_arm_to_O={a.max_o_move:.3f} A; final full SG measured, not forced')
    log(f'  v61.3 screening: sharing_shortlist={min(a.probe_shortlist,a.sharing_trials)}/{a.sharing_trials}, '
        f'preprobe_shift_limit={a.preprobe_shift_limit:.2f} A, '
        f'family_RMS={a.family_rms:.3f} A family_p99={a.family_max:.3f} A '
        f'family_volume_fraction={a.family_volume_fraction:.3f} '
        f'site_preflight={a.site_preflight} novelty_patience={a.novelty_patience} '
        f'quality_rescue={a.quality_rescue} '
        f'novelty_quality_delta={a.novelty_quality_delta}')
    if a.plan_only:return dict(plan=plan,counts=by_sg)
    try:
        from pyxtal.symmetry import Group
    except ImportError as exc:
        raise SystemExit('Normal search requires pyxtal, pymatgen and spglib on Jean Zay') from exc
    device=torch.device('cuda:0' if a.device=='auto' and torch.cuda.is_available()
                        else ('cpu' if a.device=='auto' else a.device))
    a._device=device;a._symprec=a.full_sg_audit_tolerance
    global _V50_CONFIG
    _V50_CONFIG=None
    refs=reference_ti_cifs(a.ref)
    statistics=learn_ti_reference(refs,min_k=8,max_k=a.shell_max_k,
                            soap_neighbors=24,soap_radials=5,soap_lmax=4)
    statistics=apply_ti_rule_mode(statistics,infer_ti_rules(refs,statistics),a.ti_rule_mode)
    lib,bounds,guard,_=load_local_chemistry(a.ref,a.shell_max_k,
                                           accept_A=.10,hierarchy='ti-first')
    log(f'  Loaded reference modes: TiO6={len(lib["Ti"])}; '
        f'O descriptors={len(lib["O"])}; O-O guard={float(guard):.3f} A')
    objective=TiObjective(statistics,argparse.Namespace(image_radius=2,gram_swaps=8,
        mode_temperature=.075,soap_weight=.05,ti_soap_pca='audit'),device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    archive=V611Archive(capacity=a.max_unique_per_sg,expected_ti=a.formula_units,
                       symprec=a._symprec,family_rms=a.family_rms,
                       family_max=a.family_max,
                       family_volume_fraction=a.family_volume_fraction)
    records=[];probes=[];errors=[];summaries=[];wp_reports=[];family_pairs=[]
    breadth_wp=[];breadth_cycle=[];breadth_basin=[];breadth_entrance=[];breadth_sharing=[]
    task_index=0
    for sg in parse_sgs(a.space_groups):
        tasks=[wp for sp,wp in plan if sp==sg]
        sg_pool=[];counts=collections.Counter()
        v62_sg_families=[]
        sg_time=time.perf_counter()
        sg_clocks=collections.Counter()
        ops=w594_group_operations(Group(sg))
        # Screen all distinct WP configurations before selecting ANY for
        # expensive expansion. Each expansion uses NEW metal-cycle seeds.
        # It does not repeat screening, and it does not optimize O Wyckoffs.
        progressive=bool(a.progressive_wp and a.search_mode=='full')
        jobs=[(wp,'screen' if progressive else 'full') for wp in tasks]
        screen_results=[]
        selected_expansions=[]
        for wp,budget_phase in jobs:
            run_a=copy.copy(a)
            if budget_phase=='screen':
                run_a.basins_per_wp=a.screen_basins
                run_a.entrances=a.screen_entrances
                run_a.sharing_trials=a.screen_sharing_trials
                cycle_range=range(a.screen_metal_cycles)
            elif budget_phase=='expand':
                run_a.basins_per_wp=a.basins_per_wp
                run_a.entrances=a.entrances
                run_a.sharing_trials=a.sharing_trials
                cycle_range=range(a.screen_metal_cycles,a.metal_cycles)
            else:
                cycle_range=range(a.metal_cycles)
            wp_probe_best=float('inf')
            wp_good=0
            wp_best_metal=float('inf')
            wp_t0=time.perf_counter()
            old_counts=counts.copy()
            old_clocks=sg_clocks.copy()
            v62_wp_family_start=len(v62_sg_families)
            v62_wp_best_start=min((z['score'] for z in v62_sg_families),default=float('inf'))
            task_index+=1
            log(f'[WP {tasks.index(wp)+1}/{len(tasks)} {budget_phase.upper()}] '
                f'SG{sg:03d} Ti_WP={tuple(wp)} independent_TiO6={len(wp)} '
                f'operations={len(ops)} metal_cycles={len(cycle_range)} '
                f'basins={run_a.basins_per_wp} entrances={run_a.entrances} '
                f'sharing={run_a.sharing_trials}')
            try:
                templ=template_from_token(sg,wp,device)
                if templ.n_ti!=a.formula_units:
                    raise RuntimeError('Ti Wyckoff stoichiometry mismatch')
                for cycle in cycle_range:
                    v62_cycle_started=time.perf_counter()
                    v62_cycle_families_before=len(v62_sg_families)
                    v62_cycle_best_before=min((z['score'] for z in v62_sg_families),default=float('inf'))
                    v62_cycle_probes_before=counts['probes']
                    metal_seed=str_seed(a.seed,'v60-Ti-framework',sg,wp,cycle)
                    metal_t0=time.perf_counter()
                    basins=ab59_metal_basins(templ,statistics,objective,run_a,
                                             metal_seed,device)
                    if basins:
                        wp_best_metal=min(wp_best_metal,min(float(b['metal_loss']) for b in basins))
                    sg_clocks['metal_s']+=time.perf_counter()-metal_t0
                    counts['Ti_metal_basins']+=len(basins)
                    for bi,b in enumerate(basins):
                        ti,cell=np.asarray(b['ti'],float),np.asarray(b['cell'],float)
                        with contextlib.redirect_stdout(io.StringIO()):
                            chem=Chemistry(ti,cell,lib,bounds,guard,accept=.10,
                                hierarchy='ti-first',calibration=None,target_o_cn=True)
                        chem.ablation_oxygen_weight=a.feedback_oxygen_weight
                        info=dict(sg=sg,wp=tuple(wp),metal_cycle=cycle,
                                  basin=bi,entrance=-1,metal_loss=b['metal_loss'],
                                  chem_probe=chem,chem_full=chem,ops=ops)
                        pool,diag=v61_ti_orbit_search(ti,cell,lib,chem,templ,
                            b['raw'],ops,run_a,str_seed(a.seed,'v60-LCAO',sg,wp,cycle,bi),info)
                        counts['entrances_tried']+=diag['funnel'].get('entrance_attempts',run_a.entrances)
                        counts['sharing_policy_attempts']+=diag['funnel'].get('sharing_trials',0)
                        counts['entrances_succeeded']+=diag['fresh']
                        counts['probes']+=diag['valid']
                        counts['mutations']+=diag['mutations']
                        counts['topology_rejected']+=diag['topology_rejected']
                        counts['failed_moves']+=len(diag['errors'])
                        for key,val in diag['funnel'].items():counts[key]+=val
                        for reason,n in diag['rejection_reasons'].items():
                            key=('reject_'+re.sub('[^a-z0-9]+','_',reason.lower()).strip('_'))[:90]
                            counts[key]+=int(n)
                        for key,val in diag['timings'].items():sg_clocks[key]+=val
                        v62_before=len(v62_sg_families)
                        v62_basin_improved=0
                        for candidate in pool:
                            state=v62_candidate_family_update(v62_sg_families,candidate,a,guard)
                            v62_basin_improved+=int(state=='improved')
                        breadth_basin.append(dict(initial_spgr=sg,Ti_WP=str(tuple(wp)),
                            wp_index=tasks.index(wp)+1,cycle_index=cycle+1,
                            basin_index=bi+1,probes=len(pool),
                            new_families=len(v62_sg_families)-v62_before,
                            improved_families=v62_basin_improved,
                            wall_s=round(diag['timings'].get('search_total_s',0),4),
                            metal_loss=float(b['metal_loss'])))
                        for entry in diag.get('entrance_curve',[]):
                            breadth_entrance.append(dict(initial_spgr=sg,Ti_WP=str(tuple(wp)),
                                cycle_index=cycle+1,basin_index=bi+1,**entry))
                        for entry in diag.get('sharing_curve',[]):
                            breadth_sharing.append(dict(initial_spgr=sg,Ti_WP=str(tuple(wp)),
                                cycle_index=cycle+1,basin_index=bi+1,**entry))
                        for x in pool:
                            probe_score=v61_probe_score(x,a.oo_probe_weight)
                            wp_probe_best=min(wp_probe_best,probe_score)
                            wp_good+=int(float(np.max(x['ans']['details']['localTi']))<=a.novelty_ti_loss
                                and x['ans']['min_oo_A']>=guard
                                and x['ans']['sharing_max_A']<=a.max_o_move)
                            d=x['ans']['details']
                            probes.append(dict(sg=sg,Ti_WP=str(tuple(wp)),cycle=cycle,
                                metal_basin=bi,source=x['source'],move=x['move'],
                                score=v61_probe_score(x,a.oo_probe_weight),
                                Ti_loss_max=float(np.max(d['localTi'])),
                                O_loss_max=float(np.max(d['localO'])),
                                OO_overlap_penalty=float(x['ans']['oo_overlap_penalty']),
                                min_OO_A=float(x['ans']['min_oo_A']),
                                sharing_max_A=float(x['ans']['sharing_max_A']),
                                N_good_Ti=int(np.sum(d['goodTi'])),
                                N_good_O=int(np.sum(d['goodO'])),
                                Ti_orbits=x['ans']['oct'].n_independent()))
                        screen_t0=time.perf_counter()
                        if pool:
                            min_quality=min(v61_probe_score(x,a.oo_probe_weight) for x in pool)
                            counts['best_raw_probe_score']=min(
                                float(counts.get('best_raw_probe_score',float('inf'))),min_quality)
                        local,dups=v611_diverse(pool,max(3,a.beam_width),
                            a.oo_probe_weight,a.family_rms,a.family_max,a.family_volume_fraction)
                        counts['local_family_duplicates']+=dups
                        sg_pool,dups=v611_diverse(sg_pool+local,a.pool_per_sg,
                            a.oo_probe_weight,a.family_rms,a.family_max,a.family_volume_fraction)
                        counts['sg_pool_family_duplicates']+=dups
                        sg_clocks['family_screen_s']+=time.perf_counter()-screen_t0
                        best=min((v61_probe_score(x,a.oo_probe_weight)
                                  for x in pool),default=float('nan'))
                        log(f'    Ti_basin={bi} entrances={diag["fresh"]}/{run_a.entrances} '
                            f'probes={diag["valid"]} mutations={diag["mutations"]} '
                            f'topology_rejected={diag["topology_rejected"]} '
                            f'sharing_proposed={diag["funnel"].get("sharing_proposed",0)} '
                            f'same_partition={diag["funnel"].get("duplicate_sharing_partition",0)} '
                            f'pre_selected={diag["funnel"].get("preprobe_selected",0)} '
                            f'probe_scf_s={diag["timings"].get("probe_scf_s",0.):.1f} '
                            f'OO_best={min((x["ans"]["min_oo_A"] for x in pool),default=float("nan")):.3f} A '
                            f'best_proxy={best:.5g} pool={len(sg_pool)}')
                        for err in diag['errors'][:a.print_probe_errors]:
                            log(f'      PROBE warning: {err}')
                        if diag['errors']:
                            errors.append(dict(sg=sg,wp=list(wp),cycle=cycle,basin=bi,
                                               errors=diag['errors']))
                    breadth_cycle.append(dict(initial_spgr=sg,Ti_WP=str(tuple(wp)),
                        wp_index=tasks.index(wp)+1,cycle_index=cycle+1,
                        wall_s=round(time.perf_counter()-v62_cycle_started,4),
                        probes=counts['probes']-v62_cycle_probes_before,
                        new_families=len(v62_sg_families)-v62_cycle_families_before,
                        improved_best_probe=int(min((z['score'] for z in v62_sg_families),default=float('inf'))<v62_cycle_best_before-1.e-8),
                        cumulative_families=len(v62_sg_families)))
            except Exception as exc:
                errors.append(dict(sg=sg,wp=list(wp),
                                   error=f'{type(exc).__name__}: {exc}'))
                log(f'    ERROR {type(exc).__name__}: {exc}')
                if not a.continue_on_error:raise
            # Per-WP accounting exposes waste from many independent building
            # centers, which would otherwise be concealed by the SG total.
            wp_delta=counts-old_counts
            wp_sec=time.perf_counter()-wp_t0
            wp_row=dict(initial_spgr=sg,Ti_WP=str(tuple(wp)),
                independent_TiO6=len(wp),wall_s=round(wp_sec,4),
                metal_s=round(sg_clocks['metal_s']-old_clocks['metal_s'],4),
                site_preflight_s=round(sg_clocks['site_preflight_s']-old_clocks['site_preflight_s'],4),
                entrances_s=round(sg_clocks['entrances_s']-old_clocks['entrances_s'],4),
                lcao_s=round(sg_clocks['lcao_s']-old_clocks['lcao_s'],4),
                probe_scf_s=round(sg_clocks['probe_scf_s']-old_clocks['probe_scf_s'],4),
                entrances_attempted=wp_delta['entrances_tried'],
                entrances_succeeded=wp_delta['entrances_succeeded'],
                sharing_trials=wp_delta['sharing_trials'],
                site_preflight_rejects=wp_delta['incompatible_modes'],
                preflight_skipped_basins=wp_delta['preflight_infeasible_basins'],
                center_builds=wp_delta['lcao_center_builds'],
                duplicate_policies=wp_delta['duplicate_sharing_partition'],
                topology_rejected=wp_delta['topology_rejected'],
                probes=wp_delta['probes'],
                probe_family_duplicate=wp_delta['probe_family_duplicates'],
                promising_families=wp_delta['promising_families'],
                plateau_stops=wp_delta['novelty_plateau_stops'],
                entrances_avoided=wp_delta['entrances_avoided'])
            wp_row.update(wp_index=tasks.index(wp)+1,budget_phase=budget_phase,
                best_metal_loss=(wp_best_metal if np.isfinite(wp_best_metal) else None),
                best_phase_probe=(wp_probe_best if np.isfinite(wp_probe_best) else None),
                promising_phase_probes=wp_good,
                new_families=len(v62_sg_families)-v62_wp_family_start,
                cumulative_families=len(v62_sg_families),
                best_probe_score=min((z['score'] for z in v62_sg_families),default=None),
                improved_best_probe=int(min((z['score'] for z in v62_sg_families),default=float('inf'))<v62_wp_best_start-1.e-8),
                promising_probe_families=sum(int(z['usable']) for z in v62_sg_families))
            breadth_wp.append(dict(wp_row))
            wp_reports.append(wp_row)
            log(f'  WP AUDIT SG{sg:03d} WP={tuple(wp)} phase={budget_phase} '
                f'wall={wp_sec:.1f}s '
                f'metal={wp_row["metal_s"]:.1f}s entrances={wp_row["entrances_s"]:.1f}s '
                f'LCAO={wp_row["lcao_s"]:.1f}s '
                f'probes={wp_row["probes"]} family_duplicates={wp_row["probe_family_duplicate"]} '
                f'avoided_entrances={wp_row["entrances_avoided"]} '
                f'new_families={wp_row["new_families"]} '
                f'cumulative_families={wp_row["cumulative_families"]}')
            if budget_phase=='screen':
                screen_results.append(dict(index=tasks.index(wp),wp=wp,
                    probes=wp_delta['probes'],promising=wp_good,
                    new_families=wp_row['new_families'],
                    best_score=wp_probe_best,metal_loss=wp_best_metal,
                    site_incompatible=wp_delta['incompatible_modes']))
                if len(screen_results)==len(tasks):
                    # WP diversity is broad, but deep work is bounded.
                    if a.metal_cycles>a.screen_metal_cycles and a.expand_wps>0:
                        selected_expansions=v63_pick_expansions(screen_results,
                            min(a.expand_wps,len(tasks)),a.expand_zero_probe_wps)
                        jobs.extend((entry['wp'],'expand') for entry in selected_expansions)
                    counts['v63_screened_wps']=len(tasks)
                    counts['v63_expanded_wps']=len(selected_expansions)
                    log(f'  SG{sg:03d} V63 SCREEN: WPs={len(tasks)} '
                        f'probes={sum(x["probes"] for x in screen_results)} '
                        f'promising={sum(x["promising"] for x in screen_results)} '
                        f'expanding={len(selected_expansions)} '
                        f'WP_indices={[x["index"]+1 for x in selected_expansions]} '
                        f'zero_probe_reserve={a.expand_zero_probe_wps}')
            elif budget_phase=='full':
                counts['v63_screened_wps']=len(tasks)
        screen_t0=time.perf_counter()
        finalists,duplicates=v611_diverse(sg_pool,a.refine_top,a.oo_probe_weight,
                                               a.family_rms,a.family_max,a.family_volume_fraction)
        counts['finalist_family_duplicates']+=duplicates
        sg_clocks['family_screen_s']+=time.perf_counter()-screen_t0
        counts['v62_total_families']=len(v62_sg_families)
        counts['v62_promising_families']=sum(int(x['usable']) for x in v62_sg_families)
        raw_best=float(counts.get('best_raw_probe_score',float('nan')))
        pool_best=min((v61_probe_score(x,a.oo_probe_weight) for x in sg_pool),
                      default=float('nan'))
        finalist_best=min((v61_probe_score(x,a.oo_probe_weight) for x in finalists),
                          default=float('nan'))
        counts['best_pool_probe_score']=pool_best
        counts['best_finalist_probe_score']=finalist_best
        if (np.isfinite(raw_best) and np.isfinite(finalist_best)
            and finalist_best>raw_best+1e-8):
            counts['quality_best_lost_after_probe']=1
            log(f'  SG{sg:03d} QUALITY WARNING: probe-best={raw_best:.6g} '
                f'finalist-best={finalist_best:.6g}; shortlist or family screen lost elite')
        if not finalists:
            log(f'  SG{sg:03d} FAILED: no complete TiO6 sharing-policy proposal; no CIF')
        log(f'  SG{sg:03d} FINALISTS={len(finalists)} from {counts["probes"]} probes')
        for i,item in enumerate(finalists):
            finalist_t0=time.perf_counter()
            stages=v61_final_candidates(item,a)
            sg_clocks['finalist_refine_s']+=time.perf_counter()-finalist_t0
            for phase,result in stages:
                try:
                    audit_t0=time.perf_counter()
                    record,status,_=v61_audit_item(result,item,a,archive,ops,phase,i)
                    sg_clocks['audit_archive_s']+=time.perf_counter()-audit_t0
                    if record is None:
                        counts[status]+=1
                        if status=='invalid_v596_building_unit' and counts[status]<=3:
                            log('    STRUCTURAL INVARIANT WARNING: '+_['reason'])
                        continue
                    records.append(record)
                    counts['audits']+=1
                    counts['chemistry_good']+=int(record['chemistry_ok'])
                    counts['gate10']+=int(record['gate010'])
                    counts['OO_guard_clear']+=int(record['OO_overlap_penalty']<1e-8)
                    log(f'    finalist={i+1}/{len(finalists)} {phase} '
                        f'Ti_loss_max={record["Ti_loss_max"]:.4g} '
                        f'O_loss_max={record["O_loss_max"]:.4g} '
                        f'Omin={record["min_OO_A"]:.3f} A '
                        f'Omax={record["max_o_move"]:.3f} A '
                        f'Ti_good={record["n_good_ti"]}/{record["n_ti"]} '
                        f'O_good={record["n_good_o"]}/{record["n_o"]} '
                        f'finalSG={record["measured_full_sg"]} '
                        f'graph={record["sharing_graph_violations"]}/'
                        f'{record["sharing_graph_tested"]} archive={status}')
                except Exception as exc:
                    errors.append(dict(sg=sg,finalist=i,phase=phase,
                                       error=f'{type(exc).__name__}: {exc}'))
                    log(f'    AUDIT ERROR {type(exc).__name__}: {exc}')
                    if not a.continue_on_error:raise
        archive_t0=time.perf_counter()
        archive.write_sg_final(out,sg)
        sg_clocks['cif_write_s']+=time.perf_counter()-archive_t0
        sg_clocks['sg_wall_s']=time.perf_counter()-sg_time
        log(f'  SG{sg:03d} FUNNEL entrances={counts["entrances_succeeded"]}/{counts["entrances_tried"]} '
            f'policy_trials={counts["sharing_trials"]} '
            f'policy_duplicate={counts["duplicate_sharing_partition"]} '
            f'site_incompatible={counts["site_incompatible_entrances"]} '
            f'preflight_modes_rejected={counts["incompatible_modes"]} '
            f'preflight_skipped_basins={counts["preflight_infeasible_basins"]} '
            f'LCAO_center_builds={counts["lcao_center_builds"]} '
            f'novel_families={counts["novel_physical_families"]} '
            f'probe_family_duplicate={counts["probe_family_duplicates"]} '
            f'plateau_stops={counts["novelty_plateau_stops"]} '
            f'plateau_score_resets={counts["plateau_resets_on_quality"]} '
            f'entrances_avoided={counts["entrances_avoided"]} '
            f'preprobe_reject={counts["topology_rejected"]} '
            f'preprobe_shortlisted_out={counts["preprobe_shortlisted_out"]} '
            f'quality_rescue_SCFS={counts["quality_rescue_probes"]} '
            f'SCF_runs={counts["probe_scf_runs"]} '
            f'postprobe={counts["probes"]} '
            f'finalists={len(finalists)} '
            f'physical_family_removed={counts["local_family_duplicates"]+counts["sg_pool_family_duplicates"]+counts["finalist_family_duplicates"]} '
            f'CIFs={archive.count(sg)}')
        rejection_summary={k:v for k,v in counts.items() if k.startswith('reject_') and v}
        log(f'  SG{sg:03d} REJECTIONS {rejection_summary}')
        log(f'  SG{sg:03d} TIMING total={sg_clocks["sg_wall_s"]:.1f}s '
            f'metal={sg_clocks["metal_s"]:.1f}s '
            f'orientations={sg_clocks["orientations_s"]:.1f}s '
            f'site_preflight={sg_clocks["site_preflight_s"]:.1f}s '
            f'entrances={sg_clocks["entrances_s"]:.1f}s '
            f'LCAO_preflight={sg_clocks["lcao_s"]:.1f}s '
            f'probe_SCF={sg_clocks["probe_scf_s"]:.1f}s '
            f'family_screen={sg_clocks["family_screen_s"]:.1f}s '
            f'final_refine={sg_clocks["finalist_refine_s"]:.1f}s '
            f'audit_archive={sg_clocks["audit_archive_s"]:.1f}s '
            f'CIF_write={sg_clocks["cif_write_s"]:.1f}s')
        if archive.count(sg)==0:
            log(f'  SG{sg:03d} NO SOLUTION: no final candidate has exact TiO6 symmetry '
                f'and max shared-O shift <= {a.max_o_move:.3f} A; zero CIFs')
        summaries.append(dict(initial_sg=sg,WP=len(tasks),**dict(counts),
                              **dict(sg_clocks),final_unique=archive.count(sg)))
        e593_write_table(out/'probe_metrics.csv',probes)
        e593_write_table(out/'final_candidates.csv',records)
        e593_write_table(out/'per_sg.csv',summaries)
        e593_write_table(out/'per_wp.csv',wp_reports)
        e593_write_table(out/'breadth_wp.csv',breadth_wp)
        e593_write_table(out/'breadth_cycle.csv',breadth_cycle)
        e593_write_table(out/'breadth_basin.csv',breadth_basin)
        e593_write_table(out/'breadth_entrance.csv',breadth_entrance)
        e593_write_table(out/'breadth_sharing.csv',breadth_sharing)
        e593_write_table(out/'final_family_pairs.csv',family_pairs)
        v615_emit_sg_event(v615_sg_event(sg,archive,len(tasks),counts,sg_clocks,errors,out))
        log(f'  SG{sg:03d} SUMMARY probes={counts["probes"]} '
            f'chemistry_good={counts["chemistry_good"]} '
            f'OO_guard_clear={counts["OO_guard_clear"]} '
            f'final_CIFs={archive.count(sg)}/{a.max_unique_per_sg}')
    archive.write_final(out)
    if os.environ.get('JULIETTE_GPU_CHILD')!='1':v62_aggregate_study(out)
    with (out/'errors.json').open('w') as f:json.dump(errors,f,indent=2)
    with (out/'summary.json').open('w') as f:
        json.dump(dict(config={k:v for k,v in vars(a).items() if not k.startswith('_')},
                       per_sg=summaries,per_wp=wp_reports,
                       archived_family_pairs=len(family_pairs),archive=dict(archive.stats),
                       n_errors=len(errors),n_CIFs=len(archive.written_paths)),
                  f,indent=2)
    log(f'V64 COMPLETE SG={len(by_sg)} WP={len(plan)} probes={len(probes)} '
        f'CIFs={len(archive.written_paths)} errors={len(errors)} output={out}')
    return summaries

# ============================================================================
# JULIETTE v61.5 — ONE CENTRAL PROGRESS STREAM FOR SINGLE/MULTI GPU
# ============================================================================
# Space groups, not polyhedral parameters, are the independent GPU unit.
# This launcher does not alter the v61.3 kernels, their deterministic SG seeds,
# their TiO6 symmetry constraints, or their sharing/SCF/chemistry logic.
# Each worker sees exactly one CUDA device and runs the original v61_main().

def v614_gpu_assignments(space_groups, nworkers):
    """Stable disjoint round-robin SG allocation; preserve original SG order."""
    groups=list(space_groups)
    if not groups or nworkers<1 or nworkers>len(groups):
        raise ValueError('Require 1 <= number of workers <= number of SGs')
    assignments=[groups[i::nworkers] for i in range(nworkers)]
    assert sorted(g for subset in assignments for g in subset)==sorted(groups)
    assert all(assignments)
    return assignments

def v614_gpu_inventory():
    """Devices available INSIDE the scheduler's CUDA visibility mask."""
    if not torch.cuda.is_available():
        return []
    count=torch.cuda.device_count()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    tokens=[x.strip() for x in visible.split(',') if x.strip()] if visible else []
    if tokens and len(tokens)>=count:
        return tokens[:count]
    # If there is no explicit mask, CUDA ordinal indices are valid for workers.
    if not tokens:
        return [str(i) for i in range(count)]
    raise ValueError('CUDA_VISIBLE_DEVICES specifies fewer devices than PyTorch reports')

def v614_worker_count(a, available, n_sgs):
    """Automatic mode uses all visible GPUs unless a specific device is pinned."""
    requested=str(a.gpu_workers).strip().lower()
    if requested=='auto':
        if a.device=='cpu' or (a.device!='auto' and a.device.startswith('cuda:')):
            return 1
        return min(max(1,len(available)),n_sgs)
    try:
        n=int(requested)
    except ValueError as exc:
        raise ValueError('--gpu-workers must be auto or a positive integer') from exc
    if n<1:
        raise ValueError('--gpu-workers must be auto or a positive integer')
    if n>1 and a.device=='cpu':
        raise ValueError('Cannot use several GPU workers with --device cpu')
    if n>len(available) and not (n==1 and a.device in ('cpu','auto') and not available):
        raise ValueError(f'Requested {n} GPU workers, but only {len(available)} GPUs are visible. '
                         'Request more GPUs from Slurm or reduce --gpu-workers.')
    return min(n,n_sgs)

def v614_strip_worker_cli(argv):
    """Remove parent scheduling arguments (including --flag=value syntax)."""
    value_flags={'--gpu-workers','--sg-shard-count','--sg-shard-index',
                 '--space-groups','--output','--device','--study-sgs','--study-seed'}
    bool_flags={'--gpu-plan-only','--gpu-stream-logs'}
    result=[];i=0
    while i<len(argv):
        token=argv[i]
        name=token.split('=',1)[0]
        if name in bool_flags:
            i+=1
        elif name in value_flags:
            i+=1 if '=' in token else 2
        else:
            result.append(token);i+=1
    return result

def v614_worker_command(base_argv, space_groups, out, device='cuda:0'):
    """Device 0 means the unique GPU visible inside this worker."""
    return ([sys.executable,str(Path(__file__).resolve())]
            +v614_strip_worker_cli(base_argv)
            +['--gpu-workers','1','--sg-shard-count','1','--sg-shard-index','0',
              '--space-groups',','.join(map(str,space_groups)),
              '--device',device,'--output',str(out)])

def v614_merge_tables(out, workdirs):
    """Merge per-worker CSV metrics without loading all candidate rows in RAM."""
    for filename in ('per_sg.csv','per_wp.csv','probe_metrics.csv',
                     'final_candidates.csv','final_family_pairs.csv',
                     'breadth_wp.csv','breadth_cycle.csv','breadth_basin.csv',
                     'breadth_entrance.csv','breadth_sharing.csv'):
        columns=[];sources=[]
        for i,folder in enumerate(workdirs):
            path=folder/filename
            if not path.is_file() or path.stat().st_size==0:
                continue
            with path.open(newline='') as fh:
                reader=csv.DictReader(fh)
                for field in (reader.fieldnames or []):
                    if field not in columns:columns.append(field)
            sources.append((i,path))
        if not columns:
            continue
        if 'worker_gpu' not in columns:columns.append('worker_gpu')
        with (out/filename).open('w',newline='') as dst:
            writer=csv.DictWriter(dst,fieldnames=columns,extrasaction='ignore')
            writer.writeheader()
            for i,path in sources:
                with path.open(newline='') as src:
                    for row in csv.DictReader(src):
                        row['worker_gpu']=i
                        writer.writerow(row)

def v616_identical_file(left,right):
    """Byte-verify a live CIF before reusing it in the final merge."""
    left,right=Path(left),Path(right)
    if left.stat().st_size!=right.stat().st_size:return False
    with left.open('rb') as a,right.open('rb') as b:
        while True:
            x=a.read(1024*1024);y=b.read(1024*1024)
            if x!=y:return False
            if not x:return True

def v616_atomic_copy(src,dst):
    """Publish an SG-final CIF without ever exposing a half-written file."""
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if not v616_identical_file(src,dst):
            raise FileExistsError(f'Conflicting CIF name: {dst}')
        return
    with tempfile.NamedTemporaryFile(dir=dst.parent,prefix='.juliette_',
                                     suffix='.part',delete=False) as fh:
        part=Path(fh.name)
    try:
        shutil.copyfile(src,part)
        if dst.exists():
            if not v616_identical_file(part,dst):
                raise FileExistsError(f'Conflicting CIF name: {dst}')
        else:
            os.replace(part,dst)
    finally:
        part.unlink(missing_ok=True)

def v616_publish_sg_cifs(event,worker,tracker):
    """One shared output/cif folder populated immediately after SG completion.

    Worker directories remain private scratch/diagnostics.  A completed SG
    already wrote its *final* CIFs before emitting this event.  Never copy
    intermediate CIFs or independently modify final Ti/O coordinates.
    """
    sg=int(event['sg'])
    if int(event['n_cifs'])!=len(event.get('structures',[])):
        raise ValueError(f'SG{sg} event count does not match structure list')
    out=tracker['output']
    folder=out/f'gpu_{worker:02d}'/'cif'
    for record in event['structures']:
        name=record['cif_name']
        if Path(name).name!=name or not name.lower().endswith('.cif'):
            raise ValueError(f'Unsafe CIF name in SG{sg}: {name!r}')
        if int(record['initial_spgr'])!=sg:
            raise ValueError(f'Incorrect original SG for {name}')
        src=folder/name
        if not src.is_file():
            raise FileNotFoundError(f'SG{sg} final CIF is missing: {src}')
        dst=out/'cif'/f'gpu{worker:02d}_{name}'
        v616_atomic_copy(src,dst)
    return len(event['structures'])

def v614_merge_results(out, assignments, workdirs, statuses, elapsed_s):
    """Merge only completed worker artifacts; preserve all worker originals."""
    out.mkdir(parents=True,exist_ok=True)
    combined_cifs=out/'cif';combined_cifs.mkdir(exist_ok=True)
    per_sg=[];per_wp=[];errors=[];cifs=0
    workers=[]
    for i,(sgs,folder,exit_code) in enumerate(zip(assignments,workdirs,statuses)):
        worker=dict(index=i,space_groups=sgs,exit_code=int(exit_code),
                    output=str(folder),log=str(folder/'worker.log'))
        meta=folder/'summary.json'
        if meta.is_file():
            try:
                obj=json.loads(meta.read_text())
                per_sg.extend(obj.get('per_sg',[]))
                per_wp.extend(obj.get('per_wp',[]))
                worker['n_errors']=int(obj.get('n_errors',0))
                worker['n_CIFs']=int(obj.get('n_CIFs',0))
            except (ValueError,TypeError) as exc:
                errors.append(dict(worker=i,phase='merge_summary',error=str(exc)))
        errpath=folder/'errors.json'
        if errpath.is_file():
            try:
                for entry in json.loads(errpath.read_text()):
                    errors.append(dict(worker=i,**entry))
            except (ValueError,TypeError) as exc:
                errors.append(dict(worker=i,phase='merge_errors',error=str(exc)))
        for cif in sorted((folder/'cif').glob('*.cif')):
            target=combined_cifs/f'gpu{i:02d}_{cif.name}'
            if target.exists():
                # v61.6 publishes CIFs as SGs complete; final merge verifies them.
                if not v616_identical_file(cif,target):
                    raise RuntimeError(f'Conflicting live/final CIF: {target}')
            else:
                v616_atomic_copy(cif,target)
            cifs+=1
        workers.append(worker)
    v614_merge_tables(out,workdirs)
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    summary=dict(version='v61.4',parallel='one_process_per_gpu',
                 wall_seconds=elapsed_s,workers=workers,per_sg=per_sg,
                 per_wp=per_wp,n_CIFs=cifs,n_errors=len(errors),
                 complete=all(x==0 for x in statuses),
                 notes='Each SG belongs to one worker. Per-SG family dedup is retained; '
                       'cross-worker/cross-SG StructuralMatcher dedup is not performed.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def v615_format_structure(record, rank):
    def num(value, spec):
        try:return format(float(value),spec) if value is not None else 'NA'
        except (TypeError,ValueError):return 'NA'
    return (f'    #{rank:02d} SG{record["initial_spgr"]:03d}->{record["final_spgr"] or "?"} '
            f'prim={record["N_atom_primitive"] or "?"} '
            f'Ti_loss=[{num(record["Ti_loss_min"],".4g")},'
            f'{num(record["Ti_loss_max"],".4g")}] '
            f'Ti_avg={num(record["Ti_loss_avg"],".4g")} '
            f'goodTi={record["N_good_Ti"]} goodO={record["N_good_O"]} '
            f'O-O={num(record["min_OO_A"],".3f")}A '
            f'O_shift={num(record["max_O_shift_A"],".3f")}A '
            f'graph_bad={record["sharing_graph_violations"]} '
            f'rank={num(record["rank_score"],".4g")} '
            f'CIF={record["cif_name"]}')

def v615_write_combined_structures(out,events):
    """Deterministic structure report, independent of worker completion order."""
    columns=['worker_gpu','initial_spgr','final_spgr','N_atom_primitive',
             'Ti_loss_min','Ti_loss_max','Ti_loss_avg','N_good_Ti','N_good_O',
             'min_OO_A','max_O_shift_A','rank_score','sharing_graph_violations','cif_path']
    rows=[]
    for event in events:
        for record in event.get('structures',[]):
            row={key:record.get(key) for key in columns}
            row['worker_gpu']=event['worker_gpu']
            row['cif_path']=str(Path('cif')/f'gpu{event["worker_gpu"]:02d}_{record["cif_name"]}')
            rows.append(row)
    rows.sort(key=lambda r:(float('inf') if r['Ti_loss_max'] is None
                            else float(r['Ti_loss_max']),r['initial_spgr'],
                            r['cif_path']))
    with (out/'combined_structures.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    return rows

def v615_progress_handler(line,worker,tracker):
    """Handle a single worker stdout line; returns True for an event line."""
    if not line.startswith(V615_EVENT_PREFIX):return False
    try:
        event=json.loads(line[len(V615_EVENT_PREFIX):])
        if event.get('event')!='sg_complete':return True
        sg=int(event['sg'])
        if sg not in tracker['assignments'][worker]:
            raise ValueError(f'Worker {worker} reported unassigned SG{sg}')
        with tracker['lock']:
            if sg in tracker['done']:
                raise ValueError(f'Duplicate SG{sg} progress event')
            v616_publish_sg_cifs(event,worker,tracker)
            tracker['done'].add(sg)
            event['worker_gpu']=worker
            tracker['events'].append(event)
            tracker['journal'].write(json.dumps(event,allow_nan=False)+'\n')
            tracker['journal'].flush()
            best=(min((r['Ti_loss_max'] for r in event['structures']),default=None))
            besttxt=f'{best:.5g}' if best is not None else 'none'
            finished=len(tracker['done'])
            status=('OK' if event['n_cifs'] else 'NO SOLUTION')
            log(f'[{finished:3d}/{tracker["total"]:3d}] GPU{worker} SG{sg:03d} '
                f'{status}: WPs={event["wp_tasks"]} expanded={event.get("expanded_wps",0)} probes={event["probes"]} '
                f'CIFs={event["n_cifs"]} families={event.get("probe_families",0)} '
                f'promising={event.get("promising_probe_families",0)} best_Ti_loss={besttxt} '
                f'wall={event["wall_s"]:.1f}s')
            for i,record in enumerate(sorted(event['structures'],
                                              key=lambda x:x['Ti_loss_max']),1):
                log(v615_format_structure(record,i))
            if event['n_cifs']:
                log(f'    Final CIFs available now: {tracker["output"]/"cif"}')
        return True
    except (ValueError,KeyError,TypeError,OSError) as exc:
        with tracker['lock']:
            tracker['progress_errors'].append(f'GPU{worker}: {exc}')
            log(f'  PROGRESS EVENT ERROR GPU{worker}: {exc}')
        return True

def v614_main(argv=None):
    """Compact, live per-structure progress for 1..N independent GPU workers."""
    supplied=list(sys.argv[1:] if argv is None else argv)
    if os.environ.get('JULIETTE_GPU_CHILD')=='1':
        return v61_main(supplied)  # never recursively spawn a worker
    a=v63_parse(supplied)
    if a.self_test:
        return v61_main(supplied)
    all_sgs=parse_sgs(a.space_groups)  # apply external --sg-shard-* first
    devices=v614_gpu_inventory()
    count=v614_worker_count(a,devices,len(all_sgs))
    if a.device.startswith('cuda:') and not devices:
        raise ValueError(f'--device {a.device} requested but no CUDA GPU is visible')
    if a.device.startswith('cuda:') and count==1:
        selected=int(a.device.split(':',1)[1])
        if selected>=len(devices):
            raise ValueError(f'--device {a.device} unavailable; {len(devices)} CUDA GPUs visible')
        devices=[devices[selected]]
    assigned=v614_gpu_assignments(all_sgs,count)
    out=Path(a.output)
    cpu_mode=(a.device=='cpu' or (not devices and a.device=='auto'))
    log(f'JULIETTE v64 | coupled Wyckoff state + live CIFs | '
        f'GPUs={0 if cpu_mode else count} workers={count} SGs={len(all_sgs)} '
        f'output={out}')
    if a.study_sgs:
        log(f'  STUDY SG SAMPLE n={len(all_sgs)} seed={a.study_seed}: '+','.join(map(str,all_sgs)))
    log(f'  WP per SG={a.wp_per_sg or "all within pool"}; '
        f'WP diversity pool={a.wp_diversity_pool}; '
        f'max independent Ti orbits={a.max_ti_orbits or "unlimited"}; '
        f'live CIF folder={out/"cif"}')
    log(f'  progressive_WP={a.progressive_wp}; screening: cycles={a.screen_metal_cycles}, '
        f'basins={a.screen_basins}, entrances={a.screen_entrances}, '
        f'sharing={a.screen_sharing_trials}; expansion: WPs={a.expand_wps} '
        f'(zero-probe reserve={a.expand_zero_probe_wps}), additional cycles={max(0,a.metal_cycles-a.screen_metal_cycles)}, '
        f'basins={a.basins_per_wp}, entrances={a.entrances}, sharing={a.sharing_trials}')
    for i,sgs in enumerate(assigned):
        label=('CPU' if cpu_mode else str(devices[i]))
        log(f'  worker {i}: device={label} SGs={len(sgs)} '
            f'range={sgs[0]}..{sgs[-1]}')
    if a.gpu_plan_only or a.plan_only:
        return dict(assignments=assigned,devices=devices,workers=count)
    if not cpu_mode and len(devices)<count:
        raise ValueError('Not enough visible GPUs for requested workers')
    import subprocess
    from threading import Thread, Lock
    out.mkdir(parents=True,exist_ok=True)
    workdirs=[out/f'gpu_{i:02d}' for i in range(count)]
    for folder in workdirs:
        if (folder/'summary.json').exists() or (folder/'worker.log').exists():
            raise FileExistsError(f'Worker output already exists: {folder}; '
                                  'choose a new --output to prevent mixing runs')
        folder.mkdir(parents=True,exist_ok=True)
    processes=[];threads=[];logs=[];start=time.perf_counter()
    progress_journal=(out/'progress.jsonl').open('w',buffering=1)
    (out/'cif').mkdir(exist_ok=True)
    tracker=dict(lock=Lock(),done=set(),events=[],total=len(all_sgs),
                 assignments=assigned,journal=progress_journal,
                 output=out,progress_errors=[])

    def relay(stream,loghandle,worker):
        for line in stream:
            loghandle.write(line)
            loghandle.flush()
            is_event=v615_progress_handler(line.rstrip('\n'),worker,tracker)
            diagnostic=any(marker in line for marker in (
                'Traceback (most recent call last)', 'error:', 'Error:',
                'Exception:', 'SystemExit:', 'ModuleNotFoundError:',
                'FileNotFoundError:', 'RuntimeError:', 'ValueError:'))
            if not is_event and (a.gpu_stream_logs or diagnostic):
                with tracker['lock']:
                    log(f'[GPU{worker}] {line.rstrip()}')
        stream.close()

    try:
        for i,(sgs,folder) in enumerate(zip(assigned,workdirs)):
            env=os.environ.copy()
            if not cpu_mode:env['CUDA_VISIBLE_DEVICES']=devices[i]
            env.setdefault('OMP_NUM_THREADS','1')
            env.setdefault('MKL_NUM_THREADS','1')
            env.setdefault('OPENBLAS_NUM_THREADS','1')
            env['JULIETTE_GPU_CHILD']='1'
            env['PYTHONUNBUFFERED']='1'
            cmd=v614_worker_command(supplied,sgs,folder,
                                    device='cpu' if cpu_mode else 'cuda:0')
            handle=(folder/'worker.log').open('w',buffering=1)
            logs.append(handle)
            proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            processes.append(proc)
            thread=Thread(target=relay,args=(proc.stdout,handle,i),daemon=True)
            thread.start();threads.append(thread)
        statuses=[p.wait() for p in processes]
        for t in threads:t.join()
    except BaseException:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        raise
    finally:
        progress_journal.close()
        for h in logs:h.close()
    # Never report 'no solution' as if an unsuccessful worker had searched.
    # The worker log tail includes argparse usage and full Python tracebacks.
    from collections import deque
    for worker,(status,folder) in enumerate(zip(statuses,workdirs)):
        if status:
            log(f'\n[GPU{worker}] FAILED (exit {status}); worker error log: '
                f'{folder/"worker.log"}')
            if not a.gpu_stream_logs:
                with (folder/'worker.log').open(errors='replace') as fh:
                    tail=list(deque(fh,maxlen=60))
                log(f'[GPU{worker}] ---- LAST {len(tail)} WORKER LOG LINES ----')
                for line in tail:
                    log(f'[GPU{worker}] {line.rstrip()}')
                log(f'[GPU{worker}] ---- END WORKER LOG ----')
    summary=v614_merge_results(out,assigned,workdirs,statuses,time.perf_counter()-start)
    rows=v615_write_combined_structures(out,tracker['events'])
    report=v62_aggregate_study(out)
    summary.update(version='v64',progress_events=len(tracker['events']),
                   completed_space_groups=len(tracker['done']),
                   combined_structure_rows=len(rows),breadth_recommendations=report)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    log('')
    log('=== COMBINED STRUCTURES (all GPUs; ascending Ti_loss_max) ===')
    log('GPU initial_SG final_SG primitive Ti_loss_min Ti_loss_max Ti_loss_avg '
        'good_Ti good_O min_OO_A max_O_shift_A rank_score graph_violations')
    if not rows:
        if any(statuses):log('  INCOMPLETE SEARCH: one or more workers failed; no chemical conclusion')
        else:log('  No CIFs passed the final polyhedral-displacement filter')
    for r in rows:
        f=lambda key:('NA' if r[key] is None else str(round(float(r[key]),6)))
        log(f'{r["worker_gpu"]:02d} {r["initial_spgr"]:03d} '
            f'{r["final_spgr"] if r["final_spgr"] is not None else "?"} '
            f'{r["N_atom_primitive"] if r["N_atom_primitive"] is not None else "?"} '
            f'{f("Ti_loss_min")} {f("Ti_loss_max")} {f("Ti_loss_avg")} '
            f'{r["N_good_Ti"]} {r["N_good_O"]} {f("min_OO_A")} '
            f'{f("max_O_shift_A")} {f("rank_score")} '
            f'{r["sharing_graph_violations"]}')
    log(f'V64 COMPLETE: SGs={len(tracker["done"])}/{len(all_sgs)} '
        f'GPUs={0 if cpu_mode else count} CIFs={summary["n_CIFs"]} '
        f'wall={summary["wall_seconds"]:.1f}s '
        f'failed_workers={sum(x!=0 for x in statuses)} '
        f'combined_table={out/"combined_structures.csv"} '
        f'worker_logs={out/"gpu_XX"/"worker.log"}')
    if tracker['progress_errors']:
        summary['progress_errors']=tracker['progress_errors']
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if not summary['complete'] or len(tracker['done'])!=len(all_sgs) or tracker['progress_errors']:
        for w in summary['workers']:
            if w['exit_code']:
                log(f'  FAILED GPU{w["index"]}: exit={w["exit_code"]} log={w["log"]}')
        raise SystemExit(1)
    return summary


# ==========================================================================
# JULIETTE v64 — PERSISTENT CRYSTAL STATE / COUPLED FINAL REFINEMENT
#
# v63 exploration is retained. During finalist refinement the same WP object
# owns Ti Wyckoff coordinates, lattice, coherent octahedral F matrices and an
# immutable sharing policy. O is always derived from the object (never fitted
# as independent Cartesian coordinates). Archive entries retain that complete
# object; the final CIF is regenerated from it, not flattened trial arrays.
# ==========================================================================

class V64CrystalState:
    """One independently snapshot-able crystallographic source of truth.

    `octas` stores all independent Ti Wyckoff polyhedra, their orientations,
    strains, the lattice and the attached three-arm/O sharing policy.  Atomic
    arrays are transient *views*, not independent mutable structural state.
    """
    def __init__(self, octas):
        if getattr(octas, 'sharing_registry', None) is None:
            raise V61InvalidSharing('v64 state requires a sharing policy')
        self.octas = octas.copy()

    def copy(self):
        return V64CrystalState(self.octas)

    def snapshot(self,chem,stage='v64-state'):
        return v61_snapshot(self.octas,chem,stage)

    def set_wyckoff_coordinate(self,orbit_index,coordinate_index,value):
        self.octas.orbits[orbit_index].u[coordinate_index]=float(value)%1.
        v64_refresh_state(self.octas)
        return self

    def set_cell(self,cell):
        v64_refresh_state(self.octas,cell)
        return self

    def set_sharing_from_cartesian_seeds(self,centers):
        v61_attach_lcao_policy(self.octas,centers)
        return self

    def refine(self,chem,mode_options,config):
        return v64_joint_scf(self.snapshot(chem),chem,mode_options,config)

    def write_cif(self,path):
        self.audit()
        ti,cell,oxygen=self.coordinates()
        _v44_atomic_cif(dict(ti=ti,cell=cell,answer=dict(oxygen=oxygen)),path)
        return path

    def coordinates(self):
        o, _, bad, _ = v61_derive_oxygen(self.octas)
        if bad:
            raise V61InvalidSharing('v64 derived oxygen has repeated Ti images')
        return (np.asarray(self.octas.ti, float).copy(),
                np.asarray(self.octas.cell, float).copy(),
                np.asarray(o, float).copy())

    def audit(self, ti=None, cell=None, oxygen=None, tol=2e-6):
        a, b, c = self.coordinates()
        if len(c) != 2*len(a) or b.shape != (3,3) or np.linalg.det(b) <= 0:
            raise ValueError('v64 invalid structure geometry/stoichiometry')
        if not all(np.isfinite(v).all() for v in (a,b,c)):
            raise ValueError('v64 nonfinite structure geometry')
        if v61_ideal_symmetry_error(self.octas) > 1.e-4:
            raise ValueError('v64 Wyckoff polyhedron symmetry drift')
        if coherent_strain_extrema(self.octas) > self.octas.max_strain+1e-5:
            raise ValueError('v64 coherent strain bound violated')
        if cell is not None and np.max(np.abs(b-np.asarray(cell))) > tol:
            raise ValueError('v64 archived cell differs from WP state')
        for actual,expected,label in ((a,ti,'Ti'),(c,oxygen,'oxygen')):
            if expected is not None:
                if np.asarray(expected).shape != actual.shape:
                    raise ValueError(f'v64 {label} coordinate shape drift')
                delta=minimum_image_delta(actual@b,np.asarray(expected)@b,b,
                                          np.linalg.inv(b))
                if np.max(np.linalg.norm(delta,axis=1)) > tol:
                    raise ValueError(f'v64 archived {label} differs from WP state')
        return True


def v64_refresh_state(octas, cell=None):
    """Rebuild symmetry operations and WP maps after changing Ti u or cell.

    A trial is discarded if symmetry/site-F invariance cannot be maintained.
    Do not change a site orbit's six arms independently or regenerate its
    oxygen policy. Full group operations stay fractional and crystal-invariant.
    """
    if cell is not None:
        cell=np.asarray(cell,float).reshape(3,3)
        if not np.isfinite(cell).all() or np.linalg.det(cell) <= 1e-8:
            raise WyckoffIncompatible('v64 invalid trial lattice')
        octas.cell=cell.copy()
        octas.inv=np.linalg.inv(cell)
    for orbit in octas.orbits:
        orbit.cell=octas.cell.copy()
        orbit.maps=w594_full_group_orbit_maps(octas.cell,orbit.representative(),
                    orbit.frac_sites(),octas.group_ops)
        site_ops=w594_site_ops(octas.group_ops,orbit.representative(),octas.cell)
        # Rebuild the site-constrained basis after a metric change. Reproject
        # the EXISTING full-polyhedron F, never individual vertices.
        F,B,p=w594_project_F(orbit.src(),site_ops,orbit.F,orbit.max_strain)
        if F is None or np.max(np.abs(F-orbit.F))>0.02:
            raise WyckoffIncompatible('v64 trial incompatible with Wyckoff F')
        orbit.site_ops=site_ops
        orbit.F,orbit.B,orbit.permutations=F,B,p
    octas._rebuild()
    if v61_ideal_symmetry_error(octas)>1e-4:
        raise WyckoffIncompatible('v64 SG-preserving trial failed')
    return octas


def v64_lattice_directions(kind):
    """Symmetry-compatible *length* changes, holding cell angles fixed."""
    kind=str(kind).lower()
    if kind=='cubic':return [(1.,1.,1.)]
    if kind in ('hexagonal','tetragonal','trigonal'):
        return [(1.,1.,0.),(0.,0.,1.)]
    return [(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]


def v64_candidate_score(snapshot, max_o_move):
    if snapshot['bad_triplets'] or snapshot['sharing_max_A']>max_o_move+1e-7:
        return float('inf')
    return (continuous_candidate_objective(snapshot)
            +2.*float(snapshot['oo_overlap_penalty']))


def v64_joint_scf(result, chem, mode_options, a):
    """Bounded alternating SCF: Ti Wyckoff u + lattice + full-octahedron F.

    Discrete topology stays linked while each continuous step is tested. On
    stagnation, try at most `v64_topology_updates` LCAO repartitions, accepting
    only fully re-refined, lower-loss physical crystals. Every trial is a copy
    of the SAME kind of complete Wyckoff-state object; no atom-array shortcuts.

    This is a bounded derivative-free coordinate search (not MD or DFT).
    """
    state=V64CrystalState(result['oct'])
    work=state.octas
    best=v61_snapshot(work,chem,'v64-coupled-initial')
    bestval=v64_candidate_score(best,a.max_o_move)
    if not np.isfinite(bestval):
        return result
    rng=np.random.default_rng(str_seed(a.seed,'v64-joint',
        tuple(int(o.mode) for o in work.orbits),len(work.ti),
        np.round(work.cell,5).tobytes().hex()))
    base_lattice=np.asarray(work.cell,float).copy()
    center_step=float(a.v64_center_step)
    lattice_step=float(a.v64_cell_step)
    history=[]
    topo_tries=0
    for cycle in range(int(a.v64_coupled_cycles)):
        startval=bestval
        # Work exclusively on a snapshot, so rejected moves cannot corrupt
        # the parent object or the retained best state.
        current=best['oct'].copy()
        coords=[]
        for i,orbit in enumerate(current.orbits):
            coords.extend((i,j) for j in range(orbit.dof))
        if len(coords)>a.v64_coordinate_budget:
            chosen=rng.choice(len(coords),size=a.v64_coordinate_budget,replace=False)
            coords=[coords[int(i)] for i in sorted(chosen)]
        proposals=[]
        for i,j in coords:
            for sign in (-1.,1.):
                proposals.append(('center',i,j,sign))
        for direction in v64_lattice_directions(current.template.lattice_type):
            for sign in (-1.,1.):
                proposals.append(('cell',direction,0,sign))
        # Greedy best-improvement search; each score is computed on actual
        # three-arm-derived oxygen and current Ti/lattice chemistry, not ideals.
        move_best=None; move_score=bestval
        for kind,i,j,sign in proposals:
            trial=current.copy()
            try:
                if kind=='center':
                    state_trial=V64CrystalState(trial)
                    state_trial.set_wyckoff_coordinate(i,j,
                        trial.orbits[i].u[j]+sign*center_step)
                    trial=state_trial.octas
                else:
                    factors=np.exp(sign*lattice_step*np.asarray(i,float))
                    newcell=current.cell*factors[:,None]
                    # Limit drift relative to ORIGINAL accepted metal lattice.
                    original_lengths=np.linalg.norm(base_lattice,axis=1)
                    new_lengths=np.linalg.norm(newcell,axis=1)
                    if np.max(np.abs(np.log(new_lengths/original_lengths))) > a.v64_max_lattice_change:
                        continue
                    state_trial=V64CrystalState(trial)
                    state_trial.set_cell(newcell)
                    trial=state_trial.octas
                sample=v61_snapshot(trial,chem,'v64-joint-trial')
                score=v64_candidate_score(sample,a.max_o_move)
                if score < move_score - a.v64_min_improvement:
                    move_best,move_score=sample,score
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError):
                continue
        if move_best is not None:
            best,bestval=move_best,move_score
        # The coherent F/mode fit is coupled back to any new center/lattice.
        if a.v64_inner_scf>0:
            try:
                refined=v61_scf(best['oct'],chem,mode_options,
                                steps=a.v64_inner_scf,center_mix=a.center_scf_mix)
                val=v64_candidate_score(refined,a.max_o_move)
                if val < bestval-a.v64_min_improvement:
                    best,bestval=refined,val
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError):
                pass
        # Discrete sharing is reconsidered only on a plateau; never rematch
        # at every geometry step (which would cause objective discontinuities).
        if (bestval >= startval-a.v64_min_improvement and
                topo_tries<int(a.v64_topology_updates)):
            topo_tries+=1
            try:
                trial=best['oct'].copy()
                seeds=best['oxygen']@trial.cell
                if a.v64_topology_jitter>0:
                    seeds=seeds+rng.normal(0.,a.v64_topology_jitter,seeds.shape)
                state_trial=V64CrystalState(trial)
                state_trial.set_sharing_from_cartesian_seeds(seeds)
                trial=state_trial.octas
                if not np.array_equal(trial.sharing_registry.trip,
                                      best['oct'].sharing_registry.trip):
                    candidate=v61_scf(trial,chem,mode_options,
                                steps=max(1,a.v64_inner_scf),
                                center_mix=a.center_scf_mix)
                    value=v64_candidate_score(candidate,a.max_o_move)
                    if value < bestval-a.v64_min_improvement:
                        best,bestval=candidate,value
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError):
                pass
        history.append(dict(cycle=cycle+1,objective=float(bestval),
                            max_o_shift=float(best['sharing_max_A']),
                            min_oo=float(best['min_oo_A']),
                            lattice_lengths=np.linalg.norm(best['oct'].cell,axis=1).tolist()))
        if bestval >= startval-a.v64_min_improvement:
            center_step*=.6
            lattice_step*=.6
        if center_step<1e-5 and lattice_step<1e-5:
            break
    best['v64_coupled_history']=history
    best['v64_topology_attempts']=topo_tries
    best['probe_stage']='v64-coupled-wyckoff-state'
    V64CrystalState(best['oct']).audit(best['oct'].ti,best['oct'].cell,best['oxygen'])
    return best




_v63_parse_before_v64=v63_parse
def v63_parse(argv=None):
    """Parse v64 flags while preserving every v63 argument and its behavior."""
    supplied=list(sys.argv[1:] if argv is None else argv)
    extra=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    extra.add_argument('--v64-help',action='help',help='Show v64 refinement flags')
    extra.add_argument('--v64-coupled-cycles',type=int,default=3,
                       help='Finalist joint Ti/lattice/polyhedron cycles (0 disables)')
    extra.add_argument('--v64-inner-scf',type=int,default=1,
                       help='Coherent-F fit cycles following each Ti/lattice update')
    extra.add_argument('--v64-center-step',type=float,default=.012,
                       help='Step in free Wyckoff fractional coordinates')
    extra.add_argument('--v64-cell-step',type=float,default=.012,
                       help='Log-scale lattice length step per direction')
    extra.add_argument('--v64-max-lattice-change',type=float,default=.06,
                       help='Max absolute log-length change from original metal cell')
    extra.add_argument('--v64-coordinate-budget',type=int,default=6,
                       help='Max independent Wyckoff u coordinates tried per cycle')
    extra.add_argument('--v64-topology-updates',type=int,default=1,
                       help='Max LCAO sharing reassignment attempts on plateau')
    extra.add_argument('--v64-topology-jitter',type=float,default=.12,
                       help='Cartesian jitter (A) for plateau topology proposals')
    extra.add_argument('--v64-min-improvement',type=float,default=1e-5,
                       help='Required decrease in the coupled candidate objective')
    options,remaining=extra.parse_known_args(supplied)
    a=_v63_parse_before_v64(remaining)
    # Do not silently reuse a v63 output directory when no --output is given.
    if not v63_has_flag(supplied,'--output'):
        a.output='tio2_v64'
    for name,val in vars(options).items():setattr(a,name,val)
    for name in ('v64_coupled_cycles','v64_inner_scf','v64_coordinate_budget',
                 'v64_topology_updates'):
        if getattr(a,name)<0:raise ValueError(f'{name} must be >= 0')
    for name in ('v64_center_step','v64_cell_step','v64_max_lattice_change',
                 'v64_topology_jitter','v64_min_improvement'):
        if not np.isfinite(getattr(a,name)) or getattr(a,name)<0:
            raise ValueError(f'{name} must be finite and >= 0')
    return a


_v61_final_candidates_before_v64=v61_final_candidates
def v61_final_candidates(item,a):
    """Preserve all v63 stages; add a fully owned, coupled finalist state."""
    stages=_v61_final_candidates_before_v64(item,a)
    if a.v64_coupled_cycles<=0:return stages
    try:
        source=min((r for _,r in stages if np.isfinite(v64_candidate_score(r,a.max_o_move))),
                   key=lambda r:v64_candidate_score(r,a.max_o_move),default=None)
        if source is not None:
            final=V64CrystalState(source['oct']).refine(
                item['chem_probe'],item['mode_options'],a)
            stages.append(('v64_coupled',final))
            log(f'    V64 JOINT: cycles={len(final.get("v64_coupled_history",[]))} '
                f'topology_attempts={final.get("v64_topology_attempts",0)} '
                f'objective={v64_candidate_score(final,a.max_o_move):.6g} '
                f'O_shift={final["sharing_max_A"]:.4f} A')
    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
        log('    V64 coupled refinement warning: '+type(exc).__name__+': '+str(exc))
    return stages


def v64_self_test():
    """Fast database-free test of WP state, joint Ti/lattice search and drift audit.

    This is a synthetic geometry/optimizer test, NOT chemical/physical proof
    and not a replacement for an end-to-end search with PyXtal/pymatgen.
    """
    v61_self_test()
    wp=[]
    for _ in range(3):
        wp.append(dict(label='a',dof=3,
            A=torch.eye(3,dtype=DTYPE),b=torch.zeros(3,dtype=DTYPE),
            rot=torch.eye(3,dtype=DTYPE)[None],
            tra=torch.zeros((1,3),dtype=DTYPE)))
    template=argparse.Namespace(lattice_keys=('a',),lattice_type='cubic',
                                blocks=wp,n_ti=3)
    rigid=np.asarray([[1.91,0,0],[-1.91,0,0],[0,1.91,0],
                      [0,-1.91,0],[0,0,1.91],[0,0,-1.91]],float)
    proposals=[[(0.,0,Rotation.identity())] for _ in range(3)]
    octas=WyckoffOctahedra(template,np.zeros(10),np.eye(3)*10.,
               {'Ti':[{'rigid_arms':rigid}]},proposals,
               [(np.eye(3),np.zeros(3))],max_strain=.11)
    for orbit,u in zip(octas.orbits,[(.5,.5,.5),(.54,.5,.5),(.5,.54,.5)]):
        orbit.u=np.asarray(u,float)
    v64_refresh_state(octas)
    V61SharingPolicy(octas,np.tile(np.arange(6),3)).attach(octas)
    state=V64CrystalState(octas)
    state.audit()

    class SyntheticChem:
        guard=.6
        def score(self,oxygen,details=False,**kwargs):
            energy=float((torch.linalg.norm(self.cell[0])-10.22)**2+
                         3*(self.ti[0,0]-.53)**2)
            return dict(energy=energy,collision=0.,hierarchy='ti-first',
                        goodTi=np.ones(3,bool),goodO=np.ones(6,bool))
    chem=SyntheticChem()
    initial=state.snapshot(chem)
    config=argparse.Namespace(seed=2,max_o_move=1.,v64_coupled_cycles=2,
        v64_inner_scf=0,v64_center_step=.01,v64_cell_step=.01,
        v64_max_lattice_change=.06,v64_coordinate_budget=9,
        v64_topology_updates=0,v64_topology_jitter=0.,
        v64_min_improvement=1e-8,center_scf_mix=0.)
    result=state.refine(chem,[[(0,None)] for _ in range(3)],config)
    if not v64_candidate_score(result,1.)<v64_candidate_score(initial,1.):
        raise AssertionError('v64 Ti/lattice coupled refinement failed')
    V64CrystalState(result['oct']).audit(result['oct'].ti,
                          result['oct'].cell,result['oxygen'])
    try:
        V64CrystalState(result['oct']).audit(oxygen=result['oxygen']+.03)
    except ValueError:
        pass
    else:
        raise AssertionError('v64 archival coordinate drift was not rejected')
    # Exercise the lattice subspace even when every center coordinate is frozen.
    config.v64_coordinate_budget=0
    result=state.refine(chem,[[(0,None)] for _ in range(3)],config)
    if not np.linalg.norm(result['oct'].cell[0])>10.:
        raise AssertionError('v64 lattice refinement failed')
    log('V64 SELF-TEST PASSED: full WP state, Ti/lattice coupling, '
        'oxygen reconstruction and archive drift detection')
    return True


# JULIETTE v72 generalized multi-reference entrypoint follows below.



# ============================================================================
# JULIETTE v72 — GENERALIZED SINGLE-CENTER POLICY AND SHARING ENGINE.
# v64's TiO2 GPU engine, sharing policy, refinement and archive remain intact.
# This engine does not introduce independently mutable anion coordinates.
# ============================================================================

@dataclass(frozen=True)
class V72BuildingBlock:
    """One center species + its reference-derived complete vertex environment.

    `modes` are *ordered complete polyhedra*, not independently movable bonds.
    Their only continuous DOF are WP center `u` and coherent affine `F`.
    """
    center: str
    anion: str
    coordination: int
    modes: tuple
    center_shell: np.ndarray
    anion_core_A: float
    center_anion_core_A: float
    volume_per_fu: float
    formula: str
    centers_per_fu: int
    anions_per_fu: int
    sharing_degree: int
    reference_paths: tuple
    mode_sources: tuple = ()
    max_strain: float = .11

    def __post_init__(self):
        if not (self.coordination>0 and self.centers_per_fu>0 and
                self.anions_per_fu>0 and self.sharing_degree>0):
            raise ValueError('Nonpositive building-block stoichiometry')
        if self.coordination*self.centers_per_fu!=self.sharing_degree*self.anions_per_fu:
            raise ValueError('Polyhedron-sharing degree contradicts formula stoichiometry')
        if not self.modes or any(np.shape(m)!=(self.coordination,3) for m in self.modes):
            raise ValueError('Every source mode must have exactly coordination vertices')
        if not (self.anion_core_A>0 and self.center_anion_core_A>0 and
                self.volume_per_fu>0 and self.max_strain>=0):
            raise ValueError('Invalid reference-derived geometry')


def v72_parse_chemistry(center,block,formula=None,anion_sharing=None):
    """Single center type; compounds may have e.g. B2O3 (two B per FU)."""
    import re as _re
    m=_re.fullmatch(r'([A-Z][a-z]?)([A-Z][a-z]?)([1-9][0-9]*)',block)
    if not m or m.group(1)!=center:
        raise ValueError('--block must have form CenterAnionCoordination, e.g. GeS4, BO3, TiO6')
    anion,vertices=m.group(2),int(m.group(3))
    if anion==center:raise ValueError('Center and anion elements must differ')
    if formula is None:
        defaults={'TiO6':'TiO2','GeS4':'GeS2','BO3':'B2O3','BP4':'BP'}
        formula=defaults.get(block)
        if formula is None:
            if anion_sharing is None:
                raise ValueError('For a new building block, supply --formula or --anion-sharing')
            g=math.gcd(vertices,int(anion_sharing))
            nc=int(anion_sharing)//g;na=vertices//g
            formula=center+(str(nc) if nc!=1 else '')+anion+(str(na) if na!=1 else '')
    pattern=r'([A-Z][a-z]?)([0-9]*)([A-Z][a-z]?)([0-9]*)'
    f=_re.fullmatch(pattern,formula)
    if f is None or (f.group(1),f.group(3))!=(center,anion):
        raise ValueError(f'--formula must contain only {center}, {anion} in that order')
    nc=int(f.group(2) or '1');na=int(f.group(4) or '1')
    if nc<1 or na<1 or vertices*nc%na:
        raise ValueError('Formula and block imply noninteger anion sharing')
    share=vertices*nc//na
    if anion_sharing is not None and share!=int(anion_sharing):
        raise ValueError(f'--anion-sharing {anion_sharing} contradicts formula ({share})')
    return dict(center=center,anion=anion,coordination=vertices,
                formula=formula,centers_per_fu=nc,anions_per_fu=na,
                sharing_degree=share)


def v72_reference_shell(structure,site_i,symbol,n):
    """Nearest *periodic images* of one species; count self images except zero.

    Similar images of one crystallographic site are legitimate distinct vertices
    in the coordination environment. The reference H8/H11 bug is avoided.
    """
    frac=np.asarray([x.frac_coords for x in structure if x.specie.symbol==symbol],float)
    cell=np.asarray(structure.lattice.matrix,float)
    origin=np.asarray(structure[site_i].frac_coords,float)
    if not len(frac):raise ValueError(f'Reference lacks species {symbol}')
    for rad in (2,3,4,6):
        shifts=np.asarray(list(itertools.product(range(-rad,rad+1),repeat=3)),float)
        dv=(frac[:,None,:]-origin[None,None,:]+shifts[None,:,:])@cell
        norms=np.linalg.norm(dv,axis=-1).reshape(-1)
        valid=np.flatnonzero(norms>1.e-6)
        if len(valid)<n:continue
        best=valid[np.argsort(norms[valid],kind='stable')[:n]]
        chosen_shifts=shifts[best%len(shifts)]
        if np.any(np.max(np.abs(chosen_shifts),axis=1)>=rad) and rad<6:
            continue
        return dv.reshape(-1,3)[best],norms[best]
    raise ValueError(f'Cannot obtain {n} periodic {symbol} reference neighbors')


def v72_formula_from_structure(structure, center, anion, path='<reference>'):
    """Infer reduced binary composition from CIF atom sites, not CLI guesses.

    All sites must be fully occupied and ordered. This is deliberate: a partial
    occupancy is not a unique polyhedral reference environment.
    """
    count=collections.Counter()
    for site in structure:
        try:
            symbol=site.specie.symbol
        except (AttributeError,ValueError) as exc:
            raise ValueError(f'{path}: disordered/ambiguous sites are unsupported') from exc
        if abs(float(getattr(site,'occupancy',1.))-1.)>1.e-6:
            raise ValueError(f'{path}: partially occupied reference sites unsupported')
        count[symbol]+=1
    if set(count)!={center,anion} or count[center]<1 or count[anion]<1:
        raise ValueError(f'{path}: expected only {center} and {anion}, found {dict(count)}')
    divisor=math.gcd(count[center],count[anion])
    nc=count[center]//divisor;na=count[anion]//divisor
    formula=(center+(str(nc) if nc!=1 else '')+
             anion+(str(na) if na!=1 else ''))
    return formula,nc,na,divisor


def v72_reference_paths(a, center, anion):
    """Find multi-CIF references using only --ref-folder.

    Examples:
      --ref-folder ref/GeS2  (every CIF in that compound folder)
      --ref-folder ref       (prefer ref/GeS2* folders)
      --ref-folder ref      (otherwise use GeS*.cif files in ref)

    When a supplied folder contains only top-level CIFs without useful names,
    load all of them and validate the inferred composition of every file.
    """
    root=Path(a.ref_folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'--ref-folder does not exist: {root}')
    key=(center+anion).lower()
    children=sorted((p for p in root.iterdir() if p.is_dir() and
                     p.name.lower().replace('_','').startswith(key)),key=str)
    if children:
        chosen=[p for d in children for p in d.rglob('*')
                if p.is_file() and p.suffix.lower()=='.cif']
    elif root.name.lower().replace('_','').startswith(key):
        chosen=[p for p in root.rglob('*')
                if p.is_file() and p.suffix.lower()=='.cif']
    else:
        direct=[p for p in root.iterdir()
                if p.is_file() and p.suffix.lower()=='.cif']
        named=[p for p in direct if p.stem.lower().replace('_','').startswith(key)]
        chosen=named if named else direct
    paths=tuple(sorted({p.resolve() for p in chosen},key=str))
    if not paths:
        raise FileNotFoundError(f'No reference CIFs for {center}{anion} in {root}. '
                                'Use --ref-folder with a compound-specific directory.')
    return paths


def v72_mode_fingerprint(vertices):
    """Rotation/vertex-order-invariant signature of one complete polyhedron."""
    v=np.asarray(vertices,float)
    return np.r_[np.sort(np.linalg.norm(v,axis=1)),
                 np.sort(np.linalg.norm(v[:,None]-v[None],axis=-1)[
                 np.triu_indices(len(v),1)])]


def v72_reference_evidence(info, struct, path):
    """Collect complete block modes and per-reference statistics.

    Every input CIF contributes *one* center-shell statistic and volume datum,
    regardless of its cell size and number of symmetry-equivalent centers.
    """
    c,a=info['center'],info['anion']
    nc,na=info['centers_per_fu'],info['anions_per_fu']
    formula,got_nc,got_na,units=v72_formula_from_structure(struct,c,a,path)
    if (got_nc,got_na)!=(nc,na):
        raise ValueError(f'{path}: incompatible composition {formula}, expected {info["formula"]}')
    vertices=[];center_shells=[];an_min=[];ca_min=[];gaps=[]
    for i,site in enumerate(struct):
        if site.specie.symbol==c:
            shell,d=v72_reference_shell(struct,i,a,info['coordination']+1)
            vertices.append(shell[:info['coordination']])
            ca_min.append(float(d[0]))
            gaps.append(float(d[-1]-d[-2]))
            _,md=v72_reference_shell(struct,i,c,8)
            center_shells.append(md)
        elif site.specie.symbol==a:
            _,dd=v72_reference_shell(struct,i,a,1)
            an_min.append(float(dd[0]))
    if not vertices or not an_min:
        raise ValueError(f'{path}: no complete {info["center"]}{info["anion"]} local blocks')
    return dict(path=str(path),vertices=vertices,
                center_shell=np.median(center_shells,axis=0),
                anion_min=min(an_min),center_anion_min=min(ca_min),
                volume_per_fu=float(struct.volume)/units,
                smallest_gap=min(gaps),center_count=len(vertices))


def v72_merge_reference_evidence(info, reports, max_modes=6, max_strain=.11):
    """One geometry library across all CIFs, with balanced source coverage.

    Polyhedron modes are deduplicated globally by complete internal geometry,
    not by reference filename, site order, or lattice orientation.
    """
    if not reports:raise ValueError('No usable reference evidence')
    if int(max_modes)<1:raise ValueError('--max-block-modes must be positive')
    candidates=[]
    for report in reports:
        per_ref=[]
        for vertices in report['vertices']:
            sig=v72_mode_fingerprint(vertices)
            per_ref.append((np.asarray(vertices,float),sig))
        typical=np.median([sig for _,sig in per_ref],axis=0)
        per_ref.sort(key=lambda item:float(np.linalg.norm(item[1]-typical)))
        candidates.append(per_ref)
    modes=[];sources=[];fingerprints=[]
    # Round-robin selection: at most one mode per reference each round, and
    # an oversized reference cannot crowd out a reference with one local site.
    positions=[0]*len(reports)
    while len(modes)<max_modes:
        advanced=False
        for i,report in enumerate(reports):
            pool=candidates[i]
            while positions[i]<len(pool):
                v,sig=pool[positions[i]];positions[i]+=1
                advanced=True
                if any(np.linalg.norm(sig-old)<.04 for old in fingerprints):
                    continue
                modes.append(v);fingerprints.append(sig);sources.append(report['path'])
                break
            if len(modes)>=max_modes:break
        if not advanced:break
    if not modes:raise ValueError('All references lacked valid polyhedral modes')
    for report in reports:
        if report['smallest_gap']<.05:
            log(f'V72 NOTE: {report["path"]} {info["anion"]} coordination cutoff nearly '
                f'degenerate (gap {report["smallest_gap"]:.3f} A); check '
                f'--block {info["center"]}{info["anion"]}{info["coordination"]}')
    return V72BuildingBlock(
        center=info['center'],anion=info['anion'],coordination=info['coordination'],
        modes=tuple(modes),
        center_shell=np.median([r['center_shell'] for r in reports],axis=0),
        anion_core_A=float(min(r['anion_min'] for r in reports)),
        center_anion_core_A=float(min(r['center_anion_min'] for r in reports)),
        volume_per_fu=float(np.median([r['volume_per_fu'] for r in reports])),
        formula=info['formula'],centers_per_fu=info['centers_per_fu'],
        anions_per_fu=info['anions_per_fu'],sharing_degree=info['sharing_degree'],
        reference_paths=tuple(r['path'] for r in reports),
        mode_sources=tuple(sources),max_strain=max_strain)


def v72_read_block(center,block_name,paths,max_modes=6,max_strain=.11,
                   formula_override=None,anion_sharing=None):
    """Load and validate many CIFs; infer chemistry from *their* composition."""
    from pymatgen.core import Structure
    import re as _re
    m=_re.fullmatch(r'([A-Z][a-z]?)([A-Z][a-z]?)([1-9][0-9]*)',block_name)
    if m is None or m.group(1)!=center:
        raise ValueError('Use --center Ge --block GeS4 (center symbol must agree)')
    anion=m.group(2)
    info=None;reports=[]
    for path in paths:
        with warnings.catch_warnings():
            _v46_known_missing_symop_warning()
            struct=Structure.from_file(str(path))
        formula,_,_,_=v72_formula_from_structure(struct,center,anion,path)
        if formula_override is not None and formula!=formula_override:
            raise ValueError(f'{path}: CIF gives {formula}, --formula says {formula_override}')
        if info is None:
            info=v72_parse_chemistry(center,block_name,formula,anion_sharing)
        elif formula!=info['formula']:
            raise ValueError(f'{path}: mixed formulas: {formula} versus {info["formula"]}; '
                             'all references must describe one reduced composition')
        reports.append(v72_reference_evidence(info,struct,path))
    if info is None:raise ValueError('No reference CIF files loaded')
    return info,v72_merge_reference_evidence(info,reports,max_modes,max_strain),reports


def v72_wp_tokens(group,natoms,max_orbits,cap=3000):
    """Enumerate exact-multiplicity independent center Wyckoff decompositions."""
    choices=[]
    for i in range(len(group)):
        wp=group[i];multiplicity=int(wp.multiplicity)
        if 0<multiplicity<=natoms:
            choices.append((i,multiplicity,int(wp.get_dof())))
    found=[]
    def search(start,left,tok):
        if len(found)>=cap or len(tok)>max_orbits:return
        if left==0:
            found.append(tuple(tok));return
        if len(tok)>=max_orbits:return
        for j in range(start,len(choices)):
            idx,m,d=choices[j]
            if m<=left and not (d==0 and idx in tok):
                search(j if d else j+1,left-m,tok+[idx])
                if len(found)>=cap:return
    search(0,natoms,[])
    return found


def v72_diverse_tokens(tokens,limit,rng):
    by_count=collections.defaultdict(list)
    for token in tokens:by_count[len(token)].append(token)
    selected=[]
    for size in sorted(by_count):
        pick=by_count[size];rng.shuffle(pick)
        selected.extend(pick[:max(1,limit//len(by_count))])
    extras=[t for t in tokens if t not in set(selected)]
    rng.shuffle(extras)
    selected.extend(extras[:max(0,limit-len(selected))])
    return selected[:limit]


def v72_lattice(template,volume,rng,spread=.07):
    n=len(template.lattice_keys)
    raw=torch.zeros((1,max(n,1)),dtype=DTYPE)
    size=volume**(1./3.)
    raw[:,:min(n,3)]=math.log(size)+torch.as_tensor(
        rng.normal(0.,spread,size=min(n,3)),dtype=DTYPE)
    cell=np.asarray(make_lattice(template,raw)[0][0].detach().cpu(),float)
    determinant=abs(np.linalg.det(cell))
    if determinant<=1e-10:raise ValueError('Bad lattice initialization')
    return cell*(volume/determinant)**(1./3.)


def v72_nearest(frac_a,frac_b,cell,exclude_self=False):
    """Periodic minimum neighbor distance; no input atom coordinate moves."""
    A=np.asarray(frac_a,float).reshape(-1,3)
    B=np.asarray(frac_b,float).reshape(-1,3)
    if not len(A) or not len(B):raise ValueError('Empty periodic neighbor set')
    tree,_=periodic_point_tree(B,cell)
    d,_=tree.query(A@cell,k=(2 if exclude_self else 1))
    d=np.asarray(d,float)
    return np.asarray(d[:,1] if exclude_self else d,float).reshape(-1)


@dataclass(frozen=True)
class V72AnionView:
    """Read-only analytical output of arm groups; never a mutable anion state."""
    frac: np.ndarray
    labels: np.ndarray
    shifts: np.ndarray
    arm_images: np.ndarray
    arm_distances: np.ndarray
    degrees: np.ndarray


class V72SingleCenterState:
    """Exactly v64's typed WP polyhedron objects + cell + discrete arm-sharing.

    ONLY WP center coordinates, F (3rot+6strain), and opt-in cell move can
    change. Derived anions are reconstructed as arm-group centroids on demand.
    """
    def __init__(self,sg,template,block,cell,rng,a):
        self.sg=int(sg);self.template=template;self.block=block
        self.cell=np.asarray(cell,float).reshape(3,3).copy()
        self.inv=np.linalg.inv(self.cell)
        self.max_arm_shift=float(a.max_arm_shift)
        from pyxtal.symmetry import Group
        self.group_ops=w594_group_operations(Group(self.sg))
        self.polyhedra=[]
        self.topology=None
        for wp in template.blocks:
            obj=None
            for attempt in range(a.pose_attempts):
                try:
                    obj=WyckoffPolyhedron(
                        wp,self.cell,self.group_ops,{},
                        mode=int(rng.integers(len(block.modes))),
                        rotation=Rotation.random(random_state=rng),
                        max_strain=block.max_strain,
                        u=rng.random(int(wp['dof'])),
                        site_attempts=a.site_attempts,
                        seed=int(rng.integers(2**31-1)),
                        building_center_type=block.center,building_block=block)
                    break
                except WyckoffIncompatible:pass
            if obj is None:
                raise WyckoffIncompatible('No site-symmetry-compatible polyhedron for '
                                           f'WP {wp["label"]}')
            self.polyhedra.append(obj)
        self.refresh()
        centers=len(self.centers())
        if centers%block.centers_per_fu:
            raise ValueError('Center count incompatible with formula units')
        self.expected_anions=centers//block.centers_per_fu*block.anions_per_fu
        if self.expected_anions*block.sharing_degree!=centers*block.coordination:
            raise ValueError('Incorrect arm/anion allocation')

    def copy(self):return copy.deepcopy(self)

    def centers(self):
        return np.concatenate([obj.frac_sites() for obj in self.polyhedra])

    def arms(self):
        return np.concatenate([obj.positions().reshape(-1,3) for obj in self.polyhedra])

    def refresh(self,cell=None):
        if cell is not None:
            self.cell=np.asarray(cell,float).reshape(3,3).copy()
            if np.linalg.det(self.cell)<1e-9:
                raise WyckoffIncompatible('Invalid lattice')
            self.inv=np.linalg.inv(self.cell)
        for ob in self.polyhedra:
            ob.cell=self.cell.copy()
            ob.maps=w594_full_group_orbit_maps(self.cell,ob.representative(),
                                               ob.frac_sites(),self.group_ops)
            ob.site_ops=w594_site_ops(self.group_ops,ob.representative(),self.cell)
            F,B,p=w594_project_F(ob.src(),ob.site_ops,ob.F,ob.max_strain)
            if F is None or np.max(np.abs(F-ob.F))>.02:
                raise WyckoffIncompatible('Coherent F incompatible with new center/lattice')
            ob.F,ob.B,ob.permutations=F,B,p
        return self

    def _centroids(self,labels,shifts):
        labels=np.asarray(labels,int)
        shifts=np.asarray(shifts,int).reshape(-1,3)
        n=self.expected_anions
        if len(labels)!=len(self.arms()) or np.any(labels<0) or np.any(labels>=n):
            raise ValueError('Invalid discrete arm sharing')
        counts=np.bincount(labels,minlength=n)
        if np.any(counts!=self.block.sharing_degree):
            raise ValueError('Nonuniform anion sharing degree')
        positions=self.arms()@self.inv
        summed=np.zeros((n,3),float)
        np.add.at(summed,labels,positions-shifts)
        centroids=summed/counts[:,None]
        wraps=np.floor(centroids).astype(int)
        anions=centroids%1.
        image_shift=shifts+wraps[labels]
        images=(anions[labels]+image_shift)@self.cell
        d=np.linalg.norm(self.arms()-images,axis=1)
        return V72AnionView(anions,labels.copy(),image_shift,images,d,counts)

    def derived_anions(self):
        if self.topology is None:raise ValueError('Topology not discovered')
        return self._centroids(*self.topology)

    def _balanced_assignment(self,seed):
        """Assign exactly `sharing_degree` arms to each anion site.

        Each vertex chooses the best periodic *image* of each seed. The
        Hungarian assignment on repeated site slots enforces global count.
        No seed is retained as an independent anion variable.
        """
        seed=np.asarray(seed,float).reshape(-1,3)
        if len(seed)!=self.expected_anions or not np.isfinite(seed).all():
            raise ValueError('Wrong anion seeds')
        arms=self.arms(); arm_frac=arms@self.inv
        diff=arm_frac[:,None,:]-seed[None,:,:]
        base=diff-np.rint(diff)
        # Take the true minimum image in skewed conventional cells.
        periodic=base[:,:,None,:]+np.asarray(list(itertools.product((-1.,0.,1.),repeat=3)))[None,None,:,:]
        norms=np.sum((periodic@self.cell)**2,axis=-1)
        index=np.argmin(norms,axis=-1)
        offset=np.asarray(list(itertools.product((-1,0,1),repeat=3)),int)[index]
        shifts=np.rint(diff).astype(int)-offset
        score=np.take_along_axis(norms,index[:,:,None],axis=2)[:,:,0]
        targets=np.repeat(np.arange(len(seed)),self.block.sharing_degree)
        row,col=linear_sum_assignment(score[:,targets])
        assigned=targets[col]
        labels=np.empty(len(arms),int);labels[row]=assigned
        picked_shift=shifts[np.arange(len(arms)),labels]
        return labels,picked_shift

    def rebuild_topology(self,rng,starts=4,iterations=6):
        points=(self.arms()@self.inv)%1.
        n=self.expected_anions
        if n<1 or len(points)<n:raise ValueError('Incompatible anion site target')
        best=None
        for trial_id in range(max(1,starts)):
            if trial_id==0 and self.topology is not None:
                try:seed=self.derived_anions().frac
                except ValueError:seed=None
            else:seed=None
            if seed is None:
                # Periodic farthest-point initialization from proposed vertices.
                chosen=[int(rng.integers(len(points)))]
                while len(chosen)<n:
                    delta=minimum_image_delta(points[:,None,:]@self.cell,
                        points[chosen][None,:,:]@self.cell,self.cell,self.inv)
                    distance=np.min(np.sum(delta**2,axis=-1),axis=1)
                    distance[chosen]=0
                    if np.max(distance)<1e-10:
                        remain=np.setdiff1d(np.arange(len(points)),chosen)
                        chosen.append(int(rng.choice(remain)))
                    else:chosen.append(int(np.argmax(distance)))
                seed=points[chosen]
            previous=None
            for _ in range(max(1,iterations)):
                labels,shifts=self._balanced_assignment(seed)
                view=self._centroids(labels,shifts)
                seed=view.frac
                fingerprint=(tuple(labels.tolist()),tuple(shifts.ravel().tolist()))
                if fingerprint==previous:break
                previous=fingerprint
            labels,shifts=self._balanced_assignment(seed)
            view=self._centroids(labels,shifts)
            if not np.isfinite(view.frac).all():continue
            if min(v72_nearest(view.frac,view.frac,self.cell,exclude_self=True))<.05:
                continue
            error=w594_group_symmetry_error(view.frac,self.cell,self.group_ops)
            # A good topology must minimize merging *and* avoid collapsed
            # anion images; SG mismatch is separately penalized.
            score=(np.mean(view.arm_distances**2)
                    +.5*np.max(view.arm_distances)**2
                    +3.*max(0.,.10-min(v72_nearest(view.frac,view.frac,
                                      self.cell,exclude_self=True)))**2
                    +error**2)
            if best is None or score<best[0]:best=(score,view)
        if best is None:raise ValueError('No complete, distinct anion clustering')
        self.topology=(best[1].labels.copy(),best[1].shifts.copy())
        return self.derived_anions()

    def strain_metrics(self):
        vals=[];maxvals=[]
        for ob in self.polyhedra:
            singular=np.linalg.svd(ob.F,compute_uv=False)
            vals.extend((singular-1.).tolist())
            maxvals.append(float(np.max(np.abs(singular-1.))))
        return (100.*float(np.sqrt(np.mean(np.asarray(vals)**2))),
                100.*float(max(maxvals)))

    def quality(self):
        view=self.derived_anions()
        center=self.centers();anion=view.frac
        b=self.block
        merge_rms=float(np.sqrt(np.mean(view.arm_distances**2)))
        merge_max=float(np.max(view.arm_distances))
        strain_rms,strain_max=self.strain_metrics()
        core_c=.84*float(np.min(b.center_shell))
        core_a=.85*b.anion_core_A
        core_ca=.82*b.center_anion_core_A
        dc=v72_nearest(center,center,self.cell,exclude_self=True)
        da=v72_nearest(anion,anion,self.cell,exclude_self=True)
        dca=v72_nearest(center,anion,self.cell)
        ratio_c=float(min(dc)/core_c)
        ratio_a=float(min(da)/core_a)
        ratio_ca=float(min(dca)/core_ca)
        rep_c=float(np.mean(np.maximum(0.,1.12-dc/core_c)**2))
        rep_a=float(np.mean(np.maximum(0.,1.12-da/core_a)**2))
        rep_ca=float(np.mean(np.maximum(0.,1.-dca/core_ca)**2))
        sg_error=float(w594_group_symmetry_error(anion,self.cell,self.group_ops))
        scale=max(self.max_arm_shift,1e-5)
        merge_cost=10.*(merge_rms/scale)**2+4.*(merge_max/scale)**2
        score=(merge_cost +85.*rep_c+85.*rep_a+120.*rep_ca
               +12.*(strain_rms/max(100*b.max_strain,1.))**2
               +4.*(sg_error/.2)**2)
        # v64 best-N principle: stoichiometry/distinctness are mandatory;
        # nonideal chemistry is a ranking objective, not an acceptance gate.
        # The optional strict switch applies floors and reconstruction bound.
        valid=(len(center)==self.block.centers_per_fu *
                      (len(center)//self.block.centers_per_fu)
               and len(anion)==self.expected_anions and
               min(dc)>.15 and min(da)>.15 and min(dca)>.15 and
               all(ob.site_error()<1e-4 for ob in self.polyhedra))
        chemistry_ok=(ratio_c>=1 and ratio_a>=1 and ratio_ca>=1 and
                      merge_max<=self.max_arm_shift and sg_error<.05)
        return dict(score=float(score),merge_cost=float(merge_cost),
           merge_rms_A=merge_rms,merge_max_A=merge_max,
           strain_rms_pct=float(strain_rms),strain_max_pct=float(strain_max),
           center_repulsion_cost=rep_c,anion_anion_repulsion_cost=rep_a,
           center_anion_overlap_cost=rep_ca,
           center_core_ratio_min=ratio_c,anion_core_ratio_min=ratio_a,
           center_anion_core_ratio_min=ratio_ca,
           anion_symmetry_error_A=sg_error,
           share_degree_min=int(np.min(view.degrees)),
           share_degree_max=int(np.max(view.degrees)),
           n_centers=int(len(center)),n_anions=int(len(anion)),
           chemistry_ok=bool(chemistry_ok),valid=bool(valid))

    def audit(self,strict=False):
        q=self.quality()
        if not q['valid']:raise ValueError('Non-distinct or incorrect-stoichiometry structure')
        if strict and not q['chemistry_ok']:
            raise ValueError('Strict local-chemistry certificate failed')
        return q

    def write_cif(self,path,strict=False):
        from pymatgen.core import Lattice,Structure
        from pymatgen.io.cif import CifWriter
        q=self.audit(strict)
        an=self.derived_anions()
        original=Structure(Lattice(self.cell),
             [self.block.center]*q['n_centers']+[self.block.anion]*q['n_anions'],
             np.vstack((self.centers(),an.frac)),
             coords_are_cartesian=False,validate_proximity=False)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp.cif')
        try:
            CifWriter(original,symprec=None).write_file(str(tmp))
            with warnings.catch_warnings():
                _v46_known_missing_symop_warning()
                back=Structure.from_file(str(tmp))
            expected={self.block.center:q['n_centers'],self.block.anion:q['n_anions']}
            counts=dict(collections.Counter(x.specie.symbol for x in back))
            if counts!=expected or len(back)!=len(original):
                raise ValueError(f'CIF count changed on re-read: {counts} vs {expected}')
            for sp in expected:
                a=np.asarray([s.frac_coords for s in original if s.specie.symbol==sp])
                b=np.asarray([s.frac_coords for s in back if s.specie.symbol==sp])
                dist=np.linalg.norm(minimum_image_delta(
                    a[:,None,:]@self.cell,b[None,:,:]@self.cell,
                    self.cell,self.inv),axis=-1)
                rr,cc=linear_sum_assignment(dist)
                if np.max(dist[rr,cc])>5e-4:
                    raise ValueError(f'CIF modified {sp} coordinates')
            os.replace(tmp,path)
        finally:
            if tmp.exists():tmp.unlink()
        return path


def v72_polyhedron_fit(state):
    """v64 coherent fit of each entire polyhedron against merged arm groups."""
    view=state.derived_anions();off=0
    for ob in state.polyhedra:
        n=ob.count*state.block.coordination
        images=view.arm_images[off:off+n].reshape(ob.positions().shape)
        target=images-ob.frac_sites()[:,None,:]@state.cell
        ob.fit(target,mix=.55,distortion_ridge=4.)
        off+=n
    state.refresh()
    return state


def v72_center_gradient(state,rng,a):
    """Change only Wyckoff-polyhedron center u; anions are derived each trial."""
    params=[(i,k) for i,ob in enumerate(state.polyhedra) for k in range(ob.dof)]
    if not params:return []
    rng.shuffle(params);params=params[:a.gradient_budget]
    g=np.zeros(len(params))
    current=state.quality()['score']
    for j,(oi,k) in enumerate(params):
        vals=[]
        for sign in (-1,1):
            t=state.copy();t.polyhedra[oi].u[k]=(t.polyhedra[oi].u[k]+sign*a.gradient_eps)%1.
            try:t.refresh();vals.append(t.quality()['score'])
            except (ValueError,RuntimeError,np.linalg.LinAlgError):vals.append(float('nan'))
        if np.isfinite(vals).all():g[j]=(vals[1]-vals[0])/(2*a.gradient_eps)
        elif np.isfinite(vals[1]):g[j]=(vals[1]-current)/a.gradient_eps
        elif np.isfinite(vals[0]):g[j]=(current-vals[0])/a.gradient_eps
    norm=float(np.linalg.norm(g))
    if norm<1e-10:return []
    tests=[]
    for mult in (.5,1.,2.):
        t=state.copy()
        for (oi,k),v in zip(params,-g/norm*a.center_step*mult):
            t.polyhedra[oi].u[k]=(t.polyhedra[oi].u[k]+v)%1.
        try:
            t.refresh()
            delta=minimum_image_delta(t.centers()@t.cell,
                                      state.centers()@state.cell,t.cell,t.inv)
            branch='A' if np.max(np.linalg.norm(delta,axis=-1))>a.rebuild_threshold else 'B'
            if branch=='A':t.rebuild_topology(rng,a.merge_starts,a.merge_iterations)
            q=t.quality();tests.append((q['score'],t,branch+'-gradient',q))
        except (ValueError,RuntimeError,np.linalg.LinAlgError):pass
    return tests


def v72_trial_move(state,kind,rng,a):
    t=state.copy();ob=t.polyhedra[int(rng.integers(len(t.polyhedra)))]
    if kind in ('center','global-center'):
        free=[x for x in t.polyhedra if x.dof>0]
        if not free:return None
        ob=free[int(rng.integers(len(free)))]
        step=a.global_step if kind=='global-center' else a.center_step
        ob.u=(ob.u+rng.normal(0.,step,size=ob.dof))%1.
    elif kind=='rotation':
        if not ob.move('rotate',rng,a.rotation_step):return None
    elif kind=='strain':
        E=rng.normal(size=(3,3));E=(E+E.T)/2
        E*=a.strain_step/max(np.linalg.norm(E),1e-10)
        F,B,p=w594_project_F(ob.src(),ob.site_ops,ob.F+E,ob.max_strain,
                               ob.permutations)
        if F is None:return None
        ob.F,ob.B,ob.permutations=F,B,p
    elif kind=='lattice':
        if not a.relax_lattice:return None
        directions=v64_lattice_directions(state.template.lattice_type)
        if not directions:return None
        direction=np.asarray(directions[int(rng.integers(len(directions)))],float)
        factor=np.exp(float(rng.choice([-1,1]))*a.lattice_step*direction)
        if np.max(np.abs(np.log(factor)))>a.max_lattice_change:return None
        return t.refresh(t.cell*factor[:,None])
    else:raise ValueError(kind)
    t.refresh();return t


def v72_optimize(initial,a,seed):
    rng=np.random.default_rng(seed)
    current=initial.copy();current.rebuild_topology(rng,a.merge_starts,a.merge_iterations)
    q=current.quality();score=q['score']
    best=current.copy();best_score=score
    valid=current.copy() if q['valid'] else None
    valid_score=score if valid is not None else math.inf
    counters=collections.Counter();history=[]
    for cycle in range(1,a.cycles+1):
        proposals=[]
        try:
            t=current.copy();v72_polyhedron_fit(t);qq=t.quality()
            proposals.append((qq['score'],t,'B-fit',qq))
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            counters['fit_failure']+=1
        try:proposals.extend(v72_center_gradient(current,rng,a))
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            counters['gradient_failure']+=1
        for _ in range(a.local_proposals):
            kind=str(rng.choice(['center','rotation','strain']))
            try:
                t=v72_trial_move(current,kind,rng,a)
                if t is None:continue
                branch='B'
                if kind=='center':
                    delta=minimum_image_delta(t.centers()@t.cell,
                                    current.centers()@current.cell,t.cell,t.inv)
                    if np.max(np.linalg.norm(delta,axis=-1))>a.rebuild_threshold:
                        t.rebuild_topology(rng,a.merge_starts,a.merge_iterations)
                        branch='A'
                qq=t.quality();proposals.append((qq['score'],t,branch+'-'+kind,qq))
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                counters['proposal_failure']+=1
        if cycle%a.global_every==0:
            for kind in ('global-center','lattice'):
                try:
                    t=v72_trial_move(current,kind,rng,a)
                    if t is None:continue
                    t.rebuild_topology(rng,a.merge_starts,a.merge_iterations)
                    qq=t.quality();proposals.append((qq['score'],t,'A-'+kind,qq))
                except (ValueError,RuntimeError,np.linalg.LinAlgError):
                    counters['global_failure']+=1
        if cycle%a.rebuild_every==0:
            try:
                t=current.copy();t.rebuild_topology(rng,a.merge_starts,a.merge_iterations)
                qq=t.quality();proposals.append((qq['score'],t,'A-topology',qq))
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                counters['rebuild_failure']+=1
        for ss,tt,_,qq in proposals:
            if qq['valid'] and ss<valid_score:
                valid=tt.copy();valid_score=ss
        if proposals:
            proposals.sort(key=lambda r:r[0]);ss,tt,kind,qq=proposals[0]
            delta=ss-score
            temp=a.temperature*max(0.,1.-cycle/a.cycles)
            if delta<-1e-7 or (temp>0 and delta>0 and
                                rng.random()<math.exp(-delta/temp)):
                current=tt;score=ss;counters['accepted-'+kind]+=1
        if score<best_score:best=current.copy();best_score=score
        qq=current.quality()
        if qq['valid'] and score<valid_score:
            valid=current.copy();valid_score=score
        history.append(dict(cycle=cycle,score=float(score),
             merge_rms_A=qq['merge_rms_A'],merge_max_A=qq['merge_max_A'],
             strain_rms_pct=qq['strain_rms_pct'],
             center_repulsion_cost=qq['center_repulsion_cost'],
             anion_anion_repulsion_cost=qq['anion_anion_repulsion_cost']))
    return dict(best=best,valid=valid,history=history,counters=dict(counters),
                score=float(best_score),valid_score=(float(valid_score)
                                  if np.isfinite(valid_score) else None))


def v72_search_one(job):
    sg,params,block,seed=job
    a=argparse.Namespace(**params)
    rng=np.random.default_rng(seed)
    output=dict(sg=sg,attempts=0,wp_count=0,candidates=[],errors=collections.Counter(),
                trajectories=[])
    from pyxtal.symmetry import Group
    group=Group(sg)
    ncenters=a.formula_units*block.centers_per_fu
    all_tokens=v72_wp_tokens(group,ncenters,a.max_center_orbits,a.wp_pool)
    output['wp_count']=len(all_tokens)
    chosen=v72_diverse_tokens(all_tokens,a.wp_per_sg,rng)
    retained=[]
    for tok in chosen:
        template=template_from_token(sg,tok,'cpu')
        for basin in range(a.basins):
            output['attempts']+=1
            try:
                volume=block.volume_per_fu*a.formula_units*a.volume_scale
                volume*=float(np.exp(rng.normal(0.,a.volume_jitter)))
                cell=v72_lattice(template,volume,rng)
                state=V72SingleCenterState(sg,template,block,cell,rng,a)
                result=v72_optimize(state,a,int(rng.integers(2**31-1)))
                best=result['best']
                q=best.quality()
                output['trajectories'].append(dict(
                    wp=list(tok),basin=basin,score=result['score'],
                    quality=q,counts=result['counters'],cycles=result['history']))
                finished=result['valid']
                if finished is None:
                    output['errors']['no_valid_sharing']+=1;continue
                qq=finished.audit(a.strict_chemistry)
                # Approximate symmetry-independent geometry fingerprint.
                # The complete state is always retained for CIF export.
                fingerprint=np.r_[np.sort(np.linalg.norm(finished.cell,axis=1)),
                      [qq['merge_rms_A'],qq['merge_max_A'],
                       qq['center_core_ratio_min'],qq['anion_core_ratio_min']]]
                duplicate=next((i for i,(old,fp,prev) in enumerate(retained)
                     if np.linalg.norm(fp-fingerprint)/np.sqrt(len(fp))<.025),None)
                item=(finished,fingerprint,qq)
                if duplicate is None:retained.append(item)
                elif qq['score']<retained[duplicate][2]['score']:
                    retained[duplicate]=item
                retained.sort(key=lambda record:record[2]['score'])
                retained=retained[:a.max_unique_per_sg]
            except Exception as exc:
                if a.fail_fast:raise
                output['errors'][f'{type(exc).__name__}:{str(exc)[:95]}']+=1
    output['candidates']=retained
    return output


def v72_parser():
    p=argparse.ArgumentParser(description='JULIETTE v73: cleaned, multi-reference v64 single-center polyhedra', allow_abbrev=False)
    p.add_argument('--center',default='Ge',help='Building-center element, e.g. Ti, Ge, B')
    p.add_argument('--block',default='GeS4',help='Full block: TiO6, GeS4, BO3, BP4, ...')
    p.add_argument('--engine',choices=['auto','legacy','generic'],default='auto',
                   help='auto preserves original v64 GPU TiO2; all other blocks use generic CPU')
    p.add_argument('--ref-folder',default='ref')
    p.add_argument('--space-groups',default='1')
    p.add_argument('--formula-units',type=int,default=12)
    p.add_argument('--output',default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=1,
                   help='Generic engine uses CPU workers, not GPU kernels')
    p.add_argument('--max-unique-per-sg',type=int,default=10)
    p.add_argument('--wp-per-sg',type=int,default=4)
    p.add_argument('--wp-pool',type=int,default=500)
    p.add_argument('--max-center-orbits',type=int,default=24)
    p.add_argument('--basins',type=int,default=3)
    p.add_argument('--cycles',type=int,default=20)
    p.add_argument('--local-proposals',type=int,default=3)
    p.add_argument('--gradient-budget',type=int,default=3)
    p.add_argument('--gradient-eps',type=float,default=.004)
    p.add_argument('--center-step',type=float,default=.022)
    p.add_argument('--rebuild-threshold',type=float,default=.30)
    p.add_argument('--rotation-step',type=float,default=.085)
    p.add_argument('--strain-step',type=float,default=.025)
    p.add_argument('--global-step',type=float,default=.095)
    p.add_argument('--global-every',type=int,default=4)
    p.add_argument('--rebuild-every',type=int,default=6)
    p.add_argument('--merge-starts',type=int,default=4)
    p.add_argument('--merge-iterations',type=int,default=6)
    p.add_argument('--max-arm-shift',type=float,default=1.,
                   help='Angstrom normalization for merge-cost reporting/scoring')
    p.add_argument('--max-strain',type=float,default=.11)
    p.add_argument('--max-block-modes',type=int,default=6)
    p.add_argument('--pose-attempts',type=int,default=4)
    p.add_argument('--site-attempts',type=int,default=12)
    p.add_argument('--temperature',type=float,default=.08)
    p.add_argument('--relax-lattice',action='store_true')
    p.add_argument('--lattice-step',type=float,default=.025)
    p.add_argument('--max-lattice-change',type=float,default=.12)
    p.add_argument('--volume-scale',type=float,default=1.)
    p.add_argument('--volume-jitter',type=float,default=.07)
    p.add_argument('--strict-chemistry',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--v73-self-test',action='store_true')
    return p


def v72_self_test():
    """Reference-free validation: BO3/BP4/GeS4/TiO6 parsing; sharing + v64 F."""
    examples={'TiO6':('Ti','TiO2',3),'GeS4':('Ge','GeS2',2),
              'BO3':('B','B2O3',2),'BP4':('B','BP',4)}
    for block,(center,formula,share) in examples.items():
        result=v72_parse_chemistry(center,block)
        assert result['formula']==formula and result['sharing_degree']==share
    assert v72_parse_chemistry('Al','AlF4',anion_sharing=2)['formula']=='AlF2'
    try:v72_parse_chemistry('Ge','GeS4',formula='GeS3')
    except ValueError:pass
    else:raise AssertionError('Noninteger sharing must fail')
    wp=dict(label='general',dof=3,
       A=torch.eye(3,dtype=DTYPE),b=torch.zeros(3,dtype=DTYPE),
       rot=torch.eye(3,dtype=DTYPE)[None],
       tra=torch.zeros((1,3),dtype=DTYPE))
    cell=np.eye(3)*9.
    tetra=np.asarray([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
    tetra*=2./np.sqrt(3)
    info=v72_parse_chemistry('Ge','GeS4')
    block=V72BuildingBlock('Ge','S',4,(tetra,),np.full(8,4.),
           2.5,2.,729.,'GeS2',1,2,2,('synthetic',))
    obj=WyckoffPolyhedron(wp,cell,[(np.eye(3),np.zeros(3))],{},0,
         Rotation.identity(),u=np.array([.5,.5,.5]),
         building_center_type='Ge',building_block=block)
    assert obj.src().shape==(4,3)
    state=object.__new__(V72SingleCenterState)
    state.sg=1;state.template=argparse.Namespace(lattice_type='cubic')
    state.block=block;state.cell=cell;state.inv=np.linalg.inv(cell)
    state.polyhedra=[obj];state.group_ops=[(np.eye(3),np.zeros(3))]
    state.expected_anions=2;state.max_arm_shift=1.;state.topology=None
    rng=np.random.default_rng(22)
    view=state.rebuild_topology(rng,starts=3,iterations=4)
    assert len(view.frac)==2 and np.all(view.degrees==2)
    assert len(view.labels)==4 and np.isfinite(view.arm_distances).all()
    assert state.block is block and not hasattr(state,'anion_orbits')
    assert state.derived_anions().frac.shape==(2,3)
    q=state.quality();assert q['n_centers']==1 and q['n_anions']==2
    assert all(key in q for key in ('merge_cost','strain_rms_pct',
                                   'center_repulsion_cost','anion_anion_repulsion_cost'))
    # A coherent center move changes every derived anion; there are no anion DOFs.
    old=state.derived_anions().frac.copy()
    moved=state.copy();moved.polyhedra[0].u+=.01;moved.refresh()
    assert not np.allclose(moved.derived_anions().frac,old)
    # v72 tests that need no installed pymatgen: source discovery, formula
    # reduction, strict incompatible-composition error and equal-source modes.
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    class FakeSite:
        def __init__(self,element):
            self.specie=SimpleNamespace(symbol=element)
            self.occupancy=1.
    fake=[FakeSite(x) for x in ['Ge','Ge','S','S','S','S']]
    assert v72_formula_from_structure(fake,'Ge','S')[0]=='GeS2'
    try:v72_formula_from_structure(fake,'Ge','O')
    except ValueError:pass
    else:raise AssertionError('Mismatched element was accepted')
    with TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'GeS2').mkdir()
        (root/'GeS2'/'a.cif').write_text('data_a')
        (root/'GeS2'/'b.CIF').write_text('data_b')
        (root/'TiO2.cif').write_text('data_tio2')
        aa=argparse.Namespace(ref_folder=str(root))
        selected=v72_reference_paths(aa,'Ge','S')
        assert len(selected)==2 and all(p.parent.name=='GeS2' for p in selected)
        # Passing a chemistry-specific folder accepts arbitrary CIF basenames.
        aa.ref_folder=str(root/'GeS2')
        assert len(v72_reference_paths(aa,'Ge','S'))==2
        (root/'GeS2'/'a.cif').rename(root/'GeS2'/'alpha.cif')
        assert len(v72_reference_paths(aa,'Ge','S'))==2
        aa.ref_folder=str(root)
        assert len(v72_reference_paths(aa,'Ge','S'))==2
    info=v72_parse_chemistry('Ge','GeS4','GeS2')
    t=np.asarray([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
    ra=dict(path='a.cif',vertices=[t,t*.999],center_shell=np.full(8,3.),
            anion_min=2.,center_anion_min=1.9,volume_per_fu=100.,
            smallest_gap=.5,center_count=100)
    rb=dict(path='b.cif',vertices=[t*1.15],center_shell=np.full(8,5.),
            anion_min=1.7,center_anion_min=2.1,volume_per_fu=160.,
            smallest_gap=.5,center_count=1)
    combined=v72_merge_reference_evidence(info,[ra,rb],max_modes=4)
    assert len(combined.modes)==2 and combined.mode_sources==('a.cif','b.cif')
    assert combined.volume_per_fu==130. and np.allclose(combined.center_shell,4.)
    assert combined.anion_core_A==1.7 and combined.reference_paths==('a.cif','b.cif')
    # The center-count disparity must not affect per-reference averaging.
    # Check full v64 TiO2 object-integrity path independently.
    v64_self_test()
    log('V72 SELF-TEST PASSED: multi-CIF inference and balanced mode union; arbitrary vertex count; exact sharing ratio; '
        'purely derived anions; v64 symmetry-constrained affine polyhedron; '
        'original v64 state self-test')
    return True


def v72_main(argv=None):
    a=v72_parser().parse_args(argv)
    if a.v73_self_test:return v73_self_test()
    if a.engine=='legacy':
        raise ValueError('Legacy mode only supports TiO6/TiO2: use --engine auto')
    for name in ('formula_units','workers','wp_per_sg','wp_pool','basins','cycles',
                 'merge_starts','merge_iterations','global_every','rebuild_every'):
        if getattr(a,name)<1:raise ValueError(f'{name} must be >=1')
    sgs=parse_sgs(a.space_groups)
    # The block itself determines the anion symbol; the CIF determines the ratio.
    import re as _re
    match=_re.fullmatch(r'([A-Z][a-z]?)([A-Z][a-z]?)([1-9][0-9]*)',a.block)
    if match is None or match.group(1)!=a.center:
        raise ValueError('Expected --center Ge --block GeS4')
    refs=v72_reference_paths(a,a.center,match.group(2))
    info,block,reference_reports=v72_read_block(
        a.center,a.block,refs,a.max_block_modes,a.max_strain,
        None,None)
    if a.plan_only:
        log(f'V73 PLAN: {info["formula"]} {a.block} SGs={sgs} '
            f'FU={a.formula_units} centers={a.formula_units*info["centers_per_fu"]} '
            f'arms={a.formula_units*info["centers_per_fu"]*info["coordination"]} '
            f'anions={a.formula_units*info["anions_per_fu"]} '
            f'sharing={info["sharing_degree"]} CPU workers={a.workers} '
            f'cifs={len(refs)} modes={len(block.modes)}')
        return None
    out=Path(a.output or f'{info["formula"].lower()}_v73')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Nonempty output {out}; choose a fresh output directory')
    (out/'cif').mkdir(parents=True,exist_ok=True)
    log(f'JULIETTE v73 | generic v64 WP polyhedra | '
        f'{info["formula"]} {a.block} | {len(sgs)} SGs '
        f'{a.formula_units} FU {a.workers} CPU workers')
    log(f'  {info["coordination"]*info["centers_per_fu"]} proposed vertices per FU '
        f'-> {info["anions_per_fu"]} derived {info["anion"]} '
        f'(sharing {info["sharing_degree"]}:1); '
        f'references={len(refs)} unique block modes={len(block.modes)}')
    for r in reference_reports:
        log(f'    source {Path(r["path"]).name}: centers={r["center_count"]} '
            f'volume/FU={r["volume_per_fu"]:.3f} A^3')
    log('  ALL anion positions derived from WP polyhedra; NO anion variables; '
        'rank: strain, vertex merging, center and anion repulsion')
    jobs=[(sg,vars(a),block,str_seed(a.seed,sg)) for sg in sgs]
    reports=[];t0=time.perf_counter()
    if a.workers==1:
        for job in jobs:reports.append(v72_search_one(job))
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=a.workers,
                     mp_context=get_context('spawn')) as pool:
            pending={pool.submit(v72_search_one,job):job[0] for job in jobs}
            for future in as_completed(pending):reports.append(future.result())
    records=[]
    for r in sorted(reports,key=lambda v:v['sg']):
        (out/f'SG{r["sg"]:03d}_trajectories.json').write_text(
            json.dumps(r['trajectories'],indent=2))
        if r['candidates']:
            q=r['candidates'][0][2]
            log(f' SG{r["sg"]:03d}: WPs={r["wp_count"]} attempts={r["attempts"]} '
                f'accepted={len(r["candidates"])} '
                f'merge={q["merge_rms_A"]:.3f}/{q["merge_max_A"]:.3f}A '
                f'strain={q["strain_rms_pct"]:.1f}/{q["strain_max_pct"]:.1f}% '
                f'center_rep={q["center_repulsion_cost"]:.4g} '
                f'{block.anion}-{block.anion}_rep={q["anion_anion_repulsion_cost"]:.4g}')
        else:
            log(f' SG{r["sg"]:03d}: WPs={r["wp_count"]} '
                f'attempts={r["attempts"]} no CIF; '
                f'rejects={dict(r["errors"])}')
        for rank,(state,fp,q) in enumerate(r['candidates'],1):
            name=f'SG{r["sg"]:03d}_R{rank:02d}_{info["formula"]}_v73.cif'
            try:
                state.write_cif(out/'cif'/name,a.strict_chemistry)
                records.append(dict(initial_SG=r['sg'],rank=rank,CIF=name,**q))
            except Exception as exc:
                if a.fail_fast:raise
                log(f'  CIF export rejected SG{r["sg"]:03d} rank={rank}: {exc}')
    keys=list(dict.fromkeys(k for row in records for k in row)) or ['initial_SG','rank','CIF','score']
    with (out/'combined_structures.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader();writer.writerows(records)
    summary=dict(version='v73',baseline='v64',engine='generic CPU',
        chemistry=info,reference_cifs=[str(p) for p in refs],
        reference_stats=[{k:v for k,v in r.items() if k!='vertices'
                         and not isinstance(v,np.ndarray)} for r in reference_reports],
        block_mode_sources=list(block.mode_sources),block_modes=len(block.modes),
        settings=vars(a),
        elapsed_s=time.perf_counter()-t0,final_CIFs=len(records),
        per_sg=[dict(sg=r['sg'],wp_candidates=r['wp_count'],
                     attempts=r['attempts'],kept=len(r['candidates']),
                     rejected=dict(r['errors'])) for r in reports])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    log(f'V73 COMPLETE: SGs={len(reports)} CIFs={len(records)} '
        f'wall={summary["elapsed_s"]:.1f}s output={out}')
    return records


def v73_dispatch(argv=None):
    """A single public --ref-folder, correct TiO2 GPU child routing.

    v64's internal multiprocessing argv still uses its old --ref argument.
    It is private: --ref-folder is converted once in the parent. A spawned
    child must run v61_main directly; otherwise default generic GeS4 routing
    misreads the worker's private --ref and exits before any SG is searched.
    """
    argv=list(sys.argv[1:] if argv is None else argv)
    if os.environ.get('JULIETTE_GPU_CHILD')=='1':
        return v61_main(argv)
    old_flags={'--ref','--reference-dir','--reference-cif'}
    for token in argv:
        if token.split('=',1)[0] in old_flags:
            raise ValueError(f'{token.split("=",1)[0]} is not a public option; '
                             'use --ref-folder PATH for both engines')
    if '--v73-self-test' in argv:
        return v73_self_test()
    router=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for flag in ('--center','--block','--engine','--ref-folder'):
        router.add_argument(flag)
    opt,remainder=router.parse_known_args(argv)
    center=opt.center or 'Ge'
    block=opt.block or 'GeS4'
    engine=opt.engine or 'auto'
    if engine not in ('auto','legacy','generic'):
        raise ValueError('--engine must be auto, legacy or generic')
    legacy=(engine=='legacy' or
            (engine=='auto' and center=='Ti' and block=='TiO6'))
    if legacy:
        if center!='Ti' or block!='TiO6':
            raise ValueError('Original v64 GPU engine is available only for Ti/TiO6')
        # Only this internal v64 boundary understands --ref. The public CLI,
        # documentation, and generic engine use exclusively --ref-folder.
        if opt.ref_folder is not None:
            remainder.extend(['--ref',opt.ref_folder])
        return v614_main(remainder)
    return v72_main(argv)


def v73_self_test():
    """Test multi-reference policies, public CLI and GPU-worker dispatch."""
    v72_self_test()
    p=v72_parser()
    args=p.parse_args(['--center','Ge','--block','GeS4',
                       '--ref-folder','ref/GeS2','--formula-units','12'])
    assert args.ref_folder=='ref/GeS2'
    assert not hasattr(args,'reference_dir') and not hasattr(args,'reference_cif')
    assert not hasattr(args,'formula') and not hasattr(args,'anion_sharing')
    # Confirm an actual GPU-child CLI goes straight to v61_main, preserving
    # internal v64 --ref, even when center/block were stripped by parent.
    sentinel=object()
    original=globals()['v61_main']
    old=os.environ.get('JULIETTE_GPU_CHILD')
    calls=[]
    try:
        globals()['v61_main']=lambda argv: calls.append(argv) or sentinel
        os.environ['JULIETTE_GPU_CHILD']='1'
        result=v73_dispatch(['--ref','ref_tio2','--space-groups','206',
                             '--v64-coupled-cycles','6'])
        assert result is sentinel and calls==[['--ref','ref_tio2',
              '--space-groups','206','--v64-coupled-cycles','6']]
    finally:
        globals()['v61_main']=original
        if old is None:os.environ.pop('JULIETTE_GPU_CHILD',None)
        else:os.environ['JULIETTE_GPU_CHILD']=old
    cmd=v614_worker_command(['--ref','ref_tio2','--gpu-workers','4',
                             '--space-groups','168-230'],[206],Path('tmp'))
    assert cmd.count('--ref')==1 and '--gpu-workers' in cmd
    try:
        v73_dispatch(['--center','Ti','--block','TiO6','--ref','ref_tio2'])
    except ValueError as exc:
        assert '--ref-folder' in str(exc)
    else:
        raise AssertionError('Old public --ref must fail with useful message')
    log('V73 SELF-TEST PASSED: v64 state + multi-reference inference + '
        'ref-folder-only CLI + direct GPU-child dispatch')
    return True


if __name__=='__main__':
    try:
        v73_dispatch()
    except Exception:
        # Print the full traceback to stdout, not a hidden worker stderr.
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)

