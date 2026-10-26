#!/usr/bin/env python3
"""JULIETTE v59.2 — ranked top-N TiO2 search (24 formula units by default).

Retains A+ Ti basin selection and complementary Ti-first / O-first oxygen
promotion. The hard Ti/O coordination and oxygen-sharing checks are diagnostics
rather than acceptance gates. Every finite Ti_n O_2n reconstructed candidate
can compete in a globally deduplicated, per-initial-SG top-K archive ranked by
worst post-oxygen per-Ti chemical loss (ascending).

Only the final top-K structures are written, once, at end of search. The
--stop-sg-at-cap option permits early SG capacity exhaustion after a complete
Wyckoff task; turn it off for strict best-K over the entire configured search.
The optional --no-o-cn-target still disables only the smooth oxygen CN3
optimizer penalty, leaving final CN3 measurement and reporting active.
"""
from __future__ import annotations

import argparse

import collections

import copy

import concurrent.futures

import multiprocessing

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
            sv=.065*torch.tanh(strains)
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
            exceed_o=torch.relu(res_o-o_lim)
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


import sqlite3


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

if __name__=='__main__':
    main()
