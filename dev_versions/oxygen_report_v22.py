#!/usr/bin/env python3
"""Juliette v22r1 publication report: standalone TiO6-first / LCAO-SCF benchmark.

Single script. No imports from previous Juliette versions or local modules.
Four reference CIFs supply all learned octahedral geometry and bond shells.
Other original database CIFs calibrate acceptance thresholds; each target is
excluded from its own calibration cohort, and target oxygen is used only for
full-structure duplicate screening and post-generation diagnostics. Only Ti and
lattice from the target enter oxygen construction, LCAO, SCF and global search.

Dependencies: numpy, scipy, pytorch, pymatgen, spglib, ase.
Output: one central publication tree and per-structure progress files.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import itertools
import math
import time
import traceback
import os
import re
import sys
import subprocess
import signal
import threading
import pickle
import hashlib
import multiprocessing as mp
import contextlib
import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DTYPE=torch.float64
# Species channels and their representations.  Primary shell is checked
# separately, extended shell includes radial + pair geometry, cross joins
# first-shell positions to other-species near neighbors.
CHANNELS={
    'Ti': [('O', 6, 10), ('Ti', 0, 8)],
    'O':  [('Ti', 3, 7), ('O', 0, 10)],
}
SHIFTS=np.asarray(list(itertools.product([-1.,0.,1.],repeat=3)),float)
LOCAL_SHIFTS=SHIFTS.copy()  # cheap 27-image Ti/LCAO proposal trees


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


def descriptor_np(primary_vec,other_vec,primary_count, primary_total):
    """Four groups, all in Angstrom: primary geometry, primary extension,
    other-species geometry, cross-species spatial relationships."""
    d=np.linalg.norm(primary_vec,axis=1)
    q=primary_vec[:primary_count]
    pairs=np.linalg.norm(q[:,None]-q[None,:],axis=-1)
    p=np.r_[np.sort(d[:primary_count]),np.sort(pairs[np.triu_indices(len(q),1)])]
    tail=np.sort(d[primary_count:primary_total])
    d2=np.linalg.norm(other_vec,axis=1)
    otherpair=np.linalg.norm(other_vec[:,None]-other_vec[None,:],axis=-1)
    e=np.r_[np.sort(d2),np.sort(otherpair[np.triu_indices(len(other_vec),1)])]
    cross=np.sort(np.linalg.norm(q[:,None,:]-other_vec[None,:,:],axis=-1).ravel())
    return [p,tail,e,cross]


def describe_reference(s,index,center_sp,counts):
    # Debug/helper entry point; the bulk learner uses the batched implementation.
    subindex=sum(x.specie.symbol==center_sp for x in s[:index])
    return reference_descriptor_matrices(s,counts)[center_sp][subindex]


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
    def __init__(self,ti,cell,lib,bounds,oo_guard,accept=.10,hierarchy="ti-first",calibration=None):
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
            localO=score['O']+.65*(cnO-3)**2+.7*collision
        else:
            # Keep connection RULES strong (soft CN 6/3, contact guards),
            # independently of weak, *learned* O-side topology preference.
            # Coordination remains a smooth optimizer signal, never a hard
            # candidate-rejection/SCF-stopping condition.
            localTi=score['Ti']+1.20*(cnTi-6)**2+.65*toshort
            localO=score['O']+1.20*(cnO-3)**2+1.40*collision
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


def candidate_metrics(candidate):
    d=candidate['details']
    stretch=candidate['oct'].stretch-np.eye(3)
    octstrain=np.sqrt(np.sum(stretch**2,axis=(1,2)))
    # Measured in Angstrom, not only unitless strain-matrix norm.
    # Relative to each final unit's learned mode, at its optimized orientation;
    # this is NOT the cumulative displacement from the initial SCF pose.
    arm_displacements=[]
    for i,mode in enumerate(candidate['oct'].mode):
        learned=candidate['oct'].lib['Ti'][int(mode)]['rigid_arms']
        arm_displacements.extend(np.linalg.norm(learned@stretch[i],axis=1))
    arm_displacements=np.asarray(arm_displacements)
    n_ti=len(d['hardTi']);n_o=len(d['hardO'])
    return dict(
        coordination_ok=(d['topologyBadTi']==0 and d['topologyBadO']==0),
        sharing_unique=(candidate['bad_triplets']==0),
        topology_ok=(d['topologyBadTi']==0 and d['topologyBadO']==0
                     and candidate['bad_triplets']==0),
        learned_geometry_ok=(d['geometryBadTi']==0 and d['geometryBadO']==0),
        # Chemistry-focused success: strict learned Ti environment and
        # consistent connected TiO2, independent of O extended-mode mismatch.
        ti_first_quality_ok=(d['geometryBadTi']==0 and
                            d['topologyBadTi']==0 and d['topologyBadO']==0 and
                            candidate['bad_triplets']==0 and candidate['rms']<.12 and
                            d['collision']<1e-8),
        ti_geometry_ok=(d['geometryBadTi']==0),
        hierarchy=d.get('hierarchy','v20'),
        ti_geometry_loss=d.get('geometryLossTi',float('nan')),
        o_topology_guide_loss=d.get('geometryLossO',float('nan')),
        topology_ti_fraction=float(np.mean(d['topologyTi'])) if n_ti else float('nan'),
        topology_o_fraction=float(np.mean(d['topologyO'])) if n_o else float('nan'),
        learned_geometry_ti_fraction=float(np.mean(d['geometryTi'])) if n_ti else float('nan'),
        learned_geometry_o_fraction=float(np.mean(d['geometryO'])) if n_o else float('nan'),
        mean_octahedron_strain=float(np.mean(octstrain)),
        maximum_octahedron_strain=float(np.max(octstrain)),
        learned_template_arm_displacement_rms=float(np.sqrt(np.mean(arm_displacements**2))),
        learned_template_arm_displacement_max=float(np.max(arm_displacements)),
        learned_rms_ti=np.mean(d['rTi'],axis=0).tolist(),
        learned_rms_o=np.mean(d['rO'],axis=0).tolist(),
        worst_learned_rms=float(max(np.max(d['rTi']),np.max(d['rO']))),
        topology_bad_ti=d['topologyBadTi'],topology_bad_o=d['topologyBadO'],
        geometry_bad_ti=d['geometryBadTi'],geometry_bad_o=d['geometryBadO'],
        oxygen_sharing_rms=float(candidate['rms']),
        sharing_triplets_nonunique=int(candidate['bad_triplets']),
        total_objective=continuous_candidate_objective(candidate),
        ti_cn_soft_loss=d.get('connectionCNlossTi',float('nan')),
        o_cn_soft_loss=d.get('connectionCNlossO',float('nan')))


def report(chem,o,label,detail=True):
    d=chem.score(torch.tensor(np.asarray(o),dtype=DTYPE),details=True)
    print(f'  [{label}] loss={d["energy"]:.6g} worst={d["worst"]:.5g} '
          f'primary_bad(Ti,O)=({d["primaryBadTi"]},{d["primaryBadO"]}) '
          f'full_bad(Ti,O)=({d["badTi"]},{d["badO"]}) '
          f'O--O_collision={d["collision"]:.5g}',flush=True)
    for sp in ('Ti','O'):
        r=d['rTi' if sp=='Ti' else 'rO']; mode=d['modeTi' if sp=='Ti' else 'modeO'];
        hard=d['hardTi' if sp=='Ti' else 'hardO']; good=d['goodTi' if sp=='Ti' else 'goodO'];
        target=6 if sp=='Ti' else 3
        print(f'     {sp} channel RMS A (primary,outer,secondary,cross) max='
              f'({",".join(f"{a:.3f}" for a in r.max(axis=0))}); '
              f'hard_CN={dict((int(k),int(v)) for k,v in zip(*np.unique(hard,return_counts=True)))}',flush=True)
        if detail:
            for i in range(len(good)):
                if not good[i]:
                    print(f'       BAD {sp}{i+1:02d}: CN={hard[i]}/{target} mode={mode[i]} '
                          f'channels=[{",".join(f"{a:.3f}" for a in r[i])}]',flush=True)
    return d



# ---------------- Ti-lattice OTi3 construction (no O targets) ----------------
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



# ----------------- Complete-octahedron coherent search -----------------

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
        if verbose:
            print(f'    SCF cycle {step+1:03d} rms={rms:.4f} A '
                  f'bad_O_triplets={duplicate} oct_strain={octas.distortion():.4f} '
                  f'compatible_Ti_symops={len(symmetry)}/{len(ops)}',flush=True)
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
    final.sort(key=continuous_candidate_objective)
    winner=final[0]
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
    working=result['oct'].copy()
    cell=working.cell;inv=working.inv;n=len(working.ti)
    one=torch.eye(3,dtype=DTYPE)
    ci=torch.tensor(cell,dtype=DTYPE)
    ci_inv=torch.tensor(inv,dtype=DTYPE)
    ti_cart=torch.tensor(working.ti@cell,dtype=DTYPE)
    best=result.copy()
    current_centers=result['oxygen']@cell
    total_done=0
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
        if round_best is None:
            break
        # Rebuild the LCAO network for the next collective SCF update.
        working=round_best['oct'].copy()
        current_centers=round_best['oxygen']@cell
    best['chem_feedback_steps']=total_done
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


def final_report(results,library,bounds,guard,started,args):
    print('\n'+'='*92)
    print('JULIETTE v21 — REFERENCE-SIDE COHERENT TiO6 / LCAO–SCF DIAGNOSTICS')
    print('='*92)
    print('TRAINING: anatase, brookite, hollandite, rutile; no database input.')
    print('Database O: used for full TiO2 duplicate screening and optional post-hoc comparison ONLY; not for generation.')
    print('SCF feedback includes full learned O-centered geometry; ONLY TiO6 orientation/strain are variable.')
    print('Dynamics: six-vertex TiO6 objects only; no independent oxygen motion or O Wyckoff orbit.')
    print('Topology: exact six O per Ti, three distinct periodic Ti-image neighbors per O; geometric acceptance: learned local channels.')
    print(f'Settings: starts={args.starts}, cycles={args.cycles}, max-O-peaks={args.candidate_cap}, '
          f'max-modes={args.modes_per_ti}, accept={args.accept_A:.3f} A, '
          f'strict_coincidence_RMS<0.120 A, time={time.perf_counter()-started:.1f} s.')
    print(f'LEARNED Ti-O cutoff={bounds["Ti"][0]:.4f} A, O-Ti cutoff={bounds["O"][0]:.4f} A, '
          f'O-O collision guard={guard:.4f} A')
    for name,res in results:
        print('\n'+'-'*92)
        print(f'REFERENCE {name.upper()}: {res["status"]}')
        if 'error' in res:
            print(f'  Failure: {res["error"]}')
            continue
        print(f'  Ti={res["nTi"]}, generated O={res["nO"]}, '
              f'Ti-only symmetry operations={res["operations"]}, '
              f'OTi3 candidate peaks={res["candidates"]}.')
        for i,cand in enumerate(res['starts']):
            d=cand['details']
            print(f'  START {i+1}: legacy_strict_diagnostic={cand["pass_chemistry"]} '
                  f'SCF={cand["cycles"]} entrance_RMS={cand["entrance_rms"]:.4f} A '
                  f'final_RMS={cand["rms"]:.4f} A bad_triplets={cand["bad_triplets"]} '
                  f'badTi={d["badTi"]} badO={d["badO"]} '
                  f'mean_oct_strain={cand["oct"].distortion():.4f} '
                  f'symmetry={cand["compatible_symops"]}/{cand["ti_symops"]} '
                  f'chemistry_feedback_steps={cand.get("chem_feedback_steps",0)}')
        win=res['winner'];d=win['details']
        print(f'  SELECTED START={res["winner_index"]+1}, '
              f'legacy_strict_diagnostic={win["pass_chemistry"]}, '
              f'file={res["file"]}')
        print(f'  Oxygen coincidence RMS={win["rms"]:.6f} A; '
              f'distinct-Ti sharing failures={win["bad_triplets"]}; '
              f'octahedral strain={win["oct"].distortion():.6f}; '
              f'compatible-symmetry RMS max={win["symmetry_max_rms"]:.6f} A')
        print(f'  Local chemistry: bad Ti={d["badTi"]}, bad O={d["badO"]}, '
              f'primary bad Ti={d["primaryBadTi"]}, primary bad O={d["primaryBadO"]}, '
              f'O-O collision={d["collision"]:.6g} A²')
        print(f'  TiO6 first-neighbor CN: {np.unique(d["hardTi"],return_counts=True)}')
        print(f'  OTi3 first-neighbor CN: {np.unique(d["hardO"],return_counts=True)}')
        for sp in ('Ti','O'):
            arr=d['rTi'] if sp=='Ti' else d['rO']
            cn=d['hardTi'] if sp=='Ti' else d['hardO']
            ok=d['goodTi'] if sp=='Ti' else d['goodO']
            mode=d['modeTi'] if sp=='Ti' else d['modeO']
            print(f'  {sp} channels RMS [primary,outer,same-element,cross] A: '
                  f'average {np.mean(arr,axis=0).round(5).tolist()}, '
                  f'maximum {np.max(arr,axis=0).round(5).tolist()}')
            for j in range(len(arr)):
                m=library[sp][int(mode[j])]
                print(f'    {sp}{j:02d}: {"PASS" if ok[j] else "FAIL"} '
                      f'CN={int(cn[j])}/{6 if sp=="Ti" else 3} '
                      f'nearest={m["source"]}:{m["index"]} '
                      f'channel_RMS={[round(x,5) for x in arr[j]]}')
    complete=sum(r['status']=='PASS' for _,r in results)
    print('\n'+'='*92)
    print(f'OVERALL: {complete}/{len(results)} references satisfy the old combined strict diagnostic; '
          'this does not gate database SCF or output.')
    print('No intermediate CIF, trajectories, diagnostics JSON, or oxygen-target similarity files were written.')
    print('='*92,flush=True)



def smoke_test():
    cell=np.diag([7.,8.,9.]);inv=np.linalg.inv(cell)
    a=np.array([[.15,.25,.35],[6.95,7.95,8.95]])
    b=np.array([[7.15,8.25,9.35],[6.95,-.05,-.05]])
    d=minimum_image_delta(a,b,cell,inv)
    assert np.max(np.abs(d))<1e-8,(d,a,b)
    tetra=np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])
    tgt=Rotation.from_euler('xyz',[20,30,40],degrees=True).apply(tetra)
    r,res=Rotation.align_vectors(tgt,tetra)
    assert np.linalg.norm(r.apply(tetra)-tgt)<1e-9
    # Independent of pymatgen/spglib, exercise the coherent-octahedron SCF
    # state representation, contact assignment, and fixed Ti geometry.
    lib={'Ti':[{'rigid_arms':tetra}]}
    t=np.array([[.24,.25,.25],[.74,.75,.75]])
    trials=[[(0.,0,Rotation.identity())] for _ in t]
    octa=Octahedra(t,cell,lib,trials,seed=0)
    original=octa.ti.copy()
    targets=octa.positions()-t[:,None,:]@cell
    octa.fit(targets,mix=1.0)
    assert np.max(np.abs(octa.positions()-(targets+t[:,None,:]@cell)))<1e-8
    assert np.array_equal(original,octa.ti)
    assert np.max(np.abs(octa.stretch-np.eye(3)))<1e-8



REF_NAMES=('anatase','brookite','hollandite','rutile')

def ti_distances(ti,cell,k=8):
    """Ti-Ti shell distance vectors, excluding self but retaining periodic images."""
    t=torch.tensor(ti,dtype=DTYPE)
    c=torch.tensor(cell,dtype=DTYPE)
    v,d=periodic_neighbors(t,t,k,c)
    return d.detach().cpu().numpy()

def ti_reference_similarity(ti,cell,reference_ti,neighbors=8):
    """Distributional local-shell comparison, not site-aligned structural RMSD.

    Every target Ti is matched to its closest reference Ti shell; report both
    target-to-reference and reverse distances to expose missing modes.
    """
    target=ti_distances(ti,cell,neighbors)
    out={}
    for name,(rt,rc) in reference_ti.items():
        ref=ti_distances(rt,rc,neighbors)
        pair=np.sqrt(np.mean((target[:,None,:]-ref[None,:,:])**2,axis=2))
        forward=pair.min(axis=1)
        reverse=pair.min(axis=0)
        out[name]=dict(mean=float(forward.mean()),maximum=float(forward.max()),
                       symmetric=float(.5*(forward.mean()+reverse.mean())))
    return out

def db_rows(db_path,ids):
    from ase.db import connect
    with connect(str(db_path)) as db:
        for row in db.select():
            if ids is not None and int(row.id) not in ids:continue
            atoms=row.toatoms()
            if not bool(np.all(atoms.pbc)):
                yield int(row.id),None,'nonperiodic',None;continue
            symbols=atoms.get_chemical_symbols()
            nt=symbols.count('Ti');no=symbols.count('O')
            if nt==0 or set(symbols)-{'Ti','O'} or (no and no!=2*nt):
                yield int(row.id),None,f'invalid TiO2 formula Ti={nt} O={no}',None;continue
            frac=atoms.get_scaled_positions(wrap=True)
            ti=np.array([frac[i] for i,x in enumerate(symbols) if x=='Ti'])
            # Store DB oxygen ONLY in a deferred post-hoc evaluator closure.
            # The generation loop receives exclusively ti and cell.
            cell=np.asarray(atoms.cell,float)
            original_o=np.array([frac[i] for i,x in enumerate(symbols) if x=='O']).reshape(-1,3)
            yield int(row.id),(ti,cell),None,original_o

def full_structure(ti,cell,oxygen):
    # This helper is called outside main(): imports local to main are not globals.
    from pymatgen.core import Structure
    return Structure(cell,['Ti']*len(ti)+['O']*len(oxygen),
                     np.vstack((ti,oxygen)),coords_are_cartesian=False)

def compare_oxygen_to_references(ti,cell,oxygen,refs):
    """Blind, post-generation TiO2 local geometry comparison using the same
    nearest-shell chemistry scoring (training-reference source modes).
    """
    # Reference oxygen never participates in SCF. This metric compares local
    # Ti-O and O-Ti shell distances to each reference, independent of alignment.
    tt=torch.tensor(ti,dtype=DTYPE);oo=torch.tensor(oxygen,dtype=DTYPE)
    cc=torch.tensor(cell,dtype=DTYPE)
    def shells(a,b,k):
        return periodic_neighbors(a,b,k,cc)[1].detach().cpu().numpy()
    a=shells(tt,oo,6);b=shells(oo,tt,3)
    result={}
    for name,s in refs.items():
        rt=torch.tensor(sites(s,'Ti'),dtype=DTYPE)
        ro=torch.tensor(sites(s,'O'),dtype=DTYPE)
        rc=torch.tensor(s.lattice.matrix,dtype=DTYPE)
        ra=periodic_neighbors(rt,ro,6,rc)[1].detach().cpu().numpy()
        rb=periodic_neighbors(ro,rt,3,rc)[1].detach().cpu().numpy()
        da=np.sqrt(np.mean((a[:,None,:]-ra[None,:,:])**2,axis=2))
        db=np.sqrt(np.mean((b[:,None,:]-rb[None,:,:])**2,axis=2))
        result[name]=dict(TiO=float(da.min(axis=1).mean()),
                          OTi=float(db.min(axis=1).mean()))
    return result

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
                   hierarchy=getattr(args,"hierarchy","ti-first"),calibration=active_calibration)
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
        for k in range(args.starts):
            octa=Octahedra(ti,cell,lib,proposals,seed=0 if k==0 else seed+491*k)
            ans=scf(octa,peaks,chem,options,ops,steps=args.cycles,
                    symmetry_tol=args.symmetry_tol,anneal_every=args.anneal_every,
                    seed=seed+k,verbose=args.verbose_cycles)
            raw_trials.append(ans)
            before_local=time.perf_counter()
            raw_scf_seconds+=before_local-started-raw_scf_seconds-local_feedback_seconds
            ans=coherent_chemistry_feedback(ans,chem,ops,
                    steps=args.chemistry_feedback_steps,
                    rounds=args.chemistry_feedback_rounds,accept=args.accept_A)
            trials.append(ans)
            local_feedback_seconds+=time.perf_counter()-before_local
            updates+=int(ans.get('chem_feedback_steps',0))
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

# ===========================================================================
# PUBLICATION REPORT LAYER. Everything above is the unmodified v21 physics /
# SCF / global moves except for read-only phase checkpoints and move counters.
# The structure's original O coordinates never enter search_target().
# ===========================================================================

REPORT_METHODS=('anneal','hopping','replica')
REPORT_STAGES=('entrance_scf','global_scf','global_plus_local')
METRIC_CHANNEL_NAMES=('primary','outer','same_species','cross')


def publication_angle_summary(ti,oxygen,cell,cn_ti=None):
    """Geometry measured on each COMPLETE six-coordinate TiO6, not optimized.

    Six Ti-O vectors are measured by the same periodic image kernel as v21.
    Choose the matching of three disjoint opposite pairs that minimizes the
    ideal 90/180 squared angular residual. This is a DESCRIPTIVE convention,
    not a training or generation restraint. Each Ti has 3 trans and 12 cis.
    """
    tv=torch.as_tensor(ti,dtype=DTYPE)
    ov=torch.as_tensor(oxygen,dtype=DTYPE)
    cc=torch.as_tensor(cell,dtype=DTYPE)
    vectors,dist=periodic_neighbors(tv,ov,6,cc)
    vectors=vectors.detach().cpu().numpy()
    distances=dist.detach().cpu().numpy()
    if cn_ti is None:valid=np.ones(len(ti),dtype=bool)
    else:valid=np.asarray(cn_ti,dtype=int)==6
    cis=[];trans=[];oct_all=[];bond_lengths=[];local_std=[]
    for i,(vec,dd) in enumerate(zip(vectors,distances)):
        if not valid[i]:continue
        cos=np.clip((vec@vec.T)/(dd[:,None]*dd[None,:]),-1.,1.)
        angles=np.degrees(np.arccos(cos))
        best=None;best_loss=float('inf')
        for matching in OCTAHEDRAL_OPPOSITE_MATCHINGS:
            pairset={tuple(sorted(p)) for p in matching}
            t=[float(angles[a,b]) for a,b in matching]
            c=[float(angles[a,b]) for a in range(6) for b in range(a+1,6)
               if (a,b) not in pairset]
            loss=sum((x-180.)**2 for x in t)+sum((x-90.)**2 for x in c)
            if loss<best_loss:
                best_loss=loss;best=(c,t)
        cis.extend(x-90. for x in best[0]);trans.extend(x-180. for x in best[1])
        oct_all.append(np.sqrt(best_loss/15.))
        bond_lengths.extend(dd.tolist())
        local_std.append(float(np.std(dd)))
    def rms(xs):
        return float(np.sqrt(np.mean(np.square(xs)))) if len(xs) else float('nan')
    def maximum(xs):
        return float(np.max(np.abs(xs))) if len(xs) else float('nan')
    return dict(ti_octahedra_measured=int(np.sum(valid)),
                ti_octahedra_total=int(len(ti)),
                tio_mean_A=float(np.mean(bond_lengths)) if bond_lengths else float('nan'),
                tio_std_A=float(np.std(bond_lengths)) if bond_lengths else float('nan'),
                tio_intra_oct_mean_std_A=float(np.mean(local_std)) if local_std else float('nan'),
                angle_90_rms_deg=rms(cis),angle_90_max_deg=maximum(cis),
                angle_180_rms_deg=rms(trans),angle_180_max_deg=maximum(trans),
                octahedral_angle_rms_deg=rms(cis+trans),
                octahedral_angle_max_deg=maximum(cis+trans),
                bond_lengths=np.asarray(bond_lengths,float))


def _all_opposite_matchings(indices):
    if not indices:
        yield ()
    else:
        a=indices[0]
        for j in range(1,len(indices)):
            b=indices[j]
            for tail in _all_opposite_matchings(indices[1:j]+indices[j+1:]):
                yield ((a,b),)+tail


OCTAHEDRAL_OPPOSITE_MATCHINGS=tuple(_all_opposite_matchings(tuple(range(6))))
assert len(OCTAHEDRAL_OPPOSITE_MATCHINGS)==15


def reference_standard_tio(reference):
    """A report-only reference-derived benchmark (no universal Ti-O target)."""
    bonds=[];means={}
    for name,s in reference.items():
        ti=sites(s,'Ti');oxygen=sites(s,'O');cell=np.asarray(s.lattice.matrix,float)
        distances=periodic_neighbors(torch.tensor(ti,dtype=DTYPE),
                                     torch.tensor(oxygen,dtype=DTYPE),6,
                                     torch.tensor(cell,dtype=DTYPE))[1].detach().cpu().numpy()
        bonds.extend(distances.ravel().tolist())
        means[name]=float(np.mean(distances))
    return float(np.mean(bonds)),means


def _lattice_symmetry(s,symprec):
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sg=SpacegroupAnalyzer(s,symprec=symprec)
        return str(sg.get_space_group_symbol()),int(sg.get_space_group_number())
    except Exception as exc:
        return f'unavailable: {type(exc).__name__}',None


def _histogram(array):
    u,c=np.unique(np.asarray(array,dtype=int),return_counts=True)
    return ';'.join(f'{int(k)}:{int(v)}' for k,v in zip(u,c))


def original_measurements(s,reference,reference_ti,standard_tio,lib,bounds,guard,args,
                          approx_matcher=None):
    """All references and held-out original DB structures: post-hoc only."""
    ti=sites(s,'Ti');o=sites(s,'O');cell=np.asarray(s.lattice.matrix,float)
    chem=Chemistry(ti,cell,lib,bounds,guard,accept=args.accept_A,hierarchy=args.hierarchy)
    with torch.no_grad():d=chem.score(torch.tensor(o,dtype=DTYPE),details=True)
    angles=publication_angle_summary(ti,o,cell,cn_ti=d['hardTi'])
    sim=ti_reference_similarity(ti,cell,reference_ti)
    fullsim=compare_oxygen_to_references(ti,cell,o,reference)
    spg,number=_lattice_symmetry(s,args.symprec)
    raw={
        'space_group_symbol':spg,'space_group_number':number,
        'n_ti':len(ti),'n_o':len(o),
        'original_ti_cn6_good':int(np.sum(d['topologyTi'])),
        'original_o_cn3_good':int(np.sum(d['topologyO'])),
        'original_ti_cn_hist':_histogram(d['hardTi']),
        'original_o_cn_hist':_histogram(d['hardO']),
        'original_ti_geometry_good':int(np.sum(d['geometryTi'])),
        'original_o_geometry_good':int(np.sum(d['geometryO'])),
        'original_ti_learned_primary_mean_A':float(np.mean(d['rTi'][:,0])),
        'original_o_learned_primary_mean_A':float(np.mean(d['rO'][:,0])),
        'original_tio_bond_mean_A':angles['tio_mean_A'],
        'original_tio_bond_std_A':angles['tio_std_A'],
        'original_tio_intra_oct_std_A':angles['tio_intra_oct_mean_std_A'],
        'standard_reference_TiO_A':standard_tio,
        'original_tio_mean_minus_standard_A':angles['tio_mean_A']-standard_tio,
        'original_tio_rms_vs_standard_A':float(np.sqrt(np.mean((angles['bond_lengths']-standard_tio)**2)))
           if len(angles['bond_lengths']) else float('nan'),
        'original_angle_90_rms_deg':angles['angle_90_rms_deg'],
        'original_angle_90_max_deg':angles['angle_90_max_deg'],
        'original_angle_180_rms_deg':angles['angle_180_rms_deg'],
        'original_angle_180_max_deg':angles['angle_180_max_deg'],
        'original_octahedral_angle_rms_deg':angles['octahedral_angle_rms_deg'],
        'original_measured_TiO6_count':angles['ti_octahedra_measured'],
        'nearest_reference_by_ti_shell':min(sim,key=lambda n:sim[n]['mean'])
    }
    for name in REF_NAMES:
        raw[f'{name}_ti_shell_rms_mean_A']=sim[name]['mean']
        raw[f'{name}_ti_shell_rms_symmetric_A']=sim[name]['symmetric']
        raw[f'{name}_ti_shell_rms_max_A']=sim[name]['maximum']
        raw[f'{name}_tio_shell_rms_A']=fullsim[name]['TiO']
        raw[f'{name}_oti_shell_rms_A']=fullsim[name]['OTi']
        if approx_matcher is not None:
            raw[f'{name}_full_structure_equivalent']=int(approx_matcher.fit(s,reference[name]))
    for species,key in [('ti','rTi'),('o','rO')]:
        for ci,channel in enumerate(METRIC_CHANNEL_NAMES):
            raw[f'original_{species}_{channel}_mean_residual_A']=float(np.mean(d[key][:,ci]))
            raw[f'original_{species}_{channel}_max_residual_A']=float(np.max(d[key][:,ci]))
    return raw


def candidate_publication_metrics(candidate,standard_tio):
    m=candidate_metrics(candidate)
    d=candidate['details']
    angles=publication_angle_summary(candidate['oct'].ti,candidate['oxygen'],
                                     candidate['oct'].cell,cn_ti=d['hardTi'])
    info={
        'ti_cn6_good':int(np.sum(d['topologyTi'])),
        'ti_total':len(d['topologyTi']),
        'o_cn3_good':int(np.sum(d['topologyO'])),
        'o_total':len(d['topologyO']),
        'ti_cn_hist':_histogram(d['hardTi']),
        'o_cn_hist':_histogram(d['hardO']),
        'ti_geometry_good':int(np.sum(d['geometryTi'])),
        'o_geometry_good':int(np.sum(d['geometryO'])),
        'ti_geometry_bad':int(d['geometryBadTi']),
        'o_geometry_bad':int(d['geometryBadO']),
        'ti_geometry_bad_v21':int(d['geometryBadTi_v21']),
        'o_geometry_bad_v21':int(d['geometryBadO_v21']),
        'ti_geometry_good_v21':int(np.sum(d['geometryTi_v21'])),
        'o_geometry_good_v21':int(np.sum(d['geometryO_v21'])),
        'geometry_policy':d.get('geometry_policy','v21_strict'),
        'ti_learned_primary_rms_mean_A':float(np.mean(d['rTi'][:,0])),
        'o_learned_primary_rms_mean_A':float(np.mean(d['rO'][:,0])),
        'ti_learned_full_channel_rms_A':float(np.sqrt(np.mean(d['rTi']**2))),
        'o_learned_full_channel_rms_A':float(np.sqrt(np.mean(d['rO']**2))),
        'objective_loss':m['total_objective'],
        'chemistry_energy_loss':float(d['energy']),
        'learned_ti_weighted_loss':m['ti_geometry_loss'],
        'learned_o_weak_guide_loss':m['o_topology_guide_loss'],
        'sharing_rms_A':m['oxygen_sharing_rms'],
        'sharing_duplicate_triplets':m['sharing_triplets_nonunique'],
        'collision_A2':float(d['collision']),
        'ti_first_success':bool(m['ti_first_quality_ok']),
        'ti_first_success_v21':bool(d['geometryBadTi_v21']==0 and
            d['topologyBadTi']==0 and d['topologyBadO']==0 and
            candidate['bad_triplets']==0 and candidate['rms']<.12 and
            d['collision']<1e-8),
        'strict_full_geometry_success':bool(m['learned_geometry_ok'] and m['topology_ok']),
        'correct_cn':bool(m['coordination_ok']),
        'mean_octahedral_strain':m['mean_octahedron_strain'],
        'max_octahedral_strain':m['maximum_octahedron_strain'],
        'learned_arm_displacement_rms_A':m['learned_template_arm_displacement_rms'],
        'learned_arm_displacement_max_A':m['learned_template_arm_displacement_max'],
        'generated_tio_bond_mean_A':angles['tio_mean_A'],
        'generated_tio_bond_std_A':angles['tio_std_A'],
        'generated_tio_rms_vs_standard_A':float(np.sqrt(np.mean((angles['bond_lengths']-standard_tio)**2)))
          if len(angles['bond_lengths']) else float('nan'),
        'generated_angle_90_rms_deg':angles['angle_90_rms_deg'],
        'generated_angle_180_rms_deg':angles['angle_180_rms_deg'],
        'generated_octahedral_angle_rms_deg':angles['octahedral_angle_rms_deg'],
    }
    for species,key in [('ti','rTi'),('o','rO')]:
        for ci,channel in enumerate(METRIC_CHANNEL_NAMES):
            info[f'{species}_{channel}_mean_residual_A']=float(np.mean(d[key][:,ci]))
            info[f'{species}_{channel}_max_residual_A']=float(np.max(d[key][:,ci]))
    return info


def _csv_value(v):
    if v is None:return ''
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return ''
    if isinstance(v,(np.bool_,bool)):return int(v)
    if isinstance(v,(list,tuple,dict,np.ndarray)):
        return json.dumps(v,sort_keys=True,default=str)
    return v


def write_csv(path,rows,preferred=()):
    rows=list(rows)
    keys=list(dict.fromkeys(tuple(preferred)+tuple(k for r in rows for k in r)))
    with open(path,'w',newline='',encoding='utf-8') as f:
        out=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        out.writeheader()
        for r in rows:out.writerow({k:_csv_value(r.get(k)) for k in keys})


def _markdown_table(rows,columns,limit=None):
    def fmt(x):
        if x is None:return '—'
        if isinstance(x,(float,np.floating)):
            if not np.isfinite(x):return '—'
            return f'{x:.4f}'
        if isinstance(x,(np.bool_,bool)):return 'Yes' if x else 'No'
        return str(x).replace('|','\\|').replace('\n',' ')
    r=rows[:limit] if limit is not None else rows
    out=['| '+' | '.join(label for key,label in columns)+' |',
         '| '+' | '.join('---' for _ in columns)+' |']
    out.extend('| '+' | '.join(fmt(row.get(key)) for key,label in columns)+' |'
               for row in r)
    return '\n'.join(out)


def method_stage_rows(label,source_type,db_id,outcomes,standard_tio,folder,
                      save_phase_cifs=False,stage_cif_prefix=None):
    """Three actual checkpoints from each *one* global method run.

    The nine rows per structure share only THREE global-search executions:
    entrance-SCF -> global-SCF -> global+local feedback. The cumulative
    elapsed times are intentionally overlapping, not nine separate charges.
    """
    rows=[];checkpoints=[]
    for method in REPORT_METHODS:
        if method not in outcomes:continue
        winner,stats=outcomes[method]
        initial_time=stats['phase_initial'][1]
        global_time=stats['phase_global'][1]
        final_time=stats['phase_final'][1]
        for stage,phase,seconds,increment in (
            ('entrance_scf','phase_initial',initial_time,initial_time),
            ('global_scf','phase_global',global_time,max(0.,global_time-initial_time)),
            ('global_plus_local','phase_final',final_time,max(0.,final_time-global_time))):
            cand=stats[phase][0]
            row={'structure_id':label,'source_type':source_type,'database_id':db_id,
                 'method':method,'stage':stage,'uses_global_search':stage!='entrance_scf',
                 'uses_full_local_feedback':stage=='global_plus_local',
                 'elapsed_cumulative_s':seconds,
                 'shared_preparation_seconds':stats.get('shared_preparation_seconds',0.),
                 'elapsed_with_shared_setup_s':seconds+stats.get('shared_preparation_seconds',0.),
                 'elapsed_incremental_stage_s':increment,
                 'global_proposals':stats['attempts'] if stage!='entrance_scf' else 0,
                 'global_accepted':stats['accepted'] if stage!='entrance_scf' else 0,
                 'global_improvements':stats['improvements'] if stage!='entrance_scf' else 0,
                 'global_stagnation_escapes':stats['escapes'] if stage!='entrance_scf' else 0,
                 'global_stagnation_windows':stats['stalled_intervals'] if stage!='entrance_scf' else 0,
                 'global_distinct_basin_signatures':stats['distinct_basins'] if stage!='entrance_scf' else stats['entrances'],
                 'global_basin_revisits':stats['basin_revisits'] if stage!='entrance_scf' else 0,
                 'replica_swaps_attempted':stats['swaps_attempted'] if stage!='entrance_scf' else 0,
                 'replica_swaps_accepted':stats['swaps_accepted'] if stage!='entrance_scf' else 0,
                 'lcao_scf_calls':stats['fresh_lcao_calls'] if stage!='entrance_scf' else stats['entrances'],
                 'scf_cycles_total':stats['scf_cycles'] if stage!='entrance_scf' else stats['entrances']*(5 if method!='anneal' else 1),
                 'local_feedback_steps':stats['feedback_steps'] if stage=='global_plus_local' else 0,
                 'n_entrances':stats['entrances'],
                 'stop_reason':stats['stop_reason'] if stage!='entrance_scf' else 'initial_scf',
                 'initial_objective':stats['initial_objective'],
                 'move_attempts':stats.get('move_attempts',{}) if stage!='entrance_scf' else {},
                 'move_accepted':stats.get('move_accepted',{}) if stage!='entrance_scf' else {},
                 'hierarchy':cand['details'].get('hierarchy','ti-first'),
                 }
            row.update(candidate_publication_metrics(cand,standard_tio))
            rows.append(row)
            if save_phase_cifs:
                dest=folder/f'{stage_cif_prefix or label}_{method}_{stage}.cif'
                write_final_cif(cand['oct'].ti,cand['oct'].cell,cand['oxygen'],dest)
        for checkpoint in stats.get('history',[]):
            checkpoints.append(dict(structure_id=label,source_type=source_type,
                                    database_id=db_id,method=method,**checkpoint))
    if 'baseline' in outcomes:
        cand,stats=outcomes['baseline']
        for stage,phase,sec in (
            ('entrance_scf','phase_initial',stats['phase_initial'][1]),
            ('scf_plus_local','phase_final',stats['phase_final'][1])):
            cc=stats[phase][0]
            row={'structure_id':label,'source_type':source_type,'database_id':db_id,
                 'method':'baseline','stage':stage,'uses_global_search':False,
                 'uses_full_local_feedback':stage=='scf_plus_local',
                 'elapsed_cumulative_s':sec,
                 'shared_preparation_seconds':stats.get('shared_preparation_seconds',0.),
                 'elapsed_with_shared_setup_s':sec+stats.get('shared_preparation_seconds',0.),
                 'elapsed_incremental_stage_s':sec if stage=='entrance_scf' else stats.get('baseline_local_seconds',float('nan')),
                 'global_proposals':0,'global_accepted':0,'global_improvements':0,
                 'global_stagnation_escapes':0,'global_stagnation_windows':0,
                 'global_distinct_basin_signatures':stats['distinct_basins'],
                 'replica_swaps_attempted':0,'replica_swaps_accepted':0,
                 'lcao_scf_calls':stats['fresh_lcao_calls'],'scf_cycles_total':stats['scf_cycles'],
                 'local_feedback_steps':stats['feedback_steps'] if stage=='scf_plus_local' else 0,
                 'n_entrances':stats['entrances'],'stop_reason':'fixed_v22_multistart',
                 'initial_objective':stats['initial_objective'],
                 'hierarchy':cc['details'].get('hierarchy','ti-first')}
            row.update(candidate_publication_metrics(cc,standard_tio))
            rows.append(row)
    return rows,checkpoints


def nine_way_summary(detail_rows):
    rows=[]
    baseline={(r['structure_id'],r.get('seed',0)):r for r in detail_rows
              if r['method']=='baseline' and r['stage']=='scf_plus_local'}
    for method in REPORT_METHODS:
        for stage in REPORT_STAGES:
            sub=[r for r in detail_rows if r['method']==method and r['stage']==stage]
            if not sub:
                rows.append(dict(method=method,stage=stage,n=0));continue
            arr=lambda key:np.asarray([r[key] for r in sub],dtype=float)
            success=sum(bool(r['ti_first_success']) for r in sub)
            paired=[(r,baseline[(r['structure_id'],r.get('seed',0))]) for r in sub
                    if (r['structure_id'],r.get('seed',0)) in baseline]
            speedups=[b['elapsed_with_shared_setup_s']/r['elapsed_with_shared_setup_s']
                      for r,b in paired if r['elapsed_with_shared_setup_s']>1e-9]
            improvement_ti=sum(r['ti_geometry_bad']<b['ti_geometry_bad'] for r,b in paired)
            improvement_sharing=sum(r['sharing_rms_A']+1e-4<b['sharing_rms_A'] for r,b in paired)
            rescued=sum(bool(r['ti_first_success']) and not bool(b['ti_first_success'])
                       for r,b in paired)
            rows.append(dict(method=method,stage=stage,n=len(sub),
                baseline_paired=len(paired),
                median_speedup_vs_baseline=float(np.median(speedups)) if speedups else float('nan'),
                rescued_ti_first_vs_baseline=rescued,
                improved_ti_geometry_vs_baseline=improvement_ti,
                improved_sharing_vs_baseline=improvement_sharing,
                ti_first_success=success,
                ti_cn_ok=sum(r['ti_cn6_good']==r['ti_total'] for r in sub),
                o_cn_ok=sum(r['o_cn3_good']==r['o_total'] for r in sub),
                ti_geometry_ok=sum(r['ti_geometry_bad']==0 for r in sub),
                no_sharing_triplet_defects=sum(r['sharing_duplicate_triplets']==0 for r in sub),
                median_ti_geometry_bad=float(np.median(arr('ti_geometry_bad'))),
                median_o_geometry_bad=float(np.median(arr('o_geometry_bad'))),
                median_o_sharing_A=float(np.median(arr('sharing_rms_A'))),
                median_objective=float(np.median(arr('objective_loss'))),
                median_elapsed_cumulative_s=float(np.median(arr('elapsed_cumulative_s'))),
                total_elapsed_cumulative_s=float(np.sum(arr('elapsed_cumulative_s'))),
                mean_global_proposals=float(np.mean(arr('global_proposals'))),
                total_global_proposals=int(np.sum(arr('global_proposals')))))
    return rows


def _print_final_nine_way(summary,details):
    print('\n'+'='*145)
    print('PUBLICATION BENCHMARK — NINE GLOBAL METHOD × REFINEMENT STAGE COMPARISONS')
    print('Three methods × three nested checkpoints: initial SCF, post-global SCF, post-local.')
    print('Each method is executed ONCE per structure; phase times are cumulative/overlapping, NOT nine independent jobs.')
    print('Method       Stage                 N  Ti-first  Ti-geo  Ti-CN   O-CN   Med-share(A)  Med-obj   Med-time(s)  Sum-time(s)  Total-global')
    for r in summary:
        if r['n']==0:
            print(f'{r["method"]:<12} {r["stage"]:<21} 0  not run');continue
        print(f'{r["method"]:<12} {r["stage"]:<21} '
              f'{r["n"]:>2}  {r["ti_first_success"]:>3}/{r["n"]:<3}  '
              f'{r["ti_geometry_ok"]:>3}/{r["n"]:<3} '
              f'{r["ti_cn_ok"]:>3}/{r["n"]:<3} '
              f'{r["o_cn_ok"]:>3}/{r["n"]:<3} '
              f'{r["median_o_sharing_A"]:>12.5f} '
              f'{r["median_objective"]:>9.4g} '
              f'{r["median_elapsed_cumulative_s"]:>12.3f} '
              f'{r["total_elapsed_cumulative_s"]:>12.2f} '
              f'{r["total_global_proposals"]:>12d}')
    baselines=[r for r in details if r['method']=='baseline' and r['stage']=='scf_plus_local']
    print('\nVERSUS MULTISTART LCAO-SCF BASELINE (same structure, solver+shared-setup wall time, higher speedup is faster):')
    print('Method       Stage                 Baseline pairs   Median speedup   Ti-first rescues   Ti-geometry improved   Sharing improved')
    for r in summary:
        if r['n']==0:continue
        print(f'{r["method"]:<12} {r["stage"]:<21} '
              f'{r["baseline_paired"]:>5} '
              f'{r["median_speedup_vs_baseline"]:>18.3f}x '
              f'{r["rescued_ti_first_vs_baseline"]:>14} '
              f'{r["improved_ti_geometry_vs_baseline"]:>21} '
              f'{r["improved_sharing_vs_baseline"]:>18}')
    if baselines:
        print('\nV22 MULTISTART LCAO-SCF BASELINE (independent control): '
              f'{sum(r["ti_first_success"] for r in baselines)}/{len(baselines)} Ti-first successes, '
              f'total wall time {sum(r["elapsed_cumulative_s"] for r in baselines):.2f}s.')
    print('NOTE: Angle deviations are REPORT-ONLY relative to ideal 90/180 degrees.')
    print('NOTE: Ti-O standard is the POOLED FOUR-REFERENCE MEAN, not an optimizer target.')
    print('NOTE: Similarity columns are local-shell RMS, not aligned structural RMSD.')
    print('NOTE: Other original DB oxygen calibrates acceptance; target oxygen is excluded from generation.')
    print('='*145,flush=True)


def _publication_report_md(args,original_rows,generation_rows,summary_rows,
                           skipped,standard_tio,per_ref,completed_count,seconds):
    lines=[
      '# Juliette v22 — Full-dataset publication report',
      '',
      f'**Optimizer:** v22 calibrated Ti-first hierarchy (`{args.hierarchy}`; `{args.geometry_policy}`). **Elapsed:** {seconds:.1f} s.',
      f'**Cohort:** {len(original_rows)} original TiO₂ structures '
        f'({sum(r["is_reference"] for r in original_rows)} references, '
        f'{sum(not r["is_reference"] for r in original_rows)} unique DB structures). '
        f'Generation runs completed: {completed_count}.',
      '',
      '## Protocol and definitions',
      '',
      '- Reference CIFs: anatase, brookite, hollandite, rutile. Exactly these four CIFs supply learned geometry, descriptors, neighbor thresholds and TiO₆ modes.',
      '- Ti positions are fixed for generation. Other database original oxygen calibrates geometric acceptance (not reference TiO6 modes), and the current target oxygen is excluded from its own calibration fit, entrances, LCAO, SCF, global moves, ranking and local refinement.',
      '- Complete TiO₂ duplicates are screened using pymatgen StructureMatcher with '
        f'`ltol={args.duplicate_ltol}`, `stol={args.duplicate_stol}`, '
        f'`angle_tol={args.duplicate_angle_deg}°`, `scale=False`; duplicates against references are excluded.',
      '- Space groups use pymatgen SpacegroupAnalyzer at '
        f'`symprec={args.symprec} Å`; this is a measured original-structure property, not a generation constraint.',
      '- Coordination is evaluated using v21 reference-learned Ti–O / O–Ti shell cutoffs (Ti=6, O=3).',
      '- Ti-centered and O-centered deviations are the four-channel **v21 learned descriptor residuals**; the old strict O-centered selector is retained for diagnostics.',
      '- Report-only octahedral O–Ti–O angle deviations use the least-squares pairing of three trans (180°) and twelve cis (90°) angles per fully CN6 Ti. They are NOT trained constraints.',
      f'- Report-only standard Ti–O bond length: pooled four-reference Ti–O mean **{standard_tio:.5f} Å** '
        f'(per reference: {", ".join(f"{k} {v:.5f} Å" for k,v in per_ref.items())}). No fixed physical bond length is imposed.',
      '- Table 2 hopping runtime includes shared LCAO/template preparation + hopping solver + final local feedback; individual stage elapsed times in the method CSV exclude shared preparation and have a separate with-setup column.',
      '- Reference similarity uses v21 Ti–Ti shell RMS plus Ti–O/O–Ti bond-shell RMS. Full structure equivalence is separately measured; shell RMS is not an aligned atomic RMSD.',
      '- **Nine nested checkpoints:** three global controllers (anneal, hopping, replica) each reported at initial LCAO–SCF, global LCAO–SCF best basin, and after final coherent chemistry refinement. This requires three global searches, **not nine independent reruns**.',
      '- Independent control: multistart LCAO-SCF under the SAME selected v22 policy, both initial SCF and SCF plus local feedback. Historical v21 geometry flags are reported on these outputs; v21 generation has not been rerun.',
      '- The global proposals contain a fresh LCAO network update and coherent octahedron SCF at every step. Coordinates of database O atoms never contribute to their score.',
      '- Post-run `ti_first_success`: calibrated Ti geometry pass, both CN6/CN3 complete, distinct OTi₃ triplets, sharing RMS <0.12 Å and collision score <1e-8. Strict O extended modes are diagnostics only.',
      '- Global-search objective values are comparable across methods under the same selected hierarchy, but are not physical energies. A low objective does not guarantee valid TiO₂.',
      '',
      '## Table 1 — Original structures: topology, symmetry and distortions',
      '',
      _markdown_table(original_rows,[('structure_id','Structure'),('database_id','DB ID'),
        ('is_reference','Reference'),('space_group_symbol','Space group'),
        ('space_group_number','No.'),('n_ti','Ti'),('n_o','O'),
        ('original_ti_cn6_good','Ti CN6'),('original_o_cn3_good','O CN3'),
        ('original_tio_bond_mean_A','Ti–O mean Å'),
        ('original_tio_bond_std_A','Ti–O SD Å'),
        ('original_tio_rms_vs_standard_A','Ti–O RMS vs ref Å'),
        ('original_angle_90_rms_deg','90° RMS'),('original_angle_180_rms_deg','180° RMS')]),
      '',
      'Full Table 1 in `table1_original_topology.csv` includes similarity to **all four** references and all channel residuals.',
      '',
      '## Table 2 — Generated structures: basin-hopping final outcome',
      '',
      _markdown_table(generation_rows,[('structure_id','Structure'),('status','Status'),
        ('ti_cn6_good','Ti CN6 good'),('ti_total','Ti total'),
        ('o_cn3_good','O CN3 good'),('o_total','O total'),
        ('ti_geometry_bad','Ti geom bad'),('o_geometry_bad','O geom bad'),
        ('ti_learned_full_channel_rms_A','Ti dev Å'),
        ('o_learned_full_channel_rms_A','O dev Å'),
        ('objective_loss','Objective'),('sharing_rms_A','Sharing Å'),
        ('hopping_runtime_s','Hopping s'),('ti_first_success','Ti-first')]),
      '',
      'Full Table 2 in `table2_generation_quality.csv` contains all columns, including generated geometry and best method.',
      '',
      '## Global optimizer / stage comparison (9 cells)',
      '',
      _markdown_table(summary_rows,[('method','Optimizer'),('stage','Stage'),
        ('n','N'),('ti_first_success','Ti-first pass'),
        ('ti_geometry_ok','Ti geometry pass'),('ti_cn_ok','Ti CN pass'),
        ('o_cn_ok','O CN pass'),('median_o_sharing_A','Median sharing Å'),
        ('median_objective','Median objective'),
        ('median_elapsed_cumulative_s','Median elapsed s'),
        ('total_global_proposals','Total proposals'),
        ('median_speedup_vs_baseline','Speedup ×'),
        ('rescued_ti_first_vs_baseline','Ti-first rescues')]),
      '',
      'Detailed nine-stage measurements in `method_stage_comparison.csv`; proposal history in `global_progress.csv`.',
      '',
      f'## Exclusions ({len(skipped)})',
      '',
      _markdown_table(skipped,[('structure_id','Structure'),('reason','Reason'),('duplicate_of','Duplicate of')]) if skipped else 'None.',
      '',
      '## Limitations',
      '',
      'The four reference TiO₂ structures are training examples and positive controls, not held-out validation. '
        'Visual correctness, coordination, reference-comparison residuals, and objective values measure different properties. '
        'The optimizer does not guarantee exact crystallographic symmetry coupling or physical energy minimization.',
      'Poor results for incompatible input Ti frameworks do not establish impossibility of TiO₂ at those lattice positions.',
      'The original CIF exports preserve the original structures for independent verification of symmetry and bond geometry.',
      ''
    ]
    return '\n'.join(lines)

def self_test_publication():
    assert len(OCTAHEDRAL_OPPOSITE_MATCHINGS)==15
    # Deliberately rotated/translated octahedron, no ideal templates in model.
    from scipy.spatial.transform import Rotation as R
    p=np.asarray([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])
    rot=R.from_euler('xyz',[.4,.7,.2]).as_matrix()
    vec=p@rot
    # Every ideal pair forms either 90 or 180 degrees (report-only benchmark).
    cos=vec@vec.T
    off=cos[np.triu_indices(6,1)]
    assert np.count_nonzero(np.isclose(off,-1.))==3
    assert np.count_nonzero(np.isclose(off,0.))==12
    assert abs(2.*np.pi/2.-np.pi)<1e-12
    # Exercise actual periodic Ti-centered angle reporting on an ideal unit.
    ideal_cell=np.diag([10.,10.,10.])
    ideal_ti=np.array([[.5,.5,.5]])
    ideal_o=ideal_ti+np.vstack((np.eye(3),-np.eye(3)))*.1
    ideal=publication_angle_summary(ideal_ti,ideal_o,ideal_cell,cn_ti=[6])
    assert ideal['ti_octahedra_measured']==1
    assert ideal['angle_90_rms_deg']<1e-6 and ideal['angle_180_rms_deg']<1e-6
    dummy=[]
    for m in REPORT_METHODS:
        for s in REPORT_STAGES:
            dummy.append(dict(structure_id='self_test',method=m,stage=s,ti_first_success=True,ti_cn6_good=2,
                              ti_total=2,o_cn3_good=4,o_total=4,ti_geometry_bad=0,
                              o_geometry_bad=1,sharing_duplicate_triplets=0,
                              sharing_rms_A=.04,objective_loss=.4,
                              elapsed_cumulative_s=.5,global_proposals=10))
    summary=nine_way_summary(dummy)
    assert len(summary)==9 and all(x['n']==1 for x in summary)
    assert [x['method'] for x in summary]==list(np.repeat(REPORT_METHODS,3))
    import tempfile
    with tempfile.TemporaryDirectory() as dd:
        path=Path(dd)/'nine.csv'
        write_csv(path,summary)
        with open(path,newline='') as f:
            rdr=list(csv.DictReader(f))
        assert len(rdr)==9
    smoke_test()
    print('v22_report SELF-TEST PASSED: octahedral angle pairing, nine cells, CSV, v21 smoke test.')



# ===========================================================================
# v22 empirical geometry acceptance, cross-structure calibration, and checks.
# Separate calibration channel: four references still are the ONLY templates.
# Original database oxygen contributes to leave-one-target-out *acceptance*
# calibration, NEVER target-specific generation, templates, or search ranking.
# ===========================================================================

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


def original_geometry_metrics(entry,lib,bounds,guard,args,policy):
    st=entry['original']; ti=sites(st,'Ti'); o=sites(st,'O');cell=np.asarray(st.lattice.matrix,float)
    model=Chemistry(ti,cell,lib,bounds,guard,accept=args.accept_A,
                    hierarchy=args.hierarchy,calibration=policy)
    with torch.no_grad():d=model.score(torch.tensor(o,dtype=DTYPE),details=True)
    ret={}
    for sp,label in [('Ti','ti'),('O','o')]:
        ret[f'original_{label}_geometry_good_v22']=int(np.sum(d[f'geometry{sp}']))
        ret[f'original_{label}_geometry_good_v21']=int(np.sum(d[f'geometry{sp}_v21']))
        ret[f'original_{label}_geometry_bad_v22']=int(d[f'geometryBad{sp}'])
        ret[f'original_{label}_geometry_bad_v21']=int(d[f'geometryBad{sp}_v21'])
        ret[f'original_{label}_cn_good']=int(np.sum(d[f'topology{sp}']))
        ret[f'original_{label}_cn_and_geometry_good_v22']=int(np.sum(d[f'geometry{sp}'] & d[f'topology{sp}']))
        ret[f'original_{label}_cn_and_geometry_good_v21']=int(np.sum(d[f'geometry{sp}_v21'] & d[f'topology{sp}']))
    return ret,d


def periodic_oxygen_correspondence(original,generated,cell):
    """Independent POST-HOC optimal O assignment; no solver access."""
    a=np.asarray(original,float);b=np.asarray(generated,float)
    if len(a)!=len(b) or not len(a):return None
    delta=b[:,None,:]-a[None,:,:]
    delta-=np.rint(delta)
    image=(delta[:,:,None,:]+LOCAL_SHIFTS[None,None,:,:])@cell
    dist=np.linalg.norm(image,axis=-1).min(axis=-1)
    rows,cols=linear_sum_assignment(dist)
    return cols,dict(oxygen_match_rms_A=float(np.sqrt(np.mean(dist[rows,cols]**2))),
         oxygen_match_mean_A=float(np.mean(dist[rows,cols])),
         oxygen_match_max_A=float(np.max(dist[rows,cols])),
         oxygen_match_within_0p20=int(np.sum(dist[rows,cols]<=.20)),
         oxygen_match_within_0p45=int(np.sum(dist[rows,cols]<=.45)))


def _site_diagnostics(label,seed,source,method,stage,details,lib,original_details=None,match=None):
    lines=[]
    for sp,expected,rkey,mkey,lkey in [('Ti',6,'rTi','modeTi','geometry_limitsTi'),
                                      ('O',3,'rO','modeO','geometry_limitsO')]:
        rr=np.asarray(details[rkey]);lim=np.asarray(details[lkey]);ids=np.asarray(details[mkey])
        cn=np.asarray(details['hard'+sp]);geo=np.asarray(details['geometry'+sp]);oldgeo=np.asarray(details['geometry'+sp+'_v21'])
        for idx in range(len(cn)):
            reference_index=idx if sp=='Ti' else (int(match[idx]) if match is not None else None)
            oldres=np.asarray(original_details[rkey][reference_index]) if original_details is not None and reference_index is not None else None
            record=dict(structure_id=label,seed=seed,source=source,method=method,stage=stage,
                 species=sp,site_index=idx,original_corresponding_site=reference_index,
                 cn=int(cn[idx]),target_cn=expected,cn_ok=bool(cn[idx]==expected),
                 learned_mode=int(ids[idx]),mode_group=str(tuple(lib[sp][int(ids[idx])]['counts'])),
                 v21_geometry_ok=bool(oldgeo[idx]),v22_geometry_ok=bool(geo[idx]),
                 v21_v22_reclassification=('RESCUED' if geo[idx] and not oldgeo[idx] else
                                            'NEW_REJECTION' if oldgeo[idx] and not geo[idx] else 'SAME'))
            for k,name in enumerate(('primary','outer','same_species','cross')):
                record[f'{name}_rms_A']=float(rr[idx,k])
                record[f'{name}_limit_v22_A']=float(lim[idx,k])
                record[f'{name}_excess_v22_A']=float(max(0,rr[idx,k]-lim[idx,k]))
                if oldres is not None:
                    record[f'{name}_original_rms_A']=float(oldres[k])
                    record[f'{name}_delta_vs_original_A']=float(rr[idx,k]-oldres[k])
            lines.append(record)
    return lines


def _stability_reports(method_rows):
    groups=defaultdict(list)
    for row in method_rows:
        groups[(row['structure_id'],row['method'],row['stage'])].append(row)
    by=[]
    for (sid,method,stage),items in sorted(groups.items()):
        n=len(items)
        success=np.asarray([float(x['ti_first_success']) for x in items])
        share=np.asarray([float(x['sharing_rms_A']) for x in items])
        rms=np.asarray([float(x.get('oxygen_match_rms_A',float('nan'))) for x in items])
        by.append(dict(structure_id=sid,method=method,stage=stage,n_seeds=n,
            stability_assessed=bool(n>=3),
            successes=int(success.sum()),success_rate=float(success.mean()),
            success_unanimous=bool(success.min()==success.max()),
            median_sharing_rms_A=float(np.median(share)),
            sharing_std_A=float(np.std(share)),
            median_oxygen_original_rms_A=float(np.nanmedian(rms)) if np.isfinite(rms).any() else float('nan'),
            oxygen_original_std_A=float(np.nanstd(rms)) if np.isfinite(rms).any() else float('nan')))
    overall=[]
    bymethod=defaultdict(list)
    for row in by:bymethod[(row['method'],row['stage'])].append(row)
    for (m,stage),mixed in sorted(bymethod.items()):
        for cohort in ('database','reference','all'):
            sub=([x for x in mixed if
                 (x['structure_id'].startswith('ref_') if cohort=='reference' else
                  not x['structure_id'].startswith('ref_'))]
                 if cohort!='all' else mixed)
            if not sub:continue
            overall.append(dict(cohort=cohort,method=m,stage=stage,n_structures=len(sub),
              n_seeds=min(x['n_seeds'] for x in sub),
              stability_assessed=bool(min(x['n_seeds'] for x in sub)>=3),
              fully_successful_structures=sum(x['success_rate']==1. for x in sub),
              partial_success_structures=sum(0.<x['success_rate']<1. for x in sub),
              never_successful_structures=sum(x['success_rate']==0. for x in sub),
              success_rate_over_runs=float(np.mean([x['success_rate'] for x in sub])),
              median_seed_sharing_std_A=float(np.median([x['sharing_std_A'] for x in sub]))))
    return by,overall


def _recovery_report(method_rows):
    keys=defaultdict(dict)
    for row in method_rows:
        keys[(row['structure_id'],row['seed'])][(row['method'],row['stage'])]=row
    records=[]
    for (sid,seed),stages in sorted(keys.items()):
        base=stages.get(('baseline','entrance_scf'))
        for method in REPORT_METHODS:
            ini=stages.get((method,'entrance_scf'))
            g=stages.get((method,'global_scf'))
            loc=stages.get((method,'global_plus_local'))
            if ini is None or g is None or loc is None:continue
            original_good=bool(ini['ti_first_success'])
            global_good=bool(g['ti_first_success'])
            local_good=bool(loc['ti_first_success'])
            if original_good:
                status=('DIRECT_SUCCESS_PRESERVED' if local_good else
                        'DIRECT_SUCCESS_LOST_BY_OPTIMIZATION')
            elif global_good:
                status=('RECOVERED_BY_GLOBAL' if local_good else
                        'GLOBAL_SUCCESS_LOST_BY_LOCAL')
            elif local_good:status='RECOVERED_BY_LOCAL'
            else:status='UNRESOLVED'
            records.append(dict(structure_id=sid,seed=seed,method=method,status=status,
                 baseline_direct_success=bool(base['ti_first_success']) if base else '',
                 initial_success=original_good,after_global_success=global_good,
                 after_local_success=local_good,
                 initial_v21_success=bool(ini['ti_first_success_v21']),
                 final_v21_success=bool(loc['ti_first_success_v21']),
                 initial_sharing_rms_A=ini['sharing_rms_A'],
                 final_sharing_rms_A=loc['sharing_rms_A'],
                 additional_global_seconds=max(0.,g['elapsed_cumulative_s']-ini['elapsed_cumulative_s']),
                 additional_local_seconds=max(0.,loc['elapsed_cumulative_s']-g['elapsed_cumulative_s'])))
    return records


def _v22_summary_append(report,args,calibration_env,calibration_rows,original_rows,
                        method_rows,stability_summary,recovery_rows,failures):
    text=['', '## v22 calibrated geometry judgments', '',
      f'- Acceptance: `{args.geometry_policy}`; percentile={args.calibration_quantile:.3f}; '
      f'minimum other structures={args.calibration_min_structures}; '
      f'reference-pull coefficient={args.geometry_reference_pull:.4f}.',
      '- Four reference CIFs define chemical modes and neighbor cutoffs. Distinct original database CIFs '
      'calibrate acceptance only, with the target original excluded from its own fit.',
      '- Calibration is extra labeled supervision and the database cohort is NOT an untouched holdout. '
      'Leave-one-structure-out removes direct target threshold leakage but does not replace external validation.',
      '- v22 uses calibrated channel-excess penalties in the continuous chemistry objective and coherent local feedback. '
      'A small residual attraction to the reference modes remains to regularize the optimizer.',
      '- The original v21 geometry flags are recorded on the SAME v22-generated candidates (this is NOT a v21 rerun); v21 and v22 '
      'objectives are different and their scalar values must NOT be compared across versions.',
      '- Early stopping requires calibrated Ti geometry, all Ti6/O3 coordination, unique sharing triplets, '
      'no collisions and low sharing RMS; weak O extended descriptors do not gate Ti-first success.',
      '- Original-vs-generated oxygen RMS is POST-HOC Hungarian periodic matching; never used to select or optimize oxygen.',
      '- A near-original geometry is not a proof of physical validity. Conversely a different network need not be invalid.',
      '', '### Original TiO6 and OTi3 descriptor acceptance (same original CIFs)', '',
      _markdown_table(original_rows,[('structure_id','Structure'),('original_ti_geometry_good_v21','Ti v21'),
        ('original_ti_geometry_good_v22','Ti v22'),('original_o_geometry_good_v21','O v21'),
        ('original_o_geometry_good_v22','O v22'),('original_ti_cn_good','Ti CN'),
        ('original_o_cn_good','O CN')]),
      '', '### v22 calibration thresholds per target (leave-one-target-out)', '',
      _markdown_table(calibration_rows,[('structure_id','Structure'),('calibration_structure_count','Calib N'),
        ('Ti_primary_A','Ti primary'),('Ti_outer_A','Ti outer'),
        ('Ti_same_species_A','Ti same'),('Ti_cross_A','Ti cross'),
        ('O_primary_A','O primary'),('O_outer_A','O outer')]),
      '', '### Stability across repeated seeds', '',
      _markdown_table(stability_summary,[('cohort','Cohort'),('method','Method'),('stage','Stage'),
        ('n_structures','Structures'),('n_seeds','Seeds'),
        ('fully_successful_structures','All-pass'),('partial_success_structures','Mixed'),
        ('never_successful_structures','None-pass'),('success_rate_over_runs','Pass fraction'),
        ('median_seed_sharing_std_A','Median sharing SD Å')]),
      '', '### Initial LCAO-SCF and global/local recovery counts', '',
    ]
    counts=defaultdict(lambda:defaultdict(int))
    for r in recovery_rows:counts[r['method']][r['status']]+=1
    for method in REPORT_METHODS:
        text.append(f'- {method}: '+', '.join(f'{k}={v}' for k,v in sorted(counts[method].items())))
    text.extend(['','### Limitations and interpretation','',
      '- A strict OTi3 topology may distort a physically reasonable TiO6 network containing OTi2 environments. '
      'We record such cases as failure modes; this version does NOT loosen CN6/CN3 targets or introduce OTi2 as a permitted output.',
      '- The calibrated acceptance region can contain chemically implausible structures. Coordination, sharing, '
      'collision, geometry and post-hoc fidelity must all be inspected.',
      '- The geometry calibration population contains only the original unique database structures (excluding the current target) '
      'with correctly coordinated atoms for the relevant species. Reference-only fallback is strict when too few examples exist.',
      '- Exact symmetry coupling during all SCF/local updates remains unproven.',
      '- Multiple seeds estimate algorithm variability, not confidence intervals for chemistry.',
      f'- Generation errors: {len(failures)} runs.', ''])
    return report+'\n'+'\n'.join(text)



# ==========================================================================
# Juliette v22r1 central organizer: ONE calibration, ONE seed, shared work
# queue, centrally archived structure results; no four worker report branches.
# Entire physics/search/report layer above is retained from standalone v22.
# ==========================================================================

def _atomic_pickle(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with open(tmp,'wb') as f:
            pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def _atomic_text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        tmp.write_text(value,encoding='utf-8')
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def _atomic_csv(path,rows,preferred=()):
    path=Path(path)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        write_csv(tmp,rows,preferred)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def _log(*parts):
    print(*parts,flush=True)


def _parse_v22r1(argv=None):
    ap=argparse.ArgumentParser(description='Juliette v22r1: one-seed, centrally coordinated, parallel publication report')
    ap.add_argument('--ref',default='ref')
    ap.add_argument('--db',default='tio2.db')
    ap.add_argument('--out',default='juliette_v22r1_publication')
    ap.add_argument('--db-ids',type=int,nargs='+',help='Optional testing subset (default: entire ASE database)')
    ap.add_argument('--max-targets',type=int,default=0,help='Testing-only cap; 0 means all unique database structures')
    ap.add_argument('--workers',type=int,default=4,help='Number of workers (default 4)')
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    ap.add_argument('--seed',type=int,default=42,help='One and only one generation RNG seed (default 42)')
    ap.add_argument('--resume',action='store_true',help='Continue incomplete run; skip verified finished structures')
    ap.add_argument('--aggregate-only',action='store_true',help='Rebuild summary tables from per-structure result bundles')
    ap.add_argument('--report-every',type=int,default=4,
                    help='Rebuild consolidated tables after every N completed structures (default 4)')
    ap.add_argument('--starts',type=int,default=4)
    ap.add_argument('--cycles',type=int,default=40)
    ap.add_argument('--candidate-cap',type=int,default=1500)
    ap.add_argument('--modes-per-ti',type=int,default=5)
    ap.add_argument('--octahedral-proposals',type=int,default=8)
    ap.add_argument('--symprec',type=float,default=.055)
    ap.add_argument('--symmetry-tol',type=float,default=.17)
    ap.add_argument('--accept-A',dest='accept_A',type=float,default=.10)
    ap.add_argument('--shell-max-k',type=int,default=30)
    ap.add_argument('--anneal-every',type=int,default=0)
    ap.add_argument('--chemistry-feedback-steps',type=int,default=64)
    ap.add_argument('--chemistry-feedback-rounds',type=int,default=2)
    ap.add_argument('--hierarchy',choices=('ti-first','v20'),default='ti-first')
    ap.add_argument('--methods',nargs='+',choices=('baseline','anneal','hopping','replica'),
                    default=['baseline','anneal','hopping','replica'])
    ap.add_argument('--global-steps',type=int,default=300)
    ap.add_argument('--global-scf-cycles',type=int,default=5)
    ap.add_argument('--anneal-scf-cycles',type=int,default=1)
    ap.add_argument('--global-max-seconds',type=float,default=45.)
    ap.add_argument('--match-baseline-time',action='store_true')
    ap.add_argument('--replicas',type=int,default=4)
    ap.add_argument('--exchange-every',type=int,default=12)
    ap.add_argument('--global-rotation',type=float,default=.35)
    ap.add_argument('--stagnation-window',type=int,default=24)
    ap.add_argument('--exhaustion-window',type=int,default=96)
    ap.add_argument('--min-global-steps',type=int,default=48)
    ap.add_argument('--checkpoint-every',type=int,default=25)
    ap.add_argument('--early-sharing',type=float,default=.12)
    ap.add_argument('--early-primary-A',type=float,default=.12)
    ap.add_argument('--early-confirm',type=int,default=4)
    ap.add_argument('--improvement-eps',type=float,default=1e-5)
    ap.add_argument('--geometry-policy',choices=('v21_strict','channel_q95','group_channel_q95'),default='channel_q95')
    ap.add_argument('--calibration-quantile',type=float,default=.95)
    ap.add_argument('--calibration-min-structures',type=int,default=4)
    ap.add_argument('--calibration-group-min-atoms',type=int,default=12)
    ap.add_argument('--calibration-group-min-structures',type=int,default=3)
    ap.add_argument('--geometry-reference-pull',type=float,default=.06)
    ap.add_argument('--verbose-cycles',action='store_true')
    ap.add_argument('--duplicate-ltol',type=float,default=.01)
    ap.add_argument('--duplicate-stol',type=float,default=.03)
    ap.add_argument('--duplicate-angle-deg',type=float,default=1.)
    ap.add_argument('--save-stage-cifs',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args(argv)
    if args.workers<1:ap.error('--workers must be >= 1')
    if args.report_every<1:ap.error('--report-every must be >= 1')
    if args.seed<0:ap.error('--seed must be nonnegative')
    if args.global_steps<1 or args.cycles<1 or args.starts<1:ap.error('global-steps, cycles, starts must be >=1')
    if args.global_scf_cycles<2 or args.anneal_scf_cycles<1:ap.error('global SCF cycles must be >=2; anneal SCF >=1')
    if args.replicas<2 or args.exchange_every<1:ap.error('replicas must be >=2; exchange-every >=1')
    if args.stagnation_window<2 or args.exhaustion_window<2 or args.checkpoint_every<1:
        ap.error('global windows / checkpoints must be valid')
    if len(set(args.methods))!=len(args.methods):ap.error('duplicate method selected')
    if 'hopping' not in args.methods:ap.error('hopping required for publication Table 2')
    if args.geometry_policy!='v21_strict' and args.hierarchy!='ti-first':
        ap.error('calibrated acceptance defined only for ti-first hierarchy')
    if not .50<args.calibration_quantile<1:ap.error('quantile must be in (.50,1)')
    if args.calibration_min_structures<2:ap.error('calibration-min-structures >=2')
    if args.calibration_group_min_atoms<2 or args.calibration_group_min_structures<2:
        ap.error('group calibration minimums >=2')
    if not 0<=args.geometry_reference_pull<=1:ap.error('geometry-reference-pull must be in [0,1]')
    if any(getattr(args,k)<=0 for k in ('duplicate_ltol','duplicate_stol','duplicate_angle_deg')):
        ap.error('duplicate tolerances must be positive')
    return args


def _source_identity(args):
    paths=[Path(args.ref)/f'{name}.cif' for name in REF_NAMES]+[Path(args.db)]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError('Input files missing: '+', '.join(missing))
    return {str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}


def _science_config(args):
    # Execution-resource and report frequency knobs are deliberately omitted;
    # resumes can change worker count, GPU/CPU mode and reporting cadence.
    ignore={'workers','device','resume','aggregate_only','report_every','out','self_test'}
    return {k:v for k,v in vars(args).items() if k not in ignore}


def _config_guard(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    p=out/'run_configuration.json'
    desired={'scientific_options':_science_config(args),'source_fingerprints':_source_identity(args),
             'algorithm':'v22r1-central-v1'}
    if p.exists():
        current=json.loads(p.read_text(encoding='utf-8'))
        if current!=desired:
            diff=[k for k in sorted(set(current.get('scientific_options',{}))|set(desired['scientific_options']))
                  if current.get('scientific_options',{}).get(k)!=desired['scientific_options'].get(k)]
            raise RuntimeError('Existing output belongs to DIFFERENT model/inputs; use new --out. '
                               f'Different settings={diff or "source files / algorithm"}')
        if not (args.resume or args.aggregate_only):
            raise RuntimeError('Output already initialized. Supply --resume or select a new --out.')
    else:
        if args.resume or args.aggregate_only:
            raise RuntimeError('Cannot resume: run_configuration.json absent.')
        _atomic_text(p,json.dumps(desired,indent=2,sort_keys=True)+'\n')
    return desired


def _device_init(device,index):
    torch.set_num_threads(1)
    if device=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in worker; load CUDA-enabled torch and request GPUs.')
        torch.cuda.set_device(index)
        torch.set_default_device(f'cuda:{index}')
        probe=torch.ones(1,dtype=DTYPE)
        if not probe.is_cuda:raise RuntimeError('CUDA default tensor device check failed')
        return f'CUDA:{index} {torch.cuda.get_device_name(index)}'
    torch.set_default_device('cpu')
    return 'CPU'


def _calibration_record(label,is_reference,policy,lib,args):
    rec=dict(structure_id=label,is_reference=is_reference,geometry_policy=args.geometry_policy,
             quantile=args.calibration_quantile,calibration_structure_count=policy.eligible_structures,
             Ti_calibration_sites=policy.eligible_atoms.get('Ti',0),O_calibration_sites=policy.eligible_atoms.get('O',0))
    for sp in ('Ti','O'):
        for k,name in enumerate(METRIC_CHANNEL_NAMES):
            rec[f'{sp}_{name}_A']=float(policy.global_limits[sp][k])
        for grp in {tuple(x['counts']) for x in lib[sp]}:
            support=policy.group_support[(sp,grp)]
            for k,name in enumerate(METRIC_CHANNEL_NAMES):
                if args.geometry_policy=='group_channel_q95':
                    rec[f'{sp}_group_{grp}_{name}_A']=float(policy.group_limits[(sp,grp)][k])
            rec[f'{sp}_group_{grp}_support']=f'{support[0]} atoms/{support[1]} structures/{support[2]}'
    return rec


def _prepare_context(args):
    """ONLY the central organizer does references, DB dedupe and calibration.

    Cache this prepared context atomically. No worker ever rereads the DB,
    relearns modes, reruns original geometry, or fits calibration thresholds.
    """
    out=Path(args.out);cache=out/'_central_preparation.pkl'
    if args.resume and cache.is_file():
        _log('CENTRAL CACHE HIT:',cache,'(reference learning, DB screening, calibration, original metrics reused)')
        with open(cache,'rb') as f:ctx=pickle.load(f)
        if ctx.get('version')!='v22r1-central-v1':raise RuntimeError('Incompatible cached preparation')
        return ctx
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from ase.io import write as ase_write
    from ase.db import connect
    started=time.perf_counter()
    paths=[Path(args.ref)/f'{name}.cif' for name in REF_NAMES]
    _log('CENTRAL PREPARATION: learning reference geometry once')
    lib,bounds,guard,refs=learn(paths,shell_max_k=args.shell_max_k)
    reference=dict(zip(REF_NAMES,refs))
    for name,s in reference.items():
        chem=Chemistry(sites(s,'Ti'),np.asarray(s.lattice.matrix,float),lib,bounds,guard,
                       accept=args.accept_A,hierarchy=args.hierarchy)
        with torch.no_grad():d=chem.score(torch.tensor(sites(s,'O'),dtype=DTYPE),details=True)
        if d['badTi'] or d['badO'] or d['collision']>1e-8:
            raise RuntimeError(f'Reference mode calibration FAILED: {name}')
    for mode in lib['Ti']:
        ref=reference[mode['source']]
        mode['rigid_arms']=reference_neighbors(ref,mode['index'],'O',6)[0]
        mode['framework_arms']=reference_neighbors(ref,mode['index'],'Ti',8)[0]
    reference_ti={name:(sites(s,'Ti'),np.asarray(s.lattice.matrix,float))
                  for name,s in reference.items()}
    standard_tio,per_ref=reference_standard_tio(reference)
    matcher=StructureMatcher(ltol=args.duplicate_ltol,stol=args.duplicate_stol,
                 angle_tol=args.duplicate_angle_deg,primitive_cell=True,
                 scale=False,attempt_supercell=True)
    similar_matcher=StructureMatcher(ltol=.10,stol=.20,angle_tol=5.,
                 primitive_cell=True,scale=False,attempt_supercell=True)
    original_dir=out/'original_cifs';original_dir.mkdir(parents=True,exist_ok=True)
    (out/'generated_cifs').mkdir(parents=True,exist_ok=True)
    (out/'structures').mkdir(parents=True,exist_ok=True)
    dataset=[];skipped=[];seen=[];screening=[]
    for name,s in reference.items():
        shutil.copy2(paths[list(REF_NAMES).index(name)],original_dir/f'ref_{name}_original.cif')
        dataset.append(dict(structure_id=f'ref_{name}',source_type='reference',
             database_id='',is_reference=True,original=s,
             original_cif=f'original_cifs/ref_{name}_original.cif',ref_name=name))
        seen.append((f'ref_{name}',s))
        screening.append(dict(structure_id=f'ref_{name}',status='REFERENCE_INCLUDED',
                              reason='',duplicate_of='',reference_overlap=name))
    n_db=0;n_unique=0
    _log('CENTRAL PREPARATION: screening full DB once')
    with connect(str(args.db)) as conn:
        for row in conn.select():
            rid=int(row.id)
            if args.db_ids is not None and rid not in set(args.db_ids):continue
            n_db+=1;label=f'db_{rid:06d}'
            try:
                atoms=row.toatoms();symbols=atoms.get_chemical_symbols()
                nt=symbols.count('Ti');no=symbols.count('O')
                if not bool(np.all(atoms.pbc)):reason='nonperiodic'
                elif nt==0 or no!=2*nt or set(symbols)!={'Ti','O'}:
                    reason=f'not complete periodic TiO2 (Ti={nt}, O={no})'
                else:reason=None
                if reason is not None:
                    skipped.append(dict(structure_id=label,reason=reason,duplicate_of=''))
                    screening.append(dict(structure_id=label,status='EXCLUDED_INVALID',reason=reason))
                    continue
                coords=np.asarray(atoms.get_scaled_positions(wrap=True))
                s=full_structure(coords[np.asarray(symbols)=='Ti'],np.asarray(atoms.cell,float),
                                 coords[np.asarray(symbols)=='O'])
                dupe=next((prevname for prevname,prev in seen if matcher.fit(s,prev)),None)
                overlaps=[name for name,rs in reference.items() if similar_matcher.fit(s,rs)]
                if dupe is not None:
                    reason='complete TiO2 near-exact duplicate'
                    skipped.append(dict(structure_id=label,reason=reason,duplicate_of=dupe))
                    screening.append(dict(structure_id=label,status='EXCLUDED_DUPLICATE',
                                 reason=reason,duplicate_of=dupe,reference_overlap=';'.join(overlaps)))
                    continue
                seen.append((label,s))
                if args.max_targets and n_unique>=args.max_targets:
                    reason='testing max-targets cap'
                    skipped.append(dict(structure_id=label,reason=reason,duplicate_of=''))
                    screening.append(dict(structure_id=label,status='EXCLUDED_TESTING_CAP',reason=reason))
                    continue
                dest=original_dir/f'{label}_original.cif'
                ase_write(str(dest),atoms,format='cif')
                dataset.append(dict(structure_id=label,source_type='database',database_id=rid,
                      is_reference=False,original=s,original_cif=f'original_cifs/{dest.name}',ref_name=''))
                n_unique+=1
                screening.append(dict(structure_id=label,status='INCLUDED',reason='',
                        duplicate_of='',reference_overlap=';'.join(overlaps)))
            except Exception as exc:
                reason=f'screening error: {type(exc).__name__}: {exc}'
                skipped.append(dict(structure_id=label,reason=reason,duplicate_of=''))
                screening.append(dict(structure_id=label,status='ERROR',reason=reason))
                _log('SCREENING ERROR',label,reason)
    _log(f'CENTRAL COHORT: {len(dataset)} unique structures (4 references, {n_unique} database), '
         f'{n_db} database records screened, {len(skipped)} excluded')
    _log('CENTRAL PREPARATION: building ONE calibration population and frozen leave-target-out policies')
    calibration_env,calibration_counts=collect_calibration_data(dataset,lib,bounds,guard,args)
    calibration_rows=[];original_rows=[];original_site_rows=[];prepared=[]
    for index,entry in enumerate(dataset):
        label=entry['structure_id'];s=entry['original']
        ti=sites(s,'Ti');cell=np.asarray(s.lattice.matrix,float);orig_o=sites(s,'O')
        policy=make_geometry_policy(calibration_env,lib,label,args)
        calibration_rows.append(_calibration_record(label,entry['is_reference'],policy,lib,args))
        original_base=dict(structure_id=label,source_type=entry['source_type'],
                           database_id=entry['database_id'],is_reference=entry['is_reference'],
                           original_cif=entry['original_cif'],hierarchy=args.hierarchy)
        details=None
        try:
            original_base.update(original_measurements(s,reference,reference_ti,
                     standard_tio,lib,bounds,guard,args,similar_matcher))
            geom,details=original_geometry_metrics(entry,lib,bounds,guard,args,policy)
            original_base.update(geom);original_base['geometry_policy']=args.geometry_policy
            original_site_rows.extend(_site_diagnostics(label,'','original','original','original',details,lib))
        except Exception as exc:
            original_base['original_measurement_error']=f'{type(exc).__name__}: {exc}'
            _log('ORIGINAL METRICS ERROR',label,original_base['original_measurement_error'])
            traceback.print_exc()
        original_rows.append(original_base)
        prepared.append(dict(structure_id=label,source_type=entry['source_type'],
                 database_id=entry['database_id'],is_reference=entry['is_reference'],
                 ref_name=entry['ref_name'],ti=ti,cell=cell,original_o=orig_o,
                 original_details=details,policy=policy,index=index))
        if (index+1)%10==0 or index+1==len(dataset):
            _log(f'  central original/calibration {index+1}/{len(dataset)} complete')
    ctx=dict(version='v22r1-central-v1',lib=lib,bounds=bounds,guard=guard,
        shift_vectors=np.asarray(SHIFTS,float).copy(),
        prepared=prepared,original_rows=original_rows,original_site_rows=original_site_rows,
        calibration_rows=calibration_rows,screening=screening,skipped=skipped,
        standard_tio=standard_tio,per_ref=per_ref,calibration_observations=len(calibration_env),
        elapsed_preparation_s=time.perf_counter()-started)
    _atomic_pickle(cache,ctx)
    _log(f'CENTRAL PREPARATION CACHED in {time.perf_counter()-started:.1f}s: {cache}')
    return ctx


def _results_file(out,label):
    return Path(out)/'structures'/label/'result.pkl'


def _valid_result(out,entry,args):
    label=entry['structure_id'];path=_results_file(out,label)
    if not path.is_file():return False
    try:
        with open(path,'rb') as f:r=pickle.load(f)
        if r.get('status')!='COMPLETED' or r.get('structure_id')!=label or r.get('seed')!=args.seed:
            return False
        dest=Path(out)/'generated_cifs'
        required=[dest/f'{label}_{method}_generated.cif' for method in args.methods]
        required.append(dest/f'{label}_best_generated.cif')
        if not all(p.is_file() and p.stat().st_size>0 for p in required):return False
        if args.save_stage_cifs:
            required=[dest/f'{label}_{method}_{stage}.cif'
                 for method in REPORT_METHODS if method in args.methods for stage in REPORT_STAGES]
            if not all(p.is_file() and p.stat().st_size>0 for p in required):return False
        return True
    except Exception:return False


def _structure_markdown(entry,result,args):
    label=entry['structure_id'];original=result['original'];rows=result['method_rows']
    lines=[f'# {label} — Juliette v22r1 single-seed report','',
         f'- Source: {entry["source_type"]}; seed: {args.seed}; Ti: {len(entry["ti"])}; O: {len(entry["original_o"])}.',
         f'- Execution status: {result["status"]}.',
         '- Learned modes: four reference CIFs; geometry calibration: other DB structures only.',
         '- The original oxygen positions were used only for post-hoc evaluation, not search.',
         '', '## Original geometry', '',
         _markdown_table([original], [('structure_id','Structure'),('original_ti_cn_good','Ti CN6'),
            ('original_o_cn_good','O CN3'),('original_ti_geometry_good_v21','Ti v21'),
            ('original_ti_geometry_good_v22','Ti v22'),('original_o_geometry_good_v21','O v21'),
            ('original_o_geometry_good_v22','O v22')]),
         '', '## Global methods and checkpoints', '',
         _markdown_table(rows,[('method','Method'),('stage','Stage'),('ti_first_success','Ti-first pass'),
             ('ti_cn6_good','Ti CN6'),('o_cn3_good','O CN3'),('ti_geometry_bad','Ti geometry bad'),
             ('sharing_rms_A','Sharing Å'),('oxygen_match_rms_A','Original O RMS Å'),
             ('objective_loss','Objective'),('elapsed_cumulative_s','Runtime s')]),
         '', '## Final hopping calculation', '',
         _markdown_table([result['generated']],[('status','Status'),('ti_cn6_good','Ti CN6'),
             ('o_cn3_good','O CN3'),('ti_geometry_bad_v21','Ti bad v21'),
             ('ti_geometry_bad','Ti bad v22'),('o_geometry_bad_v21','O bad v21'),
             ('o_geometry_bad','O bad v22'),('sharing_rms_A','Sharing Å'),
             ('oxygen_match_rms_A','Original O RMS Å'),('hopping_runtime_s','Runtime s')]),
         '', '## Run provenance', '',
         f'- Log: `structures/{label}/run.log`.',
         f'- Per-site diagnostics: central `site_geometry_diagnostics.csv`.',
         f'- CIFs: central `generated_cifs/{label}_*_generated.cif`.',
         f'- Result cache: `structures/{label}/result.pkl` (resume checkpoint).','']
    return '\n'.join(lines)


def _run_one_structure(entry,ctx,args,worker_id):
    label=entry['structure_id'];seed=int(args.seed)
    out=Path(args.out);target_dir=out/'structures'/label;target_dir.mkdir(parents=True,exist_ok=True)
    generated_dir=out/'generated_cifs';generated_dir.mkdir(parents=True,exist_ok=True)
    starting=time.perf_counter()
    _atomic_text(target_dir/'status.json',json.dumps(dict(structure_id=label,
          seed=seed,status='RUNNING',worker=worker_id,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    base=dict(structure_id=label,seed=seed,source_type=entry['source_type'],
              database_id=entry['database_id'],n_ti=len(entry['ti']),n_o=len(entry['original_o']),
              is_reference=entry['is_reference'],hierarchy=args.hierarchy,
              geometry_policy=args.geometry_policy,
              calibration_structures=entry['policy'].eligible_structures)
    output=dict(structure_id=label,seed=seed,status='ERROR',generated=base,method_rows=[],
                site_rows=[],checkpoints=[],elapsed_s=0.,original=ctx['original_lookup'][label],
                worker_id=worker_id)
    with open(target_dir/'run.log','a',buffering=1,encoding='utf-8') as fout:
        with contextlib.redirect_stdout(fout),contextlib.redirect_stderr(fout):
            print('\n'+'='*90,flush=True)
            print(f'v22r1 STRUCTURE {label} / seed {seed} / worker {worker_id} / '
                  f'Ti {len(entry["ti"])} / O {len(entry["original_o"])}',flush=True)
            print(f'GEOMETRY POLICY {entry["policy"].name}; '
                  f'Ti limits {np.round(entry["policy"].global_limits["Ti"],5).tolist()} '
                  f'O limits {np.round(entry["policy"].global_limits["O"],5).tolist()}',flush=True)
            try:
                offset=10007*(int(entry['database_id']) if not entry['is_reference']
                       else 100000+list(REF_NAMES).index(entry['ref_name']))
                winner,trials,npeaks,nops,outcomes=search_target(
                     entry['ti'],entry['cell'],ctx['lib'],ctx['bounds'],ctx['guard'],args,
                     seed+offset,calibration=entry['policy'])
                rows,history=method_stage_rows(label,entry['source_type'],entry['database_id'],
                           outcomes,ctx['standard_tio'],generated_dir,
                           save_phase_cifs=args.save_stage_cifs,stage_cif_prefix=label)
                diagnostics=[]
                for row in rows:
                    method=row['method'];stage=row['stage']
                    phase={'entrance_scf':'phase_initial','global_scf':'phase_global',
                           'global_plus_local':'phase_final','scf_plus_local':'phase_final'}[stage]
                    cand=outcomes[method][1][phase][0]
                    row['seed']=seed;row['geometry_policy']=args.geometry_policy
                    matched=periodic_oxygen_correspondence(entry['original_o'],cand['oxygen'],entry['cell'])
                    correspondence=None
                    if matched is not None:
                        correspondence,quality=matched
                        row.update(quality)
                    diagnostics.extend(_site_diagnostics(label,seed,'generated',method,stage,
                        cand['details'],ctx['lib'],entry['original_details'],match=correspondence))
                for check in history:check['seed']=seed
                hop,hstats=outcomes['hopping']
                base.update(candidate_publication_metrics(hop,ctx['standard_tio']))
                base['status']='COMPLETED'
                base['hopping_solver_only_s']=hstats['total_seconds']
                base['shared_preparation_seconds']=hstats.get('shared_preparation_seconds',0.)
                base['hopping_runtime_s']=hstats['total_seconds']+hstats.get('shared_preparation_seconds',0.)
                base['hopping_global_steps']=hstats['attempts']
                base['hopping_stop_reason']=hstats['stop_reason']
                base['hopping_generated_cif']=f'generated_cifs/{label}_hopping_generated.cif'
                base['best_method_by_objective']=winner['chosen_method']
                base['preparation_peaks']=npeaks;base['ti_symmetry_operations']=nops
                matched=periodic_oxygen_correspondence(entry['original_o'],hop['oxygen'],entry['cell'])
                if matched is not None:base.update(matched[1])
                for method,(cand,stats) in outcomes.items():
                    write_final_cif(entry['ti'],entry['cell'],cand['oxygen'],
                                    generated_dir/f'{label}_{method}_generated.cif')
                write_final_cif(entry['ti'],entry['cell'],winner['oxygen'],
                                generated_dir/f'{label}_best_generated.cif')
                output.update(status='COMPLETED',generated=base,method_rows=rows,
                              site_rows=diagnostics,checkpoints=history,
                              elapsed_s=time.perf_counter()-starting)
                print(f'FINISHED {label}: Ti={base["ti_cn6_good"]}/{len(entry["ti"])} '
                      f'O={base["o_cn3_good"]}/{len(entry["original_o"])} '
                      f'sharing={base["sharing_rms_A"]:.5f} '
                      f'v22-pass={base["ti_first_success"]}; '
                      f'{output["elapsed_s"]:.1f}s',flush=True)
            except BaseException as exc:
                # Python exceptions are recorded; abnormal process termination
                # will leave RUNNING and no new checkpoint, which resume retries.
                base['status']='ERROR';base['error']=f'{type(exc).__name__}: {exc}'
                output.update(status='ERROR',generated=base,error=base['error'],
                              elapsed_s=time.perf_counter()-starting)
                print(f'FAILED {label}: {base["error"]}',flush=True)
                traceback.print_exc()
    # Final report first; atomic result bundle last, so checkpoints are never
    # interpreted as complete until the written CIFs and report all exist.
    _atomic_text(target_dir/'report.md',_structure_markdown(entry,output,args))
    _atomic_pickle(target_dir/'result.pkl',output)
    _atomic_text(target_dir/'status.json',json.dumps(dict(structure_id=label,seed=seed,
                status=output['status'],worker=worker_id,elapsed_s=output['elapsed_s'],
                end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                error=output.get('error','')),indent=2)+'\n')
    return output['status'],float(output['elapsed_s'])


def _worker_process(worker_id,task_queue,result_queue,context_path,args):
    # Each process imports and loads the prepared *same* context once, not a
    # separately screened/calibrated subset. Tasks are assigned dynamically.
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    name=_device_init(args.device,worker_id if args.device=='cuda' else 0)
    with open(context_path,'rb') as f:ctx=pickle.load(f)
    global SHIFTS
    if 'shift_vectors' not in ctx:raise RuntimeError('Cached context missing learned periodic image translations')
    SHIFTS=np.asarray(ctx['shift_vectors'],dtype=float).copy()
    ctx['original_lookup']={r['structure_id']:r for r in ctx['original_rows']}
    _log(f'WORKER {worker_id} READY [{name}]; loaded shared preparation once')
    while True:
        item=task_queue.get()
        if item is None:break
        entry=ctx['prepared'][item]
        label=entry['structure_id']
        try:
            status,elapsed=_run_one_structure(entry,ctx,args,worker_id)
            result_queue.put((worker_id,label,status,elapsed,''))
        except BaseException as exc:
            err=f'{type(exc).__name__}: {exc}'
            path=Path(args.out)/'structures'/label
            path.mkdir(parents=True,exist_ok=True)
            with open(path/'run.log','a',encoding='utf-8') as f:
                f.write('\nWORKER UNHANDLED EXCEPTION: '+err+'\n'+traceback.format_exc())
            _atomic_text(path/'status.json',json.dumps(dict(structure_id=label,
                                  status='ERROR',worker=worker_id,error=err),indent=2)+'\n')
            result_queue.put((worker_id,label,'ERROR',0.,err))
    _log(f'WORKER {worker_id} EXIT')


def _aggregate(args,ctx,include_report=True,organizer_elapsed=0.):
    """Only organizer writes consolidated publication tables; one dataset.

    A structure contributes only when its atomic bundle plus CIFs are verified.
    Failed or interrupted attempts are visible in a separate status file.
    """
    out=Path(args.out);completed=[];pending=[];errors=[]
    for entry in ctx['prepared']:
        label=entry['structure_id'];bundle=_results_file(out,label)
        if _valid_result(out,entry,args):
            with open(bundle,'rb') as f:completed.append(pickle.load(f))
        else:
            state=out/'structures'/label/'status.json'
            detail={}
            if state.is_file():
                try:detail=json.loads(state.read_text(encoding='utf-8'))
                except Exception:pass
            pending.append(dict(structure_id=label,status=detail.get('status','PENDING'),
                           worker=detail.get('worker',''),error=detail.get('error','')))
            if detail.get('status')=='ERROR':errors.append(label)
    original_rows=ctx['original_rows'];generated_rows=[c['generated'] for c in completed]
    method_rows=[r for c in completed for r in c['method_rows']]
    checkpoints=[r for c in completed for r in c['checkpoints']]
    site_rows=ctx['original_site_rows']+[r for c in completed for r in c['site_rows']]
    calibration_rows=ctx['calibration_rows'];screening=ctx['screening']
    _atomic_csv(out/'table1_original_topology.csv',original_rows,
           ('structure_id','source_type','database_id','is_reference','space_group_symbol',
            'space_group_number','original_cif'))
    _atomic_csv(out/'table2_generation_quality.csv',generated_rows,
           ('structure_id','seed','source_type','database_id','status','ti_cn6_good','ti_total',
            'o_cn3_good','o_total','ti_geometry_bad','ti_geometry_bad_v21','o_geometry_bad',
            'o_geometry_bad_v21','objective_loss','sharing_rms_A','oxygen_match_rms_A','hopping_runtime_s'))
    _atomic_csv(out/'method_stage_comparison.csv',method_rows,
           ('structure_id','seed','source_type','database_id','method','stage','elapsed_cumulative_s',
            'objective_loss','sharing_rms_A','oxygen_match_rms_A'))
    _atomic_csv(out/'global_progress.csv',checkpoints,
           ('structure_id','seed','source_type','database_id','method','proposals','elapsed','objective'))
    _atomic_csv(out/'site_geometry_diagnostics.csv',site_rows,
           ('structure_id','seed','source','method','stage','species','site_index',
            'original_corresponding_site','cn','target_cn','cn_ok','v21_geometry_ok','v22_geometry_ok'))
    _atomic_csv(out/'calibration_thresholds.csv',calibration_rows,
           ('structure_id','is_reference','geometry_policy','quantile',
            'calibration_structure_count','Ti_calibration_sites','O_calibration_sites'))
    _atomic_csv(out/'screening_and_duplicates.csv',screening,
           ('structure_id','status','reason','duplicate_of','reference_overlap'))
    summary=nine_way_summary(method_rows)
    for r in summary:
        sub=[x for x in method_rows if x['method']==r['method'] and x['stage']==r['stage']]
        r['n_structures']=len({x['structure_id'] for x in sub})
        r['n_seeds']=1
        r['v21_ti_first_success_on_v22_structures']=sum(bool(x['ti_first_success_v21']) for x in sub)
        r['v22_ti_first_success_on_v22_structures']=sum(bool(x['ti_first_success']) for x in sub)
    _atomic_csv(out/'summary_9_combinations.csv',summary,
          ('method','stage','n','n_structures','n_seeds','ti_first_success','median_o_sharing_A',
           'median_objective','median_elapsed_cumulative_s'))
    by_structure,stability_summary=_stability_reports(method_rows)
    recovery=_recovery_report(method_rows)
    _atomic_csv(out/'stability_by_structure.csv',by_structure,
          ('structure_id','method','stage','n_seeds','successes','success_rate'))
    _atomic_csv(out/'stability_summary.csv',stability_summary,
          ('cohort','method','stage','n_structures','n_seeds','fully_successful_structures',
           'partial_success_structures','never_successful_structures','success_rate_over_runs'))
    _atomic_csv(out/'initial_and_global_recovery.csv',recovery,
          ('structure_id','seed','method','status','initial_success','after_global_success','after_local_success'))
    _atomic_csv(out/'pending_structures.csv',pending,('structure_id','status','worker','error'))
    status=dict(expected=len(ctx['prepared']),completed=len(completed),pending=len(pending),
                errors=len(errors),single_seed=args.seed,worker_count=args.workers,
                complete=(len(pending)==0),reference_calibration_reused=True,
                preparation_seconds=ctx['elapsed_preparation_s'],timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    _atomic_text(out/'run_status.json',json.dumps(status,indent=2)+'\n')
    if include_report:
        body=_publication_report_md(args,original_rows,generated_rows,summary,
                  ctx['skipped'],ctx['standard_tio'],ctx['per_ref'],len(completed),organizer_elapsed)
        body=_v22_summary_append(body,args,[],calibration_rows,original_rows,
                  method_rows,stability_summary,recovery,errors)
        body=body.replace('Juliette v22 — Full-dataset publication report',
                          'Juliette v22r1 — Single-seed centrally coordinated publication report')
        body=body.replace('### Stability across repeated seeds',
                          '### Single-seed outcomes (NOT seed-to-seed stability)')
        body=body.replace('- Multiple seeds estimate algorithm variability, not confidence intervals for chemistry.',
                          '- This v22r1 run uses one seed; seed-to-seed stability CANNOT be evaluated.')
        body=('> **IMPORTANT:** v22r1 runs one seed only. This report evaluates structure-level '\
              'success and optimizer differences, NOT stochastic reproducibility across seeds.\n\n'+body)
        if pending:
            body=(f'# INCOMPLETE v22r1 PUBLICATION DATA — {len(completed)}/{len(ctx["prepared"])} STRUCTURES\n\n'
                  +body)
        _atomic_text(out/'publication_report.md',body)
    return status


def _launch(args,ctx):
    out=Path(args.out)
    unfinished=[i for i,e in enumerate(ctx['prepared']) if not _valid_result(out,e,args)]
    total=len(ctx['prepared']);already=total-len(unfinished)
    _log(f'CENTRAL ORGANIZER: {total} structures; {already} verified complete; '
         f'{len(unfinished)} scheduled; {args.workers} dynamic workers, seed={args.seed}.')
    if not unfinished:
        return _aggregate(args,ctx,organizer_elapsed=0.)
    if args.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()<args.workers:
            raise RuntimeError(f'Need {args.workers} CUDA GPUs but only '
                 f'{torch.cuda.device_count() if torch.cuda.is_available() else 0} visible. '
                 'Check --gres=gpu:4, PyTorch CUDA module, and CUDA_VISIBLE_DEVICES.')
        _log('CENTRAL CUDA:',[(i,torch.cuda.get_device_name(i)) for i in range(args.workers)])
    # The longest-looking tasks are dispatched first; dynamic queue prevents
    # fixed-shard tail imbalance. This changes ONLY scheduling order, not RNG.
    unfinished.sort(key=lambda i:(len(ctx['prepared'][i]['ti']),
                                   len(ctx['prepared'][i]['original_o'])),reverse=True)
    spawn=mp.get_context('spawn')
    tasks=spawn.Queue();reports=spawn.Queue()
    for i in unfinished:tasks.put(i)
    for _ in range(args.workers):tasks.put(None)
    workers=[]
    started=time.perf_counter()
    for i in range(args.workers):
        proc=spawn.Process(target=_worker_process,
              args=(i,tasks,reports,str(out/'_central_preparation.pkl'),args),
              name=f'v22r1-worker-{i}')
        proc.start();workers.append(proc)
        _log(f'WORKER STARTED {i} pid={proc.pid}; gpu={i if args.device=="cuda" else "CPU"}')
    old_handlers={};interrupted=False
    def terminate(signum,frame):
        nonlocal interrupted
        interrupted=True
        _log(f'ORGANIZER SIGNAL {signum}; stopping workers; completed structure checkpoints remain valid.')
        for proc in workers:
            if proc.is_alive():proc.terminate()
    for signum in (signal.SIGTERM,signal.SIGINT):
        old_handlers[signum]=signal.getsignal(signum)
        signal.signal(signum,terminate)
    seen=set();count=0;last_info=time.monotonic()
    try:
        while len(seen)<len(unfinished) and not interrupted:
            try:
                worker_id,label,status,elapsed,error=reports.get(timeout=2)
            except __import__('queue').Empty:
                broken=[(i,proc.exitcode) for i,proc in enumerate(workers)
                        if proc.exitcode not in (None,0)]
                if broken:
                    _log('WORKER ABNORMAL EXIT',broken,'; remaining work can be resumed.')
                    interrupted=True
                    break
                if all(not proc.is_alive() for proc in workers):
                    _log('ALL WORKERS EXITED, but results missing; inspect per-structure logs.')
                    interrupted=True
                    break
                if time.monotonic()-last_info>=60:
                    _log(f'HEARTBEAT: completed this invocation {len(seen)}/{len(unfinished)}; '
                         f'total verified approximately {already+len(seen)}/{total}')
                    last_info=time.monotonic()
                continue
            if label in seen:continue
            seen.add(label);count+=1
            _log(f'FINISHED {already+count}/{total} {label} worker={worker_id} '
                 f'status={status} elapsed={elapsed:.1f}s'+(f' ERROR={error}' if error else ''))
            if count%args.report_every==0:
                summary=_aggregate(args,ctx,include_report=True,
                                   organizer_elapsed=time.perf_counter()-started)
                _log(f'CENTRAL REPORT updated: {summary["completed"]}/{summary["expected"]} complete')
    finally:
        if interrupted:
            for p in workers:
                if p.is_alive():p.terminate()
        for p in workers:
            p.join(timeout=8)
            if p.is_alive():p.kill();p.join(timeout=3)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        tasks.close();reports.close()
    summary=_aggregate(args,ctx,organizer_elapsed=time.perf_counter()-started)
    _log('CENTRAL SUMMARY:',json.dumps(summary,sort_keys=True))
    _log('STRUCTURE LOGS:',out/'structures'/'<structure_id>'/'run.log')
    _log('STRUCTURE REPORTS:',out/'structures'/'<structure_id>'/'report.md')
    _log('CONSOLIDATED REPORT:',out/'publication_report.md')
    if not summary['complete'] or interrupted:
        _log('INCOMPLETE: rerun SAME command with --resume; scientific settings must match.')
        raise SystemExit(2)
    return summary


def v22r1_main(argv=None):
    args=_parse_v22r1(argv)
    if args.self_test:
        self_test_publication()
        _self_test_v22r1_orchestrator()
        return
    _config_guard(args)
    _log('='*100)
    _log('JULIETTE v22r1 | ONE SEED | CENTRAL ORGANIZER | DYNAMIC GPU TASK QUEUE')
    _log('='*100)
    _log('POLICY:',args.geometry_policy,'SEED:',args.seed,'WORKERS:',args.workers,'DEVICE:',args.device)
    _log('Original target oxygen excluded from generation. Thresholds calibrated from other DB originals.')
    if args.aggregate_only:
        cache=Path(args.out)/'_central_preparation.pkl'
        if not cache.is_file():raise RuntimeError('Central preparation checkpoint missing')
        with open(cache,'rb') as f:context=pickle.load(f)
        summary=_aggregate(args,context,include_report=True)
        _log('AGGREGATE ONLY:',json.dumps(summary,indent=2))
        return
    context=_prepare_context(args)
    _launch(args,context)


def _self_test_v22r1_orchestrator():
    """Quick synthetic validation; real database test is external."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        cases=[{'structure_id':'db_000001','ti':np.zeros((1,3)),
                'original_o':np.zeros((2,3))},
               {'structure_id':'db_000002','ti':np.zeros((2,3)),
                'original_o':np.zeros((4,3))}]
        for case in cases:
            path=root/'structures'/case['structure_id']
            path.mkdir(parents=True)
            _atomic_pickle(path/'result.pkl',{'status':'COMPLETED',
                'structure_id':case['structure_id'],'seed':42})
        for c in cases:
            assert (root/'structures'/c['structure_id']/'result.pkl').is_file()
        a=argparse.Namespace(seed=42,methods=('hopping',),save_stage_cifs=False)
        assert not _valid_result(root,cases[0],a), 'should reject incomplete CIF bundle'
        _atomic_text(root/'test.txt','ok');assert (root/'test.txt').read_text()=='ok'
        _atomic_csv(root/'test.csv',[{'x':1,'y':2}]);assert (root/'test.csv').read_text().count('\n')==2
        # Validate true Python spawn lifecycle (rather than only forked mocks).
        _atomic_pickle(root/'minimal_context.pkl',{'prepared':[],'original_rows':[],
                       'shift_vectors':np.asarray(SHIFTS).copy(),
                       'geometry_policy_probe':GeometryPolicy('channel_q95',
                           np.array([.1,.2,.2,.2]),
                           {'Ti':np.array([.1,.2,.2,.2]),'O':np.array([.1,.2,.2,.2])})})
        spawn=mp.get_context('spawn')
        tq=spawn.Queue();rq=spawn.Queue();tq.put(None)
        worker=spawn.Process(target=_worker_process,
            args=(0,tq,rq,str(root/'minimal_context.pkl'),argparse.Namespace(device='cpu')))
        worker.start();worker.join(timeout=20)
        if worker.is_alive():worker.terminate();worker.join(timeout=4)
        assert worker.exitcode==0,f'spawned worker failed: {worker.exitcode}'
        tq.close();rq.close()
        _log('v22r1 central organizer atomicity / resume / spawn validation passed')


if __name__=='__main__':
    v22r1_main()
