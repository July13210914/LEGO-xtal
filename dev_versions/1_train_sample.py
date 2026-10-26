#!/usr/bin/env python3
"""Juliette v136r10 — Born-Oppenheimer B-SCF with three-state O-SCF tracking.

Minimal B-SCF repair characterization. For each of five selected space groups, generate one
independent Stage-A Ti framework, run one tested oxygen_growth Stage-B realization, then
run B-SCF without a quality rejection gate to measure before/after geometry repair.
There is no B-SCF rescue and no repeated Stage-B branching. Reports exact Ti12O24
yield, arm RMS, topology/structure diversity, timing, and exact candidates per GPU-minute.
"""
import argparse
import csv
import json
import multiprocessing as mp
import traceback
import re
from contextlib import redirect_stdout, redirect_stderr
import hashlib, math, os, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SHIFTS = np.asarray([(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)], dtype=int)
ZERO_SHIFT = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])
EPS = 1.0e-8


def encode_token(values) -> str:
    return json.dumps(list(values), separators=(",", ":"))

def decode_int_token(token: str) -> list[int]:
    return [int(x) for x in json.loads(token)]




def _hypergraph_topology_fingerprint(triplets, n_ti=12) -> str | None:
    # Permutation-invariant WL-style hash of the Ti--O incidence hypergraph.
    trips=[tuple(sorted(int(x) for x in tr)) for tr in triplets]
    if not trips:
        return None
    if any(len(tr)!=3 or min(tr)<0 or max(tr)>=int(n_ti) for tr in trips):
        return None
    n_ti=int(n_ti); n_o=len(trips); n=n_ti+n_o
    adj=[set() for _ in range(n)]
    for oi,tr in enumerate(trips):
        on=n_ti+oi
        for ti in tr:
            adj[ti].add(on); adj[on].add(ti)
    colors=['Ti']*n_ti+['O']*n_o
    for _ in range(6):
        sig=[colors[i]+'|'+','.join(sorted(colors[j] for j in adj[i])) for i in range(n)]
        palette={v:str(k) for k,v in enumerate(sorted(set(sig)))}
        colors=[palette[v] for v in sig]
    edge_colors=sorted(tuple(sorted((colors[i],colors[j]))) for i in range(n) for j in adj[i] if i<j)
    payload=repr((sorted(colors),edge_colors)).encode('utf-8')
    return hashlib.blake2b(payload,digest_size=12).hexdigest()

def deterministic_seed(seed: int, *parts) -> int:
    s = "|".join([str(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little") % (2**31-1)

def _parse_formula(formula: str) -> dict[str,int]:
    import re
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    if not out or "".join(f"{k}{v if v != 1 else ''}" for k,v in out.items()) == "":
        raise ValueError(f"Cannot parse formula {formula!r}")
    return out

@dataclass(frozen=True)
class PairChannel:
    species_i: str; species_j: str; mu: float; sigma: float

class ChemistryModel:
    def __init__(self, path: str):
        self.path = str(path)
        self.raw = json.loads(Path(path).read_text())
        if self.raw.get("schema") != "juliette_building_block_xn_v1":
            raise ValueError(f"Unsupported schema {self.raw.get('schema')!r}")
        self.items = {str(x["label"]): x for x in self.raw["construction_species"]}
        self.labels = tuple(self.items)
        self.final_formula = {k:str(v["final_formula"]) for k,v in self.items.items()}
        self.cn = {k:int(v["external_template"]["coordination_number"]) for k,v in self.items.items()}
        self.angles = {k:np.asarray(v["external_template"].get("angular_mean_deg",[]),float) for k,v in self.items.items()}
        self.angle_sigma = {k:np.maximum(np.asarray(v["external_template"].get("angular_sigma_deg",[]),float),1e-3) for k,v in self.items.items()}
        self.channels = {}
        for r in self.raw.get("pair_channels",[]):
            if r.get("relation") == "external_neighbor":
                self.channels[(str(r["species_i"]),str(r["species_j"]))] = PairChannel(
                    str(r["species_i"]),str(r["species_j"]),float(r["distance_mu_A"]),max(float(r["distance_sigma_A"]),1e-3))
        self.frameworks = {str(k):v for k,v in self.raw.get("framework_models",{}).items()}
        self.nonbonded = {}
        for r in self.raw.get("nonbonded_pair_exclusions",[]):
            a,b = str(r.get("element_i","")),str(r.get("element_j",""))
            if a and b: self.nonbonded[tuple(sorted((a,b)))] = float(r["hard_min_A"])

    def resolve_target(self, text: str):
        if "=" in text:
            formula,zs = text.split("=",1); z=int(zs)
        else: formula,z=text,1
        comp = _parse_formula(formula)
        counts = {}
        for label in self.labels:
            counts[label] = int(comp.get(self.final_formula[label],0))*z
        if sum(counts.values()) != sum(comp.values())*z:
            raise ValueError(f"Model labels cannot represent target {text!r}")
        active=[x for x,n in counts.items() if n>0]
        hi=max(self.cn[x] for x in active); lo=min(self.cn[x] for x in active)
        parents=tuple(x for x in active if self.cn[x]==hi); children=tuple(x for x in active if self.cn[x]==lo)
        if len(parents)!=1 or len(children)!=1 or hi==lo:
            raise ValueError("v93 experiment requires one high-CN parent and one low-CN child label")
        p,c=parents[0],children[0]
        if counts[p]*self.cn[p] != counts[c]*self.cn[c]:
            raise ValueError("Parent/child incidence mismatch")
        return formula,z,counts,p,c

    def pair(self,a,b): return self.channels[(a,b)]





def exact_parent_entries(spg:int, count:int, cap:int):
    """All Wyckoff multisets that exactly realize the parent count."""
    from pyxtal.symmetry import Group
    g=Group(int(spg)); mult=[int(g[i].multiplicity) for i in range(len(g))]; dof=[int(g[i].get_dof()) for i in range(len(g))]
    allowed=[i for i,m in enumerate(mult) if 1<=m<=count]
    rows=[]
    def visit(start,total,wps):
        if len(rows)>=cap: return
        if total==count:
            rows.append(encode_token(wps)); return
        for pos in range(start,len(allowed)):
            wp=allowed[pos]
            if dof[wp]==0 and wp in wps: continue
            nt=total+mult[wp]
            if nt<=count: visit(pos,nt,wps+[wp])
    visit(0,0,[])
    return list(dict.fromkeys(rows))


def _so3_power_torch(nv, centers, width, lmax):
    """Differentiable SO(3) power spectrum using Legendre addition theorem.

    nv: [B,N,K,3]. Returns normalized fingerprints [B,N,D].
    """
    r=torch.linalg.norm(nv,dim=-1).clamp_min(1e-7)
    u=nv/r[...,None]
    dots=torch.einsum("bnki,bnli->bnkl",u,u).clamp(-1.0,1.0)
    rb=torch.exp(-0.5*((r[...,None]-centers[None,None,None,:])/width)**2)
    P=[torch.ones_like(dots)]
    if lmax>=1: P.append(dots)
    for l in range(2,int(lmax)+1):
        P.append(((2*l-1)*dots*P[-1]-(l-1)*P[-2])/float(l))
    vals=[]; nr=int(centers.numel())
    for Pl in P:
        for a in range(nr):
            for b in range(a,nr):
                vals.append(torch.einsum("bnj,bnk,bnjk->bn",rb[...,a],rb[...,b],Pl))
    x=torch.stack(vals,dim=-1)
    return x/torch.linalg.norm(x,dim=-1,keepdim=True).clamp_min(1e-8)



def _astronomy_power_torch(nv, centers, width, lmax):
    """Differentiable v15 full-sky fingerprint for [B,N,K,3] neighbour vectors."""
    r=torch.linalg.norm(nv,dim=-1).clamp_min(1e-7); u=nv/r[...,None]
    dots=torch.einsum('bnji,bnki->bnjk',u,u).clamp(-1.0,1.0)
    rb=torch.exp(-0.5*((r[...,None]-centers[None,None,None,:])/max(float(width),1e-8))**2)
    occ=rb.mean(dim=2)
    P=[torch.ones_like(dots)]
    if lmax>=1: P.append(dots)
    for l in range(2,lmax+1): P.append(((2*l-1)*dots*P[-1]-(l-1)*P[-2])/l)
    vals=[]
    for Pl in P:
        for n in range(len(centers)):
            for m in range(n,len(centers)):
                vals.append(torch.einsum('bnj,bnk,bnjk->bn',rb[...,n],rb[...,m],Pl))
    power=torch.stack(vals,dim=-1)
    power=power/torch.linalg.norm(power,dim=-1,keepdim=True).clamp_min(1e-8)
    return torch.cat([occ,power],dim=-1)


def _soap_power_torch(nv, centers, width, lmax):
    r=torch.linalg.norm(nv,dim=-1).clamp_min(1e-7); u=nv/r[...,None]
    dots=torch.einsum('bnji,bnki->bnjk',u,u).clamp(-1.0,1.0)
    rb=torch.exp(-0.5*((r[...,None]-centers[None,None,None,:])/max(float(width),1e-8))**2)
    K=max(int(nv.shape[2]),1); occ=rb.sum(dim=2)/float(K)
    P=[torch.ones_like(dots)]
    if lmax>=1: P.append(dots)
    for l in range(2,lmax+1): P.append(((2*l-1)*dots*P[-1]-(l-1)*P[-2])/l)
    vals=[]
    for Pl in P:
        for n in range(len(centers)):
            for m in range(n,len(centers)):
                vals.append(torch.einsum('bnj,bnk,bnjk->bn',rb[...,n],rb[...,m],Pl)/float(K*K))
    return torch.cat([occ,torch.stack(vals,dim=-1)],dim=-1)


def _ace_power_torch(nv, centers, width, lmax):
    base=_soap_power_torch(nv,centers,width,lmax)
    r=torch.linalg.norm(nv,dim=-1).clamp_min(1e-7); u=nv/r[...,None]
    d=torch.einsum('bnji,bnki->bnjk',u,u).clamp(-1.0,1.0)
    rb=torch.exp(-0.5*((r[...,None]-centers[None,None,None,:])/max(float(width),1e-8))**2)
    K=max(int(nv.shape[2]),1); tri=[]
    for n in range(len(centers)):
        w=rb[...,n]
        tri.append(torch.einsum('bni,bnj,bnk,bnij,bnjk,bnki->bn',w,w,w,d,d,d)/float(K**3))
        area=1.0-d*d
        tri.append(torch.einsum('bni,bnj,bnk,bnij,bnjk->bn',w,w,w,area,d*d)/float(K**3))
    return torch.cat([base,torch.stack(tri,dim=-1)],dim=-1)

class TiOnlyBuilder:
    def _live(self, kind, **kw):
        q=getattr(self,'live_event_q',None)
        if q is not None:
            try: q.put({'kind':'live','event':kind,'worker':getattr(self,'live_worker_id',-1),'device':str(self.device),'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),**kw})
            except Exception: pass

    def __init__(self, model, parent, child, device="cpu", starts=24, screen_steps=12, refine_steps=24,
                 lr=0.055, neighbor_slots=6, motifs_per_ti=6, exclusion_weight=30.0,
                 motif_weight=1.0, shell_weight=0.0, aspect_weight=0.03,
                 entrance_min_factor=0.85, entrance_q90=6.0,
                 spherical_weight=1.0, gram_weight=0.15, entrance_gaussian_width=1.0, so3_weight=1.0, so3_rough_factor=1.5,
                 ambient_steps=80, ambient_lr=0.035, ambient_pressure=0.035, ambient_soft_width=0.10, so3_huber_delta=2.5,
                 natural_cutoff=5.0, natural_inner=3.8, natural_count_floor=12.0,
                 natural_lambda3_floor=0.060, natural_hole_angle=80.0,
                 natural_count_weight=0.20, natural_dim_weight=0.0, natural_coverage_weight=1.0,
                 natural_radial_restore_weight=8.0, natural_radial_restore_width=0.12,
                 natural_density_weight=1.0, natural_volume_weight=1.0,
                 natural_vacuum_weight=2.0, natural_vacuum_cos=0.45):
        self.model=model; self.parent=parent; self.child=child; self.device=torch.device(device)
        self.starts=int(starts); self.screen_steps=int(screen_steps); self.refine_steps=int(refine_steps); self.lr=float(lr)
        self.neighbor_slots=int(neighbor_slots); self.motifs_per_ti=int(motifs_per_ti)
        self.exclusion_weight=float(exclusion_weight); self.motif_weight=float(motif_weight); self.shell_weight=float(shell_weight); self.aspect_weight=float(aspect_weight)
        self.entrance_min_factor=float(entrance_min_factor); self.entrance_q90=float(entrance_q90)
        self.spherical_weight=float(spherical_weight); self.gram_weight=float(gram_weight); self.entrance_gaussian_width=max(0.05,float(entrance_gaussian_width)); self.so3_weight=float(so3_weight); self.so3_rough_factor=max(1.0,float(so3_rough_factor))
        self.ambient_steps=max(1,int(ambient_steps)); self.ambient_lr=float(ambient_lr); self.ambient_pressure=float(ambient_pressure); self.ambient_soft_width=max(0.03,float(ambient_soft_width)); self.so3_huber_delta=max(0.25,float(so3_huber_delta))
        self.natural_cutoff=float(natural_cutoff); self.natural_inner=min(float(natural_inner), self.natural_cutoff-0.05)
        self.natural_count_floor=float(natural_count_floor); self.natural_lambda3_floor=float(natural_lambda3_floor)
        self.natural_hole_cos=math.cos(math.radians(float(natural_hole_angle)))
        self.natural_count_weight=float(natural_count_weight); self.natural_dim_weight=float(natural_dim_weight); self.natural_coverage_weight=float(natural_coverage_weight)
        self.natural_radial_restore_weight=float(natural_radial_restore_weight); self.natural_radial_restore_width=max(0.03,float(natural_radial_restore_width))
        self.natural_density_weight=float(natural_density_weight); self.natural_volume_weight=float(natural_volume_weight)
        self.natural_vacuum_weight=float(natural_vacuum_weight); self.natural_vacuum_cos=float(natural_vacuum_cos)
        # Fixed approximately uniform spherical probes. They define directional coverage only;
        # they are not learned references and therefore do not encode a TiO2 polymorph.
        ng=32; ii=torch.arange(ng,dtype=torch.float32,device=self.device); golden=math.pi*(3.0-math.sqrt(5.0))
        zz=1.0-2.0*(ii+0.5)/ng; rr=torch.sqrt((1.0-zz*zz).clamp_min(0.0)); ph=golden*ii
        self.coverage_grid=torch.stack([rr*torch.cos(ph),rr*torch.sin(ph),zz],dim=1)
        self.shift=torch.as_tensor(SHIFTS,dtype=torch.float32,device=self.device)
        fw=model.frameworks.get(parent)
        if not fw:
            raise ValueError(f"No directly learned same-element framework model for {parent}; rerun 0_learn.py")
        self.fw=fw
        # v103 learner-v13 calibration: use training only to set the physical
        # radius and broad one-sided anisotropy ceilings, never a target fingerprint.
        nat=fw.get("spherical_naturalness") or {}
        ranks=nat.get("radial_rank_statistics") or {}
        if ranks.get("r12_A") and ranks.get("r16_A"):
            self.natural_inner=float(ranks["r12_A"]["median"])
            self.natural_cutoff=max(self.natural_inner+0.25,float(ranks["r16_A"]["q95"]))
        # v104test3: convert learned neighbour-rank radii into a broad cumulative
        # density profile N(R).  This is bidirectional: too few neighbours pulls
        # matter inward, while too many pushes it outward.  The same r12 scale
        # gives a model-contained volume/Ti estimate, V ~= sphere(r12)/12.
        self.density_profile=[]
        for kk in (6,12,16,20,24):
            row=ranks.get(f"r{kk}_A")
            if row and row.get("median") is not None:
                self.density_profile.append((float(row["median"]),float(kk)))
        if not self.density_profile:
            self.density_profile=[(float(self.natural_inner),float(self.natural_count_floor))]
        r12row=ranks.get("r12_A")
        r12=float(r12row["median"]) if r12row and r12row.get("median") is not None else float(self.natural_inner)
        self.volume_target_per_ti=(4.0*math.pi/3.0)*r12**3/12.0
        self.natural_lambda3_floor=max(0.02,0.80*float(nat.get("lambda3_q05",self.natural_lambda3_floor)))
        q95=nat.get("low_l_abs_pair_legendre_q95") or {}
        self.low_l_ceiling=torch.tensor([1.25*float(q95.get(str(l),0.20)) for l in range(1,5)],dtype=torch.float32,device=self.device)
        self.k=int(fw["neighbor_count"])
        self.rad_mu=torch.tensor(fw["radial_mean_A"],dtype=torch.float32,device=self.device)
        self.rad_sig=torch.tensor(fw["radial_sigma_A"],dtype=torch.float32,device=self.device).clamp_min(0.04)
        self.angle_groups=[]
        for g in fw.get("angular_shell_pair_groups",[]):
            pairs=torch.tensor(g["neighbor_rank_pairs"],dtype=torch.long,device=self.device)
            mu=torch.tensor(g["angular_mean_deg"],dtype=torch.float32,device=self.device)
            sig=torch.tensor(g["angular_sigma_deg"],dtype=torch.float32,device=self.device).clamp_min(4.0)
            self.angle_groups.append((pairs,mu,sig))
        elem=model.final_formula[parent]; self.hard=float(model.nonbonded.get(tuple(sorted((elem,elem))), float(fw.get("radial_lower_bound_A",0.5))))
        # Learned relaxed short-range scale.  Use the nonbonded reference minimum when
        # available (Ti model v9/v13: ~2.95 A), falling back to the first Ti--Ti rank.
        ref_rows=[r for r in model.raw.get("nonbonded_pair_exclusions",[]) if tuple(sorted((str(r.get("element_i","")),str(r.get("element_j","")))))==tuple(sorted((elem,elem)))]
        self.titi_restore=float(ref_rows[0].get("reference_min_A", float(self.rad_mu[0]))) if ref_rows else float(self.rad_mu[0])
        self.pass_q90=float(fw.get("score_q90_max",3.5))
        sph=fw.get("spherical_neighbor_cloud") or {}
        if not sph:
            raise ValueError("v97 requires learner v11 spherical_neighbor_cloud; rerun 0_learn_v11.py")
        if int(sph.get("neighbor_count", self.k)) != self.k:
            raise ValueError("spherical_neighbor_cloud neighbor_count disagrees with framework neighbor_count")
        self.sph_eig_mu=torch.tensor(sph["second_moment_eigenvalue_mean"],dtype=torch.float32,device=self.device)
        self.sph_eig_sig=torch.tensor(sph["second_moment_eigenvalue_sigma"],dtype=torch.float32,device=self.device).clamp_min(0.03)
        self.sph_plan_median=float(sph["planarity_ratio_lambda3_over_lambda1_median"])
        self.sph_plan_q05=float(sph["planarity_ratio_lambda3_over_lambda1_q05"])
        self.sph_plan_q95=float(sph["planarity_ratio_lambda3_over_lambda1_q95"])
        self.gram_mu=torch.tensor(sph["gram_mean"],dtype=torch.float32,device=self.device)
        self.gram_sig=torch.tensor(sph.get("gram_sigma", np.ones((self.k,self.k))),dtype=torch.float32,device=self.device).clamp_min(0.08)
        so3=fw.get("so3_environment_distribution") or {}
        if not so3:
            raise ValueError("v101 requires learner v12 so3_environment_distribution; rerun 0_learn_v12.py")
        self.so3_centers=torch.tensor(so3["radial_centers_A"],dtype=torch.float32,device=self.device)
        self.so3_width=float(so3["radial_width_A"]); self.so3_lmax=int(so3["lmax"])
        self.so3_mu=torch.tensor(so3["fingerprint_mean"],dtype=torch.float32,device=self.device)
        self.so3_eval=torch.tensor(so3["regularized_eigenvalues"],dtype=torch.float32,device=self.device).clamp_min(1e-8)
        self.so3_evec=torch.tensor(so3["covariance_eigenvectors"],dtype=torch.float32,device=self.device)
        self.so3_q95=float(so3["mahalanobis_mean_per_dim_q95"])
        self.so3_q90=float(so3["mahalanobis_mean_per_dim_q90"])
        self.env_models={}
        for key,jsonkey in (("SKY","astronomy_environment"),("SOAP","soap_environment"),("ACE","ace_environment")):
            q=fw.get(jsonkey) or {}
            if not q:
                raise ValueError(f"v128 requires learner v16 {jsonkey}; rerun 0_learn_v16.py")
            self.env_models[key]={
                "k":int(q["neighbor_count"]),"lmax":int(q["lmax"]),
                "centers":torch.tensor(q["radial_centers_A"],dtype=torch.float32,device=self.device),
                "width":float(q["radial_width_A"]),
                "med":torch.tensor(q["fingerprint_median"],dtype=torch.float32,device=self.device),
                "scale":torch.tensor(q["fingerprint_scale"],dtype=torch.float32,device=self.device).clamp_min(0.01),
                "median":float(q["score_median"]),"q90":float(q["score_q90"]),"q95":float(q["score_q95"])}
        self.environment_tau=1.0
        # v125 rigid learned external-environment SO3 distributions.  These are
        # covariance-whitened empirical basins learned separately for TiO6 and OTi3.
        # No canonical-reference MSE and no log compression is permitted here.
        ext=self.model.items[self.parent].get("external_template",{})
        oext=self.model.items[self.child].get("external_template",{})
        cv=np.asarray(ext.get("canonical_vectors_A",[]),float)
        if len(cv) < self.model.cn[self.parent]:
            raise ValueError("v125 requires Ti external_template canonical_vectors_A")
        cv=cv[:self.model.cn[self.parent]]
        def load_local_so3(e,label):
            q=e.get("so3_environment_distribution") or {}
            if not q:
                raise ValueError(f"v125 requires learner v14 local SO3 distribution for {label}; rerun 0_learn_v14.py")
            return {
                'centers':torch.tensor(q['radial_centers_A'],dtype=torch.float32,device=self.device),
                'width':float(q['radial_width_A']),'lmax':int(q['lmax']),
                'mu':torch.tensor(q['fingerprint_mean'],dtype=torch.float32,device=self.device),
                'eval':torch.tensor(q['regularized_eigenvalues'],dtype=torch.float32,device=self.device).clamp_min(1e-12),
                'evec':torch.tensor(q['covariance_eigenvectors'],dtype=torch.float32,device=self.device),
                'q90':float(q['mahalanobis_mean_per_dim_q90']),'q95':float(q['mahalanobis_mean_per_dim_q95']),
                'q99':float(q.get('mahalanobis_mean_per_dim_q99',q['mahalanobis_mean_per_dim_q95'])),
                'max_train':float(q.get('mahalanobis_mean_per_dim_max',q.get('mahalanobis_mean_per_dim_q99',q['mahalanobis_mean_per_dim_q95'])))
            }
        self.tio6_so3_model=load_local_so3(ext,'TiO6')
        self.oti3_so3_model=load_local_so3(oext,'OTi3')
        # v137: graph-free multi-basin local SOAP fields learned from accepted
        # TiO6 and OTi3 environments.  These are the Stage-B SCF chemistry.
        def load_local_soap(e,label):
            q=e.get('soap_environment_distribution') or {}
            if not q:
                raise ValueError(f"v137 requires learner v17 local SOAP distribution for {label}; rerun 0_learn_v17.py")
            return {
                'k':int(q['neighbor_count']),'centers':torch.tensor(q['radial_centers_A'],dtype=torch.float32,device=self.device),
                'width':float(q['radial_width_A']),'lmax':int(q['lmax']),
                'med':torch.tensor(q['fingerprint_median'],dtype=torch.float32,device=self.device),
                'scale':torch.tensor(q['fingerprint_scale'],dtype=torch.float32,device=self.device).clamp_min(0.01),
                'proto':torch.tensor(q['prototypes'],dtype=torch.float32,device=self.device),
                'tau':float(q.get('softmin_tau',0.1)),'q90':max(float(q.get('nearest_score_q90',1.0)),0.05),
                'q95':max(float(q.get('nearest_score_q95',1.0)),0.05)
            }
        self.tio6_soap_model=load_local_soap(ext,'TiO6')
        self.oti3_soap_model=load_local_soap(oext,'OTi3')
        # Compatibility names used only by startup reporting.
        self.oxygen_so3_centers=self.tio6_so3_model['centers']; self.oxygen_so3_width=self.tio6_so3_model['width']; self.oxygen_so3_lmax=self.tio6_so3_model['lmax']
        self.oxygen_so3_ref=self.tio6_so3_model['mu']
        # v108 rigid TiO6 proposal arms: preserve learned/canonical local cage shape.
        self.oxygen_base_vectors=torch.tensor(cv,dtype=torch.float32,device=self.device)
        cel=self.model.final_formula[self.child]
        self.oo_hard=float(self.model.nonbonded.get(tuple(sorted((cel,cel))),2.2))
        self._cache={}

    @staticmethod
    def lattice_spec(lt):
        lt=str(lt).lower()
        if lt=="cubic": return ("a",)
        if lt in {"tetragonal","hexagonal","trigonal"}: return ("a","c")
        if lt=="orthorhombic": return ("a","b","c")
        if lt=="monoclinic": return ("a","b","c","beta")
        return ("a","b","c","alpha","beta","gamma")

    @staticmethod
    def affine_map(function,dof):
        zero=np.asarray(function(np.zeros(dof)),float); A=np.zeros((3,dof),float)
        for k in range(dof):
            x=np.zeros(dof); x[k]=0.137; y=np.asarray(function(x),float); A[:,k]=((y-zero+0.5)%1.0-0.5)/0.137
        return A,zero

    def template(self,spg,wp_token):
        from pyxtal.symmetry import Group
        key=(int(spg),wp_token)
        if key in self._cache:return self._cache[key]
        g=Group(int(spg)); wps=decode_int_token(wp_token)
        dofs=[]; As=[]; bs=[]; rots=[]; trans=[]
        for wp_id in wps:
            wp=g[int(wp_id)]; d=int(wp.get_dof()); A,b=self.affine_map(lambda u,wp=wp:wp.get_position_from_free_xyzs(u),d)
            dofs.append(d); As.append(torch.tensor(A,dtype=torch.float32,device=self.device)); bs.append(torch.tensor(b,dtype=torch.float32,device=self.device))
            rots.append(torch.tensor(np.asarray([op.rotation_matrix for op in wp.ops],float),dtype=torch.float32,device=self.device))
            trans.append(torch.tensor(np.asarray([op.translation_vector for op in wp.ops],float),dtype=torch.float32,device=self.device))
        t={"spg":int(spg),"group":g,"lattice_type":str(g.lattice_type).lower(),"spec":self.lattice_spec(g.lattice_type),"wps":tuple(wps),"dofs":tuple(dofs),"A":As,"b":bs,"rots":rots,"trans":trans,"n":sum(len(x) for x in rots)}
        self._cache[key]=t; return t

    def lattice(self,t,raw):
        b=len(raw); nlat=len(t["spec"]); lengths=torch.nn.functional.softplus(raw[:,:nlat])+1.2; lt=t["lattice_type"]
        if lt=="cubic": a=lengths[:,0]; abc=torch.stack([a,a,a],1); ang=torch.full((b,3),math.pi/2,device=self.device)
        elif lt in {"tetragonal","hexagonal","trigonal"}:
            a,c=lengths[:,0],lengths[:,1]; abc=torch.stack([a,a,c],1); ang=torch.full((b,3),math.pi/2,device=self.device)
            if lt in {"hexagonal","trigonal"}: ang[:,2]=2*math.pi/3
        elif lt=="orthorhombic": abc=lengths[:,:3]; ang=torch.full((b,3),math.pi/2,device=self.device)
        elif lt=="monoclinic": abc=lengths[:,:3]; beta=math.pi/3+torch.sigmoid(raw[:,3])*math.pi/3; ang=torch.stack([torch.full_like(beta,math.pi/2),beta,torch.full_like(beta,math.pi/2)],1)
        else: abc=lengths[:,:3]; ang=math.pi/4+torch.sigmoid(raw[:,3:6])*math.pi/2
        a,bb,c=abc.unbind(1); al,be,ga=ang.unbind(1); ca,cb,cg=torch.cos(al),torch.cos(be),torch.cos(ga); sg=torch.sin(ga).clamp_min(1e-4)
        y3=c*(ca-cb*cg)/sg; z2=c*c-(c*cb)**2-y3*y3; z=torch.sqrt(z2.clamp_min(1e-6)); zero=torch.zeros_like(a)
        cell=torch.stack([torch.stack([a,zero,zero],1),torch.stack([bb*cg,bb*sg,zero],1),torch.stack([c*cb,y3,z],1)],1)
        return abc,ang,cell,z2

    def expand(self,t,coord_raw):
        centres=[]; free=[]; cur=0
        for d,A,b,R,T in zip(t["dofs"],t["A"],t["b"],t["rots"],t["trans"]):
            u=torch.sigmoid(coord_raw[:,cur:cur+d]); cur+=d; free.append(u)
            gen=(u@A.T+b)%1.0; orb=(torch.einsum("oij,bj->boi",R,gen)+T[None])%1.0; centres.append(orb)
        return torch.cat(centres,1),free

    def geometry(self,t,raw):
        nlat=len(t["spec"]); abc,ang,cell,z2=self.lattice(t,raw); centres,free=self.expand(t,raw[:,nlat:]); return abc,ang,cell,z2,centres,free

    def loss(self,t,raw,details=False):
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        delta=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vec=torch.einsum("bijnk,bkl->bijnl",delta,cell); dist=torch.linalg.norm(vec,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]; zero=torch.arange(27,device=self.device)==ZERO_SHIFT
        dist=dist.masked_fill(eye & zero[None,None,None,:],1e6)
        flatd=dist.reshape(B,N,N*27); flatv=vec.reshape(B,N,N*27,3)
        K=min(self.k,N*27-1); nd,idx=torch.topk(flatd,K,dim=-1,largest=False)
        nv=torch.gather(flatv,2,idx[...,None].expand(-1,-1,-1,3))
        rmu=self.rad_mu[:K]; rsig=self.rad_sig[:K]
        rz=torch.abs(nd-rmu[None,None,:])/rsig[None,None,:]
        radial=(rz*rz).mean(dim=(1,2))
        az_all=[]; angular_terms=[]
        u=nv/nd[...,None].clamp_min(1e-6)
        # v99 primary chemistry coordinate: smooth SO(3)-invariant power spectrum.
        # We score against the learned *distribution* in descriptor space, not a
        # discrete reference.  Covariance makes genuine TiO2 distortion directions
        # soft while off-manifold directions remain expensive.
        fp=_so3_power_torch(nv,self.so3_centers,self.so3_width,self.so3_lmax)
        dz=(fp-self.so3_mu[None,None,:])@self.so3_evec
        so3_local=(dz*dz/self.so3_eval[None,None,:]).mean(-1)
        so3_loss=so3_local.mean(1)
        # v11 diagnostics retained below; they no longer drive the main chemistry.
        # v97: retain the *joint 3D spherical distribution* of the k learned
        # first-neighbour directions.  The second-moment spectrum is invariant to
        # global rotation/reflection and immediately exposes planar collapse.
        moment=torch.einsum("bnki,bnkj->bnij",u,u)/float(max(K,1))
        eig=torch.linalg.eigvalsh(moment).flip(-1)
        emu=self.sph_eig_mu[:eig.shape[-1]]; esig=self.sph_eig_sig[:eig.shape[-1]]
        spherical=(((eig-emu[None,None,:])/esig[None,None,:])**2).mean(dim=(1,2))
        planarity=eig[...,2]/eig[...,0].clamp_min(1e-6)
        # Gram term preserves more of the neighbour-on-sphere arrangement.  Keep
        # it weak because radial-rank ties can swap nearly equivalent neighbours.
        gram=torch.einsum("bnki,bnli->bnkl",u,u)
        gmu=self.gram_mu[:K,:K]; gsig=self.gram_sig[:K,:K]
        tri=torch.triu_indices(K,K,1,device=self.device)
        gz=(gram[:,:,tri[0],tri[1]]-gmu[tri[0],tri[1]][None,None,:])/gsig[tri[0],tri[1]][None,None,:]
        gram_loss=(gz*gz).mean(dim=(1,2)) if gz.shape[-1] else torch.zeros_like(radial)
        for pairs,mu,sig in self.angle_groups:
            valid=(pairs[:,0]<K)&(pairs[:,1]<K); pp=pairs[valid]
            if len(pp)==0: continue
            dots=(u[:,:,pp[:,0],:]*u[:,:,pp[:,1],:]).sum(-1).clamp(-1+1e-6,1-1e-6)
            obs=torch.rad2deg(torch.acos(dots)); obs=torch.sort(obs,dim=-1).values
            mm=mu[:obs.shape[-1]]; ss=sig[:obs.shape[-1]]
            az=torch.abs(obs-mm[None,None,:])/ss[None,None,:]
            az_all.append(az); angular_terms.append((az*az).mean(dim=(1,2)))
        angular=torch.stack(angular_terms,0).mean(0) if angular_terms else torch.zeros_like(radial)
        local_z=torch.cat([rz]+az_all,dim=-1) if az_all else rz
        q90=torch.quantile(local_z,0.90,dim=-1).mean(1)
        mind=flatd.amin(-1).amin(-1)
        exclusion=torch.relu((self.hard-mind)/max(self.hard,0.2)).pow(2)
        aspect=(abc.amax(1)/abc.amin(1).clamp_min(1e-4)-1.0).pow(2)
        metric=torch.relu(1e-4-z2)/1e-4
        # v101 capture potential: Huber loss in whitened SO(3)/PCA coordinates.
        # Quadratic close to the learned manifold, linear far away: unlike log1p,
        # the long-range chemical driving force does not die off.
        wz=torch.abs(dz)/torch.sqrt(self.so3_eval[None,None,:])
        delta=self.so3_huber_delta
        huber=torch.where(wz<=delta,0.5*wz*wz,delta*(wz-0.5*delta))
        robust_so3_local=huber.mean(-1)
        robust_so3=robust_so3_local.mean(1)
        total=self.so3_weight*robust_so3+0.03*radial+0.01*angular+self.exclusion_weight*exclusion+self.aspect_weight*aspect+10.0*metric
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if not details:return total
        return total,{"radial_loss":radial,"angular_loss":angular,"so3_loss":so3_loss,"robust_so3_loss":robust_so3,"so3_local":so3_local,"spherical_loss":spherical,"gram_loss":gram_loss,"planarity_ratio":planarity.mean(1),"local_q90":q90,"minimum_titi_A":mind,"exclusion":exclusion,"aspect":aspect,"metric_penalty":metric,"metric_valid":z2>0,"nd":nd},(abc,ang,cell,frac,free)

    @staticmethod
    def _inv_softplus(y):
        y=torch.as_tensor(y)
        return torch.log(torch.expm1(y.clamp_min(1.0e-5)).clamp_min(1.0e-8))

    def _length_dof_count(self,t):
        return 1 if t["lattice_type"]=="cubic" else (2 if t["lattice_type"] in {"tetragonal","hexagonal","trigonal"} else 3)


    def initial_raw(self,t,nstarts=None):
        nstarts=self.starts if nstarts is None else max(1,int(nstarts))
        nlat=len(t["spec"]); ncoord=sum(t["dofs"]); raw=torch.zeros((nstarts,nlat+ncoord),device=self.device)
        # v100 deliberately starts dilute and lets Stage A compression + repulsion find
        # an ambient condensed scale.  No learned radial target is used for entrance scale.
        nlen=self._length_dof_count(t)
        base=max(5.0,4.5*(max(t["n"],1)**(1/3)))
        raw[:,:nlen]=float(self._inv_softplus(torch.tensor(base-1.2)).item())
        if nlat>nlen:
            raw[:,nlen:nlat]=0.0
        # v98: continuous Wyckoff entrances.  Avoid exact 0/1 boundaries but do not
        # quantize the crystallographic free coordinates onto an artificial grid.
        if ncoord:
            u=0.02+0.96*torch.rand((nstarts,ncoord),device=self.device)
            raw[:,nlat:]=torch.log(u/(1.0-u))
        return raw

    def ambient_loss(self,t,raw,details=False):
        """Chemistry-light conditioning: Ti--Ti soft repulsion versus hydrostatic compression.

        This stage deliberately does not use radial ranks, angles, Gram matrices, or SO(3).
        Compression prevents the trivial dilute solution; the learned Ti--Ti nonbonded wall
        prevents collapse.  It is a conditioner, never an acceptance gate.
        """
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        delta=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vec=torch.einsum("bijnk,bkl->bijnl",delta,cell); dist=torch.linalg.norm(vec,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]; zero=torch.arange(27,device=self.device)==ZERO_SHIFT
        selfzero=eye & zero[None,None,None,:]
        # Smooth wall centered at learned physical Ti--Ti hard exclusion.
        pair=torch.nn.functional.softplus((self.hard-dist)/self.ambient_soft_width).pow(2)
        pair=pair.masked_fill(selfzero,0.0)
        rep=pair.sum((1,2,3))/max(N,1)
        vol=torch.abs(torch.linalg.det(cell))/max(N,1)
        # Constant-pressure-like compression.  Linear V gives a nonzero force even for
        # very dilute cells and naturally balances the short-range repulsive wall.
        comp=self.ambient_pressure*vol
        aspect=(abc.amax(1)/abc.amin(1).clamp_min(1e-4)-1.0).pow(2)
        metric=torch.relu(1e-4-z2)/1e-4
        total=rep+comp+0.01*aspect+10.0*metric
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if not details:return total
        d=dist.masked_fill(selfzero,1e6)
        return total,{"ambient_repulsion":rep,"volume_per_ti_A3":vol,"minimum_titi_A":d.amin((1,2,3)),"metric_valid":z2>0}

    def naturalness_loss(self,t,raw,details=False):
        """Bidirectional condensed-matter regularizer.

        v104test3 keeps the successful short-contact restorative force, but replaces
        unit-sphere pair statistics with two-sided real-space density.  Learned
        neighbour-rank radii define cumulative N(R) targets; both emptiness and
        over-crowding generate gradients.  A broad volume/Ti term supplies global
        pressure in both directions.
        """
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        delta=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vec=torch.einsum("bijnk,bkl->bijnl",delta,cell); dist=torch.linalg.norm(vec,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]; zero=torch.arange(27,device=self.device)==ZERO_SHIFT
        selfzero=eye & zero[None,None,None,:]
        dist=dist.masked_fill(selfzero,1e6)
        flatd=dist.reshape(B,N,N*27); flatv=vec.reshape(B,N,N*27,3); r=flatd

        # Smooth cumulative counts N(R).  A sigmoid rather than a hard cutoff gives
        # useful forces to neighbours on either side of each learned shell radius.
        density_terms=[]; density_counts=[]
        shell_width=0.18
        for R,target in self.density_profile:
            nr=torch.sigmoid((R-r)/shell_width).sum(-1)
            # Broad relative scale: tolerate about 20% of the target count before
            # the penalty becomes order unity; Huber keeps distant entrances stable.
            z=(nr-target)/max(1.5,0.20*target)
            az=z.abs(); hub=torch.where(az<=1.0,0.5*z*z,az-0.5)
            density_terms.append(hub); density_counts.append(nr)
        density_local=torch.stack(density_terms,dim=-1).mean(-1)
        density_loss=density_local.mean(1)
        density_counts=torch.stack(density_counts,dim=-1)

        # v121 directional-vacuum penalty. Radial N(R) can be correct while a large
        # solid-angle sector is empty. Probe the sphere around every Ti and ask whether
        # each direction has a reasonably nearby Ti. The score is angular alignment
        # minus a smooth penalty for neighbours beyond the learned outer density shell.
        # We optimize only the worst quarter of probe directions, so this closes holes
        # without trying to make the Ti environment artificially isotropic.
        unitv=flatv/flatd[...,None].clamp_min(1e-6)
        probe_dot=torch.einsum("bnkd,gd->bngk",unitv,self.coverage_grid)
        Rvac=max(R for R,_ in self.density_profile)
        radial_excess=torch.nn.functional.softplus((flatd-Rvac)/0.30)
        support_score=probe_dot-0.75*radial_excess[:,:,None,:]
        tau_cov=0.08
        sw=torch.softmax(support_score/tau_cov,dim=-1)
        directional_support=(sw*support_score).sum(-1)  # [B,N,G], ~cosine of best nearby Ti
        vacuum_deficit=torch.relu(self.natural_vacuum_cos-directional_support)/0.20
        nv=max(1,self.coverage_grid.shape[0]//4)
        worst_deficit=torch.topk(vacuum_deficit,nv,dim=-1).values
        vacuum_loss=(worst_deficit*worst_deficit).mean((1,2))
        vacuum_min_support=directional_support.amin(-1).mean(1)
        vacuum_q10_support=torch.quantile(directional_support,0.10,dim=-1).mean(1)

        # Global bidirectional pressure.  The target is inferred from learned r12,
        # so no extra learner field is required.  A broad log-volume Huber penalty
        # treats expansion and compression symmetrically in multiplicative space.
        vol=torch.abs(torch.linalg.det(cell))/max(N,1)
        zv=torch.log((vol/self.volume_target_per_ti).clamp_min(1e-5))/0.30
        az=zv.abs(); volume_loss=torch.where(az<=1.0,0.5*zv*zv,az-0.5)

        # Keep v104's successful short-range restorative force.
        short=torch.nn.functional.softplus((self.titi_restore-r)/self.natural_radial_restore_width).pow(2)
        short=torch.where(r<self.titi_restore+0.75,short,torch.zeros_like(short))
        radial_restore=short.sum(-1).mean(1)/max(N,1)

        # Real-space covariance is diagnostic only in this test: it lets us see
        # whether fixing emptiness also improves chain/slab morphology.
        Rcov=max(R for R,_ in self.density_profile)
        wcov=torch.sigmoid((Rcov-r)/0.20); wsum=wcov.sum(-1).clamp_min(1e-6)
        cov=torch.einsum("bnk,bnki,bnkj->bnij",wcov,flatv,flatv)/wsum[...,None,None]
        eig=torch.linalg.eigvalsh(cov).clamp_min(1e-10)
        chain_ratio=(eig[...,1]/eig[...,2]).mean(1)
        slab_ratio=(eig[...,0]/eig[...,1]).mean(1)

        mind=flatd.amin(-1).amin(-1)
        aspect=(abc.amax(1)/abc.amin(1).clamp_min(1e-4)-1.0).pow(2)
        metric=torch.relu(1e-4-z2)/1e-4
        total=(self.natural_density_weight*density_loss
               +self.natural_vacuum_weight*vacuum_loss
               +self.natural_volume_weight*volume_loss
               +self.natural_radial_restore_weight*radial_restore
               +self.aspect_weight*aspect+10.0*metric)
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if not details:return total
        return total,{"natural_density_loss":density_loss,"natural_vacuum_loss":vacuum_loss,
                      "vacuum_min_support":vacuum_min_support,"vacuum_q10_support":vacuum_q10_support,
                      "natural_volume_loss":volume_loss,
                      "natural_radial_restore_loss":radial_restore,"density_counts":density_counts.mean(1),
                      "volume_per_ti_A3":vol,"chain_ratio_mu2_mu1":chain_ratio,
                      "slab_ratio_mu3_mu2":slab_ratio,"minimum_titi_A":mind,
                      "metric_valid":z2>0,"aspect":aspect,"metric_penalty":metric},(abc,ang,cell,frac,free)



    def overcondense_loss(self,t,raw,details=False):
        """v121 temporary topology-discovery packing potential.

        Start from the v121 condensed framework, retain its density/vacuum/short-contact
        physics, but replace the learned equilibrium volume restraint by a deliberately
        smaller temporary V/Ti target.  This state is never treated as a final crystal;
        it exists only to create a redundant pool of geometrically possible OTi3 sites.
        """
        L,N,G=self.naturalness_loss(t,raw,details=True)
        target=float(getattr(self,'overcondense_volume_A3',30.0))
        zv=torch.log((N['volume_per_ti_A3']/max(target,1e-6)).clamp_min(1e-5))/0.22
        az=zv.abs(); over_v=torch.where(az<=1.0,0.5*zv*zv,az-0.5)
        # Remove v121's equilibrium-volume term, then impose the temporary compressed target.
        total=L-self.natural_volume_weight*N['natural_volume_loss']+float(getattr(self,'overcondense_volume_weight',40.0))*over_v
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if not details:return total
        N=dict(N); N['overcondense_volume_loss']=over_v; N['overcondense_target_A3']=target
        return total,N,G










    @staticmethod
    def _axis_angle_rotation(w):
        theta=torch.linalg.norm(w,dim=-1,keepdim=True).clamp_min(1.0e-8)
        axis=w/theta; x,y,z=axis.unbind(-1); q=torch.zeros_like(x)
        K=torch.stack([q,-z,y,z,q,-x,-y,x,q],dim=-1).reshape(*w.shape[:-1],3,3)
        I=torch.eye(3,dtype=w.dtype,device=w.device).expand(*w.shape[:-1],3,3)
        return I+torch.sin(theta)[...,None]*K+(1.0-torch.cos(theta)[...,None])*torch.matmul(K,K)


    def _nearest_arm_matching(self, arms_cart, cell):
        """Dynamic nearest assignment of 72 rigid TiO6 vertices into OTi3 packs.

        No Ti3 or 6^3 hyperedge enumeration.  For every possible seed arm we find
        its nearest arm on a different Ti and then the best third arm on a third Ti,
        rank these local O3 packs by merge variance, and greedily take disjoint packs.
        Arm identities are only instantaneous geometric labels; they may switch as
        the octahedra rotate.  This makes Stage B a piecewise-smooth nearest-assignment
        problem rather than a fixed combinatorial hypergraph.
        """
        A=np.asarray(arms_cart,float).reshape(-1,3); C=np.asarray(cell,float); inv=np.linalg.inv(C)
        narm=len(A); nti=narm//6; af=(A@inv)%1.0; owner=np.arange(narm)//6
        # Minimum-image displacement/distance between all 72 current O proposals.
        df=af[None,:,:]-af[:,None,:]; df-=np.rint(df); dc=np.einsum('abi,ij->abj',df,C)
        d2=np.sum(dc*dc,axis=-1); np.fill_diagonal(d2,np.inf)
        cand=[]
        for a in range(narm):
            valid_b=np.where(owner!=owner[a])[0]
            if not len(valid_b): continue
            # A few nearest second arms are enough to avoid a bad seed choice while
            # remaining tiny (72 points).  This is local search, not enumeration.
            nb=valid_b[np.argsort(d2[a,valid_b])[:6]]
            for b in nb:
                valid_c=np.where((owner!=owner[a])&(owner!=owner[b]))[0]
                if not len(valid_c): continue
                # Unwrap b/c around seed a and choose c minimizing actual 3-point spread.
                pb=A[a]; db=af[b]-af[a]; db-=np.rint(db); pb2=pb+db@C
                dc0=af[valid_c]-af[a]; dc0-=np.rint(dc0); pc=pb+dc0@C
                cen=(pb+pb2+pc)/3.0
                var=(np.sum((pb-cen)**2,axis=1)+np.sum((pb2-cen)**2,axis=1)+np.sum((pc-cen)**2,axis=1))/3.0
                q=int(np.argmin(var)); c=int(valid_c[q]); vv=float(var[q])
                # shifts relative to raw Cartesian arm positions, for differentiable loss.
                shb=-np.rint(af[b]-af[a]).astype(int); shc=-np.rint(af[c]-af[a]).astype(int)
                cand.append((vv,(int(a),int(b),int(c)),((0,0,0),tuple(map(int,shb)),tuple(map(int,shc)))))
        cand.sort(key=lambda x:x[0])
        used=np.zeros(narm,bool); out=[]
        for vv,vs,sh in cand:
            if used[list(vs)].any(): continue
            used[list(vs)]=True
            out.append({'arms':vs,'owners':tuple(int(owner[v]) for v in vs),'shifts':sh,'rms':float(np.sqrt(max(vv,0.0)))})
            if len(out)==2*nti: break
        rms=float(np.sqrt(np.mean([g['rms']**2 for g in out]))) if out else float('inf')
        return out,{'matched_o':int(len(out)),'unused_arms':int(narm-3*len(out)),'coincidence_rms_A':rms,
                    'local_candidates':int(len(cand))}


    def _sinkhorn_o_assignment(self, arms, cell, centers_frac, tau_A2=0.30, niter=30, owner_penalty=2.0):
        """Balanced detached 72x24 arm->O assignment.

        Rows carry unit mass (each TiO6 arm is used once), columns carry mass
        three (each latent O receives three arms).  A small owner-congestion
        outer correction discourages two arms from the same Ti feeding one O.
        This is an assignment operation, not part of the autograd graph.
        """
        with torch.no_grad():
            A=arms.reshape(-1,3); C=cell; invC=torch.linalg.inv(C); af=A@invC
            cf=centers_frac.to(dtype=A.dtype,device=A.device)%1.0
            df=af[:,None,:]-cf[None,:,:]; df=df-torch.round(df)
            dc=torch.einsum('aoi,ij->aoj',df,C); d2=(dc*dc).sum(-1)
            tau=max(float(tau_A2),1.0e-4); logits=-d2/tau
            nti=A.shape[0]//6; no=cf.shape[0]
            P=None
            for outer in range(4):
                K=torch.exp((logits-logits.max()).clamp(min=-60.0)) + 1.0e-12
                # Target row sum 1, column sum 3.  Alternating scaling is tiny
                # here (72x24) and much cheaper than one geometry evaluation.
                for _ in range(max(5,int(niter))):
                    K=K/K.sum(1,keepdim=True).clamp_min(1e-12)
                    K=K*(3.0/K.sum(0,keepdim=True).clamp_min(1e-12))
                P=K/K.sum(1,keepdim=True).clamp_min(1e-12)
                P=P*(3.0/P.sum(0,keepdim=True).clamp_min(1e-12))
                # Congestion: sum of the six arms of one Ti at one O should <=1.
                own=P.reshape(nti,6,no).sum(1)
                if outer<3:
                    pen=torch.relu(own-1.0)
                    logits=logits-float(owner_penalty)*pen[:,None,:].expand(nti,6,no).reshape_as(logits)
            own=P.reshape(nti,6,no).sum(1)
            entropy=-(P.clamp_min(1e-12)*torch.log(P.clamp_min(1e-12))).sum()/P.shape[0]
            return P.detach(), {'entropy':float(entropy.cpu()),'maxP':float(P.max(1).values.mean().cpu()),
                                'oload_std':float(P.sum(0).std().cpu()),'owner_dup':float(torch.relu(own-1.0).mean().cpu())}

    def _centers_from_assignment(self, arms, cell, P, centers_ref_frac):
        """Analytic periodic weighted O centroids for a fixed assignment P."""
        A=arms.reshape(-1,3); invC=torch.linalg.inv(cell); af=A@invC
        ref=centers_ref_frac.to(dtype=A.dtype,device=A.device)
        df=af[:,None,:]-ref[None,:,:]; df=df-torch.round(df).detach()
        unwrapped=ref[None,:,:]+df
        den=P.sum(0).clamp_min(1e-8)
        cf=torch.einsum('ao,aoi->oi',P,unwrapped)/den[:,None]
        return cf, cf@cell

    def _fixed_assignment_energy(self,t,raw,rot,stretch,P,centers_ref_frac,sigma_A):
        """Fuzzy-port Stage-B energy.

        Three virtual oxygen ports only need to enter the same small capture volume.
        Inside that volume the coincidence force rapidly saturates, avoiding an
        artificial requirement that three mathematical port centres become identical.
        """
        Ln,Nn,G=self.naturalness_loss(t,raw,details=True); cell=G[2][0]; tifrac=G[3][0]
        arms=self._rigid_o_arms_torch(tifrac,cell,rot,stretch)
        cf,cc=self._centers_from_assignment(arms,cell,P,centers_ref_frac)
        A=arms.reshape(-1,3); invC=torch.linalg.inv(cell); af=A@invC
        df=af[:,None,:]-cf[None,:,:]; df=df-torch.round(df).detach()
        dc=torch.einsum('aoi,ij->aoj',df,cell); d=torch.sqrt((dc*dc).sum(-1).clamp_min(1e-12))
        sig=max(float(sigma_A),0.03); width=max(0.20*sig,0.015)
        # Smooth one-sided capture-volume penalty: almost no force once a port is
        # comfortably within sigma of the latent shared-O centroid.
        outside=torch.nn.functional.softplus((d-sig)/width)*width
        Eshare=(P*outside.pow(2)).sum()/P.sum().clamp_min(1.0)
        Estretch=0.0025*torch.tanh(stretch).pow(2).mean()
        E=Eshare+0.12*Ln.mean()+Estretch
        return E,Eshare,Nn,G,arms,cf

    @staticmethod
    def _fire_group(q,v,f,dt,alpha,npos,dtmax):
        """One FIRE update for one generalized-coordinate block."""
        with torch.no_grad():
            fn=torch.linalg.norm(f); vn=torch.linalg.norm(v)
            power=torch.sum(v*f)
            if power>0:
                npos+=1
                if fn>1e-12 and vn>1e-12:
                    v.mul_(1.0-alpha).add_(f*(alpha*vn/fn))
                if npos>5:
                    dt=min(dt*1.10,dtmax); alpha*=0.99
            else:
                v.zero_(); dt=max(dt*0.5,1e-5); alpha=0.10; npos=0
            v.add_(f,alpha=dt)
            q.add_(v,alpha=dt)
        return dt,alpha,npos,float(fn.detach().cpu())

    def _hard_balanced_o_assignment(self, P):
        """Exact 72->24 assignment: one arm each, three arms/O, <=1 arm from each Ti/O.

        This is a tiny min-cost flow (1728 arm/O choices), used only after the
        soft topology has hardened.  It is not part of the gradient graph.
        """
        import networkx as nx
        Q=np.asarray(P.detach().cpu(),float); narm,no=Q.shape; nti=narm//6
        G=nx.DiGraph(); src='s'; snk='t'
        G.add_node(src,demand=-narm); G.add_node(snk,demand=narm)
        for a in range(narm):
            an=f'a{a}'; G.add_node(an,demand=0); G.add_edge(src,an,capacity=1,weight=0)
            ti=a//6
            for o in range(no):
                x=f'x{ti}_{o}'
                if x not in G: G.add_node(x,demand=0)
                cost=int(round(-1.0e5*math.log(max(Q[a,o],1e-12))))
                G.add_edge(an,x,capacity=1,weight=cost)
        for ti in range(nti):
            for o in range(no):
                x=f'x{ti}_{o}'; on=f'o{o}'
                if on not in G: G.add_node(on,demand=0)
                G.add_edge(x,on,capacity=1,weight=0)
        for o in range(no): G.add_edge(f'o{o}',snk,capacity=3,weight=0)
        flow=nx.min_cost_flow(G)
        groups=[[] for _ in range(no)]
        for a in range(narm):
            an=f'a{a}'
            for x,v in flow[an].items():
                if v:
                    o=int(x.split('_')[1]); groups[o].append(a); break
        ok=(all(len(g)==3 for g in groups) and all(len({a//6 for a in g})==3 for g in groups))
        return groups,ok

    def _joint_force_assignment(self,t,raw0,rot0,total_steps=60,fire_block=5,raw_dt=0.020,rot_dt=0.045,
                                tau_start=0.45,tau_end=0.06,sinkhorn_iters=30,token_tag='?',sigma_A=0.20):
        """Optimize crystal/lattice + rigid-angular TiO6 rotations + radial stretches.

        Oxygen is represented only by fuzzy virtual ports. Assignment is 72 ports to
        24 latent O volumes (3 ports/O); the physical O is instantiated afterwards at
        the centroid of its three assigned ports.
        """
        raw=raw0.detach().clone().requires_grad_(True); rot=rot0.detach().clone().requires_grad_(True)
        stretch=torch.zeros((rot.shape[0],6),dtype=rot.dtype,device=rot.device)  # rigid Ti-O lengths: NOT an optimization variable
        vr=torch.zeros_like(raw); vo=torch.zeros_like(rot)
        dtr,aro,npr=float(raw_dt),0.10,0; dto,ao,npo=float(rot_dt),0.10,0
        ncycle=max(1,int(math.ceil(total_steps/max(1,fire_block)))); trace=[]; tstart=time.time(); centers_ref=None; lastP=None; stable=0
        for cyc in range(ncycle):
            # Gaussian assignment width corresponding to the requested port volume.
            tau=max(2.0*float(sigma_A)**2,1e-4)
            with torch.no_grad():
                _,_,G0=self.naturalness_loss(t,raw,details=True); c0=G0[2][0]; f0=G0[3][0]
                a0=self._rigid_o_arms_torch(f0,c0,rot,stretch)
                if centers_ref is None:
                    groups,_=self._nearest_arm_matching(a0.cpu().numpy(),c0.cpu().numpy()); inv=np.linalg.inv(c0.cpu().numpy()); cen=[]; flat=a0.cpu().numpy().reshape(-1,3)
                    for g in groups[:24]:
                        pts=[flat[ai]+np.asarray(sh,float)@c0.cpu().numpy() for ai,sh in zip(g['arms'],g['shifts'])]; cen.append((np.mean(pts,axis=0)@inv)%1.0)
                    if len(cen)<24: cen.extend((f0.cpu().numpy()[i%len(f0)] for i in range(24-len(cen))))
                    centers_ref=torch.tensor(np.asarray(cen[:24]),dtype=a0.dtype,device=a0.device)
                P,adiag=self._sinkhorn_o_assignment(a0,c0,centers_ref,tau_A2=tau,niter=sinkhorn_iters)
                centers_ref=self._centers_from_assignment(a0,c0,P,centers_ref)[0].detach()%1.0; lastP=P.detach()
            last=None
            for kk in range(max(1,int(fire_block))):
                for q in (raw,rot): q.grad=None
                E,Es,Nn,G,arms,cf=self._fixed_assignment_energy(t,raw,rot,stretch,P,centers_ref,sigma_A); E.backward()
                fr=-raw.grad.detach(); fo=-rot.grad.detach()
                for f,lim in ((fr,100.),(fo,100.)):
                    n=torch.linalg.norm(f)
                    if n>lim: f.mul_(lim/n)
                dtr,aro,npr,fnr=self._fire_group(raw,vr,fr,dtr,aro,npr,max(float(raw_dt)*4,raw_dt))
                dto,ao,npo,fno=self._fire_group(rot,vo,fo,dto,ao,npo,max(float(rot_dt)*4,rot_dt))
                last=(float(E.detach().cpu()),float(Es.detach().cpu()),fnr,fno,0.0)
            with torch.no_grad():
                _,Nd,Gd=self.naturalness_loss(t,raw,details=True); cd=Gd[2][0]; fd=Gd[3][0]; ad=self._rigid_o_arms_torch(fd,cd,rot,stretch)
                gg,dd=self._nearest_arm_matching(ad.cpu().numpy(),cd.cpu().numpy()); cnt={r:int(sum(g['rms']<=r for g in gg)) for r in (0.10,0.15,0.20,0.25,0.30,0.40)}
                rec={'cycle':cyc+1,'sigma_A':float(sigma_A),'merge_rms_A':float(dd['coincidence_rms_A']),'assign_entropy':adiag['entropy'],'assign_maxP':adiag['maxP'],'assign_owner_dup':adiag['owner_dup'],'capture_counts':cnt,'share_E':last[1] if last else float('nan'),'volume_per_ti_A3':float(Nd['volume_per_ti_A3'][0].cpu())}; trace.append(rec)
            if adiag['maxP']>=0.95 and adiag['entropy']<=0.12 and adiag['owner_dup']<=0.02: stable+=1
            else: stable=0
            if stable>=2: break
        return raw.detach(),rot.detach(),stretch.detach(),trace,lastP,centers_ref.detach()

    def _solve_shared_o_sites(self, frac, cell, arm_groups, arms_cart, steps=70, lr=0.045, dir_weight=0.65):
        """v121: turn each hard 3-arm assignment into one chemically placed O.

        The centroid used in v118 can be very far from one or more assigned Ti.
        Here every O is solved locally against its three assigned Ti centers.  The
        objective asks for three Ti-O distances near the learned r0 while retaining
        the three inherited rigid-octahedron arm directions.  All 24 O problems are
        independent, so they are optimized as one small vectorized batch.
        """
        cn=np.asarray(cell,float); inv=np.linalg.inv(cn)
        tf=np.asarray(frac,float)%1.0; tc=tf@cn
        flat=np.asarray(arms_cart,float).reshape(-1,3)
        r0=float(self.model.pair(self.parent,self.child).mu)
        centers=[]; ti_unwrap=[]; dirs=[]
        for g in arm_groups:
            a0=int(g[0]); p0=flat[a0]; pts=[p0]
            for a in g[1:]:
                df=(flat[int(a)]-p0)@inv; df-=np.round(df); pts.append(p0+df@cn)
            cen=np.mean(pts,axis=0); centers.append(cen)
            tus=[]; dus=[]
            for a in g:
                a=int(a); ti=a//6
                # Put this Ti image nearest the current shared-O centroid.
                df=tf[ti]-(cen@inv); df-=np.round(df); tip=cen+df@cn
                tus.append(tip)
                # Inherited arm direction, minimum-imaged from its Ti owner.
                da=(flat[a]-tc[ti])@inv; da-=np.round(da); dv=da@cn
                dv=dv/max(np.linalg.norm(dv),1e-8); dus.append(dv)
            ti_unwrap.append(tus); dirs.append(dus)
        O=torch.tensor(np.asarray(centers),dtype=torch.float32,device=self.device,requires_grad=True)
        T=torch.tensor(np.asarray(ti_unwrap),dtype=torch.float32,device=self.device)
        U=torch.tensor(np.asarray(dirs),dtype=torch.float32,device=self.device)
        opt=torch.optim.Adam([O],lr=float(lr))
        for _ in range(max(1,int(steps))):
            opt.zero_grad(set_to_none=True)
            v=O[:,None,:]-T; d=torch.linalg.norm(v,dim=-1).clamp_min(1e-6); u=v/d[...,None]
            # Smooth bond restoration plus directional memory.  Direction is a
            # preference, not a hard constraint: shared O must compromise among 3 Ti.
            eb=((d-r0)/max(0.16*r0,1e-6))**2
            edir=(1.0-(u*U).sum(-1))**2
            loss=(eb + float(dir_weight)*edir).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_([O],5.0); opt.step()
        with torch.no_grad():
            v=O[:,None,:]-T; d=torch.linalg.norm(v,dim=-1); u=v/d.clamp_min(1e-6)[...,None]
            bond_rms=torch.sqrt(torch.mean((d-r0)**2,dim=1))
            dir_loss=torch.mean((1.0-(u*U).sum(-1))**2,dim=1)
            resid=(bond_rms/max(r0,1e-6))**2 + float(dir_weight)*dir_loss
            of=((O@torch.tensor(inv,dtype=torch.float32,device=self.device))%1.0).cpu().numpy()
            diag={'local_o_resid_mean':float(resid.mean().cpu()),
                  'local_o_resid_max':float(resid.max().cpu()),
                  'local_o_bond_rms_mean_A':float(bond_rms.mean().cpu()),
                  'local_o_bond_rms_max_A':float(bond_rms.max().cpu()),
                  'local_o_dir_loss_mean':float(dir_loss.mean().cpu()),
                  'local_o_tio_min_A':float(d.min().cpu()),
                  'local_o_tio_mean_A':float(d.mean().cpu()),
                  'local_o_tio_max_A':float(d.max().cpu())}
        return of,diag


    def _polish_octahedral_oxygen(self, frac, cell, opos_frac, triplets, steps=120, lr=0.025, angle_weight=3.0):
        """Stage-B oxygen-only polish for a fixed Ti6/O3 incidence topology.

        Every oxygen remains attached to the same three Ti selected by the hard
        72-arm -> 24-O assignment.  Ti and lattice are frozen.  Oxygen positions
        are optimized jointly for Ti-O bond lengths and permutation-invariant
        octahedral O-Ti-O geometry (3 trans angles near 180 deg, 12 cis near 90 deg).
        This is intentionally a B-level repair: C is not involved.
        """
        tf=torch.as_tensor(np.asarray(frac,float)%1.0,dtype=torch.float32,device=self.device)
        cn=torch.as_tensor(np.asarray(cell,float),dtype=torch.float32,device=self.device)
        of=torch.tensor(np.asarray(opos_frac,float)%1.0,dtype=torch.float32,device=self.device,requires_grad=True)
        nti=len(tf); no=len(of); r0=float(self.model.pair(self.parent,self.child).mu)
        by_ti=[[] for _ in range(nti)]
        for oi,tr in enumerate(triplets):
            for ti in tr:
                ti=int(ti)
                if 0 <= ti < nti: by_ti[ti].append(int(oi))
        if no != 24 or any(len(x)!=6 for x in by_ti):
            return np.asarray(opos_frac,float)%1.0, {'octa_valid':False,'octa_angle_rms_deg':float('inf'),'octa_angle_q90_abs_deg':float('inf'),'octa_min_angle_deg':0.0,'octa_bond_rms_A':float('inf')}
        ti_o=torch.tensor([[ti,oi] for ti,ids in enumerate(by_ti) for oi in ids],dtype=torch.long,device=self.device)
        oct_ids=torch.tensor(by_ti,dtype=torch.long,device=self.device)
        iu=torch.triu_indices(6,6,offset=1,device=self.device)
        target_dot=torch.cat([torch.full((3,),-1.0,device=self.device),torch.zeros((12,),device=self.device)])
        opt=torch.optim.Adam([of],lr=float(lr))
        best=None; best_loss=float('inf')
        for _ in range(max(1,int(steps))):
            opt.zero_grad(set_to_none=True)
            # Minimum-image map is piecewise differentiable; detach the integer image.
            df=of[ti_o[:,1]]-tf[ti_o[:,0]]; df=df-torch.round(df).detach()
            vv=df@cn; dd=torch.linalg.norm(vv,dim=1).clamp_min(1e-6)
            bond=((dd-r0)/max(0.12*r0,1e-6)).pow(2).mean()
            odf=of[oct_ids]-tf[:,None,:]; odf=odf-torch.round(odf).detach()
            ov=odf@cn; ou=ov/torch.linalg.norm(ov,dim=-1,keepdim=True).clamp_min(1e-6)
            gram=torch.matmul(ou,ou.transpose(1,2)); dots=gram[:,iu[0],iu[1]]
            sdots,_=torch.sort(dots,dim=1)
            ang=((sdots-target_dot[None,:])/0.20).pow(2).mean()
            loss=bond+float(angle_weight)*ang
            loss.backward(); torch.nn.utils.clip_grad_norm_([of],5.0); opt.step()
            with torch.no_grad():
                if float(loss)<best_loss and torch.isfinite(loss):
                    best_loss=float(loss); best=of.detach().clone()
        if best is not None: of=best
        with torch.no_grad():
            df=of[ti_o[:,1]]-tf[ti_o[:,0]]; df=df-torch.round(df)
            vv=df@cn; dd=torch.linalg.norm(vv,dim=1)
            odf=of[oct_ids]-tf[:,None,:]; odf=odf-torch.round(odf)
            ov=odf@cn; ou=ov/torch.linalg.norm(ov,dim=-1,keepdim=True).clamp_min(1e-6)
            dots=torch.matmul(ou,ou.transpose(1,2))[:,iu[0],iu[1]].clamp(-1,1)
            angles=torch.rad2deg(torch.acos(dots)); sang,_=torch.sort(angles,dim=1)
            # 12 cis are the smaller angles, 3 trans the larger angles.
            target=torch.cat([torch.full((12,),90.0,device=self.device),torch.full((3,),180.0,device=self.device)])
            err=(sang-target[None,:]).abs(); rms=torch.sqrt((err**2).mean())
            diag={'octa_valid':True,'octa_angle_rms_deg':float(rms.cpu()),
                  'octa_angle_q90_abs_deg':float(torch.quantile(err.flatten(),0.90).cpu()),
                  'octa_min_angle_deg':float(angles.min().cpu()),
                  'octa_max_angle_deg':float(angles.max().cpu()),
                  'octa_bond_rms_A':float(torch.sqrt(((dd-r0)**2).mean()).cpu()),
                  'octa_tio_min_A':float(dd.min().cpu()),'octa_tio_mean_A':float(dd.mean().cpu()),'octa_tio_max_A':float(dd.max().cpu())}
            out=(of%1.0).detach().cpu().numpy()
        return out,diag



    def _lock_rigid_triplets(self,t,raw,rot,stretch,groups,centers_ref,steps=240):
        """After fuzzy topology discovery, minimize exact three-port coincidence.

        STRICT RIGID-TiO6 TEST: no Ti-O stretching and no angular deformation.
        Only crystallographic Ti/framework + lattice raw variables and one SO(3)
        rotation per TiO6 are optimized.  The 72->24 OTi3 assignment is frozen.
        """
        narm=sum(len(g) for g in groups); no=len(groups)
        P=torch.zeros((narm,no),dtype=raw.dtype,device=raw.device)
        for oi,g in enumerate(groups):
            for aa in g: P[int(aa),oi]=1.0
        r=raw.detach().clone().requires_grad_(True)
        q=rot.detach().clone().requires_grad_(True)
        st=torch.zeros_like(stretch)  # hard-fixed radial geometry
        vr=torch.zeros_like(r); vq=torch.zeros_like(q)
        dtr,ar,npr=0.010,0.10,0; dtq,aq,npq=0.020,0.10,0
        cref=centers_ref.detach().clone(); trajectory=[]; best=None; stale=0
        report_every=max(10,int(steps)//8)
        for it in range(max(1,int(steps))):
            for x in (r,q): x.grad=None
            Ln,Nn,G=self.naturalness_loss(t,r,details=True); cell=G[2][0]; tifrac=G[3][0]
            arms=self._rigid_o_arms_torch(tifrac,cell,q,st)
            cf,cc=self._centers_from_assignment(arms,cell,P,cref)
            A=arms.reshape(-1,3); invC=torch.linalg.inv(cell); af=A@invC
            df=af[:,None,:]-cf[None,:,:]; df=df-torch.round(df).detach()
            dc=torch.einsum('aoi,ij->aoj',df,cell)
            # No fuzzy volume here: exact variance of each assigned 3-port cluster.
            Eshare=(P*(dc*dc).sum(-1)).sum()/P.sum().clamp_min(1.0)
            # Very weak framework regularizer prevents pathological cell/Ti drift;
            # it does not alter the rigid TiO6 internal geometry.
            E=Eshare + 0.01*Ln.mean()
            E.backward()
            fr=-r.grad.detach(); fq=-q.grad.detach()
            for f,lim in ((fr,120.),(fq,120.)):
                nn=torch.linalg.norm(f)
                if nn>lim: f.mul_(lim/nn)
            dtr,ar,npr,_=self._fire_group(r,vr,fr,dtr,ar,npr,0.050)
            dtq,aq,npq,_=self._fire_group(q,vq,fq,dtq,aq,npq,0.080)
            es=float(Eshare.detach().cpu())
            if best is None or es < best-1e-7: best=es; stale=0
            else: stale+=1
            if (it+1)%report_every==0 or it==steps-1:
                trajectory.append({'lock_step':int(it+1),'share_mse_A2':es})
            if stale>=80: break
        with torch.no_grad():
            _,_,G=self.naturalness_loss(t,r,details=True); cell=G[2][0]; frac=G[3][0]
            arms=self._rigid_o_arms_torch(frac,cell,q,st)
            cref,_=self._centers_from_assignment(arms,cell,P,cref); cref=cref.detach()%1.0
            cn=cell.detach().cpu().numpy(); fn=frac.detach().cpu().numpy()%1.; flat=arms.detach().cpu().numpy().reshape(-1,3); inv=np.linalg.inv(cn)
            opos=[]; rms=[]; maxspread=[]; trip=[]
            for g in groups:
                p0=flat[g[0]]; pts=[p0]
                for aa in g[1:]:
                    df=(flat[aa]-p0)@inv; df-=np.round(df); pts.append(p0+df@cn)
                pts=np.asarray(pts); cen=pts.mean(0); opos.append((cen@inv)%1.0); trip.append(tuple(int(aa)//6 for aa in g))
                rms.append(float(np.sqrt(np.mean(np.sum((pts-cen)**2,axis=1)))))
                maxspread.append(float(max(np.linalg.norm(pts[i]-pts[j]) for i in range(3) for j in range(i))))
            diag=self._fuzzy_final_geometry_diag(fn,cn,np.asarray(opos),trip,flat.reshape(len(fn),6,3))
            final={'sigma_A':0.0,'phase':'rigid_direct_lock','lock_steps_done':int(it+1),'port_rms_A':float(np.mean(rms)),
                   'port_max_pair_A':float(np.max(maxspread)),'stage_b_cell':cn.copy(),'stage_b_frac':fn.copy(),
                   'stage_b_port_frac':(flat@inv)%1.0,'o_growth_positions':np.asarray(opos,float),'o_growth_triplets':trip,**diag}
        return r.detach(),q.detach(),st.detach(),cref.detach(),trajectory,final

    def oxygen_growth(self,t,row):
        """Broad fuzzy topology discovery, then rigid-TiO6 direct coincidence lock."""
        raw0=row.get('raw_tensor')
        if raw0 is None: return {'o_growth_n':0,'o_growth_positions':np.empty((0,3)),'o_growth_triplets':[],'anneal_trajectory':[]}
        nti=len(row['frac']); search_sigma=float(getattr(self,'oxygen_search_sigma_A',0.30)); total=max(1,int(getattr(self,'octahedron_rotate_steps',60))); block=max(1,int(getattr(self,'oxygen_fire_block',5))); sink=int(getattr(self,'oxygen_sinkhorn_iters',30)); t0=time.perf_counter()
        rv=torch.zeros((nti,3),device=self.device)
        rb,rv,stretch,trace,P,cref=self._joint_force_assignment(t,raw0,rv,total_steps=total,fire_block=block,raw_dt=float(getattr(self,'oxygen_stageb_lr',0.020)),rot_dt=float(getattr(self,'octahedron_rotate_lr',0.045)),sinkhorn_iters=sink,token_tag=str(getattr(self,'progress_tag','token=?')),sigma_A=search_sigma)
        with torch.no_grad():
            _,_,G=self.naturalness_loss(t,rb,details=True); cell=G[2][0]; frac=G[3][0]; arms_t=self._rigid_o_arms_torch(frac,cell,rv,stretch); Pf,ad=self._sinkhorn_o_assignment(arms_t,cell,cref,tau_A2=max(2*search_sigma*search_sigma,1e-4),niter=sink)
        groups,ok=self._hard_balanced_o_assignment(Pf)
        # Evaluate the broad-search state before annealing.
        cn=cell.detach().cpu().numpy(); fn=frac.detach().cpu().numpy()%1.; flat=arms_t.detach().cpu().numpy().reshape(-1,3); inv=np.linalg.inv(cn); opos=[]; trip=[]; rms=[]; mx=[]
        for g in groups:
            p0=flat[g[0]]; pts=[p0]
            for a in g[1:]:
                df=(flat[a]-p0)@inv; df-=np.round(df); pts.append(p0+df@cn)
            pts=np.asarray(pts); cen=pts.mean(0); opos.append((cen@inv)%1.0); trip.append(tuple(int(a)//6 for a in g)); rms.append(float(np.sqrt(np.mean(np.sum((pts-cen)**2,axis=1))))); mx.append(float(max(np.linalg.norm(pts[i]-pts[j]) for i in range(3) for j in range(i))))
        search_diag=self._fuzzy_final_geometry_diag(fn,cn,np.asarray(opos),trip,arms_t.detach().cpu().numpy())
        trajectory=[{'sigma_A':search_sigma,'port_rms_A':float(np.mean(rms)),'port_max_pair_A':float(np.max(mx)),'stage_b_cell':cn.copy(),'stage_b_frac':fn.copy(),'stage_b_port_frac':(flat@inv)%1.0,'o_growth_positions':np.asarray(opos,float),'o_growth_triplets':trip,**search_diag}]
        rb,rv,stretch,cref,lock_trace,locked=self._lock_rigid_triplets(t,rb,rv,stretch,groups,cref,int(getattr(self,'oxygen_lock_steps',240))); trajectory.append(locked)
        fin=trajectory[-1]; incidence=np.zeros(nti,float)
        for tr in fin['o_growth_triplets']:
            for ti in tr: incidence[int(ti)]+=1
        inc_loss=float(np.mean(((incidence-6.0)/6.0)**2)) if nti else 9.0
        return {'o_growth_n':len(fin['o_growth_positions']),'o_growth_positions':fin['o_growth_positions'],'o_growth_triplets':fin['o_growth_triplets'],'o_growth_arm_groups':[tuple(g) for g in groups],
                'o_growth_incidence_loss':inc_loss,'o_match_matched':24 if ok else 0,'o_match_unused_arms':0 if ok else 72-3*len(groups),'o_match_rms_A':fin['port_rms_A'],'o_match_all24_rms_A':fin['port_rms_A'],
                'port_rms_A':fin['port_rms_A'],'port_max_pair_A':fin['port_max_pair_A'],'port_sigma_A':fin['sigma_A'],'assignment_entropy':ad['entropy'],'assignment_maxP':ad['maxP'],
                'stage_b_raw':rb,'stage_b_cell':fin['stage_b_cell'],'stage_b_frac':fin['stage_b_frac'],'stage_b_port_frac':fin['stage_b_port_frac'],
                'stage_b_profile':{'total_s':time.perf_counter()-t0,'search_trace':trace},'anneal_trajectory':trajectory,**{k:v for k,v in fin.items() if k.startswith('final_')}}

    def _fuzzy_final_geometry_diag(self,frac,cell,opos,trip,arms):
        """Measure damage caused by replacing three fuzzy ports by their centroid O."""
        tf=np.asarray(frac,float)%1.; of=np.asarray(opos,float)%1.; C=np.asarray(cell,float); nti=len(tf)
        # Target pair-angle matrix from the fixed learned TiO6 skeleton.
        base=self.oxygen_base_vectors.detach().cpu().numpy(); bu=base/np.linalg.norm(base,axis=1,keepdims=True); target=np.degrees(np.arccos(np.clip(bu@bu.T,-1,1)))
        angle_err=[]; minang=180.; bond_err=[]; tio=[]; oo_min=float('inf')
        rtarget=np.linalg.norm(np.asarray(arms).reshape(nti,6,3)-(tf@C)[:,None,:],axis=2)
        # map each physical O to each Ti owner; compare sorted local angular spectrum to target spectrum (rotation/permutation invariant)
        target_angles=np.sort(target[np.triu_indices(6,1)])
        for ti in range(nti):
            ids=[oi for oi,tr in enumerate(trip) if ti in tr]
            if len(ids)!=6: continue
            vec=[]
            for oi in ids:
                df=of[oi]-tf[ti]; df-=np.round(df); v=df@C; vec.append(v); tio.append(np.linalg.norm(v))
            vec=np.asarray(vec); d=np.linalg.norm(vec,axis=1); u=vec/np.maximum(d[:,None],1e-12); ang=np.degrees(np.arccos(np.clip(u@u.T,-1,1))); aa=np.sort(ang[np.triu_indices(6,1)]); angle_err.extend((aa-target_angles).tolist()); minang=min(minang,float(aa.min()))
            bond_err.extend((np.sort(d)-np.sort(rtarget[ti])).tolist())
        oc=of@C
        for i in range(len(of)):
            for j in range(i):
                df=of[i]-of[j]; df-=np.round(df); oo_min=min(oo_min,float(np.linalg.norm(df@C)))
        ae=np.asarray(angle_err,float); be=np.asarray(bond_err,float); tio=np.asarray(tio,float)
        return {'final_angle_rms_deg':float(np.sqrt(np.mean(ae*ae))) if len(ae) else float('inf'),'final_angle_max_abs_deg':float(np.max(np.abs(ae))) if len(ae) else float('inf'),'final_min_oti_angle_deg':float(minang),'final_bond_rms_A':float(np.sqrt(np.mean(be*be))) if len(be) else float('inf'),'final_tio_min_A':float(tio.min()) if len(tio) else float('nan'),'final_tio_mean_A':float(tio.mean()) if len(tio) else float('nan'),'final_tio_max_A':float(tio.max()) if len(tio) else float('nan'),'final_oo_min_A':float(oo_min)}

    def _rigid_o_arms_torch(self, tifrac, cell, rotvec, stretch=None):
        """Fixed-angular learned TiO6 ports with six bounded radial stretches per Ti.

        The angular skeleton is the learner's canonical TiO6 geometry. Rotation is
        free; each arm may stretch by +/-12%, but no O-Ti-O angular DOF exists.
        These are virtual oxygen-port centres, not physical oxygen atoms.
        """
        ti=tifrac @ cell
        R=self._axis_angle_rotation(rotvec)
        base=self.oxygen_base_vectors
        if stretch is None:
            scale=torch.ones((len(tifrac),6),dtype=base.dtype,device=base.device)
        else:
            scale=1.0 + float(getattr(self,'oxygen_stretch_fraction',0.12))*torch.tanh(stretch)
        vec=torch.einsum('aj,nkj->nak',base,R) * scale[...,None]
        return ti[:,None,:]+vec

    def _joint_loss(self,t,ti_raw,o_raw,triplets,arm_groups=None,details=False):
        """v123 B-SCF: force-balanced local chemistry with fixed Ti6/O3 topology.

        Stage B supplies an exact 72-arm -> 24-O topology.  Each arm keeps its
        original rigid-octahedron slot (0..5), so B-SCF never has to re-discover
        which O pairs are trans.  The 12 cis pairs are driven to 90 degrees by
        (u_i.u_j)^2 and the 3 inherited trans pairs to 180 degrees by
        (u_i.u_j+1)^2.  Bond, angle, collision and target-volume terms are all
        explicit and differentiable; SO3 is diagnostic/optional only.
        """
        Ln,Nn,G=self.naturalness_loss(t,ti_raw,details=True); abc,ang,cell,tifrac,free=G
        B=tifrac.shape[0]; ofrac=o_raw
        dfrac=ofrac[:,None,:,None,:]-tifrac[:,:,None,None,:]+self.shift[None,None,None,:,:]
        dcart=torch.einsum('bnoqi,bij->bnoqj',dfrac,cell); dd=torch.linalg.norm(dcart,dim=-1)
        dmin,idx=dd.min(dim=-1); qidx=idx[...,None,None].expand(-1,-1,-1,1,3); vec=torch.gather(dcart,3,qidx).squeeze(3)
        ch=self.model.pair(self.parent,self.child); r0=float(ch.mu)

        # Recover, for every Ti, its six physical O and the original rigid-arm slot.
        per_ti=[[] for _ in range(tifrac.shape[1])]; assigned=[]
        if arm_groups is not None and len(arm_groups)==len(triplets):
            for oi,(tr,grp) in enumerate(zip(triplets,arm_groups)):
                owner_to_slot={int(a)//6:int(a)%6 for a in grp}
                for ti in tr:
                    ti=int(ti); assigned.append(dmin[:,ti,oi]); per_ti[ti].append((int(oi),owner_to_slot[ti]))
        else:
            # Compatibility fallback; normally v121 always receives arm_groups.
            for oi,tr in enumerate(triplets):
                for ti in tr:
                    ti=int(ti); assigned.append(dmin[:,ti,oi]); per_ti[ti].append((int(oi),len(per_ti[ti])))
        adb=torch.stack(assigned,dim=-1) if assigned else torch.full((B,1),10.0,device=self.device)
        rel=adb/max(r0,1e-6)

        # v123: assigned Ti-O bonds remain a distinct force channel.  The topology is
        # already fixed by Stage B; C must first make every selected OTi3/TiO6 bond
        # chemically local before spending freedom on octahedral angles or volume.
        z=(rel-1.0)/0.055
        bond_restore=torch.nn.functional.smooth_l1_loss(z,torch.zeros_like(z),beta=1.0,reduction='none')
        short=torch.nn.functional.softplus((0.68-rel)/0.035)**2
        long=torch.nn.functional.softplus((rel-1.25)/0.035)**2
        bond_loss=(5.0*bond_restore+12.0*short+24.0*long).mean(-1)

        # v123 explicit 7th+ Ti-O exclusion.  The 72 selected topology bonds are
        # exempt; every other Ti-O pair is nonbonded and is pushed outside the
        # first coordination shell.  This prevents good six-bond statistics from
        # hiding TiO7/TiO8 environments.
        bonded_mask=torch.zeros((tifrac.shape[1],ofrac.shape[1]),dtype=torch.bool,device=self.device)
        for oi,tr in enumerate(triplets):
            for ti in tr: bonded_mask[int(ti),int(oi)]=True
        nonbond_d=dmin[:,~bonded_mask]
        nonbond_floor=1.16*r0
        tio_nonbond_loss=(torch.nn.functional.softplus((nonbond_floor-nonbond_d)/(0.055*r0))**2).mean(1) if nonbond_d.numel() else torch.zeros(B,device=self.device)

        so3_rows=[]; cis_rows=[]; trans_rows=[]
        base=self.oxygen_base_vectors
        baseu=base/torch.linalg.norm(base,dim=-1,keepdim=True).clamp_min(1e-6)
        for ti,entries in enumerate(per_ti):
            if len(entries)!=6:
                bad=torch.full((B,),100.0,device=self.device)
                so3_rows.append(bad); cis_rows.append(bad); trans_rows.append(bad); continue
            oids=[e[0] for e in entries]; slots=[e[1] for e in entries]
            vv=vec[:,ti,torch.as_tensor(oids,dtype=torch.long,device=self.device),:]
            uv=vv/torch.linalg.norm(vv,dim=-1,keepdim=True).clamp_min(1e-6)
            cis=[]; trans=[]
            for i in range(5):
                for j in range(i+1,6):
                    c=(uv[:,i,:]*uv[:,j,:]).sum(-1)
                    bref=float((baseu[slots[i]]*baseu[slots[j]]).sum().detach().cpu())
                    if bref < -0.75: trans.append((c+1.0)**2)
                    else: cis.append(c**2)
            # Exact inherited octahedron should give 12 cis and 3 trans pairs.
            cis_rows.append(torch.stack(cis,dim=1).mean(1) if cis else torch.full((B,),100.0,device=self.device))
            trans_rows.append(torch.stack(trans,dim=1).mean(1) if trans else torch.full((B,),100.0,device=self.device))
            M=self.tio6_so3_model
            fp=_so3_power_torch(vv[:,None,:,:],M['centers'],M['width'],M['lmax'])[:,0]
            dz=(fp-M['mu'][None,:])@M['evec']
            so3_rows.append((dz*dz/M['eval'][None,:]).mean(-1))
        so3_site=torch.stack(so3_rows,dim=1); cis_site=torch.stack(cis_rows,dim=1); trans_site=torch.stack(trans_rows,dim=1)
        # Tail-aware rigid chemistry: bad local cages must not hide in a mean.
        kt=max(1,min(3,so3_site.shape[1])); so3=torch.topk(so3_site,kt,dim=1).values.mean(1)
        cis_loss=cis_site.mean(1); trans_loss=trans_site.mean(1); oct_loss=cis_loss+trans_loss

        # O-centred SO3 on the three topology-assigned Ti.  Site identities are
        # fixed by Stage B, so no nearest-neighbour switching occurs.
        oti3_rows=[]
        for oi,tr in enumerate(triplets):
            tids=torch.as_tensor([int(q) for q in tr],dtype=torch.long,device=self.device)
            vv=-vec[:,tids,int(oi),:]  # O -> Ti nearest-image vectors
            M=self.oti3_so3_model
            fp=_so3_power_torch(vv[:,None,:,:],M['centers'],M['width'],M['lmax'])[:,0]
            dz=(fp-M['mu'][None,:])@M['evec']
            oti3_rows.append((dz*dz/M['eval'][None,:]).mean(-1))
        oti3_site=torch.stack(oti3_rows,dim=1) if oti3_rows else torch.full((B,1),1.0e6,device=self.device)
        ko=max(1,min(6,oti3_site.shape[1])); oti3_so3=torch.topk(oti3_site,ko,dim=1).values.mean(1)

        oo=[]
        for i in range(ofrac.shape[1]-1):
            df=ofrac[:,i+1:,:]-ofrac[:,i:i+1,:]
            dc=torch.einsum('boqi,bij->boqj',df[:,:,None,:]+self.shift[None,None,:,:],cell)
            dm=torch.linalg.norm(dc,dim=-1).min(dim=-1).values; oo.append(torch.nn.functional.softplus((self.oo_hard-dm)/0.08)**2)
        oo_loss=torch.cat(oo,dim=1).mean(1) if oo else torch.zeros(B,device=self.device)

        # v125: spherical Ti-framework shape is an independent force family.  It was
        # previously only a diagnostic and was therefore absent from the normalized
        # B-SCF mechanics despite being scientifically important.
        _,Dt,_=self.loss(t,ti_raw,details=True)
        spherical_loss=Dt['spherical_loss']
        # Remove the embedded Stage-A volume target from the C 'natural' channel.
        # Otherwise force normalization would secretly give the old target a second
        # independent vote and C would not be a genuine pressure release.
        natural_no_volume=(self.natural_density_weight*Nn['natural_density_loss']
                           +self.natural_vacuum_weight*Nn['natural_vacuum_loss']
                           +self.natural_radial_restore_weight*Nn['natural_radial_restore_loss']
                           +self.aspect_weight*Nn['aspect']+10.0*Nn['metric_penalty'])

        # Volume is a weak release regularizer, not a chemistry criterion.
        vol=Nn["volume_per_ti_A3"]; vtarget=max(float(self.volume_target_per_ti),1e-6)
        volume_loss=((vol-vtarget)/vtarget)**2
        # Scalar total is retained only for diagnostics/backward compatibility.
        # v123 B-SCF does NOT differentiate this weighted sum; joint_refine takes
        # gradients of the physical channels separately and normalizes them.
        total=(bond_loss + tio_nonbond_loss + oo_loss + cis_loss + trans_loss
               + so3 + oti3_so3 + 0.05*Ln + volume_loss)
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if details:
            return total,{'joint_bond_loss':bond_loss,'joint_tio_nonbond_loss':tio_nonbond_loss,
                'joint_oct_loss':oct_loss,'joint_cis90_loss':cis_loss,'joint_trans180_loss':trans_loss,
                'joint_oct_sites':cis_site+trans_site,'joint_tio6_so3':so3,'joint_tio6_so3_raw_sites':so3_site,'joint_tio6_so3_sites':so3_site,
                'joint_oti3_so3':oti3_so3,'joint_oti3_so3_raw_sites':oti3_site,'joint_oti3_so3_sites':oti3_site,
                'joint_coord_loss':oct_loss,'joint_oo_loss':oo_loss,'joint_volume_pressure':volume_loss,'joint_volume_per_ti':vol,'joint_min_tio':adb.min(1).values,'joint_max_tio':adb.max(1).values,
                'joint_mean_tio':adb.mean(1),'joint_min_nonbond_tio':nonbond_d.min(1).values if nonbond_d.numel() else torch.full((B,),99.0,device=self.device),
                'cell':cell,'tifrac':tifrac,'ofrac':ofrac,'natural':Nn,'spherical_loss':spherical_loss,
                'force_channels':{'bond':bond_loss,'tio_nonbond':tio_nonbond_loss,'oo':oo_loss,'oct':oct_loss,'tio6_so3':so3,'oti3_so3':oti3_so3,'natural':natural_no_volume,'spherical':spherical_loss,'volume':volume_loss}}
        return total

    def _topology_only_loss(self,t,ti_raw,o_raw,triplets,arm_groups=None,details=False):
        """v135r10 B-SCF objective: learned TiO6 + OTi3 only.

        No Stage-A naturalness, framework SOAP, volume target, collision, bond, or angle
        objective is evaluated here. Geometry is reconstructed only to obtain the current
        cell/Ti positions. Fixed Stage-B Ti6/O3 incidence defines the local environments.
        """
        _,_,cell,_,tifrac,_=self.geometry(t,ti_raw); B=tifrac.shape[0]; ofrac=o_raw
        dfrac=ofrac[:,None,:,None,:]-tifrac[:,:,None,None,:]+self.shift[None,None,None,:,:]
        dcart=torch.einsum('bnoqi,bij->bnoqj',dfrac,cell); dd=torch.linalg.norm(dcart,dim=-1)
        idx=dd.argmin(dim=-1); qidx=idx[...,None,None].expand(-1,-1,-1,1,3); vec=torch.gather(dcart,3,qidx).squeeze(3)
        dmin=dd.min(dim=-1).values
        per_ti=[[] for _ in range(tifrac.shape[1])]; assigned=[]
        for oi,tr in enumerate(triplets):
            grp=arm_groups[oi] if arm_groups is not None and oi<len(arm_groups) else None
            owner_to_slot={int(a)//6:int(a)%6 for a in grp} if grp is not None else {}
            for ti in tr:
                ti=int(ti); assigned.append(dmin[:,ti,oi]); per_ti[ti].append((oi,owner_to_slot.get(ti,len(per_ti[ti]))))
        adb=torch.stack(assigned,dim=-1)
        rows=[]
        for ti,entries in enumerate(per_ti):
            if len(entries)!=6: rows.append(torch.full((B,),1e9,device=self.device)); continue
            oids=[e[0] for e in entries]; vv=vec[:,ti,torch.as_tensor(oids,dtype=torch.long,device=self.device),:]
            M=self.tio6_so3_model; fp=_so3_power_torch(vv[:,None,:,:],M['centers'],M['width'],M['lmax'])[:,0]
            dz=(fp-M['mu'][None,:])@M['evec']; rows.append((dz*dz/M['eval'][None,:]).mean(-1))
        ts=torch.stack(rows,dim=1); tio6=torch.topk(ts,max(1,min(3,ts.shape[1])),dim=1).values.mean(1)
        orows=[]
        for oi,tr in enumerate(triplets):
            tids=torch.as_tensor([int(q) for q in tr],dtype=torch.long,device=self.device); vv=-vec[:,tids,int(oi),:]
            M=self.oti3_so3_model; fp=_so3_power_torch(vv[:,None,:,:],M['centers'],M['width'],M['lmax'])[:,0]
            dz=(fp-M['mu'][None,:])@M['evec']; orows.append((dz*dz/M['eval'][None,:]).mean(-1))
        os=torch.stack(orows,dim=1); oti3=torch.topk(os,max(1,min(6,os.shape[1])),dim=1).values.mean(1)
        # TiO6 has four times the authority of OTi3. Raw magnitude is diagnostic;
        # joint_refine still RMS-normalizes each channel gradient before combination.
        total=4.0*tio6+oti3
        if details:
            vol=torch.abs(torch.linalg.det(cell))/max(int(tifrac.shape[1]),1)
            return total,{'joint_tio6_so3':tio6,'joint_tio6_so3_sites':ts,'joint_oti3_so3':oti3,'joint_oti3_so3_sites':os,
                          'joint_min_tio':adb.min(1).values,'joint_mean_tio':adb.mean(1),'joint_max_tio':adb.max(1).values,
                          'joint_volume_per_ti':vol,'cell':cell,'tifrac':tifrac,'ofrac':ofrac,
                          'force_channels':{'tio6_so3':tio6,'oti3_so3':oti3}}
        return total

    def joint_refine(self,t,row,growth):
        """v123 force-balanced B-SCF.

        Topology is fixed (Ti6/O3).  Each physical energy family is differentiated
        separately.  Its generalized gradient is RMS-normalized before channels are
        combined, so descriptor scale cannot silently dominate the trajectory.  The
        magnitude pattern *within* each channel is preserved, hence badly stretched
        bonds/collisions still receive stronger forces than small errors.
        """
        need=int(round(len(row['frac'])*self.model.cn[self.parent]/self.model.cn[self.child]))
        if growth['o_growth_n']<need:return None
        # B-SCF must start from the *realized Stage-B framework*. oxygen_growth()
        # jointly moves the Ti/Wyckoff/lattice raw variables and constructs O positions
        # in that final B cell.  Starting from row['raw_tensor'] (the pre-B A framework)
        # would mix B oxygen coordinates with an A lattice/Ti geometry.
        ti_raw=growth.get('stage_b_raw')
        if ti_raw is None:
            ti_raw=row.get('raw_tensor')  # compatibility only for legacy growth records
        if ti_raw is None:return None
        x=ti_raw.detach().clone().to(self.device).requires_grad_(True)
        o=torch.tensor(growth['o_growth_positions'],dtype=torch.float32,device=self.device)[None,:,:].requires_grad_(True)

        # Fail loudly on any future A/B coordinate-frame regression.  Stage-B stores
        # both the optimized raw framework and its realized cell/Ti fractional positions.
        with torch.no_grad():
            _,_,cell_chk,_,ti_chk,_=self.geometry(t,x)
            if growth.get('stage_b_cell') is not None:
                ref_cell=torch.as_tensor(growth['stage_b_cell'],dtype=cell_chk.dtype,device=self.device)
                if not torch.allclose(cell_chk[0],ref_cell,rtol=2e-5,atol=2e-5):
                    raise RuntimeError('B-SCF initialization mismatch: stage_b_raw does not reproduce stage_b_cell')
            if growth.get('stage_b_frac') is not None:
                ref_ti=torch.as_tensor(growth['stage_b_frac'],dtype=ti_chk.dtype,device=self.device)
                # Fractional coordinates are periodic; compare wrapped displacement.
                df=ti_chk[0]-ref_ti
                df=df-torch.round(df)
                if float(df.abs().max().cpu())>2e-5:
                    raise RuntimeError('B-SCF initialization mismatch: stage_b_raw does not reproduce stage_b_frac')
        trip=growth['o_growth_triplets']; groups=growth.get('o_growth_arm_groups')
        nlat=len(t['spec']); r0=float(self.model.pair(self.parent,self.child).mu)

        # Relative channel authority after normalization.  These are intentionally
        # O(1): bond is first priority, while both SO3 environments and exclusions
        # have independent voices instead of depending on arbitrary raw loss scales.
        # v135r10: B-SCF optimizes ONLY the learned local topology environments.
        # TiO6 is deliberately dominant; OTi3 is secondary. Other geometry terms remain
        # diagnostics only and contribute no gradient in B-SCF.
        cw={'tio6_so3':4.00,'oti3_so3':1.00}

        def report(tag,force_rms=None):
            with torch.no_grad():
                L,D=self._topology_only_loss(t,x,o,trip,groups,details=True)
                extra='' if not force_rms else ' F['+' '.join(f'{k}={v:.2f}' for k,v in force_rms.items())+']'
                print(f"    [v125 C] {tag:<10s} TiO={float(D['joint_min_tio'][0]):.3f}/{float(D['joint_mean_tio'][0]):.3f}/{float(D['joint_max_tio'][0]):.3f}A TiO6SO3={float(D['joint_tio6_so3'][0]):.3f} OTi3SO3={float(D['joint_oti3_so3'][0]):.3f} V/Ti={float(D['joint_volume_per_ti'][0]):.2f}{extra}",flush=True)
                return L,D

        L0,D0=report('pre')
        mn0=float(D0['joint_min_tio'][0]); mx0=float(D0['joint_max_tio'][0])
        # Characterization mode: do not reject bad starting geometry.  The purpose of
        # this run is to measure how much B-SCF can repair it.  Only non-finite
        # arithmetic is treated as an execution error by the caller.
        soap_pre=float('nan')  # v135r10: framework SOAP is outside B-SCF objective

        def balanced_step(active_x,active_o,lr,mask_lattice=False):
            # Rebuild one graph, then differentiate each physical channel separately.
            L,D=self._topology_only_loss(t,x,o,trip,groups,details=True)
            channels=D['force_channels']; gx=torch.zeros_like(x) if active_x else None; go=torch.zeros_like(o) if active_o else None
            raw_rms={}
            keys=['tio6_so3','oti3_so3']
            for ki,k in enumerate(keys):
                targets=[]
                if active_x: targets.append(x)
                if active_o: targets.append(o)
                grads=torch.autograd.grad(channels[k].mean(),targets,retain_graph=(ki<len(keys)-1),allow_unused=True)
                pos=0; parts=[]
                if active_x:
                    g=grads[pos]; pos+=1
                    if g is None:g=torch.zeros_like(x)
                    if mask_lattice:g=g.clone(); g[:,:nlat]=0.0
                    parts.append(g.reshape(-1))
                if active_o:
                    g2=grads[pos] if pos<len(grads) else None
                    if g2 is None:g2=torch.zeros_like(o)
                    parts.append(g2.reshape(-1))
                flat=torch.cat(parts) if parts else torch.zeros(1,device=self.device)
                rms=torch.sqrt(torch.mean(flat*flat)+1e-16)
                raw_rms[k]=float(rms.detach().cpu())
                # v125: gradient normalization is conditioning only.  Absolute
                # violation gates channel authority, so a nearly satisfied family
                # cannot acquire unit force merely because its tiny gradient is normalized.
                if float(rms.detach().cpu())<=1e-7:
                    scale=0.0
                else:
                    val=max(float(channels[k].mean().detach().cpu()),0.0)
                    if k=='tio6_so3': ref=max(self.tio6_so3_model['q95'],1e-8); q=val/ref; act=q/(1.0+q)
                    elif k=='oti3_so3': ref=max(self.oti3_so3_model['q95'],1e-8); q=val/ref; act=q/(1.0+q)
                    elif k=='volume':
                        relv=abs(float(D['joint_volume_per_ti'][0].detach().cpu())/max(self.volume_target_per_ti,1e-8)-1.0); act=relv/(relv+0.20)
                    else:
                        act=val/(1.0+val)
                    scale=cw[k]*float(act)/rms.clamp_min(1e-8)
                pos=0
                if active_x:
                    g=grads[pos]; pos+=1
                    if g is None:g=torch.zeros_like(x)
                    if mask_lattice:g=g.clone(); g[:,:nlat]=0.0
                    gx=gx+scale*g
                if active_o:
                    g2=grads[pos] if pos<len(grads) else None
                    if g2 is None:g2=torch.zeros_like(o)
                    go=go+scale*g2
            with torch.no_grad():
                # Normalize the combined generalized force per variable family; this
                # fixes step scale without destroying competition between channels.
                if active_x:
                    xr=torch.sqrt(torch.mean(gx*gx)+1e-12); x.add_(-lr*gx/xr.clamp_min(1e-6))
                if active_o:
                    orr=torch.sqrt(torch.mean(go*go)+1e-12); o.add_(-lr*go/orr.clamp_min(1e-6))
            return raw_rms

        nsteps=max(3,int(self.joint_steps)); n_o=max(1,int(round(.40*nsteps))); n_ti=max(1,int(round(.35*nsteps))); n_all=max(1,nsteps-n_o-n_ti)
        lr=float(self.joint_lr); fr=None
        for step in range(n_o):
            fr=balanced_step(False,True,0.55*lr)
            if (step+1)%10==0: report(f'O {step+1}',fr)
        _,D_o=report('O-only',fr)
        for step in range(n_ti):
            fr=balanced_step(True,True,0.35*lr,mask_lattice=True)
            if (step+1)%10==0: report(f'Ti {step+1}',fr)
        _,D_ti=report('O+Ti',fr)
        for step in range(n_all):
            fr=balanced_step(True,True,0.20*lr,mask_lattice=False)
            if (step+1)%10==0: report(f'all {step+1}',fr)

        with torch.no_grad():
            L,D=self._topology_only_loss(t,x,o,trip,groups,details=True)
            mn=float(D['joint_min_tio'][0]); mx=float(D['joint_max_tio'][0]); mean=float(D['joint_mean_tio'][0])
            plausible=bool(np.isfinite(float(L[0])))
            report('final',fr)
            soap_post=float('nan')
            return {'joint_loss':float(L[0]),'stage_c_soap_pre':float('nan'),'stage_c_soap_post':float('nan'),'stage_c_soap_delta':float('nan'),
                    'stage_c_tio6_so3_pre':float(D0['joint_tio6_so3'][0]),'stage_c_tio6_so3_delta':float(D['joint_tio6_so3'][0]-D0['joint_tio6_so3'][0]),
                    'stage_c_oti3_so3_pre':float(D0['joint_oti3_so3'][0]),'stage_c_oti3_so3_delta':float(D['joint_oti3_so3'][0]-D0['joint_oti3_so3'][0]),
                    'stage_c_min_tio_pre':mn0,'stage_c_mean_tio_pre':float(D0['joint_mean_tio'][0]),'stage_c_max_tio_pre':mx0,'stage_c_volume_pre':float(D0['joint_volume_per_ti'][0]),
                    'joint_tio6_so3':float(D['joint_tio6_so3'][0]),'joint_tio6_so3_sites':D['joint_tio6_so3_sites'][0].cpu().numpy().tolist(),
                    'joint_oti3_so3':float(D['joint_oti3_so3'][0]),'joint_oti3_so3_sites':D['joint_oti3_so3_sites'][0].cpu().numpy().tolist(),
                    'joint_min_tio':mn,'joint_mean_tio':mean,'joint_max_tio':mx,'joint_plausible':plausible,'joint_volume_per_ti':float(D['joint_volume_per_ti'][0]),
                    'joint_cell':D['cell'][0].cpu().numpy(),'joint_ti_frac':D['tifrac'][0].cpu().numpy()%1.0,'joint_o_frac':D['ofrac'][0].cpu().numpy()%1.0,
                    'phase_o_cell':D_o['cell'][0].cpu().numpy(),'phase_o_ti_frac':D_o['tifrac'][0].cpu().numpy()%1.0,'phase_o_o_frac':D_o['ofrac'][0].cpu().numpy()%1.0,
                    'phase_ti_cell':D_ti['cell'][0].cpu().numpy(),'phase_ti_ti_frac':D_ti['tifrac'][0].cpu().numpy()%1.0,'phase_ti_o_frac':D_ti['ofrac'][0].cpu().numpy()%1.0}

    def environment_loss(self,t,raw,method,details=False):
        M=self.env_models[method]
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        delta=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vec=torch.einsum("bijnk,bkl->bijnl",delta,cell); dist=torch.linalg.norm(vec,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]; zero=torch.arange(27,device=self.device)==ZERO_SHIFT
        dist=dist.masked_fill(eye & zero[None,None,None,:],1e6)
        flatd=dist.reshape(B,N,N*27); flatv=vec.reshape(B,N,N*27,3)
        K=min(M["k"],N*27-1); nd,idx=torch.topk(flatd,K,dim=-1,largest=False)
        nv=torch.gather(flatv,2,idx[...,None].expand(-1,-1,-1,3))
        if method=="SKY": fp=_astronomy_power_torch(nv,M["centers"],M["width"],M["lmax"])
        elif method=="SOAP": fp=_soap_power_torch(nv,M["centers"],M["width"],M["lmax"])
        else: fp=_ace_power_torch(nv,M["centers"],M["width"],M["lmax"])
        z=(fp-M["med"][None,None,:])/M["scale"][None,None,:]
        site=torch.mean(z*z,dim=-1)
        tau=max(float(self.environment_tau),0.05)
        worst=tau*(torch.logsumexp(site/tau,dim=1)-math.log(max(N,1)))
        mind=nd[...,0].min(1).values
        collision=torch.nn.functional.softplus((self.hard-mind)/0.06)**2
        aspect=(torch.log(abc.max(1).values/abc.min(1).values.clamp_min(1e-6)))**2
        metric=torch.relu(-z2/0.02)**2
        total=worst+20.0*collision+0.03*aspect+10.0*metric
        if not details:return torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        vol=torch.det(cell).abs()/float(max(N,1))
        return total,{"site":site,"best":site.min(1).values,"median":site.median(1).values,
                      "q90":torch.quantile(site,.90,dim=1),"worst":site.max(1).values,
                      "minimum_titi_A":mind,"volume_per_ti_A3":vol,"metric_valid":z2>0},(abc,ang,cell,frac,free)

    def environment_shake(self,t,raw,method="SOAP",steps=60,lr=0.016,freeze_lattice=False):
        """Local descriptor relaxation.  v129 can freeze lattice during basin-proposal settling."""
        x=raw.detach().clone().requires_grad_(True); opt=torch.optim.Adam([x],lr=lr)
        tic=time.perf_counter(); nlat=len(t["spec"])
        for step in range(max(1,int(steps))):
            f=step/max(int(steps)-1,1); cur=lr*(0.12+0.88*0.5*(1+math.cos(math.pi*f)))
            for pg in opt.param_groups: pg['lr']=cur
            opt.zero_grad(set_to_none=True); L=self.environment_loss(t,x,method); L.sum().backward()
            if freeze_lattice and x.grad is not None: x.grad[:,:nlat]=0.0
            torch.nn.utils.clip_grad_norm_([x],5.0); opt.step()
        return x.detach(),time.perf_counter()-tic



    @staticmethod
    def _logit_np(u):
        u=np.clip(np.asarray(u,float),1e-4,1-1e-4)
        return np.log(u/(1-u))


    def _random_relocation_candidates(self,t,before,nproposal=8,rng=None):
        """Control: reset one movable Wyckoff generator to a random point in its legal free-coordinate cube."""
        rng=np.random.default_rng() if rng is None else rng; nlat=len(t['spec'])
        blocks=[]; cur=0
        for d in t['dofs']:
            if d>0: blocks.append((nlat+cur,d))
            cur+=d
        if not blocks:return []
        out=[]
        for _ in range(max(1,int(nproposal))):
            st,d=blocks[int(rng.integers(len(blocks)))]; u=rng.uniform(.02,.98,size=d)
            x=before.clone(); x[0,st:st+d]=torch.tensor(self._logit_np(u),dtype=x.dtype,device=x.device); out.append(x)
        return out

    def adaptive_mc_search(self,t,probe,nproposal=128,keep=6,settle_steps=12,settle_lr=0.016,rng=None):
        """v130 global search: batch random symmetry-legal orbit mutations, SOAP-rank once,
        then short fixed-lattice SOAP-settle only the best survivors as one GPU batch.
        The unmutated probe state is always retained as a control candidate.
        """
        tic=time.perf_counter(); rng=np.random.default_rng() if rng is None else rng
        props=self._random_relocation_candidates(t,probe,int(nproposal),rng)
        if not props:
            return probe.clone(),time.perf_counter()-tic,{'proposals':0,'valid':0,'kept':0,'best_raw_worst':float('inf'),'best_settle_worst':float('inf')}
        P=torch.cat(props,dim=0)
        with torch.no_grad():
            _,D,_=self.environment_loss(t,P,'SOAP',details=True)
            valid=D['metric_valid'] & torch.isfinite(D['worst']) & (D['minimum_titi_A']>=0.88*self.hard)
            ids=torch.where(valid)[0]
            nvalid=int(ids.numel())
            if nvalid==0:
                return probe.clone(),time.perf_counter()-tic,{'proposals':len(props),'valid':0,'kept':0,'best_raw_worst':float('inf'),'best_settle_worst':float('inf')}
            ids=ids[torch.argsort(D['worst'][ids])[:min(int(keep),nvalid)]]
            best_raw=float(D['worst'][ids[0]])
            K=P[ids].detach().clone()
        # One batched optimizer for all finalists; lattice is frozen during basin settling.
        Y,_=self.environment_shake(t,K,'SOAP',steps=int(settle_steps),lr=settle_lr,freeze_lattice=True)
        with torch.no_grad():
            _,Dy,_=self.environment_loss(t,Y,'SOAP',details=True)
            good=torch.isfinite(Dy['worst']) & (Dy['minimum_titi_A']>=0.88*self.hard)
            if not bool(good.any()):
                return probe.clone(),time.perf_counter()-tic,{'proposals':len(props),'valid':nvalid,'kept':len(ids),'best_raw_worst':best_raw,'best_settle_worst':float('inf')}
            score=Dy['worst'].clone(); score[~good]=float('inf'); j=int(torch.argmin(score))
            best=Y[j:j+1].detach().clone(); best_settle=float(score[j])
        return best,time.perf_counter()-tic,{'proposals':len(props),'valid':nvalid,'kept':len(ids),'best_raw_worst':best_raw,'best_settle_worst':best_settle}

    def optimize(self,t,raw,steps,lr,mode="chemistry"):
        x=raw.detach().clone().requires_grad_(True); opt=torch.optim.Adam([x],lr=lr)
        n=max(1,int(steps))
        for step in range(n):
            # Gentle cosine decay matters especially for the longer v101 ambient pre-relaxation.
            frac=step/max(n-1,1)
            cur_lr=lr*(0.15+0.85*0.5*(1.0+math.cos(math.pi*frac)))
            for pg in opt.param_groups: pg["lr"]=cur_lr
            opt.zero_grad(set_to_none=True)
            if mode=="ambient":
                L=self.ambient_loss(t,x)
            elif mode=="natural":
                L=self.naturalness_loss(t,x)
            elif mode=="overcondense":
                L=self.overcondense_loss(t,x)
            elif mode=="exact":
                _,D,_=self.loss(t,x,details=True)
                L=(self.so3_weight*D["so3_loss"] + 0.03*D["radial_loss"] + 0.01*D["angular_loss"]
                   + self.exclusion_weight*D["exclusion"] + self.aspect_weight*D["aspect"] + 10.0*D["metric_penalty"])
                L=torch.nan_to_num(L,nan=1e9,posinf=1e9,neginf=1e9)
            else:
                L=self.loss(t,x)
            L.sum().backward(); torch.nn.utils.clip_grad_norm_([x],10.0); opt.step()
        return x.detach()

    def _stage_a_batch(self,t,spg,batch_index,nstarts,keep):
        """Generate a small vectorized batch of independent Stage-A Ti entrances."""
        tic=time.perf_counter()
        self._live('stageA_batch_start',spg=int(spg),batch_index=int(batch_index),nstarts=int(nstarts),keep=int(keep))
        raw=self.initial_raw(t,nstarts=max(1,int(nstarts)))
        a1=self.optimize(t,raw,self.ambient_steps,self.ambient_lr,mode='natural')
        ps=max(0,int(getattr(self,'stage_a_polish_steps',80)))
        a=self.optimize(t,a1,ps,max(.008,.35*self.ambient_lr),mode='natural') if ps else a1
        with torch.no_grad():
            La,Na,_=self.naturalness_loss(t,a,details=True)
            vr=Na['volume_per_ti_A3']/max(self.volume_target_per_ti,1e-8)
            gate=(Na['metric_valid'] & torch.isfinite(La) & (Na['minimum_titi_A']>=.88*self.hard) &
                  (vr>=.45) & (vr<=2.20) & (Na['natural_density_loss']<=8.0))
            ids=torch.where(gate)[0]
            if len(ids): ids=ids[torch.argsort(La[ids])[:min(int(keep),len(ids))]]
        sec=float(time.perf_counter()-tic)
        self._live('stageA_batch_done',spg=int(spg),batch_index=int(batch_index),seconds=sec,retained=int(len(ids)))
        return a,ids,sec

    @staticmethod
    def _b_quality(cand):
        """Final Stage-B quality. Stage-A SOAP is deliberately not used to rank B outcomes."""
        exact=(int(cand.get('o_growth_n',0))==24 and int(cand.get('o_match_matched',0))==24 and
               int(cand.get('o_match_unused_arms',72))==0 and float(cand.get('o_growth_incidence_loss',9.0))<1e-10)
        arm=float(cand.get('o_match_all24_rms_A',cand.get('o_match_rms_A',float('inf'))))
        local=float(cand.get('o_local_resid_mean',9.0))
        if not np.isfinite(arm): arm=float('inf')
        if not np.isfinite(local): local=9.0
        # Lexicographic: exact chemistry first, then realized arm geometry, then local OTi3 residual.
        return bool(exact),arm,local

    def _local_soap_softmin(self, fp, M):
        """Smooth distance to a learned multi-basin local SOAP manifold."""
        dz=(fp[:,:,None,:]-M['proto'][None,None,:,:])/M['scale'][None,None,None,:]
        d2=(dz*dz).mean(-1)
        tau=max(float(M['tau']),1e-4)
        # log-mean-exp rather than log-sum-exp: equal prototype distances map back
        # to that same distance and prototype count cannot lower the energy.
        return -tau*(torch.logsumexp(-d2/tau,dim=-1)-math.log(max(int(M['proto'].shape[0]),1)))

    def _oxygen_field_energy(self, ti_frac, o_frac, cell, details=False):
        """v137 graph-free local SOAP SCF field.

        Every Ti independently observes its six nearest periodic O neighbours and
        every O independently observes its three nearest periodic Ti neighbours.
        No Ti--O ownership graph, triplet assignment, ideal octahedral target, or
        explicit coordination penalty is used.  Chemistry is the learned smooth
        multi-basin SOAP naturalness of those local observations.
        """
        B=o_frac.shape[0]; no=o_frac.shape[1]
        df=o_frac[:,None,:,None,:]-ti_frac[:,:,None,None,:]+self.shift[None,None,None,:,:]
        dc=torch.einsum('bnoqi,bij->bnoqj',df,cell)
        dd=torch.linalg.norm(dc,dim=-1)
        qi=dd.argmin(-1); gather=qi[...,None,None].expand(-1,-1,-1,1,3)
        vec=torch.gather(dc,3,gather).squeeze(3); dist=dd.min(-1).values

        # Ti observers: six nearest O.
        _,ix6=torch.topk(dist,k=self.tio6_soap_model['k'],dim=-1,largest=False)
        v6=torch.gather(vec,2,ix6[...,None].expand(-1,-1,-1,3))
        fp_ti=_soap_power_torch(v6,self.tio6_soap_model['centers'],self.tio6_soap_model['width'],self.tio6_soap_model['lmax'])
        ti_site=self._local_soap_softmin(fp_ti,self.tio6_soap_model)

        # O observers: three nearest Ti; reverse Ti->O vectors to O->Ti.
        dist_o=dist.transpose(1,2); vec_o=-vec.transpose(1,2)
        _,ix3=torch.topk(dist_o,k=self.oti3_soap_model['k'],dim=-1,largest=False)
        v3=torch.gather(vec_o,2,ix3[...,None].expand(-1,-1,-1,3))
        fp_o=_soap_power_torch(v3,self.oti3_soap_model['centers'],self.oti3_soap_model['width'],self.oti3_soap_model['lmax'])
        o_site=self._local_soap_softmin(fp_o,self.oti3_soap_model)

        # Normalize by learned 95th-percentile in-manifold distance so Ti and O
        # observers have comparable authority despite different descriptor sizes.
        e_ti=ti_site.mean(1)/float(self.tio6_soap_model['q95'])
        e_o=o_site.mean(1)/float(self.oti3_soap_model['q95'])

        # Retain only a hard local O--O exclusion as a numerical/physical guardrail.
        oo_dist=[]
        for i in range(no-1):
            od=o_frac[:,i+1:,:]-o_frac[:,i:i+1,:]
            oc=torch.einsum('boqi,bij->boqj',od[:,:,None,:]+self.shift[None,None,:,:],cell)
            oo_dist.append(torch.linalg.norm(oc,dim=-1).min(-1).values)
        if oo_dist:
            oo_d=torch.cat(oo_dist,dim=1)
            oo_pair=torch.nn.functional.softplus((1.75-oo_d)/0.05).pow(2)
            e_oo=oo_pair.sum(1)/float(no); oo_min=oo_d.min(1).values
            oo_lt14=(oo_d<1.40).sum(1).to(cell.dtype); oo_lt16=(oo_d<1.60).sum(1).to(cell.dtype); oo_lt18=(oo_d<1.80).sum(1).to(cell.dtype)
        else:
            e_oo=torch.zeros(B,device=self.device); oo_min=torch.full((B,),float('inf'),device=self.device)
            oo_lt14=oo_lt16=oo_lt18=torch.zeros(B,device=self.device)
        total=e_ti+e_o+4.0*e_oo
        if details:
            return total,{'soap_ti':e_ti,'soap_o':e_o,'soap_ti_site_mean':ti_site.mean(1),'soap_o_site_mean':o_site.mean(1),
                          'soap_ti_site_max':ti_site.max(1).values,'soap_o_site_max':o_site.max(1).values,
                          'oo':e_oo,'oo_min':oo_min,'oo_lt14':oo_lt14,'oo_lt16':oo_lt16,'oo_lt18':oo_lt18,
                          'tio_min':dist.amin(dim=(1,2)),'tio_mean6':torch.topk(dist,k=6,dim=-1,largest=False).values.mean((1,2))}
        return total

    def _run_oxygen_field(self,t,ti_raw,ntraj=16,steps=500,lr=0.010,temp0=0.025,temp1=0.0005):
        """Explicit 24-O Langevin search with Ti and lattice frozen."""
        with torch.no_grad():
            _,_,cell,_,ti_frac,_=self.geometry(t,ti_raw)
        cell=cell.detach(); ti_frac=ti_frac.detach(); B=int(ntraj)
        cell=cell.expand(B,-1,-1).contiguous(); ti=ti_frac.expand(B,-1,-1).contiguous()
        # Start from 24 diffuse O particles uniformly over the current Stage-A cell.
        o=torch.rand((B,24,3),device=self.device,dtype=cell.dtype)
        bestE=torch.full((B,),float('inf'),device=self.device); best=o.clone(); trace=[]
        for it in range(int(steps)):
            o=o.detach().requires_grad_(True)
            E,D=self._oxygen_field_energy(ti,o,cell,details=True)
            g=torch.autograd.grad(E.sum(),o)[0]
            # clip per-particle force so one collision cannot launch an O across the cell
            gn=torch.linalg.norm(g,dim=-1,keepdim=True); g=g*torch.clamp(8.0/torch.clamp(gn,min=1e-8),max=1.0)
            q=it/max(int(steps)-1,1); temp=float(temp0)*(float(temp1)/float(temp0))**q
            eta=float(lr)*(0.35+0.65*(1.0-q))
            with torch.no_grad():
                o=(o-eta*g + math.sqrt(max(2.0*eta*temp,0.0))*torch.randn_like(o))%1.0
                En=self._oxygen_field_energy(ti,o,cell)
                imp=En<bestE; bestE=torch.where(imp,En,bestE); best=torch.where(imp[:,None,None],o,best)
                if it in (0,9,24,49,99,199,int(steps)-1):
                    trace.append({'step':int(it+1),'best_energy':float(bestE.min().cpu()),'median_energy':float(bestE.median().cpu()),'temperature':temp})
        j=int(torch.argmin(bestE).item()); of=best[j:j+1]
        with torch.no_grad(): E,D=self._oxygen_field_energy(ti[j:j+1],of,cell[j:j+1],details=True)
        return {'energy':float(E[0]),'components':{k:float(v[0]) for k,v in D.items()},'o_frac':of[0].cpu().numpy(),
                'ti_frac':ti[j].cpu().numpy(),'cell':cell[j].cpu().numpy(),'trajectory':trace,'ntraj':B,'steps':int(steps)}

    def _oxygen_wyckoff_template(self, spg, token):
        """Build an exact symmetry expansion for 24 oxygen sites from a Wyckoff multiset."""
        from pyxtal.symmetry import Group
        g=Group(int(spg)); wps=decode_int_token(token)
        dofs=[]; As=[]; bs=[]; rots=[]; trans=[]
        for wp_id in wps:
            wp=g[int(wp_id)]; d=int(wp.get_dof())
            A,b=self.affine_map(lambda u,wp=wp:wp.get_position_from_free_xyzs(u),d)
            dofs.append(d)
            As.append(torch.tensor(A,dtype=torch.float32,device=self.device))
            bs.append(torch.tensor(b,dtype=torch.float32,device=self.device))
            rots.append(torch.tensor(np.asarray([op.rotation_matrix for op in wp.ops],float),dtype=torch.float32,device=self.device))
            trans.append(torch.tensor(np.asarray([op.translation_vector for op in wp.ops],float),dtype=torch.float32,device=self.device))
        n=sum(len(x) for x in rots)
        if n != 24: raise ValueError(f'O Wyckoff token expands to {n}, expected 24: {token}')
        return {'token':token,'wps':tuple(wps),'dofs':tuple(dofs),'A':As,'b':bs,'rots':rots,'trans':trans,'ncoord':sum(dofs)}

    def _expand_oxygen_wyckoff(self, wt, raw):
        centres=[]; cur=0
        for d,A,b,R,T in zip(wt['dofs'],wt['A'],wt['b'],wt['rots'],wt['trans']):
            if d:
                u=torch.sigmoid(raw[:,cur:cur+d]); cur+=d
                gen=(u@A.T+b)%1.0
            else:
                gen=b[None,:].expand(raw.shape[0],-1)%1.0
            orb=(torch.einsum('oij,bj->boi',R,gen)+T[None])%1.0
            centres.append(orb)
        return torch.cat(centres,1)

    def _profile_sync(self):
        """Synchronize the active CUDA device only at coarse profiling boundaries."""
        if str(self.device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _run_oxygen_field_wyckoff(self,t,ti_raw,ntraj=16,steps=500,lr=0.010,temp0=0.025,temp1=0.0005):
        """Batched symmetry-exact oxygen search (v136r4).

        Physical objective, screening, trajectory allocation, Langevin schedule,
        best-ever retention and r2 convergence criteria are unchanged.  The
        selected Wyckoff templates are optimized in one padded batched graph so
        the GPU sees all active template/trajectory rows at each field step.
        """
        prof={'geometry':0.0,'enumerate_templates':0.0,'screen_total':0.0,
              'optimize_total':0.0,'final_eval_total':0.0,'templates_total':0,
              'templates_chosen':0,'screen_starts':0,'optimization_steps':0,
              'batched_global_steps':0,'chosen':[],
              'kernel_timing':{'expand':0.0,'forward':0.0,'backward':0.0,
                               'update':0.0,'best_eval':0.0,'convergence':0.0,
                               'grad_clip':0.0,'noise':0.0,'apply_update':0.0,
                               'best_forward':0.0,'best_retain':0.0}}
        self._profile_sync(); tic=time.perf_counter()
        with torch.no_grad(): _,_,cell0,_,ti0,_=self.geometry(t,ti_raw)
        cell0=cell0.detach(); ti0=ti0.detach(); spg=int(t['spg'])
        self._profile_sync(); prof['geometry']=time.perf_counter()-tic

        self._profile_sync(); tic=time.perf_counter()
        toks=exact_parent_entries(spg,24,128)
        if not toks: raise RuntimeError(f'no exact 24-O Wyckoff decomposition for SG {spg}')
        rng=np.random.default_rng(int(getattr(self,'global_seed',0))+1009*spg)
        if len(toks)>32:
            ids=np.sort(rng.choice(len(toks),size=32,replace=False)); toks=[toks[int(i)] for i in ids]
        prof['templates_total']=len(toks)
        self._profile_sync(); prof['enumerate_templates']=time.perf_counter()-tic

        self._profile_sync(); tic=time.perf_counter()
        screened=[]
        for tok in toks:
            wt=self._oxygen_wyckoff_template(spg,tok); d=wt['ncoord']; ns=4 if d else 1
            prof['screen_starts'] += ns
            if d:
                u=torch.tensor(rng.uniform(.02,.98,size=(ns,d)),device=self.device,dtype=cell0.dtype)
                raw=torch.log(u/(1-u))
            else: raw=torch.zeros((1,0),device=self.device,dtype=cell0.dtype)
            with torch.no_grad():
                of=self._expand_oxygen_wyckoff(wt,raw)
                e=self._oxygen_field_energy(ti0.expand(len(raw),-1,-1),of,cell0.expand(len(raw),-1,-1))
                screened.append((float(e.min().cpu()),tok,wt))
        screened.sort(key=lambda x:x[0])
        self._profile_sync(); prof['screen_total']=time.perf_counter()-tic

        nkeep=min(8,len(screened),max(1,int(ntraj))); chosen=screened[:nkeep]
        prof['templates_chosen']=len(chosen)
        base=max(1,int(ntraj)//nkeep); rem=max(0,int(ntraj)-base*nkeep)

        # Build one padded raw-coordinate tensor.  Padding coordinates never enter
        # any symmetry expansion and therefore have exactly zero gradient.
        specs=[]; maxd=max([int(wt['ncoord']) for _,_,wt in chosen]+[0]); row0=0
        for kk,(e0,tok,wt) in enumerate(chosen):
            B=(base+(1 if kk<rem else 0)) if wt['ncoord'] else 1
            d=int(wt['ncoord']); sl=slice(row0,row0+B); row0+=B
            specs.append({'e0':e0,'tok':tok,'wt':wt,'d':d,'B':B,'slice':sl,
                          'active':bool(d),'stop_reason':'max_steps','stop_hits':0,
                          'check_history':[],'n_exec':0,'best_step':0,
                          'best_energy':float('inf'),'conv':[]})
        N=row0
        raw=torch.zeros((N,maxd),device=self.device,dtype=cell0.dtype)
        for sp in specs:
            if sp['d']:
                u=torch.tensor(rng.uniform(.02,.98,size=(sp['B'],sp['d'])),device=self.device,dtype=cell0.dtype)
                raw[sp['slice'],:sp['d']]=torch.log(u/(1-u))
        bestE=torch.full((N,),float('inf'),device=self.device); bestraw=raw.detach().clone()
        # v136r4: keep best-step bookkeeping on-device.  r3 synchronized the GPU
        # once per active template on every field step just to update reporting
        # metadata; this does not participate in the objective or convergence.
        bestStepRows=torch.zeros((N,),device=self.device,dtype=torch.int64)
        coord_mask=torch.zeros((N,maxd),device=self.device,dtype=cell0.dtype)
        for sp in specs:
            if sp['d']: coord_mask[sp['slice'],:sp['d']]=1.0
        ti_all=ti0.expand(N,-1,-1).contiguous(); cell_all=cell0.expand(N,-1,-1).contiguous()

        def expand_all(r):
            return torch.cat([self._expand_oxygen_wyckoff(sp['wt'],r[sp['slice'],:sp['d']]) for sp in specs],0)
        def snap_for(sp,E,g,step):
            sl=sp['slice']; d=sp['d']
            if not d:
                return {'step':int(step),'energy_best':float(E[sl].min().detach().cpu()),
                        'energy_median':float(E[sl].median().detach().cpu()),'grad_rms_best':0.0,
                        'grad_rms_median':0.0,'grad_max_best':0.0,'grad_max_median':0.0}
            gg=g[sl,:d]; pr=torch.sqrt(torch.mean(gg*gg,dim=-1)); pm=torch.amax(torch.abs(gg),dim=-1)
            return {'step':int(step),'energy_best':float(E[sl].min().detach().cpu()),
                    'energy_median':float(E[sl].median().detach().cpu()),
                    'grad_rms_best':float(pr.min().detach().cpu()),'grad_rms_median':float(pr.median().detach().cpu()),
                    'grad_max_best':float(pm.min().detach().cpu()),'grad_max_median':float(pm.median().detach().cpu())}

        check_interval=25; min_steps=min(100,int(steps)); patience=3; de_tol=0.05; grad_rms_tol=0.50
        checkpoints={x for x in (0,100,200,300,400,int(steps)) if 0<=x<=int(steps)}
        active_rows=torch.zeros((N,1),device=self.device,dtype=cell0.dtype)
        for sp in specs:
            if sp['active']: active_rows[sp['slice']]=1.0
            else: sp['stop_reason']='zero_dof'

        # CUDA events give device time without synchronizing every operation.
        use_events=str(self.device).startswith('cuda') and torch.cuda.is_available()
        ev=[]
        def mark(name):
            if use_events:
                e=torch.cuda.Event(enable_timing=True); e.record(); ev.append((name,e))
        self._profile_sync(); opt_tic=time.perf_counter()

        # Initial diagnostics, using one shared graph/backward only when requested.
        if 0 in checkpoints:
            rr=raw.detach().requires_grad_(True); of=expand_all(rr); E=self._oxygen_field_energy(ti_all,of,cell_all)
            g=torch.autograd.grad(E.sum(),rr)[0]
            for sp in specs: sp['conv'].append(snap_for(sp,E,g,0))

        for it in range(int(steps)):
            if not any(sp['active'] for sp in specs): break
            rr=raw.detach().requires_grad_(True)
            mark('expand'); of=expand_all(rr); mark('forward')
            E=self._oxygen_field_energy(ti_all,of,cell_all); mark('backward')
            g=torch.autograd.grad(E.sum(),rr)[0]; mark('update')
            # Same r2 per-trajectory norm clipping, now masked for converged rows.
            gn=torch.linalg.norm(g,dim=-1,keepdim=True); gc=g*torch.clamp(8.0/torch.clamp(gn,min=1e-8),max=1.0); mark('grad_clip')
            q=it/max(int(steps)-1,1); temp=float(temp0)*(float(temp1)/float(temp0))**q; eta=float(lr)*(0.35+0.65*(1-q))
            with torch.no_grad():
                # Preserve r3 RNG consumption exactly: generate the full padded tensor,
                # then mask padding rather than changing the random draw shape.
                noise=math.sqrt(max(2*eta*temp,0.0))*torch.randn_like(raw); mark('noise')
                raw=(raw + active_rows*(-eta*gc+noise))*coord_mask
            mark('apply_update')
            ofn=expand_all(raw); En=self._oxygen_field_energy(ti_all,ofn,cell_all); mark('best_forward')
            with torch.no_grad():
                imp=(En<bestE) & (active_rows[:,0]>0)
                bestE=torch.where(imp,En,bestE); bestraw=torch.where(imp[:,None],raw,bestraw)
                # Reporting-only best-step bookkeeping stays on GPU; this removes the
                # per-template .cpu() synchronization that dominated r3's update path.
                step_tensor=torch.full_like(bestStepRows,it+1)
                bestStepRows=torch.where(imp,step_tensor,bestStepRows)
                for sp in specs:
                    if sp['active']: sp['n_exec']+=1
            mark('best_retain')
            global_step=it+1; prof['batched_global_steps']=global_step

            need_diag=(global_step in checkpoints) or (global_step>=min_steps and global_step%check_interval==0)
            if need_diag:
                # Preserve r2 convergence semantics: diagnose the post-update
                # state.  Unlike r2 this is one shared diagnostic backward for
                # all active templates rather than one backward per template.
                rd=raw.detach().requires_grad_(True); od=expand_all(rd)
                Ed=self._oxygen_field_energy(ti_all,od,cell_all)
                gd=torch.autograd.grad(Ed.sum(),rd)[0]
                for sp in specs:
                    if not sp['active']: continue
                    snap=snap_for(sp,Ed,gd,sp['n_exec'])
                    if global_step in checkpoints: sp['conv'].append(snap)
                    if sp['n_exec']>=min_steps and sp['n_exec']%check_interval==0:
                        sp['check_history'].append((sp['n_exec'],float(bestE[sp['slice']].min().detach().cpu()),snap['grad_rms_best']))
                        if len(sp['check_history'])>=patience+1:
                            h=sp['check_history']; stagnant=(h[-(patience+1)][1]-h[-1][1])<de_tol
                            low=snap['grad_rms_best']<grad_rms_tol
                            sp['stop_hits']=sp['stop_hits']+1 if (stagnant and low) else 0
                            if sp['stop_hits']>=patience:
                                sp['active']=False; sp['stop_reason']='converged'; active_rows[sp['slice']]=0.0
                                if global_step not in checkpoints: sp['conv'].append(snap)

        self._profile_sync(); prof['optimize_total']=time.perf_counter()-opt_tic
        # Resolve CUDA event intervals into coarse kernel categories.  Each marker
        # denotes the end of the named phase; intervals are accumulated by phase.
        if use_events and ev:
            torch.cuda.synchronize(self.device)
            # Consecutive marker intervals: previous marker -> current marker.
            for (pn,pe),(cn,ce) in zip(ev[:-1],ev[1:]):
                if cn in prof['kernel_timing']:
                    prof['kernel_timing'][cn]+=pe.elapsed_time(ce)/1000.0
            kt=prof['kernel_timing']
            # v136r4 detailed update-path accounting. Legacy 'best_eval' and
            # 'convergence' fields remain present for log/parser compatibility.
            kt['update']=kt['grad_clip']+kt['noise']+kt['apply_update']
            kt['best_eval']=kt['best_forward']+kt['best_retain']

        global_best=None; tested=[]
        for sp in specs:
            sl=sp['slice']; d=sp['d']
            if not d:
                with torch.no_grad():
                    of0=self._expand_oxygen_wyckoff(sp['wt'],raw[sl,:0]); e0v=self._oxygen_field_energy(ti0.expand(sp['B'],-1,-1),of0,cell0.expand(sp['B'],-1,-1))
                    bestE[sl]=e0v; bestraw[sl]=raw[sl]; sp['best_energy']=float(e0v.min().cpu()); sp['best_step']=0
                sp['conv']=[{'step':0,'energy_best':sp['best_energy'],'energy_median':sp['best_energy'],'grad_rms_best':0.0,'grad_rms_median':0.0,'grad_max_best':0.0,'grad_max_median':0.0}]
            prof['optimization_steps']+=sp['n_exec']
            jlocal=int(torch.argmin(bestE[sl]).item()); j=sl.start+jlocal
            if d:
                sp['best_energy']=float(bestE[j].detach().cpu())
                sp['best_step']=int(bestStepRows[j].detach().cpu())
            rr=bestraw[j:j+1,:d]
            self._profile_sync(); ft=time.perf_counter()
            with torch.no_grad():
                of=self._expand_oxygen_wyckoff(sp['wt'],rr); Ef,D=self._oxygen_field_energy(ti0,of,cell0,details=True)
            self._profile_sync(); fsec=time.perf_counter()-ft; prof['final_eval_total']+=fsec
            # In a batched run wall time is shared, so per-template opt_seconds is
            # intentionally omitted; steps and global batch time are the meaningful quantities.
            prof['chosen'].append({'token':sp['tok'],'ncoord':d,'ntraj':sp['B'],'steps':sp['n_exec'],
                'stop_reason':sp['stop_reason'],'best_step':sp['best_step'],'best_energy':sp['best_energy'],
                'opt_seconds':float('nan'),'final_seconds':fsec,'seconds_per_step':float('nan'),'convergence':sp['conv']})
            item={'energy':float(Ef[0]),'components':{k:float(v[0]) for k,v in D.items()},'o_frac':of[0].cpu().numpy(),
                  'ti_frac':ti0[0].cpu().numpy(),'cell':cell0[0].cpu().numpy(),'token':sp['tok'],'ncoord':d,'screen_energy':float(sp['e0']),
                  'o_raw':rr[0].detach().cpu().numpy()}
            tested.append({'token':sp['tok'],'ncoord':d,'energy':float(Ef[0]),'screen_energy':float(sp['e0'])})
            if global_best is None or item['energy']<global_best['energy']: global_best=item
        global_best['tested_wyckoff']=tested; global_best['ntraj']=int(ntraj); global_best['steps']=int(steps); global_best['profile']=prof
        return global_best

    def _coordination_counts(self, ti_frac, o_frac, cell):
        """Diagnostic hard coordination counts using the field's CN midpoint cutoff."""
        with torch.no_grad():
            df=o_frac[:,None,:,None,:]-ti_frac[:,:,None,None,:]+self.shift[None,None,None,:,:]
            dc=torch.einsum('bnoqi,bij->bnoqj',df,cell)
            dist=torch.linalg.norm(dc,dim=-1).min(-1).values
            r0=float(self.model.pair(self.parent,self.child).mu)
            cut=r0+0.38
            cti=(dist<cut).sum(-1); co=(dist<cut).sum(1)
            out={'coord_cut_A':float(cut),'Nbond':int((dist<cut).sum().item()),
                 'Ti_abs_defect':float(torch.abs(cti.to(torch.float32)-6).sum().item()),
                 'O_abs_defect':float(torch.abs(co.to(torch.float32)-3).sum().item())}
            for k in range(3,9): out[f'TiO{k}']=int((cti==k).sum().item())
            for k in range(1,6): out[f'OTi{k}']=int((co==k).sum().item())
            out['Ti_under6']=int((cti<6).sum().item()); out['Ti_over6']=int((cti>6).sum().item())
            out['O_under3']=int((co<3).sum().item()); out['O_over3']=int((co>3).sum().item())
            return out

    def _run_stage_b_electron_phonon(self,t,ti_raw_b,sym,max_cycles=80,o_inner=80,phonon_inner=1,
                                      energy_tol=1e-3,volume_weight=120.0,strain_weight=60.0,
                                      ti_anchor_weight=8.0,o_grad_tol=3e-3,o_track_tol=1e-1,
                                      slow_lr=0.018,temp0=0.018,temp1=0.0005,
                                      max_ti_step_A=0.12,max_vol_step=0.025,max_retries=5):
        """r10 adiabatic BO-Langevin B dynamics with three-state O-SCF tracking.

        Oxygen is the fast subsystem and is warm-started/re-solved after EVERY accepted
        Ti+lattice move.  Only the slow Ti/lattice variables receive Langevin noise.
        A slow proposal is retried at half step if the O solve fails or if the proposed
        geometry jump is too large.  Exact SG/Wyckoff symmetry is retained throughout.
        The temperature is annealed to zero and a deterministic BO quench closes B.
        """
        wt=self._oxygen_wyckoff_template(int(t['spg']),sym['token'])
        x=ti_raw_b.detach().clone().requires_grad_(True)
        d=int(wt['ncoord'])
        if d:
            y=torch.tensor(np.asarray(sym['o_raw'],float),device=self.device,dtype=x.dtype)[None].detach().clone().requires_grad_(True)
        else:
            y=torch.empty((1,0),device=self.device,dtype=x.dtype,requires_grad=False)

        def state():
            _,_,cell,_,ti,_=self.geometry(t,x)
            oo=self._expand_oxygen_wyckoff(wt,y)
            return cell,ti,oo

        with torch.no_grad():
            c0,ti0,o0=state(); E0,D0=self._oxygen_field_energy(ti0,o0,c0,details=True)
            b_cell=c0.detach().clone(); b_inv=torch.linalg.inv(b_cell); b_vol=torch.det(b_cell)
            b_ti_cart=torch.einsum('bni,bij->bnj',ti0,b_cell).detach().clone()

        def objective(include_trust=True,details=False):
            cell,ti,oo=state(); field,D=self._oxygen_field_energy(ti,oo,cell,details=True)
            F=cell@b_inv; I=torch.eye(3,device=self.device,dtype=cell.dtype)[None]
            eps=0.5*(F+F.transpose(-1,-2))-I
            tr=torch.diagonal(eps,dim1=-2,dim2=-1).sum(-1)/3.0; dev=eps-tr[:,None,None]*I
            vr=torch.det(cell)/b_vol
            ti_cart=torch.einsum('bni,bij->bnj',ti,cell)
            e_vol=float(volume_weight)*torch.log(torch.clamp(vr,min=1e-8)).pow(2)
            e_str=float(strain_weight)*(dev*dev).mean((1,2))
            e_ti=float(ti_anchor_weight)*((ti_cart-b_ti_cart)**2).mean((1,2))
            total=field+(e_vol+e_str+e_ti if include_trust else 0.0)
            if details:return total,field,D,e_vol,e_str,e_ti,cell,ti,oo,eps,vr
            return total

        def grad_rms(params):
            gs=[p.grad.reshape(-1) for p in params if p.grad is not None and p.grad.numel()]
            if not gs:return float('nan')
            g=torch.cat(gs); return float(torch.sqrt(torch.mean(g*g)).detach().cpu())

        def run_o_scf(max_iter=None):
            if not d:return True,0.0,0
            mi=int(o_inner if max_iter is None else max_iter)
            opt=torch.optim.LBFGS([y],lr=0.5,max_iter=mi,max_eval=mi*2,tolerance_grad=1e-7,
                                  tolerance_change=1e-9,history_size=12,line_search_fn='strong_wolfe')
            gr=[float('inf')]; calls=[0]
            def closure():
                opt.zero_grad(set_to_none=True); E=objective(False).sum(); calls[0]+=1
                if not torch.isfinite(E):raise FloatingPointError('non-finite O-SCF energy')
                E.backward(); gr[0]=grad_rms([y]); return E
            try:
                opt.step(closure)
                # Explicit final residual: LBFGS success is not synonymous with SCF convergence.
                if y.grad is not None:y.grad=None
                Ef=objective(False).sum(); Ef.backward(); gr[0]=grad_rms([y])
                # Three-state O-SCF classification.  ``converged`` is the strict
                # electronic tolerance; ``trackable`` is a finite, bounded residual that
                # is safe to warm-start across a small BO move; larger/non-finite residuals
                # are true failures.  This changes acceptance logic only -- no energy term
                # or Langevin parameter is modified.
                if not np.isfinite(gr[0]):
                    status='failed'
                elif gr[0] <= float(o_grad_tol):
                    status='converged'
                elif gr[0] <= float(o_track_tol):
                    status='trackable'
                else:
                    status='failed'
                if status != 'failed':
                    return status,gr[0],calls[0]
                # v137 rescue: SOAP soft-min fields can have broad/curved basin
                # boundaries where quasi-Newton history is unhelpful.  Take a short
                # damped Adam descent, then retry LBFGS from the rescued point.
                rescue=torch.optim.Adam([y],lr=0.015)
                for _ in range(40):
                    rescue.zero_grad(set_to_none=True); er=objective(False).sum(); calls[0]+=1
                    if not torch.isfinite(er): break
                    er.backward(); rescue.step()
                if y.grad is not None:y.grad=None
                opt2=torch.optim.LBFGS([y],lr=0.35,max_iter=max(20,mi//2),max_eval=max(40,mi),
                                       tolerance_grad=1e-7,tolerance_change=1e-9,history_size=8,line_search_fn='strong_wolfe')
                def closure2():
                    opt2.zero_grad(set_to_none=True); er=objective(False).sum(); calls[0]+=1
                    if not torch.isfinite(er): raise FloatingPointError('non-finite rescued O-SCF energy')
                    er.backward(); return er
                opt2.step(closure2)
                if y.grad is not None:y.grad=None
                ef=objective(False).sum(); ef.backward(); gr[0]=grad_rms([y])
                if not np.isfinite(gr[0]): status='failed'
                elif gr[0] <= float(o_grad_tol): status='converged'
                elif gr[0] <= float(o_track_tol): status='trackable'
                else: status='failed'
                return status,gr[0],calls[0]
            except Exception:
                return 'failed',gr[0],calls[0]

        trace=[]
        def snap(cyc,phase,accepted=True,grad=float('nan'),extra=None):
            with torch.no_grad():
                E,F,D,ev,es,et,cell,ti,oo,eps,vr=objective(True,True)
                tc=torch.einsum('bni,bij->bnj',ti,cell); disp=torch.linalg.norm(tc-b_ti_cart,dim=-1)
                comp={k:float(v[0]) for k,v in D.items()}; coord=self._coordination_counts(ti,oo,cell)
                z={'cycle':int(cyc),'phase':phase,'accepted':bool(accepted),'objective':float(E[0]),'field_energy':float(F[0]),
                   'components':comp,'coordination':coord,'elastic_volume':float(ev[0]),'elastic_strain':float(es[0]),
                   'ti_anchor':float(et[0]),'volume_ratio':float(vr[0]),'strain_rms':float(torch.sqrt((eps*eps).mean())),
                   'ti_disp_rms_A':float(torch.sqrt((disp*disp).mean())),'ti_disp_max_A':float(disp.max()),'grad_rms':float(grad),
                   'cell':cell[0].cpu().numpy(),'ti_frac':ti[0].cpu().numpy(),'o_frac':oo[0].cpu().numpy()}
                if extra:z.update(extra)
                trace.append(z); return z

        # Initial electronic ground-state solve.  From here on every accepted slow
        # geometry has a corresponding converged, warm-started O state.
        o_status,g_o,n_o=run_o_scf(max(o_inner,120))
        last_o_grad=float(g_o)
        snap(0,'O_SCF_initial',o_status != 'failed',g_o,{'o_scf_calls':n_o,'o_scf_status':o_status})
        if o_status == 'failed':
            final=trace[-1]
            return {'energy':final['objective'],'field_energy':final['field_energy'],'components':final['components'],
                    'coordination':final['coordination'],'cell':final['cell'],'ti_frac':final['ti_frac'],'o_frac':final['o_frac'],
                    'trace':trace,'cycles':0,'stop_reason':'initial_O_SCF_failed','escape_used':False}

        nlat=len(t['spec']); gen=torch.Generator(device=self.device)
        gen.manual_seed(int(getattr(self,'global_seed',2025))+100003*int(t['spg'])+9000007)
        stable=0; stop='max_cycles'; accepted_steps=0
        for cyc in range(1,int(max_cycles)+1):
            q=(cyc-1)/max(int(max_cycles)-1,1); temp=float(temp0)*(1-q)+float(temp1)*q
            xbase=x.detach().clone(); ybase=y.detach().clone()
            with torch.no_grad(): cbase,tibase,obase=state(); vbase=float((torch.det(cbase)/b_vol)[0].cpu())
            base_obj=float(trace[-1]['objective']); accepted=False; retry_info=[]

            # BO force: O is frozen at its converged state while differentiating slow variables.
            if x.grad is not None:x.grad=None
            Eslow=objective(True).sum(); Eslow.backward(); gx=x.grad.detach().clone()
            gslow=float(torch.sqrt(torch.mean(gx*gx)).cpu())
            # Normalize force by subsystem so raw parameter scales do not dominate.
            glat=torch.sqrt(torch.mean(gx[:,:nlat]**2)).clamp_min(1e-8) if nlat else None
            gti=torch.sqrt(torch.mean(gx[:,nlat:]**2)).clamp_min(1e-8) if gx.shape[1]>nlat else None

            for retry in range(int(max_retries)+1):
                # Reduce slow motion when the fast subsystem is only loosely tracked.
                qscf=1.0 if last_o_grad <= float(o_grad_tol) else max(0.20,min(1.0,math.sqrt(float(o_track_tol)/max(last_o_grad,1e-12))*0.50))
                scale=(0.5**retry)*qscf
                with torch.no_grad():
                    x.copy_(xbase); y.copy_(ybase) if d else None
                    # Slow BO-Langevin step. Lattice receives weaker stochastic motion;
                    # its deterministic BO force remains fully active.
                    if nlat:
                        nl=torch.randn(x[:,:nlat].shape,device=self.device,dtype=x.dtype,generator=gen)
                        x[:,:nlat].add_(-float(slow_lr)*scale*gx[:,:nlat]/glat +
                                        math.sqrt(2.0*float(slow_lr)*temp)*scale*0.20*nl)
                    if gx.shape[1]>nlat:
                        nt=torch.randn(x[:,nlat:].shape,device=self.device,dtype=x.dtype,generator=gen)
                        x[:,nlat:].add_(-float(slow_lr)*scale*gx[:,nlat:]/gti +
                                        math.sqrt(2.0*float(slow_lr)*temp)*scale*nt)
                    x.clamp_(-10.0,10.0)
                    cprop,tiprop,_=state()
                    ti0c=torch.einsum('bni,bij->bnj',tibase,cbase)
                    tipc=torch.einsum('bni,bij->bnj',tiprop,cprop)
                    ti_step=float(torch.sqrt(torch.mean(torch.linalg.norm(tipc-ti0c,dim=-1)**2)).cpu())
                    vprop=float((torch.det(cprop)/b_vol)[0].cpu()); dv=abs(vprop-vbase)
                if ti_step>float(max_ti_step_A) or dv>float(max_vol_step):
                    retry_info.append({'retry':retry,'reason':'slow_step_too_large','ti_step_A':ti_step,'dVratio':dv})
                    continue

                # Critical BO tracking step: warm-start O from ybase and converge it at
                # the proposed Ti/lattice geometry BEFORE accepting that geometry.
                o_status,g,ncall=run_o_scf()
                retry_info.append({'retry':retry,'reason':('ok' if o_status == 'converged' else ('O_SCF_trackable' if o_status == 'trackable' else 'O_SCF_failed')),'ti_step_A':ti_step,
                                   'dVratio':dv,'o_grad_rms':g,'o_scf_calls':ncall,'o_scf_status':o_status})
                if o_status != 'failed':
                    accepted=True; accepted_steps+=1; last_o_grad=float(g); break

            if not accepted:
                with torch.no_grad():
                    x.copy_(xbase)
                    if d:y.copy_(ybase)
                snap(cyc,'BO_step_rejected',False,gslow,{'temperature':temp,'retries':retry_info})
                stop='O_tracking_failed'; break

            ss=snap(cyc,'BO_Langevin',True,gslow,{'temperature':temp,'retries':retry_info,
                                                  'slow_step_A':retry_info[-1]['ti_step_A'],
                                                  'o_grad_rms':retry_info[-1]['o_grad_rms'],
                                                  'o_scf_status':retry_info[-1]['o_scf_status']})
            imp=base_obj-ss['objective']
            # At finite T uphill steps are intentional. Only test stagnation in the cold tail.
            if q>0.75 and abs(imp)<float(energy_tol) and gslow<5e-3: stable+=1
            else: stable=0
            if stable>=3:
                stop='cold_converged'; break

        # Zero-temperature BO quench: tightly converge O, then alternate a few
        # deterministic slow quasi-Newton steps with a fresh O-SCF after every step.
        for k in range(8):
            o_status,g,ncall=run_o_scf(max(o_inner,120))
            snap(int(max_cycles)+k+1,'O_SCF_quench',o_status != 'failed',g,{'o_scf_calls':ncall,'o_scf_status':o_status})
            if o_status == 'failed': stop='final_O_SCF_failed'; break
            xpre=x.detach().clone(); ypre=y.detach().clone(); epre=trace[-1]['objective']
            opt=torch.optim.LBFGS([x],lr=0.25,max_iter=1,max_eval=4,tolerance_grad=1e-7,
                                  tolerance_change=1e-9,history_size=8,line_search_fn='strong_wolfe')
            gs=[float('inf')]
            def cl():
                opt.zero_grad(set_to_none=True); ee=objective(True).sum(); ee.backward(); gs[0]=grad_rms([x]); return ee
            try: opt.step(cl)
            except Exception:
                with torch.no_grad():x.copy_(xpre)
                stop='final_slow_failed'; break
            o_status2,g2,n2=run_o_scf(max(o_inner,120))
            if o_status2 == 'failed':
                with torch.no_grad():
                    x.copy_(xpre)
                    if d:y.copy_(ypre)
                stop='final_O_tracking_failed'; break
            sq=snap(int(max_cycles)+k+1,'BO_quench',True,gs[0],{'o_grad_rms':g2,'o_scf_calls':n2,'o_scf_status':o_status2})
            if epre-sq['objective']>=0 and epre-sq['objective']<float(energy_tol) and gs[0]<5e-3:
                stop='quenched'; break

        final=trace[-1]
        return {'energy':final['objective'],'field_energy':final['field_energy'],'components':final['components'],
                'coordination':final['coordination'],'cell':final['cell'],'ti_frac':final['ti_frac'],'o_frac':final['o_frac'],
                'trace':trace,'cycles':max(z['cycle'] for z in trace),'stop_reason':stop,'escape_used':True,
                'accepted_slow_steps':accepted_steps}

    def build(self,spg,wp_token,nstarts=None):
        """One conditioned Stage-A Ti framework -> explicit oxygen field search; Ti/cell fixed."""
        t=self.template(spg,wp_token); batch_starts=max(1,int(getattr(self,'stage_a_batch_starts',64)))
        self._live('sg_start',spg=int(spg),nstarts=batch_starts,target_a=1)
        a,ids,asec=self._stage_a_batch(t,spg,0,batch_starts,1)
        if len(ids)==0:return []
        qi=int(ids[0]); abase=a[qi:qi+1].detach().clone(); probe,_=self.environment_shake(t,abase,'SOAP',self.probe_steps,self.environment_lr,False); start=probe
        remain=max(0,int(self.environment_steps)-int(self.probe_steps)); afinal,_=(self.environment_shake(t,start,'SOAP',remain,self.environment_lr,False) if remain else (start,0.0))
        with torch.no_grad(): _,DA,_=self.environment_loss(t,afinal,'SOAP',details=True); soapA=float(DA['worst'][0])
        ntr=int(getattr(self,'oxygen_field_trajectories',16)); nst=int(getattr(self,'oxygen_field_steps',500)); flr=float(getattr(self,'oxygen_field_lr',.010)); t0=float(getattr(self,'oxygen_field_temp0',.025)); t1=float(getattr(self,'oxygen_field_temp1',.0005))
        tic=time.perf_counter(); sym=self._run_oxygen_field_wyckoff(t,afinal,ntr,nst,flr,t0,t1); sec_sym=time.perf_counter()-tic
        self._live('oxygen_field_done',mode='wyckoff',spg=int(spg),energy=float(sym['energy']),seconds=float(sec_sym),components=sym['components'],wyckoff_token=sym['token'],wyckoff_dof=sym['ncoord'],profile=sym.get('profile',{}))
        bcoord=self._coordination_counts(torch.tensor(sym['ti_frac'],device=self.device,dtype=afinal.dtype)[None],torch.tensor(sym['o_frac'],device=self.device,dtype=afinal.dtype)[None],torch.tensor(sym['cell'],device=self.device,dtype=afinal.dtype)[None])
        tic=time.perf_counter(); cres=self._run_stage_b_electron_phonon(t,afinal,sym); sec_c=time.perf_counter()-tic
        self._live('stageB_scf_done',spg=int(spg),seconds=float(sec_c),energy_before=float(sym['energy']),energy_after=float(cres['field_energy']),regularized_energy_after=float(cres['energy']),components_before=sym['components'],components_after=cres['components'],coord_before=bcoord,coord_after=cres['coordination'],cycles=int(cres['cycles']),stop_reason=cres['stop_reason'],trace=cres['trace'])
        rec={'spg':int(spg),'a_source_index':qi,'soap_stageA':soapA,'stageA_seconds_sg':float(asec),
             'stageB_seconds_total':float(sec_sym),'wyckoff_seconds':float(sec_sym),'stageB_scf_seconds':float(sec_c),
             'field_energy':sym['energy'],'field_components':sym['components'],'stageB_coordination':bcoord,
             'stageB_cell':sym['cell'],'stageB_ti_frac':sym['ti_frac'],'stageB_o_frac':sym['o_frac'],
             'wyckoff_energy':sym['energy'],'wyckoff_components':sym['components'],'wyckoff_token':sym['token'],'wyckoff_dof':sym['ncoord'],
             'wyckoff_tested':sym['tested_wyckoff'],'wyckoff_profile':sym.get('profile',{}),'wyckoff_cell':sym['cell'],'wyckoff_ti_frac':sym['ti_frac'],'wyckoff_o_frac':sym['o_frac'],
             'stageB_scf_energy':cres['field_energy'],'stageB_scf_regularized_energy':cres['energy'],'stageB_scf_components':cres['components'],'stageB_scf_coordination':cres['coordination'],
             'stageB_scf_cell':cres['cell'],'stageB_scf_ti_frac':cres['ti_frac'],'stageB_scf_o_frac':cres['o_frac'],'stageB_scf_trace':cres['trace'],'stageB_scf_cycles':cres['cycles'],'stageB_scf_stop_reason':cres['stop_reason'],
             'ntraj':ntr,'steps':nst}
        self._live('sg_compute_done',spg=int(spg),a_entrances=1,cycles=1+int(cres['cycles']),stop_reason='stageB_scf_'+cres['stop_reason'],stageA_seconds=float(asec),stageB_seconds=float(sec_sym)+float(sec_c),stageB_scf_seconds=float(sec_c)); return [rec]

def _expand_ops(seed, ops):
    out=[]
    for f in seed:
        for op in ops:
            q=np.asarray(op(np.asarray(f,float)),float)%1.0
            if not any(np.max(np.abs(((q-r+.5)%1)-.5))<1e-6 for r in out): out.append(q)
    return np.asarray(out,float)

def _reference_tio2_structures():
    """Built-in crystallographic calibration structures (conventional cells)."""
    refs={}
    # Rutile P42/mnm, experimental a/c and oxygen internal parameter u.
    u=.3053; ti=np.array([[0,0,0],[.5,.5,.5]],float)
    o=np.array([[u,u,0],[1-u,1-u,0],[.5+u,.5-u,.5],[.5-u,.5+u,.5]],float)%1
    refs['rutile']=(np.diag([4.5936,4.5936,2.9587]),np.vstack([ti,o]),['Ti']*2+['O']*4)
    # Anatase I41/amd conventional cell, standard origin choice.
    z=.2080
    ti=np.array([[0,0,0],[0,.5,.25],[.5,.5,.5],[.5,0,.75]],float)
    o=np.array([[0,0,z],[0,.5,.25+z],[.5,.5,.5+z],[.5,0,.75+z],
                [0,.5,.25-z],[0,0,1-z],[.5,0,.75-z],[.5,.5,.5-z]],float)%1
    refs['anatase']=(np.diag([3.784,3.784,9.515]),np.vstack([ti,o]),['Ti']*4+['O']*8)
    # Brookite Pbca: Meagher/Lager-type asymmetric coordinates expanded by the 8 ops.
    ops61=[lambda p:p,lambda p:np.array([p[0],.5-p[1],.5+p[2]]),
           lambda p:np.array([-p[0],.5+p[1],.5-p[2]]),lambda p:np.array([.5-p[0],.5+p[1],p[2]]),
           lambda p:np.array([.5+p[0],.5-p[1],-p[2]]),lambda p:np.array([.5+p[0],p[1],.5-p[2]]),
           lambda p:np.array([.5-p[0],-p[1],.5+p[2]]),lambda p:-p]
    ti=_expand_ops([[.1290,.0999,.8628]],ops61); o1=_expand_ops([[.0098,.1484,.1831]],ops61); o2=_expand_ops([[.2322,.1117,.5366]],ops61)
    refs['brookite']=(np.diag([9.184,5.447,5.145]),np.vstack([ti,o1,o2]),['Ti']*len(ti)+['O']*(len(o1)+len(o2)))
    # Hollandite-type TiO2(H), I4/m. Lattice from K0.008TiO2; B/O internal
    # coordinates from a tetragonal Ti-bearing hollandite refinement. Tunnel A sites omitted.
    def ops87(p):
        x,y,z=p; base=[(x,y,z),(-y,x,z),(-x,-y,z),(y,-x,z)]
        return [np.array(q) for q in base]+[np.array(q)+np.array([.5,.5,.5]) for q in base]
    def ex87(p): return _expand_ops([p],sum(([lambda x,q=q:q(x)] for q in []),[]))
    # expand explicitly because closures over coordinate triples are clearer here.
    def e87(p):
        arr=[]
        for q in ops87(np.asarray(p,float)):
            q=q%1
            if not any(np.max(np.abs(((q-r+.5)%1)-.5))<1e-6 for r in arr): arr.append(q)
        return np.asarray(arr,float)
    ti=e87([.35132,.16737,0]); o1=e87([.1552,.2022,0]); o2=e87([.5417,.1679,0])
    refs['hollandite']=(np.diag([10.1631,10.1631,2.9639]),np.vstack([ti,o1,o2]),['Ti']*len(ti)+['O']*(len(o1)+len(o2)))
    return refs

def _write_simple_cif(path, cell, frac, symbols, title='TiO2'):
    """Dependency-free P1 CIF writer for exact realized coordinates."""
    cell=np.asarray(cell,float); frac=np.asarray(frac,float)%1.0
    def lensang(c):
        a,b,cc=[np.linalg.norm(v) for v in c]
        def ang(u,v): return np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))
        return a,b,cc,ang(c[1],c[2]),ang(c[0],c[2]),ang(c[0],c[1])
    a,b,c,al,be,ga=lensang(cell)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        f.write(f'data_{title}\n_cell_length_a {a:.8f}\n_cell_length_b {b:.8f}\n_cell_length_c {c:.8f}\n')
        f.write(f'_cell_angle_alpha {al:.8f}\n_cell_angle_beta {be:.8f}\n_cell_angle_gamma {ga:.8f}\n')
        f.write("_symmetry_space_group_name_H-M 'P 1'\n_symmetry_Int_Tables_number 1\n")
        f.write('loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n')
        counts={}
        for sy,q in zip(symbols,frac):
            counts[sy]=counts.get(sy,0)+1
            f.write(f'{sy}{counts[sy]} {sy} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f}\n')

def _v135_worker(worker_id, device, tasks, cfg, event_q):
    """One GPU worker; all shared logging is serialized through the manager queue."""
    rows=[]
    try:
        if str(device).startswith('cuda:'):
            torch.cuda.set_device(int(str(device).split(':')[-1]))
        torch.set_num_threads(1)
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        model=ChemistryModel(cfg['model'])
        formula,z,counts,parent,child=model.resolve_target(cfg['target'])
        b=TiOnlyBuilder(model,parent,child,device=device,starts=cfg['starts'],
                        ambient_steps=cfg['ambient_steps'],ambient_lr=cfg['ambient_lr'],lr=cfg['builder_lr'])
        for k,v in cfg['builder_attrs'].items(): setattr(b,k,v)
        b.live_event_q=event_q; b.live_worker_id=worker_id
        event_q.put({'kind':'worker_start','worker':worker_id,'device':device,'ntasks':len(tasks)})
        for it,(spg,tok) in enumerate(tasks,1):
            tic=time.perf_counter(); err=None; rr=[]
            import io
            buf=io.StringIO()
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    rr=b.build(spg,tok)
                if rr is None: rr=[]
                if not isinstance(rr,list):
                    raise TypeError(f'builder.build returned {type(rr).__name__}, expected list')
            except Exception as e:
                err=f'{type(e).__name__}: {e}'
                traceback.print_exc(file=buf)
                rr=[]
            elapsed=time.perf_counter()-tic
            for r in rr:
                if not isinstance(r,dict):
                    raise TypeError(f'build row is {type(r).__name__}, expected dict')
                r['worker']=worker_id; r['device']=device; rows.append(r)
            event_q.put({'kind':'search_done','worker':worker_id,'device':device,'index':it,'ntasks':len(tasks),
                         'spg':int(spg),'elapsed_s':elapsed,'rows':rr,'error':err,'detail':buf.getvalue(),
                         'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S')})
        event_q.put({'kind':'worker_done','worker':worker_id,'device':device,'rows':rows})
    except Exception as e:
        event_q.put({'kind':'worker_fatal','worker':worker_id,'device':device,
                     'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'rows':rows})

def _v135_jsonable(x):
    if x is None or isinstance(x,(str,bool,int,float)): return x
    if isinstance(x,(np.floating,np.integer)): return x.item()
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x,Path): return str(x)
    if isinstance(x,dict): return {str(k):_v135_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)): return [_v135_jsonable(v) for v in x]
    raise TypeError(f'not JSON serializable: {type(x).__name__}')

def _formula_prefix(target):
    # "TiO2=12" -> "TiO2"; safe for directory names and general targets.
    f=str(target).split('=',1)[0].strip()
    f=re.sub(r'[^A-Za-z0-9_.+-]+','_',f)
    return f or 'chemistry'

def _make_supercell(cell, frac, symbols, reps):
    """Replicate a periodic reference cell without changing its local geometry."""
    reps=tuple(int(x) for x in reps)
    c=np.asarray(cell,float).copy(); c[0]*=reps[0]; c[1]*=reps[1]; c[2]*=reps[2]
    f=np.asarray(frac,float); out=[]; sy=[]
    den=np.asarray(reps,float)
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                shift=np.array([i,j,k],float)
                out.append((f+shift)/den)
                sy.extend(symbols)
    return c,np.vstack(out),sy

def _reference_field_metrics(gen):
    rows=[]
    for name,(cell,frac,symbols) in _reference_tio2_structures().items():
        try:
            # The field explicitly selects six distinct O particles per Ti.  Small
            # conventional cells (rutile has only four O) therefore must be expanded;
            # periodic nearest images alone cannot create six distinct particle indices.
            n_o=sum(sy=='O' for sy in symbols)
            if n_o < 6:
                cell,frac,symbols=_make_supercell(cell,frac,symbols,(2,2,2))
            ti=np.asarray([q for q,sy in zip(frac,symbols) if sy=='Ti'],float); oo=np.asarray([q for q,sy in zip(frac,symbols) if sy=='O'],float)
            tc=torch.tensor(ti,dtype=torch.float32,device=gen.device)[None]; oc=torch.tensor(oo,dtype=torch.float32,device=gen.device)[None]; cc=torch.tensor(cell,dtype=torch.float32,device=gen.device)[None]
            with torch.no_grad(): E,D=gen._oxygen_field_energy(tc,oc,cc,details=True)
            rows.append({'name':name,'energy':float(E[0]),'components':{k:float(v[0]) for k,v in D.items()},'cell':cell,'frac':frac,'symbols':symbols})
        except Exception as e: rows.append({'name':name,'error':f'{type(e).__name__}: {e}'})
    return rows

def main():
    p=argparse.ArgumentParser(description='Juliette v136r10: 4-SG adiabatic BO-Langevin B dynamics with three-state O-SCF tracking; no Stage C')
    p.add_argument('--model',default='data/xn_templates/chemistry_model.json'); p.add_argument('--target',default='TiO2=12'); p.add_argument('--output',default=None)
    p.add_argument('--ngpu',type=int,default=-1); p.add_argument('--workers',type=int,default=0); p.add_argument('--starts',type=int,default=64); p.add_argument('--space-groups',type=int,default=4); p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--trajectories',type=int,default=16); p.add_argument('--field-steps',type=int,default=500); p.add_argument('--field-lr',type=float,default=.010); p.add_argument('--temp0',type=float,default=.025); p.add_argument('--temp1',type=float,default=.0005)
    p.add_argument('--environment-steps',type=int,default=60); p.add_argument('--probe-steps',type=int,default=30); p.add_argument('--environment-lr',type=float,default=.016)
    p.add_argument('--ambient-steps',type=int,default=100); p.add_argument('--ambient-lr',type=float,default=.035); p.add_argument('--builder-lr',type=float,default=.055)
    args=p.parse_args(); slurm_ntasks=int(os.environ.get('SLURM_NTASKS','1') or 1); slurm_cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','1') or 1)
    if slurm_ntasks>1: raise SystemExit(f'v135 requires one SLURM task, but SLURM_NTASKS={slurm_ntasks}.')
    np.random.seed(args.seed); torch.manual_seed(args.seed); outdir=Path(args.output or f'{_formula_prefix(args.target)}_v137_SOAP_BO_4SG'); outdir.mkdir(parents=True,exist_ok=True); (outdir/'cif').mkdir(exist_ok=True)
    logfile=outdir/'run.log'; progressfile=outdir/'progress.jsonl'; visible=torch.cuda.device_count() if torch.cuda.is_available() else 0; ngpu=visible if args.ngpu<0 else min(max(args.ngpu,0),visible); nworkers=(ngpu if args.workers<=0 else args.workers) if ngpu else 1; nworkers=max(1,nworkers); devices=[f'cuda:{i%max(ngpu,1)}' for i in range(nworkers)] if ngpu else ['cpu']*nworkers
    model=ChemistryModel(args.model); formula,z,counts,parent,child=model.resolve_target(args.target); choices=[]
    for spg in range(1,231):
        try:toks=exact_parent_entries(spg,counts[parent],cap=1)
        except Exception:toks=[]
        if toks:choices.append((spg,toks[0]))
    rng=np.random.default_rng(args.seed); rng.shuffle(choices); choices=choices[:args.space_groups]; chunks=[choices[i::nworkers] for i in range(nworkers)]
    attrs={'stage_a_batch_starts':args.starts,'stage_a_batch_keep':1,'environment_steps':args.environment_steps,'probe_steps':min(args.probe_steps,args.environment_steps),'environment_lr':args.environment_lr,'global_seed':args.seed,'oxygen_field_trajectories':args.trajectories,'oxygen_field_steps':args.field_steps,'oxygen_field_lr':args.field_lr,'oxygen_field_temp0':args.temp0,'oxygen_field_temp1':args.temp1}
    cfg={'model':args.model,'target':args.target,'starts':args.starts,'ambient_steps':args.ambient_steps,'ambient_lr':args.ambient_lr,'builder_lr':args.builder_lr,'builder_attrs':attrs}
    def emit(x):
        print(x,flush=True)
        with logfile.open('a') as f:f.write(x+'\n')
    emit('--- Juliette v137 local-SOAP BO-SCF diagnostic ---'); emit(f'Target={args.target}; SG={len(choices)}; trajectories/SG={args.trajectories}; field steps={args.field_steps}')
    emit(f'Resources: SLURM_NTASKS={slurm_ntasks}; SLURM_CPUS_PER_TASK={slurm_cpus}; visible GPUs={visible}; workers={nworkers}; devices={devices}')
    emit('POLICY: Local graph-free SOAP chemistry. Exact SG/Wyckoff symmetry. O is the fast subsystem: warm-started O-SCF is classified converged/trackable/failed after EVERY proposed slow move; converged and trackable states may be accepted. Ti+lattice are slow BO variables and receive annealed Langevin dynamics; lattice noise is weak. Oversized or O-untrackable slow proposals are halved/retried. Finish with zero-T BO quench. NO Stage C.')
    # reference calibration on GPU0/CPU before worker fanout
    try:
        g=TiOnlyBuilder(model,parent,child,device=devices[0],starts=args.starts,ambient_steps=args.ambient_steps,ambient_lr=args.ambient_lr,lr=args.builder_lr);
        for k,v in attrs.items():setattr(g,k,v)
        refs=_reference_field_metrics(g); refdir=outdir/'reference_calibration'; refdir.mkdir(exist_ok=True)
        emit('--- Oxygen-field reference energies ---')
        for r in refs:
            if r.get('error'):emit(f"[REF {r['name']}] ERROR {r['error']}")
            else:
                c=r['components']; emit(f"[REF {r['name']}] E={r['energy']:.3f} oct={c['oct']:.3f} rad={c['rad']:.3f} CNti={c['cn_ti']:.3f} CNo={c['cn_o']:.3f} OO={c['oo']:.3f}"); _write_simple_cif(refdir/f"{r['name']}.cif",r['cell'],r['frac'],r['symbols'],r['name'])
        (refdir/'metrics.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('cell','frac','symbols')} for r in refs],indent=2,default=_v135_jsonable))
        del g
    except Exception as e:emit(f'[REF calibration] ERROR {type(e).__name__}: {e}')
    ctx=mp.get_context('spawn'); q=ctx.Queue(); procs=[]
    for wid,(dev,tasks) in enumerate(zip(devices,chunks)):
        pr=ctx.Process(target=_v135_worker,args=(wid,dev,tasks,cfg,q),daemon=False); pr.start(); procs.append(pr)
    allrows=[]; done=0; total=0.0; fatals=[]
    while done<len(procs):
        msg=q.get(); kind=msg.get('kind')
        if kind=='live':
            with logfile.open('a') as f:f.write('[LIVE] '+json.dumps(msg,default=_v135_jsonable)+'\n')
            if msg.get('event')=='stageA_batch_done':emit(f"[GPU{msg['worker']} SG {msg['spg']}] Stage-A retained={msg.get('retained')} in {msg.get('seconds',0):.1f}s")
            elif msg.get('event')=='oxygen_field_done':
                c=msg.get('components',{}); emit(f"[GPU{msg['worker']} SG {msg['spg']}] O-field[{msg.get('mode','?')}] E={msg.get('energy',float('nan')):.3f} SOAPti={c.get('soap_ti',float('nan')):.3f} SOAPo={c.get('soap_o',float('nan')):.3f} OO={c.get('oo',float('nan')):.3f} OOmin={c.get('oo_min',float('nan')):.3f} {msg.get('seconds',0):.1f}s")
                if msg.get('mode')=='wyckoff':
                    p=msg.get('profile',{}); ch=p.get('chosen',[])
                    emit(f"[GPU{msg['worker']} SG {msg['spg']}] GPU-TIME BREAKDOWN wyckoff: geometry={p.get('geometry',0):.2f}s enumerate={p.get('enumerate_templates',0):.2f}s screen={p.get('screen_total',0):.2f}s optimize={p.get('optimize_total',0):.2f}s final={p.get('final_eval_total',0):.2f}s | templates={p.get('templates_total',0)} chosen={p.get('templates_chosen',0)} screen_starts={p.get('screen_starts',0)} optimization_steps={p.get('optimization_steps',0)}")
                    kt=p.get('kernel_timing',{}); emit(f"[GPU{msg['worker']} SG {msg['spg']}] BATCH PROFILE: global_steps={p.get('batched_global_steps',0)} expand={kt.get('expand',0):.2f}s forward={kt.get('forward',0):.2f}s backward={kt.get('backward',0):.2f}s update={kt.get('update',0):.2f}s best_eval={kt.get('best_eval',0):.2f}s convergence={kt.get('convergence',0):.2f}s")
                    emit(f"[GPU{msg['worker']} SG {msg['spg']}] UPDATE PROFILE: grad_clip={kt.get('grad_clip',0):.2f}s noise={kt.get('noise',0):.2f}s apply={kt.get('apply_update',0):.2f}s best_forward={kt.get('best_forward',0):.2f}s best_retain={kt.get('best_retain',0):.2f}s")
                    if ch:
                        emit(f"[GPU{msg['worker']} SG {msg['spg']}] WYCKOFF OPT DETAIL: "+' | '.join(f"{x['token']} dof={x['ncoord']} traj={x['ntraj']} steps={x['steps']} stop={x.get('stop_reason','?')} best=s{x.get('best_step',0)} E={x.get('best_energy',float('nan')):.3f}" for x in ch))
                        for x in ch:
                            cv=x.get('convergence',[])
                            if cv:
                                emit(f"[GPU{msg['worker']} SG {msg['spg']}] CONVERGENCE {x['token']}: "+' | '.join(
                                    f"s{z['step']} E={z['energy_best']:.3f}/{z['energy_median']:.3f} gRMS={z['grad_rms_best']:.3e}/{z['grad_rms_median']:.3e} gMAX={z['grad_max_best']:.3e}/{z['grad_max_median']:.3e}" for z in cv))
            elif msg.get('event')=='stageB_scf_done':
                b=msg.get('components_before',{}); a=msg.get('components_after',{}); cb=msg.get('coord_before',{}); ca=msg.get('coord_after',{})
                emit(f"[GPU{msg['worker']} SG {msg['spg']}] BO-B field {msg.get('energy_before',float('nan')):.3f}->{msg.get('energy_after',float('nan')):.3f} regE={msg.get('regularized_energy_after',float('nan')):.3f} | SOAPti {b.get('soap_ti',float('nan')):.3f}->{a.get('soap_ti',float('nan')):.3f} SOAPo {b.get('soap_o',float('nan')):.3f}->{a.get('soap_o',float('nan')):.3f} OO {b.get('oo',float('nan')):.3f}->{a.get('oo',float('nan')):.3f} OOmin {b.get('oo_min',float('nan')):.3f}->{a.get('oo_min',float('nan')):.3f} | cycles={msg.get('cycles')} stop={msg.get('stop_reason')} {msg.get('seconds',0):.1f}s")
                emit(f"[GPU{msg['worker']} SG {msg['spg']}] COORD frozen-B->BO-B: TiO4 {cb.get('TiO4',0)}->{ca.get('TiO4',0)} TiO5 {cb.get('TiO5',0)}->{ca.get('TiO5',0)} TiO6 {cb.get('TiO6',0)}->{ca.get('TiO6',0)} | OTi2 {cb.get('OTi2',0)}->{ca.get('OTi2',0)} OTi3 {cb.get('OTi3',0)}->{ca.get('OTi3',0)} OTi4 {cb.get('OTi4',0)}->{ca.get('OTi4',0)}")
                for z in msg.get('trace',[]):
                    c=z.get('components',{}); coord=z.get('coordination',{}); emit(f"[GPU{msg['worker']} SG {msg['spg']}] BO-B {z.get('cycle',0):02d}/{z.get('phase','?')} obj={z.get('objective',float('nan')):.3f} field={z.get('field_energy',float('nan')):.3f} SOAPti={c.get('soap_ti',float('nan')):.3f} SOAPo={c.get('soap_o',float('nan')):.3f} OO={c.get('oo',float('nan')):.3f} OOmin={c.get('oo_min',float('nan')):.3f} Nbond={coord.get('Nbond','?')} TiDef={coord.get('Ti_abs_defect',float('nan')):.0f} ODef={coord.get('O_abs_defect',float('nan')):.0f} TiRMS={z.get('ti_disp_rms_A',float('nan')):.3f}A strainRMS={z.get('strain_rms',float('nan')):.4f} V/V0={z.get('volume_ratio',float('nan')):.4f} gRMS={z.get('grad_rms',float('nan')):.3e}")
            continue
        if kind=='worker_start':emit(f"[worker {msg['worker']} {msg['device']}] started; assigned {msg['ntasks']} SG searches")
        elif kind=='search_done':
            rr=msg.get('rows') or []; allrows.extend(rr); total+=float(msg.get('elapsed_s',0));
            if msg.get('error'):
                emit(f"[SG {msg['spg']:3d} | GPU{msg['worker']}] ERROR {msg['error']}")
                detail=(msg.get('detail') or '').strip()
                if detail:
                    with logfile.open('a') as f: f.write(detail+'\n')
            for r in rr:
                sg=int(r['spg'])
                if r.get('wyckoff_cell') is not None:_write_simple_cif(outdir/'cif'/f"SG_{sg:03d}_B_final.cif",r['wyckoff_cell'],np.vstack([r['wyckoff_ti_frac'],r['wyckoff_o_frac']]),['Ti']*12+['O']*24,f"SG{sg}_B_final")
                for z in r.get('stageB_scf_trace',[]):
                    _write_simple_cif(outdir/'cif'/f"SG_{sg:03d}_BSCF_cycle_{int(z['cycle']):02d}_{z.get('phase','state')}.cif",z['cell'],np.vstack([z['ti_frac'],z['o_frac']]),['Ti']*12+['O']*24,f"SG{sg}_BSCF_cycle_{int(z['cycle']):02d}_{z.get('phase','state')}")
                if r.get('stageB_scf_cell') is not None:_write_simple_cif(outdir/'cif'/f"SG_{sg:03d}_BSCF_final.cif",r['stageB_scf_cell'],np.vstack([r['stageB_scf_ti_frac'],r['stageB_scf_o_frac']]),['Ti']*12+['O']*24,f"SG{sg}_BSCF_final")
            if rr:emit(f"[SG {msg['spg']:3d} | GPU{msg['worker']}] B E={rr[0]['field_energy']:.3f} -> BO-B E={rr[0].get('stageB_scf_energy',float('nan')):.3f} | elapsed={msg['elapsed_s']:.1f}s")
        elif kind=='worker_done':done+=1; emit(f"[worker {msg['worker']} {msg['device']}] finished")
        elif kind=='worker_fatal':done+=1; fatals.append(msg); emit(f"[worker {msg['worker']}] FATAL {msg['error']}")
    for pr in procs:pr.join()
    with (outdir/'results.jsonl').open('w') as f:
        for r in allrows:f.write(json.dumps(r,default=_v135_jsonable)+'\n')
    summary={'version':'v137-local-SOAP-BO-4SG','rows':len(allrows),'gpu_minutes':total/60,'fatal_workers':len(fatals),'energies':[r.get('field_energy') for r in allrows]}; (outdir/'summary.json').write_text(json.dumps(summary,indent=2,default=_v135_jsonable))
    emit(f'=== v136r9 BO-LANGEVIN TRACKING DIAGNOSTIC DONE: SG={len(choices)} rows={len(allrows)} GPU-min={total/60:.2f} fatal_workers={len(fatals)} ===')

if __name__=='__main__':main()
