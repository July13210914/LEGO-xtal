#!/usr/bin/env python3
"""Juliette v149 — accelerated broad SG/Wyckoff generation from one learner-v39 model.

Stage A uses a cheap learned Ti density/volume capture field on a large random pool,
keeps a nonredundant shortlist, then applies the exact self-supervised Ti-Ti latent
manifold only to that shortlist. Stage B retains the v140 symmetry-exact oxygen/BO
engine. Stage C uses learner-v39 Ti-O Bayesian reference support plus the Ti-Ti
manifold. Final structures are ranked against every authoritative reference field.
Only the unified chemistry_model.json learner contract is read; no secondary or built-in crystallographic model is used.
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

def _v30_tio_dim(nmax=6,lmax=6):
    return int(nmax + (lmax+1)*nmax*(nmax+1)//2)


@dataclass(frozen=True)
class PairChannel:
    species_i: str; species_j: str; mu: float; sigma: float

class ChemistryModel:
    """Strict reader for the single-file learner-v39 generation contract."""
    def __init__(self, path: str):
        self.path=str(path)
        self.raw=json.loads(Path(path).read_text())
        if self.raw.get('schema')!='juliette_generation_model_v39':
            raise ValueError(
                f"Generator v149 requires learner schema 'juliette_generation_model_v39'; "
                f"got {self.raw.get('schema')!r}. Run 0_learn_v39.py and pass that one JSON file."
            )
        g=self.raw.get('generation_contract') or {}
        if int(g.get('contract_version',0))!=2:
            raise ValueError(f"Generator v149 requires generation_contract version 2; got {g.get('contract_version')!r}")
        required=('chemistry','pair_scales','conditioning','tio_reference_field','titi_manifold','local_force_models')
        miss=[k for k in required if not g.get(k)]
        if miss: raise ValueError(f'Incomplete learner-v39 generation_contract; missing {miss}')
        self.generation=g
        ch=g['chemistry']; self.parent=str(ch['parent_element']); self.child=str(ch['child_element'])
        self.formula_unit={str(k):int(v) for k,v in ch['formula_unit'].items()}
        self.cn={self.parent:int(ch['parent_coordination']),self.child:int(ch['child_coordination'])}
        ps=g['pair_scales']
        self.pair_channel=PairChannel(self.parent,self.child,float(ps['parent_child_mu_A']),max(float(ps['parent_child_sigma_A']),1e-3))
        self.coord_cutoff=float(ps['coordination_cutoff_A'])
        self.titi_hard=float(ps['parent_parent_hard_min_A']); self.titi_reference_min=float(ps['parent_parent_reference_min_A'])
        self.oo_hard=float(ps['child_child_hard_min_A']); self.oo_reference_min=float(ps['child_child_reference_min_A'])
        co=g['conditioning']; self.volume_per_parent_median=float(co['volume_per_parent_median_A3'])
        self.volume_per_parent_q10=float(co['volume_per_parent_q10_A3']); self.volume_per_parent_q90=float(co['volume_per_parent_q90_A3'])
        prof=co.get('titi_cumulative_density_profile') or []
        if not prof: raise ValueError('Learner-v39 is missing learned Ti-Ti cumulative density profile required by fast Stage A')
        self.titi_density_profile=tuple((float(x['radius_A']),float(x['rank'])) for x in prof)
        self.titi_density_shell_width=float(co.get('density_shell_width_A',0.18))
        self.tio_reference_field=g['tio_reference_field']; self.titi_manifold=g['titi_manifold']; self.local_force_models=g['local_force_models']
        # Startup contract validation catches learner/generator drift before GPU work starts.
        desc=self.titi_manifold.get('descriptor') or {}; nn=self.titi_manifold.get('nn_metric') or {}
        arch=[int(x) for x in nn.get('architecture',[])]
        nmax=int(desc.get('radial_basis_nmax',0)); lmax=int(desc.get('angular_lmax',-1))
        dim=nmax+(lmax+1)*nmax*(nmax+1)//2 if nmax and lmax>=0 else 0
        anchors=np.asarray(self.titi_manifold.get('latent_anchors',[]),float)
        if len(arch)!=4 or arch[0]!=dim or not nn.get('state_dict') or anchors.ndim!=2 or anchors.shape[1]!=arch[-1]:
            raise ValueError('Learner-v39 Ti-Ti manifold contract is internally inconsistent')
        self.titi_descriptor_neighbor_cap=int(self.titi_manifold.get('descriptor_neighbor_cap',0))
        self.titi_max_training_neighbors=int(self.titi_manifold.get('max_training_neighbors_within_rmax',0))
        if self.titi_descriptor_neighbor_cap < max(1,self.titi_max_training_neighbors):
            raise ValueError('Learner-v39 Ti-Ti descriptor neighbor cap is smaller than training support')
        tref=self.tio_reference_field.get('references') or {}
        if not tref: raise ValueError('Learner-v39 contains no Ti-O reference fields')
        for name,r in tref.items():
            X=np.asarray(r.get('X',[]),float); K=np.asarray(r.get('Kinv',[]),float)
            if X.ndim!=2 or X.shape[1]!=_v30_tio_dim() or K.shape!=(len(X),len(X)):
                raise ValueError(f'Learner-v39 Ti-O reference {name!r} has inconsistent descriptor/Kinv dimensions')
            if float(r.get('acceptance_q90U',0))<=0: raise ValueError(f'Learner-v39 Ti-O reference {name!r} has invalid acceptance threshold')

    def resolve_target(self,text: str):
        if '=' in text:
            formula,zs=text.split('=',1); z=int(zs)
        else: formula,z=text,1
        comp=_parse_formula(formula)
        if set(comp)!=set(self.formula_unit):
            raise ValueError(f'Target {formula!r} elements {sorted(comp)} disagree with learner model {sorted(self.formula_unit)}')
        ratios=[comp[k]/max(self.formula_unit[k],1) for k in self.formula_unit]
        if max(ratios)-min(ratios)>1e-12:
            raise ValueError(f'Target {formula!r} stoichiometry disagrees with learner model formula unit {self.formula_unit}')
        scale=int(round(ratios[0]))
        if scale<1 or any(comp[k]!=scale*self.formula_unit[k] for k in comp):
            raise ValueError(f'Target formula must be an integer multiple of learner formula unit {self.formula_unit}')
        counts={k:int(comp[k])*int(z) for k in comp}
        return formula,int(z),counts,self.parent,self.child

    def pair(self,a,b):
        if {str(a),str(b)}!={self.parent,self.child}:
            raise KeyError(f'Learner-v39 only defines active parent-child pair {self.parent}-{self.child}; requested {a}-{b}')
        return self.pair_channel


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
        if parent!=model.parent or child!=model.child:
            raise ValueError(f'Builder parent/child {parent}/{child} disagree with learner contract {model.parent}/{model.child}')
        self.model=model; self.parent=parent; self.child=child; self.device=torch.device(device)
        self.starts=int(starts); self.screen_steps=int(screen_steps); self.refine_steps=int(refine_steps); self.lr=float(lr)
        self.ambient_steps=max(1,int(ambient_steps)); self.ambient_lr=float(ambient_lr); self.aspect_weight=float(aspect_weight)
        self.hard=float(model.titi_hard); self.oo_hard=float(model.oo_hard); self.titi_restore=float(model.titi_reference_min)
        self.volume_target_per_ti=float(model.volume_per_parent_median); self.coord_cutoff=float(model.coord_cutoff)
        self.titi_density_profile=tuple(model.titi_density_profile); self.titi_density_shell_width=float(model.titi_density_shell_width)
        self.shift=torch.as_tensor(SHIFTS,dtype=torch.float32,device=self.device); self._cache={}
        # v143 Stage A is deliberately hierarchical: a cheap learned condensed-matter
        # capture field handles the broad random entrance pool; the expensive 870-D
        # Ti-Ti manifold is evaluated/refined only on a small diverse shortlist.
        self.stage_a_manifold_weight=4.0; self.stage_a_repulsion_weight=4.0; self.stage_a_volume_weight=0.8
        self.stage_a_density_weight=1.0; self.stage_a_volume_log_width=0.35; self.stage_a_repulsion_width=0.08
        # Exact Ti-Ti descriptor/NN/observed anchors from the same learner file.
        tm=model.titi_manifold; desc=tm['descriptor']; nn=tm['nn_metric']; arch=[int(v) for v in nn['architecture']]
        anchors=np.asarray(tm['latent_anchors'],float)
        self.titi_manifold={
            'rmax':float(desc['outer_cutoff_A']),'sigma':float(desc['sigma_A']),
            'nmax':int(desc['radial_basis_nmax']),'lmax':int(desc['angular_lmax']),
            'radius':max(float(tm['allowed_distance_q90_0p10A']),1e-8),
            'neighbor_cap':int(model.titi_descriptor_neighbor_cap),
            'max_training_neighbors':int(model.titi_max_training_neighbors),
            'anchors':torch.tensor(anchors,dtype=torch.float32,device=self.device),'arch':arch,
            'state':{k:torch.tensor(v,dtype=torch.float32,device=self.device) for k,v in nn['state_dict'].items()}}
        # Exact Ti-O Bayesian reference channels from the same learner file.
        td=model.tio_reference_field.get('descriptor') or {}; self.tio_sigma=float(td.get('sigma_A',0.28)); self.tio_refs=[]
        for name,r in model.tio_reference_field['references'].items():
            self.tio_refs.append({'name':str(name),'cutoff':float(r['cutoff_A']),'threshold':max(float(r['acceptance_q90U']),1e-10),
                                  'exponent':float(r['kernel_exponent']),
                                  'X':torch.tensor(r['X'],dtype=torch.float32,device=self.device),
                                  'Kinv':torch.tensor(r['Kinv'],dtype=torch.float32,device=self.device)})
        lf=model.local_force_models; vm=lf['v140_style']; sm=lf['so3']
        self.local_v140={'ti_r':torch.tensor([q['r'] for q in vm['tio6_anchors']],dtype=torch.float32,device=self.device),'ti_c':torch.tensor([q['cos'] for q in vm['tio6_anchors']],dtype=torch.float32,device=self.device),'o_r':torch.tensor([q['r'] for q in vm['oti3_anchors']],dtype=torch.float32,device=self.device),'o_c':torch.tensor([q['cos'] for q in vm['oti3_anchors']],dtype=torch.float32,device=self.device),'rs':float(vm['radial_scale_A']),'cs':float(vm['angular_scale'])}
        self.local_so3={'centers':torch.tensor(sm['centers_A'],dtype=torch.float32,device=self.device),'width':float(sm['width_A']),'lmax':int(sm['lmax']),'ti':torch.tensor(sm['tio6_anchors'],dtype=torch.float32,device=self.device),'o':torch.tensor(sm['oti3_anchors'],dtype=torch.float32,device=self.device),'scale':max(float(sm.get('scale',.05)),.02)}

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
        # v143 starts near the broad learned volume scale.  This is only an entrance
        # scale, not a final structural target; free Wyckoff coordinates remain random.
        nlen=self._length_dof_count(t)
        base=max(4.0,(max(self.volume_target_per_ti,1.0)*max(t["n"],1))**(1/3))
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

    def _nearest_local_geometry_loss(self,t,ti_raw,o_frac,details=False):
        """Local observer field for Stage C: each Ti sees its nearest 6 O; each O sees its nearest 3 Ti.

        No incidence graph is imposed. This avoids the old 1e9 sentinel caused by forcing an
        arbitrary O->Ti3 assignment that did not give every Ti exactly six incident O atoms.
        """
        _,_,cell,_,ti,_=self.geometry(t,ti_raw); B,Nti,_=ti.shape; No=o_frac.shape[1]
        d=o_frac[:,None,:,None,:]-ti[:,:,None,None,:]+self.shift[None,None,None,:,:]
        vc=torch.einsum('btoqi,bij->btoqj',d,cell); dd=torch.linalg.norm(vc,dim=-1)
        flatd=dd.reshape(B,Nti,No*27); flatv=vc.reshape(B,Nti,No*27,3)
        kd,ki=torch.topk(flatd,min(6,No*27),dim=-1,largest=False)
        kv=torch.gather(flatv,2,ki[...,None].expand(-1,-1,-1,3))
        if kv.shape[2] < 6:
            bad=torch.full((B,),1e9,device=self.device,dtype=cell.dtype)
            return (bad,{'tio6_so3':bad,'oti3_so3':bad}) if details else bad
        kv=kv[:,:,:6,:]
        M=self.tio6_so3_model; fp=_so3_power_torch(kv,M['centers'],M['width'],M['lmax'])
        dz=(fp-M['mu'][None,None,:])@M['evec']; ti_sites=(dz*dz/M['eval'][None,None,:]).mean(-1)
        tio6=ti_sites.mean(1)

        # O observer: reverse the same periodic Ti-O vectors and independently choose nearest 3 Ti.
        ov=(-vc).permute(0,2,1,3,4).reshape(B,No,Nti*27,3)
        od=dd.permute(0,2,1,3).reshape(B,No,Nti*27)
        okd,oki=torch.topk(od,min(3,Nti*27),dim=-1,largest=False)
        okv=torch.gather(ov,2,oki[...,None].expand(-1,-1,-1,3))
        if okv.shape[2] < 3:
            bad=torch.full((B,),1e9,device=self.device,dtype=cell.dtype)
            return (bad,{'tio6_so3':tio6,'oti3_so3':bad}) if details else bad
        okv=okv[:,:,:3,:]
        M=self.oti3_so3_model; fp=_so3_power_torch(okv,M['centers'],M['width'],M['lmax'])
        dz=(fp-M['mu'][None,None,:])@M['evec']; o_sites=(dz*dz/M['eval'][None,None,:]).mean(-1)
        oti3=o_sites.mean(1)
        total=4.0*tio6+oti3
        if details:
            return total,{'tio6_so3':tio6,'oti3_so3':oti3,'tio6_sites':ti_sites,'oti3_sites':o_sites,
                          'tio_min':kd[:,:,:6].amin(dim=(1,2)),'tio_mean':kd[:,:,:6].mean(dim=(1,2)),
                          'tio_max':kd[:,:,:6].amax(dim=(1,2)),'cell':cell,'tifrac':ti,'ofrac':o_frac,
                          'volume_per_ti':torch.abs(torch.linalg.det(cell))/max(Nti,1)}
        return total

    def _titi_manifold_descriptor(self, ti, cell):
        """Differentiable v37 Ti<-Ti SOAP, with exact learned-neighbour truncation.

        The learner records the maximum Ti count inside rmax and exports a cap above
        that support.  We therefore build the expensive angular tensor only for the
        nearest ``cap`` periodic candidates.  For every environment within learned
        support this is descriptor-exact, while reducing O((27N)^2) to O(cap^2).
        """
        M=self.titi_manifold; B,N,_=ti.shape; rmax=float(M['rmax']); nmax=int(M['nmax']); lmax=int(M['lmax'])
        df=ti[:,None,:,None,:]+self.shift[None,None,None,:,:]-ti[:,:,None,None,:]
        vv_all=torch.einsum('bijnk,bkl->bijnl',df,cell)
        rr_all=torch.linalg.norm(vv_all,dim=-1).clamp_min(1e-8)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]
        zero=(torch.arange(27,device=self.device)==ZERO_SHIFT)[None,None,None,:]
        rr_all=rr_all.masked_fill(eye & zero,1e6)
        flatr=rr_all.reshape(B,N,N*27); flatv=vv_all.reshape(B,N,N*27,3)
        K=min(int(M.get('neighbor_cap',32)),N*27-1)
        rr,ix=torch.topk(flatr,K,dim=-1,largest=False)
        vv=torch.gather(flatv,2,ix[...,None].expand(-1,-1,-1,3))
        mask=(rr<=rmax); u=vv/rr[...,None].clamp_min(1e-8)
        lo=max(0.35,0.12*rmax); centers=torch.linspace(lo,rmax,nmax,dtype=ti.dtype,device=self.device)
        width=max(float(M['sigma']),0.55*rmax/max(nmax-1,1))
        fc=0.5*(torch.cos(math.pi*torch.clamp(rr/rmax,0.0,1.0))+1.0)*mask.to(ti.dtype)
        g=torch.exp(-0.5*((rr[...,None]-centers[None,None,None,:])/width)**2)*fc[...,None]
        feats=[g.sum(dim=2)]
        dots=torch.einsum('bnji,bnki->bnjk',u,u).clamp(-1.0,1.0)
        P0=torch.ones_like(dots); Pm2=P0; Pm1=dots
        for l in range(lmax+1):
            if l==0: Pl=P0
            elif l==1: Pl=dots
            else:
                Pl=((2*l-1)*dots*Pm1-(l-1)*Pm2)/float(l); Pm2,Pm1=Pm1,Pl
            vals=[]
            for a in range(nmax):
                ga=g[...,a]
                for b in range(a,nmax):
                    vals.append(torch.einsum('bnj,bnk,bnjk->bn',ga,g[...,b],Pl))
            feats.append(torch.stack(vals,dim=-1))
        x=torch.cat(feats,dim=-1)
        return x/torch.linalg.norm(x,dim=-1,keepdim=True).clamp_min(1e-12)

    def _titi_manifold_latent(self, x):
        M=self.titi_manifold; sd=M['state']
        # v34r5 network: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> GELU -> Linear -> L2.
        def linear(q,prefix): return torch.nn.functional.linear(q,sd[prefix+'.weight'],sd[prefix+'.bias'])
        def ln(q,prefix): return torch.nn.functional.layer_norm(q,(q.shape[-1],),sd[prefix+'.weight'],sd[prefix+'.bias'],1e-5)
        q=torch.nn.functional.gelu(ln(linear(x,'net.0'),'net.1'))
        q=torch.nn.functional.gelu(ln(linear(q,'net.3'),'net.4'))
        q=linear(q,'net.6')
        return torch.nn.functional.normalize(q,p=2,dim=-1,eps=1e-8)

    def _titi_manifold_score(self, ti, cell):
        M=self.titi_manifold; z=self._titi_manifold_latent(self._titi_manifold_descriptor(ti,cell))
        # Anchors are observed/reference sites only. No interpolated centers.
        d=(1.0-torch.einsum('bnz,az->bna',z,M['anchors'])).clamp_min(0.0).min(dim=-1).values
        excess=torch.relu(d-float(M['radius']))/float(M['radius'])
        return excess.pow(2).mean(dim=1),d,excess

    def _tio_descriptor(self,vv,cutoff):
        """Differentiable reproduction of learner-v39 normalized Ti<-O SOAP basis."""
        nmax=6; lmax=6; rr=torch.linalg.norm(vv,dim=-1).clamp_min(1e-8); u=vv/rr[...,None]
        centers=torch.linspace(max(0.35,0.18*float(cutoff)),float(cutoff),nmax,dtype=vv.dtype,device=vv.device)
        fc=0.5*(torch.cos(math.pi*torch.clamp(rr/float(cutoff),0.0,1.0))+1.0)
        g=torch.exp(-0.5*((rr[...,None]-centers[None,None,None,:])/max(self.tio_sigma,1e-3))**2)*fc[...,None]
        feats=[g.sum(dim=2)]; dots=torch.einsum('bnji,bnki->bnjk',u,u).clamp(-1.0,1.0)
        P0=torch.ones_like(dots); Pm2=P0; Pm1=dots
        for l in range(lmax+1):
            if l==0: Pl=P0
            elif l==1: Pl=dots
            else:
                Pl=((2*l-1)*dots*Pm1-(l-1)*Pm2)/float(l); Pm2,Pm1=Pm1,Pl
            vals=[]
            for n in range(nmax):
                for m in range(n,nmax): vals.append(torch.einsum('bnj,bnk,bnjk->bn',g[...,n],g[...,m],Pl))
            feats.append(torch.stack(vals,dim=-1))
        x=torch.cat(feats,dim=-1)
        return x/torch.linalg.norm(x,dim=-1,keepdim=True).clamp_min(1e-12)

    def _tio_reference_distances(self,ti,o,cell):
        """Normalized Bayesian D for every Ti site against every authoritative reference."""
        B,N,_=ti.shape; Nc=o.shape[1]
        df=o[:,None,:,None,:]-ti[:,:,None,None,:]+self.shift[None,None,None,:,:]
        vv=torch.einsum('bncqi,bij->bncqj',df,cell).reshape(B,N,Nc*27,3)
        dd=torch.linalg.norm(vv,dim=-1)
        K=min(24,Nc*27); _,ix=torch.topk(dd,K,dim=-1,largest=False)
        near=torch.gather(vv,2,ix[...,None].expand(-1,-1,-1,3))
        Ds=[]
        for R in self.tio_refs:
            fp=self._tio_descriptor(near,R['cutoff'])
            ss=torch.einsum('bnd,rd->bnr',fp,R['X']).clamp(0.0,1.0)
            k=ss.pow(float(R['exponent']))
            u=(1.0-torch.einsum('bnr,rs,bns->bn',k,R['Kinv'],k)).clamp_min(0.0)
            Ds.append(u/float(R['threshold']))
        return torch.stack(Ds,dim=-1)

    def _tio_reference_support(self,ti,o,cell):
        """Per-Ti normalized Bayesian distance to the nearest authoritative Ti-O reference."""
        D=self._tio_reference_distances(ti,o,cell); best=D.min(dim=-1).values
        excess=torch.relu(best-1.0)
        return excess.pow(2).mean(1),best,excess

    def reference_similarity_metrics(self,ti,o,cell):
        """Structure-level ranking metrics against all learner-v39 CIF reference fields."""
        D=self._tio_reference_distances(ti,o,cell)
        best,bix=D.min(dim=-1); names=[r['name'] for r in self.tio_refs]
        B=best.shape[0]; rows=[]
        for b in range(B):
            bv=best[b]
            per={}
            for j,name in enumerate(names):
                q=D[b,:,j]
                per[name]={'mean':float(q.mean().cpu()),'q50':float(torch.quantile(q,.50).cpu()),
                           'q90':float(torch.quantile(q,.90).cpu()),'max':float(q.max().cpu()),
                           'fraction_D_le_1':float((q<=1.0).to(q.dtype).mean().cpu())}
            counts={name:int((bix[b]==j).sum().cpu()) for j,name in enumerate(names)}
            ml,md,mex=self._titi_manifold_score(ti[b:b+1],cell[b:b+1])
            rows.append({'local_D_mean':float(bv.mean().cpu()),'local_D_q50':float(torch.quantile(bv,.50).cpu()),
                         'local_D_q90':float(torch.quantile(bv,.90).cpu()),'local_D_max':float(bv.max().cpu()),
                         'fraction_D_le_0p25':float((bv<=.25).to(bv.dtype).mean().cpu()),
                         'fraction_D_le_0p5':float((bv<=.5).to(bv.dtype).mean().cpu()),
                         'fraction_D_le_1':float((bv<=1.0).to(bv.dtype).mean().cpu()),
                         'fraction_D_le_2':float((bv<=2.0).to(bv.dtype).mean().cpu()),
                         'nearest_reference_site_counts':counts,'per_reference':per,
                         'titi_q90':float(torch.quantile(md[0],.90).cpu()),'titi_max':float(md[0].max().cpu()),
                         'titi_fraction_inside':float((md[0]<=float(self.titi_manifold['radius'])).to(md.dtype).mean().cpu())})
        return rows

    def titi_conditioning_loss(self,t,raw,details=False):
        """Cheap learned Stage-A capture field; no SOAP/NN evaluation.

        It learns only broad condensed Ti-framework scales from the v37 contract:
        cumulative Ti counts N(R), volume/Ti, and the Ti-Ti hard wall.  This is a
        funnel conditioner, never the final similarity criterion.
        """
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        df=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vv=torch.einsum('bijnk,bkl->bijnl',df,cell); dist=torch.linalg.norm(vv,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]
        zero=(torch.arange(27,device=self.device)==ZERO_SHIFT)[None,None,None,:]
        selfzero=eye & zero; dist=dist.masked_fill(selfzero,1e6)
        r=dist.reshape(B,N,N*27)
        terms=[]; counts=[]
        sw=max(float(self.titi_density_shell_width),0.05)
        for R,target in self.titi_density_profile:
            nr=torch.sigmoid((float(R)-r)/sw).sum(-1)
            z=(nr-float(target))/max(1.5,0.20*float(target))
            az=z.abs(); hub=torch.where(az<=1.0,0.5*z*z,az-0.5)
            terms.append(hub); counts.append(nr)
        density_local=torch.stack(terms,dim=-1).mean(-1); density=density_local.mean(1)
        pair=torch.nn.functional.softplus((self.hard-dist)/self.stage_a_repulsion_width).pow(2)
        pair=pair.masked_fill(selfzero,0.0); rep=pair.sum((1,2,3))/max(N,1)
        mind=r.amin((1,2))
        vol=torch.abs(torch.linalg.det(cell))/max(N,1)
        zv=torch.log((vol/max(self.volume_target_per_ti,1e-8)).clamp_min(1e-6))/self.stage_a_volume_log_width
        av=zv.abs(); vloss=torch.where(av<=1.0,0.5*zv*zv,av-0.5)
        aspect=torch.log(abc.amax(1)/abc.amin(1).clamp_min(1e-6)).pow(2)
        metric=torch.relu(1e-4-z2)/1e-4
        total=(self.stage_a_density_weight*density+self.stage_a_repulsion_weight*rep+
               self.stage_a_volume_weight*vloss+0.02*aspect+10.0*metric)
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if not details:return total
        return total,{'conditioning_density':density,'density_counts':torch.stack(counts,dim=-1).mean(1),
                      'repulsion':rep,'minimum_titi_A':mind,'volume_per_ti_A3':vol,
                      'volume_loss':vloss,'metric_valid':z2>0},(abc,ang,cell,frac,free)

    def _stage_a_signature(self,t,raw):
        """Cheap continuous signature used only to avoid near-duplicate Stage-A proposals."""
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        df=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vv=torch.einsum('bijnk,bkl->bijnl',df,cell); dist=torch.linalg.norm(vv,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]
        zero=(torch.arange(27,device=self.device)==ZERO_SHIFT)[None,None,None,:]
        dist=dist.masked_fill(eye & zero,1e6).reshape(B,N,N*27)
        K=min(12,N*27-1); nd=torch.topk(dist,K,dim=-1,largest=False).values
        # mean/std ranked distances plus log cell ratios and volume; dimension is tiny.
        mu=nd.mean(1)/max(self.titi_restore,1e-6); sd=nd.std(1,unbiased=False)/max(self.titi_restore,1e-6)
        ratios=torch.log(abc/abc.mean(1,keepdim=True).clamp_min(1e-6))
        vol=(torch.abs(torch.linalg.det(cell))/max(N,1))/max(self.volume_target_per_ti,1e-8)
        return torch.cat([mu,sd,ratios,torch.log(vol.clamp_min(1e-6))[:,None]],dim=1)

    @staticmethod
    def _greedy_diverse_indices(scores, signatures, keep, min_sep):
        order=torch.argsort(scores).detach().cpu().tolist(); sig=signatures.detach().cpu()
        chosen=[]
        for j in order:
            if not chosen or min(float(torch.linalg.norm(sig[j]-sig[k])) for k in chosen)>=float(min_sep):
                chosen.append(int(j))
                if len(chosen)>=int(keep): break
        if len(chosen)<int(keep):
            for j in order:
                if int(j) not in chosen:
                    chosen.append(int(j))
                    if len(chosen)>=int(keep): break
        return chosen

    def titi_stageA_loss(self,t,raw,details=False):
        """Ti-only Wyckoff potential: learned Ti-Ti manifold + broad physical conditioning."""
        abc,ang,cell,z2,frac,free=self.geometry(t,raw); B,N,_=frac.shape
        mloss,mdist,mex=self._titi_manifold_score(frac,cell)
        df=frac[:,None,:,None,:]+self.shift[None,None,None,:,:]-frac[:,:,None,None,:]
        vv=torch.einsum('bijnk,bkl->bijnl',df,cell); dist=torch.linalg.norm(vv,dim=-1).clamp_min(1e-7)
        eye=torch.eye(N,dtype=torch.bool,device=self.device)[None,:,:,None]; zero=torch.arange(27,device=self.device)==ZERO_SHIFT
        selfzero=eye & zero[None,None,None,:]; dist=dist.masked_fill(selfzero,1e6)
        pair=torch.nn.functional.softplus((self.hard-dist)/self.stage_a_repulsion_width).pow(2)
        rep=pair.sum((1,2,3))/max(N,1); mind=dist.amin((1,2,3))
        vol=torch.abs(torch.linalg.det(cell))/max(N,1)
        zv=torch.log((vol/max(self.volume_target_per_ti,1e-8)).clamp_min(1e-6))/self.stage_a_volume_log_width
        av=zv.abs(); vloss=torch.where(av<=1.0,0.5*zv*zv,av-0.5)
        aspect=torch.log(abc.amax(1)/abc.amin(1).clamp_min(1e-6)).pow(2); metric=torch.relu(1e-4-z2)/1e-4
        total=(self.stage_a_manifold_weight*mloss+self.stage_a_repulsion_weight*rep+self.stage_a_volume_weight*vloss+0.02*aspect+10.0*metric)
        total=torch.nan_to_num(total,nan=1e9,posinf=1e9,neginf=1e9)
        if not details:return total
        return total,{'titi_manifold_loss':mloss,'titi_q90':torch.quantile(mdist,.90,dim=1),'titi_max':mdist.max(1).values,
                      'titi_excess_q90':torch.quantile(mex,.90,dim=1),'repulsion':rep,'minimum_titi_A':mind,
                      'volume_per_ti_A3':vol,'volume_loss':vloss,'metric_valid':z2>0},(abc,ang,cell,frac,free)

    def _learned_local_force_loss(self,ti,o,cell,mode):
        """Reference-derived TiO6/OTi3 force. No ideal bond lengths or angles."""
        B,Nt,_=ti.shape; No=o.shape[1]
        # Ti -> six nearest O
        df=o[:,None,:,None,:]-ti[:,:,None,None,:]+self.shift[None,None,None,:,:]
        vv=torch.einsum('btoqi,bij->btoqj',df,cell)
        vv=vv.reshape(B,Nt,No*27,3); dd=torch.linalg.norm(vv,dim=-1)
        _,ix=torch.topk(dd,min(6,No*27),dim=-1,largest=False); tv=torch.gather(vv,2,ix[...,None].expand(-1,-1,-1,3))
        # O -> three nearest Ti
        df2=ti[:,None,:,None,:]-o[:,:,None,None,:]+self.shift[None,None,None,:,:]
        ov=torch.einsum('botqi,bij->botqj',df2,cell)
        ov=ov.reshape(B,No,Nt*27,3); od=torch.linalg.norm(ov,dim=-1)
        _,iox=torch.topk(od,min(3,Nt*27),dim=-1,largest=False); ov=torch.gather(ov,2,iox[...,None].expand(-1,-1,-1,3))
        if mode=='so3':
            M=self.local_so3
            tf=_so3_power_torch(tv,M['centers'],M['width'],M['lmax']); of=_so3_power_torch(ov,M['centers'],M['width'],M['lmax'])
            td=((tf[:,:,None,:]-M['ti'][None,None,:,:])/M['scale']).pow(2).mean(-1).min(-1).values
            od2=((of[:,:,None,:]-M['o'][None,None,:,:])/M['scale']).pow(2).mean(-1).min(-1).values
            return td.mean(1)+od2.mean(1),td,od2
        M=self.local_v140
        def inv(v):
            r=torch.sort(torch.linalg.norm(v,dim=-1),dim=-1).values; u=v/torch.linalg.norm(v,dim=-1,keepdim=True).clamp_min(1e-8)
            G=torch.einsum('bnji,bnki->bnjk',u,u); K=v.shape[2]; tri=torch.triu_indices(K,K,1,device=v.device); c=torch.sort(G[:,:,tri[0],tri[1]],dim=-1).values
            return r,c
        tr,tc=inv(tv); orr,oc=inv(ov)
        td=(((tr[:,:,None,:]-M['ti_r'][None,None,:,:])/M['rs']).pow(2).mean(-1)+((tc[:,:,None,:]-M['ti_c'][None,None,:,:])/M['cs']).pow(2).mean(-1)).min(-1).values
        od2=(((orr[:,:,None,:]-M['o_r'][None,None,:,:])/M['rs']).pow(2).mean(-1)+((oc[:,:,None,:]-M['o_c'][None,None,:,:])/M['cs']).pow(2).mean(-1)).min(-1).values
        return td.mean(1)+od2.mean(1),td,od2

    def final_geometry_refine(self,t,bo,steps=12,lr=0.010,local_mode="so3"):
        """Stage C: learner-v39 Ti-O reference support + Ti-Ti manifold, symmetry exact."""
        x=bo['raw_tensor'].detach().clone().to(self.device).requires_grad_(True)
        tok=bo.get('o_wyckoff_token'); yr=bo.get('o_raw_tensor')
        if tok is None or yr is None: raise KeyError('Stage C requires BO oxygen Wyckoff token/raw coordinates')
        wt=self._oxygen_wyckoff_template(int(t['spg']),tok); y=yr.detach().clone().to(self.device)
        if y.ndim==1:y=y[None]
        y=y.requires_grad_(bool(wt['ncoord']))
        wtio=float(getattr(self,'stage_c_tio_weight',4.0)); wm=float(getattr(self,'stage_c_titi_manifold_weight',4.0))
        wr=float(getattr(self,'stage_c_repulsion_weight',8.0)); wc=float(getattr(self,'stage_c_coordination_weight',1.0))
        def geom():
            _,_,cell,_,pa,_=self.geometry(t,x); ch=self._expand_oxygen_wyckoff(wt,y); return cell,pa,ch
        def loss_details():
            cell,pa,ch=geom(); B,Np,_=pa.shape; Nc=ch.shape[1]
            tloss,tD,tex=self._tio_reference_support(pa,ch,cell); mloss,mdist,mex=self._titi_manifold_score(pa,cell); local,local_ti,local_o=self._learned_local_force_loss(pa,ch,cell,local_mode)
            # Learned first-shell cutoff supplies only coordination bookkeeping; no target angles/modes.
            df=ch[:,None,:,None,:]-pa[:,:,None,None,:]+self.shift[None,None,None,:,:]
            vc=torch.einsum('bpcqi,bij->bpcqj',df,cell); dd=torch.linalg.norm(vc,dim=-1).min(-1).values
            occ=torch.sigmoid((self.coord_cutoff-dd)/0.08); cti=occ.sum(-1); co=occ.sum(1)
            coord=(((cti-float(self.model.cn[self.parent]))/1.0)**2).mean(1)+(((co-float(self.model.cn[self.child]))/0.75)**2).mean(1)
            d6,_=torch.topk(dd,min(6,Nc),dim=-1,largest=False)
            oo_terms=[]; oo_min=torch.full((B,),float('inf'),device=self.device,dtype=cell.dtype)
            for i in range(Nc-1):
                df2=ch[:,i+1:,:]-ch[:,i:i+1,:]; dc=torch.einsum('bnqi,bij->bnqj',df2[:,:,None,:]+self.shift[None,None,:,:],cell)
                dm=torch.linalg.norm(dc,dim=-1).min(-1).values; oo_min=torch.minimum(oo_min,dm.min(1).values)
                oo_terms.append(torch.nn.functional.softplus((self.oo_hard-dm)/0.08).pow(2))
            oo_loss=torch.cat(oo_terms,dim=1).mean(1) if oo_terms else torch.zeros(B,device=self.device,dtype=cell.dtype)
            total=8.0*local+1.0*wtio*tloss+wm*mloss+0.5*wc*coord+wr*oo_loss
            return total,{'local_force':local,'local_ti':local_ti.mean(1),'local_o':local_o.mean(1),'tio_reference':tloss,'tio_D_q90':torch.quantile(tD,.90,dim=1),'tio_D_max':tD.max(1).values,
                          'tio_excess_q90':torch.quantile(tex,.90,dim=1),'titi_manifold':mloss,
                          'titi_manifold_q90':torch.quantile(mdist,.90,dim=1),'titi_manifold_max':mdist.max(1).values,
                          'titi_manifold_excess_q90':torch.quantile(mex,.90,dim=1),'coordination_loss':coord,
                          'oo_repulsion':oo_loss,'oo_min':oo_min,'bond_min':d6.amin((1,2)),'bond_mean':d6.mean((1,2)),
                          'bond_max':d6.amax((1,2)),'bond_std':d6.std((1,2),unbiased=False),'cell':cell,'tifrac':pa,'ofrac':ch,
                          'volume_per_parent':torch.abs(torch.linalg.det(cell))/max(Np,1)}
        def state():
            with torch.no_grad():
                L,D=loss_details(); coord=self._coordination_counts(D['tifrac'],D['ofrac'],D['cell'])
                return {'objective':float(L[0]),'local_force':float(D['local_force'][0]),'local_mode':local_mode,'tio_reference':float(D['tio_reference'][0]),'tio_D_q90':float(D['tio_D_q90'][0]),
                        'tio_D_max':float(D['tio_D_max'][0]),'tio_excess_q90':float(D['tio_excess_q90'][0]),
                        'titi_manifold':float(D['titi_manifold'][0]),'titi_manifold_q90':float(D['titi_manifold_q90'][0]),
                        'titi_manifold_max':float(D['titi_manifold_max'][0]),'titi_manifold_excess_q90':float(D['titi_manifold_excess_q90'][0]),
                        'coordination_loss':float(D['coordination_loss'][0]),'oo_repulsion':float(D['oo_repulsion'][0]),'oo_min_A':float(D['oo_min'][0]),
                        'tio_min_A':float(D['bond_min'][0]),'tio_mean_A':float(D['bond_mean'][0]),'tio_max_A':float(D['bond_max'][0]),
                        'tio_std_A':float(D['bond_std'][0]),'volume_per_ti_A3':float(D['volume_per_parent'][0]),'coordination':coord,
                        'cell':D['cell'][0].cpu().numpy(),'ti_frac':D['tifrac'][0].cpu().numpy()%1.0,'o_frac':D['ofrac'][0].cpu().numpy()%1.0}
        params=[x]+([y] if wt['ncoord'] else []); pre=state(); opt=torch.optim.Adam(params,lr=float(lr))
        best=float(pre['objective']); bestx=x.detach().clone(); besty=y.detach().clone(); last_improve=0
        patience=max(4,min(8,max(1,int(steps))//3)); done=0
        for k in range(max(1,int(steps))):
            q=k/max(int(steps)-1,1); cur=float(lr)*(0.15+0.85*0.5*(1+math.cos(math.pi*q)))
            for pg in opt.param_groups:pg['lr']=cur
            opt.zero_grad(set_to_none=True); loss=loss_details()[0].mean()
            if not torch.isfinite(loss):break
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,3.0); opt.step(); done=k+1
            with torch.no_grad():val=float(loss_details()[0][0])
            tol=max(0.005,5e-4*max(abs(best),1.0))
            if val<best-tol: best=val; bestx=x.detach().clone(); besty=y.detach().clone(); last_improve=done
            elif done>=6 and done-last_improve>=patience:break
        with torch.no_grad():x.copy_(bestx); y.copy_(besty)
        post=state()
        return {'pre':pre,'post':post,'steps':int(done),'lr':float(lr),'stop_reason':('energy_stagnant' if done<int(steps) else 'max_steps'),
                'o_wyckoff_token':tok,'o_raw_tensor':y.detach().clone()}

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

    def optimize(self,t,raw,steps,lr,mode="manifold"):
        x=raw.detach().clone().requires_grad_(True); opt=torch.optim.Adam([x],lr=lr); n=max(1,int(steps))
        for step in range(n):
            frac=step/max(n-1,1); cur_lr=lr*(0.15+0.85*0.5*(1.0+math.cos(math.pi*frac)))
            for pg in opt.param_groups:pg['lr']=cur_lr
            opt.zero_grad(set_to_none=True)
            if mode=='cheap': L=self.titi_conditioning_loss(t,x)
            elif mode=='manifold': L=self.titi_stageA_loss(t,x)
            else: raise ValueError(f'v145 Ti optimizer mode {mode!r} is unsupported')
            L.sum().backward(); torch.nn.utils.clip_grad_norm_([x],10.0); opt.step()
        return x.detach()

    def _stage_a_batch(self,t,spg,batch_index,nstarts,keep):
        """Fast hierarchical Stage A: many cheap starts -> diverse shortlist -> manifold."""
        tic=time.perf_counter(); self._live('stageA_batch_start',spg=int(spg),batch_index=int(batch_index),nstarts=int(nstarts),keep=int(keep))
        factor=max(1,int(getattr(self,'stage_a_proposal_factor',2)))
        # v145: equalize search effort by actual crystallographic freedom rather
        # than spending the same 128 entrances on every SG.  Lattice metric DOFs
        # and free Ti-Wyckoff coordinates both count; quota is bounded.
        free_dof=int(len(t['spec'])+sum(t['dofs']))
        qbase=max(1,int(getattr(self,'stage_a_quota_base',8)))
        qper=max(0,int(getattr(self,'stage_a_quota_per_dof',8)))
        qmin=max(1,int(getattr(self,'stage_a_quota_min',8)))
        qmax=max(qmin,int(getattr(self,'stage_a_quota_max',128)))
        nraw=max(qmin,min(qmax,qbase+qper*free_dof))*factor
        raw=self.initial_raw(t,nstarts=nraw)
        cheap_steps=max(1,int(getattr(self,'stage_a_cheap_steps',36)))
        cheap=self.optimize(t,raw,cheap_steps,float(getattr(self,'stage_a_cheap_lr',self.ambient_lr)),mode='cheap')
        with torch.no_grad():
            Lc,Dc,_=self.titi_conditioning_loss(t,cheap,details=True)
            vr=Dc['volume_per_ti_A3']/max(self.volume_target_per_ti,1e-8)
            gate=(Dc['metric_valid'] & torch.isfinite(Lc) & (Dc['minimum_titi_A']>=.88*self.hard) & (vr>=.55) & (vr<=1.80))
            cand=torch.where(gate)[0]
            # If an SG is difficult, do not throw away the whole search: keep finite noncollapsed states.
            if cand.numel()==0:
                cand=torch.where(Dc['metric_valid'] & torch.isfinite(Lc) & (Dc['minimum_titi_A']>=.82*self.hard))[0]
            shortlist=max(1,int(getattr(self,'stage_a_shortlist',8)))
            if cand.numel():
                sig=self._stage_a_signature(t,cheap[cand])
                chosen_local=self._greedy_diverse_indices(Lc[cand],sig,min(shortlist,int(cand.numel())),float(getattr(self,'stage_a_diversity',0.10)))
                sel=cand[torch.as_tensor(chosen_local,dtype=torch.long,device=self.device)]
            else:
                sel=torch.empty((0,),dtype=torch.long,device=self.device)
        if sel.numel()==0:
            sec=float(time.perf_counter()-tic)
            self._live('stageA_batch_done',spg=int(spg),batch_index=int(batch_index),seconds=sec,retained=0,
                       raw_proposals=int(nraw),free_dof=int(free_dof),cheap_valid=0,shortlist=0)
            return cheap,sel,sec
        short=cheap[sel].detach().clone()
        refine_steps=max(1,int(getattr(self,'stage_a_manifold_steps',18)))
        refined=self.optimize(t,short,refine_steps,float(getattr(self,'stage_a_manifold_lr',.014)),mode='manifold')
        with torch.no_grad():
            Lm,Dm,_=self.titi_stageA_loss(t,refined,details=True)
            good=Dm['metric_valid'] & torch.isfinite(Lm) & (Dm['minimum_titi_A']>=.88*self.hard)
            gids=torch.where(good)[0]
            if gids.numel()==0: gids=torch.where(torch.isfinite(Lm))[0]
            if gids.numel(): gids=gids[torch.argsort(Lm[gids])[:min(int(keep),int(gids.numel()))]]
            # Return only the refined shortlist; IDs are relative to this tensor.
            out=refined; ids=gids
        sec=float(time.perf_counter()-tic)
        self._live('stageA_batch_done',spg=int(spg),batch_index=int(batch_index),seconds=sec,retained=int(ids.numel()),
                   raw_proposals=int(nraw),free_dof=int(free_dof),cheap_valid=int(cand.numel()),shortlist=int(sel.numel()))
        return out,ids,sec

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

    def _oxygen_field_energy(self, ti_frac, o_frac, cell, details=False):
        """Cheap differentiable oxygen field used inside the early Langevin B prototype.

        Ti is fixed.  The field rewards six near-r0 O around every Ti with an
        octahedral angular spectrum, rewards O coordination near three, and has a
        steep O--O exclusion.  No discrete O ownership/triplet assignment is made.
        """
        B=o_frac.shape[0]; nti=ti_frac.shape[1]; no=o_frac.shape[1]
        # nearest-image Ti--O vectors
        df=o_frac[:,None,:,None,:]-ti_frac[:,:,None,None,:]+self.shift[None,None,None,:,:]
        dc=torch.einsum('bnoqi,bij->bnoqj',df,cell)
        dd=torch.linalg.norm(dc,dim=-1)
        qi=dd.argmin(-1); gather=qi[...,None,None].expand(-1,-1,-1,1,3)
        vec=torch.gather(dc,3,gather).squeeze(3); dist=dd.min(-1).values
        r0=float(self.model.pair(self.parent,self.child).mu)
        # smooth coordination; chosen only as a cheap inner-loop field, not final chemistry score
        # ONE shared differentiable Ti--O bond-order matrix. Both coordination
        # fields are exact marginals of the same matrix, so sum(CN_Ti)==sum(CN_O)
        # by construction. This is the r6 self-consistency invariant.
        occ=torch.sigmoid((r0+0.38-dist)/0.10)
        cn_ti=occ.sum(-1); cn_o=occ.sum(1)
        bond_order_total=occ.sum((1,2))
        e_cn_ti=((cn_ti-6.0)/1.0).pow(2).mean(1)
        e_cn_o=((cn_o-3.0)/0.75).pow(2).mean(1)
        # nearest six O define a continuously differentiable local branch between identity swaps
        d6,ix=torch.topk(dist,k=6,dim=-1,largest=False)
        vv=torch.gather(vec,2,ix[...,None].expand(-1,-1,-1,3))
        e_rad=((d6-r0)/0.18).pow(2).mean((1,2))
        u=vv/torch.clamp(torch.linalg.norm(vv,dim=-1,keepdim=True),min=1e-8)
        pair=[]
        for a in range(5):
            for b in range(a+1,6): pair.append(torch.abs((u[:,:,a,:]*u[:,:,b,:]).sum(-1)))
        ac=torch.stack(pair,dim=-1)
        ac=torch.sort(ac,dim=-1,descending=True).values
        # octahedron: three opposite pairs |cos|=1; twelve cis pairs |cos|=0
        target=torch.zeros_like(ac); target[:,:,:3]=1.0
        e_oct=((ac-target)/0.16).pow(2).mean((1,2))
        # O--O excluded volume under PBC.  IMPORTANT: this is a short-range
        # barrier, not a statistical O--O target.  The previous implementation
        # averaged over all 276 O--O pairs, so one catastrophic 1.2--1.5 A
        # contact was diluted almost to zero.  Normalize by N_O instead: one
        # bad contact must remain expensive regardless of how many harmless
        # long-range pairs are present.
        oo_dist=[]
        for i in range(no-1):
            od=o_frac[:,i+1:,:]-o_frac[:,i:i+1,:]
            oc=torch.einsum('boqi,bij->boqj',od[:,:,None,:]+self.shift[None,None,:,:],cell)
            oo_dist.append(torch.linalg.norm(oc,dim=-1).min(-1).values)
        if oo_dist:
            oo_d=torch.cat(oo_dist,dim=1)
            # Essentially zero above ~1.8 A, but rises very sharply below 1.75 A.
            # softplus keeps the force differentiable for Langevin/autograd.
            oo_pair=torch.nn.functional.softplus((1.75-oo_d)/0.05).pow(2)
            e_oo=oo_pair.sum(1)/float(no)
            oo_min=oo_d.min(1).values
            oo_lt14=(oo_d<1.40).sum(1).to(cell.dtype)
            oo_lt16=(oo_d<1.60).sum(1).to(cell.dtype)
            oo_lt18=(oo_d<1.80).sum(1).to(cell.dtype)
        else:
            e_oo=torch.zeros(B,device=self.device); oo_min=torch.full((B,),float('inf'),device=self.device)
            oo_lt14=oo_lt16=oo_lt18=torch.zeros(B,device=self.device)
        total=3.0*e_oct+1.5*e_rad+2.0*e_cn_ti+1.5*e_cn_o+4.0*e_oo
        if details:
            return total,{'oct':e_oct,'rad':e_rad,'cn_ti':e_cn_ti,'cn_o':e_cn_o,'oo':e_oo,
                          'oo_min':oo_min,'oo_lt14':oo_lt14,'oo_lt16':oo_lt16,'oo_lt18':oo_lt18,
                          'cn_ti_mean':cn_ti.mean(1),'cn_o_mean':cn_o.mean(1),
                          'bond_order_total':bond_order_total,
                          'cn_balance_error':torch.abs(cn_ti.sum(1)-cn_o.sum(1)),
                          'tio_min':dist.amin(dim=(1,2)),'tio_mean6':d6.mean((1,2))}
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

    def _lcao_oxygen_seeds(self, wt, ti_frac, cell, nseed, seed):
        """LCAO-inspired symmetry-exact O entrance guesses.

        This is an initialization field, not an electronic energy: candidate O
        Wyckoff coordinates are ranked by overlap with three Ti-centred radial
        bond orbitals at the learned Ti-O distance, while penalizing a fourth
        Ti entering the first shell.  The returned coordinates preserve the
        requested SG/Wyckoff token exactly.
        """
        d=int(wt['ncoord']); nseed=max(1,int(nseed))
        if not d:
            return torch.zeros((1,0),device=self.device,dtype=cell.dtype)
        ncand=max(32,8*nseed,12*d)
        eng=torch.quasirandom.SobolEngine(dimension=d,scramble=True,seed=int(seed)&0x7fffffff)
        u=eng.draw(ncand).to(device=self.device,dtype=cell.dtype)
        u=0.02+0.96*u
        raw=torch.log(u/(1.0-u))
        with torch.no_grad():
            of=self._expand_oxygen_wyckoff(wt,raw)
            ti=ti_frac.expand(ncand,-1,-1); cc=cell.expand(ncand,-1,-1)
            df=of[:,None,:,None,:]-ti[:,:,None,None,:]+self.shift[None,None,None,:,:]
            dc=torch.einsum('bnoqi,bij->bnoqj',df,cc)
            dist=torch.linalg.norm(dc,dim=-1).min(-1).values.transpose(1,2) # B,O,Ti
            ds=torch.sort(dist,dim=-1).values
            r0=float(self.model.pair(self.parent,self.child).mu)
            sig=max(float(self.model.pair(self.parent,self.child).sigma),0.08)
            top3=ds[...,:3]
            bond=((top3-r0)/sig).pow(2).mean((1,2))
            if ds.shape[-1]>3:
                fourth=ds[...,3]
                extra=torch.nn.functional.softplus((float(self.coord_cutoff)-fourth)/0.10).pow(2).mean(1)
            else:
                extra=torch.zeros_like(bond)
            # Population term is the local-orbital analogue: three strong Ti
            # overlaps should contribute at each oxygen site.
            orb=torch.exp(-0.5*((dist-r0)/sig).pow(2))
            pop=(orb.sum(-1)-3.0).pow(2).mean(1)
            score=bond+0.5*pop+2.0*extra
            ids=torch.argsort(score)[:min(nseed,ncand)]
        return raw[ids].detach()

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
            raw=self._lcao_oxygen_seeds(wt,ti0,cell0,ns,int(getattr(self,'global_seed',0))+7919*spg+37*int.from_bytes(hashlib.blake2b(str(tok).encode(),digest_size=4).digest(),'little')%100000)
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
                          'active':bool(d),'stop_reason':'max_steps',
                          'check_history':[],'n_exec':0,'best_step':0,
                          'last_improve_step':0,'last_improve_energy':float('inf'),
                          'best_energy':float('inf'),'conv':[]})
        N=row0
        raw=torch.zeros((N,maxd),device=self.device,dtype=cell0.dtype)
        for kk,sp in enumerate(specs):
            if sp['d']:
                seeds=self._lcao_oxygen_seeds(sp['wt'],ti0,cell0,sp['B'],int(getattr(self,'global_seed',0))+104729*spg+1009*kk)
                raw[sp['slice'],:sp['d']]=seeds[:sp['B']]
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

        check_interval=25; min_steps=min(100,int(steps)); patience_steps=min(100,max(25,int(steps)//2)); de_tol_abs=0.01; de_tol_rel=1e-4
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
                        cur_best=float(bestE[sp['slice']].min().detach().cpu())
                        sp['check_history'].append((sp['n_exec'],cur_best,snap['grad_rms_best']))
                        tol=max(de_tol_abs,de_tol_rel*max(abs(cur_best),1.0))
                        if (not np.isfinite(sp['last_improve_energy'])) or cur_best < sp['last_improve_energy']-tol:
                            sp['last_improve_energy']=cur_best; sp['last_improve_step']=sp['n_exec']
                        elif sp['n_exec']-sp['last_improve_step']>=patience_steps:
                            sp['active']=False; sp['stop_reason']='energy_stagnant'; active_rows[sp['slice']]=0.0
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
            cut=float(self.coord_cutoff)
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
        snap(0,'O_SCF_initial',o_status != 'failed',g_o,{'o_scf_calls':n_o,'o_scf_status':o_status})
        if o_status == 'failed':
            final=trace[-1]
            return {'energy':final['objective'],'field_energy':final['field_energy'],'components':final['components'],
                    'coordination':final['coordination'],'cell':final['cell'],'ti_frac':final['ti_frac'],'o_frac':final['o_frac'],
                    'raw_tensor':x.detach().clone(),'o_raw_tensor':y.detach().clone(),'o_wyckoff_token':sym['token'],
                    'trace':trace,'cycles':0,'stop_reason':'initial_O_SCF_failed','escape_used':False,'accepted_slow_steps':0}

        nlat=len(t['spec']); gen=torch.Generator(device=self.device)
        gen.manual_seed(int(getattr(self,'global_seed',2025))+100003*int(t['spg'])+9000007)
        stop='max_cycles'; accepted_steps=0
        bo_min_cycles=10; bo_patience=12; last_improve_cycle=0; best_obj=float(trace[-1]['objective']); best_def=int(trace[-1]['coordination'].get('Ti_abs_defect',0)+trace[-1]['coordination'].get('O_abs_defect',0))
        probe_cycles=max(3,int(getattr(self,'bo_probe_cycles',6))); probe_obj0=float(best_obj); probe_def0=int(best_def)
        hop_used=False; hop_cycle=None
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
                scale=0.5**retry
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
                    accepted=True; accepted_steps+=1; break

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
            # BO is an exploratory Langevin search, not a deterministic minimizer. Stop when
            # accepted slow moves cease to discover a meaningfully better chemistry/objective.
            cur_def=int(ss['coordination'].get('Ti_abs_defect',0)+ss['coordination'].get('O_abs_defect',0))
            tol=max(0.01,1e-4*max(abs(best_obj),1.0))
            meaningful=False
            if cur_def < best_def:
                best_def=cur_def; best_obj=float(ss['objective']); meaningful=True
            elif float(ss['objective']) < best_obj-tol:
                best_obj=float(ss['objective']); meaningful=True
            if meaningful: last_improve_cycle=cyc

            # v145 capture probe: the first few ionic cycles must demonstrate a
            # downhill chemistry trend.  A flat entrance receives one large,
            # symmetry-preserving hop in raw crystallographic coordinates; if the
            # post-hop probe is also flat, terminate before expensive long B/C work.
            probe_due=(cyc==probe_cycles) or (hop_used and hop_cycle is not None and cyc==hop_cycle+probe_cycles)
            if probe_due:
                improved=(best_def < probe_def0) or (best_obj < probe_obj0-max(0.25,0.02*max(abs(probe_obj0),1.0)))
                if not improved and not hop_used:
                    xsave=x.detach().clone(); ysave=y.detach().clone()
                    with torch.no_grad():
                        # Larger motion for free Ti coordinates, modest lattice hop;
                        # SG/Wyckoff maps still generate every physical position.
                        if nlat:
                            x[:,:nlat].add_(0.12*torch.randn(x[:,:nlat].shape,device=self.device,dtype=x.dtype,generator=gen))
                        if x.shape[1]>nlat:
                            x[:,nlat:].add_(float(getattr(self,'bo_large_hop',0.85))*torch.randn(x[:,nlat:].shape,device=self.device,dtype=x.dtype,generator=gen))
                        if d:
                            y.add_(float(getattr(self,'bo_large_hop',0.85))*torch.randn(y.shape,device=self.device,dtype=y.dtype,generator=gen))
                        x.clamp_(-10.0,10.0)
                    hs,hg,hn=run_o_scf(max(o_inner,120))
                    if hs=='failed':
                        with torch.no_grad(): x.copy_(xsave); y.copy_(ysave) if d else None
                        stop='early_frozen_hop_failed'; break
                    hz=snap(cyc,'BO_large_hop',True,hg,{'o_scf_calls':hn,'o_scf_status':hs})
                    hop_used=True; hop_cycle=cyc; probe_obj0=float(hz['objective']); probe_def0=int(hz['coordination'].get('Ti_abs_defect',0)+hz['coordination'].get('O_abs_defect',0))
                    best_obj=min(best_obj,probe_obj0); best_def=min(best_def,probe_def0); last_improve_cycle=cyc
                    continue
                elif not improved and hop_used:
                    stop='early_frozen_after_hop'; break
                else:
                    probe_obj0=float(best_obj); probe_def0=int(best_def)
            if cyc>=bo_min_cycles and cyc-last_improve_cycle>=bo_patience:
                stop='energy_stagnant'; break

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
                'raw_tensor':x.detach().clone(), 'o_raw_tensor':y.detach().clone(), 'o_wyckoff_token':sym['token'],
                'trace':trace,'cycles':max(z['cycle'] for z in trace),'stop_reason':stop,'escape_used':True,
                'accepted_slow_steps':accepted_steps}

    def build(self,spg,wp_token,nstarts=None):
        """v145: DOF-scaled Ti Stage A -> LCAO-seeded O/BO capture -> learned Stage C."""
        t=self.template(spg,wp_token); batch_starts=max(1,int(getattr(self,'stage_a_batch_starts',64)))
        self._live('sg_start',spg=int(spg),nstarts=batch_starts,target_a=1)
        a,ids,asec=self._stage_a_batch(t,spg,0,batch_starts,1)
        if len(ids)==0:return []
        qi=int(ids[0]); abase=a[qi:qi+1].detach().clone()
        remain=max(0,int(getattr(self,'environment_steps',60)))
        afinal=self.optimize(t,abase,remain,float(getattr(self,'environment_lr',.016)),mode='manifold') if remain else abase
        with torch.no_grad(): _,DA,_=self.titi_stageA_loss(t,afinal,details=True); manA=float(DA['titi_q90'][0]); manMax=float(DA['titi_max'][0])
        with torch.no_grad():
            _,_,acell,_,afrac,_=self.geometry(t,afinal)
            acell_np=acell[0].detach().cpu().numpy(); afrac_np=afrac[0].detach().cpu().numpy()
        arad=float(self.titi_manifold['radius']); maxratio=float(getattr(self,'stage_a_max_q90_ratio',5.0))
        if manA > maxratio*arad:
            self._live('stageA_manifold_reject',spg=int(spg),q90=float(manA),maximum=float(manMax),radius=float(arad),ratio=float(manA/arad),limit=float(maxratio))
            return []
        ntr=int(getattr(self,'oxygen_field_trajectories',16)); nst=int(getattr(self,'oxygen_field_steps',500)); flr=float(getattr(self,'oxygen_field_lr',.010)); t0=float(getattr(self,'oxygen_field_temp0',.025)); t1=float(getattr(self,'oxygen_field_temp1',.0005))
        tic=time.perf_counter(); sym=self._run_oxygen_field_wyckoff(t,afinal,ntr,nst,flr,t0,t1); sec_sym=time.perf_counter()-tic
        self._live('oxygen_field_done',mode='wyckoff',spg=int(spg),energy=float(sym['energy']),seconds=float(sec_sym),components=sym['components'],wyckoff_token=sym['token'],wyckoff_dof=sym['ncoord'],profile=sym.get('profile',{}))
        bcoord=self._coordination_counts(torch.tensor(sym['ti_frac'],device=self.device,dtype=afinal.dtype)[None],torch.tensor(sym['o_frac'],device=self.device,dtype=afinal.dtype)[None],torch.tensor(sym['cell'],device=self.device,dtype=afinal.dtype)[None])
        tic=time.perf_counter(); cres=self._run_stage_b_electron_phonon(t,afinal,sym); sec_c=time.perf_counter()-tic
        self._live('stageB_scf_done',spg=int(spg),seconds=float(sec_c),energy_before=float(sym['energy']),energy_after=float(cres['field_energy']),regularized_energy_after=float(cres['energy']),components_before=sym['components'],components_after=cres['components'],coord_before=bcoord,coord_after=cres['coordination'],cycles=int(cres['cycles']),stop_reason=cres['stop_reason'],trace=cres['trace'])
        frozen=str(cres.get('stop_reason','')).startswith('early_frozen')
        if frozen:
            # Do not spend Stage-C time polishing a B entrance that failed both
            # the capture probe and its one allowed large hop.
            sec_final=0.0; cfin={'pre':{},'post':{},'steps':0,'lr':float(getattr(self,'stage_c_lr',.010)),'stop_reason':'skipped_frozen_B'}
            self._live('stageC_skipped',spg=int(spg),reason='frozen_B_after_probe_hop')
            rc,rt,ro=cres['cell'],cres['ti_frac'],cres['o_frac']
        else:
            modes=['so3','v140'] if str(getattr(self,'stage_c_local_force','compare'))=='compare' else [str(getattr(self,'stage_c_local_force','so3'))]
            branch=[]
            for lm in modes:
                tic=time.perf_counter(); cf=self.final_geometry_refine(t,cres,getattr(self,'stage_c_steps',12),getattr(self,'stage_c_lr',.010),local_mode=lm); ss=time.perf_counter()-tic
                self._live('stageC_done',spg=int(spg),seconds=float(ss),pre=cf['pre'],post=cf['post'],steps=cf['steps'],local_mode=lm)
                branch.append((lm,cf,ss))
            cfin=branch[0][1]; sec_final=branch[0][2]; rc,rt,ro=cfin['post']['cell'],cfin['post']['ti_frac'],cfin['post']['o_frac']
        with torch.no_grad():
            rs=self.reference_similarity_metrics(
                torch.tensor(rt,dtype=afinal.dtype,device=self.device)[None],
                torch.tensor(ro,dtype=afinal.dtype,device=self.device)[None],
                torch.tensor(rc,dtype=afinal.dtype,device=self.device)[None])[0]
        pid=hashlib.blake2b(str(wp_token).encode('utf-8'),digest_size=4).hexdigest()
        rec={'spg':int(spg),'proposal_id':pid,'wyckoff_parent_token':str(wp_token),'a_source_index':qi,'stageA_effective_dof':int(len(t['spec'])+sum(t['dofs'])),'stageA_raw_quota':int(max(int(getattr(self,'stage_a_quota_min',4)),min(int(getattr(self,'stage_a_quota_max',64)),int(getattr(self,'stage_a_quota_base',4))+int(getattr(self,'stage_a_quota_per_dof',4))*int(len(t['spec'])+sum(t['dofs']))))*max(1,int(getattr(self,'stage_a_proposal_factor',2)))),'titi_stageA_q90':manA,'titi_stageA_max':manMax,'stageA_seconds_sg':float(asec),'stageA_cell':acell_np,'stageA_ti_frac':afrac_np,
             'stageB_seconds_total':float(sec_sym),'wyckoff_seconds':float(sec_sym),'stageB_scf_seconds':float(sec_c),
             'field_energy':sym['energy'],'field_components':sym['components'],'stageB_coordination':bcoord,'stageB_cell':sym['cell'],'stageB_ti_frac':sym['ti_frac'],'stageB_o_frac':sym['o_frac'],
             'wyckoff_energy':sym['energy'],'wyckoff_components':sym['components'],'wyckoff_token':sym['token'],'wyckoff_dof':sym['ncoord'],'wyckoff_tested':sym['tested_wyckoff'],'wyckoff_profile':sym.get('profile',{}),'wyckoff_cell':sym['cell'],'wyckoff_ti_frac':sym['ti_frac'],'wyckoff_o_frac':sym['o_frac'],
             'stageB_scf_energy':cres['field_energy'],'stageB_scf_regularized_energy':cres['energy'],'stageB_scf_components':cres['components'],'stageB_scf_coordination':cres['coordination'],'stageB_scf_cell':cres['cell'],'stageB_scf_ti_frac':cres['ti_frac'],'stageB_scf_o_frac':cres['o_frac'],'stageB_scf_trace':cres['trace'],'stageB_scf_cycles':cres['cycles'],'stageB_scf_stop_reason':cres['stop_reason'],
             'stageC_seconds':float(sec_final),'stageC_pre':cfin['pre'],'stageC_post':cfin['post'],'stageC_steps':cfin['steps'],'stageC_lr':cfin['lr'],'stageC_stop_reason':cfin.get('stop_reason','?'),'stageC_cell':(None if frozen else cfin['post']['cell']),'stageC_ti_frac':(None if frozen else cfin['post']['ti_frac']),'stageC_o_frac':(None if frozen else cfin['post']['o_frac']),
             'reference_similarity':rs,'ntraj':ntr,'steps':nst}
        out=[]
        branches=(branch if not frozen else [('skipped',cfin,sec_final)])
        for lm,cf,ss in branches:
            rr=dict(rec); rr['stageC_local_force_mode']=lm; rr['proposal_id']=pid+'_'+lm; rr['stageC_seconds']=float(ss); rr['stageC_pre']=cf['pre']; rr['stageC_post']=cf['post']; rr['stageC_steps']=cf['steps']; rr['stageC_stop_reason']=cf.get('stop_reason','?')
            if not frozen:
                rr['stageC_cell']=cf['post']['cell']; rr['stageC_ti_frac']=cf['post']['ti_frac']; rr['stageC_o_frac']=cf['post']['o_frac']
                with torch.no_grad(): rr['reference_similarity']=self.reference_similarity_metrics(torch.tensor(rr['stageC_ti_frac'],dtype=afinal.dtype,device=self.device)[None],torch.tensor(rr['stageC_o_frac'],dtype=afinal.dtype,device=self.device)[None],torch.tensor(rr['stageC_cell'],dtype=afinal.dtype,device=self.device)[None])[0]
            out.append(rr)
        self._live('sg_compute_done',spg=int(spg),a_entrances=1,cycles=1+int(cres['cycles']),stop_reason='stageB_scf_'+cres['stop_reason'],stageA_seconds=float(asec),stageB_seconds=float(sec_sym)+float(sec_c),stageB_scf_seconds=float(sec_c)); return out


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
        model=ChemistryModel(cfg['learner_model'])
        formula,z,counts,parent,child=model.resolve_target(cfg['target'])
        b=TiOnlyBuilder(model,parent,child,device=device,starts=cfg['starts'],
                        ambient_steps=cfg['ambient_steps'],ambient_lr=cfg['ambient_lr'],lr=cfg['builder_lr'])
        for k,v in cfg['builder_attrs'].items(): setattr(b,k,v)
        b.live_event_q=event_q; b.live_worker_id=worker_id
        event_q.put({'kind':'worker_start','worker':worker_id,'device':device,'ntasks':len(tasks)})
        for it,(spg,tok) in enumerate(tasks,1):
            # Task-specific deterministic RNG: independent of worker scheduling and never
            # reuses the same random entrance stream for two SG/Wyckoff proposals.
            tseed=deterministic_seed(int(cfg.get('builder_attrs',{}).get('global_seed',2025)),int(spg),str(tok))
            np.random.seed(tseed); torch.manual_seed(tseed)
            if str(device).startswith('cuda:') and torch.cuda.is_available(): torch.cuda.manual_seed(tseed)
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

def main():
    p=argparse.ArgumentParser(description='Juliette v149: v140 efficient engine driven exclusively by learner-v39 single-file TiO/TiTi model')
    p.add_argument('--learner-model',default='data/xn_templates/chemistry_model.json'); p.add_argument('--target',default='TiO2=12'); p.add_argument('--output',default=None); p.add_argument('--stage-c-local-force',choices=['compare','so3','v140'],default='compare')
    p.add_argument('--ngpu',type=int,default=-1); p.add_argument('--workers',type=int,default=0); p.add_argument('--starts',type=int,default=64); p.add_argument('--space-groups',type=int,default=4); p.add_argument('--wyckoff-variants-per-sg',type=int,default=1); p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--trajectories',type=int,default=16); p.add_argument('--field-steps',type=int,default=500); p.add_argument('--field-lr',type=float,default=.010); p.add_argument('--temp0',type=float,default=.025); p.add_argument('--temp1',type=float,default=.0005)
    p.add_argument('--environment-steps',type=int,default=24); p.add_argument('--probe-steps',type=int,default=0); p.add_argument('--environment-lr',type=float,default=.014)
    p.add_argument('--ambient-steps',type=int,default=36); p.add_argument('--ambient-lr',type=float,default=.035); p.add_argument('--builder-lr',type=float,default=.055)
    p.add_argument('--stage-a-proposal-factor',type=int,default=2); p.add_argument('--stage-a-shortlist',type=int,default=8)
    p.add_argument('--stage-a-quota-base',type=int,default=4); p.add_argument('--stage-a-quota-per-dof',type=int,default=4); p.add_argument('--stage-a-quota-min',type=int,default=4); p.add_argument('--stage-a-quota-max',type=int,default=64)
    p.add_argument('--stage-a-cheap-steps',type=int,default=36); p.add_argument('--stage-a-cheap-lr',type=float,default=.035)
    p.add_argument('--stage-a-manifold-steps',type=int,default=18); p.add_argument('--stage-a-manifold-lr',type=float,default=.014); p.add_argument('--stage-a-diversity',type=float,default=.10); p.add_argument('--stage-a-max-q90-ratio',type=float,default=5.0)
    p.add_argument('--stage-c-steps',type=int,default=12); p.add_argument('--stage-c-lr',type=float,default=.010); p.add_argument('--stage-c-tio-weight',type=float,default=4.0); p.add_argument('--stage-c-titi-manifold-weight',type=float,default=4.0); p.add_argument('--stage-c-coordination-weight',type=float,default=1.0); p.add_argument('--stage-c-repulsion-weight',type=float,default=8.0)
    p.add_argument('--bo-probe-cycles',type=int,default=6); p.add_argument('--bo-large-hop',type=float,default=.85)
    p.add_argument('--verbose-bo-trace',action='store_true',help='Print every BO cycle; default prints sparse checkpoints while full trace remains in results.jsonl')
    args=p.parse_args(); slurm_ntasks=int(os.environ.get('SLURM_NTASKS','1') or 1); slurm_cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','1') or 1)
    if slurm_ntasks>1: raise SystemExit(f'v145 requires one SLURM task, but SLURM_NTASKS={slurm_ntasks}.')
    np.random.seed(args.seed); torch.manual_seed(args.seed); outdir=Path(args.output or f'{_formula_prefix(args.target)}_v146_{int(args.space_groups)}SG'); outdir.mkdir(parents=True,exist_ok=True); (outdir/'cif').mkdir(exist_ok=True)
    logfile=outdir/'run.log'; progressfile=outdir/'progress.jsonl'; visible=torch.cuda.device_count() if torch.cuda.is_available() else 0; ngpu=visible if args.ngpu<0 else min(max(args.ngpu,0),visible); nworkers=(ngpu if args.workers<=0 else args.workers) if ngpu else 1; nworkers=max(1,nworkers); devices=[f'cuda:{i%max(ngpu,1)}' for i in range(nworkers)] if ngpu else ['cpu']*nworkers
    model=ChemistryModel(args.learner_model); formula,z,counts,parent,child=model.resolve_target(args.target)
    rng=np.random.default_rng(args.seed); bysg={}
    cap=max(4,4*max(1,int(args.wyckoff_variants_per_sg)))
    for spg in range(1,231):
        try:toks=exact_parent_entries(spg,counts[parent],cap=cap)
        except Exception:toks=[]
        if toks:
            toks=list(dict.fromkeys(toks)); rng.shuffle(toks); bysg[int(spg)]=toks
    # v146 benchmark selection: spread requested SGs across the actual Ti
    # crystallographic DOF range instead of taking a purely random SG sample.
    # For the default one variant/SG, choose one deterministic token per SG,
    # then sample approximately uniformly through sorted effective DOF.
    from pyxtal.symmetry import Group
    def benchmark_dof(spg,tok):
        g=Group(int(spg)); lt=str(g.lattice_type).lower()
        nlat=1 if lt=='cubic' else (2 if lt in {'tetragonal','hexagonal','trigonal'} else (3 if lt=='orthorhombic' else (4 if lt=='monoclinic' else 6)))
        return int(nlat+sum(int(g[int(i)].get_dof()) for i in decode_int_token(tok)))
    candidates=[]
    for spg,toks in bysg.items():
        tok=toks[0]
        candidates.append((benchmark_dof(spg,tok),int(spg),tok))
    # Shuffle before stable sort so ties in DOF do not systematically favor SG number.
    rng.shuffle(candidates); candidates.sort(key=lambda x:x[0])
    want=min(max(1,int(args.space_groups)),len(candidates))
    pick_idx=np.unique(np.rint(np.linspace(0,len(candidates)-1,want)).astype(int))
    if len(pick_idx)<want:
        used=set(int(x) for x in pick_idx); pick_idx=list(map(int,pick_idx))
        for i in range(len(candidates)):
            if i not in used: pick_idx.append(i); used.add(i)
            if len(pick_idx)>=want: break
    selected=[candidates[int(i)] for i in pick_idx[:want]]
    choices=[]
    selected_dof={}
    for dof,spg,tok0 in selected:
        selected_dof[int(spg)]=int(dof)
        toks=bysg[spg][:max(1,int(args.wyckoff_variants_per_sg))]
        for tok in toks: choices.append((spg,tok))
    rng.shuffle(choices); chunks=[choices[i::nworkers] for i in range(nworkers)]
    attrs={'stage_a_batch_starts':args.starts,'stage_a_batch_keep':1,'stage_a_proposal_factor':args.stage_a_proposal_factor,'stage_a_shortlist':args.stage_a_shortlist,'stage_a_quota_base':args.stage_a_quota_base,'stage_a_quota_per_dof':args.stage_a_quota_per_dof,'stage_a_quota_min':args.stage_a_quota_min,'stage_a_quota_max':args.stage_a_quota_max,
           'stage_a_cheap_steps':args.stage_a_cheap_steps,'stage_a_cheap_lr':args.stage_a_cheap_lr,'stage_a_manifold_steps':args.stage_a_manifold_steps,
           'stage_a_manifold_lr':args.stage_a_manifold_lr,'stage_a_diversity':args.stage_a_diversity,'stage_a_max_q90_ratio':args.stage_a_max_q90_ratio,
           'environment_steps':args.environment_steps,'probe_steps':min(args.probe_steps,args.environment_steps),'environment_lr':args.environment_lr,'global_seed':args.seed,'oxygen_field_trajectories':args.trajectories,'oxygen_field_steps':args.field_steps,'oxygen_field_lr':args.field_lr,'oxygen_field_temp0':args.temp0,'oxygen_field_temp1':args.temp1,'stage_c_steps':args.stage_c_steps,'stage_c_lr':args.stage_c_lr,'stage_c_tio_weight':args.stage_c_tio_weight,'stage_c_titi_manifold_weight':args.stage_c_titi_manifold_weight,'stage_c_coordination_weight':args.stage_c_coordination_weight,'stage_c_repulsion_weight':args.stage_c_repulsion_weight,'bo_probe_cycles':args.bo_probe_cycles,'bo_large_hop':args.bo_large_hop,'stage_c_local_force':args.stage_c_local_force}
    cfg={'learner_model':args.learner_model,'target':args.target,'starts':args.starts,'ambient_steps':args.ambient_steps,'ambient_lr':args.ambient_lr,'builder_lr':args.builder_lr,'builder_attrs':attrs}
    def emit(x):
        print(x,flush=True)
        with logfile.open('a') as f:f.write(x+'\n')
    emit('--- Juliette v149 4-SG learned-local-force A/B benchmark / single learner-model generation ---'); emit(f'Target={args.target}; SG={len(choices)}; trajectories/SG={args.trajectories}; field steps={args.field_steps}')
    emit(f'Resources: SLURM_NTASKS={slurm_ntasks}; SLURM_CPUS_PER_TASK={slurm_cpus}; visible GPUs={visible}; workers={nworkers}; devices={devices}')
    emit('Benchmark SG/DOF: '+', '.join(f'{sg}:{selected_dof.get(sg,"?")}' for sg,_ in choices))
    emit('POLICY: ONE learner-v39 JSON only. Stage A = DOF-scaled cheap learned capture -> diverse shortlist -> exact Ti-Ti latent manifold. SG/Wyckoff symmetry remains exact. B uses LCAO-inspired symmetry-exact oxygen entrances + BO capture probe/hop; C uses learner-v39 Ti-O Bayesian reference support + Ti-Ti manifold. No legacy model, no interpolated centers.')
    # No built-in/legacy reference calibration is consulted in v145.
    # All chemistry/reference support comes from --learner-model.
    wall_start=time.perf_counter(); ctx=mp.get_context('spawn'); q=ctx.Queue(); procs=[]
    for wid,(dev,tasks) in enumerate(zip(devices,chunks)):
        pr=ctx.Process(target=_v135_worker,args=(wid,dev,tasks,cfg,q),daemon=False); pr.start(); procs.append(pr)
    allrows=[]; done=0; total=0.0; fatals=[]
    while done<len(procs):
        msg=q.get(); kind=msg.get('kind')
        if kind=='live':
            with logfile.open('a') as f:f.write('[LIVE] '+json.dumps(msg,default=_v135_jsonable)+'\n')
            if msg.get('event')=='stageA_batch_done':emit(f"[GPU{msg['worker']} SG {msg['spg']}] Stage-A dof={msg.get('free_dof','?')} raw={msg.get('raw_proposals','?')} cheap_valid={msg.get('cheap_valid','?')} shortlist={msg.get('shortlist','?')} retained={msg.get('retained')} in {msg.get('seconds',0):.1f}s")
            elif msg.get('event')=='stageA_manifold_reject':emit(f"[GPU{msg['worker']} SG {msg['spg']}] Stage-A reject: TiMan q90={msg.get('q90',float('nan')):.5f} = {msg.get('ratio',float('nan')):.2f}x radius > {msg.get('limit',float('nan')):.2f}x")
            elif msg.get('event')=='oxygen_field_done':
                c=msg.get('components',{}); emit(f"[GPU{msg['worker']} SG {msg['spg']}] O-field[{msg.get('mode','?')}] E={msg.get('energy',float('nan')):.3f} oct={c.get('oct',float('nan')):.3f} rad={c.get('rad',float('nan')):.3f} CNti={c.get('cn_ti',float('nan')):.3f} CNo={c.get('cn_o',float('nan')):.3f} OO={c.get('oo',float('nan')):.3f} OOmin={c.get('oo_min',float('nan')):.3f} {msg.get('seconds',0):.1f}s")
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
            elif msg.get('event')=='stageC_done':
                a=msg.get('pre',{}); b=msg.get('post',{}); ca=a.get('coordination',{}); cb=b.get('coordination',{})
                emit(f"[GPU{msg['worker']} SG {msg['spg']}] Stage-C local-force {a.get('objective',float('nan')):.3f}->{b.get('objective',float('nan')):.3f} TiO-D90={a.get('tio_D_q90',float('nan')):.3f}->{b.get('tio_D_q90',float('nan')):.3f} TiO-Dmax={a.get('tio_D_max',float('nan')):.3f}->{b.get('tio_D_max',float('nan')):.3f} | TiMan q90={a.get('titi_manifold_q90',float('nan')):.5f}->{b.get('titi_manifold_q90',float('nan')):.5f} max={a.get('titi_manifold_max',float('nan')):.5f}->{b.get('titi_manifold_max',float('nan')):.5f} | TiO mean/std={a.get('tio_mean_A',float('nan')):.3f}/{a.get('tio_std_A',float('nan')):.3f}->{b.get('tio_mean_A',float('nan')):.3f}/{b.get('tio_std_A',float('nan')):.3f}A OOmin={a.get('oo_min_A',float('nan')):.3f}->{b.get('oo_min_A',float('nan')):.3f}A | TiO6 {ca.get('TiO6',0)}->{cb.get('TiO6',0)} OTi3 {ca.get('OTi3',0)}->{cb.get('OTi3',0)} | steps={msg.get('steps','?')} {msg.get('seconds',0):.1f}s")
            elif msg.get('event')=='stageB_scf_done':
                b=msg.get('components_before',{}); a=msg.get('components_after',{}); cb=msg.get('coord_before',{}); ca=msg.get('coord_after',{})
                emit(f"[GPU{msg['worker']} SG {msg['spg']}] BO-B field {msg.get('energy_before',float('nan')):.3f}->{msg.get('energy_after',float('nan')):.3f} regE={msg.get('regularized_energy_after',float('nan')):.3f} | oct {b.get('oct',float('nan')):.3f}->{a.get('oct',float('nan')):.3f} rad {b.get('rad',float('nan')):.3f}->{a.get('rad',float('nan')):.3f} CNti {b.get('cn_ti',float('nan')):.3f}->{a.get('cn_ti',float('nan')):.3f} CNo {b.get('cn_o',float('nan')):.3f}->{a.get('cn_o',float('nan')):.3f} OO {b.get('oo',float('nan')):.3f}->{a.get('oo',float('nan')):.3f} OOmin {b.get('oo_min',float('nan')):.3f}->{a.get('oo_min',float('nan')):.3f} | cycles={msg.get('cycles')} stop={msg.get('stop_reason')} {msg.get('seconds',0):.1f}s")
                emit(f"[GPU{msg['worker']} SG {msg['spg']}] COORD frozen-B->BO-B: TiO4 {cb.get('TiO4',0)}->{ca.get('TiO4',0)} TiO5 {cb.get('TiO5',0)}->{ca.get('TiO5',0)} TiO6 {cb.get('TiO6',0)}->{ca.get('TiO6',0)} | OTi2 {cb.get('OTi2',0)}->{ca.get('OTi2',0)} OTi3 {cb.get('OTi3',0)}->{ca.get('OTi3',0)} OTi4 {cb.get('OTi4',0)}->{ca.get('OTi4',0)}")
                tr=msg.get('trace',[])
                for iz,z in enumerate(tr):
                    cyc=int(z.get('cycle',0)); phase=str(z.get('phase','?'))
                    show=bool(args.verbose_bo_trace or iz==0 or iz==len(tr)-1 or cyc%10==0 or 'quench' in phase.lower())
                    if not show: continue
                    c=z.get('components',{}); coord=z.get('coordination',{}); emit(f"[GPU{msg['worker']} SG {msg['spg']}] BO-B {cyc:02d}/{phase} obj={z.get('objective',float('nan')):.3f} field={z.get('field_energy',float('nan')):.3f} oct={c.get('oct',float('nan')):.3f} rad={c.get('rad',float('nan')):.3f} CNti={c.get('cn_ti',float('nan')):.3f} CNo={c.get('cn_o',float('nan')):.3f} OO={c.get('oo',float('nan')):.3f} OOmin={c.get('oo_min',float('nan')):.3f} Nbond={coord.get('Nbond','?')} TiDef={coord.get('Ti_abs_defect',float('nan')):.0f} ODef={coord.get('O_abs_defect',float('nan')):.0f} TiRMS={z.get('ti_disp_rms_A',float('nan')):.3f}A strainRMS={z.get('strain_rms',float('nan')):.4f} V/V0={z.get('volume_ratio',float('nan')):.4f} gRMS={z.get('grad_rms',float('nan')):.3e}")
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
                sg=int(r['spg']); pid=str(r.get('proposal_id','p0'))
                stem=f"SG_{sg:03d}_{pid}"
                # v145: exactly three trajectory artifacts per successful proposal.
                if r.get('stageA_cell') is not None:_write_simple_cif(outdir/'cif'/f"{stem}_A.cif",r['stageA_cell'],r['stageA_ti_frac'],['Ti']*len(r['stageA_ti_frac']),f"{stem}_A")
                if r.get('stageB_scf_cell') is not None:_write_simple_cif(outdir/'cif'/f"{stem}_B.cif",r['stageB_scf_cell'],np.vstack([r['stageB_scf_ti_frac'],r['stageB_scf_o_frac']]),['Ti']*len(r['stageB_scf_ti_frac'])+['O']*len(r['stageB_scf_o_frac']),f"{stem}_B")
                if r.get('stageC_cell') is not None:_write_simple_cif(outdir/'cif'/f"{stem}_C.cif",r['stageC_cell'],np.vstack([r['stageC_ti_frac'],r['stageC_o_frac']]),['Ti']*len(r['stageC_ti_frac'])+['O']*len(r['stageC_o_frac']),f"{stem}_C")
            if rr:emit(f"[SG {msg['spg']:3d} | GPU{msg['worker']}] B E={rr[0]['field_energy']:.3f} -> BO-B E={rr[0].get('stageB_scf_energy',float('nan')):.3f} | elapsed={msg['elapsed_s']:.1f}s")
        elif kind=='worker_done':done+=1; emit(f"[worker {msg['worker']} {msg['device']}] finished")
        elif kind=='worker_fatal':done+=1; fatals.append(msg); emit(f"[worker {msg['worker']}] FATAL {msg['error']}")
    for pr in procs:pr.join()
    with (outdir/'results.jsonl').open('w') as f:
        for r in allrows:f.write(json.dumps(r,default=_v135_jsonable)+'\n')
    metric_fields=['rank','spg','proposal_id','stageC_local_force_mode','wyckoff_parent_token','stageA_effective_dof','stageA_raw_quota','stageA_seconds_sg','wyckoff_seconds','stageB_scf_seconds','stageB_scf_cycles','stageB_scf_stop_reason','stageC_seconds','stageC_stop_reason',
                   'A_titi_q90','A_titi_max','B_field_energy','BO_field_energy','C_objective_pre','C_objective_post',
                   'C_tio_D_q90_pre','C_tio_D_q90_post','C_tio_D_max_pre','C_tio_D_max_post',
                   'C_titi_q90_pre','C_titi_q90_post','C_titi_max_pre','C_titi_max_post',
                   'C_oo_min_pre_A','C_oo_min_post_A','C_tio_std_pre_A','C_tio_std_post_A',
                   'TiO6_B','OTi3_B','TiO6_BO','OTi3_BO','TiO6_C','OTi3_C',
                   'ref_D_mean','ref_D_q50','ref_D_q90','ref_D_max','ref_frac_D_le_0p25','ref_frac_D_le_0p5','ref_frac_D_le_1','ref_frac_D_le_2','ref_nearest_site_counts','ref_titi_q90','ref_titi_max','ref_titi_fraction_inside']
    with (outdir/'generation_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=metric_fields); w.writeheader()
        ranked=sorted(allrows,key=lambda z:(float((z.get('reference_similarity') or {}).get('local_D_q90',float('inf'))),-float((z.get('reference_similarity') or {}).get('fraction_D_le_1',0.0))))
        for ir,r in enumerate(ranked,1):
            b=r.get('stageB_coordination',{}); bo=r.get('stageB_scf_coordination',{}); cp=r.get('stageC_pre',{}); cq=r.get('stageC_post',{}); cc=cq.get('coordination',{})
            rs=r.get('reference_similarity') or {}
            row={'rank':ir,'spg':r.get('spg'),'proposal_id':r.get('proposal_id'),'stageC_local_force_mode':r.get('stageC_local_force_mode'),'wyckoff_parent_token':r.get('wyckoff_parent_token'),'stageA_effective_dof':r.get('stageA_effective_dof'),'stageA_raw_quota':r.get('stageA_raw_quota'),'stageA_seconds_sg':r.get('stageA_seconds_sg'),'wyckoff_seconds':r.get('wyckoff_seconds'),'stageB_scf_seconds':r.get('stageB_scf_seconds'),'stageB_scf_cycles':r.get('stageB_scf_cycles'),'stageB_scf_stop_reason':r.get('stageB_scf_stop_reason'),'stageC_seconds':r.get('stageC_seconds'),'stageC_stop_reason':r.get('stageC_stop_reason'),
                 'A_titi_q90':r.get('titi_stageA_q90'),'A_titi_max':r.get('titi_stageA_max'),'B_field_energy':r.get('field_energy'),'BO_field_energy':r.get('stageB_scf_energy'),
                 'C_objective_pre':cp.get('objective'),'C_objective_post':cq.get('objective'),'C_tio_D_q90_pre':cp.get('tio_D_q90'),'C_tio_D_q90_post':cq.get('tio_D_q90'),
                 'C_tio_D_max_pre':cp.get('tio_D_max'),'C_tio_D_max_post':cq.get('tio_D_max'),'C_titi_q90_pre':cp.get('titi_manifold_q90'),'C_titi_q90_post':cq.get('titi_manifold_q90'),
                 'C_titi_max_pre':cp.get('titi_manifold_max'),'C_titi_max_post':cq.get('titi_manifold_max'),'C_oo_min_pre_A':cp.get('oo_min_A'),'C_oo_min_post_A':cq.get('oo_min_A'),
                 'C_tio_std_pre_A':cp.get('tio_std_A'),'C_tio_std_post_A':cq.get('tio_std_A'),
                 'ref_D_mean':rs.get('local_D_mean'),'ref_D_q50':rs.get('local_D_q50'),'ref_D_q90':rs.get('local_D_q90'),'ref_D_max':rs.get('local_D_max'),
                 'ref_frac_D_le_0p25':rs.get('fraction_D_le_0p25'),'ref_frac_D_le_0p5':rs.get('fraction_D_le_0p5'),'ref_frac_D_le_1':rs.get('fraction_D_le_1'),'ref_frac_D_le_2':rs.get('fraction_D_le_2'),
                 'ref_nearest_site_counts':json.dumps(rs.get('nearest_reference_site_counts',{}),sort_keys=True),'ref_titi_q90':rs.get('titi_q90'),'ref_titi_max':rs.get('titi_max'),'ref_titi_fraction_inside':rs.get('titi_fraction_inside')}
            for tag,d in [('B',b),('BO',bo),('C',cc)]:
                for k in ('TiO6','OTi3'): row[f'{k}_{tag}']=d.get(k,0)
            w.writerow(row)
    # Long-form per-reference ranking, directly exposing which CIF reference each output resembles.
    ref_fields=['global_rank','spg','proposal_id','reference','mean_D','q50_D','q90_D','max_D','fraction_D_le_1']
    with (outdir/'reference_similarity_by_reference.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ref_fields); w.writeheader()
        ranked=sorted(allrows,key=lambda z:float((z.get('reference_similarity') or {}).get('local_D_q90',float('inf'))))
        for ir,r in enumerate(ranked,1):
            per=(r.get('reference_similarity') or {}).get('per_reference',{})
            for name,d in sorted(per.items(),key=lambda kv:float(kv[1].get('q90',float('inf')))):
                w.writerow({'global_rank':ir,'spg':r.get('spg'),'proposal_id':r.get('proposal_id'),'reference':name,
                            'mean_D':d.get('mean'),'q50_D':d.get('q50'),'q90_D':d.get('q90'),'max_D':d.get('max'),'fraction_D_le_1':d.get('fraction_D_le_1')})
    # Two primary benchmark targets: throughput and density of low-reference-distance outputs.
    dvals=np.asarray([float((r.get('reference_similarity') or {}).get('local_D_q90',np.inf)) for r in allrows],float)
    finite=dvals[np.isfinite(dvals)]
    bins=[0.0,0.25,0.5,0.75,1.0,1.5,2.0,4.0,float('inf')]
    hist=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        hist.append({'lo':lo,'hi':hi,'count':int(np.sum((finite>=lo)&(finite<hi)))})
    wall_seconds=float(time.perf_counter()-wall_start)
    wall_proxy=max([float(r.get('stageA_seconds_sg',0))+float(r.get('wyckoff_seconds',0))+float(r.get('stageB_scf_seconds',0))+float(r.get('stageC_seconds',0)) for r in allrows]+[0.0])
    per_ref_summary={}
    for name in model.tio_reference_field['references']:
        vals=[]
        for r in allrows:
            d=((r.get('reference_similarity') or {}).get('per_reference',{}).get(name) or {}).get('q90')
            if d is not None and np.isfinite(float(d)): vals.append(float(d))
        a=np.asarray(vals,float)
        per_ref_summary[name]={'n':int(len(a)),'median_q90_D':float(np.median(a)) if len(a) else None,
                               'density_q90_D_le_1':float(np.mean(a<=1.0)) if len(a) else 0.0,
                               'density_q90_D_le_0p5':float(np.mean(a<=.5)) if len(a) else 0.0}
    nlow=int(np.sum(finite<=1.0)); nvlow=int(np.sum(finite<=.5)); ntask=max(len(choices),1)
    summary={'version':'v149-split-learning-local-force-4sg-ab','requested_space_groups':int(args.space_groups),'proposal_tasks':len(choices),'rows':len(allrows),
             'generation_yield':float(len(allrows)/ntask),'gpu_minutes':total/60,'wall_seconds':wall_seconds,'fatal_workers':len(fatals),
             'throughput_structures_per_gpu_hour':(len(allrows)/(total/3600) if total>0 else 0.0),'throughput_structures_per_wall_hour':(len(allrows)/(wall_seconds/3600) if wall_seconds>0 else 0.0),
             'low_distance_structures_per_gpu_hour':(nlow/(total/3600) if total>0 else 0.0),
             'max_single_pipeline_seconds':wall_proxy,'reference_D_q90_median':float(np.median(finite)) if len(finite) else None,
             'low_distance_density_Dq90_le_1':float(np.mean(finite<=1.0)) if len(finite) else 0.0,
             'very_low_distance_density_Dq90_le_0p5':float(np.mean(finite<=0.5)) if len(finite) else 0.0,
             'effective_low_distance_yield_per_proposal':float(nlow/ntask),'effective_very_low_distance_yield_per_proposal':float(nvlow/ntask),
             'reference_D_q90_histogram':hist,'per_reference_summary':per_ref_summary,'reference_names':list(model.tio_reference_field['references']),
             'reference_directory_from_learner':model.generation.get('provenance',{}).get('reference_directory')}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2,default=_v135_jsonable))
    (outdir/'reference_distance_distribution.json').write_text(json.dumps({'bins':hist,'Dq90_values':finite.tolist()},indent=2))
    emit(f"=== v149 SPLIT-LEARNING LOCAL-FORCE A/B DONE: tasks={len(choices)} rows={len(allrows)} GPU-min={total/60:.2f} low-D(q90<=1)={summary['low_distance_density_Dq90_le_1']:.3f} throughput={summary['throughput_structures_per_gpu_hour']:.2f}/GPU-h wall={summary['throughput_structures_per_wall_hour']:.2f}/h fatal_workers={len(fatals)} ===")

if __name__=='__main__':main()
