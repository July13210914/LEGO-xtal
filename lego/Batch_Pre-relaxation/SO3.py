import torch
from torch import cos, sin, sqrt
import numpy as np
import torch.optim as optim
import pandas as pd
import math
import time
import logging
import os
import torch.nn.functional as F
import torch.jit as jit

@jit.script
def _assoc_legendre_script(ell: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """
    TorchScript-compiled associated Legendre P_ell^m over a flat tensor `x`.
    Input  : x  shape (BN,)  — any dtype/ device
    Output :     shape (BN,) — same dtype/ device
    The tiny ℓ-loop is now inside the scripted graph (C++ / CUDA).
    """
    # P_m^m
    if m == 0:
        Pmm = torch.ones_like(x)
    else:
        # (-1)^m (2m-1)!! (1-x²)^{m/2}
        double_fact = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        for i in range(1, m + 1):
            double_fact = double_fact * (2 * i - 1)               # (2m-1)!!
        Pmm = ((-1.)**m) * double_fact * ((1 - x)*(1 + x)).pow(0.5 * m)

    if ell == m:
        return Pmm

    # P_{m+1}^m
    Pm1m = x * (2*m + 1) * Pmm
    if ell == m + 1:
        return Pm1m

    # upward recurrence (still scalar loop, now scripted)
    Pllm2 = Pmm
    Pllm1 = Pm1m
    for l in range(m + 2, ell + 1):
        new_val = ((2 * l - 1) * x * Pllm1 - (l + m - 1) * Pllm2) / (l - m)
        Pllm2 = Pllm1
        Pllm1 = new_val
    return Pllm1
@jit.script
def _assoc_legendre_batch(l_vec: torch.Tensor,  # (L,)
                        m_vec: torch.Tensor,  # (L,)
                        x_flat: torch.Tensor  # (B·N,)
                        ) -> torch.Tensor:    # → (B·N, L)
    L  = l_vec.size(0)
    out = torch.empty(x_flat.size(0), L,
                    dtype=x_flat.dtype, device=x_flat.device)
    for i in range(L):
        out[:, i] = _assoc_legendre_script(
            int(l_vec[i]), int(m_vec[i]), x_flat)
    return out


@jit.script
def spherical_bessel_in(lmax: int, x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=1e-6, max=5e2)

    i0 = torch.sinh(x) / x  
    i1 = (torch.sinh(x) / (x * x)) - (torch.cosh(x) / x)

    Bessels_list = [i0.unsqueeze(-1)]
    if lmax > 0:
        Bessels_list.append(i1.unsqueeze(-1))

    i_l_minus_2 = i0.unsqueeze(-1)
    i_l_minus_1 = i1.unsqueeze(-1)

    for l in range(2, lmax + 1):
        i_l = ((2 * (l - 1) + 1) / x.unsqueeze(-1)) * i_l_minus_1 + i_l_minus_2
        Bessels_list.append(i_l)
        i_l_minus_2 = i_l_minus_1
        i_l_minus_1 = i_l

    return torch.cat(Bessels_list, dim=-1) if lmax > 0 else Bessels_list[0]

class SO3:
    def __init__(self, lmax=4, nmax=2, alpha=2.0, rcut=2.0, max_N=200):
        self.lmax = lmax
        self.nmax = nmax
        self.alpha = alpha
        self.rcut = rcut
        self.max_N = max_N
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def build_neighbor_list_local_batched(self, cell, positions, numbers, atom_ids):
        """
        Batched version of neighbor-list construction with padding.
        Returns fixed-size tensors using NaN on missing neighbors.
        """
        device = self.device
        B, N, _ = positions.shape
        _, M = atom_ids.shape

        coords = torch.arange(-1, 2, dtype=torch.float64, device=device)  # [-1,0,1]
        x1, y1, z1 = torch.meshgrid(coords, coords, coords, indexing='ij')
        VECTORS = torch.stack([x1, y1, z1], dim=-1).reshape(-1, 3)  # (27,3)

        vectors_batched = VECTORS.unsqueeze(0).expand(B, -1, -1)
        cell_shifts = torch.bmm(vectors_batched, cell)

        ref_pos = positions.unsqueeze(2)  # => (B,N,1,3)
        cell_shifts_4d = cell_shifts.unsqueeze(1)  # => (B,1,27,3)
        ref_pos = ref_pos + cell_shifts_4d  # => (B,N,27,3)

        valid_centers = (atom_ids != -1)  # (B,M)

        dummy_atom_ids = torch.full_like(atom_ids, fill_value=N, dtype=torch.long)  
        atom_ids_valid = torch.where(valid_centers, atom_ids, dummy_atom_ids)

        positions_padded = torch.cat([positions, torch.full((B, 1, 3), torch.nan, device=device, dtype=torch.float64)], dim=1)

        gather_index = atom_ids_valid.unsqueeze(-1).expand(-1, -1, 3)  
        center_atoms = torch.gather(positions_padded, dim=1, index=gather_index)  

        center_atoms[~valid_centers.unsqueeze(-1).expand_as(center_atoms)] = torch.nan

        diff = ref_pos.unsqueeze(1) - center_atoms.unsqueeze(2).unsqueeze(2)  
        dists = torch.norm(diff, dim=-1)

        nan_mask = torch.isnan(dists)
        dists[nan_mask] = torch.inf

        mask = (dists > 1e-3) & (dists < self.rcut)

        valid_mask = atom_ids != -1  
        neighbors_per_atom = (mask * valid_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=(2, 3))  


        #print('max_N1',self.max_N)


        valid_b, valid_center_idx, valid_atom_idx, valid_image_idx = torch.where(mask)

        neighbor_pos = torch.full((B, self.max_N, 3),
                                  -1000, dtype=torch.float64, device=device)
        atomic_weights = torch.full((B, self.max_N),
                                    -1, dtype=torch.complex128, device=device)
        neighbor_indices = torch.full((B, self.max_N, 3),
                                      -1, dtype=torch.long, device=device)

        if valid_b.numel() == 0:
            return neighbor_pos, atomic_weights, neighbor_indices

        sort_idx = torch.argsort(valid_b)
        vb_sorted = valid_b[sort_idx]
        vc_sorted = valid_center_idx[sort_idx]
        va_sorted = valid_atom_idx[sort_idx]
        vi_sorted = valid_image_idx[sort_idx]

        counts = torch.bincount(vb_sorted, minlength=B)

        # ---------- vectorised scatter (no Python for‑loop over batches) ----------
        #
        # Build a rank-within-batch for every valid neighbour entry, then
        # scatter into the padded tensors.  This keeps at most `self.max_N`
        # neighbours per batch, matching the previous behaviour of `c_store`.
        #
        if vb_sorted.numel() > 0:
            num_valid = vb_sorted.size(0)
            idx_all   = torch.arange(num_valid, device=device)

            # rank_in_batch = how many neighbours already seen for that batch
            first_occ_mask = torch.zeros_like(vb_sorted, dtype=torch.bool)
            first_occ_mask[0] = True
            first_occ_mask[1:] = vb_sorted[1:] != vb_sorted[:-1]

            start_idx_per_batch = torch.full((B,), -1,
                                             dtype=torch.long, device=device)
            start_idx_per_batch[vb_sorted[first_occ_mask]] = idx_all[first_occ_mask]

            rank_in_batch = idx_all - start_idx_per_batch[vb_sorted]

            # Respect cap of self.max_N neighbours per batch
            keep_mask = rank_in_batch < self.max_N

            rb = vb_sorted[keep_mask]          # batch indices
            rr = rank_in_batch[keep_mask]      # slot within batch (0 … max_N-1)

            src_va = va_sorted[keep_mask]      # atom index
            src_vi = vi_sorted[keep_mask]      # image index
            src_vc = vc_sorted[keep_mask]      # centre index within atom_ids

            # neighbour positions
            neighbor_pos[rb, rr, :] = (
                ref_pos[rb, src_va, src_vi, :] - center_atoms[rb, src_vc, :]
            )
            # atomic weights
            atomic_weights[rb, rr] = numbers[rb, src_va].to(torch.complex128)

            # neighbour indices: [centre, neighbour, image]
            neighbor_indices[rb, rr, 0] = atom_ids[rb, src_vc]
            neighbor_indices[rb, rr, 1] = src_va
            neighbor_indices[rb, rr, 2] = src_vi
        # ---------- end vectorised scatter ---------------------------------------

        return neighbor_pos, atomic_weights, neighbor_indices
        

    def Cosine(self,Rij, Rc, derivative=False):
        """
        Cosine cutoff function: 0.5*(cos(pi*Rij/Rc)+1) ; Rij<=Rc
        """
        if not derivative:
            result = 0.5 * (torch.cos(torch.pi * Rij / Rc) + 1.)
        else:
            result = -0.5 * torch.pi / Rc * torch.sin(torch.pi * Rij / Rc)
        result = torch.where(Rij <= Rc, result, torch.zeros_like(result))
        return result
    

    def W(self,nmax):
        alpha = torch.arange(1, nmax + 1, dtype=torch.float64, device=self.device).view(-1, 1)
        beta  = torch.arange(1, nmax + 1, dtype=torch.float64, device=self.device).view(1, -1)
    
        temp1 = (2*alpha + 5)*(2*alpha + 6)*(2*alpha + 7)
        temp2 = (2*beta  + 5)*(2*beta  + 6)*(2*beta  + 7)
        numer = torch.sqrt(temp1 * temp2)
        denom = (5 + alpha + beta)*(6 + alpha + beta)*(7 + alpha + beta)
        arr   = numer/denom
        sinv  = torch.linalg.inv(arr)
        eigvals, V = torch.linalg.eigh(sinv)
        sqrtD = torch.diag(torch.sqrt(eigvals))
        arr_ortho = V @ sqrtD @ V.T
        return arr_ortho
    

    def phi(self,r, alpha, rcut):
        """
        φ_alpha(r) = (rcut - r)^(alpha + 2)/Normalization
        """
        normalization = (2 * rcut**(2*alpha + 7)) / ((2*alpha + 5)*(2*alpha + 6)*(2*alpha + 7))
        return (rcut - r)**(alpha + 2)/torch.sqrt(normalization)
    

    def g(self,r, n, nmax, rcut, w):
        alpha_vals = torch.arange(1, nmax + 1, dtype=torch.float64, device=r.device).view(1, -1)
        phi_vals   = self.phi(r.view(-1,1), alpha_vals, rcut)
        Sum        = torch.matmul(phi_vals, w[n-1, :nmax].view(-1, 1)).squeeze()
        return Sum
    

    def factorial(self,n):
        if isinstance(n, int):
            n = torch.tensor(n, dtype=torch.float64, device=self.device)
        return torch.exp(torch.lgamma(n + 1))
    


    def factorial_torch(self,n):
        return torch.exp(torch.lgamma(n+1))

   
 
    def spherical_harmonic(self, l_vec, m_vec, thetas_2d, phis_2d):

        device = thetas_2d.device
        B, N = thetas_2d.shape
        L = l_vec.numel()
    
        cos_theta_2d = torch.clamp(torch.cos(thetas_2d), min=-0.999999, max=0.999999)
    
        # Compute Legendre polynomials
        P_vals = torch.zeros(B, N, L, dtype=torch.float64, device=device)
        m_abs_vec = m_vec.abs()
        #serial loop
        # Compute all P_ℓ^{|m|} at once (no Python loop)
        P_flat = _assoc_legendre_batch(
                    l_vec.to(torch.int64),
                    m_abs_vec.to(torch.int64),
                    cos_theta_2d.reshape(-1))          # (B·N, L)
        P_vals = P_flat.view(B, N, L)                   # (B, N, L)
    
        # Normalization Factor
        two_ell_plus_1 = (2.0 * l_vec + 1.0).to(torch.float64)
        num = self.factorial_torch(l_vec - m_abs_vec)
        denom = self.factorial_torch(l_vec + m_abs_vec)
        prefactors_1d = torch.sqrt((two_ell_plus_1 / (4.0 * math.pi)) * (num / denom))
    
        # Condon–Shortley Phase
        #sign_condon_1d = (-1.0) ** m_vec.abs()
        sign_condon_1d = torch.where(m_vec < 0, (-1.0) ** m_vec.abs(), torch.ones_like(m_vec, dtype=torch.float64))
    
        # Fix shape mismatch
        P_vals *= sign_condon_1d.view(1, 1, -1)  # Dynamic broadcasting
    
        # Compute e^(i m phi)
        m_3d = m_vec.view(1, 1, L).to(phis_2d.dtype)
        phi_3d = phis_2d.unsqueeze(-1)
        phi_3d = torch.where(torch.isfinite(phi_3d), phi_3d, torch.tensor(0.0, device=phi_3d.device))  # Replace NaNs

        phase = torch.exp(1j * m_3d * phi_3d)
    
        # Final result
        Y_all = P_vals * prefactors_1d.view(1, 1, L) * phase
   
        return Y_all
    
    def compute_spherical_harmonics(self,lmax, thetas, phis):
        """
        Returns shape (B,N,lmax+1,2*lmax+1), complex
        """
        device = thetas.device
        B, N   = thetas.shape
    
        l_grid, m_shift_grid = torch.meshgrid(
            torch.arange(lmax+1, device=device),
            torch.arange(2*lmax+1, device=device),
            indexing='ij'
        )
        valid_mask = (m_shift_grid >= (lmax - l_grid)) & (m_shift_grid <= (lmax + l_grid))
        l_vec = l_grid[valid_mask]
        m_vec = m_shift_grid[valid_mask] - lmax


    
        Y_all = self.spherical_harmonic(l_vec, m_vec, thetas, phis)
        Y_lm  = torch.zeros((B,N,lmax+1,2*lmax+1), dtype=Y_all.dtype, device=device)
    
        # Scatter results
        Y_lm_updated = Y_lm.clone()
        Y_lm_updated[:, :, l_grid[valid_mask], m_shift_grid[valid_mask]] = Y_all
        return Y_lm_updated

    def GaussLegendreQuadrature(self,nmax, lmax):
        """
        Demo routine returning Chebyshev-like nodes in [-1,1].
        """
        import math
        NQuad = (nmax + lmax + 1)*10
        i     = torch.arange(1, NQuad+1, dtype=torch.float64, device=self.device)
        quad_array = torch.cos((2.0*i - 1.0)*torch.pi/(2.0*NQuad))
        weight     = torch.tensor(torch.pi/(NQuad), device=self.device, dtype=torch.float64)
        return quad_array, weight

    def compute_cs(self,batched_pos, nmax, lmax, rcut, alpha, cutoff):

        valid_mask = ~torch.isnan(batched_pos).any(dim=-1)  # (B,N)
        Ris = torch.norm(batched_pos, dim=-1)
        #Ris = torch.where(Ris > 1e-6, Ris, torch.tensor(1e-6, dtype=Ris.dtype, device=Ris.device))
        
    
        GCQuadrature, weight = self.GaussLegendreQuadrature(self.nmax, self.lmax)

        Quadrature = rcut/2.0*(GCQuadrature + 1.0)
 
        alpha_values = torch.arange(1, nmax+1, dtype=torch.float64, device=self.device).view(-1,1)
 
        phi_values   = self.phi(Quadrature, alpha_values, rcut)  # (nmax, Q)

        w_matrix = self.W(nmax).to(self.device)  # (nmax,nmax)
        Gs = torch.matmul(w_matrix[:, :nmax], phi_values)  # (nmax, Q)
    
        # Multiply by measure = r^2 exp(-alpha r^2) * sqrt(1 - x^2) * weight
        # (just as in your example)
        Quad_Squared = Quadrature**2
        Gs *= Quad_Squared * torch.exp(-alpha * Quad_Squared) * torch.sqrt(torch.clamp(1 - GCQuadrature**2, min=0.0)) * weight
       
        # Bessels => shape (B,N,Q,lmax+1)
        BesselArgs=2.0 * alpha * torch.einsum('bn,q->bnq', Ris, Quadrature)
        #print("BesselArgs (Modified Spherical Bessel Function Args):", BesselArgs[:,:11,:20])

        Bessels    = spherical_bessel_in(self.lmax, BesselArgs)
        #print("Bessels (Modified Spherical Bessel Functions):", Bessels[:,:20,8:11,:])

        # Integrate => (B,N,nmax,lmax+1)
        integral_array = torch.einsum('ij,bkjl->bkil', Gs, Bessels)
        # Ensure integral_array properly ignores masked values


        integral_array = torch.nan_to_num(integral_array, nan=0.0) #, posinf=1e6, neginf=-1e6)
        # Compute angles safely
        cos_theta = batched_pos[..., 2] / Ris
        cos_theta = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)

        thetas=torch.arccos(cos_theta)
        thetas = torch.where(
            valid_mask,
            torch.arccos(cos_theta),  # Apply arccos safely
            torch.full_like(batched_pos[..., 2], float(-1)) 
        )
         
        eps = 1e-8  # Small epsilon to avoid division by zero
        phis = torch.atan2(
            torch.where((batched_pos[..., 0] == 0) & (batched_pos[..., 1] == 0), 
                        torch.full_like(batched_pos[..., 1], eps), 
                        batched_pos[..., 1]),
            torch.where((batched_pos[..., 0] == 0) & (batched_pos[..., 1] == 0), 
                        torch.full_like(batched_pos[..., 0], eps), 
                        batched_pos[..., 0])
        )
        
        Y_lm = self.compute_spherical_harmonics(self.lmax, thetas, phis)

        #print("ylms (Spherical Harmonics):", Y_lm[:1,:11,:].sum())
        # cutoff & normalization => shape (B,N
        Ris = torch.norm(batched_pos, dim=-1)

        cutoff_vals = cutoff(Ris, rcut)  # (B,N)
        norm_vals   = 4*math.pi*torch.exp(-alpha*Ris**2)
        exparray    = norm_vals*cutoff_vals  # (B,N)
        exparray    = exparray.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Combine
        E = torch.einsum('bnlm,bnil->bnilm', Y_lm, integral_array)

        exparray = exparray.expand(-1, -1, E.shape[2], -1, -1)  # Ensure compatible shape
      
        C = E * exparray  # Now element-wise multiplication works
        #print('C', C[:1,:11].sum())
   
        return C



    def compute_p(self,cell, positions, numbers, atom_ids):
        """
        Computes the power spectrum from atomic neighbor lists using spherical harmonics.
    
        Args:
            cell (torch.Tensor): Shape (B, 3, 3), unit cell matrices.
            positions (torch.Tensor): Shape (B, N, 3), atomic positions.
            numbers (torch.Tensor): Shape (B, N), atomic numbers.
            atom_ids (torch.Tensor): Shape (B, M), indices of central atoms.
    
        Returns:
            torch.Tensor: Shape (B, M, ncoefs), power spectrum.
        """

    
        # Move tensors to device
        cell = cell.to(self.device)
        positions = positions.to(self.device)
        numbers = numbers.to(self.device)
        atom_ids = atom_ids.to(self.device)
        # Build neighbor list
        neighbor_pos, atomic_weights, neighbor_indices = self.build_neighbor_list_local_batched(
            cell, positions, numbers, atom_ids
        )
        # Mask invalid neighbors
        mask = atomic_weights == -1  # Shape (B, max_N)
        
        #neighbor_pos[mask] = float(-1000)  # Set invalid positions to NaN
        neighbor_indices[mask] = -1000  # Mark invalid indices

        # Compute coefficients (SO3 features)
        cs = self.compute_cs(neighbor_pos, nmax=self.nmax, lmax=self.lmax, rcut=self.rcut, alpha=self.alpha, cutoff=self.Cosine)
        #print('neighbor_pos', neighbor_pos)
        #print('cs',cs.shape)
        # Apply atomic weights
        atomic_weights_expanded = atomic_weights[:, :, None, None, None]  # Shape: (B, max_N, 1, 1, 1)
        atomic_weights_expanded = torch.nan_to_num(atomic_weights_expanded, nan=0.0)

        C_s_weighted = cs * atomic_weights_expanded
    
        # Compute normalization factors
        ls = torch.arange(self.lmax + 1, device=self.device, dtype=torch.float64)
        norm = torch.sqrt(
            2 * torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=self.device))
            * torch.tensor(math.pi, dtype=torch.float64, device=self.device)
            / torch.sqrt(2 * ls + 1)
        ).to(torch.complex128)
        cs_norm = torch.einsum('bnilm,l->bnilm', C_s_weighted, norm)
        #print('cs_norm',cs_norm[0,:11].sum())
    
        # Initialize power spectrum storage
        ncoefs = self.nmax * (self.nmax + 1) // 2 * (self.lmax + 1)
        B, M = atom_ids.shape
        plist = torch.zeros((B, M, ncoefs), dtype=torch.float64, device=self.device)
        CN_max=12
        dists = -torch.ones((B,M, CN_max),dtype=torch.float64, device=self.device)
    
        # Compute power spectrum
        # Serial loop over batches and atoms
        # ---------- vectorised power-spectrum block (no Python loops) ----------
        # cs_norm : (B, P, nR, L, K)      neighbour coefficients
        B, P, nR, L, K = cs_norm.shape
        ncoefs = self.nmax * (self.nmax + 1) // 2 * (self.lmax + 1)   # unchanged

        # a) accumulate φ_tot for every (batch, centre-atom)
        N_atoms       = numbers.shape[1]                          # max index + 1
        batch_offset  = (torch.arange(B, device=self.device) * N_atoms).unsqueeze(1)  # (B,1)
        centres       = neighbor_indices[..., 0]                  # (B, P)
        valid_rows    = centres >= 0                              # rows kept after “-1000” sentinel
        glob_centres  = (centres + batch_offset)[valid_rows]      # 1-D (N_pairs,)

        phi_tot = torch.zeros((B * N_atoms, nR, L, K),
                              dtype=cs_norm.dtype,
                              device=self.device)                 # (C, nR, L, K)
        phi_tot.index_add_(
            0,
            glob_centres.reshape(-1).long(),
            cs_norm[valid_rows].reshape(-1, nR, L, K)
        )                                                         #   CUDA kernel

        # b) power spectrum  P(c,r,s,l) = Σ_k φ(c,r,l,k) φ*(c,s,l,k)
        P = torch.einsum('crlk,cslk->crsl',                       # (C,nR,L,K) × (C,nR,L,K)
                         phi_tot, torch.conj(phi_tot)).real       # → (C,nR,nR,L)

        # c) gather only the atoms requested in `atom_ids`
        plist = torch.zeros((B, atom_ids.shape[1], ncoefs),
                            dtype=torch.float64,
                            device=self.device)

        valid_atoms     = atom_ids >= 0                           # boolean mask
        glob_atom_ids   = (atom_ids + batch_offset)               # (B, M)
        P_sel = P[glob_atom_ids[valid_atoms].long()]             # (#valid, nR, nR, L)

        # flatten lower-triangular (r ≤ s) for every l
        r_idx, s_idx    = torch.tril_indices(nR, nR, device=self.device)
        coeffs          = P_sel[:, r_idx, s_idx, :].reshape(-1, ncoefs)  # (#valid, ncoefs)

        plist[valid_atoms] = coeffs
        # ---------- end of vectorised block -----------------------------------

        # 1. distance of every neighbour row ------------------------
        dist_all = torch.norm(neighbor_pos, dim=2)           # (B, P)

        # 2. mask[b,m,p] == True  ⟺  neighbour p belongs to atom_ids[b,m]
        mask = (atom_ids.unsqueeze(-1) ==
                neighbor_indices[:, :, 0].unsqueeze(1))      # (B, M, P)
        mask &= (atom_ids.unsqueeze(-1) != -1)               # drop padded centres

        # 3. running rank of each True along P ----------------------
        rank = torch.cumsum(mask.int(), dim=2) - 1           # (B, M, P)
        rank[~mask] = -1                                     # put –1 elsewhere

        # 4. keep only first CN_max neighbours ----------------------
        keep = (rank >= 0) & (rank < CN_max)                 # (B, M, P)

        # 5. scatter into output tensor -----------------------------
        dists = torch.full((B, M, CN_max), -1.0,
                        dtype=neighbor_pos.dtype,
                        device=neighbor_pos.device)

        b_idx, m_idx, p_idx = keep.nonzero(as_tuple=True)    # flat indices
        slot                = rank[b_idx, m_idx, p_idx]      # 0 … CN_max-1
        dists[b_idx, m_idx, slot] = dist_all[b_idx, p_idx]   # write distances
        rdf = self.dist2rdf(dists)
        #return plist
        return torch.cat((plist, 0.3*rdf), dim=2)
    
    def dist2rdf(self, dists, num_bins=20, sigma=0.5):
        """
        Convert a batch of distance arrays to radial distribution functions (RDFs).
        
        Args:
            dists (torch.Tensor): Shape (B, M, N), where:
                                  B = batch size, 
                                  M = number of central atoms,
                                  N = number of neighbors.
            num_bins (int): Number of histogram bins.
            sigma (float): Standard deviation Gaussian smoothing.
    
        Returns:
            torch.Tensor: Shape (B, M, num_bins), the batch-wise RDFs.
        """
        B, M, N = dists.shape  # Batch size, number of central atoms, number of neighbors
        device = dists.device
    
        dr = self.rcut / num_bins  # Bin width
        rdf = torch.zeros((B, M, num_bins), device=device)  # Initialize RDF histogram
    
        # 🔹 Step 1: Filter valid distances
        valid_mask = (dists > 0) & (dists < self.rcut)  # Exclude zero and cutoff values
        valid_dists = dists * valid_mask  # Zero-out invalid distances
    
        # 🔹 Step 2: Compute bin indices
        bin_indices = (valid_dists / dr).long()  # Convert distances to bin indices
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  # Ensure indices are valid
    
        # 🔹 Step 3: Compute histogram in a vectorized way
        weights = valid_mask.float()  
        
        # Vectorized scatter_add_ to accumulate counts in bins
        rdf.scatter_add_(2, bin_indices, weights)
    
        def gaussian_kernel(sigma, kernel_size, device):
            x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=device)
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()  # Normalize the kernel
            return kernel.view(1, 1, -1)  # Shape: (out_channels, in_channels, kernel_size)
        # Define kernel_size based on sigma
        kernel_size = 3
        
        kernel = gaussian_kernel(sigma, kernel_size, device=device)
    
        # 🔹 Step 5: Apply Gaussian smoothing
        padding = kernel_size // 2  # Compute required padding
        rdf = F.conv1d(rdf.view(B * M, 1, num_bins), kernel, padding=padding).view(B, M, num_bins)
    
        return rdf
