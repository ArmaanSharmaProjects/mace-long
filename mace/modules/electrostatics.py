import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class GaussianSmearing(nn.Module):
    def __init__(self, start: float = -1.0, stop: float = 1.0, num_gaussians: int = 16):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1))**2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class ElectronegativityModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QeqSolver(nn.Module):
    def __init__(self, hardness_init: Dict[int, float], covalent_radii: Dict[int, float]):
        super().__init__()
        self.hardness = nn.ParameterDict({
            str(z): nn.Parameter(torch.tensor(val, dtype=torch.float64))
            for z, val in hardness_init.items()
        })

        self.register_buffer("covalent_radii", torch.tensor(
            [covalent_radii.get(z, 1.0) for z in range(119)], dtype=torch.float64
        ))

    def get_parameters(self, atomic_numbers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = atomic_numbers.device
        J = torch.zeros_like(atomic_numbers, dtype=torch.float64)
        sigma = self.covalent_radii[atomic_numbers].to(device)
        
        unique_z = torch.unique(atomic_numbers)
        for z in unique_z:
            z_str = str(z.item())
            if z_str in self.hardness:
                mask = (atomic_numbers == z)
                J[mask] = self.hardness[z_str]
                
        return J, sigma

    def forward(self, 
                edge_index: torch.Tensor, 
                node_attrs: torch.Tensor, 
                chi: torch.Tensor, 
                ptr: torch.Tensor, 
                total_charge: torch.Tensor,
                positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        atomic_numbers = node_attrs[:, 0].long()
        J, sigma = self.get_parameters(atomic_numbers)
        
        charges_list = []
        energies_list = []
        
        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i+1]
            n_atoms = end - start
            
            pos_batch = positions[start:end]
            J_batch = J[start:end]
            sigma_batch = sigma[start:end]
            chi_batch = chi[start:end]
            q_tot = total_charge[i]

            
            r_vec = pos_batch.unsqueeze(0) - pos_batch.unsqueeze(1)
            r_ij = torch.norm(r_vec, dim=-1)

            sigma_sq = sigma_batch**2
            gamma_ij = torch.sqrt(sigma_sq.unsqueeze(0) + sigma_sq.unsqueeze(1))

            mask = torch.eye(n_atoms, device=pos_batch.device).bool()
            safe_r = torch.where(mask, torch.ones_like(r_ij), r_ij)
            inv_r = 1.0 / safe_r

            off_diag = torch.erf(safe_r / (1.41421356 * gamma_ij)) * inv_r
            off_diag = torch.where(mask, torch.zeros_like(off_diag), off_diag)

            diag_val = J_batch + 1.0 / (sigma_batch * 1.77245385)
            A = torch.where(mask, torch.diag(diag_val), off_diag)
            
            
            M = torch.zeros((n_atoms + 1, n_atoms + 1), device=pos_batch.device, dtype=torch.float64)
            M[:n_atoms, :n_atoms] = A
            M[:n_atoms, n_atoms] = 1.0
            M[n_atoms, :n_atoms] = 1.0
            
            rhs = torch.cat([-chi_batch, q_tot.view(1)])
            
            solution = torch.linalg.solve(M, rhs)
            Q = solution[:n_atoms]
            
            charges_list.append(Q)


            A_off_diag = torch.where(mask, torch.zeros_like(A), A)
            term1 = 0.5 * torch.sum(Q.unsqueeze(0) * A_off_diag * Q.unsqueeze(1))
            

            term2 = torch.sum(Q**2 / (2 * sigma_batch * 1.77245385))
            
            energies_list.append(term1 + term2)
            
        print(f"DEBUG: Number of graphs: {len(ptr)-1}")
        print(f"DEBUG: Atoms per graph: {[ptr[i+1]-ptr[i] for i in range(len(ptr)-1)]}")

        return torch.cat(charges_list), torch.stack(energies_list)