import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import torch.nn.init as init
import math
from timm.layers import trunc_normal_


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, dropout: float,
                 rs_lora: bool = False, init_method: str = 'bert'):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, r))
        self.r = r
        self.alpha = alpha
        
        if rs_lora:
            self.scaling = alpha / math.sqrt(r)
        else:
            self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if init_method == 'lora':
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif init_method == 'gaussian':
            nn.init.normal_(self.A, std=1. / self.r)
            nn.init.zeros_(self.B)
        elif init_method == 'bert':
            trunc_normal_(self.A, std=.02)
            trunc_normal_(self.B, std=.02)
        else:
            raise ValueError(f'Unknown init method: {init_method}')

    def forward(self, x: torch.Tensor):
        return (self.dropout(x) @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)) * self.scaling


class BatchedLoRALayers(nn.Module):
    def __init__(self, num_experts: int, in_dim: int, out_dim: int, r: int, 
                 alpha: float, dropout: float, rs_lora: bool = False, init_method: str = 'bert'):
        super().__init__()
        self.num_experts = num_experts
        self.r = r
        self.scaling = alpha / (math.sqrt(r) if rs_lora else r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.experts_A = nn.Parameter(torch.empty(num_experts, r, in_dim))
        self.experts_B = nn.Parameter(torch.empty(num_experts, out_dim, r))
        
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        for i in range(self.num_experts):
            if init_method == 'lora':
                nn.init.kaiming_uniform_(self.experts_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.experts_B[i])
            elif init_method == 'gaussian':
                nn.init.normal_(self.experts_A[i], std=1. / self.r)
                nn.init.zeros_(self.experts_B[i])
            elif init_method == 'bert':
                trunc_normal_(self.experts_A[i], std=.02)
                trunc_normal_(self.experts_B[i], std=.02)

    def forward(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.dropout(x)
        

        lora_A_out = torch.einsum('bd, erd -> bre', x, self.experts_A)


        full_gate_weights = torch.zeros(batch_size, self.num_experts, device=x.device, dtype=x.dtype)
        full_gate_weights.scatter_(1, indices, weights)


        lora_A_out = lora_A_out * full_gate_weights.unsqueeze(1)


        out = torch.einsum('bre, eor -> bo', lora_A_out, self.experts_B)

        return out * self.scaling



class Gate(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        n_routed_experts: int = 8,
        n_activated_experts: int = 2,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0,
        lfb_update_rate: float = 1e-3,
        momentum: float = 0.9,       
        norm_type: str = "rmsnorm"   
    ):
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.lfb_update_rate = lfb_update_rate
        self.n_routed_experts = n_routed_experts
        self.momentum = momentum     


        if norm_type == "rmsnorm":
            self.router_norm = RMSNorm(dim)
        elif norm_type == "layernorm":
            self.router_norm = nn.LayerNorm(dim)
        else:
            self.router_norm = nn.Identity()

        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        

        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.register_buffer("accumulated_counts", torch.zeros(n_routed_experts))
        

        self.register_buffer("running_load", torch.ones(n_routed_experts) / n_routed_experts)
        
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def accumulate_stats(self, counts: torch.Tensor):

        self.accumulated_counts += counts.float()

    @torch.no_grad()
    def apply_bias_update(self):

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.accumulated_counts, op=torch.distributed.ReduceOp.SUM)
            

        total_tokens = self.accumulated_counts.sum()
        if total_tokens < 1: 

            self.accumulated_counts.zero_()
            return

        current_dist = self.accumulated_counts / total_tokens
        

        # running_load = momentum * old + (1 - momentum) * new
        self.running_load = self.momentum * self.running_load + (1 - self.momentum) * current_dist
        

        target_load = 1.0 / self.n_routed_experts
        error = self.running_load - target_load
        

        self.expert_bias -= self.lfb_update_rate * torch.sign(error)


        self.accumulated_counts.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x_norm = self.router_norm(x)
        

        scores = F.linear(x_norm, self.weight)
        
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        
        original_scores = scores
        biased_scores = scores + self.expert_bias

        indices = torch.topk(biased_scores, self.topk, dim=-1)[1]
        

        weights = original_scores.gather(1, indices)

        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)

        weights *= self.route_scale
        return weights.type_as(x), indices


class camoe(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        out_dim: int = 768,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        n_activated_experts: int = 1,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0,
        lora_rank: int = 64,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        rs_lora: bool = False,
        init_method: str = "bert",
        use_load_balance: bool = True,

        lfb_momentum: float = 0.9,     
        router_norm: str = "rmsnorm",
        lfb_update_rate: float = 1e-3,   
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_shared_experts = n_shared_experts

        # Gate 初始化时传入新参数
        self.gate = Gate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            score_func=score_func,
            route_scale=route_scale,
            lfb_update_rate=lfb_update_rate,
            momentum=lfb_momentum,   
            norm_type=router_norm   
        )

        # Routed Experts
        self.routed_experts = BatchedLoRALayers(
            num_experts=n_routed_experts,
            in_dim=dim,
            out_dim=out_dim,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            rs_lora=rs_lora,
            init_method=init_method
        )

        # Shared Experts
        self.shared_experts = nn.ModuleList([
            LoRALayer(
                in_dim=dim,
                out_dim=out_dim,
                r=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                rs_lora=rs_lora,
                init_method=init_method
            )
            for _ in range(n_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if len(original_shape) == 3:
            x_flat = x.view(-1, self.dim)
        else:
            x_flat = x
            
        gate_weights, indices = self.gate(x_flat)
        y = self.routed_experts(x_flat, indices, gate_weights)

        if self.n_shared_experts > 0:
            z = sum(expert(x_flat) for expert in self.shared_experts) / self.n_shared_experts
            y = y + z

        if self.training:
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
            self.gate.accumulate_stats(counts)

        if self.dim == self.out_dim:
            return y.view(original_shape)
        else:
            new_shape = list(original_shape)
            new_shape[-1] = self.out_dim
            return y.view(new_shape)

    @torch.no_grad()
    def update_load_balance(self):

        if self.training:
            self.gate.apply_bias_update()