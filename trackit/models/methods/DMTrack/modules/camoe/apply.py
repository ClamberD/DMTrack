import torch
from torch import nn
from .model import camoe
from typing import List, Literal

    
class LinearWith_camoe(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        rs_lora: bool = False,
        init_method: str = "bert",
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        n_activated_experts: int = 1,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0,
        use_load_balance: bool = True,
        lfb_momentum: float = 0.9,
        router_norm: str = "rmsnorm",
        lfb_update_rate: float = 1e-4, 
    ):
        super().__init__()
        self.linear = linear

        self.camoe = camoe(
            dim=linear.in_features,
            out_dim=linear.out_features,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            n_activated_experts=n_activated_experts,
            score_func=score_func,
            route_scale=route_scale,
            lora_rank=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            rs_lora=rs_lora,
            init_method=init_method,
            use_load_balance=use_load_balance,
            lfb_momentum=lfb_momentum,
            router_norm=router_norm,
            lfb_update_rate=lfb_update_rate,  
        )

    def forward(self, x: torch.Tensor):
        return self.linear(x) + self.camoe(x)
    

class LinearWith_camoe_TimmQKV(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        rs_lora: bool = False,
        init_method: str = "bert",
        target_q: bool = True,
        target_k: bool = True,
        target_v: bool = True,

        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        n_activated_experts: int = 1,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0,
        use_load_balance: bool = True,
        lfb_momentum: float = 0.9,
        router_norm: str = "rmsnorm",
        lfb_update_rate: float = 1e-3,
    ):
        super().__init__()

        dim = linear.in_features
        bias = linear.bias is not None

        q = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        k = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)

        q.weight.data.copy_(linear.weight.data[:dim])
        k.weight.data.copy_(linear.weight.data[dim:2 * dim])
        v.weight.data.copy_(linear.weight.data[2 * dim:])

        q.weight.requires_grad = k.weight.requires_grad = v.weight.requires_grad = linear.weight.requires_grad
        if bias:
            q.bias.data.copy_(linear.bias.data[:dim])
            k.bias.data.copy_(linear.bias.data[dim:2 * dim])
            v.bias.data.copy_(linear.bias.data[2 * dim:])
            q.bias.requires_grad = k.bias.requires_grad = v.bias.requires_grad = linear.bias.requires_grad

        def wrap(m: nn.Linear):
            return LinearWith_camoe(
                m, r, alpha, dropout, rs_lora, init_method,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                n_activated_experts=n_activated_experts,
                score_func=score_func,
                route_scale=route_scale,
                use_load_balance=use_load_balance,
                lfb_momentum=lfb_momentum,
                router_norm=router_norm,
                lfb_update_rate=lfb_update_rate,
            )

        self.q = wrap(q) if target_q else q
        self.k = wrap(k) if target_k else k
        self.v = wrap(v) if target_v else v

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat((q, k, v), dim=-1)



def find_all_frozen_nn_linear_names(model, inference=False):
    cls = nn.Linear
    lora_module_names = set()

    for name, module in model.named_modules():
        if not inference:
            if isinstance(module, cls) and not module.weight.requires_grad:
                lora_module_names.add(name)
        else:
            if isinstance(module, cls):
                lora_module_names.add(name)

    return list(lora_module_names)


def apply_camoe(
    model: nn.Module,
    lora_module_names: List[str],
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_rslora: bool = False,
    init_method: str = "bert",

    n_routed_experts: int = 4,
    n_shared_experts: int = 1,
    n_activated_experts: int = 1,
    score_func: str = "softmax",
    route_scale: float = 1.0,
    use_load_balance: bool = True,
    lfb_momentum: float = 0.9,
    router_norm: str = "rmsnorm",
    lfb_update_rate: float = 1e-4,   
):
    for lora_module_name in lora_module_names:
        tokens = lora_module_name.split('.')
        parent_module_name = '.'.join(tokens[:-1])
        module_name = tokens[-1]
        parent_module = model.get_submodule(parent_module_name)

        if module_name == 'qkv':
            layer_cls = LinearWith_camoe_TimmQKV
        else:
            layer_cls = LinearWith_camoe

        setattr(
            parent_module,
            module_name,
            layer_cls(
                getattr(parent_module, module_name),
                lora_r,
                lora_alpha,
                lora_dropout,
                use_rslora,
                init_method,

                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                n_activated_experts=n_activated_experts,
                score_func=score_func,
                route_scale=route_scale,
                use_load_balance=use_load_balance,
                lfb_momentum=lfb_momentum,
                router_norm=router_norm,
                lfb_update_rate=lfb_update_rate,  
            )
        )

