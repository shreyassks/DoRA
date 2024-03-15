import torch
import torch.nn as nn


lora_r = 64
lora_alpha = 128
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = True


class LoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank, dtype=torch.float32))
        self.A = nn.Parameter(torch.randn(in_dim, self.rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(self.rank, out_dim))

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearLayerWithLora(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoraLayer(self.linear.in_features, self.linear.out_features, rank, alpha)

    def forward(self, x):
        x = self.linear(x) + self.lora(x)
        return x


class LinearLayerWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoraLayer(self.linear.in_features, self.linear.out_features, rank, alpha)
        self.m = nn.Parameter(torch.ones(1, self.linear.out_features))

    def forward(self, x):
        linear_output = self.linear(x)
        lora_output = self.lora(x)
        dora_norm = lora_output / (lora_output.norm(p=2, dim=1, keepdim=True) + 1e-9)
        dora_output = self.m * dora_norm
        return linear_output + dora_output
    
    
def set_trainable_params(model):
    for module in model.children():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = False
        else:
            set_trainable_params(module)


def add_dora_weights(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        if lora_query:
            layer.self_attn.q_proj = LinearLayerWithDoRA(layer.self_attn.q_proj, lora_r, lora_alpha)
        if lora_value:
            layer.self_attn.v_proj = LinearLayerWithDoRA(layer.self_attn.v_proj, lora_r, lora_alpha)
        if lora_key:
            layer.self_attn.k_proj = LinearLayerWithDoRA(layer.self_attn.k_proj, lora_r, lora_alpha)
        if lora_projection:
            layer.self_attn.o_proj = LinearLayerWithDoRA(layer.self_attn.o_proj, lora_r, lora_alpha)
        if lora_mlp:
            layer.mlp.gate_proj = LinearLayerWithDoRA(layer.mlp.gate_proj, lora_r, lora_alpha)
            layer.mlp.up_proj = LinearLayerWithDoRA(layer.mlp.up_proj, lora_r, lora_alpha)
            layer.mlp.down_proj = LinearLayerWithDoRA(layer.mlp.down_proj, lora_r, lora_alpha)

    set_trainable_params(model)
    return model
