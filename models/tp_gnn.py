import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class TPGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__(aggr='add')  # 论文要求 add
        self.lin = nn.Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, trust_scores, prop_coeffs):
        x = self.lin(x)

        row, col = edge_index  # row: source(j), col: target(i)
        trust_j = trust_scores[row]  # [E]

        return self.propagate(edge_index, x=x, t_gate=trust_j, p_coeff=prop_coeffs)

    def message(self, x_j, t_gate, p_coeff):
        # 稳定性：避免 gate 极端饱和导致梯度消失，可做轻度裁剪（可按论文设定范围调整）
        t_gate = torch.clamp(t_gate, 0.0, 1.0)          # trust 通常在[0,1]
        p_coeff = torch.clamp(p_coeff, -5.0, 5.0)       # 防止传播系数过大

        gate = torch.sigmoid(t_gate * p_coeff).unsqueeze(-1)
        msg = gate * x_j

        if self.dropout > 0:
            msg = F.dropout(msg, p=self.dropout, training=self.training)
        return msg


class TPGNN(nn.Module):
    """
    工程增强：
    - dropout/residual/layernorm (可控)
    - freeze/unfreeze 接口（用于主结果 vs 消融）
    """
    def __init__(self, input_dim, hidden_dim, output_dim=153, dropout=0.1, use_layernorm=True, use_residual=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual

        self.conv1 = TPGNNConv(input_dim, hidden_dim, dropout=dropout)
        self.conv2 = TPGNNConv(hidden_dim, hidden_dim, dropout=dropout)
        self.conv3 = TPGNNConv(hidden_dim, output_dim, dropout=0.0)

        self.ln1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.ln3 = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, x, edge_index, trust_scores, prop_coeffs):
        h1 = self.conv1(x, edge_index, trust_scores, prop_coeffs)
        h1 = F.relu(self.ln1(h1))

        h2 = self.conv2(h1, edge_index, trust_scores, prop_coeffs)
        if self.use_residual:
            h2 = h2 + h1
        h2 = F.relu(self.ln2(h2))

        out = self.conv3(h2, edge_index, trust_scores, prop_coeffs)
        out = self.ln3(out)
        return out

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
