import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils_good as utils

class Attention(nn.Module):
    def __init__(self, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.d_k = cfg.MODEL.ATT_HIDDEN_SIZE // num_heads

        self.Wq = nn.Linear(cfg.MODEL.RNN_SIZE, cfg.MODEL.ATT_HIDDEN_SIZE, bias=False)
        self.Wk = nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_HIDDEN_SIZE, bias=False)
        self.Wv = nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_HIDDEN_SIZE, bias=False)
        self.Wo = nn.Linear(cfg.MODEL.ATT_HIDDEN_SIZE, cfg.MODEL.ATT_HIDDEN_SIZE, bias=False)

        self.scale = 1.0 / (self.d_k ** 0.5)  # Scaled dot-product
        self.dropout = nn.Dropout(cfg.MODEL.ATT_HIDDEN_DROP) if cfg.MODEL.ATT_HIDDEN_DROP > 0 else None
        self.norm = nn.LayerNorm(cfg.MODEL.ATT_HIDDEN_SIZE)
        self.gate = nn.Sequential(
            nn.Linear(2 * cfg.MODEL.ATT_HIDDEN_SIZE, cfg.MODEL.ATT_HIDDEN_SIZE),
            nn.Sigmoid()
        )

    def split_heads(self, x, batch_size):
        """ Split the last dimension into (heads, depth)."""
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, h, att_feats):
        batch_size = h.size(0)
        q = self.split_heads(self.Wq(h), batch_size)
        k = self.split_heads(self.Wk(att_feats), batch_size)
        v = self.split_heads(self.Wv(att_feats), batch_size)

        # Scaled dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        att = torch.matmul(attn_weights, v)
        att = att.transpose(1, 2).contiguous().view(batch_size, -1, cfg.MODEL.ATT_HIDDEN_SIZE)
        att = self.Wo(att)

        # Apply gating mechanism
        combined = torch.cat([att, h], dim=-1)
        gate_values = self.gate(combined)
        att = gate_values * att + (1 - gate_values) * h  # Gated combination

        att = self.norm(att)  # Layer normalization
        return att
