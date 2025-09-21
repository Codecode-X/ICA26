import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class CAT(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(CAT, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar, adjust=False, attn_mask=None):
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        if adjust:
            src2 = self.self_attn(src, tar, tar, attn_mask=None, key_padding_mask=None)[0]
        else:
            src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1)
        return src
