import torch
import math
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self,input_size,head_em,num_heads,out_size,mask=False):
        super().__init__()

        self.input_size = input_size
        self.head_em = head_em
        self.num_heads = num_heads
        self.out_size = out_size
        self.mask= mask

        self.q=nn.Linear(self.input_size,self.head_em*self.num_heads)
        self.k=nn.Linear(self.input_size,self.head_em*self.num_heads)
        self.v=nn.Linear(self.input_size,self.head_em*self.num_heads)

        self.scale = math.sqrt(self.head_em)

        self.em_full_size=self.head_em*self.num_heads

        self.softmax=nn.Softmax(dim=-1)

        self.out_layer=nn.Linear(self.em_full_size,self.out_size)

        self.drop_out=nn.Dropout(.1)

    def forward(self, query, keys, values, padding_mask=None):
        Bq, Sq, _ = query.shape
        Bk, Sk, _ = keys.shape
        Bv, Sv, _ = values.shape

        # Q/K/V
        q_values = self.q(query).reshape(Bq, Sq, self.num_heads, self.head_em).transpose(1, 2)
        k_values = self.k(keys).reshape(Bk, Sk, self.num_heads, self.head_em).transpose(1, 2)
        v_values = self.v(values).reshape(Bv, Sv, self.num_heads, self.head_em).transpose(1, 2)

        # Attention scores
        qk = torch.einsum("BHQE, BHKE -> BHQK", q_values, k_values)
        qk = qk / self.scale

        # 1) Causal Mask
        if self.mask:
            causal_mask = torch.triu(torch.ones_like(qk), diagonal=1)
            qk = qk.masked_fill(causal_mask == 1, float('-inf'))

        # 2) Padding Mask   (B, K) → (B, 1, 1, K)
        if padding_mask is not None:
            # padding_mask shape: (batch, seq_len)
            # Expand to match scores: (B, 1, 1, Sk)
            pad = padding_mask[:, None, None, :].to(qk.device)


            # True = ignore
            qk = qk.masked_fill(pad == True, float('-inf'))

        # Softmax
        qk_prob = self.softmax(qk)
        
        qk_prob = self.drop_out(qk_prob)

        # Weighted sum
        qv = torch.einsum("BHQK, BHKE -> BHQE", qk_prob, v_values)

        # Back to (B, S, E)
        qv = qv.transpose(1, 2).contiguous().reshape(Bq, Sq, self.em_full_size)

        out = self.out_layer(qv)

        return out
