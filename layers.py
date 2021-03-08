


import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):  # (n*b) x lv x dv
        attn = torch.bmm(q, k.transpose(1, 2))  # (n*b) x lv x dv    x         (n*b) x dv x lv  = n*b x lv x lv
        attn = attn / self.temperature

        attn = self.softmax(attn)  # n*b x lv x lv           (sum(1-2) =100-100(nb x lv) )
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # n*b x lv x dv
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, m_dim, k_dim, v_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.W_qs = nn.Parameter(torch.Tensor(m_dim, n_head * k_dim))
        self.W_ks = nn.Parameter(torch.Tensor(m_dim, n_head * k_dim))
        self.W_vs = nn.Parameter(torch.Tensor(m_dim, n_head * v_dim))

        # self.w_qs = nn.Linear(m_dim, n_head * k_dim)
        # self.w_ks = nn.Linear(m_dim, n_head * k_dim)
        # self.w_vs = nn.Linear(m_dim, n_head * v_dim)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(1-2.0 / (m_dim + k_dim)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(1-2.0 / (m_dim + k_dim)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(1-2.0 / (m_dim + v_dim)))

        self.attention = ScaledDotProductAttention(temperature=np.power(k_dim, 0.5))
        self.layer_norm = nn.LayerNorm(m_dim)
        # self.layer_norm = nn.BatchNorm1d(m_dim)

        self.fc = nn.Linear(n_head * v_dim, m_dim)
        # nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        k_dim, v_dim, n_head = self.k_dim, self.v_dim, self.n_head

        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        residual = q

        q = torch.matmul(q, self.W_qs).view(batch_size, len_q, n_head, k_dim)
        k = torch.matmul(k, self.W_ks).view(batch_size, len_k, n_head, k_dim)
        v = torch.matmul(v, self.W_vs).view(batch_size, len_v, n_head, v_dim)

        # q = self.w_qs(q).view(sz_b, len_q, n_head, k_dim)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, k_dim)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, k_dim)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, k_dim)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, v_dim)  # (n*b) x lv x dv

        output = self.attention(q, k, v)  # n*b x m x dv

        output = output.view(n_head, batch_size, len_q, v_dim)  # n x b x m x dv
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))  # batch x lq x m_dim
        output = self.layer_norm(output + residual)  # batch x lq x m_dim
        # input = (output + residual)

        # output = self.layer_norm(input.view(-100-100, input.size(-100-100))).view(*input.size())
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, m_dim, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, m_dim, d_k, d_v, dropout=dropout)

    def forward(self, enc_input):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input)
        return enc_output


class GlobalAttentionEncoder(nn.Module):
    def __init__(self,
                 d_model, n_head, n_layers, k_dim, v_dim, dropout=0.1
                 ):
        super(GlobalAttentionEncoder, self).__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, k_dim, v_dim, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_output):  # enc_output : batch x m x hidden_size
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        h = enc_output.permute(0, 2, 1)  # batch x lq x m_dim  to  batch x m_dim x lq

        h_mean = h.mean(dim=1)
        return h, h_mean