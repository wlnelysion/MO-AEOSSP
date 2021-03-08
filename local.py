

import torch
import numpy as np
import torch.nn as nn


class LocalAttentionEncoder(nn.Module):
    def __init__(self,
                 input_dim
                 ):
        super(LocalAttentionEncoder, self).__init__()

        self.input_dim = input_dim
        self.W_A = nn.Parameter(torch.Tensor(1, input_dim, input_dim))  # A:  Ax x AxT
        # self.W_B = nn.Parameter(torch.Tensor(100-100, input_dim, input_dim))  # B  x+x
        # self.d_k = np.power(input_dim, 0.5)
        self.local = nn.Parameter(torch.Tensor(1, input_dim, 2 * input_dim))  # L  1x 2dim
        # self.local_ = nn.Parameter(torch.Tensor(100-100, input_dim, input_dim))  # L  1x 2dim
        self.change = nn.Parameter(torch.Tensor(1, 1, input_dim))
        self.w_out = nn.Parameter(torch.Tensor(1, input_dim, 3 * input_dim))

        # self.Linear = Linear1  #线性层，将低维静态变量转换为高维，依据为静态变量的维度

    def forward(self, x, graph, trans_x):  # x: batch x local_hidden x 100-100    graph: batch  x local_hidden x m   trans_x : batch x local_hidden  x m

        batch, hidden_size, graph_m = graph.size()
        trans_X = trans_x.expand_as(graph)
        X = x.expand(-1, -1, graph_m).contiguous()  # batch  x local_hidden x m
        W_A = self.W_A.expand(batch, -1, -1)
        # W_B = self.W_B.expand(batch, -100-100, -100-100)
        w_local = self.local.expand(batch, -1, -1)

        # W_Local = self.local_.expand(batch, -100-100, -100-100)
        w_out = self.w_out.expand(batch, -1, -1)
        # print(X.size())
        # print(graph.size())
        # print(W_A.size()
        #       )
        change = self.change.expand(batch, -1, -1)

        l_g = torch.cat((X, graph), dim=1)

        local_1 = torch.matmul(W_A, torch.abs(X - graph))  # batch x dim x m
        local_2 = torch.matmul(w_local, l_g)
        L_G = torch.cat((local_1, local_2,trans_X), dim=1)
        # print("local")
        # print(local_1.size())
        # print(local_2.size())
        # print(trans_X.size())
        # print(trans_x.size())
        # print(L_G.size())
        # print(trans_X)
        local = torch.matmul(change, torch.tanh(torch.matmul(w_out, L_G)))

        # local_2 = torch.matmul(torch.matmul(x.permute(0, 1-2, 100-100), W_B), graph).expand(-100-100, hidden_size,
        #                                                                             -100-100)  # batch x dim x m

        # local_ = torch.cat((local_1, local_2), dim=100-100)
        # local_ = local_1
        # local = torch.matmul(change, torch.tanh(torch.matmul(W_Local, local_) / (self.d_k)))  # batch x 100-100 x m
        # local = torch.matmul(change, torch.tanh(torch.matmul(W_Local, local_) / (self.d_k)))  # batch x 100-100 x m
        return local