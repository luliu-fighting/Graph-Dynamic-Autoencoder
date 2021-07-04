import torch
import torch.nn as nn
import numpy as np
import dgl
import dgl.function as fn

def construct_graph(data, neighbor=5):
    n, c = data.shape
    # nodes feature
    g = dgl.DGLGraph()
    g.add_nodes(n)
    g.ndata['h'] = torch.Tensor(data)
    # edges feature
    for i in range(n):
        if i - neighbor < 0:
            left = 0
            right = left + neighbor * 2
        elif i + neighbor > n - 1:
            right = n - 1
            left = right - neighbor * 2
        else:
            left = i - neighbor
            right = i + neighbor
        neighbor_nodes = [x for x in range(left, right+1) if x != i]
        di = []
        for j in range(neighbor*2):
            d = np.linalg.norm(data[i,:]-data[neighbor_nodes[j],:])
            di.append(d)
        beta = np.mean(di)
        di = [np.exp(-x/beta) for x in di]
        di = [x/np.sum(di) for x in di]
        g.add_edges([i]*neighbor*2, neighbor_nodes)
        g.edges[[i]*neighbor*2, neighbor_nodes].data['h_e'] = torch.Tensor(di)
    g = dgl.add_self_loop(g)
    for i in range(n):
        g.edges[i, i].data['h_e'] = torch.Tensor([1.0])
    return g

def updataEdgefeature(data, neighbor=5):
    n = data.shape[0]
    data = data.cpu().numpy()
    dis = []
    for i in range(n):
        if i - neighbor < 0:
            left = 0
            right = left + neighbor * 2
        elif i + neighbor > n - 1:
            right = n - 1
            left = right - neighbor * 2
        else:
            left = i - neighbor
            right = i + neighbor
        neighbor_nodes = [x for x in range(left, right+1) if x != i]
        di = []
        for j in range(neighbor*2):
            d = np.linalg.norm(data[i,:]-data[neighbor_nodes[j],:])
            di.append(d)
        beta = np.mean(di)
        di = [np.exp(-x/beta) for x in di]
        di = [x/np.sum(di) for x in di]
        dis.extend(di)
    for i in range(n):
        dis.append(1.0)
    dis = torch.Tensor(dis)
    dis = dis.cuda()
    return dis

class GAElayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, SelfTraining=False):
        super(GAElayer, self).__init__()
        # if input_dim is None or output_dim is None:
        #     raise ValueError
        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training_self = SelfTraining  # Indicates whether to pre-train layer by layer or train the entire network

        self.msg_func = fn.u_mul_e('h', 'h_e', 'm')
        self.reduce_func = fn.sum('m', 'h')

        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=True),
            nn.ReLU()  # Unified activation with ReLU
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.out_features, self.in_features, bias=True),
            nn.ReLU()
        )

    def forward(self, inputs):
        g, h = inputs
        g.ndata['h'] = h
        g.edata['h_e'] = updataEdgefeature(h, neighbor=5)
        g.update_all(self.msg_func, self.reduce_func)
        h = g.ndata.pop('h')
        out = self.encoder(h)
        if self.is_training_self:
            return g, self.decoder(out)
        else:
            return g, out

    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter
    def is_training_layer(self, other: bool):
        self.is_training_self = other

class GraphStackAE(nn.Module):
    """
    Construct the whole network with layers_list
    """
    def __init__(self, layers_list=None):
        super(GraphStackAE, self).__init__()
        self.layers_list = layers_list
        self.initialize()

        self.encoder_1 = self.layers_list[0]
        self.encoder_2 = self.layers_list[1]
        self.decoder_1 = self.layers_list[2]
        self.decoder_2 = self.layers_list[3]

    def initialize(self):
        for layer in self.layers_list:
            layer.is_training_layer = False
            # for param in layer.parameters():
            #     param.requires_grad = True

    def forward(self, inputs):
        g, out = inputs
        g, out = self.encoder_1([g, out])
        g, out = self.encoder_2([g, out])
        self.hidden_feature = out
        g, out = self.decoder_1([g, out])
        g, out = self.decoder_2([g, out])
        return out


