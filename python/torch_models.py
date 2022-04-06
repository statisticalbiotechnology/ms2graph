#!/usr/bin/env python3

from utils import MZEncoder
import torch
from torch._C import dtype
from torch.nn.modules.activation import ReLU, Sigmoid
import torch_geometric
from torch_geometric.nn.conv import GATv2Conv, GatedGraphConv, GCNConv
from torch_geometric.nn.glob import GlobalAttention, GraphMultisetTransformer, Set2Set, global_add_pool
from torch import nn
from torch_geometric.nn.glob.glob import global_add_pool, global_max_pool
from utils import generate_tgt_mask, PositionalEncoding


class GNN2Seq(torch.nn.Module):
    def __init__(self, node_state_dim, edge_dim, out_dim=20):
        super(GNN2Seq, self).__init__()
        self.node_state_dim = node_state_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        self.encoder = torch_geometric.nn.Sequential('x, edge_index, edge_attr',
                [(GATv2Conv(in_channels=node_state_dim, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )
        # Propagation/Message Passing model
        self.propagation = torch_geometric.nn.Sequential('x, edge_index, edge_attr',
                [(GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )

        # Output model for state k+1 construction
        self.out_next_state = nn.Sequential(
            nn.Linear(self.node_state_dim*8, self.node_state_dim*8),
                                 nn.ReLU(),
                                 nn.Linear(self.node_state_dim*8, self.node_state_dim*8)
                                )

        # Output model for sequence element prediction
        _h_gate = nn.Sequential(nn.Linear(self.node_state_dim * 8, node_state_dim * 8),
                                nn.ReLU(),
                                nn.Linear(self.node_state_dim * 8, 1))
        self.global_attention = GlobalAttention(_h_gate, nn.Linear(self.node_state_dim*8,
                                                                   self.node_state_dim*8))
        #self.global_attention = Set2Set(self.node_state_dim*64 , self.node_state_dim*64)
        #self.global_attention = global_add_pool

        self.out_element = nn.Sequential(
            nn.Linear(self.node_state_dim * 8,self.node_state_dim *4),
            nn.ReLU(),
            nn.Linear(self.node_state_dim*4 , self.out_dim)
                                         )
        self.soft_max = nn.Softmax(dim=1)

        self.out_element.apply(self.__init_w)


    def __init_w(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, input, batch, device, n_steps=5):
        out = torch.tensor([], device=device)
        edge_index = input.edge_index
        edge_attr = input.edge_attr
        state = torch.nan_to_num(input.x)
        # print(state.shape)
        # print(torch.count_nonzero(torch.isnan(state)))
        state = self.encoder(state, edge_index=edge_index, edge_attr=edge_attr)
        # print(torch.count_nonzero(torch.isnan(state)))
        # print(state)
        for i in range(n_steps):
            # Compute propagation
            state = self.propagation(state, edge_index=edge_index, edge_attr=edge_attr)

            # Generate next state
            next_state = self.out_next_state(state)

            # Generate prediction for step i
            graph_level_representation = self.global_attention(state, batch) #
            out_element = self.out_element(graph_level_representation)
            #out = out_element
            out = torch.cat([out, out_element])
            # Update state
            state = next_state
        return out



class GNN2Edges(torch.nn.Module):
    # Work in progress
    def __init__(self, node_state_dim, edge_dim, out_dim=20):
        super(GNN2Edges, self).__init__()
        self.node_state_dim = node_state_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        # Propagation/Message Passing model
        self.propagation = torch_geometric.nn.Sequential('x, edge_index, edge_attr',
                [(GATv2Conv(in_channels=node_state_dim, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )

        # Output model for prediction at edge level
        self.bilinear = nn.Bilinear(node_state_dim*2*16, edge_dim, node_state_dim *8 + edge_dim)
        self.out_edge = nn.Sequential(nn.Linear(node_state_dim * 8 + edge_dim, node_state_dim * 8 + edge_dim),
                                        nn.ReLU(),
                                        nn.Linear(node_state_dim*8 + edge_dim, node_state_dim),
                                        nn.ReLU(),
                                        nn.Linear(node_state_dim, 1),
                                        nn.Sigmoid()
                                        )

    def forward(self, input, batch, device):
        out = torch.tensor([], device=device)
        edge_index = input.edge_index
        edge_attr = input.edge_attr
        state = input.x
        # State prediction
        state = self.propagation(state, edge_index=edge_index, edge_attr=edge_attr)

        # Predict edges
        node_embeddings_0 = torch.index_select(state, dim=0, index=edge_index[0])
        node_embeddings_1 = torch.index_select(state, dim=0, index=edge_index[1])

        edge_repre = torch.cat((node_embeddings_0, node_embeddings_1), dim=1)
        bi = self.bilinear(edge_repre, edge_attr)
        out = self.out_edge(bi)

        return out



class GNN2Transformer(torch.nn.Module):
    def __init__(self, node_state_dim, edge_dim, out_dim=20):
        super(GNN2Transformer, self).__init__()
        self.node_state_dim = node_state_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.pos_encoding = PositionalEncoding(22, max_len=40)
        self.mz_encoding = MZEncoder()

        self.encoder = torch_geometric.nn.Sequential('x, edge_index, edge_attr',
                [(GATv2Conv(in_channels=node_state_dim, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*32,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 (GATv2Conv(in_channels=node_state_dim*32, out_channels=node_state_dim*32,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*32, out_channels=node_state_dim*32,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*32, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*8,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=22,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=22, nhead=2, dim_feedforward=1028)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.out_linear = nn.Linear(22, 22)
        self.soft_max = nn.Softmax(dim=2)

    def __init_w(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input, tgt, batch=None, device='cuda'):
        out = torch.tensor([], device=device)
        edge_index = input.edge_index
        edge_attr = input.edge_attr
        state = input.x # torch.nan_to_num(input.x)
        #state = self.mz_encoding(state[:, :1])
        #state = torch.cat((state, input.x), dim=1)
        state = self.encoder(state, edge_index=edge_index, edge_attr=edge_attr)
        # Decoding

        #print(state.shape)
        state, mask = torch_geometric.torch_geometric.utils.to_dense_batch(state, batch)
        state = torch.transpose(state, 0, 1)
        #print(state.shape)
        tgt = self.__tgt_tokenize(tgt, batch, device)
        tgt = self.pos_encoding(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[0]).to(device)
        out = self.transformer_decoder(tgt, state)
        out = self.out_linear(out)
        #out = self.soft_max(out)
        return out



    def __tgt_tokenize(self, tgt, batch, device):
        # TODO putting precursor info as StartOfString might help
        shape = (int(tgt.shape[0] / (int(torch.max(batch)) + 1)),
                                 int(torch.max(batch)) + 1)
        tgt = torch.reshape(tgt, shape)
        sos = torch.tensor([20 for a in range(tgt.shape[1])], device=device)
        sos = sos.unsqueeze(0)

        tgt = torch.cat((sos, tgt))
        tgt = torch.nn.functional.one_hot(tgt, 22).float()
        return tgt
