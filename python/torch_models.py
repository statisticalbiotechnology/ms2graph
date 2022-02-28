#!/usr/bin/env python3

import torch
from torch.nn.modules.activation import ReLU, Sigmoid
import torch_geometric
from torch_geometric.nn.conv import GATv2Conv, GatedGraphConv, GCNConv
from torch_geometric.nn.glob import GlobalAttention, GraphMultisetTransformer, Set2Set, global_add_pool
from torch import nn
from torch_geometric.nn.glob.glob import global_add_pool, global_max_pool

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
                 (GATv2Conv(in_channels=node_state_dim*8, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )
        # Propagation/Message Passing model
        self.propagation = torch_geometric.nn.Sequential('x, edge_index, edge_attr',
                [(GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*16,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*32,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*32, out_channels=node_state_dim*64,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*64, out_channels=node_state_dim*64,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*64, out_channels=node_state_dim*64,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )

        # Output model for state k+1 construction
        self.out_next_state = nn.Sequential(
            nn.Linear(self.node_state_dim*64, self.node_state_dim*16),
                                 nn.ReLU(),
                                 nn.Linear(self.node_state_dim*16, self.node_state_dim*16)
                                )

        # Output model for sequence element prediction
        _h_gate = nn.Sequential(nn.Linear(20,1),)
        self.global_attention = GlobalAttention(_h_gate, nn.Linear(self.node_state_dim*64,
                                                                   self.node_state_dim*64))
        #self.global_attention = Set2Set(self.node_state_dim*64 , self.node_state_dim*64)
        #self.global_attention = global_add_pool

        self.out_element = nn.Sequential(
            nn.Linear(self.node_state_dim * 64,self.node_state_dim *16),
            nn.ReLU(),
            nn.Linear(self.node_state_dim*16 , self.out_dim)
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
        state = input.x
        state = self.encoder(state, edge_index=edge_index, edge_attr=edge_attr)

        for i in range(n_steps):
            # Compute propagation
            state = self.propagation(state, edge_index=edge_index, edge_attr=edge_attr)

            # Generate next state
            next_state = self.out_next_state(state)

            # Generate prediction for step i
            graph_level_representation = global_add_pool(state, batch) #
            out_element = self.out_element(graph_level_representation)
            #out = out_element
            out = torch.cat([out, out_element])
            # Update state
            state = next_state
        return out



class GNN2Edge(torch.nn.Module):
    # Work in progress
    def __init__(self, node_state_dim, edge_dim, out_dim=20):
        super(GNN2Edge, self).__init__()
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
                 (GATv2Conv(in_channels=node_state_dim*16, out_channels=node_state_dim*32,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*32, out_channels=node_state_dim*64,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*64, out_channels=node_state_dim*64,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
                  'x, edge_index, edge_attr -> x'),
                 nn.ReLU(),
                 (GATv2Conv(in_channels=node_state_dim*64, out_channels=node_state_dim*64,
                        add_self_loops=True, heads=1, edge_dim=edge_dim),
              'x, edge_index, edge_attr -> x')]
        )

        # Output model for prediction at edge level
        self.out_edge = nn.Sequential(nn.Bilinear(node_state_dim * 2, edge_dim, node_state_dim + edge_dim),
                                        nn.ReLU(),
                                        nn.Linear(node_state_dim + edge_dim, node_state_dim + edge_dim),
                                        nn.ReLU(),
                                        nn.Linear(node_state_dim + edge_dim, node_state_dim),
                                        nn.ReLU(),
                                        nn.Linear(node_state_dim, 1),
                                        nn.Sigmoid()
                                        )

        self.out_element.apply(self.__init_w)
    def __init_w(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.fill_(0.01)
    def forward(self, input, batch, device, n_steps=5):
        out = torch.tensor([], device=device)
        edge_index = input.edge_index
        edge_attr = input.edge_attr
        state = input.x
        for i in range(n_steps):
            # Compute propagation
            state = self.propagation(state, edge_index=edge_index, edge_attr=edge_attr)

            # Predict edges
        return out



