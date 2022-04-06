#!/usr/bin/env python3

import collections
import ms2graph
import graph_utils
from torch_geometric.data import Data
from torch_geometric import utils
import networkx as nx
import pickle
import numpy as np
from os import path, listdir
import torch
import time
from torch.nn.functional import one_hot
from collections import Counter


class Spectrum(object):
    """
    Class containing Spectrum data.

    Attributes
    ----------
    idx : int
        Spectrum scan index
    graph : torch_geometric.data.Data
        Graph in pyG format. It contains mainly node features and edges
        definitions.
    intensities : np.array
        Numpy array containing intensities values for each peak
    mz_array : np.array
        Numpy array containing mass to charge values of the mass spectrum
    precursor_data : dict
        Dictionary containing info about precursor (mass, charge, m/z)
        keys{'m': ,'z':, 'mz }
    peptide_seq : str, optional
        Peptide sequence corresponding to spectrum

    Methods
    -------
    show_spectrum(out='../tmp/spectrum_plot_{idx}.html'.format(idx=self.idx))
        Save spectrum plot on a html file. Relies on spectrum_utils package.
    show_graph()
        Display a graph representation using NetworkX (network analysis package
        for python)
    print_stats()
        Print graph and spectrum statistics

    """

    def __init__(self, data, pickle_dir=None, y_edges=False, only_stats=False, compute_completness=True,
                 compute_n_paths=False):
        self.idx = data['idx']
        self.mz_array = data['mz_array']

        self.intensities = data['intensities']

        self.precursor = {'m': data['p_m'], 'z': data['p_z'], 'mz':
                          data['p_mz']}
        #self.edges_aa = data['edges_aa']
        #self.edge_features = data['edge_features']
        self.node_features = data['node_features']
        if 'peptide_seq' in data.keys():
            self.peptide_seq = data['peptide_seq']
        self.max_miss_aa = data['max_miss_aa']
        self.graph = self.__generate_pyg_graph(data['from'], data['to'],
                                               data['edges_aa'], None, y_edges=y_edges,
                                               max_miss_fragments=self.max_miss_aa)

        if 'b_max_miss' in data and 'y_max_miss' in data:
            self.missing_values = (data['b_max_miss'], data['y_max_miss'], data['all_max_miss'])
        if 'mass_ions' in data:
            self.mass_ions = data['mass_ions']
        if y_edges:
            self.true_edges = len(self.graph.y)

        self.set_stats(compute_completness, compute_n_paths)
        if only_stats:
            self.graph = None
        if pickle_dir is not None:
            with open(path.join(pickle_dir, str(self.idx) + ".pickle"), 'wb') as f:
                pickle.dump(self, f)


    def __generate_pyg_graph(self, _from, _to, _edges, _edges_features, use_intensities=True,
                             node_padding=None, set_y=True, y_edges=False, max_miss_fragments=2,
                             collapse_edge_representation=None):
        """
        Encodes _from, _to, _edges lists into a torch_geometric.data.Data class instance.

        Parameters
        ----------
        _from : list of int
            List of nodes indices
        _to : list of int
            List of nodes indices
        _edges : list of str
            List of amino acids corresponding to each edge
        use_intensities : bool
            Boolean parameter to either consider intensities as node features,
            should be present in Spectrum instance.
        node_padding : int
            Pad node embeddings with n zeros
        set_y : bool
            Set y into data graph structure if petide sequence is given

        Returns
        -------
        graph : torch_geometric.data.Data
            Pytorch geometric graph class instance
        """
        assert len(_from) == len(_to) == len(_edges)
        num_nodes = len(self.mz_array)
        print(f"Generating graph with {num_nodes} nodes")

        # Managing edges collasping (usually collapsed for sequential predictions)
        if collapse_edge_representation is None:
            collapse_edge_representation = not y_edges

        # Set edge features
        if y_edges or (not collapse_edge_representation and not y_edges):
            padding = max_miss_fragments
        else:
            padding = False
        _edge_attr = [Spectrum._encode_sequence(edge, flatten=True, padding=padding,
                                                collapse_edge_representation=collapse_edge_representation).unsqueeze(0)
                         for edge in _edges]
        _tmp_f = _from[0]
        _edges_nodes_idx = [0]
        for idx, el in enumerate(_from[1:]):
            if el != _tmp_f:
                _tmp_f = el
                _edges_nodes_idx.append(idx)
                continue
            _edges_nodes_idx.append(_edges_nodes_idx[-1])

        _edges_nodes_idx = [idx for idx, f in enumerate(_from) ]
        _filtering_edges = [idx for idx, edge_feat in enumerate(_edge_attr)\
                            if any([(edge_feat == k).all() for k in\
                                    _edge_attr[_edges_nodes_idx[idx]:idx]])] # filter different amino acids permutations
        _from = [el for i, el in enumerate(_from) if i not in _filtering_edges]
        _to = [el for i, el in enumerate(_to) if i not in _filtering_edges]
        _edge_attr = [el for i, el in enumerate(_edge_attr) if i not in _filtering_edges]
        if _edges_features is not None:
            _edge_features = [el for i, el in enumerate(_edges_features) if i not in _filtering_edges]
            _edge_features = torch.tensor(_edge_features, dtype=torch.float).unsqueeze(1)
        edge_index = torch.tensor([_from, _to], dtype=torch.long)

        edge_attr = torch.cat(_edge_attr, dim=0)
        if _edges_features is not None:
            edge_attr = torch.cat((edge_attr, _edge_features), dim=1)
        edge_attr = edge_attr.type(torch.float)


        x = torch.tensor(self.mz_array, dtype=torch.float).unsqueeze(1)
        if use_intensities and self.intensities is not None:
            assert len(self.intensities) == num_nodes
            x_intensities = torch.tensor(self.intensities, dtype=torch.float).unsqueeze(1)
            x = torch.cat([x, x_intensities], dim=1)
        if self.node_features is not None:
            x_features = torch.tensor(self.node_features, dtype=torch.float)
            x = torch.cat([x, x_features], dim=1)
        if node_padding is not None:
            dim_nodes = x.shape[1]
            pad_size = node_padding - dim_nodes
            pad_tensor = torch.zeros((x.shape[0], pad_size))
            x = torch.cat((x, pad_tensor), dim=1)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if set_y and self.peptide_seq is not None:
            if not y_edges:
                y = Spectrum._encode_sequence(self.peptide_seq,
                                              get_classes_list=True)
            else:
                s, e = ms2graph.get_source_sink_ions(self.mz_array, self.precursor['m'],
                                                     orientation=-1)
                y = ms2graph.right_path_all_paths(self.peptide_seq, _from, _to, _edges,
                                                  -1, 1, self.mz_array, s, verbose=True,
                                                  first_recursion=True, start_time=time.time())
                flat_list = [item for sublist in y for item in sublist]
                y = list(set(flat_list))
                print(f"Total number of edges = {len(_from)} , of which true = {len(y)} (ratio = {len(y) / len(_from)})")
            graph.y = y

        return graph

    def show_graph(self):
        """
        Show graph structure using NetworkX.

        """
        assert self.graph is not None
        nx_graph = utils.to_networkx(self.graph,
                                     to_undirected=False)
        node_attributes = {key: round(value, 2) for (key, value) in
                           enumerate(list(self.intensities))}

        # edge_attributes missing because of many overlapping edges

        nx_graph = nx.relabel_nodes(nx_graph, node_attributes)
        pos = nx.kamada_kawai_layout(nx_graph)
        nx.draw_networkx(nx_graph, pos=pos)
        plt.show()


    def set_stats(self, compute_completness=True, compute_n_paths=False):
        if self.graph is None:
            return
        self.num_nodes = self.graph.num_nodes
        self.num_edges = self.graph.num_edges
        self.isolated_nodes = int(self.graph.has_isolated_nodes())
        self.avg_node_degree = float(self.graph.num_edges /
                                        self.graph.num_nodes)
        self.peptide_seq_len = len(self.peptide_seq)
        self.seq_len_node_ratio = self.graph.num_nodes / len(self.peptide_seq)

        if compute_completness and not hasattr(self, 'missing_values'):
            missing_values = ms2graph.check_spectrum_completness(self.mz_array,
                                                                 self.peptide_seq)
            self.missing_values = missing_values[0]
            self.mass_ions = missing_values[1]

        if compute_n_paths and not hasattr(self, 'n_paths'):
            n_paths =  graph_utils.paths_number_source_sink(
                graph_utils.torch_graph_to_dict_graph(self.graph),
                ms2graph.get_source_sink_ions(self.mz_array,
                                              self.precursor['m']))
            self.n_paths = n_paths


        if hasattr(self, 'true_edges'):
            self.tf_edges_ratio = self.true_edges / self.num_edges


    def get_stats(self, compute_completness=False, compute_n_paths=False):
        data = {}
        stats_attributes = ['num_nodes', 'num_edges', 'isolated_nodes', 'avg_node_degree',
                            'peptide_seq_len', 'seq_len_node_ratio', 'max_miss_aa']

        optional_attributes = ['missing_values', 'n_paths', 'true_edges', 'tf_edges_ratio', 'mass_ions']

        if hasattr(self, 'graph') and ((not hasattr(self, 'missing_values') and compute_completness) or\
           (not hasattr(self, 'n_paths') and compute_n_paths)):
            self.set_stats(compute_completness, compute_n_paths)

        for opt_attr in optional_attributes:
            if hasattr(self, opt_attr):
                stats_attributes.append(opt_attr)


        for attribute in stats_attributes:
            data[attribute] = getattr(self, attribute)
        return data

    def print_stats(self, compute_completness=False, compute_n_paths=False):
        """
        Print basic information about the graph.

        (nodes number, edges number, isolated nodes, directed or not)
        """

        data = self.get_stats(compute_completness=compute_completness, compute_n_paths=compute_n_paths)

        for key, value in data.values():
            if type(value) == float:
                print(f"{key:17} = {value:7.3f}")
            if type(value) == int:
                print(f"{key:17} = {value:10d}")
            else: # consider string
                print(f"{key:17} = {str(value)}")




    @staticmethod
    def _encode_sequence(psm_peptide_seq, flatten=False, padding=None, get_classes_list=False,
                         collapse_edge_representation=False):
        """
        Return one hot encoded tensor for a given protein peptide sequence

        """
        _AMINOACIDS = "ARNDCQEGHILKMFPSTWYV"
        int_rep = torch.tensor([_AMINOACIDS.index(el) for el in psm_peptide_seq], dtype=torch.long)
        if get_classes_list:
            return int_rep
        encoded_seq = one_hot(int_rep, num_classes=20)
        if padding is not None and padding > encoded_seq.shape[0]:
            pad_size = padding - encoded_seq.shape[0]
            pad_tensor = torch.zeros((pad_size, 20))
            encoded_seq = torch.cat((encoded_seq, pad_tensor))
        if flatten:
            if collapse_edge_representation:
                encoded_seq = torch.sum(encoded_seq, dim=0)
            else:
                encoded_seq = torch.flatten(encoded_seq)
        return encoded_seq


    @staticmethod
    def decode_sequence(edge_attr):
        """
        Return list of psms given edge attr

        """
        _AMINOACIDS = "ARNDCQEGHILKMFPSTWYV"
        _aa_dict = {}
        for i, aa in enumerate(_AMINOACIDS):
            _aa_dict[i] = aa
            
        n_aa = edge_attr.shape[1] // 20

        edge_attr = torch.reshape(edge_attr, (edge_attr.shape[0], 20, n_aa))
        labels = torch.argmax(edge_attr, dim=1)
        labels = torch.reshape(labels, (edge_attr.shape[0], n_aa))
        labels = labels.tolist()

        return labels




if __name__ == "__main__":
    # Graph Spectrum instance test
    # print("\n\n Graph Instance Test")
    # mzFile = "../data/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
    # mzScan = 32688
    # psmPeptide = "IANVQSQLEK"
    # precursorCharge = 2
    # print(f"Recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}")
    # fragment_tol_mass = 10
    # fragment_tol_mode = 'ppm'
    # data = read_sample_spectrum(mzScan, mzFile, False,
    #                             fragment_tol_mass,
    #                             fragment_tol_mode,
    #                             psmPeptide=psmPeptide)
    # sample_spectrum = Spectrum(data)
    # sample_spectrum.print_stats()
    # sample_spectrum.show_graph()
    pass
