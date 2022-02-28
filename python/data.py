#!/usr/bin/env python3
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric import utils
import networkx as nx
from torch_geometric.data.dataset import Dataset
from ms2graph import read_sample_spectrum, processSpectra, readPSMs
import pyteomics.mzml as mz
import pyteomics.mass as mass
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from os import path
import pickle
import glob



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

    def __init__(self, data):
        self.idx = data['idx']
        self.mz_array = data['mz_array']

        self.intensities = data['intensities']

        self.precursor = {'m': data['p_m'], 'z': data['p_z'], 'mz':
                          data['p_mz']}
        self.edges_aa = data['edges_aa']
        self.edge_features = data['edge_features']
        self.node_features = data['node_features']
        if 'peptide_seq' in data.keys():
            self.peptide_seq = data['peptide_seq']
        self.graph = self.__generate_pyg_graph(data['from'], data['to'],
                                               data['edges_aa'])

    def __generate_pyg_graph(self, _from, _to, _edges, use_intensities=True,
                             node_padding=None, set_y=True):
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
        print(len(_edges))
        # Set edge features
        _edge_attr = [SpectraDataset._encode_sequence(edge, flatten=True, collapse_edge_representation=True).unsqueeze(0)
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
        if self.edge_features is not None:
            _edge_features = [el for i, el in enumerate(self.edge_features) if i not in _filtering_edges]
            _edge_features = torch.tensor(_edge_features, dtype=torch.float).unsqueeze(1)
        edge_index = torch.tensor([_from, _to], dtype=torch.long)

        edge_attr = torch.cat(_edge_attr, dim=0)
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
            y = SpectraDataset._encode_sequence(self.peptide_seq,
                                   get_classes_list=True)
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

    def get_stats(self):
        data = {}
        data['num_nodes'] = self.graph.num_nodes
        data['num_edges'] = self.graph.num_edges
        data['isolated_nodes'] = int(self.graph.has_isolated_nodes())
        data['avg_node_degree'] = float(self.graph.num_edges /
                                        self.graph.num_nodes)
        data['peptide_seq_len'] = len(self.peptide_seq)
        data['seq_len_node_ratio'] = self.graph.num_nodes / len(self.peptide_seq)
        return data

    def print_stats(self):
        """
        Print basic information about the graph.

        (nodes number, edges number, isolated nodes, directed or not)
        """
        num_info = "{title:17} = {value:5d}"
        float_info = "{title:17} = {value:3.2f}"
        bool_info = "{title:17} = {value:5}"

        print(num_info.format(title="Nodes Number",
              value=self.graph.num_nodes))
        print(num_info.format(title="Edges Number",
              value=self.graph.num_edges))
        print(num_info.format(title="Isolated Nodes",
              value=self.graph.has_isolated_nodes()))
        print(float_info.format(title="Avg Node Degree",
                               value=self.graph.num_edges / self.graph.num_nodes))
        print(bool_info.format(title="Directed",
                               value=str(self.graph.is_directed())))


class SpectraDataset():
    def __init__(self, mzml_dir="../data/converted/", mzml_files_wildcard="*",
                 psm_dir="../data/crux/crux-output/", psm_file="percolator.target.psms.txt",
                 serialized_dataset="../data/serialized/spectra_dataset.pickle",
                 save_pickled_dataset=True, fdr_threhold=0.01, max_spectra=None):
        self.mzml_dir = mzml_dir
        self.mzml_files_wildcard = mzml_files_wildcard
        self.psm_dir = mzml_dir
        self.psm_files_wildcard = mzml_files_wildcard

        # Build dataset
        _count = 0
        self.dataset = {}
        print(path.join(psm_dir, psm_file))
        _scan2charge, _scan2peptide = readPSMs(path.join(psm_dir, psm_file),
                                               fdrThreshold=fdr_threhold)

        print(len(_scan2charge))
        print(len(_scan2peptide))
        mzml_files = glob.glob(path.join(mzml_dir, mzml_files_wildcard))
        for mzml_file in mzml_files:
            print("Analyzing {mzml_file}".format(mzml_file=mzml_file))

            if max_spectra is not None and _count >= max_spectra:
                break
            _ds = processSpectra(mzml_file, _scan2charge, _scan2peptide, verbose=True,
                                 max_spectra=max_spectra - _count)
            for spectrum_data in _ds:
                _spectrum_instance = Spectrum(spectrum_data)
                _idx = spectrum_data['idx']
                self.dataset[_idx] = _spectrum_instance
                _count += 1
                if max_spectra is not None and _count >= max_spectra:
                    break
        print("Dataset loaded (number of spectra = {spectra_num})".
              format(spectra_num=len(self.dataset.keys())))

        # Serialize dataset
        if save_pickled_dataset:
            with open(serialized_dataset, 'wb') as f:
                pickle.dump(self, f)
                print("Dataset serialized on path : {path_pickle}".
                  format(path_pickle=serialized_dataset))

    def get_data(self, normalize=True, max_size=None):
        """
        Returns training vectors X

        Return list X with pytorch_geometric.Data.data graph instances
        A normalization step on graph feature vector can be performed.

        Parameters
        ----------
        normalize : bool
            Apply normalization on nodes features.  # TODO but probably handled in GNN layers
        max_size : int or None
            If not none returns at most max_size elements.

        Returns
        -------
        X : list of torch_geometric.Data.data instances
            graph instances, with y data if present in graph.y attribute
        """
        keys = self.dataset.keys()
        if max_size is not None:
            keys = keys[min(max_size, len(keys) - 1):]


        data = [self.dataset[key].graph for key in keys]

        return data

    def rebuild_graphs(self, serialized_dataset, padding=10):
        """
        Temporary/Utils method
        """
        for key in self.dataset.keys():
            # element = self.dataset[key]
            graph = self.dataset[key].graph
            #edge_attr = self.dataset[key].graph.edge_attr
            #edge_attr = edge_attr.type(torch.float)
            # edges = self.dataset[key].edges_aa
            # x = graph.x
            # dim_nodes = x.shape[1]
            # pad_size = padding - dim_nodes
            # pad_tensor = torch.zeros((x.shape[0], pad_size))
            # x = torch.cat((x, pad_tensor), dim=1)
            #lgraph.edge_attr = edge_attr
            graph.y = self._encode_sequence(self.dataset[key].peptide_seq,
                                   get_classes_list=True)
            self.dataset[key].graph = graph


        with open(serialized_dataset, 'wb') as f:
            pickle.dump(self, f)
            print("Dataset serialized on path : {path_pickle}".
                  format(path_pickle=serialized_dataset))

    
    def print_stats(self):
        int_info = "{title:17} = {value:5d}"
        float_info = "{title:25} = {value:3.2f}"
        bool_info = "{title:17} = {value:5}"

        # Retrieve stats from spectrum/graphs and compute mean
        stats_list = [graph.get_stats() for graph in self.dataset.values()]
        avg_nodes = sum(data['num_nodes'] for data in stats_list) / len(stats_list)
        avg_edges = sum(data['num_edges'] for data in stats_list) / len(stats_list)
        avg_node_degree = sum(data['avg_node_degree'] for data in stats_list) / len(stats_list)
        avg_peptide_seq_len = sum(data['peptide_seq_len'] for data in stats_list) / len(stats_list)
        avg_node_seq_len_ratio = sum(data['seq_len_node_ratio'] for data in stats_list) / len(stats_list)



        print(int_info.format(title="Number of spectra", value=len(self.dataset.keys())))
        print(float_info.format(title="Average nodes number", value=avg_nodes))
        print(float_info.format(title="Average edges number", value=avg_edges))
        print(float_info.format(title="Average nodes degree", value=avg_node_degree))
        print(float_info.format(title="Average peptide seq len", value=avg_peptide_seq_len))
        print(float_info.format(title="Avg node/seq_len ratio", value=avg_node_seq_len_ratio))

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
    def load_pickled_dataset(serialized_dataset="../data/serialized/spectra_dataset.pickle"):
        with open(serialized_dataset, 'rb') as f:
            ds = pickle.load(f)
        print("Dataset Loaded")
        return ds


if __name__ == '__main__':
    # Testing of module classes instances
    a = SpectraDataset._encode_sequence("AL", flatten=True, padding=3)
    print(a)



    pickle_file = "../data/serialized/spectra_dataset.pickle"

    # Dataset Spectra instance test
    print("Dataset Spectra instance test")
    if path.isfile(pickle_file):
        ds = SpectraDataset.load_pickled_dataset(pickle_file)
    else:
        ds = SpectraDataset(serialized_dataset=pickle_file,
                            max_spectra=700, mzml_files_wildcard="*01.mz*",)
    #ds.rebuild_graphs("../data/serialized/spectra_dataset_200.pickle")
    all_seq = ""
    for spectrum in ds.dataset.values():
        seq = spectrum.peptide_seq
        all_seq += seq
    print(len(all_seq))
    ds.print_stats()


    # Graph Spectrum instance test
    print("\n\n Graph Instance Test")
    mzFile = "../data/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
    mzScan = 32688
    psmPeptide = "IANVQSQLEK"
    precursorCharge = 2
    print(f"Recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}")
    fragment_tol_mass = 10
    fragment_tol_mode = 'ppm'
    data = read_sample_spectrum(mzScan, mzFile, False,
                                fragment_tol_mass,
                                fragment_tol_mode,
                                psmPeptide=psmPeptide)
    sample_spectrum = Spectrum(data)
    sample_spectrum.print_stats()
    sample_spectrum.show_graph()

    pass
