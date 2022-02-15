#!/usr/bin/env python3
import torch
from torch_geometric.data import Data
from torch_geometric import utils
import networkx as nx
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
        if 'peptide_seq' in data.keys():
            self.peptide_seq = data['peptide_seq']
        self.graph = self.__generate_pyg_graph(data['from'], data['to'],
                                               data['edges_aa'])

    def __generate_pyg_graph(self, _from, _to, _edges, use_intensities=True):
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

        Returns
        -------
        graph : torch_geometric.data.Data
            Pytorch geometric graph class instance
        """
        assert len(_from) == len(_to) == len(_edges)
        num_nodes = len(self.mz_array)
        edge_index = torch.tensor([_from, _to], dtype=torch.float)
        x = torch.tensor(self.mz_array, dtype=torch.float).unsqueeze(1)
        if use_intensities and self.intensities is not None:
            assert len(self.intensities) == num_nodes
            x_intensities = torch.tensor(self.intensities, dtype=torch.float).unsqueeze(1)
            x = torch.cat([x, x_intensities], dim=1)

        graph = Data(x=x, edge_index=edge_index)
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
                 save_pickled_dataset=True, fdr_threhold=0.01):
        self.mzml_dir = mzml_dir
        self.mzml_files_wildcard = mzml_files_wildcard
        self.psm_dir = mzml_dir
        self.psm_files_wildcard = mzml_files_wildcard

        # Build dataset
        self.dataset = {}
        print(path.join(psm_dir, psm_file))
        _scan2charge, _scan2peptide = readPSMs(path.join(psm_dir, psm_file),
                                               fdrThreshold=fdr_threhold)
        mzml_files = glob.glob(path.join(mzml_dir, mzml_files_wildcard))
        for mzml_file in mzml_files:
            print("Analyzing {mzml_file}".format(mzml_file=mzml_file))
            _ds = processSpectra(mzml_file, _scan2charge, _scan2peptide, verbose=False)
            for spectrum_data in _ds:
                _spectrum_instance = Spectrum(spectrum_data)
                _idx = spectrum_data['idx']
                self.dataset[_idx] = _spectrum_instance
        print("Dataset loaded (number of spectra = {spectra_num})".
              format(spectra_num=len(self.dataset.keys())))

        # Serialize dataset
        if save_pickled_dataset:
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
    def load_pickled_dataset(serialized_dataset="../data/serialized/spectra_dataset.pickle"):
        with open(serialized_dataset, 'rb') as f:
            ds = pickle.load(f)
        print("Dataset Loaded")
        return ds


if __name__ == '__main__':
    # Testing of module classes instances

    # Dataset Spectra instance test
    print("Dataset Spectra instance test")
    if path.isfile("../data/serialized/spectra_dataset.pickle"):
        ds = SpectraDataset.load_pickled_dataset()
    else:
        ds = SpectraDataset()
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
                                   fragment_tol_mode)
    sample_spectrum = Spectrum(data)
    sample_spectrum.print_stats()
    sample_spectrum.show_graph()

    pass
