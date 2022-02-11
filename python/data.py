#!/usr/bin/env python3
import torch
from torch_geometric.data import Data
from torch_geometric import utils
import networkx as nx
from ms2graph import process_sample_spectrum
import pyteomics.mzml as mz
import pyteomics.mass as mass
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from os import path



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
        self.intensities = np.insert(self.intensities, 0, 0)  # c term
        self.intensities = np.insert(self.intensities, 0, 0)  # n term
        self.intensities = np.append(self.intensities, 0)  # precursor + c term
        self.intensities = np.append(self.intensities, 0)  # precursor + n term

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

        if use_intensities and self.intensities is not None:
            assert len(self.intensities) == num_nodes
            x = torch.tensor(self.intensities, dtype=torch.float)
        else:
            x = torch.tensor(range(num_nodes), dtype=torch.float)

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
                 psm_dir="../data/crux/crux-output/", psm_files_wildcard="*",
                 serialized_dataset="spectra_dataset.pickle"):
        self.mzml_dir = mzml_dir
        self.mzml_files_wildcard = mzml_files_wildcard
        self.psm_dir = mzml_dir
        self.psm_files_wildcard = mzml_files_wildcard



if __name__ == '__main__':
    # Testing of module classes instances

    # Graph Spectrum instance test
    mzFile = "../data/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
    mzScan = 32688
    psmPeptide = "IANVQSQLEK"
    precursorCharge = 2
    print(f"Recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}")
    fragment_tol_mass = 10
    fragment_tol_mode = 'ppm'
    data = process_sample_spectrum(mzScan, mzFile, False,
                                   fragment_tol_mass,
                                   fragment_tol_mode)
    sample_spectrum = Spectrum(data)
    sample_spectrum.print_stats()
    sample_spectrum.show_graph()

    pass
