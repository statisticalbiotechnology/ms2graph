#!/usr/bin/env python3
import torch
from torch_geometric.data.dataset import Dataset
from ms2graph import processSpectra, readPSMs, check_spectrum_completness, right_path_all_paths, get_source_sink_ions
import pyteomics.mzml as mz
import pyteomics.mass as mass
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from os import path, listdir
import pickle
import glob
import pandas
import seaborn as sns
import pandas as pd
from graph_utils import filter_nodes, filter_spectrum, paths_number_source_sink, torch_graph_to_dict_graph , torch_graph_to_f_t_c_graph
import copy




class SpectraDataset():
    def __init__(self, mzml_dir="../data/converted/", mzml_files_wildcard="*",
                 psm_dir="../data/crux/crux-output/", psm_file="percolator.target.psms.txt",
                 serialized_dataset_dir="../data/serialized/spectra_dataset/",
                 save_pickled_dataset=True, fdr_threhold=0.01, max_spectra=None, empty=False,
                 filtering_max_miss=None, y_edges=False, overwrite=True, only_stats=False,
                 max_miss_aa=2, compute_completness=True, compute_n_paths=False):
        self.mzml_dir = mzml_dir
        self.mzml_files_wildcard = mzml_files_wildcard
        self.psm_dir = mzml_dir
        self.psm_files_wildcard = mzml_files_wildcard
        self.max_miss_aa = max_miss_aa
        self.serialized_dataset_dir = path.join(serialized_dataset_dir)

        self.__name__ = serialized_dataset_dir.split('/')[-2]
        print(f"Loading Dataset {self.__name__}")

        self.dataset = {}
        # Handle empty generation
        if empty:
            return

        # Build dataset
        _count = 0
        print(path.join(psm_dir, psm_file))
        _scan2charge, _scan2peptide = readPSMs(path.join(psm_dir, psm_file),
                                               fdrThreshold=fdr_threhold)

        mzml_files = glob.glob(path.join(mzml_dir, mzml_files_wildcard))
        for mzml_file in mzml_files:
            print("Analyzing {mzml_file}".format(mzml_file=mzml_file))

            if max_spectra is not None and _count >= max_spectra:
                break
            _ds = processSpectra(mzml_file, _scan2charge, _scan2peptide, verbose=True,
                                 max_spectra=max_spectra - _count,
                                 filtering_max_miss=filtering_max_miss,
                                 pickle_dir=serialized_dataset_dir, y_edges=y_edges, log=True,
                                 overwrite=overwrite, only_stats=only_stats, max_miss_aa=max_miss_aa,
                                 compute_n_paths=compute_n_paths, compute_completness=compute_completness)
            _count += len(_ds)
            if max_spectra is not None and _count >= max_spectra:
                break
        print("Dataset saved (number of spectra = {spectra_num})".
              format(spectra_num=len(self.dataset.keys())))

        ds = self.load_pickled_dataset(serialized_dataset=serialized_dataset_dir,
                                  max_spectra=max_spectra)
        self.dataset = copy.deepcopy(ds.dataset)

    def get_data(self, normalize=True, max_size=None, y_edges=False, graph_filtering=False,
                 filter_precursor_mass_error=10):
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
        print(f"Retrieving model data from dataset of length = {len(list(keys))}")
        if max_size is not None:
            keys = keys[min(max_size, len(keys) - 1):]

        to_remove = []


        if y_edges:
            ratio = [0, 0]
            for key in keys:
                start_node, end_node = get_source_sink_ions(self.dataset[key].mz_array,
                                                            self.dataset[key].precursor['m'] - mass.Composition({'H':2, 'O':1}).mass(),
                                                            orientation=-1)
                if graph_filtering:

                    self.dataset[key].graph, self.dataset[key].mz_array, self.dataset[key].intensities, self.dataset[key].edges_aa = \
                        filter_spectrum(self.dataset[key].graph, self.dataset[key].mz_array,
                                        self.dataset[key].intensities, self.dataset[key].edges_aa,
                                        node_pairs=[(start_node, end_node)])
                if filter_precursor_mass_error:
                    _peptide_mass = mass.calculate_mass(parsed_sequence=list(self.dataset[key].peptide_seq))
                    _precursor_mass = self.dataset[key].precursor['m'] - \
                        mass.Composition({'H':2, 'O':1}).mass()
                    if abs(_peptide_mass - _precursor_mass) / (_precursor_mass / 1000000) > 10:
                        to_remove.append(key)

                y = torch.zeros((self.dataset[key].graph.edge_index.shape[1], ),)
                y_mask = self.dataset[key].graph.y
                y[y_mask] = 1
                self.dataset[key].graph.y = y
                self.dataset[key].graph.seq = self.dataset[key].peptide_seq
                self.dataset[key].graph.source_sink_tuple = (start_node, end_node)
                ratio[0] += len(y_mask)
                ratio[1] += (self.dataset[key].graph.edge_index.shape[1])
            print(f"Ratio positive_edges / total_edges : {ratio[0] / ratio[1]} ")
        print(f"Node features : {self.dataset[list(self.dataset.keys())[0]].graph.x.shape[1]}")
        data = [self.dataset[key].graph for key in keys if key not in to_remove]

        return data

    def rebuild_graphs(self, serialized_dataset, padding=10):
        """
        Temporary/Utils method
        """
        pass

        with open(serialized_dataset, 'wb') as f:
            pickle.dump(self, f)
            print("Dataset serialized on path : {path_pickle}".
                  format(path_pickle=serialized_dataset))

    def get_stats(self, compute_completness=False, compute_n_paths=False, save_attributes=True):
        stats_list = [graph.get_stats(compute_completness, compute_n_paths) for graph in self.dataset.values()]
        print(f"Retrieve stats for dataset with {len(self.dataset.keys())} elements")
        df = pandas.DataFrame(stats_list)
        if save_attributes:
            self.stats = df
        return df

    
    def print_stats(self, compute_completness=False, compute_n_paths=False,
                    plot_dir="../results/plots/"):
        float_info = "{title:25} = {value:3.2f}"

        # Retrieve stats from spectrum/graphs and compute mean
        df = self.get_stats(compute_completness, compute_n_paths)


        stats = (f"Spectra Number           : {df.shape[0]:8d} \n") +\
        (f"Average nodes number     : {df['num_nodes'].mean():10.2f} \n") +\
        (f"Average edges number     : {df['num_edges'].mean():10.1f} \n") +\
        (f"Isolated nodes           : {df['isolated_nodes'].mean():10.2f} \n") +\
        (f"Average peptide length   : {df['peptide_seq_len'].mean():10.2f} \n")+\
        (f"Average len(seq)/n_nodes : {df['seq_len_node_ratio'].mean():10.2f} \n")+\
        (f"Average n_edges/n_nodes  : {(df['num_edges'] / df['num_nodes']).mean():10.2f} \n")


        plot = sns.regplot(data=df, x='num_nodes', y='num_edges')
        plot.figure.savefig(path.join(plot_dir, f"n_nodes_n_edges_{self.get_name()}.png"))
        plt.clf()
        if 'missing_values' in df.columns:
            b_miss, y_miss, all_miss = list(zip(*df['missing_values']))

            df_missing = pd.DataFrame({'max_missing_masses': list(b_miss + y_miss + all_miss),
                               'ion_type':['b' for i in range(len(b_miss))] +\
                                       ['y' for i in range(len(y_miss))] +\
                                       ['all' for i in range(len(all_miss))]})

            #plot = sns.displot(df_missing, x='max_missing_masses', hue='ion_type', kde=False,
            #                   stat='probability', bins=np.arange(-0.25,
            #                    max(df_missing['max_missing_masses']), 0.5),
            #                   multiple='dodge')

            plot = sns.ecdfplot(df_missing, x='max_missing_masses', hue='ion_type',
                               stat='proportion',
                                )

            # plot = sns.countplot(data=df_missing, x='max_missing_masses', hue='ion_type')
            plt.xticks(list(range(max(df_missing['max_missing_masses']))))
            plt.gcf().set_size_inches(15, 8)
            plot.figure.savefig(path.join(plot_dir, f"max_ecdf_missing_masses_{self.get_name()}.png"))
            plt.clf()

            stats += (float_info.format(title="\nAverage max subseq missing b ions", value=sum(b_miss) / len(b_miss)))
            stats += (float_info.format(title="\nAverage max subseq missing y ions", value=sum(y_miss) / len(y_miss)))


        if 'n_paths' in df.columns:

            plot = sns.regplot(data=df, x='num_nodes', y='n_paths')
            plot.figure.savefig(path.join(plot_dir, f"n_nodes_n_paths_{self.get_name()}.png"))
            plt.clf()

            stats += (float_info.format(title="\nAverage number of paths on y ions", value=df['n_paths'].mean()))
            stats += (float_info.format(title="\nAverage ratio n_paths / n_peaks", value=(df['n_paths'] / df['n_nodes']).mean()))

        if 'true_edges' in df.columns:
            plot = sns.regplot(data=df, x='num_edges', y='tf_edges_ratio')
            plot.figure.savefig(path.join(plot_dir, f"n_edges_true_edges_ratio_{self.__name__}.png"))
            plt.clf()

            stats += (f"\nAverage number of true edges : {df['true_edges'].mean():10.2f}")
            stats += (f"\nAverage number of true edges ratio :  {df['tf_edges_ratio'].mean():8.4f}")
        self.log_stats(stats)
    
    @staticmethod
    def load_pickled_dataset(serialized_dataset="../data/serialized/spectra_dataset/",
                             max_spectra=None):
        if path.isfile(serialized_dataset):
            with open(serialized_dataset, 'rb') as f:
                ds = pickle.load(f)
                return ds


        ds = SpectraDataset(empty=True, serialized_dataset_dir=serialized_dataset)
        for i, file_name in enumerate(listdir(serialized_dataset)):
            if max_spectra is not None and i > max_spectra:
                break
            with open(path.join(serialized_dataset, file_name), 'rb') as f:
                ds.dataset[file_name.split('.')[0]] = pickle.load(f)
        ds.__name__ = serialized_dataset.split('/')[-2]
        ds.get_stats(save_attributes=True)
        print("Dataset Loaded : {ds.get_name()}")
        return ds

    def log_stats(self, stats, stats_dir="../results/stats/"):
        print(stats)
        stats_str = f"\n---------\n{self.get_name()}\n--------\n{stats}"
        print(stats_str)
        with open(path.join(stats_dir, self.get_name() + '.log'), 'w') as f:
            f.write(stats_str)

    def get_name(self):
        if hasattr(self, '__name__'):
            return self.__name__
        return "{len(self.dataset.keys())}_dataset"


if __name__ == '__main__':



    pickle_dir = "../data/serialized/ds_2_miss_5_2/"
    LOAD_PICKLE = True

    # Dataset Spectra instance test
    print("Dataset Spectra instance test")


    if not LOAD_PICKLE:
        ds = SpectraDataset(serialized_dataset_dir=pickle_dir,
                            max_spectra=10000, mzml_files_wildcard="*01.mz*",
                            filtering_max_miss=(5, 2), y_edges=False, overwrite=False,
                            only_stats=False, max_miss_aa=2, compute_n_paths=False)
    else:
        ds = SpectraDataset.load_pickled_dataset(pickle_dir, max_spectra=8000)
    ds.print_stats(compute_completness=True, compute_n_paths=False)



    pass
