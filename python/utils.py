#!/usr/bin/env python3
from torch import nn
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pyteomics.mzml as mz
import pyteomics.mass as mass
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from os import path
from matplotlib import pyplot as plt




class MZEncoder(nn.Module):
    # Took inspiration from deptcharge/components/encoders.py
    def __init__(self, dim=20, min_mz=0.0001, max_mz = 10000):
        super().__init__()
        n_sin = dim // 2
        n_cos = dim - n_sin

        self.sin_t = (max_mz / min_mz) * ((min_mz / (torch.pi * 2)) ** \
            (2 * torch.arange(0, n_sin).float() / (dim)))

        self.cos_t = (max_mz / min_mz) * ((min_mz / (torch.pi * 2)) ** \
            (2 * torch.arange(n_sin, dim).float() / (dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        self.sin_t = self.sin_t.to('cuda')
        self.cos_t = self.cos_t.to('cuda')
        sin_mz = torch.sin(x / self.sin_t)
        cos_mz = torch.cos(x / self.cos_t)
        return torch.cat((sin_mz, cos_mz), dim=1)


def generate_tgt_mask(sz):
    """Generate a square mask for the sequence. The masked positions
    are filled with float('-inf'). Unmasked positions are filled with
    float(0.0).
    This function is a slight modification of the version in the PyTorch
    repository.
    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask




def minibatch_list(graph_list, max_batch_size=32):
    """
    Return minibatch indices list so that each minibatch contains element with same
    seq length.
    """
    indices = sorted(range(len(graph_list)), key=lambda k: graph_list[k].y.shape[0])
    graph_list = list(graph_list)
    graph_list.sort(key=lambda x: x.y.shape[0])
    batch_list = []
    len_y = 0
    count = 0
    batch = []
    for i, graph in enumerate(graph_list):
        n_len_y = graph.y.shape[0]
        if n_len_y == len_y and count < max_batch_size:
            count += 1
            batch.append(indices[i])
        else:
            count = 1
            if len(batch) > 0:
                batch_list.append(batch)
            batch = [indices[i]]
            len_y = n_len_y
    batch_list.sort(key=lambda x : len(x))
    return batch_list



class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class MyBatchSampler(Sampler):
     def __init__(self, batches):
         self.batches = batches

     def __iter__(self):
         for batch in self.batches:
             yield batch

     def __len__(self):
         return len(self.batches)



# precompute molecules masses for time performance purposes
DICT_MOL = {'h2o': mass.Composition({'H': 2, 'O': 1}).mass(),
            'nh3': mass.Composition({'N': 1, 'H': 3}).mass(),
            'co' : mass.Composition({'C': 1, 'O': 1}).mass(),
            'h+': mass.Composition({'H+': 1}).mass()}


B_OFFSET_FUNCTIONS = (
        lambda x, charge: (x / charge + DICT_MOL['h+']), # b ion

        lambda x, charge: (x / charge + DICT_MOL['h+'] \
            - DICT_MOL['nh3'] / charge), # b ion - nh3
        lambda x, charge: (x / charge + DICT_MOL['h+'])\
            - DICT_MOL['h2o'] / charge, # b ion - h2o

        lambda x, charge: (x / charge + DICT_MOL['h+'])\
            - 2 * DICT_MOL['h2o'] / charge, # b ion - h2o - h2o
        lambda x, charge: (x + DICT_MOL['h+']) -
            DICT_MOL['co'] # a ions
    )


B_INVERSE_OFFSET_FUNCTIONS = (
        lambda x, charge: charge * (x - DICT_MOL['h+']), # b ion

        lambda x, charge: charge * (x - DICT_MOL['h+']) +
            DICT_MOL['nh3'], # b ion - NH3
        lambda x, charge: (x - DICT_MOL['h+']) * charge +\
            DICT_MOL['h2o'], # b ion - h2o


        lambda x, charge: (x - DICT_MOL['h+']) * charge +\
            2 * DICT_MOL['h2o'], # b ion - h2o -h2o
        lambda x, charge: (x - DICT_MOL['h+']) +\
        DICT_MOL['co'] # a ions
    )

Y_OFFSET_FUNCTIONS = (
        lambda x, charge: (x / charge + DICT_MOL['h2o'] / charge + DICT_MOL['h+']), # y ion

        lambda x, charge: (x / charge + DICT_MOL['h2o'] / charge + DICT_MOL['h+'])\
            - DICT_MOL['nh3'] / charge,           # y ion - nh3
        lambda x, charge: (x / charge + DICT_MOL['h+']), # y ion - h2o

        lambda x, charge: (x / charge + DICT_MOL['h+'])  \
            - DICT_MOL['h2o'] / charge, # y ion -h2o-h2o
    )


Y_INVERSE_OFFSET_FUNCTIONS = (
        lambda x, charge: charge * (x - DICT_MOL['h2o'] / charge - DICT_MOL['h+']),  # y ion


        lambda x, charge: charge * (x - DICT_MOL['h2o'] / charge - DICT_MOL['h+']) + \
            DICT_MOL['nh3'],  # y ion - Nh3
        lambda x, charge: charge * (x - DICT_MOL['h+']), # y ion - h2o


        lambda x, charge: charge * (x - DICT_MOL['h+']) +\
            DICT_MOL['h2o'], # y ion - h2o - h2o
    )

def test_offset_functions(offset_functions, inverse_offset_functions):
    """
    Test for each function if f^-1(f(x))==x
    """
    assert len(offset_functions) == len(inverse_offset_functions)
    for i in range(len(offset_functions)):
        for charge in range(1, 4):
            x = 100
            #print(i)
            #print(offset_functions[i](x, charge))
            #print(abs(inverse_offset_functions[i](offset_functions[i](x, charge), charge)))
            assert abs(inverse_offset_functions[i](offset_functions[i](x, charge), charge) - x) < 0.000001



def stats_PSMs(file_path, plot_dir="../results/plots/"):
    df = pd.read_csv(file_path, sep='\t')
    #df = df.set_index("scan")
    df['diff'] = abs(df['spectrum neutral mass'] - df['peptide mass'])
    print(df['diff'].mean())
    df['ppm'] = df['diff'] / (df['spectrum neutral mass'] / 1000000)
    df['ppm'] = df['ppm'].clip(upper=100)
    print(df)
    #df.plot(x='ppm', kind='bar')
    sns.set(rc = {'figure.figsize':(15,8)})
    #
    bins = list(np.arange(-0.5,120.5, 1))
    p_da = sns.histplot(x=df['diff'], bins=bins,stat='probability')
    p_da.set_xlabel("abs(precursor_mass - psm_mass) in Da")
    p_da.figure.savefig(path.join(plot_dir, "distr_mass_diff_da.png"))
    plt.clf()


    bins = list(np.arange(-0.00075,0.05, 0.0005))
    p_da_0 = sns.histplot(x=df['diff'], bins=bins,stat='probability')
    p_da_0.set_xlabel("abs(precursor_mass - psm_mass) in Da")
    p_da_0.figure.savefig(path.join(plot_dir, "distr_mass_diff_da_around_0.png"))
    plt.clf()


    bins = list(np.arange(-0.5,101.5, 1))
    p_ppm = sns.histplot(x=df['ppm'], bins=bins,stat='probability')
    p_ppm.set_xlabel("abs(precursor_mass - psm_mass) in ppm")
    p_ppm.figure.savefig(path.join(plot_dir, "distr_mass_diff_ppm.png"))
    plt.clf()

    over_20_da = df[df['diff'] > 20]['diff'].count() / df['diff'].count() * 100
    over_10_ppm = df[df['ppm'] > 10]['ppm'].count() / df['ppm'].count() * 100

    print("{over_20_da} % over 20 dalton difference (precursor mass - psm mass)".format(over_20_da=over_20_da))
    print("{over_10_ppm} % over 10 ppm difference (precursor mass - psm mass)".format(over_10_ppm=over_10_ppm))


class PositionalEncoding(nn.Module):
    # got it from pytorch doc
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
    #stats_missing_values("./log.log")
    stats_PSMs("../data/crux/crux-output/percolator.target.psms.txt")



    test_offset_functions(B_OFFSET_FUNCTIONS, B_INVERSE_OFFSET_FUNCTIONS)
    test_offset_functions(Y_OFFSET_FUNCTIONS, Y_INVERSE_OFFSET_FUNCTIONS)
