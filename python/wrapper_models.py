#!/usr/bin/env python3
from base_wrapper import BaseModelWrapper
from torch_models import GNN2Seq
from torch_geometric.loader import DataLoader
import torch
import statistics
from os import path
from data import SpectraDataset, Spectrum

# List of putative prediction models inherited from BaseModelWrapper

class GNN2SeqWrapper(BaseModelWrapper):
    def __init__(self, cuda=True):
        self.model = GNN2Seq(5,21)
        self.cuda = cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        print(sum(param.numel() for param in self.model.parameters()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, amsgrad=True)
        self.loss = torch.nn.CrossEntropyLoss()


    def fit(self, data, epochs=10000, minibatch_size=32, valid_split_size=None, max_len_pred=None):
        # Assert target presence
        for graph in data:
            assert graph.y is not None

        _total_elements = len(data)

        if valid_split_size is not None:
            data_valid = data[:int(_total_elements * valid_split_size)]
            data = data[int(_total_elements * valid_split_size):]


        self.model.train()

        for epoch in range(epochs):
            print("Starting epoch : {epoch}".format(epoch=epoch))
            # utils
            _losses = []
            _n_steps = 5 # length of pred, fixed len temporarily
            if max_len_pred is not None:
                _n_steps=max_len_pred
            for i, x in enumerate(data):
                data[i].y = x.y[:_n_steps]
            loader = DataLoader(data, batch_size=minibatch_size, shuffle=True)
            for i, data_batch in enumerate(loader):
                self.optimizer.zero_grad()
                data_batch.to(self.device)
                y = data_batch.y
                batch = data_batch.batch

                out = self.model(data_batch, batch, self.device, n_steps=_n_steps)
                loss = self.loss(out, y)
                loss.backward()
                self.optimizer.step()
                _losses.append(loss)
                if i % 300 == 0:
                    print("Training loss at epcoh {epoch} = {loss:3.2f}".format(epoch=epoch,
                                                    loss=statistics.fmean(_losses)))
                    # print("y: " + str(y))
                    # print("out: " + str(torch.argmax(out,1))) #
                    _losses = []

    def predict(self, data):
        pass # to be implemented



if __name__ ==  '__main__':
    # Test models prototypes
    print("Loading Spectra Dataset")
    pickle_dataset = "../data/serialized/spectra_dataset_200_noclaim.pickle"
    if path.isfile(pickle_dataset):
        ds = SpectraDataset.load_pickled_dataset(pickle_dataset)
    else:
        ds = SpectraDataset()
    ds.print_stats()

    data = ds.get_data()
    print("Training vectors of length : {length} retrieved".format(length=len(data)))

    model_gnn = GNN2SeqWrapper()
    model_gnn.fit(data, minibatch_size=4, max_len_pred=4)
    pass
