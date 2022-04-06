#!/usr/bin/env python3
from base_wrapper import BaseModelWrapper
from torch_models import GNN2Seq, GNN2Transformer, GNN2Edges
from torch_geometric.loader import DataLoader
import torch
import statistics
from os import path
from data import SpectraDataset
from utils import minibatch_list, GraphDataset, MyBatchSampler
from graph_utils import filter_nodes, k_best_dijkstra, torch_graph_to_dict_graph, k_best_bellmanford
from spectrum import Spectrum
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


class GNN2SeqWrapper(BaseModelWrapper):
    def __init__(self, cuda=True):
        super(GNN2SeqWrapper, self).__init__()
        self.model = GNN2Seq(12,20)
        self.cuda = cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.log_message(sum(param.numel() for param in self.model.parameters()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.00001)
        self.loss = torch.nn.CrossEntropyLoss()


    def fit(self, data, epochs=10000, minibatch_size=32, valid_split_size=None, max_len_pred=5,
            **kwargs):
        # Assert target presence
        for graph in data:
            assert graph.y is not None

        _total_elements = len(data)

        if valid_split_size is not None:
            data_valid = data[:int(_total_elements * valid_split_size)]
            data = data[int(_total_elements * valid_split_size):]


        if max_len_pred is not None:
            _n_steps=max_len_pred
        for i, x in enumerate(data):
            data[i].y = x.y[:_n_steps]

        if valid_split_size is not None:
            for i, x in enumerate(data_valid):
                data_valid[i].y = x.y[:_n_steps]
                loader_valid = DataLoader(data_valid, batch_size=8, shuffle=False)

        # Start training epochs
        for epoch in range(epochs):
            self.model.train()
            #if epoch == 40:
            #    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, amsgrad=True)
            #if epoch == 100:
            #    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, amsgrad=True)
            self.log_message("\n-----------------------\nStarting epoch : {epoch}".format(epoch=epoch))
            # utils
            _losses = []
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

            self.model.eval()
            self.log_message("Training loss at epcoh {epoch} = {loss:3.3f}".format(epoch=epoch,
                                                                            loss=statistics.fmean(_losses)))
            self.log_message("y: " + str(y))
            self.log_message("out: " + str(torch.argmax(out,1))) #
            _losses = []
            with torch.no_grad():
                if valid_split_size is not None:
                    torch.cuda.empty_cache()
                    _valid_losses = []
                    for i, data_batch in enumerate(loader_valid):
                        data_batch.to(self.device)
                        y_valid = data_batch.y
                        batch_valid = data_batch.batch
                        out_valid = self.model(data_batch, batch_valid, self.device,
                                               n_steps=_n_steps)
                        loss = self.loss(out_valid, y_valid)
                        _valid_losses.append(loss)

                    self.log_message("Valid loss at epcoh {epoch} = {loss:3.3f}".format(epoch=epoch,
                                                                loss=statistics.fmean(_valid_losses)))
                    self.log_message("y: " + str(y_valid))
                    self.log_message("out: " + str(torch.argmax(out_valid,1))) #

    def predict(self, data):
        pass # to be implemented


class GNN2TransformerWrapper(BaseModelWrapper):
    def __init__(self, cuda=True):
        super(GNN2TransformerWrapper, self).__init__()
        self.model = GNN2Transformer(12,20)
        self.cuda = cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.log_message(sum(param.numel() for param in self.model.parameters()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001,
                                          amsgrad=True, weight_decay=0.000000001)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        #lr = 5.0  # learning rate
        self.loss = torch.nn.CrossEntropyLoss()


    def fit(self, data, epochs=10000, minibatch_size=32, valid_split_size=None, **kwargs):
        # Assert target presence
        for graph in data:
            assert graph.y is not None
        _total_elements = len(data)

        if valid_split_size is not None:
            data_valid = data[:int(_total_elements * valid_split_size)]
            data = data[int(_total_elements * valid_split_size):]


        if valid_split_size is not None:
            for i, x in enumerate(data_valid):
                loader_valid = DataLoader(data_valid, batch_size=1, shuffle=False)

        batch_list = minibatch_list(data, minibatch_size)
        data = GraphDataset(data)
        sampler = MyBatchSampler(batch_list)
        # Start training epochs
        for epoch in range(epochs):
            self.model.train()
            #if epoch == 40:
            #    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, amsgrad=True)
            #if epoch == 100:
            #    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, amsgrad=True)
            self.log_message("\n-----------------------\nStarting epoch : {epoch}".format(epoch=epoch))
            # utils
            _losses = []

            loader = DataLoader(data, batch_sampler=sampler)
            for i, data_batch in enumerate(loader):
                self.optimizer.zero_grad()
                data_batch.to(self.device)
                y = data_batch.y
                batch = data_batch.batch
                out = self.model(data_batch, y, batch, self.device)
                y = self._add_eos(y, batch, self.device)
                out = out.transpose(0, 1)
                y = y.transpose(0, 1)
                out = out.reshape((out.shape[0]*out.shape[1], out.shape[2]))
                y = y.reshape((y.shape[0]*y.shape[1]))
                loss = self.loss(out, y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                _losses.append(loss)

            self.model.eval()
            self.log_message("Training loss at epcoh {epoch} = {loss:3.3f}".format(epoch=epoch,
                                                                            loss=statistics.fmean(_losses)))
            self.log_message("y: " + str(y))
            self.log_message("out: " + str(torch.argmax(out,1))) #
            _losses = []
            with torch.no_grad():
                if valid_split_size is not None:
                    torch.cuda.empty_cache()
                    _valid_losses = []
                    for i, data_batch in enumerate(loader_valid):
                        data_batch.to(self.device)
                        y_valid = data_batch.y
                        batch_valid = data_batch.batch
                        out_valid = self.model(data_batch, y_valid, batch_valid, self.device,
                                               )


                        y_valid = self._add_eos(y_valid, batch_valid, self.device)
                        out_valid = out_valid.transpose(0, 1)
                        y_valid = y_valid.transpose(0, 1)
                        out_valid = out_valid.reshape((out_valid.shape[0]*out_valid.shape[1],
                                                       out_valid.shape[2]))
                        y_valid = y_valid.reshape((y_valid.shape[0] * y_valid.shape[1]))

                        loss = self.loss(out_valid, y_valid)
                        _valid_losses.append(loss)

                    self.log_message("Valid loss at epcoh {epoch} = {loss:3.3f}".format(epoch=epoch,
                                                                loss=statistics.fmean(_valid_losses)))
                    self.log_message("y: " + str(y_valid))
                    self.log_message("out: " + str(torch.argmax(out_valid,1))) #

    def predict(self, data):
        pass # to be implemented

    def _add_eos(self, y, batch, device):
        shape = (int(y.shape[0] / (int(torch.max(batch)) + 1)),
                                 int(torch.max(batch)) + 1)
        y = torch.reshape(y, shape)
        eos = torch.tensor([21 for a in range(y.shape[1])], device=device)
        eos = eos.unsqueeze(0)
        y = torch.cat((y, eos))
        # y = torch.nn.functional.one_hot(y, 22)
        return y






class GNN2EdgesWrapper(BaseModelWrapper):
    def __init__(self, cuda=True):
        super(GNN2EdgesWrapper, self).__init__()
        self.pytorch_model_class = GNN2Edges

        self.node_dim = 12
        self.edge_dim = 40

        self.model = self.pytorch_model_class(self.node_dim, self.edge_dim)
        self.cuda = cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.n_parameters = sum(param.numel() for param in self.model.parameters())
        self.log_message(f"Parameters number : {self.n_parameters}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.00)
        self.loss = torch.nn.BCELoss()


    def fit(self, data, epochs=10000, minibatch_size=32, valid_split_size=None,
            serialize_epoch_interval=None):
        # Assert target presence
        for graph in data:
            assert graph.y is not None

        _total_elements = len(data)

        if valid_split_size is not None:
            data_valid = data[:int(_total_elements * valid_split_size)]
            data = data[int(_total_elements * valid_split_size):]
            self.log_message(f"Training set   | size : {len(data)}")
            self.log_message(f"Validation set | size : {len(data_valid)}")
        if valid_split_size is not None:
            loader_valid = DataLoader(data_valid, batch_size=1, shuffle=False)

        # Start training epochs
        for epoch in range(epochs):
            self.model.train()
            self.log_message(f"-----------------------\nStarting epoch : {epoch}")
            # utils
            _losses = []
            all_y = torch.tensor([], device=self.device)
            all_out = torch.tensor([], device=self.device)

            loader = DataLoader(data, batch_size=minibatch_size, shuffle=True)
            for i, data_batch in enumerate(loader):
                self.optimizer.zero_grad()
                data_batch.to(self.device)
                y = data_batch.y
                batch = data_batch.batch

                out = self.model(data_batch, batch, self.device)
                y = y.unsqueeze(1)
                loss = self.loss(out, y)


                all_y = torch.cat((all_y, y))
                all_out = torch.cat((all_out, out))
                loss.backward()
                self.optimizer.step()
                self.log_message(loss)
                _losses.append(loss)
            self.model.eval()
            perf = self.score_binary(all_y, all_out)
            self.log_message("Training loss at epcoh {epoch} = {loss:3.3f}".format(epoch=epoch,
                                                                            loss=statistics.fmean(_losses)))
            self.log_message(f"Perf at epcoh {epoch} = {str(perf)}")
            self.log_message(f"epoch = {epoch} - sample y : {y.flatten()}")
            self.log_message(f"epoch = {epoch} - sample out : {str(torch.argmax(out,1))}")  #
            _losses = []
            with torch.no_grad():
                if valid_split_size is not None:
                    torch.cuda.empty_cache()
                    all_y_valid = torch.tensor([], device=self.device)
                    all_out_valid = torch.tensor([], device=self.device)
                    _valid_losses = []
                    for i, data_batch in enumerate(loader_valid):
                        data_batch.to(self.device)
                        y_valid = data_batch.y
                        y_valid = y_valid.unsqueeze(1)
                        batch_valid = data_batch.batch
                        out_valid = self.model(data_batch, batch_valid, self.device)
                        loss = self.loss(out_valid, y_valid)

                        # Save data for stats
                        all_y_valid = torch.cat((all_y_valid, y_valid))
                        all_out_valid = torch.cat((all_out_valid, out_valid))
                        _valid_losses.append(loss)

                    valid_perf = self.score_binary(all_y_valid, all_out_valid, plot=True, plot_label=str(epoch))
                    self.log_message("Valid loss at epcoh {epoch} = {loss:3.3f}".format(epoch=epoch,
                                                                loss=statistics.fmean(_valid_losses)))
                    self.log_message(f"Valid perf at epcoh {epoch} = {str(valid_perf)}")
                    self.log_message(f"epoch = {epoch} - sample valid_y : {y_valid.flatten()}")
                    self.log_message(f"epoch = {epoch} - sample out : {str(torch.argmax(out_valid, 1))}") #
            if serialize_epoch_interval is not None and epoch % serialize_epoch_interval == 0:
                self.serialize_model(str_suffix=str(epoch))

    def predict(self, data, evaluate=False):
        loader = DataLoader(data, batch_size=1, shuffle=False)

        if evaluate:
            y_present = [hasattr(graph, 'y') for graph in data]
            if not all(y_present): evaluate = False


        all_out = torch.tensor([], device=self.device)
        if evaluate:
            all_y = torch.tensor([], device=self.device)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for i, data_batch in enumerate(loader):
                data_batch.to(self.device)
                batch = data_batch.batch
                out = self.model(data_batch, batch, self.device)
                preds.append(out)
                if evaluate:
                    y = data_batch.y.unsqueeze(1)
                    all_y = torch.cat((all_y, y))
                all_out = torch.cat((all_out, out))

        if evaluate:
            loss = self.loss(all_out, all_y)
            perf = self.score_binary(all_y, all_out, plot=True, plot_label="_predict")

            self.log_message(f"Loss = {loss}")
            self.log_message(f"Perf = {perf}")

        #self.edges_to_peptide_seq(list(loader), preds)

        return all_out

    def edges_to_peptide_seq(self, graphs, y_hats):
        assert len(graphs) == len(y_hats)
        for graph, y_hat in zip(graphs, y_hats):
            y_hat = - y_hat.cpu().squeeze()
            dict_graph, edges, chars = torch_graph_to_dict_graph(graph, y_hat.tolist(), return_edges=True)
            paths = k_best_bellmanford(dict_graph, edges, graph.source_sink_tuple[0])
            print(paths)
            for path in paths:
                print(path.get())
            # DEBUG

    
    def score_binary(self, y, y_hat, plot=False, plot_label=""):
        scores = {}
        fpr, tpr, thresholds = metrics.roc_curve(y.cpu().detach().numpy(),
                                                 y_hat.cpu().detach().numpy())
        pre_p, tpr_p, thresholds_p = metrics.precision_recall_curve(y.cpu().detach().numpy(),
                                                 y_hat.cpu().detach().numpy())
        print(len(thresholds))
        if plot:

            fpr_c = fpr[::max(1, len(fpr) // 10000)]
            tpr_c = tpr[::max(1, len(tpr) // 10000)]
            thresholds_c = fpr[::max(1, len(fpr) // 10000)]
            plot = sns.lineplot(x=fpr_c, y=tpr_c)
            plot.set_xlabel("fpr")
            plot.set_ylabel("tpr")
            plot.set(title="ROC Curve")
            percentiles_range = list(np.arange(0.05, 1, 0.1))
            for ix, perc in enumerate(percentiles_range):
                point = int(len(thresholds_c) * perc)
                plot.text(fpr_c[point] + 0.02, tpr_c[point] + 0.02, str(round(perc, 2)))
            plot.figure.savefig("../results/plots/tmp_roc_" + plot_label + ".png")
            plt.clf()

        scores['auc'] = metrics.auc(fpr, tpr)
        scores['auprc'] = metrics.average_precision_score(y.cpu().detach().numpy(),
                                                          y_hat.cpu().detach().numpy())
        return scores







if __name__ ==  '__main__':
    # Test models prototypes
    print("Loading Spectra Dataset")
    pickle_dataset = "../data/serialized/ds_2_miss_5_2"
    test_split = 0.6

    ds = SpectraDataset.load_pickled_dataset(pickle_dataset, max_spectra=80000)

    data = ds.get_data(y_edges=True, graph_filtering=False)
    _total_elements = len(data)

    if test_split is not None:
        data_test = data[int(_total_elements * test_split):]
        data = data[:int(_total_elements * test_split)]
    print("Training samples retrieved : {length}".format(length=len(data)))

    #model_gnn = GNN2SeqWrapper()
    #model_gnn = GNN2EdgesWrapper()
    model_gnn = GNN2EdgesWrapper.load_serialized_model("../models/GNN2EdgesWrapper_20.pickle")

    model_gnn.fit(data, minibatch_size=1, valid_split_size=0.3, serialize_epoch_interval=5)



    if test_split is not None:
        # Prediction test
        model_gnn.predict(data_test, evaluate=True)
    pass
