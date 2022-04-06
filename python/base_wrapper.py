#!/usr/bin/env python3
from abc import ABCMeta, abstractclassmethod
from os import path
from time import gmtime, strftime, localtime
import pickle
import copy


class BaseModelWrapper:
    __metaclass__ = ABCMeta


    def __init__(self, name=None):
        if name is None:
            self.__name__ = self.__class__.__name__ # useful for identifying each model on the basis of the subclass name
        else:
            self.__name__ = name


    @abstractclassmethod
    def fit(self, X, Y):
        """
        Abstract class for model training
        """
        pass

    @abstractclassmethod
    def predict(self, X):
        """
        Abstract class for model prediction phase
        """
        pass


    def load_serialized_model(model_path):
        print(model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        state_dict = model.model

        model.model = model.pytorch_model_class(model.node_dim, model.edge_dim)
        model.model.load_state_dict(state_dict)
        model.model.to(model.device)
        model.log_message(f"Loaded model at path : {model_path} containing {model.n_parameters} parameters")
        return model

    def log_message(self, message, terminal_log=True, log_dir="../results/", save_time=True):
        """
        Function to log a message on terminal and/or on file
        """
        message = str(message)

        for msg in message.split('\n'):
            log_message = str(msg)

            if save_time:
                time = strftime("%d/%m/%Y %H:%M:%S -> ", localtime())
                log_message = time + log_message
            if log_dir is not None:
                file_log = path.join(log_dir, self.__name__) + '.log'
                with open(file_log, 'a+') as f:
                    f.write('\n' + log_message)

            if terminal_log:
                print(f"LOG = {log_message}")

    def serialize_model(self, models_path="../models/", str_suffix=""):
        file_name = f"{self.__name__}_{str_suffix}.pickle"

        serialize_object = copy.deepcopy(self)
        serialize_object.model = serialize_object.model.state_dict()

        with open(path.join(models_path, file_name), 'wb') as f:
            pickle.dump(serialize_object, f)
            self.log_message(f"Model : {self.__name__} has been serialized on {models_path} as {file_name}")

    def get_name(self):
        return self.__name__

    def score(self, y, y_hat):
        """
        Function to compute evaluation scores on peptide prediciton
        (to be improved with graph metrics)

        Parameters
        ----------
        y : list of str / or list of list of int / list of int / list of chars
            List of strings containing actual peptide sequences or list of amino acids
        y_hat : list of str / or list of list of int / list of int / list of chars
            List of strings containing predicted peptide sequences or list of amino acids

        Returns
        -------
        performances : dict
            Dictionary containing evaluation metrics

        """
        total_preds = len(y)
        performances = {}

        _correct_peptide_sequences = sum(1 for (a, b) in zip(y, y_hat) if a == b)

        peptide_accuracy = _correct_peptide_sequences / total_preds

        performances['peptide_accuracy'] = peptide_accuracy
        return performances
