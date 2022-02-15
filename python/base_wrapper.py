#!/usr/bin/env python3
from abc import ABCMeta, abstractclassmethod

class BaseModelWrapper:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

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

    def score(self, y, y_hat):
        """
        Function to compute evaluation scores on peptide prediciton
        (to be improved with graph metrics)

        Parameters
        ----------
        y : list of str
            List of strings containing actual peptide sequences
        y_hat : list of str
            List of strings containing predicted peptide sequences

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
