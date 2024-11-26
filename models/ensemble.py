import numpy as np

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def train(self, features, target):
        for model in self.models:
            model.train(features, target)

    def predict(self, future_data):
        predictions = [model.predict(future_data) for model in self.models]
        predictions_mean = np.mean(predictions, axis=0)
        return predictions_mean
