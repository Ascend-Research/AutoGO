import numpy as np


class Normalizer:
    def __init__(self, data):
        self.subtractor, self.divisor = None, None
        if data is not None:
            self.generate_params(data)
    
    def generate_params(self, data):
        targets = [item['acc_mean'] for item in data]
        self.subtractor = np.mean(targets)
        self.divisor = np.std(targets)

    def transform(self, data):
        targets = [item['acc_mean'] for item in data]
        new_targets = (targets - self.subtractor) / self.divisor
        new_targets = new_targets.tolist()
        return self._reassign(data, new_targets)

    @staticmethod
    def _reassign(data, new_targets):
        for idx, instance in enumerate(data):
            instance['acc_mean'] = new_targets[idx]
        return data
