import numpy as np

class ArrayNormalizer:

    def __init__(self, arr):
        self.min = arr.min()
        self.max = arr.max()

    def transform(self, arr):
        arr = (arr - self.min) / (self.max - self.min)
        return arr

    def transform_train_test(self, train, test):
        train = self.transform(train)
        test = self.transform(test)
        return train, test

    def inverse_transform(self, arr):
        arr = arr * (self.max - self.min) + self.min
        return arr