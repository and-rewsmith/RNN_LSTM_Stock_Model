from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ArrayNormalizer import ArrayNormalizer

arr1 = [ [0, 1, 2], [3, 4, 5], [8, 9, 10] ]
sents = [-1, -2, -3]

for i in range(0, len(arr1)):
    arr1[i].append(sents[i])

print(arr1)
