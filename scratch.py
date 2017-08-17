from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ArrayNormalizer import ArrayNormalizer

arr1 = [ [0, 1, 2], [3, 4, 5], [8, 9, 10] ]

arr1 = np.asarray(arr1)

print(type(arr1))
arr1 = arr1.tolist()
print(type(arr1))



# arr1 = np.asarray(arr1)
#
# print(arr1[0].tolist().insert(0, 1))
#
# print(arr1[0])