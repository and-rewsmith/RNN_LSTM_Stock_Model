from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ArrayNormalizer import ArrayNormalizer

arr1 = [ [[0,2],[3,4]], [[5,6],[7,8]], [[6,8],[9,10]], [[12,15],[7,3]] ]
arr1 =np.asarray(arr1)


norm = ArrayNormalizer(arr1)
print(norm.transform(arr1))
