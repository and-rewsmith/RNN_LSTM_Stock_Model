from sklearn.preprocessing import MinMaxScaler
import numpy as np

arr = [[0,1,3],
        [4,5,6]]

arr = np.asarray(arr)

arr = (np.hstack(arr))
mmscaler = MinMaxScaler(feature_range=(0, 1))
#
# print(arr)
# print()
#
print((mmscaler.fit_transform(arr)))
