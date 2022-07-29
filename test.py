import numpy as np

# a = np.array([[1, 1, 1],
#               [1, 0, 1],
#               [0, 0, 0]])
a = np.array([[[0], [0]]])
# b = [[1, 1, 1],
#      [1, 0, 1],
#      [0, 1, 0]]
b = (a == 0).astype(np.uint8)
print((a == 0).astype(np.uint8))
print(np.sum(b))
# print(a.shape)
