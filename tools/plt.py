import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
Seed = 2

rdm = np.random.Randomstate(Seed)

X = rdm.randn(300, 2)

Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
print(Y_[0], Y_[1], Y_[2])
Y_c = [['red' if y else 'blue'] for y in Y_]

X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

print(Y_c)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()