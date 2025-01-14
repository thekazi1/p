from scipy import *  # This is not recommended due to potential name conflicts with other modules. Use specific imports instead.
import numpy as np  # Replace "from scipy import *" with this import if needed.
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt


class RBF:
    def __init__(self, indim, num_centers, outdim):
        self.indim = indim
        self.outdim = outdim
        self.num_centers = num_centers
        self.centers = [np.random.uniform(-1, 1, indim) for _ in range(num_centers)]
        self.beta = 8
        self.W = np.random.rand(num_centers, outdim)

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-self.beta * norm(c - d) ** 2)

    def _calc_act(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.num_centers), dtype=float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
           y: column vector of dimension n x 1 """
        # choose random center vectors from training set
        rand_idx = np.random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rand_idx]
        print("Center:", self.centers)
        # calculate activations of RBFs
        G = self._calc_act(X)
        print("Activations:", G)
        # calculate output weights (pseudo inverse)
        self.W = np.dot(np.linalg.pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calc_act(X)
        Y = np.dot(G, self.W)
        return Y

if __name__ == "__main__":
    # 1D example
    n = 100
    x = np.mgrid[-1:1:complex(0, n)].reshape(n, 1)
    # Set y and add random noise
    y = np.sin(3 * (x + 0.5) ** 3 - 1)
    # y += np.random.normal(0, 0.1, y.shape)
    # RBF Regression
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # Plot Original Data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, y, 'k-')
    # Plot Learned Model
    ax.plot(x, z, 'r-', lw=2)
    # Plot RBF centers
    ax.plot(rbf.centers, np.zeros(rbf.num_centers), 'gs')
    for c in rbf.centers:
        # RBF Prediction Lines
        cx = np.arange(c - 0.7, c + 0.7, 0.01)
        cy = [rbf._basisfunc(np.array([cx_]), np.array([c])) for cx_ in cx]
        ax.plot(cx, cy, '-', color='gray', lw=0.2)
    ax.set_xlim(-1.2, 1.2)
    plt.show()
