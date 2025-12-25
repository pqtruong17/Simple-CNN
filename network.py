import numpy as np


class Linear:
    # (in_size, out_size) is (row, collumn)
    def __init__(self, in_size, out_size):
        self.weight = np.random.rand(in_size, out_size)
        self.bias = np.zeros((out_size, 1))
        self.params = [self.weight, self.bias]
        self.gradWeight = None
        self.gradBias = None
        self.gradInput = None

    def foward(self, X):
        self.X = X
        self.gradInput = np.matmul(self.weight, self.X)
        return self.gradInput

    def backward(self, nextGrad):
        self.gradWeight = np.matmul(nextGrad, self.X.T)
        self.gradInput = np.matmul(nextGrad.T, self.weight)
        return self.gradInput, self.gradWeight


class Network:
    # Take in layers objects
    def __init__(self):
        self.layers = []
        self.gradInput = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def foward(self, nextGrad):
        for layer in self.layers:
            nextGrad = layer.foward(nextGrad)
        self.gradInput = nextGrad
        return self.gradInput

    def backward(self):
        self.gradWeights = []
        for layer in reversed(self.layers):
            gradInput = layer.gradInput
            gradInput, gradWeight = layer.backward(gradInput)
            self.gradWeights.append(gradWeight)
        return gradInput


X = np.random.rand(4, 1)
nn = Network()
nn.addLayer(Linear(3, 4))
nn.addLayer(Linear(2, 3))
f = nn.foward(X)
# print(f)
b = nn.backward()
for w in nn.gradWeights:
    print(w)
    print()
