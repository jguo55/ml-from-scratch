import numpy as np

class DHNN:
    def __init__(self, patterns):
        self.dim = patterns[0].shape
        self.n = len(patterns[0].flatten())
        self.wmatrix = np.zeros((self.n,self.n))
        for p in patterns:
            pflat = p.flatten()
            self.wmatrix += np.outer(pflat, pflat)
        np.fill_diagonal(self.wmatrix, 0)
        self.wmatrix = self.wmatrix / len(patterns)

    def predict_async(self, state, num_iter=100):
        state = np.array(state).flatten()
        for _ in range(num_iter):
            i = np.random.randint(self.n-1)
            input_sum = np.dot(self.wmatrix[i], state)
            state[i] = 1 if input_sum >= 0 else -1
        return
    
    def display(self, state):
        state = state.reshape(self.dim)
        for row in state:
            print('[ ' + ''.join('* ' if val == 1 else '  ' for val in row) + ']')

zero = np.array([
    [-1, 1, 1, 1, -1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [-1, 1, 1, 1, -1]
])

one = np.array([
    [-1, 1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1]
])

two = np.array([
    [1, 1, 1, -1, -1],
    [-1, -1, -1, 1, -1],
    [-1, -1, -1, 1, -1],
    [-1, 1, 1, -1, -1],
    [1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1]
])

patterns = [zero, one, two]
dhnn = DHNN(patterns)

new = np.array([
    [-1, -1, -1, 1, 1],
    [1, 1, 1, -1, 1],
    [-1, -1, -1, 1, -1],
    [1, -1, -1, 1, 1],
    [1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])

rand = np.random.choice([0,1],size=(6,5))

dhnn.display(dhnn.predict_async(rand))
