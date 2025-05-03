import numpy as np
from dhnnpatterns import patterns, pattern_names

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

    def predict_async(self, state, num_iter=1000):
        state = np.array(state).flatten()
        for _ in range(num_iter):
            i = np.random.randint(self.n-1)
            input_sum = np.dot(self.wmatrix[i], state)
            state[i] = 1 if input_sum >= 0 else -1
        return state
    
    def display(self, state):
        state = state.reshape(self.dim)
        for row in state:
            print('[ ' + ''.join('* ' if val == 1 else '  ' for val in row) + ']')

indices = [18, 0, 31, 13, 4, 61, 14, 3]#, 67, 10, 17, 16, 22, 11, 64]
candidates = [patterns[i] for i in indices]
candidate_names = [pattern_names[i] for i in indices]

dhnn = DHNN(candidates)

rand = np.random.choice([-1,1],size=(10,10))

print(candidate_names)
dhnn.display(dhnn.predict_async(rand))
