import numpy as np
from itertools import combinations

import numpy as np

# 0
zero = np.full((10, 10), -1)
zero[0:10, 0] = zero[0:10, 9] = 1
zero[0, 1:9] = zero[9, 1:9] = 1

# 1
one = np.full((10, 10), -1)
one[0:10, 4] = 1
one[9, 0:5] = 1

# 2
two = np.full((10, 10), -1)
two[0, 1:9] = two[4, 1:9] = two[9, 1:9] = 1
two[5:9, 8] = 1

# 3
three = np.full((10, 10), -1)
three[0, 1:9] = three[4, 1:9] = three[9, 1:9] = 1
three[5:9, 8] = 1

# 4
four = np.full((10, 10), -1)
four[0:5, 4] = 1
four[5, 0:10] = 1

# 5
five = np.full((10, 10), -1)
five[0, 1:9] = five[4, 1:9] = five[9, 1:9] = 1
five[1:5, 0] = 1

# 6
six = np.full((10, 10), -1)
six[0, 1:9] = six[4, 1:9] = six[9, 1:9] = 1
six[0:5, 0] = 1

# 7
seven = np.full((10, 10), -1)
seven[0, 1:9] = seven[9, 1:9] = 1
seven[5:9, 8] = 1

# 8
eight = np.full((10, 10), -1)
eight[0, 1:9] = eight[4, 1:9] = eight[9, 1:9] = 1
eight[5:9, 8] = 1

# 9
nine = np.full((10, 10), -1)
nine[0, 1:9] = nine[4, 1:9] = nine[9, 1:9] = 1
nine[1:5, 0] = 1

# A
A = np.full((10, 10), -1)
A[1, 4:6] = A[2, 3:7] = A[3, 2:8] = A[4, 1:9] = A[5, 0:10] = 1

# B
B = np.full((10, 10), -1)
B[0:10, 0] = B[1, 1:5] = B[2, 1:5] = B[4, 1:5] = B[5, 1:5] = B[9, 1:5] = 1

# C
C = np.full((10, 10), -1)
C[0, 1:9] = C[9, 1:9] = C[1:9, 0] = 1

# D
D = np.full((10, 10), -1)
D[0:10, 0] = D[1, 1:5] = D[2, 1:5] = D[4, 1:5] = D[5, 1:5] = D[9, 1:5] = 1

# E
E = np.full((10, 10), -1)
E[0, 1:9] = E[4, 1:9] = E[9, 1:9] = E[1:10, 0] = 1

# F
F = np.full((10, 10), -1)
F[0, 1:9] = F[4, 1:9] = F[1:10, 0] = 1

# G
G = np.full((10, 10), -1)
G[0, 1:9] = G[9, 1:9] = G[1:9, 0] = G[5, 6:9] = 1

# H
H = np.full((10, 10), -1)
H[0:10, 0] = H[0:10, 9] = H[4, 0:10] = 1

# I
I = np.full((10, 10), -1)
I[0:10, 4] = 1
I[9, 0:10] = 1

# J
J = np.full((10, 10), -1)
J[0:10, 4] = J[9, 0:5] = 1

# K
K = np.full((10, 10), -1)
K[0:10, 0] = K[4, 0:10] = K[9, 0] = 1

# L
L = np.full((10, 10), -1)
L[0:10, 0] = L[9, 0:10] = 1

# M
M = np.full((10, 10), -1)
M[0, 0:10] = M[9, 0:10] = M[4, 0:10] = 1

# N
N = np.full((10, 10), -1)
N[0, 0:10] = N[9, 0:10] = N[4, 0:10] = 1

# O
O = np.full((10, 10), -1)
O[0, 1:9] = O[9, 1:9] = O[1:9, 0] = O[1:9, 9] = 1

# P
P = np.full((10, 10), -1)
P[0, 1:9] = P[4, 1:9] = P[9, 1:9] = P[1:5, 0] = 1

# Q
Q = np.full((10, 10), -1)
Q[0, 1:9] = Q[9, 1:9] = Q[1:9, 0] = Q[1:9, 9] = 1

# R
R = np.full((10, 10), -1)
R[0, 1:9] = R[4, 1:9] = R[9, 1:9] = R[1:5, 0] = 1

# S
S = np.full((10, 10), -1)
S[0, 1:9] = S[4, 1:9] = S[9, 1:9] = S[1:5, 0] = 1

# T
T = np.full((10, 10), -1)
T[0, 1:9] = T[9, 1:9] = 1

# U
U = np.full((10, 10), -1)
U[0:10, 0] = U[9, 0:10] = U[1:9, 0] = 1

# V
V = np.full((10, 10), -1)
V[0, 1:9] = V[9, 1:9] = 1

# W
W = np.full((10, 10), -1)
W[0, 0:10] = W[9, 0:10] = W[4, 0:10] = 1

# X
X = np.full((10, 10), -1)
X[0, 0:10] = X[9, 0:10] = X[4, 0:10] = 1

# Y
Y = np.full((10, 10), -1)
Y[0, 1:9] = Y[9, 1:9] = Y[4, 0:10] = 1

# Z
Z = np.full((10, 10), -1)
Z[0, 1:9] = Z[4, 1:9] = Z[9, 1:9] = 1


patterns = [zero, one, two, three, four, five, six, seven, eight, nine, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z]
pattern_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

M = 5  # Number of patterns in each combo
N = 100  # Neurons per pattern
num_pairs = M * (M - 1) / 2

# Track best/worst
min_score = float('inf')
max_score = float('-inf')
min_combo = None
max_combo = None

# Loop through all 3-pattern combinations
for combo_indices in combinations(range(len(patterns)), M):
    combo = [patterns[i] for i in combo_indices]
    total = 0
    for i in range(M):
        for j in range(i + 1, M):
            raw_dot = np.dot(combo[i].flatten(), combo[j].flatten())
            norm_dot = abs(raw_dot / N)
            total += norm_dot

    score = total / num_pairs    
    if score < min_score:
        min_score = score
        min_combo = combo_indices

    if score > max_score:
        max_score = score
        max_combo = combo_indices

    #print(f"Combo {combo_indices}: {score:.4f}")

# Final results
print("\nLowest Orthogonality Index:")
print(f"Combo indices: {min_combo} — Score: {min_score:.4f}")

print("\nHighest Orthogonality Index:")
print(f"Combo indices: {max_combo} — Score: {max_score:.4f}")

