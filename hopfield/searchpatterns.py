import numpy as np
import random
from dhnnpatterns import patterns, pattern_names

def orthogonality_index(indices, N=100):
    total = 0
    M = len(indices)
    for i in range(M):
        for j in range(i + 1, M):
            a = patterns[indices[i]].flatten()
            b = patterns[indices[j]].flatten()
            total += abs(np.dot(a, b) / N)
    return total / (M * (M - 1) / 2)

def simulated_annealing(pattern_count=5, max_iter=10000, temp_start=1.0, temp_end=0.001, alpha=0.995):
    N = 100  # neurons
    all_indices = list(range(len(patterns)))

    # Initial random selection
    current_indices = random.sample(all_indices, pattern_count)
    current_score = orthogonality_index(current_indices, N)
    best_indices = list(current_indices)
    best_score = current_score
    temperature = temp_start

    for iter in range(max_iter):
        # Propose a new neighbor by swapping one pattern
        new_indices = list(current_indices)
        out_idx = random.choice(new_indices)
        remaining = list(set(all_indices) - set(new_indices))
        in_idx = random.choice(remaining)
        new_indices[new_indices.index(out_idx)] = in_idx

        new_score = orthogonality_index(new_indices, N)

        # Acceptance condition
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_indices = new_indices
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_indices = new_indices

        # Cooling
        temperature = max(temp_end, temperature * alpha)

        if iter % 500 == 0 or iter == max_iter - 1:
            print(f"Iter {iter}, Current Score: {current_score:.4f}, Best: {best_score:.4f}")

    return best_indices, best_score

# Run it
best_indices, best_score = simulated_annealing()
print("\nBest Combo:")
print("Indices:", best_indices)
print("Names:", [pattern_names[i] for i in best_indices])
print("Orthogonality Index:", best_score)
