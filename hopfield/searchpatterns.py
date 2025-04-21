import numpy as np
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

def greedy_pattern_selection(target_size=15):
    N = 100
    all_indices = list(range(len(patterns)))

    # Step 1: Start with the pattern with lowest average dot product vs others
    avg_scores = []
    for i in all_indices:
        total = 0
        for j in all_indices:
            if i != j:
                a = patterns[i].flatten()
                b = patterns[j].flatten()
                total += abs(np.dot(a, b) / N)
        avg_scores.append((i, total / (len(all_indices) - 1)))
    seed = min(avg_scores, key=lambda x: x[1])[0]
    selected = [seed]

    # Step 2: Greedily grow the set
    while len(selected) < target_size:
        best_candidate = None
        best_score = float('inf')
        for idx in all_indices:
            if idx in selected:
                continue
            temp = selected + [idx]
            score = orthogonality_index(temp, N)
            if score < best_score:
                best_score = score
                best_candidate = idx
        selected.append(best_candidate)
        print(f"Added {pattern_names[best_candidate]:<5} (index {best_candidate}) — New score: {best_score:.4f} — N={len(selected)}")

    final_score = orthogonality_index(selected, N)
    return selected, final_score

# Run the greedy selection
indices, score = greedy_pattern_selection(target_size=15)
print("\nFinal Greedy Selection:")
print("Indices:", indices)
print("Names:", [pattern_names[i] for i in indices])
print("Orthogonality Index:", score)

