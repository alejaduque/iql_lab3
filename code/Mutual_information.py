import os
import numpy as np
import pandas as pd
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from scipy.stats import norm

# Dictionary for the pairs of words
def create_dataframe(words_list, distance):
    # Find all the pairs at given distance 
    pairs = [(words_list[i], words_list[i + distance], i) for i in range(len(words_list) - distance)]
    # Save into a dataframe
    df = pd.DataFrame(pairs, columns=['Token x', 'Token y', 'Position of Token x'])
    df['Distance'] = distance
    return df

def collect_positions(df):
    df_grouped = df.groupby(['Token x', 'Token y']).agg({'Position of Token x': list}).reset_index()
    return df_grouped

# Entropies
def H_X(pairs) -> float:
    token_counts = Counter(pairs[0])
    F = len(pairs[0])
    H = 0

    for token_x, fx in token_counts.items():
        if fx != 0:
            H += fx * np.log(fx)

    H /= F
    H = np.log(F) - H

    return H

def H_Y(pairs) -> float:
    token_counts = Counter(pairs[1])
    F = len(pairs[0])
    H = 0

    for token_y, fy in token_counts.items():
        if fy != 0:
            H += fy * np.log(fy)

    H /= F
    H = np.log(F) - H
    return H

def H_XY(pairs, pairs_gr) -> float:
    F = len(pairs[0])
    H = 0
    for pair in pairs_gr:
        fr = len(pair[2])
        if fr != 0:
            H += fr * np.log(fr)

    H /= F
    H = np.log(F) - H

    return H

# Mutual information
def I(pairs, pairs_gr) -> float: 
    HX = H_X(pairs)
    HY = H_Y(pairs)
    H = H_XY(pairs, pairs_gr)
    return HX + HY - H

def mutual_information(tokens, max_d):
    MI = np.zeros(max_d)
    for i in range(1, max_d):
        pairs = create_dataframe(tokens, i)
        pairs_grouped = collect_positions(pairs)

        pairs_np = np.transpose(pairs.to_numpy())
        pairs_grouped_np = pairs_grouped.to_numpy()

        MI[i] = I(pairs_np, pairs_grouped_np)
    return MI

def shuffle_tokens(tokens):
    shuffled = tokens[:]
    random.shuffle(shuffled)
    return shuffled

def calculate_shuffled_mi(tokens, max_d, num_shuffles):
    with Pool(cpu_count()) as pool:
        shuffled_mis = pool.starmap(mutual_information, [(shuffle_tokens(tokens), max_d) for _ in range(num_shuffles)])
    return np.array(shuffled_mis)

def calculate_p_values(observed_mi, shuffled_mis):
    p_values = np.zeros_like(observed_mi)
    for d in range(1, len(observed_mi)):
        shuffled_mi_d = shuffled_mis[:, d]
        mean_shuffled = np.mean(shuffled_mi_d)
        std_shuffled = np.std(shuffled_mi_d)
        z_score = (observed_mi[d] - mean_shuffled) / std_shuffled
        p_values[d] = 1 - norm.cdf(z_score)
    return p_values

def calculate_mi_and_p_values(file_path, max_d, num_shuffles):
    print(f"Processing {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    observed_mi = mutual_information(tokens, max_d)
    shuffled_mis = calculate_shuffled_mi(tokens, max_d, num_shuffles)
    p_values = calculate_p_values(observed_mi, shuffled_mis)
    avg_shuffled_mi = np.mean(shuffled_mis, axis=0)
    return os.path.basename(file_path), observed_mi, p_values, avg_shuffled_mi

def save_results(result, output_dir):
    filename, mi, p_values, avg_shuffled_mi = result
    os.makedirs(output_dir, exist_ok=True)
    mi_save_path = os.path.join(output_dir, f"{filename}.mi")
    pv_save_path = os.path.join(output_dir, f"{filename}.pvalues")
    avg_save_path = os.path.join(output_dir, f"{filename}.avg_shuffled_mi")
    np.savetxt(mi_save_path, mi)
    np.savetxt(pv_save_path, p_values)
    np.savetxt(avg_save_path, avg_shuffled_mi)
    print(f"Saved mutual information, p-values, and average shuffled MI for {filename}")

if __name__ == "__main__":
    tokenized_dir = "data/tokenized"
    output_dir = "data/mi_results"
    max_d = 30  # Define maximum distance
    num_shuffles = 40  # Define the number of shuffles for p-value calculation

    print(f"Maximum distance (max_d): {max_d}")
    print(f"Number of shuffles: {num_shuffles}")

    # List all tokenized files
    tokenized_files = [os.path.join(tokenized_dir, f) for f in os.listdir(tokenized_dir) if f.endswith('.tokens')]

    print("Starting mutual information and p-value calculation...")
    for file in tokenized_files:
        result = calculate_mi_and_p_values(file, max_d, num_shuffles)
        save_results(result, output_dir)
    
    print("\nMutual information and p-value calculation and saving complete.")
