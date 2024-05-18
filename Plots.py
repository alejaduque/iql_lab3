import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    return np.loadtxt(file_path)

def plot_mi_d(mi_data, pvalues_data, output_path, log_scale=True):
    d_values = np.arange(1, len(mi_data))
    mi_values = mi_data[1:]  # Ignore the zero-distance value
    pvalues = pvalues_data[1:]  # Ignore the zero-distance value

    plt.figure(figsize=(10, 6))
    
    # Scatter plot for all points
    plt.scatter(d_values, mi_values, color='red', label='p-value >= 0.05')

    # Highlight points with p-value < 0.05
    significant = pvalues < 0.05
    plt.scatter(d_values[significant], mi_values[significant], color='blue', label='p-value < 0.05')
    
    plt.xlabel('Distance (d)')
    plt.ylabel('Mutual Information I(d)')
    plt.title('Mutual Information I(d) vs. Distance (d)')
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.title('I(d) vs. d')

    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    mi_results_dir = "data/mi_results"
    output_plot_dir = "data/plots"
    os.makedirs(output_plot_dir, exist_ok=True)

    mi_files = [f for f in os.listdir(mi_results_dir) if f.endswith('.mi')]

    for mi_file in mi_files:
        base_name = os.path.splitext(mi_file)[0]
        mi_file_path = os.path.join(mi_results_dir, mi_file)
        pvalues_file_path = os.path.join(mi_results_dir, f"{base_name}.pvalues")
        
        mi_data = load_data(mi_file_path)
        pvalues_data = load_data(pvalues_file_path)
        
        output_plot_path = os.path.join(output_plot_dir, f"{base_name}.png")
        plot_mi_d(mi_data, pvalues_data, output_plot_path)

    print("All plots generated.")
