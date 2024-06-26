import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

def load_data(file_path):
    return np.loadtxt(file_path)

def plot_mi_d(mi_data, pvalues_data, avg_shuffled_mi_data, output_path, log_scale=True):
    d_values = np.arange(1, len(mi_data))
    mi_values = mi_data[1:]  # Ignore the zero-distance value
    pvalues = pvalues_data[1:]  # Ignore the zero-distance value
    avg_shuffled_mi_values = avg_shuffled_mi_data[1:]  # Ignore the zero-distance value

    plt.figure(figsize=(7, 5))
    
    # Scatter plot for all points
    plt.scatter(d_values, mi_values, color='red', label='p-value >= 0.05')

    # Highlight points with p-value < 0.05
    significant = pvalues < 0.05
    plt.scatter(d_values[significant], mi_values[significant], color='blue', label='p-value < 0.05')

    # Plot the avg_shuffled_mi_data as an orange line
    plt.plot(d_values, avg_shuffled_mi_values, color='orange', label='Random MI', linestyle='-')

    # Perform Theil-Sen regression on log-transformed significant points
    if np.any(significant):
        log_d_values = np.log(d_values[significant])
        log_mi_values = np.log(mi_values[significant])
        theil_sen_slope, theil_sen_intercept, _, _ = theilslopes(log_mi_values, log_d_values)
        
        last_significant_point = d_values[significant][-1]
        regression_d_values = np.linspace(d_values[significant][0], last_significant_point, 100)
        regression_line_log = theil_sen_intercept + theil_sen_slope * np.log(regression_d_values)
        regression_line = np.exp(regression_line_log)
        
        plt.plot(regression_d_values, regression_line, color='green', label=r'$I(d) \sim d^\alpha$')

    plt.xlabel('Distance (d)')
    plt.ylabel('Mutual Information I(d)')
    plt.title('Mutual Information I(d) vs. Distance (d)')
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.title('I(d) vs. d')

    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Adjust layout to remove extra whitespace
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

    return theil_sen_slope, theil_sen_intercept

def save_theil_sen_data(mi_files, mi_results_dir, output_file):
    with open(output_file, 'w') as f:
        f.write("File,Slope,Intercept\n")
        for mi_file in mi_files:
            base_name = os.path.splitext(mi_file)[0]
            mi_file_path = os.path.join(mi_results_dir, mi_file)
            pvalues_file_path = os.path.join(mi_results_dir, f"{base_name}.pvalues")
            avg_shuffled_mi_file_path = os.path.join(mi_results_dir, f"{base_name}.avg_shuffled_mi")
            
            mi_data = load_data(mi_file_path)
            pvalues_data = load_data(pvalues_file_path)
            avg_shuffled_mi_data = load_data(avg_shuffled_mi_file_path)
            
            output_plot_path = os.path.join("data/plots", f"{base_name}.png")
            theil_sen_slope, theil_sen_intercept = plot_mi_d(mi_data, pvalues_data, avg_shuffled_mi_data, output_plot_path)
            f.write(f"{base_name},{theil_sen_slope},{theil_sen_intercept}\n")
    print(f"Theil-Sen data saved to {output_file}")

if __name__ == "__main__":
    mi_results_dir = "data/mi_results"
    output_plot_dir = "data/plots"
    output_theil_sen_file = "data/csv/theil_sen_data.csv"
    os.makedirs(output_plot_dir, exist_ok=True)

    mi_files = [f for f in os.listdir(mi_results_dir) if f.endswith('.mi')]

    for mi_file in mi_files:
        base_name = os.path.splitext(mi_file)[0]
        mi_file_path = os.path.join(mi_results_dir, mi_file)
        pvalues_file_path = os.path.join(mi_results_dir, f"{base_name}.pvalues")
        avg_shuffled_mi_file_path = os.path.join(mi_results_dir, f"{base_name}.avg_shuffled_mi")
        
        mi_data = load_data(mi_file_path)
        pvalues_data = load_data(pvalues_file_path)
        avg_shuffled_mi_data = load_data(avg_shuffled_mi_file_path)
        
        output_plot_path = os.path.join(output_plot_dir, f"{base_name}.png")
        plot_mi_d(mi_data, pvalues_data, avg_shuffled_mi_data, output_plot_path)

    save_theil_sen_data(mi_files, mi_results_dir, output_theil_sen_file)
    print("All plots and Theil-Sen data generated.")
