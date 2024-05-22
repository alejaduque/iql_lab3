import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(file_path):
    return np.loadtxt(file_path)

def power_law_with_constant(x, C, alpha, D):
    return C * x ** (-alpha) + D

def plot_mi_d(mi_data, pvalues_data, avg_shuffled_mi_data, output_path, log_scale=True):
    d_values = np.arange(1, len(mi_data))
    mi_values = mi_data[1:]  # Ignore the zero-distance value
    pvalues = pvalues_data[1:]  # Ignore the zero-distance value
    avg_shuffled_mi_values = avg_shuffled_mi_data[1:]  # Ignore the zero-distance value

    plt.figure(figsize=(7, 5))
    
    plt.scatter(d_values, mi_values, color='red', label='p-value >= 0.05')
    significant = pvalues < 0.05
    plt.scatter(d_values[significant], mi_values[significant], color='blue', label='p-value < 0.05')
    plt.plot(d_values, avg_shuffled_mi_values, color='orange', label='Random MI', linestyle='-')

    if np.any(significant):
        try:
            params, _ = curve_fit(power_law_with_constant, d_values[significant], mi_values[significant], p0=[1, 1, 1])
            regression_d_values = np.linspace(d_values[significant][0], d_values[significant][-1], 100)
            regression_line = power_law_with_constant(regression_d_values, *params)
            plt.plot(regression_d_values, regression_line, color='green', label=r'$I(d) \sim d^{-\alpha} + D$')
        except RuntimeError:
            print(f"Optimal parameters not found for {output_path}")

    plt.xlabel('Distance (d)')
    plt.ylabel('Mutual Information I(d)')
    plt.title('Mutual Information I(d) vs. Distance (d)')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.title('I(d) vs. d')

    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def save_theil_sen_data(mi_files, mi_results_dir, output_file):
    with open(output_file, 'w') as f:
        f.write("File,C,Alpha,D\n")
        for mi_file in mi_files:
            base_name = os.path.splitext(mi_file)[0]
            mi_file_path = os.path.join(mi_results_dir, mi_file)
            pvalues_file_path = os.path.join(mi_results_dir, f"{base_name}.pvalues")
            avg_shuffled_mi_file_path = os.path.join(mi_results_dir, f"{base_name}.avg_shuffled_mi")
            
            mi_data = load_data(mi_file_path)
            pvalues_data = load_data(pvalues_file_path)
            avg_shuffled_mi_data = load_data(avg_shuffled_mi_file_path)
            
            output_plot_path = os.path.join("data/test_plots", f"{base_name}.png")
            plot_mi_d(mi_data, pvalues_data, avg_shuffled_mi_data, output_plot_path)

            if np.any(pvalues_data[1:] < 0.05):
                d_values = np.arange(1, len(mi_data))
                significant = pvalues_data[1:] < 0.05
                try:
                    params, _ = curve_fit(power_law_with_constant, d_values[significant], mi_data[1:][significant], p0=[1, 1, 1])
                    f.write(f"{base_name},{params[0]},{params[1]},{params[2]}\n")
                except RuntimeError:
                    f.write(f"{base_name},NaN,NaN,NaN\n")
    print(f"Theil-Sen data saved to {output_file}")

if __name__ == "__main__":
    mi_results_dir = "data/mi_results"
    output_plot_dir = "data/test_plots"
    output_theil_sen_file = "data/csv/power_ct.csv"
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
    print("All plots and fitting data generated.")
