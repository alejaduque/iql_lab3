import pandas as pd
import os

def csv_to_latex(input_csv, output_tex):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)
    
    # Generate LaTeX table code from the DataFrame
    latex_table = df.to_latex(index=False)
    
    # Write the LaTeX table code to an output .tex file
    with open(output_tex, 'w') as f:
        f.write(latex_table)

def process_all_csv_files(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            # Define input and output paths
            input_csv_path = os.path.join(input_directory, filename)
            output_tex_filename = filename.replace('.csv', '.tex')
            output_tex_path = os.path.join(output_directory, output_tex_filename)
            
            # Convert the CSV file to a LaTeX table
            csv_to_latex(input_csv_path, output_tex_path)

# Directories setup
input_directory = 'data/csv'
output_directory = 'data/tables'

# Process all CSV files in the input directory
process_all_csv_files(input_directory, output_directory)
