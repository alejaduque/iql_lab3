import os
import csv

def analyze_tokens_files(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Path for the output CSV file
    output_csv_path = os.path.join(output_directory, 'lenghts.csv')
    
    # Create or open the CSV file at the output path
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Initialize the CSV writer
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['File Name', 'Languaje', 'Length'])
        
        # Walk through the input directory
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                # Check if the file has a .tokens extension
                if file.endswith('.tokens'):
                    # Extract the base file name without .txt.tokens
                    base_name = file.replace('.txt.tokens', '')
                    # Get the first two letters of the file name
                    first_two_letters = base_name[:2]
                    # Compute the length of the file in terms of tokens
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tokens = f.read().split()
                            length = len(tokens)
                        # Write to the CSV
                        writer.writerow([base_name, first_two_letters, length])
                    except UnicodeDecodeError as e:
                        print(f"Error reading {file_path}: {e}")

# Directories setup
input_directory = 'data/tokenized'
output_directory = 'data/csv'

# Running the function with specified directories
analyze_tokens_files(input_directory, output_directory)
