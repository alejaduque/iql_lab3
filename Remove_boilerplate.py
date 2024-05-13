import os
import re

def remove_gutenberg_boilerplate(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if re.match(r"\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG EBOOK", line, re.IGNORECASE):
            start_index = i
        elif re.match(r"\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG EBOOK", line, re.IGNORECASE):
            end_index = i

    if start_index is not None and end_index is not None:
        cleaned_lines = lines[start_index+1:end_index]
        cleaned_text = ''.join(cleaned_lines)
    else:
        cleaned_text = ''.join(lines)

    return cleaned_text

def remove_gutenberg_boilerplate_from_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            cleaned_text = remove_gutenberg_boilerplate(input_file_path)
            # Write the cleaned text to the output file
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            print(f"Boilerplate removed from: {filename}")

input_folder_path = "data/original"

output_folder_path = "data/no_boilerplate"

remove_gutenberg_boilerplate_from_folder(input_folder_path, output_folder_path)
