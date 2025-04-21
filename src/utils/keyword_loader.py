import os

def load_keywords_from_txt_directory(directory_path="data/keywords/"):
    """Read keywords from all text files in a directory and return them as a concatenated list of strings."""
    all_keywords = []
    for filename in os.listdir(directory_path):
        # Only process .txt files
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                # Read each line, strip leading/trailing whitespace, and store it in a list
                keywords = [line.strip() for line in file.readlines()]
                all_keywords.extend(keywords)  # Add the keywords from this file to the list
    return all_keywords
