def load_keywords_from_text_file(file_path):
    """Read keywords from a text file and return them as a list of strings."""
    with open(file_path, 'r') as file:
        # Read each line, strip leading/trailing whitespace, and store it in a list
        keywords = [line.strip() for line in file.readlines()]
    return keywords
