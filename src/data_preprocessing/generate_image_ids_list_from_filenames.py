import os

def group_files_by_prefix(root_dir):
    """
    This function will generate the same pattern as the 'image_ids' in the geojson file, i.e., the splitpages' file paths of the same deed document would be grouped together, while the file path of the deed document with only one txt file will be stored in an individual list.
    This function will recursively traverse the directory tree and group all .txt files by their prefix (before the last '_').
    It will return a list of lists, where each inner list contains the file paths of .txt files that are generated from the same deed document (i.e., they share the same prefix before the last '_').
    :param root_dir: root directory path to search for .txt files
    :return: the grouped file paths as a list of lists
    """
    # to store all file paths of .txt files (without suffix)
    all_files = []
    
    # traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # process all files in the current directory
        for filename in filenames:
            if filename.endswith('.txt'):
                # construct the full file path (without suffix)
                file_path = os.path.join(dirpath, filename[:-4])
                # file_path = file_path.split('/', 10)[-1]
                all_files.append(file_path)
    
    # sort all the file paths
    all_files.sort()
    groups = {}
    # print(all_files[:20])

    for file_path in all_files:
        # get the file names, without path
        base_name = os.path.basename(file_path)
        
        # split by the last '_'
        parts = base_name.rsplit('_', 1)
        
        if len(parts) >= 2:
            prefix, number = parts
            # prefix as key, add entire path to the list
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(file_path)
        else:
            # if the path is unable to be splitted, store in an individual list
            groups[file_path] = [file_path]
    
    # dict to list conversion
    result = list(groups.values())
    
    return result