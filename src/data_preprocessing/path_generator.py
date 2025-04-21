import os
from src.utils.config_loader import load_config

_, _, ocrtxt_path, s3_path = load_config()

# Define a function to construct the paths
def create_s3_paths(row, folder):
    """Generate a list of the txt paths stored on S3 for each image_id of a geometry."""
    paths = [f"{s3_path}{folder}/{image_id}.txt" for image_id in row['image_ids']]
    return paths

def create_download_paths(row, folder):
    """Generate a list of the local folder paths for each image_id of a geometry.""" 
    path2 = [f"{ocrtxt_path}{folder}/{image_id}".rsplit('/', 1)[0] for image_id in row['image_ids']]
    return path2

def generate_saved_path(row, folder):
    """Generate a list of the local txt file paths for each image_id of a geometry."""
    return [f"{ocrtxt_path}{folder}/{image_id}.txt" for image_id in row['image_ids']]

def create_command_paths(row, folder):
    """Generate a list of the commands to copy the txt file from S3 to local for each image_id of a geometry."""
    path3 = [
        f'./aws/local/aws-cli/v2/2.26.0/bin/aws s3 cp "{s3_path}{folder}/{image_id}.txt" "{ocrtxt_path}{folder}/{image_id}"'.rsplit("/", 1)[0] for image_id in row['image_ids']
    ]
    path3 = [f'{x}/"' for x in path3]
    return path3

# # Apply the function to each row to create new paths
# gdf['s3_path'] = gdf.apply(create_s3_paths, axis=1)
# gdf['download_path'] = gdf.apply(create_download_paths, axis=1)
# gdf['command'] = gdf.apply(create_command_paths, axis=1)

def save_commands_as_txt(gdf, command_path="commands.txt"):
    commands = []
    for command in gdf['command']:
        if isinstance(command, str):
            commands.append(command)
        elif isinstance(command, list):
            for cmd in command:
                commands.append(cmd)

    commands = sorted(list(set(commands)))
# command_path = folder_path + "/" + folder + "-commands.txt"

    with open(command_path, "w") as f:
        for command in commands:
            f.write(command + "\n")
    print("Commands saved to {}".format(command_path))
# 将命令保存到 TXT 文件
