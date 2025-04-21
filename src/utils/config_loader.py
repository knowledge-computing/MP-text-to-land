import os
from dotenv import load_dotenv

def load_config():
    # Load the environment variables from the .env file
    load_dotenv()

    # Get the values from the .env file
    geojson_path = os.getenv('GEOJSON_PATH')
    folder_names = os.getenv('FOLDER_NAMES').split(',')
    ocrtxt_path = os.getenv('OCRTXT_PATH')
    s3_path = os.getenv('S3_PATH')
    
    return geojson_path, folder_names, ocrtxt_path, s3_path