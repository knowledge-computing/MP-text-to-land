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

# def load_config_shp():
#     # Load the environment variables from the .env file
#     load_dotenv()

#     # Get the values from the .env file
#     shp_path = os.getenv('SHP_PATH')
    
#     return shp_path

def load_openai_config():
    # Load the environment variables from the .env file
    load_dotenv()

    # Get the values from the .env file
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    return openai_api_key