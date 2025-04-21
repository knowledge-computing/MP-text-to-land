# Import specific functions or classes from submodules to make them available at the package level
from src.utils.config_loader import load_config
from src.utils.keyword_loader import load_keywords_from_txt_directory
from src.data_preprocessing.path_generator import generate_saved_path, create_s3_paths, create_download_paths, create_command_paths, save_commands_as_txt
from src.data_preprocessing.geojson_loader import load_geojson_to_gdf
from src.data_preprocessing.geosentence_identifier import filter_relevant_sentences
# If needed, initialize other things or perform logging setup
import logging

# Set up basic logging for the entire src package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Initializing the src package.")

# Optionally, define an `__all__` list to specify the public API of this package
# This limits what gets imported when using `from src import *`
__all__ = [
    'load_config',
    'create_s3_paths', 
    'create_download_paths', 
    'generate_saved_path', 
    'create_command_paths',
    'save_commands_as_txt',
    'load_geojson_to_gdf',
    'load_keywords_from_txt_directory',
    'filter_relevant_sentences'
]
