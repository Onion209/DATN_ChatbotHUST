import os
import json
import logging

logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise