import yaml
import logging

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """
    Loads the YAML configuration file containing budget, weights, and scenario parameters.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Validate existence of strict required keys
        required_keys = ['budget', 'unit_cost', 'nsga3', 'parallel', 'weights']
        if not all(key in config for key in required_keys):
            raise KeyError("Configuration file is missing required fields.")

        # Set default values for optional parameters to ensure stability
        config.setdefault('sampling', {'random_init_ratio': 0.25})
        config.setdefault('mutation', {
            'directed_ratio': 0.7,
            'random_ratio': 0.3,
            'replace_min': 0.05,
            'replace_max': 0.15
        })

        logging.info(f"Configuration loaded successfully: {config_path}")
        return config

    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise