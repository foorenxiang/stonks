import yaml
import logging

logger = logging.getLogger(__name__)


def yaml_load(yaml_filepath):
    with open(yaml_filepath, "r") as fp:
        try:
            data = yaml.safe_load(fp)
            return data
        except yaml.YAMLError:
            logger.exception(f"Failed to read yaml file at {yaml_filepath}")
