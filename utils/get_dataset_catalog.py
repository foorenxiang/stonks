from utils.yaml_loader import yaml_load
from from_root import from_root


def get_dataset_catalog():
    dataset_catalog = yaml_load(from_root("datasets_catalog.yaml"))
    return dataset_catalog
