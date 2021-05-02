from kedro.pipeline import Pipeline, node

from .nodes import create_master_table, preprocess_companies, preprocess_shuttles


def create_pipeline(**kwargs):
    return Pipeline([])
