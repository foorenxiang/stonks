from from_root import from_root

PROJECT_ROOT = from_root(".")

DATASETS = PROJECT_ROOT / "datasets"
RAW_DATASETS = DATASETS / "raw"
PREPROCESSED_DATASETS = DATASETS / "preprocessed"

AUTOGLUON_PACKAGE = PROJECT_ROOT / "StonksML"
AUTOGLUON_LOGS = AUTOGLUON_PACKAGE / "autogluon_logs"
AUTOGLUON_MODEL = AUTOGLUON_PACKAGE / "autogluon_model"

AUTOTS_PACKAGE = PROJECT_ROOT / "StonksML"
AUTOTS_LOGS = AUTOTS_PACKAGE / "autots_logs"
AUTOTS_MODEL_DUMPS = AUTOTS_PACKAGE / "autots_model_dumps"

ML_UTILS = PROJECT_ROOT / "StonksML" / "utils"

STREAMLIT_PACKAGE = PROJECT_ROOT / "StonksStreamlit"
