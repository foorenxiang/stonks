from autots import AutoTS
import math


class AutoTSConfigs:

    FAST = "FAST"
    ACCURATE = "ACCURATE"
    GPU = "GPU"
    DEFAULT = "DEFAULT"

    FORECAST_LENGTH = 7

    _suggested_validation_trials = math.floor(FORECAST_LENGTH / 3)
    validation_trials = (
        _suggested_validation_trials if _suggested_validation_trials > 2 else 2
    )

    _autots_mode_configs = dict()
    _autots_mode_configs[FAST] = lambda: AutoTS(
        forecast_length=AutoTSConfigs.FORECAST_LENGTH,
        frequency="infer",
        prediction_interval=0.9,
        ensemble=None,
        model_list="superfast",
        transformer_list="fast",
        max_generations=5,
        num_validations=2,
        validation_method="backwards",
    )
    _autots_mode_configs[ACCURATE] = lambda: AutoTS(
        forecast_length=AutoTSConfigs.FORECAST_LENGTH,
        frequency="infer",
        prediction_interval=0.9,
        ensemble="all",
        model_list="parallel",
        transformer_list="fast",
        max_generations=50,
        num_validations=AutoTSConfigs.validation_trials,
        validation_method="backwards",
    )
    _autots_mode_configs[GPU] = lambda: AutoTS(
        forecast_length=AutoTSConfigs.FORECAST_LENGTH,
        frequency="infer",
        prediction_interval=0.9,
        ensemble="auto",
        model_list="gpu",
        transformer_list="all",
        max_generations=50,
        num_validations=AutoTSConfigs.validation_trials,
        validation_method="backwards",
    )
    _autots_mode_configs[DEFAULT] = lambda: AutoTS(
        forecast_length=AutoTSConfigs.FORECAST_LENGTH,
        frequency="infer",
        prediction_interval=0.9,
        ensemble="auto",
        model_list="default",
        transformer_list="all",
        max_generations=50,
        num_validations=AutoTSConfigs.validation_trials,
        validation_method="backwards",
    )

    @classmethod
    def create_model_lambda(cls, selectedMode):
        return cls._autots_mode_configs[getattr(cls, selectedMode)]
