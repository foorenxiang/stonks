from mlflow.pyfunc import PythonModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericModel(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input=None):
        context
        if model_input:
            logger.info("Running model prediction without input")
            return self.model.predict(model_input)
        logger.info("Running model prediction with input")
        return self.model.predict()

    def access_model_attribute(self, attribute: str) -> any:
        return getattr(self.model, attribute)

    def call_model_method(
        self, method: str, *method_args: any, **method_kwargs: any
    ) -> any:
        return self.access_model_attribute(method)(*method_args, **method_kwargs)


if __name__ == "__main__":
    """Example from https://mlflow.org/docs/latest/models.html#custom-python-models"""
    from mlflow.pyfunc import save_model, load_model

    # Define the model class
    class AddN(PythonModel):
        def __init__(self, n):
            self.n = n

        def predict(self, context, model_input):
            context
            return model_input.apply(lambda column: column + self.n)

    # Construct and save the model
    model_path = "add_n_model"
    add5_model = AddN(n=5)
    save_model(path=model_path, python_model=add5_model)

    # Load the model in `python_function` format
    loaded_model = load_model(model_path)

    # Evaluate the model
    import pandas as pd

    model_input = pd.DataFrame([range(10)])
    model_output = loaded_model.predict(model_input)
    assert model_output.equals(pd.DataFrame([range(5, 15)]))
