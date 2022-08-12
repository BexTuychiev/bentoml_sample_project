import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# Get the runner
xgb_runner = bentoml.models.get("xgb_booster:latest").to_runner()

# Create a Service object
svc = bentoml.Service("xgb_service", runners=[xgb_runner])


# Create an endpoint named classify
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series) -> np.ndarray:
    # Convert the input string to numpy array
    label = xgb_runner.predict.run(input_series)

    return label
