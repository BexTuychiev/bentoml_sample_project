import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, Text

# Get the runner
xgb_runner = bentoml.models.get("xgb_final").to_runner()

# Create a Service object
svc = bentoml.Service("xgb_service", runners=[xgb_runner])


# Create an endpoint named classify
@svc.api(input=Text(), output=NumpyNdarray())
def classify(input_series) -> np.ndarray:
    # Convert the input string to numpy array
    array = np.fromstring(input_series, np.uint8)

    label = xgb_runner.predict.run(array)

    return label
