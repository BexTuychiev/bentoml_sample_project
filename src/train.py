import pandas as pd
import bentoml
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


def train_xgb_save(X, y, tag_name="xgb_final"):
    """
    A simple function to train a model and save it to BentoML model store.
    """

    # Create a model
    model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Train
    model.fit(X, y)

    bentoml.sklearn.save_model(tag_name, model)


if __name__ == "__main__":
    # Load and prep the data
    data = pd.read_csv("data/data.csv")
    X, y = data.drop("target", axis=1), data[["target"]]

    # Train and save
    train_xgb_save(X, y, "xgb_booster")
