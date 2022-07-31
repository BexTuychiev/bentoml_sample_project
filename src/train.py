import xgboost as xgb
import pandas as pd
import bentoml


def train_xgb_save(X, y, tag_name="xgb_final"):
    """
    A simple function to train a model and save it to BentoML model store.
    """
    # Initialize a classifier
    clf = xgb.XGBClassifier(tree_method="gpu_hist")

    # Train and save
    clf.fit(X, y)

    bentoml.sklearn.save_model(tag_name, clf)


if __name__ == "__main__":
    # Load and prep the data
    data = pd.read_csv("data/data.csv")
    X, y = data.drop("target", axis=1), data[["target"]]

    # Train and save
    train_xgb_save(X, y)
