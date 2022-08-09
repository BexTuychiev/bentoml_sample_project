import xgboost as xgb
import pandas as pd
import bentoml
import warnings

warnings.filterwarnings("ignore")


def train_xgb_save(X, y, tag_name="xgb_final"):
    """
    A simple function to train a model and save it to BentoML model store.
    """
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    # Specify parameters for a binary classification problem
    params = {"objective": "binary:logistic", "booster": "gbtree",
              "tree_method": "gpu_hist", "eval_metric": "auc"}

    # Train
    booster = xgb.train(params, dtrain, num_boost_round=20)

    bentoml.xgboost.save_model(tag_name, booster)


if __name__ == "__main__":
    # Load and prep the data
    data = pd.read_csv("data/data.csv")
    X, y = data.drop("target", axis=1), data[["target"]]

    # Train and save
    train_xgb_save(X, y, "xgb_booster")
