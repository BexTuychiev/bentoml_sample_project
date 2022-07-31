from sklearn.datasets import make_classification
import pandas as pd


def generate_data(n_samples, n_features, n_informative, path):
    """
    A simple function to save a synthetic dataset to path.
    """
    # Generate the data
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative)

    # Save it as a CSV
    feature_names = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    df.to_csv(path, index=False)


if __name__ == "__main__":
    n_samples, n_features = 10000, 7
    generate_data(n_samples, n_features, 5, "data/data.csv")
