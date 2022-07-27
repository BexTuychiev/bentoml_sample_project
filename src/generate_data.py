from sklearn.datasets import make_classification
import pandas as pd

n_samples = 100000
n_features = 10
seed = 42

# Generate the data
X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=seed)

col_names = [f'feature_{i}' for i in range(n_features)] + ['target']

df = pd.DataFrame(data=X)
df['target'] = y

df.columns = col_names

# Save
df.to_csv('data/data.csv', index=False)
