from sklearn import datasets
import pandas as pd
from pathlib import Path

p = Path('data')
p.mkdir(exist_ok=True)
iris = datasets.load_iris(as_frame=True)
df = iris.frame
# add numeric target column
df['target'] = iris.target
csv_path = p / 'iris.csv'
df.to_csv(csv_path, index=False)
print('Saved iris to', csv_path)
