import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('corpus.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_data.to_csv('train_dataset.csv', index=False)
val_data.to_csv('validation_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)
