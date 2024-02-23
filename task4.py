# # Apparently surprise only works on python 3.7 in a .py file :P

import pandas as pd
from surprise import Dataset, SVD, Reader

# Define a reader for Surprise
reader = Reader(rating_scale=(1, 5))

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_set = Dataset.load_from_df(train_df[['User ID', 'Movie ID', 'Rating']], reader).build_full_trainset()

# Initialize and train SVD algorithm
algo = SVD()
algo.fit(train_set)

# Make predictions on the test set
predictions = []
for idx, row in test_df.iterrows():
    pred = algo.predict(row['User ID'], row['Movie ID'])
    predictions.append(pred.est)

print(predictions)