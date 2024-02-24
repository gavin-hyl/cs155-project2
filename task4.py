# # Apparently surprise only works on python 3.7 in a .py file :P
import numpy as np
import pandas as pd
from surprise import Dataset, SVD, Reader
import matplotlib.pyplot as plt

# Define a reader for Surprise
reader = Reader(rating_scale=(1, 5))

# Read the csv files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
movies = pd.read_csv('movies.csv')
train_df = train_df[['User ID', 'Movie ID', 'Rating']]
print(train_df.describe())
# create a training set
train_set = Dataset.load_from_df(train_df, reader).build_full_trainset()

# Initialize and train SVD algorithm
algo = SVD(reg_all=0)
algo.fit(train_set)

# validate the model
preds = []
print(test_df)
for _, (uid, iid, rating) in test_df.iterrows():
    preds.append(algo.predict(uid, iid).est - rating)
print(f"error: {sum(preds) / len(test_df.index)}")

# check results
U = algo.pu
print(f"U shape: {U.shape}")
V = algo.qi
print(f"V shape: {V.shape}")

# make sure that V's columns match the movies in movies.csv
zero_cnt = 0
for iid in movies['Movie ID']:
    # for some reason unique has to be added here...
    if iid not in train_df['Movie ID'].unique():
        V = np.insert(V, iid, values=0, axis=0)
        zero_cnt += 1
print(f"inserted zero rows: {zero_cnt}")
V = V.T # maintain consistency with project description
print(f"V shape after zeroes: {V.shape}")

# perform SVD on the movie (V) matrix
A, S, B = np.linalg.svd(V)
A_thin = A.T[0:2]
print(f"A thin shape: {A_thin.shape}")
print((A_thin @ V).shape)     # check that the dimensions is correct
# project V onto the two most important dimensions
V_tilde = (A_thin @ V)              

def get_name_from_iid(iid: int):
    return movies['Movie Title'][iid-1]

popular_iid = [50, 258, 100, 181, 294, 286, 288, 1, 300, 121]
best_iid = [814, 1599, 1201, 1122, 1653, 1293, 1500, 1189, 1536, 1467]
random_iid = [i for i in range(1, 11)]


def produce_plot(iids: list, name: str):
    for iid in iids:
        x = V_tilde[0][iid - 1]
        y = V_tilde[1][iid - 1]
        plt.text(x, y, f"{movies['Movie Title'][iid-1]}", fontsize=7)
        plt.scatter(x, y, c="black", s=6)

    plt.autoscale(enable=True)
    plt.savefig(f'figs/task4-{name}.png')
    plt.cla()

produce_plot(popular_iid, 'popular')
produce_plot(best_iid, 'best')
produce_plot(random_iid, 'random')

# three genres: comedy, action, and documentary
action_iid = []
comedy_iid = []
documentary_iid = []