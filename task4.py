# Apparently surprise only works on python<=3.7, and installing ipykernel on this version of python was taking forever
import numpy as np
import pandas as pd
from surprise import Dataset, SVD, Reader
import matplotlib.pyplot as plt

# Load the movies csv, which is used in multiple places below. This is read-only!
MOVIES = pd.read_csv('movies.csv')
print(len(MOVIES['Movie ID']))
scores = {}
for iid in MOVIES['Movie ID']:
    scores.update({iid: })

# Define a reader for Surprise
reader = Reader()

# Read the csv files
train_df = pd.read_csv('data.csv')
test_df = pd.read_csv('test.csv')
train_df = train_df[['User ID', 'Movie ID', 'Rating']]
print(train_df.describe())
# create a training set
train_set = Dataset.load_from_df(train_df, reader).build_full_trainset()

# Initialize and train SVD algorithm
algo = SVD(lr_all=0.03, reg_all=0.1)
algo.fit(train_set)

# Validate the model using mean error - sanity check
# errors = []
# print(test_df)
# for _, (uid, iid, rating) in test_df.iterrows():
#     errors.append(algo.predict(uid, iid).est - rating)
# print(f"error: {sum(errors) / len(test_df.index)}")

# Check factorization results
U = algo.pu
print(f"U shape: {U.shape}")
V = algo.qi
print(f"V shape: {V.shape}")

V = V.T # maintain consistency with project description
print(f"V shape (new): {V.shape}")

# perform SVD on the movie (V) matrix
A, S, B = np.linalg.svd(V)
print(A.shape)
A_thin = (A[:, :2]).T
print(f"A thin shape: {A_thin.shape}")
print(f"project shape: {(A_thin @ V).shape}")     # check that the dimensions is correct
V_tilde = np.matmul(A_thin, V)        # project V onto the two most important dimensions

# props to task 1 and 2!
popular_iid = [50, 258, 100, 181, 294, 286, 288, 1, 300, 121]
best_iid = [814, 1599, 1201, 1122, 1653, 1293, 1500, 1189, 1536, 1467]

def produce_plot(iids: list, name: str):
    for iid in iids:
        x, y = V_tilde[:, iid-1]
        plt.annotate(f"{MOVIES['Movie Title'][iid-1]}", (x, y), fontsize=7)
        plt.scatter(x, y, c="tab:blue", s=6)
    plt.axvline(x=0, linestyle='--')
    plt.axhline(y=0, linestyle='--')
    plt.autoscale(enable=True)
    plt.savefig(f'figs/task4-{name}.png', dpi=720)
    plt.cla()

produce_plot(popular_iid, 'popular')
produce_plot(best_iid, 'best')

# just chose the first ten movies
random_iid = [i for i in range(1, 11)]
produce_plot(random_iid, 'random')

def first_movies_of_genre(genre: str, lim: int = 10):
    iids = []
    for _, row in MOVIES.iterrows():
        if row[genre] == 1:
            iids.append(row['Movie ID'])
    counts = train_df['Movie ID'].value_counts()
    iids.sort(key=lambda id: counts.get(id), reverse=True)
    return iids

# three genres: comedy, action, and documentary
genres = ['Action', 'Comedy', 'Documentary']
for genre in genres:
    produce_plot(first_movies_of_genre(genre), genre.lower())


produce_plot([82, 143, 161, 231, 257, 272, 288, 403, 318, 1065], 'chosen')