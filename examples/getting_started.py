from collections import defaultdict
import csv
import numpy
import random
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

import tensorrec

import logging
logging.getLogger().setLevel(logging.INFO)

# Open and read in the ratings file
# NOTE: This expects the ratings.csv file to be in the same folder as this Python file
print('Loading ratings')
with open('ratings.csv', 'r') as ratings_file:
    ratings_file_reader = csv.reader(ratings_file)
    raw_ratings = list(ratings_file_reader)
    raw_ratings_header = raw_ratings.pop(0)

# Iterate through the input to map MovieLens IDs to new internal IDs
# The new internal IDs will be created by the defaultdict on insertion
movielens_to_internal_user_ids = defaultdict(lambda: len(movielens_to_internal_user_ids))
movielens_to_internal_item_ids = defaultdict(lambda: len(movielens_to_internal_item_ids))
for row in raw_ratings:
    row[0] = movielens_to_internal_user_ids[int(row[0])]
    row[1] = movielens_to_internal_item_ids[int(row[1])]
    row[2] = float(row[2])
n_users = len(movielens_to_internal_user_ids)
n_items = len(movielens_to_internal_item_ids)

# Look at an example raw rating
print("Raw ratings example:\n{}\n{}".format(raw_ratings_header, raw_ratings[0]))

# Shuffle the ratings and split them in to train/test sets 80%/20%
random.shuffle(raw_ratings)  # Shuffles the list in-place
cutoff = int(.8 * len(raw_ratings))
train_ratings = raw_ratings[:cutoff]
test_ratings = raw_ratings[cutoff:]
print("{} train ratings, {} test ratings".format(len(train_ratings), len(test_ratings)))


# This method converts a list of (user, item, rating, time) to a sparse matrix
def interactions_list_to_sparse_matrix(interactions):
    users_column, items_column, ratings_column, _ = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)),
                             shape=(n_users, n_items))


# Create sparse matrices of interaction data
sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings)
sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings)

# Construct indicator features for users and items
user_indicator_features = sparse.identity(n_users)
item_indicator_features = sparse.identity(n_items)

# Build a matrix factorization collaborative filter model
cf_model = tensorrec.TensorRec(n_components=5)

# Fit the collaborative filter model
print("Training collaborative filter")
cf_model.fit(interactions=sparse_train_ratings,
             user_features=user_indicator_features,
             item_features=item_indicator_features)

# Create sets of train/test interactions that are only ratings >= 4.0
sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 4.0)
sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 4.0)


# This method predicts item ranks for each user and prints out recall@10 train/test metrics
def check_results(ranks):
    train_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_train_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_test_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    print("Recall at 10: Train: {:.4f} Test: {:.4f}".format(train_recall_at_10,
                                                            test_recall_at_10))


# Check the results of the MF CF model
print("Matrix factorization collaborative filter:")
predicted_ranks = cf_model.predict_rank(user_features=user_indicator_features,
                                        item_features=item_indicator_features)
check_results(predicted_ranks)

# Let's try a new loss function: WMRB
print("Training collaborative filter with WMRB loss")
ranking_cf_model = tensorrec.TensorRec(n_components=5,
                                       loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
ranking_cf_model.fit(interactions=sparse_train_ratings_4plus,
                     user_features=user_indicator_features,
                     item_features=item_indicator_features,
                     n_sampled_items=int(n_items * .01))

# Check the results of the WMRB MF CF model
print("WMRB matrix factorization collaborative filter:")
predicted_ranks = ranking_cf_model.predict_rank(user_features=user_indicator_features,
                                                item_features=item_indicator_features)
check_results(predicted_ranks)

# To improve the recommendations, lets read in the movie genres
print('Loading movie metadata')
with open('movies.csv', 'r') as movies_file:
    movies_file_reader = csv.reader(movies_file)
    raw_movie_metadata = list(movies_file_reader)
    raw_movie_metadata_header = raw_movie_metadata.pop(0)

# Map the MovieLens IDs to our internal IDs and keep track of the genres and titles
movie_genres_by_internal_id = {}
movie_titles_by_internal_id = {}
for row in raw_movie_metadata:
    row[0] = movielens_to_internal_item_ids[int(row[0])]  # Map to IDs
    row[2] = row[2].split('|')  # Split up the genres
    movie_genres_by_internal_id[row[0]] = row[2]
    movie_titles_by_internal_id[row[0]] = row[1]

# Look at an example movie metadata row
print("Raw metadata example:\n{}\n{}".format(raw_movie_metadata_header,
                                             raw_movie_metadata[0]))

# Build a list of genres where the index is the internal movie ID and
# the value is a list of [Genre, Genre, ...]
movie_genres = [movie_genres_by_internal_id[internal_id]
                for internal_id in range(n_items)]

# Transform the genres into binarized labels using scikit's MultiLabelBinarizer
movie_genre_features = MultiLabelBinarizer().fit_transform(movie_genres)
n_genres = movie_genre_features.shape[1]
print("Binarized genres example for movie {}:\n{}".format(movie_titles_by_internal_id[0],
                                                          movie_genre_features[0]))

# Coerce the movie genre features to a sparse matrix, which TensorRec expects
movie_genre_features = sparse.coo_matrix(movie_genre_features)

# Fit a content-based model using the genres as item features
print("Training content-based recommender")
content_model = tensorrec.TensorRec(
    n_components=n_genres,
    item_repr_graph=tensorrec.representation_graphs.FeaturePassThroughRepresentationGraph(),
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph()
)
content_model.fit(interactions=sparse_train_ratings_4plus,
                  user_features=user_indicator_features,
                  item_features=movie_genre_features,
                  n_sampled_items=int(n_items * .01))

# Check the results of the content-based model
print("Content-based recommender:")
predicted_ranks = content_model.predict_rank(user_features=user_indicator_features,
                                             item_features=movie_genre_features)
check_results(predicted_ranks)

# Try concatenating the genres on to the indicator features for a hybrid recommender system
full_item_features = sparse.hstack([item_indicator_features, movie_genre_features])

print("Training hybrid recommender")
hybrid_model = tensorrec.TensorRec(
    n_components=5,
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph()
)
hybrid_model.fit(interactions=sparse_train_ratings_4plus,
                 user_features=user_indicator_features,
                 item_features=full_item_features,
                 n_sampled_items=int(n_items * .01))

print("Hybrid recommender:")
predicted_ranks = hybrid_model.predict_rank(user_features=user_indicator_features,
                                            item_features=full_item_features)
check_results(predicted_ranks)

# Print out movies that User #432 has liked
print("User 432 liked:")
for m in sparse_train_ratings_4plus[432].indices:
    print(movie_titles_by_internal_id[m])

# Pull user 432's features out of the user features matrix and predict movie ranks for just that user
u432_features = sparse.csr_matrix(user_indicator_features)[432]
u432_rankings = hybrid_model.predict_rank(user_features=u432_features,
                                          item_features=full_item_features)[0]

# Get internal IDs of User 432's top 10 recommendations
# These are sorted by item ID, not by rank
# This may contain items with which User 432 has already interacted
u432_top_ten_recs = numpy.where(u432_rankings <= 10)[0]
print("User 432 recommendations:")
for m in u432_top_ten_recs:
    print(movie_titles_by_internal_id[m])

# Print out User 432's held-out interactions
print("User 432's held-out movies:")
for m in sparse_test_ratings_4plus[432].indices:
    print(movie_titles_by_internal_id[m])
