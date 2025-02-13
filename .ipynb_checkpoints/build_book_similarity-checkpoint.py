"""
Script to build book recommendation systems.
"""
import json
import pickle
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def read_raw_data(_num_samples, _fn):
    """
    Reads and processes the raw Goodreads data.
    :param _num_samples: The number of rows to sample from the dataframe.
    :param _fn: Formatted filename of output.
    :return: None, saves the output dataframe.
    """
    _df = pd.read_csv("goodreads_interactions.csv", nrows=_num_samples)
    _df = _df[_df.is_read == 1]
    _df = _df[0:_num_samples]
    _df.to_csv('goodreads_{}.csv'.format(_fn, index=False))


def build_rating_matrix(_df):
    """
    Converts a dataframe to a user-item interaction matrix.
    :param _df: The input dataframe.
    :return: Numpy matrix representing user-interaction ratings.
    """
    _n_users = len(_df.user_id.unique()) + 1  # python indices start at zero, user_ids start at 1
    _n_books = _df.book_idx.max() + 1  # python indices start at zero, book_ids start at 1
    print('Users: {}'.format(_n_users))
    print('Books: {}'.format(_n_books))
    _ratings = np.zeros((_n_users, _n_books))
    for _, row in tqdm(_df.iterrows()):
        i = row.user_id
        j = row.book_idx
        _ratings[i, j] = row.rating
    return _ratings


def recommend_item_similarity(_matrix, _eps, _n_latent):
    """
    Builds item similarities using truncated SVD.
    :param _matrix: The user-item rating matrix.
    :param _eps: The epsilon parameter for truncated SVD.
    :param _n_latent: The number of latent features for truncated SVD.
    :return: _sparse_features, The sparse matrix of item-similarity features.
    """
    _item_svd = TruncatedSVD(n_components=_n_latent)
    _item_features = _item_svd.fit_transform(_matrix.transpose())
    print('Converting to sparse')
    _sparse_features = sparse.csr_matrix(_item_features)
    return _sparse_features


def generate_similarity_matrix(_features, _metric):
    """
    Generates the similarity matrix from either item or user features
    based on the given similarity metric.
    :param _features: The matrix of user or item features.
    :param _metric: A string indicating which similarity metric should be used.
    :return: _similarity_matrix, The final similarity matrix.
    """
    assert _metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    print('Computing similarity')
    _similarity_matrix = pairwise_distances(_features, metric=_metric)
    return _similarity_matrix


def merge_meta(_meta_path, _map_path, _ratings):
    """
    Merges book metadata with ratings.

    :param _meta_path: Path to book metadata csv.
    :param _map_path: Path to book ID mapping.
    :param _ratings: Dataframe of rating interactions.
    :return: _ratings_meta, a dataframe of metadata and ratings and
    _metadata_lookup, dictionary for the UI.
    """
    _meta = pd.read_csv(_meta_path)
    _map = pd.read_csv(_map_path)
    _ratings_map = _ratings.merge(_map, how='left',
                                  left_on='book_id', right_on='book_id_csv')
    _ratings_map = _ratings_map[['user_id', 'book_id_csv', 'is_read',
                                 'rating', 'is_reviewed', 'book_id_y']]
    _ratings_map.columns = ['user_id', 'book_idx', 'is_read',
                            'rating', 'is_reviewed', 'book_id']
    _metadata_lookup = {}
    for _, row in _ratings_map.iterrows():
        _md = _meta[_meta['book_id'] == row['book_id']]
        _metadata_lookup[str(row.book_idx)] = {
            'title': _md['title'].values[0],
            'link': _md['link'].values[0]}
    return _ratings_map, _metadata_lookup


if __name__ == "__main__":
    NS = 8000
    FN = '8k'
    EPS = 1e-9
    FACTORS = 7
    METRIC = 'cityblock'
    try:
        goodreads = pd.read_csv('goodreads_{}.csv'.format(FN))
    except FileNotFoundError:
        read_raw_data(NS, FN)
        goodreads = pd.read_csv('goodreads_{}.csv'.format(FN))
    ratings_meta, metadata_lookup = merge_meta(
        'book_metadata.csv',
        'book_id_map.csv', goodreads)
    print('Saving metadata')
    with open('books_metadata_{records}.json'.format(records=FN), 'w', encoding='utf-8') as m:
        json.dump(metadata_lookup, m)
    ratings = build_rating_matrix(ratings_meta)
    item_features = recommend_item_similarity(ratings, EPS, FACTORS)
    sim = generate_similarity_matrix(item_features, METRIC)
    print('Saving similarity')
    with open('book_similarity_{factors}_{records}_{metric}.pkl'.format(factors=FACTORS,
                                                                        records=FN,
                                                                        metric=METRIC), 'wb') as f:
        pickle.dump(sim, f)
