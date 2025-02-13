"""
Script to test recommender systems.
"""
import json
import pickle


def test_recommender(_search, _similarity, _metadata):
    """
    A function to test our recommender system.

    :param _search: A book ID to search for.
    :param _similarity: Our recommender similarity matrix.
    :param _metadata: Mapping of book ID to title.
    :return: List of titles of top 5 most similar books.
    """
    row_sims = _similarity[_search, ]
    res = sorted(range(len(row_sims)), key=lambda sub: row_sims[sub])[-5:]
    print('Searched for book: {sb}'.format(sb=_metadata[str(_search)]['title']))
    for j, _ in enumerate(res):
        print('Match {idx}: {book}'.format(idx=j, book=_metadata[str(res[j])]['title']))


if __name__ == "__main__":
    #  Make sure these are updated to match your models from 'build_book_similarity.py'
    NS = 8000
    FN = '8k'
    EPS = 1e-9
    FACTORS = 7
    METRIC = 'cityblock'

    SIM_PATH = 'book_similarity_{factors}_{records}_{metric}.pkl'.format(factors=FACTORS,
                                                                         records=FN,
                                                                         metric=METRIC)
    META_PATH = 'books_metadata_{records}.json'.format(records=FN)
    with open(SIM_PATH, 'rb') as f:
        sim = pickle.load(f)
    print('Loaded similarity')
    with open(META_PATH, 'r', encoding='utf-8') as m:
        metadata_lookup = json.load(m)
    print('Loaded metadata')
    #  Try different indices for the first parameter of the following function
    test_recommender(948, sim, metadata_lookup)
