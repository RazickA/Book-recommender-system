import json
import pandas as pd

from tqdm import tqdm


def load_book_graph(_path):
    """
    Processes book metadata.

    :param _path: The path to the metadata file.
    :return: _meta, A dataframe of book metadata.
    """
    bid = []
    title = []
    lnk = []
    with open(_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            bid.append(data['book_id'])
            title.append(data['title'])
            lnk.append(data['link'])
    _meta = pd.DataFrame({'book_id': bid,
                          'title': title,
                          'link': lnk})
    return _meta


if __name__ == "__main__":
    meta = load_book_graph('goodreads_books.json')
    meta.to_csv('book_metadata.csv', index=False)
