import numpy as np
import pandas as pd
from themes import clean, bayes_ngrams
from authors import divide_author
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def vectorization(train, name, n, ret=4, min_df=15, vec=None):
    if vec is None:
        vec = CountVectorizer(strip_accents=None, lowercase=True, ngram_range=(n, n), min_df=min_df,
                              stop_words='english')
        count = np.array(vec.fit_transform(train[name].to_numpy()).toarray(), dtype=np.int_)
    else:
        count = np.array(vec.transform(train[name].to_numpy()).toarray(), dtype=np.int_)
    p1, _ = bayes_ngrams(count, train['label'])
    name1 = np.array(vec.get_feature_names_out())
    v1_max = np.zeros((count.shape[0], ret))
    tmp = np.array([np.sort(p1[c > 0])[-ret // 2:] for c in count])
    for i, t in enumerate(tmp):
        for j, v in enumerate(t):
            v1_max[i, j] = v
    tmp = np.array([np.sort(p1[c > 0])[:ret // 2] for c in count])
    for i, t in enumerate(tmp):
        for j, v in enumerate(t):
            v1_max[i, j + 2] = v
    return v1_max, p1, name1, vec


def prepare_data(train, test=None, authors_power=None, vec=None):
    if vec is None:
        vec = [None, None, None, None]
    print("==== autorzy")
    aut = train[train['author'].notna()]['author']
    aut = [divide_author(a) for a in aut]

    authors_for_art = np.zeros(shape=train['author'].shape, dtype=np.int_)  # liczba autorów dla artykułu
    authors_for_art[train['author'].notna()] = np.array([len(a) for a in aut])

    if authors_power is None:
        authors_power = dict()
        for tab_a in aut:  # zliczanie artykułów autora
            for a in tab_a:
                authors_power[a] = 1 + authors_power.get(a, 0)

    ap_for_art = np.zeros_like(authors_for_art)  # przypisywanie najbardziej doswiadczonego autora do artykulu
    ap_for_art[train['author'].notna()] = np.array([np.max([authors_power.get(a, 0) for a in tab_a]) for tab_a in aut])

    print("==== tytuly")
    # Tytuł
    # czy go nie ma
    title_exist = np.ones_like(train['title'])
    title_exist[train['title'].notna()] = 0
    # po dwa najważniejsze unigramy i trigramy
    train['title'] = train['title'].map(clean)
    v1_max, p1, n1, vec[0] = vectorization(train, 'title', 1, vec=vec[0])
    v3_max, p3, n3, vec[1] = vectorization(train, 'title', 3, vec=vec[1])

    print("==== tekst")
    # Tekst
    # długość
    train['text'] = train['text'].map(clean)
    text_len = train['text'].map(len).to_numpy()
    # po 4 unigramy i trigramy
    tv1_max, tp1, tn1, vec[2] = vectorization(train, 'text', 1, 8, 150, vec[2])
    tv3_max, tp3, tn3, vec[3] = vectorization(train, 'text', 3, 8, 150, vec[3])

    print("==== laczenie")
    st = np.stack([authors_for_art, ap_for_art, title_exist, text_len], axis=1)
    tt = pd.DataFrame(
        np.concatenate([st, v1_max, v3_max, tv1_max, tv3_max], axis=1),
        columns=['#autors', 'experience', 'not_title', 'text_len', 'v11_max', 'v12_max', 'v11_min', 'v12_min',
                 'v31_max', 'v32_max', 'v31_min', 'v32_min', 'tv11_max', 'tv12_max', 'tv13_max', 'tv14_max',
                 'tv11_min', 'tv12_min', 'tv13_min', 'tv14_min', 'tv31_max', 'tv32_max', 'tv33_max', 'tv34_max',
                 'tv31_min', 'tv32_min', 'tv33_min', 'tv34_min'])
    prepared = pd.concat([train['label'], tt], axis=1)
    print(np.sum(train['label'].isna()), train['label'].shape)
    print(np.sum(prepared['label'].isna()), prepared['label'].shape)

    p_test = None
    if test is not None:
        p_test, _ = prepare_data(test, None, authors_power, vec)

    return prepared, p_test


def raw2prep():
    table = pd.read_csv("data/train.csv")
    print(np.sum(table['label'].isna()))
    table = table.sample(frac=1).reset_index(drop=False)
    train = table[:int(table.shape[0] * 0.7)]
    test = table[int(table.shape[0] * 0.7):]
    print(np.sum(train['label'].isna()))
    print(np.sum(test['label'].isna()))
    t1, t2 = prepare_data(train, test)
    print(np.sum(t2[:t2.shape[0]//2]['label'].isna()))
    t1.to_csv("data/prepared_train.csv")
    t2.to_csv("data/prepared_test.csv")


if __name__ == "__main__":
    # raw2prep()
    t = pd.read_csv("data/prepared_test.csv")
    print(np.sum(t[t.shape[0]//2:]['label'].isna()))
    print(t[t.shape[0]//2:].to_numpy)

    # table = pd.read_csv("data/train.csv")
    # print(np.sum(table['label'].isna()))
