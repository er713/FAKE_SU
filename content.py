import numpy as np
from themes import clean, create_word_cloud, create_bar_chart, get_grams_with_thresh, bayes_ngrams
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def not_str(x):
    if not isinstance(x, str):
        return -1
    else:
        return len(x)


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv', index_col='id')
    con_lab = data[['text', 'label']]
    print('Artykuły bez zawartości:', con_lab[con_lab['text'].isna()].shape)
    # print(data[con_lab['text'].isna()]['title'])
    con_lab = con_lab[con_lab['text'].notna()]

    len_lab = con_lab.copy()
    len_lab['text'] = con_lab['text'].map(len)
    len_lab = len_lab.sort_values('text')
    # print(len_lab.tail())
    print(len_lab.shape)
    print("Korelacja\t\t\t", len_lab['text'].corr(len_lab['label']))
    print("Korelacja spearmana\t", len_lab['text'].corr(len_lab['label'], method='spearman'))

    con_lab['text'] = con_lab['text'].map(clean)
    len_lab['text'] = con_lab['text'].map(len)
    len_lab = len_lab[len_lab['text'] > 10]
    print("=====Po Clean=======")
    print("Korelacja\t\t\t", len_lab['text'].corr(len_lab['label']))
    print("Korelacja spearmana\t", len_lab['text'].corr(len_lab['label'], method='spearman'))
    print("Korelacja kendalla\t", len_lab['text'].corr(len_lab['label'], method='kendall'))
    # len_lab['text'].hist(bins=[0, 100, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 130000])
    print(len_lab['text'].describe())
    # plt.show()

    # word cloud
    vec = CountVectorizer(strip_accents=None, lowercase=True, ngram_range=(1, 1), min_df=150, stop_words='english')
    count = vec.fit_transform(con_lab['text'].to_numpy())
    tab_count = np.array(count.toarray(), dtype=np.int_)

    prob, quantity = bayes_ngrams(tab_count, con_lab['label'])
    names = np.array(vec.get_feature_names_out())
    su = np.sum(tab_count, axis=0)
    print(names, np.max(su), np.min(su), np.mean(su), np.median(su))
    print(quantity[prob > 0.99], np.mean(quantity[prob > 0.99]), np.median(quantity[prob > 0.99]))
    print(names[prob > 0.99])

    th = 0.8
    w = get_grams_with_thresh(names, prob, threshold=th)
    print(w)
    print(len(w))
    ss = sorted(w.items(), key=lambda x: x[1], reverse=True)
    print(ss)
    create_word_cloud(w)
    # create_bar_chart(ss, max_items=11, threshold=0.85, label_size=20)
