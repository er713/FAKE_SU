from typing import List, Tuple, Iterable
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import matplotlib
from wordcloud import WordCloud


def clean(text: str) -> str:
    t = text.encode('ascii', 'ignore').decode()
    t = re.sub(r'\n', ' ', t)
    t = re.sub(r'\W', ' ', t)
    t = re.sub(r'[0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t, flags=re.I)
    return t


def bayes_ngrams(tab: np.ndarray, labels: pd.DataFrame) -> (np.ndarray, np.ndarray):
    l = labels.to_numpy()
    p_size = np.sum(l)
    size = l.shape[0]
    # n_size = size - p_size
    P_p = p_size / size
    # P_n = n_size / size
    p_mask = labels == 1
    n_mask = np.logical_not(p_mask)
    p_count = np.sum(tab[p_mask], axis=0)
    n_count = np.sum(tab[n_mask], axis=0)
    count = n_count + p_count
    P_ng_p = p_count.astype(np.float_) / p_size
    # P_ng_n = n_count.astype(np.float_) / n_size
    P_ng = count.astype(np.float_) / size
    P_p_ng = np.divide(np.multiply(P_ng_p, P_p), P_ng)
    return P_p_ng, count


def get_grams_with_thresh(names: np.ndarray, probability: np.ndarray, threshold=0.8):
    return dict(zip(names[probability > threshold], np.round(probability[probability > threshold], 4)))


def create_word_cloud(words):
    wc = WordCloud(background_color='white', scale=3, collocations=False, relative_scaling=1,
                   width=4000, height=3000, colormap='tab20c')
    plt.figure(figsize=(40, 30))
    img = wc.generate_from_frequencies(words)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def create_bar_chart(sorted_, threshold=0.8, max_items=30, label_size=20):
    matplotlib.rc('ytick', labelsize=label_size)
    matplotlib.rc('xtick', labelsize=label_size * 3 // 4)
    matplotlib.rc('axes', labelsize=label_size * 3 // 4)
    sorted_words = sorted_[:max_items]
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.5, right=0.8, bottom=0.1, top=0.9)
    width = 0.01
    space = 2.5 / 2
    ax.barh(np.arange(len(sorted_words))[::-1] * (width * space), [s[1] for s in sorted_words], align='center',
            height=width)
    # plt.axis('off')
    ax.set_yticks(np.arange(len(sorted_words))[::-1] * (width * space))
    ax.set_yticklabels([s[0] for s in sorted_words])
    ax.set_xlim([threshold, 1.])
    # ax.xlim()
    ax.set_xlabel('Prawdopodobieństwo')
    ax.set_xticks([threshold, 1.])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv', index_col='id')
    tit_lab = data[['title', 'label']]
    tit_lab_nan = tit_lab[tit_lab['title'].isna()]
    tit_lab_clean = tit_lab[tit_lab['title'].notna()]
    print(tit_lab.shape, tit_lab_nan.shape, tit_lab_clean.shape)

    y_clean = tit_lab_clean['label'].to_numpy()
    y_nan = tit_lab_nan['label'].to_numpy()
    print('=====================')
    print('Średnia fakeow przy nieznanym tytule', np.mean(y_nan))
    print('Średnia fakeow przy znanym tytule', np.mean(y_clean))
    print('=====================')
    print(y_nan.shape[0] - np.sum(y_nan))

    vec = CountVectorizer(strip_accents=None, lowercase=True, ngram_range=(3, 3), min_df=15, stop_words=None)
    counts = vec.fit_transform(tit_lab_clean['title'].map(clean).to_numpy())
    tab_count = np.array(counts.toarray(), dtype=np.int_)
    # print(tab_count[:5])
    prob, quantity = bayes_ngrams(tab_count, tit_lab_clean['label'])
    names = np.array(vec.get_feature_names())
    su = np.sum(tab_count, axis=0)
    print(names, np.max(su), np.min(su),
          np.mean(su), np.median(su))
    # print(prob, np.sum(prob > 0.99), prob.shape[0])
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
