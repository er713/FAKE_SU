import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy

if __name__ == "__main__":
    data = pd.read_csv('test.csv', )
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # nlp.add_pipe("lemmatizer")

    tokens = nlp("This is a horrible sentence.")
    for t in tokens:
        print(len(t.vector))
        print(t.text, t.lemma_, t.sentiment)
