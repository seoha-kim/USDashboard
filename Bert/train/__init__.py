from ..model import Model
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


class Train:
    def import_model(self):
        Model_ = Model(MAX_SEQ_LEN=150)
        model = Model_.nlp_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2")
        movie_reviews = pd.read_csv("../IMDB Dataset.csv")
        movie_reviews.head(5)
        movie_reviews = movie_reviews.sample(frac=1)
        return Model_, model, movie_reviews

    def import_tokenizer(self):
        slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        save_path = "bert_base_uncased/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        slow_tokenizer.save_pretrained(save_path)

        # Load the fast tokenizer from saved file
        tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
        tokenizer.enable_truncation(Model.MAX_SEQ_LEN - 2)
        return slow_tokenizer, tokenizer

    def preprocessing_reviews(self, train_count = 20000, test_count = 2000):
        reviews = []
        Model_, model, movie_reviews = self.import_model()
        slow_tokenizer, tokenizer = self.import_tokenizer()
        sentences = list(movie_reviews["review"])
        for sen in sentences:
            reviews.append(model.preprocess_text(sen))
        y = np.array(movie_reviews["sentiment"])

        X_train = Model_.convert_sentences_to_features(reviews[:train_count], tokenizer)
        X_test = Model_.convert_sentences_to_features(reviews[train_count:train_count+test_count], tokenizer)
        one_hot_encoded = Model.to_categorical(y)

        Y_train = one_hot_encoded[:train_count]
        Y_test = one_hot_encoded[train_count:train_count + test_count]
        return X_train, X_test, Y_train, Y_test, model