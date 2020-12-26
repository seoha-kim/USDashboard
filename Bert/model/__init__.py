import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tqdm import tqdm
import re
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

class Model:
    def __init__(self,MAX_SEQ_LEN):
        self.MAX_SEQ_LEN = MAX_SEQ_LEN  # max sequence length

    def get_masks(self, tokens):
        """Masks: 1 for real tokens and 0 for paddings"""
        return [1] * len(tokens) + [0] * (self.MAX_SEQ_LEN - len(tokens))

    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (self.MAX_SEQ_LEN - len(tokens))

    def get_ids(self, ids):
        """Token ids from Tokenizer vocab"""
        token_ids = ids
        input_ids = token_ids + [0] * (self.MAX_SEQ_LEN - len(token_ids))
        return input_ids

    def create_single_input(self, sentence, tokenizer):
        """Create an input from a sentence"""
        encoded = tokenizer.encode(sentence)
        ids = self.get_ids(encoded.ids)
        masks = self.get_masks(encoded.tokens)
        segments = self.get_segments(encoded.tokens)
        return ids, masks, segments

    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(sen)
        # Remove punctuations and numbers
        sentence = re.sub("[^a-zA-Z0-9]", " ", sen)
        return sentence

    def remove_tags(self, text):
        TAG_RE = re.compile(r"<[^>]+>")
        return TAG_RE.sub("", text)

    def convert_sentences_to_features(self, sentences, tokenizer):
        """Convert sentences to features: input_ids, input_masks and input_segments"""
        input_ids, input_masks, input_segments = [], [], []
        for sentence in tqdm(sentences, position=0, leave=True):
            ids, masks, segments = self.create_single_input(sentence, tokenizer)
            assert len(ids) == self.MAX_SEQ_LEN
            assert len(masks) == self.MAX_SEQ_LEN
            assert len(segments) == self.MAX_SEQ_LEN
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
        return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)]

    def nlp_model(self, callable_object):
        # Load the pre-trained BERT base model
        bert_layer = hub.KerasLayer(handle=callable_object, trainable=True)

        # BERT layer three inputs: ids, masks and segments
        input_ids = Input(shape=(self.MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")
        input_masks = Input(shape=(self.MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")
        input_segments = Input(shape=(self.MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")

        inputs = [input_ids, input_masks, input_segments]  # BERT inputs
        pooled_output, sequence_output = bert_layer(inputs)  # BERT outputs

        # Add a hidden layer
        x = Dense(units=768, activation="relu")(pooled_output)
        x = Dropout(0.1)(x)

        # Add output layer
        outputs = Dense(2, activation="softmax")(x)

        # Construct a new model
        model = Model(inputs=inputs, outputs=outputs)
        return model
