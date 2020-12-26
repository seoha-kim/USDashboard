import pandas as pd
import tensorflow as tf
import os
import shutil

class DataPrep:
    def __init__(self):
        URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",
                                          origin=URL,
                                          untar=True,
                                          cache_dir='..',
                                          cache_subdir='')

        main_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
        train_dir = os.path.join(main_dir, 'train')
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

        train = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/train', batch_size=30000, seed=123)

        for i in train.take(1):
          train_feat = i[0].numpy()
          train_lab = i[1].numpy()

        train = pd.DataFrame([train_feat, train_lab]).T
        train.columns = ['review', 'sentiment']
        train['review'] = train['review'].str.decode("utf-8")
        train.to_csv('../IMDB Dataset.csv', index=False)