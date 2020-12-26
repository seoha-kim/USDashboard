from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from Bert.model import Model
from tokenizers import BertWordPieceTokenizer

class SentimentAnalyzer:
    def adjust(self, df_line):
        if 'funn' in df_line['sentence'].lower() or 'wtf' in df_line['sentence'].lower():
          df_line['neg'] = df_line['neg']+0.7
          df_line['pos'] = df_line['pos']*0.5
          sum = df_line['neg'] + df_line['pos']
          df_line['neg'] /= sum
          df_line['pos'] /= sum
          labels = ["Negative", "Positive"]
          sentiment = np.argmax([df_line['neg'], df_line['pos']])
          df_line['sentiment'] = labels[sentiment]
          return df_line

    def sentiment_predict(self, sentence, tokenizer):
        Model_ = Model()
        model = load_model('nlp_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        to_predict = Model_.convert_sentences_to_features(sentence, tokenizer)
        pred = model.predict(to_predict)

        result_df = pd.DataFrame(columns=['sentence', 'neg', 'pos', 'sentiment'])
        result_df['sentence'] = sentence
        result_df['neg'] = pred[:,0]
        result_df['pos'] = pred[:,1]
        result_df = result_df.apply(self.adjust, axis = 1)
        return result_df, model

    def get_top(df, top_count):
        neg = df[df['sentiment'] == "Negative"].sort_values(['neg'], ascending=False)[:top_count]
        pos = df[df['sentiment'] == "Positive"].sort_values(['pos'], ascending=False)[:top_count]
        neg = neg.append(pos)
        return neg

if __name__ == "__main__":
    sentimanet_analyzer = SentimentAnalyzer(); model_def = Model()
    model = load_model('nlp_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

    # let's prepare youtube data
    NEWS = ['nbc', 'cnn', 'fox']
    files = []
    for news in NEWS:
        files.append('../data/{}.csv'.format(news))
    comments = \
    pd.concat([pd.read_csv(f, encoding='latin_1') if 'nbc' in f else pd.read_csv(f) for f in files], ignore_index=True)[
        'comment']  # first

    keywords = {'trump': ['trump', 'donald', 'orange'], 'biden': ['joe', 'biden', 'binden', 'jeff dunhams', 'vice']}
    tcount, bcount = 0, 0
    c_type = []
    for c in comments:
        c = c.lower()
        for tk in keywords['trump']:
            tcount += c.count(tk)
        for bk in keywords['biden']:
            bcount += c.count(bk)
        if tcount > bcount:
            c_type.append('T')
        elif tcount < bcount:
            c_type.append('B')
        else:
            c_type.append('etc')
        tcount, bcount = 0, 0

    to_pred = []
    for c in comments:
        to_pred.append(model_def.preprocess_text(c))

    ytb_result = pd.DataFrame(columns=['sentence', 'neg', 'pos', 'sentiment'])
    ytb_result = model_def.sentiment_predict(to_pred, tokenizer, model, ytb_result)
    ytb_result['category'] = c_type

    t_result = ytb_result[ytb_result['category'] == "T"]
    b_result = ytb_result[ytb_result['category'] == "B"]
    etc_result = ytb_result[ytb_result['category'] == "etc"]

    COUNT = 200
    get_top = sentimanet_analyzer.get_top(COUNT)
    t_result = get_top(t_result, COUNT)
    T_neg, T_pos = t_result[:COUNT], t_result[COUNT:]
    b_result = get_top(b_result, COUNT)
    B_neg, B_pos = b_result[:COUNT], b_result[COUNT:]
    etc_result = get_top(etc_result, COUNT)
    E_neg, E_pos = etc_result[:COUNT], etc_result[COUNT:]

    T_neg.to_csv("sent_data/Trump_Neg.csv", index=False)
    B_neg.to_csv("sent_data/Biden_Neg.csv", index=False)
    E_neg.to_csv("sent_data/etc_Neg.csv", index=False)
    T_pos.to_csv("sent_data/Trump_Pos.csv", index=False)
    B_pos.to_csv("sent_data/Biden_Pos.csv", index=False)
    E_pos.to_csv("sent_data/etc_Pos.csv", index=False)