import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def make_cloud(filename, masking, cmap):
    txt = list(pd.read_csv(filename)['sentence'])
    txt = ' '.join(txt)
    txt = txt.lower()
    clean_txt = [word for word in txt.split() if word not in stop]
    clean_txt = ' '.join([str(elem) for elem in clean_txt])
    wc = WordCloud(background_color=None,
                   mode="RGBA",
                   max_font_size=40,
                   mask = masking,
                   colormap = cmap
                  ).generate(clean_txt)
    return wc

if __name__ == "main":
    stop = set(line.strip() for line in open('../Stopwords.txt', encoding='utf-8'))
    biden_coloring = np.array(Image.open('biden.png'))
    trump_coloring = np.array(Image.open('trump.png'))

    neg_cmap = plt.cm.Oranges
    pos_cmap = plt.cm.Blues

    wc = make_cloud('Trump_Neg.csv', trump_coloring, neg_cmap)
    plt.figure(figsize=(25,15))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()