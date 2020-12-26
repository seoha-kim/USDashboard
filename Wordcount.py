import pyTextMiner as ptm
corpus = ptm.CorpusFromFieldDelimitedFile('data/cnn.txt', 1)

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.helper.ToLowerCase(),
                        ptm.tokenizer.Whitespace(),
                        ptm.helper.StopwordFilter(file='Stopwords.txt'),
                        ptm.tagger.NLTK(),
                        ptm.lemmatizer.WordNet(),
                        ptm.helper.POSFilter('N*', 'J*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.counter.WordCounter())

result = pipeline.processCorpus(corpus)
print(result)
