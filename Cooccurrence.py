import pyTextMiner as ptm

if __name__ == '__main__':
    corpus = ptm.CorpusFromFieldDelimitedFile('data/nbc.txt', 2)
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.counter.WordCounter(),
                            ptm.tokenizer.Whitespace(),
                            ptm.helper.StopwordFilter(file='Stopwords.txt'),
                            ptm.tagger.NLTK(),
                            ptm.lemmatizer.WordNet(),
                            ptm.helper.POSFilter('N*', 'J*'),
                            ptm.helper.SelectWordOnly())

    result = pipeline.processCorpus(corpus)

    co_occurrence, vocabulary = ptm.cooccurrence.CooccurrenceManager().computeCooccurence(result)
    print(co_occurrence)

    ptm.graphml.GraphMLCreator().createGraphML(co_occurrence, vocabulary, "data/nbc.graphml")
