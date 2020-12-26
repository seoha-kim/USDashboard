''' tuple => tuple '''

class BaseTagger:
    IN_TYPE = [tuple]
    OUT_TYPE = [tuple]

class WordNet(BaseTagger):
    def __init__(self):
        import nltk
        nltk.download('wordnet')

        from nltk.stem import WordNetLemmatizer
        self.inst = WordNetLemmatizer()

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        from nltk.corpus import wordnet
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def __call__(self, *args, **kwargs):
        tag = WordNet.get_wordnet_pos(args[0][1])
        return self.inst.lemmatize(args[0][0], tag if tag else 'n'), args[0][1]
