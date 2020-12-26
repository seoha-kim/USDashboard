''' str => list(str) '''

class BaseSplitter:
    IN_TYPE = [str]
    OUT_TYPE = [list, str]

class NLTK(BaseSplitter):
    def __init__(self):
        import nltk.tokenize
        self.func = nltk.tokenize.sent_tokenize

    def __call__(self, *args, **kwargs):
        return self.func(*args)