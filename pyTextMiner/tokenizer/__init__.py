''' str => list(str) '''

class BaseTokenizer:
    IN_TYPE = [str]
    OUT_TYPE = [list, str]

# [English]
class Tweet(BaseTokenizer):
    def __init__(self):
        import nltk.tokenize
        self.inst = nltk.tokenize.TweetTokenizer()

    def __call__(self, *args, **kwargs):
        return self.inst.tokenize(*args)

class Whitespace(BaseTokenizer):
    def __init__(self):
        import nltk.tokenize
        self.inst = nltk.tokenize.WhitespaceTokenizer()

    def __call__(self, *args, **kwargs):
        return self.inst.tokenize(*args)

class Word(BaseTokenizer):
    def __init__(self):
        import nltk.tokenize
        self.inst = nltk.tokenize.word_tokenize

    def __call__(self, *args, **kwargs):
        return self.inst(*args)

# [Korean]
class Komoran(BaseTokenizer):
    def __init__(self):
        from konlpy.tag import Komoran
        self.inst = Komoran()
        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        return self.inst.pos(args[0])

class TwitterKorean(BaseTokenizer):
    def __init__(self):
        from konlpy.tag import Twitter
        self.inst = Twitter()
        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        return self.inst.pos(args[0])
