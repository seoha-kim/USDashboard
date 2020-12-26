import string
from collections import Counter
import os
from nltk import bigrams
from collections import defaultdict
import operator

class CooccurrenceManager:
    def computeCooccurence(self, list):
        com = defaultdict(lambda: defaultdict(int))
        count_all = Counter()
        count_all1 = Counter()

        uniqueList = []
        for _array in list:
            for line in _array:
                for word in line:
                    if word not in uniqueList:
                        uniqueList.append(word)

                terms_bigram = bigrams(line)
                # Update the counter
                count_all.update(line)
                count_all1.update(terms_bigram)

                # Build co-occurrence matrix
                for i in range(len(line) - 1):
                    for j in range(i + 1, len(line)):
                        w1, w2 = sorted([line[i], line[j]])
                        if w1 != w2:
                            com[w1][w2] += 1

        com_max = []
        # For each term, look for the most common co-occurrent terms
        for t1 in com:
            t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
            for t2, t2_count in t1_max_terms:
                com_max.append(((t1, t2), t2_count))
        # Get the most frequent co-occurrences
        terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)

        return terms_max, uniqueList

