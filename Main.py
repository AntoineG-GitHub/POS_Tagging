from nltk.probability import *
from nltk.corpus.reader import TaggedCorpusReader
from Preprocessing import *


if __name__ == '__main__':
    # nltk.download('tagsets')
    train = TaggedCorpusReader(root="resources", fileids="BAWE_train.retagged.txt")
    test = TaggedCorpusReader(root="resources", fileids="BAWE_test.retagged.txt")
    # nltk.help.upenn_tagset("NN")
    train = train.tagged_words()
    print(train)

    print('---------Preprocessing--------------')
    most_freq = mostFrequent(train)
    print("the three most common tags are", most_freq)
    words = mostFreqTag_Relative(train)
    print("Before noun relative frequency", words)
    verbs = mostFreq_verb(train)
    print("The verbs relative frequency", verbs)
