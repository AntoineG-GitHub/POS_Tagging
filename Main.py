import nltk as nltk
from nltk.corpus.reader import TaggedCorpusReader


if __name__ == '__main__':
    nltk.download('tagsets')
    train = TaggedCorpusReader(root="resources", fileids="BAWE_train.retagged.txt")
    test = TaggedCorpusReader(root="resources", fileids="BAWE_test.retagged.txt")
    nltk.help.upenn_tagset("NN")