from nltk.probability import *
from nltk.corpus.reader import TaggedCorpusReader
from Preprocessing import *
from homographs import get_homographs_one_word
from POS_tagger import build_tagger
from HMM import *

if __name__ == '__main__':
    # nltk.download('tagsets')
    train = TaggedCorpusReader(root="resources", fileids="BAWE_train.retagged.txt")
    test = TaggedCorpusReader(root="resources", fileids="BAWE_test.retagged.txt")
    # nltk.help.upenn_tagset("VBP")
    train_tagged = train.tagged_words()
    # print(train)

    print('---------Preprocessing--------------')
    most_freq = mostFrequent(train_tagged)
    print("the three most common tags are", most_freq)
    words = mostFreqTag_Relative(train_tagged)
    print("Before noun relative frequency", words)
    verbs = mostFreq_verb(train_tagged)
    print("The verbs relative frequency", verbs)
    print('---------Homographs--------------')
    pos_tags = get_homographs_one_word(train, 'The')
    print(pos_tags)
    print('---------POS Tagger--------------')
    tagger = build_tagger(train)
    sentence = "Can you tag this simple sentence with the tagger you just built ?"
    print(tagger.tag(sentence.split()))
    print(round(100 - (tagger.evaluate(test.tagged_sents()) * 100), ndigits=2))
    print('---------HMM Tagger--------------')
    tagger_hmm = hmm(train)
    print(tagger_hmm.tag(sentence.split()))
    print(round(100 - (tagger_hmm.evaluate(test.tagged_sents()) * 100), ndigits=2))
