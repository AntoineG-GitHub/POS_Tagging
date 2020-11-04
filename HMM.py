import nltk


def hmm(corpus):
    train_sents = corpus.tagged_sents()
    my_tag = nltk.HiddenMarkovModelTagger.train(train_sents)
    return my_tag
