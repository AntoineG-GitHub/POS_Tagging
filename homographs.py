import nltk


def get_homographs_one_word(corpus, the_word):
    """
    Provides the different PoS tags for a given word

    Parameters
    -----------
    corpus: a `TaggedCorpusReader` object created from some
        text representation of the corpus (i.e. BAWE_train)
    the_word: string
        word for which we want to retrieve the PoS tags

    Returns
    -------
    pos_tags: set of PoS tags for the_word
    """
    corpus = corpus.tagged_words()
    lexicon = nltk.FreqDist(word for (word, tag) in corpus)
    if the_word not in lexicon:
        the_word = "<UNK>"
    pos_tags = nltk.FreqDist(tag for (word, tag) in corpus if word == the_word)
    pos_tags = set(dict(pos_tags).keys())
    return pos_tags
