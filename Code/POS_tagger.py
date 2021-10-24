import nltk


def build_tagger(corpus):
    """
    Returns a unigram nltk tagger trained on the given corpus with
    the specificity that the backoff should be obtained as the most
    frequent tag associated to <UNK> where <UNK> is associated to
    the first occurrence of any word in the text.

    Parameters
    -----------
    corpus: a `TaggedCorpusReader` object created from some
        text representation of the corpus (ie BAWE_train)

    Returns
    -------
    Returns an nltk Tagger which assigns the most likely tag to any
    word from the corpus and falls back to a DefaultTagger which
    returns the most likely tag for the <UNK> token.

    e.g. this object can be obtained by nltk.UnigramTagger(text)

    Warning
    -------
    Pay attention to the fact that when training your tagger, the
    first occurence of any word should be replaced by the <UNK> token.
    """
    tagged_sents = corpus.tagged_sents()
    lexicon = {}
    lexicon_word = []
    word_index = 0
    new = []
    for sentence in tagged_sents:
        tmp = []
        for (word, tag) in sentence:
            if word in lexicon_word:
                word_index += 1
                tmp.append((word, tag))
                pass
            else:
                lexicon_word.append(word)
                tmp.append(('<UNK>', tag))
                word_index += 1
                if tag not in lexicon:
                    lexicon[tag] = 1
                else:
                    lexicon[tag] = lexicon[tag] + 1
            word_index = 0
        new.append(tmp)
    most_common = max(lexicon, key=lexicon.get)
    tagger = nltk.UnigramTagger(new, backoff=nltk.DefaultTagger(most_common))
    return tagger
