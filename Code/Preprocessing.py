import nltk


def mostFrequent(train):
    """
    Returns most frequent tags
    :param train: train data set
    :return: a dictionary with the most frequent tags
    """
    pos_counts = nltk.FreqDist(tag for (word, tag) in train)
    pos_counts = pos_counts.most_common(3)
    most_freq_tags = dict(pos_counts)
    return most_freq_tags


def mostFreqTag_Relative(train):
    """
    Returns relative most frequent tags
    :param train: train data set
    :return: the most frequent relative tags
    """
    pos_counts = nltk.bigrams(tag for (word, tag) in train)
    prec = {}
    for i in list(pos_counts):
        if i[1] == 'NN':
            if i[0] in prec:
                prec[str(i[0])] += 1
            else:
                prec[str(i[0])] = 1
    dic_mostFreq = dict(nltk.FreqDist(prec).most_common(3))
    most = {}
    for (key) in dic_mostFreq:
        most[key] = round(dic_mostFreq[key]/sum(prec.values()), ndigits=2)
    return most


def mostFreq_verb(train):
    """
    Returns most frequent verbs
    :param train: train data set
    :return: the most frequent verbs
    """
    pos_counts = nltk.FreqDist(word for (word, tag) in train if tag.startswith("VB"))
    most_freq = pos_counts.most_common(3)
    pos_counts = dict(pos_counts)
    most_freq = dict(most_freq)
    total_value = sum(pos_counts.values())
    for key in most_freq:
        most_freq[key] = round(most_freq[key] / total_value, ndigits=3)
    return most_freq
