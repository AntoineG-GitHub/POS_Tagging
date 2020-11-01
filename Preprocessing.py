import nltk


def mostFrequent(train):
    pos_counts = nltk.FreqDist(tag for (word, tag) in train)
    pos_counts = pos_counts.most_common(3)
    most_freq_tags = dict(pos_counts)
    return most_freq_tags


def mostFreqTag_Relative(train):
    pos_counts = nltk.util.bigrams(train)
    most_freq = pos_counts.most_common(5)[1:4]  # Returns the 5 most frequent tags
    pos_counts = dict(pos_counts)
    most_freq = dict(most_freq)
    total_value = sum(pos_counts.values())
    for key in most_freq:
        most_freq[key] = round(most_freq[key] / total_value, ndigits=3)
    return most_freq


def mostFreq_verb(train):
    pos_counts = nltk.FreqDist(word for (word, tag) in train if tag.startswith("VB"))
    most_freq = pos_counts.most_common(3)
    pos_counts = dict(pos_counts)
    most_freq = dict(most_freq)
    total_value = sum(pos_counts.values())
    for key in most_freq:
        most_freq[key] = round(most_freq[key] / total_value, ndigits=3)
    return most_freq
