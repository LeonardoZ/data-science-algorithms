# coding=utf8
from collections import defaultdict, Counter
import math
from ml import split_data
import glob, re
import random

def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)

def count_words(training_set):
    """ pares (message, is_spam) """
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """ word_counts para lista de triplas w, p(w | spam) e p(w / ~spam) """ 
    return [(w,
            (spam + k) / (total_spams + 2 * k),
            (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.iteritems()]    

def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    #itera cada palavra do vocabulário
    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # se a palavra aparecer na mensagem,
        # adicione a probabilidade log de vela
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # se a palavra não aparece na mesagem
        # adicione a probabilidade log de não vê-la
        # que é log(1 - probabilidade de vê-la)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # conta mensagens spam e não spam
        num_spams = len([is_spam for message, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        # roda dados de trainamento pela nosa pipeline
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts, num_spams, num_non_spams, self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def identify():
    """ Download from http://spamassassin.apache.org/publiccorpus/ 2002210 files and set path"""
    path = r"spam/*/*"

    data = []

    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn, 'r') as file:
            for line in file:
                if line.startswith("Subject:"):
                    # remove o primeiro Subject:
                    subject = re.sub(r"^Subject: ", "", line).strip()
                    data.append((subject, is_spam))

    random.seed(0)
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    # triplas (subject, isspam real e probabilidade)
    classified = [(subject, is_spam, classifier.classify(subject)) for subject, is_spam in test_data]


    # sendo spam_prob > 0.5 igual a previsão
    # conta combinações
    counts = Counter((is_spam, spam_probability > 0.5) for _, is_spam, spam_probability in classified)
    
    #probability de maior para menor
    classified.sort(key = lambda row: row[2])

    # maiores probabilidades de spam prevista entre os não spams
    spammiest_hams = filter(lambda row: not row[1], classified)[-5:]

    # as menores probabilidades do spam previsto estar entre os spams
    hammiest_spams = filter(lambda row: row[1], classified)[:5]

    print "\nSpammiest: ", spammiest_hams
    print "\nHammiest: ", hammiest_spams
    print "\nCounts: ", counts

    def p_spam_given_word(word_prob):
        """ usa o teorema de bayes para computar p(spam / message contains word) """
        word, prob_if_spam, prob_if_not_spam = word_prob
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)

    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_hams = words[-5:]
    hammiest_spams = words[:5]
    print "\nSorted Spammiest: ", spammiest_hams
    print "\nSorted Hammiest", hammiest_spams

identify()