# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import nltk
import math


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here

    train_set = process(train_set)
    dev_set = process(dev_set)

    pos_dict, neg_dict = build_dictionaries(train_set, train_labels)
    pos_count = sum(pos_dict.values())
    neg_count = sum(neg_dict.values())
    vocab_size = get_total_words_in(train_set)

    predictions = []

    no_of_words = vocab_size + 1
    for each_email in dev_set:
        pos_prob = math.log10(pos_prior)
        neg_prob = math.log10(1 - pos_prior)
        for each_word in each_email:
            if (pos_dict.get(each_word, 0) + smoothing_parameter) != 0:
                pos_prob += math.log10(
                    (pos_dict.get(each_word, 0) + smoothing_parameter) / (pos_count + (smoothing_parameter * no_of_words)))
            if (neg_dict.get(each_word, 0) + smoothing_parameter) != 0:
                neg_prob += math.log10(
                    (neg_dict.get(each_word, 0) + smoothing_parameter) / (neg_count + (smoothing_parameter * no_of_words)))

        if pos_prob > neg_prob:
            predictions.append(1)
        else:
            predictions.append(0)

    #predictions = classify_set(dev_set, pos_dict, neg_dict, pos_word_count, neg_word_count, smoothing_parameter, vocab_size, pos_prior)

    # return predicted labels of development set
    return predictions

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=0.8, bigram_lambda=0.05,pos_prior=0.9):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here

    train_set = process(train_set)
    dev_set = process(dev_set)

    pos_dict, neg_dict = build_dictionaries(train_set, train_labels)
    pos_word_count = sum(pos_dict.values())
    neg_word_count = sum(neg_dict.values())
    vocab_size = get_total_words_in(train_set)

    pos_bigram_dict, neg_bigram_dict = build_dict_bigrams(train_set, train_labels)
    pos_bigram_count = sum(pos_bigram_dict.values())
    neg_bigram_count = sum(neg_bigram_dict.values())
    vocab_size_bigrams = get_total_bigram_count(train_set)

    #Mixture model

    predictions = []

    no_of_words = vocab_size + 1
    no_of_bigrams = vocab_size_bigrams + 1

    for each_email in dev_set:
        unigram_pos_prob = math.log10(pos_prior)
        unigram_neg_prob = math.log10(1 - pos_prior)

        for each_word in each_email:
            if (pos_dict.get(each_word, 0) + unigram_smoothing_parameter) != 0:
                unigram_pos_prob += math.log10((pos_dict.get(each_word, 0) + unigram_smoothing_parameter) / (pos_word_count + (unigram_smoothing_parameter * no_of_words)))
            if (neg_dict.get(each_word, 0) + unigram_smoothing_parameter) != 0:
                unigram_neg_prob += math.log10((neg_dict.get(each_word, 0) + unigram_smoothing_parameter) / (neg_word_count + (unigram_smoothing_parameter * no_of_words)))

        bigram_pos_prob = math.log10(pos_prior)
        bigram_neg_prob = math.log10(1 - pos_prior)
        bigrams = list(nltk.bigrams(each_email))

        for each_bigram in bigrams:
            if (pos_bigram_dict.get(each_bigram, 0) + bigram_smoothing_parameter) != 0:
                bigram_pos_prob += math.log10((pos_bigram_dict.get(each_bigram, 0) + bigram_smoothing_parameter) / (pos_bigram_count + (bigram_smoothing_parameter * no_of_bigrams)))
            if (neg_bigram_dict.get(each_bigram, 0) + bigram_smoothing_parameter) != 0:
                bigram_neg_prob += math.log10((neg_bigram_dict.get(each_bigram, 0) + bigram_smoothing_parameter) / (neg_bigram_count + (bigram_smoothing_parameter * no_of_bigrams)))

        pos_prob = ((1 - bigram_lambda) * unigram_pos_prob) + (bigram_lambda * bigram_pos_prob)
        neg_prob = ((1 - bigram_lambda) * unigram_neg_prob) + (bigram_lambda * bigram_neg_prob)

        if pos_prob > neg_prob:
            predictions.append(1)
        else:
            predictions.append(0)

    #mixture_predictions = classify_mixture_model(dev_set, pos_dict, neg_dict, pos_word_count, neg_word_count, pos_bigram_dict, neg_bigram_dict, pos_bigram_count, neg_bigram_count, unigram_smoothing_parameter, bigram_smoothing_parameter, vocab_size, vocab_size_bigrams, bigram_lambda, pos_prior)
    
    # return predicted labels of development set using a bigram model
    return predictions




def classify_set_bigram(dev_set, pos_bigram_dict, neg_bigram_dict, pos_bigram_count, neg_bigram_count, smoothing_param, vocab_size, pos_prior):
    predictions = []
    no_of_words = vocab_size + 1

    for each_email in dev_set:
        pos_prob = math.log10(pos_prior)
        neg_prob = math.log10(1 - pos_prior)
        bigrams = list(nltk.bigrams(each_email))
        
        for each_bigram in bigrams:
            if(pos_bigram_dict.get(each_bigram, 0) + smoothing_param) != 0:
                pos_prob += math.log10((pos_bigram_dict.get(each_bigram, 0) + smoothing_param) / (pos_bigram_count + (smoothing_param * no_of_words)))
            if(neg_bigram_dict.get(each_bigram, 0) + smoothing_param) != 0:
                neg_prob += math.log10((neg_bigram_dict.get(each_bigram, 0) + smoothing_param) / (neg_bigram_count + (smoothing_param * no_of_words)))

        if pos_prob > neg_prob:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def get_total_words_in(email_set):

    flat_list = [item for sublist in email_set for item in sublist]

    return len(set(flat_list))

def process(email_set):
    #removes stop words from data sets

    #print("Starting to process")

    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    result = []

    for each_email in email_set:
        #print("Next email")
        result_nest = []
        for each_word in each_email:
            if each_word not in stopwords:
                result_nest.append(each_word)
        if result_nest:
            result.append(result_nest)

    return result

def build_dictionaries(train_set, train_labels):

    pos_dict = {}
    neg_dict = {}

    for email_index in range(len(train_labels)):

        each_email = train_set[email_index]
        email_label = train_labels[email_index]

        for each_word in each_email:
            if email_label == 1:
                pos_dict[each_word] = pos_dict.get(each_word, 0) + 1
            else:
                neg_dict[each_word] = neg_dict.get(each_word, 0) + 1

    return pos_dict, neg_dict


def build_dict_bigrams(train_set, train_labels):

    pos_bigram_dict = {}
    neg_bigram_dict = {}


    for email_index in range(len(train_labels)):

        each_email = train_set[email_index]
        email_label = train_labels[email_index]

        bigrams = list(nltk.bigrams(each_email))
        #print("Bigrams from nltk: ", bigrams)

        for each_bigram in bigrams:
            if email_label == 1:
                pos_bigram_dict[each_bigram] = pos_bigram_dict.get(each_bigram, 0) + 1
            else:
                neg_bigram_dict[each_bigram] = neg_bigram_dict.get(each_bigram, 0) + 1

    return pos_bigram_dict, neg_bigram_dict

def get_total_bigram_count(email_set):

    count = 0
    for each_email in email_set:
        bigrams = list(nltk.bigrams(each_email))
        count += len(bigrams)
    return count



""""
def classify_set_idf(dev_set, pos_dict, neg_dict, pos_word_count, neg_word_count, df_dict, no_of_docs, smoothing_parameter, vocab_size, pos_prior):

    predictions = []
    no_of_words = vocab_size + 1
    for each_email in dev_set:
        pos_prob = math.log10(pos_prior)
        neg_prob = math.log10(1 - pos_prior)

        for each_word in each_email:
            word_df = df_dict.get(each_word, 0)
            idf_value = 1

            if word_df != 0:
                #print(word_df)
                idf_value = math.log10((no_of_docs + 1) / word_df)
            if (pos_dict.get(each_word, 0) + smoothing_parameter) != 0:
                #print(pos_dict.get(each_word, 0))
                #print(idf_value)
                pos_prob += math.log10((pos_dict.get(each_word, 0) * abs(idf_value) + smoothing_parameter) / (pos_word_count + smoothing_parameter * no_of_words))
            if (neg_dict.get(each_word, 0) + smoothing_parameter) != 0:
                neg_prob += math.log10((neg_dict.get(each_word, 0) * abs(idf_value) + smoothing_parameter) / (neg_word_count + smoothing_parameter * no_of_words))

        if pos_prob > neg_prob:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
"""
