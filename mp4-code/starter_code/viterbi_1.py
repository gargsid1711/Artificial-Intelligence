"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
import numpy as np


def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    P_s, P_t, P_e, global_tag_count = compute_parameters(train)
    unique_tags = list(global_tag_count.keys())
    P_s, P_t, P_e = laplace_smoothing(P_s, P_t, P_e, len(train), unique_tags, global_tag_count, laplace=0.00001)

    print(P_t)

    predictions = []
    i = 0
    for sentence in test:
        predictions.append(execute_viterbi(P_s, P_t, P_e, sentence, unique_tags))
        break


    return predictions

def compute_parameters(train):

    #computing probabilities of a tags on the first word in the sentence
    P_s = {}
    P_t = {}
    P_e = {}

    #print(sum(P_s.values()))

    global_tag_count = {}
    for sentence in train:
        for word in sentence:
            curr_tag = word[1]
            if curr_tag in ['START', 'END']:
                continue
            global_tag_count[curr_tag] = global_tag_count.get(curr_tag, 0) + 1

    for sentence in train:
        prev_tag = 'START'
        for curr_word, curr_tag in sentence:
            tag_dict = {}
            if curr_tag == 'START':
                continue
            elif prev_tag == 'START':
                #computing initial probabilities
                P_s[curr_tag] = P_s.get(curr_tag, 0) + 1
            else:
                #computing transition probabilites
                P_t[(prev_tag, curr_tag)] = P_t.get((prev_tag, curr_tag), 0) + 1

            if curr_word in P_e.keys():
                #computing emission probabilities
                tag_dict = P_e.get(curr_word)
                tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1
            else:
                tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1
                P_e[curr_word] = tag_dict

            prev_tag = curr_tag

    return P_s, P_t, P_e, global_tag_count


def laplace_smoothing(P_s, P_t, P_e, no_of_sent, unique_tags, global_tag_count, laplace):

    for tag in unique_tags:
        P_s[tag] = (P_s.get(tag) + laplace) / (no_of_sent + laplace * (len(unique_tags) + 1))
    P_s["UNK"] = laplace/ (no_of_sent + laplace * (len(unique_tags) + 1))


    for word in P_e:
        tag_dict = P_e.get(word)
        word_count = sum(tag_dict.values())
        for tag in unique_tags:
            tag_dict[tag] = (tag_dict.get(tag, 0) + laplace) / word_count + (laplace * (len(unique_tags) + 1))
    P_e["UNK"] = laplace / (word_count + laplace * (len(unique_tags) + 1))


    for (prev_tag, curr_tag) in P_t.keys():
        P_t[(prev_tag, curr_tag)] = (P_t.get((prev_tag, curr_tag)) + laplace) / (sum(P_t.values()) + (laplace * (len(unique_tags) + 1)))

    P_t["UNK"] = laplace / (sum(P_t.values()) + (laplace * (len(unique_tags) + 1)))

    return P_s, P_t, P_e



def execute_viterbi(P_s, P_t, P_e, sentence, unique_tags):

    trellis = np.zeros(shape=(len(unique_tags), len(sentence)))
    b_arr = ["" for i in range(len(sentence))]
    b_arr[0] = "START"
    b_arr[-1] = "END"

    #populating first column of array
    for i in range(len(unique_tags)):
        tag = unique_tags[i]
        first_word = sentence[1]
        if first_word in P_s.keys():
            if first_word in P_e.keys():
                trellis[i][1] = math.log10(P_s.get(tag, 0)) + math.log10(P_e.get(sentence[1]).get(tag))
            else:
                trellis[i][1] = math.log10(P_s.get(tag, 0)) + math.log10(P_e.get("UNK"))
        elif first_word in P_e.keys():
            trellis[i][1] = math.log10(P_s.get("UNK")) + math.log10(P_e.get(sentence[1]).get(tag))
        else:
            trellis[i][1] = math.log10(P_s.get("UNK")) + math.log10(P_e.get("UNK"))

    #populating rest of the columns of the trellis
    if len(sentence) > 2:
        for j in range(2, len(sentence)):
            for i in range(len(unique_tags)):
                max_prob = 0
                back_tag = None
                curr_tag = unique_tags[i]
                curr_word = sentence[j]
                for k in range(len(unique_tags)):
                    if curr_word in P_e.keys():
                        if (unique_tags[k], curr_tag) in P_t.keys():
                            prob_v = trellis[k, j-1] + math.log10(P_t.get((unique_tags[k], curr_tag))) + math.log10(P_e.get(curr_word).get(curr_tag))
                        else:
                            prob_v = trellis[k, j-1] + math.log10(P_t.get("UNK")) + math.log10(P_e.get(curr_word).get(curr_tag))
                    elif (unique_tags[k], curr_tag) in P_t.keys():
                        prob_v = trellis[k, j-1] + math.log10(P_t.get((unique_tags[k], curr_tag))) + math.log10(P_e.get("UNK"))
                    else:
                        prob_v = trellis[k, j-1] + math.log10(P_t.get("UNK")) + math.log10(P_e.get("UNK"))

                    if prob_v < max_prob:
                        print(unique_tags[k], prob_v)
                        max_prob = prob_v
                        back_tag = unique_tags[k]
                trellis[i, j] = max_prob
                b_arr[j-1] = back_tag
    print(b_arr)
    print(trellis)

    #print(trellis)

    tagged_sentence = []
    for i in range(len(sentence)):
        tagged_sentence.append((sentence[i], b_arr[i]))

    print(tagged_sentence)
    return tagged_sentence

""""
max_index = np.argmax(trellis[:, j - 1])
b_arr[j - 1] = unique_tags[max_index]
"""