"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_dict = {}
    global_tag_freq = {}

    for sentence in train:
        for word in sentence:
            tag_dict = {}
            curr_word = word[0]
            curr_tag = word[1]
            if curr_word in ['START', 'END']:
                continue

            global_tag_freq[curr_tag] = global_tag_freq.get(curr_tag, 0) + 1

            if curr_word in word_dict.keys():
                tag_dict = word_dict.get(curr_word)
                tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1
            else:
                tag_dict[curr_tag] = 1
                word_dict[curr_word] = tag_dict

    max_freq_tag = keywithmaxval(global_tag_freq)
    print(max_freq_tag)

    data = []
    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            #print(word)
            if word in word_dict.keys():
                #print(word)
                #print(word_dict[word])
                tag = keywithmaxval(word_dict[word])
                tagged_sentence.append((word, tag))
                #if word == 'the':
                    #print(tag)
            else:
                tagged_sentence.append((word, max_freq_tag))
        #print(tagged_sentence)
        data.append(tagged_sentence)

    return data


def keywithmaxval(dict):
    return max(dict, key=dict.get)