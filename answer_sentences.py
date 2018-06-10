from utils import (nltk, get_bow, get_sentences, match_trees, better_bow, 
                    match_sent_structs, model, stopwords, wn_story_dict, pattern_matcher)
import operator
from nltk.stem.porter import *

def choose_sentence(question, story):

    diff = question['difficulty']
    sentence = get_best_wordnet_sent(question, story)

    
    if diff == 'Discourse':
        #get sentence index so we can retrive sentences before /after for discourse
        print('discourse question')
        if(isinstance(story["sch"], str)):
            sentences = story["sch"]
        else:
            sentences = story["text"]
        sentences = nltk.sent_tokenize(sentences)
        i = 0
        sent_ind = 0
        for sent in sentences:
            if sent == sentence:
                sent_ind = i
            i+=1

        print("sent_ind: " + str(sent_ind))

        discourse_type = ''
        qwords = nltk.word_tokenize(question['text'])
        lowered_qwords = []
        for qword in qwords: lowered_qwords.append(qword.lower())
        qwords = lowered_qwords
        print("tokenized qwords:" + str(qwords))

        # if question has 'after' in it
        if 'after' in qwords:
            discourse_type = 'after'
        # in question has 'before' in it
        elif 'first' in qwords:
            discourse_type = 'first'
        elif 'before' in qwords:
            discourse_type = 'before'

        print('discourse_type: ' + discourse_type)

        sent_words = nltk.word_tokenize(sentence)
        lowered_sent = []
        for word in sent_words: lowered_sent.append(word.lower())

        if discourse_type in sent_words:
            return sentence
        else:
            if discourse_type == 'after' and sent_ind<len(sentences):
                print('returning sentence after wordnet matched sentence')
                return sentences[sent_ind:sent_ind+2] 
            if (discourse_type == 'first' or discourse_type == 'before') and sent_ind>0:
                print('returning sentence before wordnet matched sentence')
                return sentences[sent_ind-1:sent_ind+1]
        
        if discourse_type == '' and sent_ind>0:
            return sentences[sent_ind-1:sent_ind+1]
    # else:
    #     sentence = None
    #     # find_answer()
    # # (S (NP (*)) (VP (*) (PP)))
    # print("Choose_sentence: " + str(sentence))

    return sentence


#baseline sentence matching by overlap
def baseline(qbow, sentences, stopwords):
    # Collect all the candidate answers

    answers = []
    for sent in sentences:
        # feature_dict = W2vecextractor.get_doc2vec_feature_dict(sent)
        # A list of all the word tokens in the sentence
        # lmtzr = WordNetLemmatizer()
        sbow = get_bow(sent, stopwords)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        
        answers.append((overlap, sent))
        
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = (answers[0])[1]    


    return best_answer

def get_best_wordnet_sent(question, story, use_sch=True):
    # qbow = get_bow(question['text'])
    #initialize stemmer
    stemmer = PorterStemmer()
    best_sent = None
    best_score = 0.0

    #get right version of text to pull best sentence out of
    if(isinstance(story["sch"], str) and use_sch == True):
        sentences = story["sch"]
        print("using sch")
        # print(sentences)
    else:
        sentences = story["text"]
        print("using text")
    sentences = nltk.sent_tokenize(sentences)
    
    #first check qwords against wordnet words
    qwords = nltk.word_tokenize(question['text'])
    qwords = get_bow(get_sentences(question['text'])[0], stopwords)
    # print("better qbow:")
    # print(better_bow(question))

    i = 0
    for sent in wn_story_dict[question['sid']]:
        sent_score = 0.0
        # did_match = False
        words_not_found = set(qwords)
    
        for qword in qwords:
            for word in sent:
                if str(qword) == word:
                    # print('matched ' + qword + ' with ' + word)
                    sent_score += 1
                    # print('sent ' + str(i) + ' score: ' + str(sent_score))
                    # print('sent ' + str(i) + ': ' + sentences[i])
                    words_not_found.remove(qword)
                    # break

        print("words not found: " + str(words_not_found))
        # if words not in wordnet data, try factoring in word similarity a bit
        
        for qword in words_not_found:
            highest_sim = 0
            for word in sent:
                if word in model.vocab and qword in model.vocab:
                    sim = model.similarity(word, str(qword))               
                    # print("sim of '" + word + "' and '" + qword + "' = " + str(sim))
                    if sim > highest_sim:
                        highest_sim = sim
                        # print("sim of '" + word + "' and '" + qword + "' = " + str(sim))
                        # print('sent ' + str(i) + ' score: ' + str(sent_score))
                        # print('sent ' + str(i) + ': ' + sentences[i])

            if highest_sim > 0.3:
                sent_score += highest_sim

        if sent_score > best_score:
            best_score = sent_score
            best_sent = sentences[i]
        print(sent)
        i += 1
    
    #check if we're using default_answer if so, use full text instead of scherazade
    if best_sent == None and use_sch == True:
        return get_best_wordnet_sent(question, story, False)
    return best_sent
        

