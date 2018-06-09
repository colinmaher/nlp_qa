from utils import (nltk, get_bow, get_sentences, match_trees, better_bow, 
                    match_sent_structs, model, stopwords, wn_story_dict)
import operator
from nltk.stem.porter import *

#"What" question specific function
def get_best_what_sentence(filtered_sents, filtered_question, tree):

    
    # for subtree in tree:
    #     print(subtree)
    #     phrases = []

        # pattern1 = nltk.ParentedTree.fromstring("(VP)")
        # phrases += pattern_matcher(pattern1, subtree)

    # print(phrases)

    #print q dep
    # print(filtered_question[1])


    current_best = (filtered_sents[0][1], 0)
    current_best_con_graph = filtered_sents[0][2]
    current_best_dep_graph = filtered_sents[0][3]
    for pair in filtered_sents:
        sent_sim_weight_total = 0
        significant_weights = 0

        for word in pair[0]:
        
            for qword in filtered_question[0]:
                #check if words are in the model
                if qword in model.vocab and word in model.vocab:
                    sim = model.similarity(word, qword)
                    
                    # if sim > 0.98:
                    #     sent_sim_weight_total += weights[0]
                    #     significant_weights += 1
                    #     print('same word *= ' + str(1.25))
                    if sim > 0.1:
                        # print(word, qword)
                        # print(sim)
                        sent_sim_weight_total += sim
                        significant_weights += 1

                elif qword == word: #for words not in model (like names)
                    sent_sim_weight_total += 2
                    significant_weights += 1
                    # print('same name += 2')
                    
        # print(sent_sim_weight_total, significant_weights)
        if significant_weights is not 0:
            avg_weight = sent_sim_weight_total/len(pair[0])
        else:
            avg_weight = 0
        if avg_weight > current_best[1]:
            current_best = (pair[1], avg_weight)
            current_best_con_graph = pair[2]
            current_best_dep_graph = pair[3]
        # print(avg_weight)
    # print("current best graph: ")
    # print(current_best_graph)

    # print(find_answer(current_best_graph))
    pattern = "(NP)"
    # print(current_best_dep_graph)
    # find_answer(current_best_con_graph, current_best_dep_graph, filtered_question[1], pattern)

    return current_best[0]


#decides which algorithm to use
def choose_sentence(question, story):
    # question_word = question['text'].split(' ', 1)[0].lower()
    # # try:
    # #     tree = story["sch_par"]
    # # except:
    # tree = story["story_par"]

    # sentence = None
    # pattern = nltk.ParentedTree.fromstring("(ROOT)")
    # sentence_structs = match_sent_structs(pattern, tree)
    # sent_deps = story['story_dep']
    # filtered_sents = match_trees(pattern, tree, sentence_structs, sent_deps)
    # #change here if we don't want qbow:
    # filtered_question = (get_bow(get_sentences(question['text'])[0], stopwords), question['dep'])

    # if question_word == "what":
        # sentence = get_best_what_sentence(filtered_sents, filtered_question, tree)
    diff = question['difficulty']
    # if diff != 'Discourse':
    sentence = get_best_wordnet_sent(question, story)
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
                if qword == word:
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
                    sim = model.similarity(word, qword)               
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
        

