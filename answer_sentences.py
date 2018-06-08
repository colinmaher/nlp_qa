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
    if diff != 'Discourse':
        sentence = get_best_wordnet_sent(question, story)
    else:
        sentence = None
        # find_answer()
    # (S (NP (*)) (VP (*) (PP)))

    # elif question_word == "where":
    #     pattern = nltk.ParentedTree.fromstring("(S)")
    #     filtered_sents = match_trees(pattern, tree)
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_where_sentence(filtered_sents, filtered_question)

    # elif question_word == "when":
    #     pattern = nltk.ParentedTree.fromstring("(S)")
    #     filtered_sents = match_trees(pattern, tree)
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_sentence(filtered_sents, filtered_question, question_word)

    # elif question_word == "why":
    #     pattern = nltk.ParentedTree.fromstring("(S)")
    #     fsentence = match_trees(pattern, tree)
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_sentence(filtered_sents, filtered_question, question_word)

    # else:
    #     pattern = nltk.ParentedTree.fromstring("(ROOT)")
    #     filtered_sents = match_trees(pattern, tree)
    #     #change here if we don't want qbow:
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_what_sentence(filtered_sents, filtered_question)

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

def get_best_wordnet_sent(question, story):
    # qbow = get_bow(question['text'])
    #initialize stemmer
    stemmer = PorterStemmer()
    best_sent = 'default_answer'
    best_score = 0.0

    #get right version of text to pull best sentence out of
    if(isinstance(story["sch"], str)):
        sentences = story["sch"]
        # print(sentences)
    else:
        sentences = story["text"]
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
        for word in sent:
            for qword in qwords:
                if qword == word:
                    # print('matched ' + qword + ' with ' + word)
                    sent_score += 1
                    # print('sent ' + str(i) + ' score: ' + str(sent_score))

        # if words not in wordnet data, try factoring in word similarity a bit
        for word in sent:
            for qword in qwords:
                if word in model.vocab and qword in model.vocab:
                    sim = model.similarity(word, qword)                
                    # print("sim of " + word + " " + qword + " = " + str(sim))
                    if sim > 0.5:
                        sent_score += 0.1

        if sent_score > best_score:
            best_score = sent_score
            best_sent = sentences[i]
        i += 1
    #after similarity, check for exact matches between better qbow and sentence words (no need to bow)
    print("best_sent")
    print(best_sent)
    return best_sent
        

