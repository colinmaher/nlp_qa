from utils import (nltk, get_bow, get_sentences, match_trees, 
                    match_sent_structs, model, stopwords, pattern_matcher)
from nltk.corpus import wordnet as wn
from wordnet.wordnet_demo import load_wordnet_ids
import operator

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
        print(avg_weight)
    # print("current best graph: ")
    # print(current_best_graph)

    # print(find_answer(current_best_graph))
    pattern = "(NP)"
    # print(current_best_dep_graph)
    # find_answer(current_best_con_graph, current_best_dep_graph, filtered_question[1], pattern)

    return current_best[0]


def find_answer(s_con_graph, patterns, question):
    #match sentence's constituency graph with pattern
    qwords = nltk.word_tokenize(question)
    qword = qwords[0].lower()
    important_q_words = qwords[1:]
    # patt2 = nltk.ParentedTree.fromstring(pattern[1])
    # print(s_con_graph)
    phrase_structs = []
    for pat in patterns:
        patt = nltk.ParentedTree.fromstring(pat)
        phrase_structs += pattern_matcher(patt, s_con_graph)

    # compare phrase similarity with question (not working well)

    # most_sim_phrase = phrase_structs[0]
    # highest_phrase_sim = 0
    # for phrase in phrase_structs:
    #     phrase_text = phrase.leaves()
    #     phrase_sim = 0
    #     significant_weights = 0

    #     for word in phrase_text:
    #         for qword in important_q_words:
    #             if qword in model.vocab and word in model.vocab:
    #                 sim = model.similarity(word, qword)
    #                 if sim > 0.1:
    #                     significant_weights += 1
    #                     # phrase_sim += model.similarity(word, qword)
    #                     phrase_sim += 1
    #                 # elif sim > 0.3:
    #                 #     significant_weights += 1
    #                 #     phrase_sim += 1
    #             elif word == qword:
    #                 phrase_sim += 2
    #     if significant_weights > 0:
    #         phrase_sim = phrase_sim
    #     else:
    #         phrase_sim = 0
    #     if phrase_sim > highest_phrase_sim:
    #         highest_phrase_sim = phrase_sim
    #         most_sim_phrase = phrase
                    

    # phrase_structs += pattern_matcher(patt2, s_con_graph)
    print("Phrase matches: " + str(phrase_structs))
        # if highest_phrase_sim > 0.1:
        #     answer = " ".join(most_sim_phrase.leaves())
        # else:
    
    if(not phrase_structs):
        return
    else:
        largest_phrase = phrase_structs[0]
    for phrase in phrase_structs:
        print(len(phrase))
        size = len(phrase.leaves())
        if size > len(largest_phrase.leaves()):
            largest_phrase = phrase
    
    answer = " ".join(largest_phrase.leaves())
        # size = 0
        # for node in phrase:
        #     size += 1

    return answer

# decides which algorithm to use
# we should pass in the sentence here as a tuple of the text and the dep graph
# also need to use scherezade when applicable
def choose_sentence(question, story):
    question_word = question['text'].split(' ', 1)[0].lower()
    tree = story["story_dep"]
    answer = None

    baseline_con = sentences[0]
    for sent in sentences:
        if " ".join(sent.leaves()) == baseline:
            baseline_con = sent

    # print(baseline_con)

    if question_word == "what":
        pattern = nltk.ParentedTree.fromstring("(ROOT)")
        sentences = match_sent_structs(pattern, tree)
        sent_deps = story['story_dep']
        filtered_sents = match_trees(pattern, tree, sentences, sent_deps)
        #change here if we don't want qbow:
        filtered_question = (get_bow(get_sentences(question['text'])[0], stopwords), question['dep'])
        sentence_tuple = get_best_what_sentence(filtered_sents, filtered_question, tree)
        what_pattern = ["(NP *)"]

        answer = find_answer(sentence_tuple[2], what_pattern, question['text'])
    # (S (NP (*)) (VP (*) (PP)))

    elif question_word == "where":
        where_pattern = ["(PP *)"]
        if(find_answer):
            answer = find_answer(baseline_con, where_pattern, question['text'])
    # elif question_word == "when":
    #     pattern = nltk.ParentedTree.fromstring("(S)")
    #     filtered_sents = match_trees(pattern, tree)
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_sentence(filtered_sents, filtered_question, question_word)

    # elif question_word == "why":
    #     pattern = nltk.ParentedTree.fromstring("(S)")
    #     filtered_sents = match_trees(pattern, tree)
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_sentence(filtered_sents, filtered_question, question_word)
    elif question_word == "who":
        where_pattern = ["(NNP *)", "(NP *)"]
        answer = find_answer(baseline_con, where_pattern, question['text'])

    # elif question_word == "which":
    #     where_pattern = ["(NNP *)", "(NP *)"]
    #     sentence = find_answer(baseline_con, where_pattern, question['text'])



    # else:
    #     pattern = nltk.ParentedTree.fromstring("(ROOT)")
    #     filtered_sents = match_trees(pattern, tree)
    #     #change here if we don't want qbow:
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_what_sentence(filtered_sents, filtered_question)

    return answer


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