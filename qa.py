
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
# from stubs. import baseline_stub_word2vec_demo as baseline
# import chunk_demo as chunk
# import constituency_demo_stub as cgraph
import operator
import gensim
import re
import nltk
from nltk.corpus import brown

model = gensim.models.KeyedVectors.load_word2vec_format('pruned_word2vec.txt', binary=False)
stopwords = set(nltk.corpus.stopwords.words("english"))

# def recursive_find_ans_word(node_indicies):
#     for dep in node['deps'].items():
#         if re.match(r'^V*', dep[0]):
#             highest_subj_ind = dep[1][0]
#         else:
#             if len(q_dep_graph.get_by_address(dep[1])['deps']) > 0:
#                 nodes_to_search += [dep[1]]
#     if highest_subj_ind == 0:
#         find_ans_word(q_dep_graph, nodes_to_search)

def find_ans_word(q_dep_graph):
    highest_subj_ind = 0
    highest_subj = ''
    root_word = ''
    nodes_to_search = []
    for nodeNum in q_dep_graph.nodes:
        node = q_dep_graph.get_by_address(nodeNum)
        print(node)
        if node['rel'] == 'root':
            root_word = node['word']
            print('root deps:')
            # print(node['deps'].items())
            #if root is qword, recursively find nsub
            # if re.match(r'^W*', node['tag']):
            #     dep_nodes_list = []
            #     for node in node['deps']:
            #         dep_nodes_list.append(node[1])
            #     recursive_find_ans_word(dep_nodes_list)
            for dep in node['deps'].items():
                if q_dep_graph.get_by_address(dep[1][0])['rel'] == 'nsubj':
                    highest_subj_ind = dep[1][0]
    highest_subj = q_dep_graph.get_by_address(highest_subj_ind)['word']
    if highest_subj is None:
        for dep in node['deps'].items():
            if q_dep_graph.get_by_address(dep[1][0])['tag'].lower()[0] == 'v':
                highest_subj_ind = dep[1][0]
        if highest_subj_ind == 0:
            highest_subj = root_word
        else:
            highest_subj = q_dep_graph.get_by_address(highest_subj_ind)['word']

    print("best q word: " + highest_subj)
    return highest_subj

def find_answer(s_con_graph, s_dep_graph, q_dep_graph, pattern):
    pattern = nltk.ParentedTree.fromstring(pattern)
    phrases = pattern_matcher(pattern, s_con_graph)
    phrases += pattern_matcher("(VP)", s_con_graph)
    phrase_sims = []

    important_q_word = find_ans_word(q_dep_graph)
    word_in_ans_phrase = " "


    # most_similar_word = ""
    high_sim = 0
    print(s_con_graph)
    for nodeNum in s_dep_graph.nodes:
        node = s_dep_graph.get_by_address(nodeNum)
        print(node)
        # if node['word'] == important_q_word:
        if node['word'] in model.vocab and important_q_word in model.vocab:
            word_sim = model.similarity(node['word'], important_q_word)
            if word_sim > high_sim:
                high_sim = word_sim
                if node['head'] != None:
                    word_in_ans_phrase = s_dep_graph.get_by_address(node['head'])["word"]
                    print(word_in_ans_phrase)
                else:
                    if node['word'] != None:
                        word_in_ans_phrase = node['word']
                        print(word_in_ans_phrase)

    # print("word in ans: " + word_in_ans_phrase)
    highest_sim = 0
    best_phrase = phrases[0]
    for phrase in phrases:
        for word in phrase.leaves():
            print("Word: " + word)
            if word in model.vocab and word_in_ans_phrase in model.vocab:
                word_sim = model.similarity(word, word_in_ans_phrase)
                if word_sim > highest_sim:
                    highest_sim = word_sim
                    best_phrase = phrase

    return " ".join(best_phrase.leaves())
    #use dependency relations to decide which noun phrase contains the correct answer


#"What" question specific function
def get_best_what_sentence(filtered_sents, filtered_question, tree):

    
    # for subtree in tree:
    #     print(subtree)
    #     phrases = []

        # pattern1 = nltk.ParentedTree.fromstring("(VP)")
        # phrases += pattern_matcher(pattern1, subtree)

    # print(phrases)

    #print q dep
    print(filtered_question[1])


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
            current_best_graph = pair[2]
            current_best_dep_graph = pair[3]
        print(avg_weight)
    # print("current best graph: ")
    # print(current_best_graph)

    # print(find_answer(current_best_graph))
    pattern = "(NP)"
    # print(current_best_dep_graph)
    return find_answer(current_best_con_graph, current_best_dep_graph, filtered_question[1], pattern)
    # return find_answer(current_best_con_graph, current_best_dep_graph, question_dep_graph, pattern)
    # return current_best[0]

def get_best_where_sentence(filtered_sents, filtered_question):
    current_best = (filtered_sents[0][1], 0)
    for pair in filtered_sents:
        sent_sim_weight_total = 0
        significant_weights = 0

        # print(pair[0])
        for word in pair[0]:
        
            for qword in filtered_question:
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
                    
        # print(sent_sim_weight_total)
        if significant_weights is not 0:
            avg_weight = sent_sim_weight_total/len(pair[0])
        else:
            avg_weight = 0
        if avg_weight > current_best[1]:
            current_best = (pair[1], avg_weight)
        print(avg_weight)
    # print("current best: ")
    # print(current_best[0])
    return current_best[0]

# See if our pattern matches the current root of the tree
def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None: 
        return root
        
    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:                
        return root
        
    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:                   
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild) 
            if match is None:
                return None 
        return root
    
    return None

def pattern_matcher(pattern, tree):
    nodes = []
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            nodes.append(node)
    return nodes

def match_sent_structs(pattern, tree):
    nodes = []
    for subtree in tree:
        node = matches(pattern, subtree)
        if node is not None:
            nodes.append(node)
    return nodes

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences	

def get_bow(tagged_tokens, stopwords):
    return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords and re.match(r"\w+", t[0].lower()) is not None])

def match_trees(pattern, tree, sent_structs, sent_deps):
    possible_sents = []
    filtered_sents = []
    # # Match our pattern to the tree
    for subtree in tree:
        subtrees = pattern_matcher(pattern, subtree)
        # print(subtrees)
        if len(subtrees) > 0:
            possible_sents.append(" ".join(subtrees[0].leaves()))
    sent_num = 0
    for sent in possible_sents:
        filtered_sents.append((get_bow(get_sentences(sent)[0], stopwords), sent, sent_structs[sent_num], sent_deps[sent_num]))
        sent_num += 1
    # print(filtered_sents)
    return filtered_sents

#decides which algorithm to use
def choose_sentence(question, story):
    question_word = question['text'].split(' ', 1)[0].lower()
    # try:
    #     tree = story["sch_par"]
    # except:
    tree = story["story_par"]

    sentence = None
    if question_word == "what":
        pattern = nltk.ParentedTree.fromstring("(ROOT)")
        sentence_structs = match_sent_structs(pattern, tree)
        sent_deps = story['story_dep']
        filtered_sents = match_trees(pattern, tree, sentence_structs, sent_deps)
        #change here if we don't want qbow:
        filtered_question = (get_bow(get_sentences(question['text'])[0], stopwords), question['dep'])
        sentence = get_best_what_sentence(filtered_sents, filtered_question, tree)
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
    #     filtered_sents = match_trees(pattern, tree)
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_sentence(filtered_sents, filtered_question, question_word)

    # else:
    #     pattern = nltk.ParentedTree.fromstring("(ROOT)")
    #     filtered_sents = match_trees(pattern, tree)
    #     #change here if we don't want qbow:
    #     filtered_question = get_bow(get_sentences(question)[0], stopwords)
    #     sentence = get_best_what_sentence(filtered_sents, filtered_question)

    return sentence

    
    # if isinstance(story["sch"], str):
    #     tree = story["sch_par"]
    # else:


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


def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str

    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        id  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    ###     Your Code Goes Here         ###
    # print(story["text"])

    # use sch if it's there
    if(isinstance(story["sch"], str)):
        sentences = get_sentences(story["sch"])
    else:
        sentences = get_sentences(story["text"])
    # sentences = get_sentences(story["text"])

    # print("\n" + question_word + "\n")
    
    print(question['qid'] + ": " + question["text"])
    # print(question['dep'])

    qbow = get_bow(get_sentences(question["text"])[0], stopwords)
    print("qbow:" + str(qbow))
    answer = " ".join([t[0] for t in baseline(qbow, sentences, stopwords)])

    # #if sch is not available use our algorithm
    # if(not isinstance(story["sch"], str)):
    #choose sentence arbitrates strategy to use  for finding best sentence
    sentence = choose_sentence(question, story)
    if sentence is not None:
        answer = sentence

    

    # print(answer + "\n")
    # if(isinstance(story["sch"], str)):
    #     print("Scherezade\n")

    ###     End of Your Code         ###
    return answer



#############################################################
###     Dont change the code below here
#############################################################

class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
