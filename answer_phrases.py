import operator
import re
from utils import nltk, model, stopwords, pattern_matcher, match_sent_structs, get_bow, get_sentences

#function to recursively go up the dependency tree to find the word in the question we wish
#to look for in the answer
# def recursive_find_ans_word(node_indicies):
#     for dep in node['deps'].items():
#         if re.match(r'^V*', dep[0]):
#             highest_subj_ind = dep[1][0]
#         else:
#             if len(q_dep_graph.get_by_address(dep[1])['deps']) > 0:
#                 nodes_to_search += [dep[1]]
#     if highest_subj_ind == 0:
#         find_ans_word(q_dep_graph, nodes_to_search)

# def find_ans_word(q_dep_graph):
#     highest_subj_ind = 0
#     highest_subj = ''
#     root_word = ''
#     nodes_to_search = []
#     for nodeNum in q_dep_graph.nodes:
#         node = q_dep_graph.get_by_address(nodeNum)
#         # print(node)
#         if node['rel'] == 'root':
#             root_word = node['word']
#             # print('root deps:')
#             # print(node['deps'].items())
#             #if root is qword, recursively find nsub
#             # if re.match(r'^W*', node['tag']):
#             #     dep_nodes_list = []
#             #     for node in node['deps']:
#             #         dep_nodes_list.append(node[1])
#             #     recursive_find_ans_word(dep_nodes_list)
#             for dep in node['deps'].items():
#                 if q_dep_graph.get_by_address(dep[1][0])['rel'] == 'nsubj':
#                     highest_subj_ind = dep[1][0]
#     highest_subj = q_dep_graph.get_by_address(highest_subj_ind)['word']
#     if highest_subj is None:
#         for dep in node['deps'].items():
#             if q_dep_graph.get_by_address(dep[1][0])['tag'].lower()[0] == 'v':
#                 highest_subj_ind = dep[1][0]
#         if highest_subj_ind == 0:
#             highest_subj = root_word
#         else:
#             highest_subj = q_dep_graph.get_by_address(highest_subj_ind)['word']

#     # print("best q word: " + highest_subj)
#     return highest_subj

def find_answer(question, sent_dep, sent_con):
    print('in find ans')
    #get right types of phrase based on question first
    qtokens = nltk.word_tokenize(question['text'])
    qword = qtokens[0].lower()
    qbow = get_bow(get_sentences(question['text'])[0], stopwords)
    
    phrases = ""
    print(sent_con)
    if qword == 'what':
        # print("sent constuency graph:")
        # for tree in sent_con.subtrees():
        #     print(tree)
        pattern = nltk.ParentedTree.fromstring("(NP)")
        phrases = pattern_matcher(pattern, sent_con)
        # pattern = nltk.ParentedTree.fromstring("(VP)")
        # phrases += pattern_matcher(pattern, sent_con)
        
    if qword == 'where':
        pattern = nltk.ParentedTree.fromstring("(PP)")
        phrases = pattern_matcher(pattern, sent_con)

    elif qword == 'who':
        pattern = nltk.ParentedTree.fromstring("(NP)")
        phrases = pattern_matcher(pattern, sent_con)
        # pattern = nltk.ParentedTree.fromstring("(NNP)")
        # phrases += pattern_matcher(pattern, sent_con)
        pattern = nltk.ParentedTree.fromstring("(MD)")
        phrases += pattern_matcher(pattern, sent_con)

    elif qword == 'when':
        pattern = nltk.ParentedTree.fromstring("(NP)")
        phrases = pattern_matcher(pattern, sent_con)
        pattern = nltk.ParentedTree.fromstring("(PP)")
        phrases += pattern_matcher(pattern, sent_con)
    
    #look at phrases with 'because'
    elif qword == 'why':
        pattern = nltk.ParentedTree.fromstring("(SBAR)")
        phrases = pattern_matcher(pattern, sent_con)

    elif qword == 'which':
        pattern = nltk.ParentedTree.fromstring("(NP)")
        phrases = pattern_matcher(pattern, sent_con)
        

    elif qword == 'how':
        # pattern = nltk.ParentedTree.fromstring("(NP)")
        # phrases = pattern_matcher(pattern, sent_con)
        pattern = nltk.ParentedTree.fromstring("(PP)")
        phrases = pattern_matcher(pattern, sent_con)
        pattern = nltk.ParentedTree.fromstring("(VP)")
        phrases += pattern_matcher(pattern, sent_con)

    # else:
    #     pattern = nltk.ParentedTree.fromstring("(NP)")
    #     phrases = pattern_matcher(pattern, sent_con)
    #     pattern = nltk.ParentedTree.fromstring("(VP)")
    #     phrases += pattern_matcher(pattern, sent_con)

    if phrases != "":
        joined_phrases = ""
        for phrase_tree in phrases:
            phrase = phrase_tree.leaves()
            print("phrase leaves: ")
            print(phrase)
            use_phrase = True
            for word in phrase:
                print("qbow: ")
                print(qbow)
                if word in qbow:
                    use_phrase = False
            if use_phrase: 
                joined_phrases += " ".join(phrase) + " "

        print("phrases:")
        print(joined_phrases)
        if joined_phrases != "": 
            # phrase_tokens = " ".join(join)
            return joined_phrases

    for node in sent_dep.nodes.values():

        if node['rel'] == "root":
            deps = get_dependents(node, sent_dep)
            
            deps = sorted(deps+[node], key=operator.itemgetter("address"))

            
            return " ".join(dep["word"] for dep in deps if re.match(r"\w+", dep["word"]) != None)


def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
        
    return results