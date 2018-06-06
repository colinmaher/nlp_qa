# import utils
from utils import nltk, pattern_matcher, model

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


def find_ans_word(q_dep_graph):
    highest_subj_ind = 0
    highest_subj = ''
    root_word = ''
    nodes_to_search = []
    for nodeNum in q_dep_graph.nodes:
        node = q_dep_graph.get_by_address(nodeNum)
        # print(node)
        if node['rel'] == 'root':
            root_word = node['word']
            # print('root deps:')
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

    # print("best q word: " + highest_subj)
    return highest_subj

def find_answer(s_con_graph, s_dep_graph, q_dep_graph, pattern):
    pattern = nltk.ParentedTree.fromstring(pattern)
    phrases = pattern_matcher(pattern, s_con_graph)
    phrases += pattern_matcher("(VP)", s_con_graph)
    phrase_sims = []

    important_q_word = find_ans_word(q_dep_graph)
    word_in_ans = "some bull"


    # most_similar_word = ""
    high_sim = 0
    # print(s_con_graph)
    for nodeNum in s_dep_graph.nodes:
        node = s_dep_graph.get_by_address(nodeNum)
        # print(node)
        # if node['word'] == important_q_word:
        if node['word'] is not None:
            if node['word'] in model.vocab and important_q_word in model.vocab:
                word_sim = model.similarity(node['word'], important_q_word)
                if word_sim > high_sim:
                    high_sim = word_sim
                    if node['head'] != 0:
                        word_in_ans = s_dep_graph.get_by_address(node['head'])
                        word_in_ans = word_in_ans["word"]
                        # print(word_in_ans)
                    else:
                        if node['word'] != None:
                            word_in_ans = node['word']
                            # print(word_in_ans)
            else:
                if node['word'].lower() == important_q_word.lower():
                    if node['head'] != 0:
                            word_in_ans = s_dep_graph.get_by_address(node['head'])
                            word_in_ans = word_in_ans["word"]
                            # print(word_in_ans)
                    else:
                        if node['word'] != None:
                            word_in_ans = node['word']
                            # print(word_in_ans)

    # print("word in ans: " + word_in_ans)
    # print("s_graph:")
    # print(s_con_graph)
    # for node in s_con_graph.subtrees(lambda s_con_graph: len(s_con_graph.leaves()) == 1 and word_in_ans == s_con_graph.leaves()[0]):
        # print(node)
        #find smallest tree containing word_in_ans
            #now find the parent noun phrase until the parent is not a noun phrase

        # print(node.parent())


    # for phrase in phrases:
    #     for word in phrase.leaves():
            # print("Word: " + word)
            # if word in model.vocab and word_in_ans in model.vocab:
            #     word_sim = model.similarity(word, word_in_ans)
            #     if word_sim > highest_sim:
            #         highest_sim = word_sim
            #         best_phrase = phrase

            #find the place of the the word_in_ans in the story constituency tree, 
            # go up trees until the tree is no longer an NP, extract the highest NP phrase we ended on

    # return
    #use dependency relations to decide which noun phrase contains the correct answer