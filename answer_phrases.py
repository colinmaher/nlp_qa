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

def find_answer(question, sent_dep):
    return None