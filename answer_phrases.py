import operator
import re
from utils import nltk, model, stopwords, pattern_matcher, match_sent_structs, get_bow, get_sentences

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

        
    if qword == 'where' or qword == 'in' or qword == 'after':
        pattern = nltk.ParentedTree.fromstring("(PP)")
        phrases = pattern_matcher(pattern, sent_con)

    elif qword == 'who':
        pattern = nltk.ParentedTree.fromstring("(NP (DT) (*) (NN))")
        phrases = pattern_matcher(pattern, sent_con)
        pattern = nltk.ParentedTree.fromstring("(NNP)")
        phrases += pattern_matcher(pattern, sent_con)
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
        pattern = nltk.ParentedTree.fromstring("(RB)")
        phrases = pattern_matcher(pattern, sent_con)
        # pattern = nltk.ParentedTree.fromstring("(VP)")
        # phrases += pattern_matcher(pattern, sent_con)

    elif qword == 'had':
        return "no"

    else:
        pattern = nltk.ParentedTree.fromstring("(NP)")
        phrases = pattern_matcher(pattern, sent_con)

    if phrases != "":
        joined_phrases = ""
        num_phrases = 0
        for phrase_tree in phrases:
            phrase = phrase_tree.leaves()
            print("phrase leaves: ")
            print(phrase)
            use_phrase = True
            words_in_q = 0
            for word in phrase:
                print("qbow: ")
                print(qbow)
                if word in qbow:
                    use_phrase = False
            if use_phrase and num_phrases < 3: 
                joined_phrases += " ".join(phrase) + " "
                num_phrases += 1

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