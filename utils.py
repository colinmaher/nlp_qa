import nltk, re, gensim
from nltk.corpus import wordnet as wn
from wordnet.wordnet_demo import load_wordnet_ids
from nltk.corpus import brown
from nltk.stem.porter import *

try: #check if model is generated and saved
    model = gensim.models.KeyedVectors.load('pruned_w2v_model')
except:
    #remake and save if not
    model = gensim.models.KeyedVectors.load_word2vec_format('pruned_word2vec.txt', binary=False)
    model.save("pruned_w2v_model")
    model = gensim.models.KeyedVectors.load('pruned_w2v_model')

stopwords = set(nltk.corpus.stopwords.words("english"))

# collocations from brown corpus
brown_collocations = []
def generate_collocations():
    global brown_collocations
    if brown_collocations is None:
        brown_words_per_sentence = [brown.words(fileid) for fileid in brown.fileids()]
        brown_words = [word.lower() for sublist in brown_words_per_sentence for word in sublist]
        brown_text = nltk.Text(brown_words)
        # ignored_words = stopwords.words('english')
        finder = nltk.collocations.BigramCollocationFinder.from_words(brown_words, 2)
        finder.apply_freq_filter(2)
        # finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
        brown_collocations = brown_text.collocations()
        print(brown_collocations)

# load in wordnet data about stories
noun_ids = load_wordnet_ids("{}/{}".format("./wordnet", "Wordnet_nouns.csv"))
verb_ids = load_wordnet_ids("{}/{}".format("./wordnet", "Wordnet_verbs.csv"))

# print("noun ids: ")
# print(noun_ids)
# print("verb ids: ")
# print(verb_ids)

# generate synonyms, hypernyms, hyponyms of nouns and verbs in story sentences
# and create a new list containing an array of the synonyms etc that were found
# in the given csv files
wn_story_dict = {}
def generate_wn_list(story):
    #define lemmatizer
    wn_lem = PorterStemmer()
    
    sid = story['sid']
    if sid in wn_story_dict:
        return
    else:
        if isinstance(story["sch"], str):
            story_txt = story['sch']
        else:
            story_txt = story['text']

        wn_sent_list = []
        wn_story_dict[sid] = []
        sents_tagged = get_sentences(story_txt)
        sents = nltk.sent_tokenize(story_txt)
        # print(sents)
        j = 0
        for sent in sents:
            word_list = set()   
            sbow = get_bow(sents_tagged[j], stopwords)
            # words = nltk.word_tokenize(sent)
            # print(sent)
            # print(sbow)
            word_is_annotated = False
            for word in sbow:
                word_lem = wn_lem.stem(word)
                # print("word_lem: " + word_lem)
                for key in noun_ids.keys():
                    if word_lem in noun_ids[key]["story_noun"]:
                        word_is_annotated = True
                        ss = wn.synset(key)
                        # fetch synonyms, hypernyms, hyponyms of noun
                        synonyms = [lemma.name() for lemma in ss.lemmas()]
                        hypernyms_lemmas = [hyp.lemma_names() for hyp in ss.hypernyms()]
                        hypernyms = []

                        for lemmas in hypernyms_lemmas:
                            hypernyms += lemmas

                        hyponyms_lemmas = [hyp.lemma_names() for hyp in ss.hyponyms()]
                        hyponyms = []

                        for lemmas in hyponyms_lemmas:
                            hyponyms += lemmas

                        for syn in synonyms:
                            word_list.add(syn)
                        for hyp in hypernyms:
                            word_list.add(hyp)
                        for hypo in hyponyms:
                            word_list.add(hypo)
                    
                for key in verb_ids.keys():
                    if word_lem in verb_ids[key]["story_verb"]:
                        word_is_annotated = True
                        ss = wn.synset(key)
                        # fetch synonyms, hypernyms, hyponyms of verb
                        synonyms = [lemma.name() for lemma in ss.lemmas()]

                        hypernyms_lemmas = [hyp.lemma_names() for hyp in ss.hypernyms()]
                        hypernyms = []

                        for lemmas in hypernyms_lemmas:
                            hypernyms += lemmas

                        hyponyms_lemmas = [hyp.lemma_names() for hyp in ss.hyponyms()]
                        hyponyms = []

                        for lemmas in hyponyms_lemmas:
                            hyponyms += lemmas

                        for syn in synonyms:
                            word_list.add(syn)
                        for hyp in hypernyms:
                            word_list.add(hyp)
                        for hypo in hyponyms:
                            word_list.add(hypo)

                # add each sbow word to its sentence list
                #lemmatize word before adding, because we'll compare to lemmatized qwords
                word_list.add(word)
            j+=1
            # print(word_list)
            wn_sent_list.append(word_list)
        wn_story_dict[sid] = wn_sent_list
        
        
        print(wn_story_dict[sid])
            


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

   