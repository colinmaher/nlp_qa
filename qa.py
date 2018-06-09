'''

NLP Story Q/A bot
by csmaher and vmelkote

qa.py: program entry point

python3 qa.py > output

'''

import utils
from utils import (nltk, stopwords, get_sentences, get_bow, 
                    generate_collocations, generate_wn_list)
# import operator, re, nltk, utils
from answer_sentences import (baseline, choose_sentence)
from answer_phrases import find_answer

from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers


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
    
    #generate resources needed throughout the program first
    generate_collocations()
    generate_wn_list(story)
    
    # # use sch if it's there
    if(isinstance(story["sch"], str)):
        sentences = get_sentences(story["sch"])
        # print(sentences)
    else:
        sentences = get_sentences(story["text"])

    # # print("\n" + question_word + "\n")
    
    # print(question)
    # print(question['qid'] + ": " + question["text"])
    # print(question['dep'])
    # print(story['text'])
    # print(story['sch'])
    qbow = get_bow(get_sentences(question["text"])[0], stopwords)
    print("qbow:" + str(qbow))
    answer = " ".join([t[0] for t in baseline(qbow, sentences, stopwords)])


    # print(question['difficulty'])
    # if (question['difficulty'] == 'Discourse'):
    #     None
    #     qword_text = nltk.Text(nltk.word_tokenize(question['text']))
    #     print(qword_text.collocations(10))
    # else:
    #     # #if sch is not available use our algorithm
    #     # if(not isinstance(story["sch"], str)):
    #     #choose sentence arbitrates strategy to use  for finding best sentence
    #     # noun_ids = load_wordnet_ids("wordnet/Wordnet_nouns.csv")
    #     # verb_ids = load_wordnet_ids("wordnet/Wordnet_verbs.csv")

    #     #get nouns and verbs out of question so we can run them through wordNet
    #     q_dep_graph = question["dep"]
    #     # print(q_dep_graph)
    #     q_verbs = []
    #     q_nouns = []
    #     important_qbow = []
    #     q_noun_synsets = {}
    #     q_verb_synsets = {}

    #     for nodeNum in q_dep_graph.nodes:
    #         node = q_dep_graph.get_by_address(nodeNum)
    #         # print(node)
    #         if node['tag'][0] is 'V':
    #             q_verbs.append(node['word'])
    #         elif node['tag'][0] is 'N':
    #             q_nouns.append(node['word'])
    #     #add any qbow words that weren't verbs or nouns
    #     for qword in qbow:
    #         if qword not in q_verbs and qword not in q_nouns:
    #             important_qbow.append(qword)
    #     for noun in q_nouns:
    #         q_noun_synsets[noun] = wn.synsets(noun)
    #     for verb in q_verbs:
    #         q_verb_synsets[verb] = wn.synsets(verb)
    #     # print(question['text'])


    #     print(q_verbs, q_nouns, important_qbow)
        # for qword in qbow:
        #     q_synsets = wn.synsets(qword)
        #     if q_synsets is not None:
        #         for q_synset in q_synsets:

        #         q_hypo = q_synset.hyponyms()
        #         q_hyper = q_synset.hypernyms()
        #     for sent in sentences:

    sentence = choose_sentence(question, story)
    if sentence != None:
        answer = sentence
        #call function to get part relevant of sentence out
        # s_dep
        
        if(isinstance(story["sch"], str)):
            sentences = nltk.sent_tokenize(story["sch"])
            s_dep = story['sch_dep']
            # print(sentences)
        else:
            sentences = nltk.sent_tokenize(story["text"])
            s_dep = story['story_dep']
        i = 0
        for sent in sentences:
            if sent == sentence:
                # print(s_dep[i])
                answer = find_answer(question, s_dep[i])
            i+=1

    # print(answer + "\n")
    # if(isinstance(story["sch"], str)):
    #     print("Scherezade\n")

    ###     End of Your Code         ###
    print("answer:")
    print(answer)
    print()
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
