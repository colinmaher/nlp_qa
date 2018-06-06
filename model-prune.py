import nltk
import gensim
from nltk.corpus import brown

if __name__ == "__main__":

    #get google word2vec and prune using brown corpus
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews_vectors_negative300.bin', binary=True)
    words = set(brown.words())
    out_file = 'pruned_word2vec.txt'
    f = open(out_file, 'w')
    word_presented = words.intersection(model.vocab.keys())
    f.write('{} {}\n'.format(len(word_presented),len(model['word'])))

    for word in word_presented:
        f.write('{} {}\n'.format(word, ' '.join(str(value) for value in model[word])))

    f.close()