import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from transformers import AutoTokenizer
stemmer = PorterStemmer()

def nltk_tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

 
def bag_of_words(sentence, words):
    sentence_words = [stem(word) for word in sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag