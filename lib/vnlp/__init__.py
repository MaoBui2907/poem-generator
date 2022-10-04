import numpy as np
import fasttext
import fasttext.util

class VNlp:
    def __init__(self, model, dim=300):
        self.ft = fasttext.load_model(model)
        fasttext.util.reduce_model(self.ft, dim)
        self.word_dimension = dim

    def to_vector(self, word: str):
        control_toks = {
            "<SOS>": np.asfarray([0]*self.word_dimension),
            "<EOS>": np.asfarray([0]*self.word_dimension),
            "<PAD>": np.asfarray([0]*self.word_dimension),
            "<BRK>": np.asfarray([0]*self.word_dimension),
        }
        word = self.normalize(word)
        if word in control_toks.keys():
            return control_toks[word]
        else:
            return self.ft.get_word_vector(word)

    @staticmethod
    def normalize(word: str):
        w = " ".join(word.split())
        return w.lower()

    def combined_vector(self, words: list):
        vecs = []
        for w in words:
            vecs.append(self.to_vector(w))
        vecs = np.asfarray(vecs)
        return np.mean(vecs, axis=0)

    @staticmethod
    def generate_bow_vector(word_size, index):
        res = [0] * word_size
        res[index] = 1
        return np.array(res, dtype=np.float64)