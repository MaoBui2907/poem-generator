import json
from tqdm import tqdm
import torch
import numpy as np

from lib import dictionary, vnlp, data_utils
from lib.data_utils import SPECIAL_CONTROLS

data_path = 'raws/output.json'
output_path = 'raws/toks.json'
fasttext_bin_path = '../raws/wiki.vi.bin'
poem_tok_number = 32    # = 7 * 4 + 3 + 2 - 1
window_size = 14
word_dimension = 300
vectorized_path = 'vectorized/data.json'
inp_vec_path = '/data/simple_data/vectorized/inp.pkl'
out_vec_path = '/data/simple_data/vectorized/out.pkl'
sen_vec_path = '/data/simple_data/vectorized/sen.pkl'
dictionary_path = '/data/simple_data/vectorized/dict.pkl'
config_path = '/data/simple_data/vectorized/config.ini'

nlp = vnlp.VNlp(fasttext_bin_path, dim=word_dimension)
word_dict = dictionary.Dictionary(word_dimension + 1)   # +1 for special tokens

for idx, tok in enumerate(SPECIAL_CONTROLS):
    word_dict.add_word(tok, [0] * word_dimension + [idx + 1])

with open(data_path, 'r', encoding='utf8') as data:
    json_data = json.load(data)
    poems = json_data["poems"]
    inp_toks = []
    inp_vecs = []
    out_toks = []
    out_vecs = []
    sen_toks = []
    sen_vecs = []
    for poem in tqdm(poems):
        temp_inp_tok = []
        for tok in poem["tokens"][:-1]:
            tok_vector = nlp.to_vector(tok)
            tok_idx = word_dict.add_word(tok, np.append(tok_vector, 0.1))
            temp_inp_tok.append(tok_idx)
            inp_tok = [word_dict.word2idx.get(SPECIAL_CONTROLS[2])] * (window_size - len(temp_inp_tok[-window_size:])) + temp_inp_tok[-window_size:]
            inp_toks.append(inp_tok)
            inp_vecs.append([word_dict.vectors[tok] for tok in inp_tok])
            out_toks.append(tok_idx)
        del out_toks[0]
        out_toks.append(word_dict.word2idx.get(SPECIAL_CONTROLS[1]))
    out_vecs = [nlp.generate_bow_vector(len(word_dict), tok) for tok in out_toks]
    with open(vectorized_path, 'w') as output:
        json.dump({
            "INPUT": inp_toks,
            "OUTPUT": out_toks,
        }, output)
    data_utils.pickle_data(inp_vec_path, np.array(inp_vecs, dtype=np.float64))
    data_utils.pickle_data(out_vec_path, np.array(out_vecs, dtype=np.float64).reshape((-1, len(word_dict))))
    data_utils.pickle_data(dictionary_path, word_dict)
    config = {
        "BOW_SIZE": len(word_dict),
        "SEQ_LEN": poem_tok_number,
        "WINDOW_SIZE": window_size,
        "WORD_SIZE": word_dimension,
        "INP_SIZE": torch.FloatTensor(np.array(inp_vecs)).size(),
        "OUT_SIZE": torch.FloatTensor(np.array(out_vecs)).reshape(-1, len(word_dict)).size(),
    }
    data_utils.save_config(config_path, config)
    