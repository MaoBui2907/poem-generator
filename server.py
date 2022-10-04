from flask.templating import render_template
import torch
import argparse
import numpy as np
from lib import data_utils, vnlp
from model import PoemGeneratorLightning
from flask import Flask, make_response, request, jsonify

parser = argparse.ArgumentParser()

parser.add_argument("--config", default="/data/vectorized/config.ini")
parser.add_argument("--chkp", default="lightning_logs/version_10/checkpoints/epoch=1-step=2508.ckpt")

args = parser.parse_args()

config = data_utils.load_config(args.config)

embed_size = config.getint('DEFAULT', 'WORD_SIZE') + 1
word_size = config.getint('DEFAULT', 'bow_size')
hid_size = 500
seq_len = config.getint('DEFAULT', 'seq_len')
window_size = config.getint('DEFAULT', 'window_size')

chkp_path = args.chkp

nlp = vnlp.VNlp('data/raws/wiki.vi.bin', embed_size - 1)
corpus = data_utils.unpickle_file("/data/vectorized/dict.pkl")

model = PoemGeneratorLightning.load_from_checkpoint(chkp_path, strict=True, word_size=word_size, embed_size=embed_size, hid_size=hid_size, window_size=window_size, seq_len=seq_len)
model.eval()
model.freeze()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    inp_ = np.concatenate([[corpus.vectors[corpus.word2idx[i]] for i in ([data_utils.SPECIAL_CONTROLS[2]] * (window_size - 2) + [data_utils.SPECIAL_CONTROLS[0]])], [np.append(nlp.to_vector(data.get('start_text')), 0.1)]])
    inp_ = torch.FloatTensor(inp_)
    inp_ = inp_.reshape(1, window_size, -1)

    sentiments = data.get("keywords")
    sent = torch.FloatTensor(nlp.combined_vector(sentiments))
    sent = sent.reshape(1, -1)

    outputs = []
    for i in range(seq_len):
        out = model(inp_, sent)
        out_ = out.squeeze().exp()
        out_ = torch.multinomial(out_, 1)[0]
        # out_ = torch.argmax(out_)
        tok = corpus.idx2word[int(out_)]
        outputs.append(tok)
        inp_ = inp_.reshape(window_size, -1)
        inp_ = torch.FloatTensor(torch.cat([inp_[1:], torch.FloatTensor([corpus.vectors[int(out_)]])]))
        inp_ = inp_.reshape(1, window_size, -1)
    print(len(outputs))
    print(" ".join(outputs))
    return make_response(jsonify({'result': ' '.join(outputs)}), 200)

app.run("0.0.0.0", 1998)