from math import fabs
import re
import json
from tqdm import tqdm

from lib.data_utils import SPECIAL_CONTROLS

data_path = './raws/simp_input.txt'
output_path = './raws/output.json'
prepared_data = {
    "version": '1.0',
    "number": 0,
    "poems": []
}
blacklist_token = "[\W_]+"
with open(data_path, 'r', encoding='utf8') as data:
    raw = data.read()
    poems = raw.lower().split("\n\n\n")
    for block in tqdm(poems):
        lines = block.split("\n")
        poem_toks = [SPECIAL_CONTROLS[0]]
        for l in lines:
            l_norm = re.sub(blacklist_token, " ", l)
            l_norm = re.sub("\s+", " ", l_norm)
            l_toks = list(filter(lambda tok: tok, l_norm.split(" "))) + [SPECIAL_CONTROLS[3]]
            poem_toks.extend(l_toks)
        poem_toks[-1] = SPECIAL_CONTROLS[1]
        prepared_data["poems"].append({
            "sentiments": [],
            "tokens": poem_toks
        })
        prepared_data["number"] = len(prepared_data["poems"])
    with open(output_path, 'w', encoding='utf8') as output:
        json.dump(prepared_data, output, ensure_ascii=False)