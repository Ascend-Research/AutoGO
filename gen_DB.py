import codecs
import pickle
import time
from collections import defaultdict
from os.path import sep as P_SEP

import pandas as pd
import sentencepiece as spm
from constants import CG_NAMES as cg_names
from encoder.utils import get_segment2subgraph
from motifs.motif_structures.utils import *
from motifs.motif_structures.utils import load_nx_graphs_from_cg_name
from params import *


family = 'two_path'
vsize= 500
sub_str= False
encoding = "res_ratio"
model_type = "bpe"

with open("cache_sentence_piece/h+i+n15+n2+t/h+i+n15+n2+t_encoder_shp.pkl", "rb") as f:
    nt = pickle.load(f)


sp = spm.SentencePieceProcessor()
sp.load("cache_sentence_piece/h+i+n15+n2+t/models/h+i+n15+n2+t_vsize2000_bpe_shp.model")

family_graphs = []
for family in ["hiaml", "inception", "two_path", "nb101", "nb201"]:
    family_graphs += load_nx_graphs_from_cg_name(cg_names[family])

s_time = time.time()
segment2subgraph = get_segment2subgraph(family_graphs, nt, sp, family,
                                        encoding=encoding, sub_str=sub_str, model_type=model_type, save=False)

t_time = time.time()

print('{} sec took for db generation'.format(t_time-s_time))

with codecs.open("cache_sentence_piece/h+i+n15+n2+t/combined_segment2subgraph_res_ratio.pkl", 'wb+') as f:
    pickle.dump(segment2subgraph, f)

op_names, res_in, res_out, kernel, padding, stride, properties, lats, stds = [], [], [], [], [], [], [], [], []
cnames = ['op', 'input', 'output', 'kernel', 'padding', 'stride', 'properties', 'lats', 'stds']
lat_dict = defaultdict(lambda: [])

def get_in_res_str(res):
    return str([1, res[0], res[2], res[4]])
def get_out_res_str(res):
    return str([1, res[1], res[3], res[5]])

seg_df = pd.DataFrame(columns=['seg_str', 'num_inputs', 'num_outputs', 'res_in', 'res_out', 'subgraph', 'lat'])
seg_strs, num_inputs, num_outputs, res_in, res_out, lats, subgraphs = [], [], [], [], [], [], []
for seg, _subgraphs in segment2subgraph.items():
    nan = False
    for subgraph in _subgraphs:
        seg_strs.append(seg)
        in_nodes = [n for i, n in subgraph.nodes(data=True) if subgraph.in_degree(i) == 0]
        out_nodes = [n for i, n in subgraph.nodes(data=True) if subgraph.out_degree(i) == 0]
        num_inputs.append(len(in_nodes))
        num_outputs.append(len(out_nodes))
        _res_in = [n['resolution'] for n in in_nodes]
        _res_out = [n['resolution'] for n in out_nodes]
        res_in.append(str(_res_in))
        res_out.append(str(_res_out))
        subgraphs.append(subgraph)
        lats.append(0)

seg_df['seg_str'] = seg_strs
seg_df['num_inputs'] = num_inputs
seg_df['num_outputs'] = num_outputs
seg_df['res_in'] = res_in
seg_df['res_out'] = res_out
seg_df['subgraph'] = subgraphs
seg_df['lat'] = lats


with codecs.open("cache_sentence_piece/h+i+n15+n2+t/combined_segment_DB_res_ratio.pkl", 'wb+') as f:
    pickle.dump(seg_df, f)
