from motifs.motif_structures.utils import cg_to_nx
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from params import prepare_global_params
from test_encoder import load_and_test_encoder
from encoder.utils import get_psc_subgraphs_dict
import pickle
from tqdm import tqdm

from constants import abbrv_families_names
from params import DATA_DIR


"""
!!IMPORTANT!!
DO NOT SHUFFLE THE CACHES, the seg_meta_dict depends on the indices of the caches to be constant
THE LAST ENTRY IN THE OUTPUT PKL IS THE METADATA CACHE!
"""

def prepare_local_params(parser):
    parser.add_argument("-encoder_families", type=str, required=False, default="hiaml+two_path+nb201+nb101_5k+inception", help="separate families by +")
    parser.add_argument("-encoding", type=str, default="shp")
    parser.add_argument("-vocab_size", type=int, default=2000)
    parser.add_argument("-data_family", type=str, required=True, default=None, help="'nb201c10'")
    return parser.parse_args()


def main(params):
    encoder_families = list(v for v in set(params.encoder_families.split("+")) if len(v) > 0)
    encoder_families.sort()

    encoding = params.encoding
    vocab_size = params.vocab_size
    data_family = params.data_family

    sub_str = False
    model_type = "bpe"

    print("Params:", encoder_families, data_family, encoding, vocab_size)

    sp, nt = load_and_test_encoder(families=encoder_families, vocab_size=vocab_size, sub_str=sub_str, model_type=model_type, encoding=encoding)
    fdm = FamilyDataManager(families=(data_family,), cache_dir=DATA_DIR)
    cgs, _, _ = fdm.get_src_train_dev_test_sets(0, 0)

    new_cgs = []
    seg_meta_dict = {}
    for i, cg in tqdm(enumerate(cgs), total=len(cgs), desc=f"Adding segments to {data_family}"):

        cg_nx_list = cg_to_nx(cg['compute graph'], family=data_family, relu6=False, keep_name=True)
        subgraph_dict = get_psc_subgraphs_dict(cg_nx_list, nt, sp, data_family)
        new_cgs.append({**cg, "segments": subgraph_dict})

        segs_count = {}
        for seg_token in subgraph_dict.keys():
            segs_count[seg_token] = len(subgraph_dict[seg_token]['predecessor_cg'])

        for seg_token, seg_count in segs_count.items():
            if seg_token not in seg_meta_dict.keys():
                seg_meta_dict[seg_token] = [[i, seg_count]]
            else:
                seg_meta_dict[seg_token].append([i, seg_count])

    new_cgs.append(seg_meta_dict)

    filename = f"cache/gpi_comp_graph_cache_E-{'+'.join([abbrv_families_names[family] for family in encoder_families])}_D-{abbrv_families_names[data_family]}_{encoding}_{vocab_size}.pkl" # {params.char_type}
    with open(filename, 'wb') as f:
        pickle.dump(new_cgs, f)
    print("Saved to", filename)

if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    main(_args)