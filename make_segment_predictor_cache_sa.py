from params import CACHE_DIR, prepare_global_params
import numpy as np
import pickle
from tqdm import tqdm
import os
from constants import abbrv_families_names
import numpy as np


def main(params):
    print(params)

    num_samples = 'all' if params.num_samples=='all' else float(params.num_samples)
    suffix = 'ns'+params.num_samples

    encoder_families = list(v for v in set(params.encoder_families.split("+")) if len(v) > 0)
    encoder_families.sort()
    abbrv_families_str = '+'.join([abbrv_families_names[family] for family in encoder_families])
    data_family = params.data_family

    np.random.seed(params.seed)

    cache_file = params.cache if params.cache is not None \
        else os.path.join('cache', \
            f"gpi_comp_graph_cache_E-{abbrv_families_str}_D-{abbrv_families_names[data_family]}_{params.encoding}_{params.vocab_size}.pkl")
    cache_file = "../automutate_dev/" + cache_file
    print("Loading", cache_file)
    with open(cache_file, "rb") as f:
        raw_cache = pickle.load(f)
    print("Loading is done")

    cgs_meta = raw_cache.pop()
    new_cache = []

    for seg_key, seg_val in tqdm(cgs_meta.items(), total= len(cgs_meta), desc="Segments"):
        seg_val_selected = seg_val
        if num_samples != 'all':
            num_selected_samples = num_samples
            if num_samples < 1.:
                num_selected_samples = np.ceil(num_samples * len(seg_val))
            seg_val_selected = np.random.permutation(seg_val)[:min(len(seg_val), int(num_selected_samples))]
        for seg_val_idx, [anchor_idx, _] in enumerate(seg_val_selected):
            anchor_cg = raw_cache[anchor_idx]

            for seg_idx in range(len(anchor_cg["segments"][seg_key]["predecessor_cg"])):
                new_dict = {
                    "acc_mean": anchor_cg['acc'], 
                    "predecessor_cg": anchor_cg["segments"][seg_key]["predecessor_cg"][seg_idx], 
                    "segment_cg": anchor_cg["segments"][seg_key]["segment_cg"][seg_idx], 
                    "successor_cg": anchor_cg["segments"][seg_key]["successor_cg"][seg_idx], 
                }
                new_cache.append(new_dict)
    print("number of instances:", len(new_cache))

    suffix = 'ns'+params.num_samples
    out_file = os.path.join(CACHE_DIR, \
        f"gpi_comp_graph_cache_E-{abbrv_families_str}_D-{abbrv_families_names[data_family]}_{params.encoding}_{params.vocab_size}_s_{suffix}.pkl") # {params.char_type}
    print("Saving cache to", out_file)
    with open(out_file, "wb") as f:
        pickle.dump(new_cache, f, protocol=4)
    print("Done")


def prepare_local_params(parser):
    parser.add_argument("-encoder_families", type=str, required=False, default="hiaml+two_path+nb201+nb101_5k+inception", help="separate families by +")
    parser.add_argument("-data_family", type=str, required=True, default=None, help="'nb201c10'")
    parser.add_argument("-encoding", type=str, default="shp")
    parser.add_argument("-vocab_size", type=int, default=2000)
    parser.add_argument("-num_samples", type=str, default='all')
    parser.add_argument(
        "-cache", help="name of the raw input cache", type=str, default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    _parser = prepare_global_params()
    params = prepare_local_params(_parser)
    main(params)
