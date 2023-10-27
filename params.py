import os
import argparse
from os.path import sep as P_SEP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = P_SEP.join([BASE_DIR, "cache"])
CACHE_MOTIFS_DIR = P_SEP.join([BASE_DIR, "cache_motifs"])
CACHE_SPM_DIR = P_SEP.join([BASE_DIR, "cache_sentence_piece"])
PLOT_DIR = P_SEP.join([BASE_DIR, "plots"])
DATA_DIR = P_SEP.join([BASE_DIR, "data"])
NAS_MACRO_DATA_DIR = P_SEP.join([BASE_DIR, "nasbench_macro", "data"])
SAVED_MODELS_DIR = P_SEP.join([BASE_DIR, "saved_models"])
LOGS_DIR = P_SEP.join([BASE_DIR, "logs"])
# CACHE_DIR_MOTIFS = P_SEP.join([ROOT_DIR, "cache"])


if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
# if not os.path.exists(CACHE_DIR_MOTIFS): os.makedirs(CACHE_DIR_MOTIFS)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(SAVED_MODELS_DIR): os.makedirs(SAVED_MODELS_DIR)
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)


def prepare_global_params():

    parser = argparse.ArgumentParser()

    parser.add_argument("-device_str", type=str, required=False,
                        default=None)
    parser.add_argument("-seed", required=False, type=int,
                        default=12345)
    parser.add_argument("-max_gradient_norm", required=False, type=float,
                        default=5.0)
    parser.add_argument("-word_delim", required=False, type=str,
                        default=" ")
    parser.add_argument("-logs_dir", required=False, type=str,
                        default=LOGS_DIR)
    parser.add_argument("-saved_models_dir", required=False, type=str,
                        default=SAVED_MODELS_DIR)
    parser.add_argument("-saved_model_file", required=False, type=str,
                        default=P_SEP.join([SAVED_MODELS_DIR, "default_model.pt"]))
    parser.add_argument("-run_test_only", action="store_true", required=False,
                        default=False)
    parser.add_argument("-allow_parallel", required=False, action="store_true",
                        default=False)
    parser.add_argument("-num_workers", required=False, type=int,
                        default=2)
    parser.add_argument("-disable_checkpoint", action="store_true", required=False,
                        default=False)

    return parser
