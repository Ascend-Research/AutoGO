import pickle
import sentencepiece as spm 
from params import *
from encoder.network_encoder import Network_Encoder
from constants import abbrv_families_names


def create_encoder_for_families(families=["hiaml", "two_path"], encoding="op"):

    abbrv_families_str = '+'.join([abbrv_families_names[family] for family in families])
    
    net_encoder = Network_Encoder(families=families, encoding=encoding)

    graph_tokens, graph_tokens_str = net_encoder.encode_all_graphs()

    path = P_SEP.join([CACHE_SPM_DIR, abbrv_families_str])
    if not os.path.isdir(path):
        os.makedirs(path)
    print('writing graph tokens to {}_graph_tokens_{}.txt'.format(abbrv_families_str, encoding))
    with open(
        P_SEP.join([path, "{}_graph_tokens_{}.txt".format(abbrv_families_str, encoding)]),
        "wb",
    ) as f:
        f.write(graph_tokens_str.encode(encoding="utf8"))
    print('pickling the encoder to {}_encoder_{}.pkl'.format(abbrv_families_str, encoding))

    with open(
        P_SEP.join([path, "{}_encoder_{}.pkl".format(abbrv_families_str, encoding)]),
        "wb",
    ) as f:
        pickle.dump(net_encoder, f)
    return net_encoder


def load_and_test_encoder(
    families=["hiaml", "two_path"], vocab_size=500, sub_str=False, model_type="bpe", encoding="op"):
    
    abbrv_families_str = '+'.join([abbrv_families_names[family] for family in families])
    path = P_SEP.join([CACHE_SPM_DIR, abbrv_families_str])

    print("loading the network encoder at {}...".format(path))
    with open(
        P_SEP.join([path, "{}_encoder_{}.pkl".format(abbrv_families_str, encoding)]),
        "rb",
    ) as f:
        nt = pickle.load(f)
    print(
        "nt loaded with {} graphs and encoding vocab size of {}".format(
            abbrv_families_str, len(nt.op2vocab)
        )
    )

    print("loading the graph tokens txt at {}...".format(path))
    if sub_str:
        print("the file includes sub-strings")
        input_text = P_SEP.join(
            [path, "{}_100k_sub_strs_{}.txt".format(abbrv_families_str, encoding)]
        )
        with open(input_text, "rb") as f:
            gt_utf8 = f.read()

    else:
        input_text = P_SEP.join(
            [path, "{}_graph_tokens_{}.txt".format(abbrv_families_str, encoding)]
        )

    model_name = "{}_vsize{}_{}{}_{}".format(
        abbrv_families_str, vocab_size, model_type, "_sub-str" if sub_str else "", encoding
    )

    if not os.path.isdir(P_SEP.join([path, "models"])):
        os.makedirs(P_SEP.join([path, "models"]))
    if os.path.isfile(P_SEP.join([path, "models", model_name+'.model'])):
        print('found the model in cache, loading {}'.format(model_name))
    else:
        print("training the model and save to {}".format(model_name))
        spm.SentencePieceTrainer.train(
            "--input={} --model_prefix={} --vocab_size={} --model_type={} --character_coverage=1.0 --split_by_unicode_script=True --split_by_number=False --split_by_whitespace=False --add_dummy_prefix=False --required_chars={}".format(
                input_text,
                P_SEP.join([path, "models", model_name]),
                vocab_size,
                model_type,
                "".join(list(nt.vocab2op.keys()))[0],
            )
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load("{}.model".format(P_SEP.join([path, "models", model_name])))

    vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    print("segment vocab size:", len(vocabs))
    return sp, nt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-families", type=str, default="hiaml+two_path")
    parser.add_argument("-g_family", type=str, default="")
    parser.add_argument("-vocab_size", type=int, default=500)
    parser.add_argument("-model_type", type=str, default="bpe")
    parser.add_argument("-encoding", type=str, default="op")
    parser.add_argument("-remap", required=False, action="store_true", default=False)
    parser.add_argument("-sub_str", required=False, action="store_true", default=False)
    args = parser.parse_args()
    print(args.__dict__)

    families = list(v for v in set(args.families.split("+")) if len(v) > 0)
    families.sort()

    if args.remap:
        create_encoder_for_families(families, encoding=args.encoding)

    sp, nt = load_and_test_encoder(
        families,
        vocab_size=args.vocab_size,
        sub_str=args.sub_str,
        model_type=args.model_type,
        encoding=args.encoding,
    )
