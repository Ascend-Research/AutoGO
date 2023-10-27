import torch
import random
import numpy as np
from params import *
import torch_geometric
from constants import *
import utils.model_utils as m_util
from model_src.model_helpers import BookKeeper
from model_src.comp_graph.tf_comp_graph import OP2I
from model_src.comp_graph.tf_comp_graph_models import make_cg_regressor
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from model_src.comp_graph.tf_comp_graph_dataloaders import CGRegressDataLoader
from utils.model_utils import set_random_seed, device, add_weight_decay, get_activ_by_name
from model_src.predictor.model_perf_predictor import train_predictor
import time
from model_src.demo_functions import get_reg_truth_and_preds, pure_regressor_metrics, correlation_metrics


def prepare_local_params(parser, ext_args=None):
    parser.add_argument("-model_name", required=False, type=str,
                        default="GNN_predictor")
    parser.add_argument("-data_families", required=False, type=str,
                        default="hiaml+two_path+nb201c10+nb101_5k+inception"
                        )
    parser.add_argument("-dev_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-test_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-epochs", required=False, type=int,
                        default=40)
    parser.add_argument("-fine_tune_epochs", required=False, type=int,
                        default=0)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=32)
    parser.add_argument("-initial_lr", required=False, type=float,
                        default=0.0001)
    parser.add_argument("-in_channels", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-hidden_size", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-out_channels", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-num_layers", help="", type=int,
                        default=6, required=False)
    parser.add_argument("-dropout_prob", help="", type=float,
                        default=0.0, required=False)
    parser.add_argument("-aggr_method", required=False, type=str,
                        default="mean")
    parser.add_argument("-gnn_activ", required=False, type=str,
                        default="tanh")
    parser.add_argument("-reg_activ", required=False, type=str,
                        default=None)
    parser.add_argument("-normalize_HW_per_family", required=False, action="store_true",
                    default=False)
    parser.add_argument('-gnn_type', required=False, default="GraphConv")
    parser.add_argument('-num_seeds', type=int, default=1, required=False)

    return parser.parse_args(ext_args)


def get_family_train_size_dict(args):
    if args is None:
        return {}
    rv = {}
    for arg in args:
        if "#" in arg:
            fam, size = arg.split("#")
        else:
            fam = arg
            size = 0
        rv[fam] = int(float(size))
    return rv


def main(params):
    data_families = list(v for v in params.data_families.split("+") if len(v) > 0)
    data_families.sort()
    data_families_abbrv = '+'.join([abbrv_families_names[family] for family in data_families])
    
    params.model_name = f"gpi_{params.model_name}_predictor_D-{data_families_abbrv}_seed{params.seed}"
    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=float("inf"), eval_perf_comp_func=lambda old, new: new < old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)

    book_keeper.log("Params: {}".format(params), verbose=False)
    set_random_seed(params.seed, log_f=book_keeper.log)
    book_keeper.log("Train Families: {}".format(data_families))

    data_manager = FamilyDataManager(data_families, log_f=book_keeper.log, cache_dir=DATA_DIR)
    family2sets = \
        data_manager.get_regress_train_dev_test_sets(params.dev_ratio, params.test_ratio,
                                                normalize_HW_per_family=params.normalize_HW_per_family,
                                                normalize_target=False, group_by_family=True)

    train_data, dev_data = [], []
    test_data = {}
    for f, (fam_train, fam_dev, fam_test) in family2sets.items():
        train_data.extend(fam_train)
        dev_data.extend(fam_dev)
        test_data[f] = fam_test

    random.shuffle(train_data)
    random.shuffle(dev_data)
    book_keeper.log("Train size: {}".format(len(train_data)))
    book_keeper.log("Dev size: {}".format(len(dev_data)))
    test_sizes = [[f, len(test_data[f])] for f in test_data.keys()]
    book_keeper.log("Test sizes: {}".format(test_sizes))

    train_loader = CGRegressDataLoader(params.batch_size, train_data,)
    dev_loader = CGRegressDataLoader(params.batch_size, dev_data,)
    test_loader = {}
    for f in data_families:
        test_loader[f] = CGRegressDataLoader(params.batch_size, test_data[f],)

    book_keeper.log(
        "{} overlap(s) between train/dev loaders".format(train_loader.get_overlapping_data_count(dev_loader)))
    book_keeper.log("Initializing {}".format(params.model_name))


    if "GINConv" in params.gnn_type:
        def gnn_constructor(in_channels, out_channels):
            nn = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels),
                                     torch.nn.Linear(in_channels, out_channels),
                                     )
            return torch_geometric.nn.GINConv(nn=nn)
    else:
        def gnn_constructor(in_channels, out_channels):
            return eval("torch_geometric.nn.%s(%d, %d)"
                        % (params.gnn_type, in_channels, out_channels))

    model = make_cg_regressor(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=params.in_channels,
                              shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                              hidden_size=params.hidden_size, out_channels=params.out_channels,
                              gnn_constructor=gnn_constructor,
                              gnn_activ=get_activ_by_name(params.gnn_activ), n_gnn_layers=params.num_layers,
                              dropout_prob=params.dropout_prob, aggr_method=params.aggr_method,
                              regressor_activ=get_activ_by_name(params.reg_activ)).to(device())

    perf_criterion = torch.nn.MSELoss()
    model_params = add_weight_decay(model, weight_decay=0.)
    optimizer = torch.optim.Adam(model_params, lr=params.initial_lr)

    book_keeper.log(model)
    book_keeper.log("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    book_keeper.log("Number of trainable parameters: {}".format(n_params))

    reg_metrics = ["MSE", "MAE", "MAPE"]

    def _batch_fwd_func(_model, _batch):
        regular_node_inds = _batch[DK_BATCH_CG_REGULAR_IDX]
        regular_node_shapes = _batch[DK_BATCH_CG_REGULAR_SHAPES]
        weighted_node_inds = _batch[DK_BATCH_CG_WEIGHTED_IDX]
        weighted_node_shapes = _batch[DK_BATCH_CG_WEIGHTED_SHAPES]
        weighted_node_kernels = _batch[DK_BATCH_CG_WEIGHTED_KERNELS]
        weighted_node_bias = _batch[DK_BATCH_CG_WEIGHTED_BIAS]
        edge_tsr_list = _batch[DK_BATCH_EDGE_TSR_LIST]
        batch_last_node_idx_list = _batch[DK_BATCH_LAST_NODE_IDX_LIST]

        return _model(regular_node_inds, regular_node_shapes, weighted_node_inds, weighted_node_shapes,
                      weighted_node_kernels, weighted_node_bias, edge_tsr_list, batch_last_node_idx_list,
                      ext_feat=[0, 0])

    book_keeper.log("Training for {} epochs".format(params.epochs))
    start = time.time()
    try:
        train_predictor(_batch_fwd_func, model, train_loader, perf_criterion, optimizer, book_keeper,
                        num_epochs=params.epochs, max_gradient_norm=params.max_gradient_norm,
                        dev_loader=dev_loader)
    except KeyboardInterrupt:
        book_keeper.log("Training interrupted")

    book_keeper.report_curr_best()
    book_keeper.load_model_checkpoint(model, allow_silent_fail=True, skip_eval_perfs=True,
                                      checkpoint_file=P_SEP.join([book_keeper.saved_models_dir,
                                                                  params.model_name + "_best.pt"]))
    end = time.time()
    with torch.no_grad():
        model.eval()
        book_keeper.log("===============Overall Test===============")
        for f in data_families:
            book_keeper.log(f"Family: {f}")

            test_lab_mu, test_pred_mu = get_reg_truth_and_preds(model, test_loader[f], _batch_fwd_func)
            test_reg_mu = pure_regressor_metrics(test_lab_mu, test_pred_mu)
            for i, metric in enumerate(reg_metrics):
                book_keeper.log("Test {}: {}".format(metric, test_reg_mu[i]))
                if metric is "MAE":
                    metric_list = ["Train Mean MAE"]
                    results_list = [test_reg_mu[i]]

            [overall_sp_mu] = correlation_metrics(test_lab_mu, test_pred_mu, pearson=False)
            book_keeper.log("Test Mean Spearman Rho: {}".format(overall_sp_mu))
        
        book_keeper.log("Total time: %s" % (end - start))
        metric_list.append("Mean SRCC")
        results_list.append(overall_sp_mu)

    return metric_list, results_list


if __name__ == "__main__":
    _parser = prepare_global_params()
    params = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = params.device_str

    if params.num_seeds == 1:
        main(params)
    else:
        original_model_name = params.model_name
        book_keeper = BookKeeper(log_file_name=original_model_name + "_acc_allseeds.txt",
                                 model_name=params.model_name,
                                 logs_dir=params.logs_dir)

        book_keeper.log("Params: {}".format(params), verbose=False)
        metrics_dict = {'Mean': np.mean,
                        'S.Dev': np.std,
                        'Max': np.max,
                        'Min': np.min}

        all_results = []
        for i in range(params.num_seeds):
            params.seed = SEEDS_RAW[i % len(SEEDS_RAW)]
            if params.num_seeds > len(SEEDS_RAW):
                params.seed += i
            params.model_name = original_model_name
            metric_list, result_list = main(params)
            all_results.append(result_list)

        result_mat = np.matrix(all_results)
        banner_msg = ", ".join(metric_list)

        for i, metric in enumerate(metric_list):
            book_keeper.log(metric)
            for measure in metrics_dict.keys():
                computed_metric = metrics_dict[measure](result_mat[:, i]).squeeze()
                book_keeper.log("%s: %.6f" % (measure, computed_metric))
