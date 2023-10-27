#!/usr/bin/env python
"""
Main file of automutate
"""

import argparse
import os
import pickle
import sys
from argparse import PARSER
from collections import OrderedDict
from datetime import datetime
from functools import partial

import networkx as nx
import matplotlib.colors as plt_colors

import numpy as np
import pandas as pd
import torch
import torch_geometric
# Import some constants
from constants import (DK_BATCH_CG_REGULAR_IDX, DK_BATCH_CG_REGULAR_SHAPES,
                       DK_BATCH_CG_WEIGHTED_BIAS, DK_BATCH_CG_WEIGHTED_IDX,
                       DK_BATCH_CG_WEIGHTED_KERNELS,
                       DK_BATCH_CG_WEIGHTED_SHAPES, DK_BATCH_EDGE_TSR_LIST,
                       DK_BATCH_LAST_NODE_IDX_LIST, DK_BATCH_REG_NODE_OFFSET,
                       DK_BATCH_REGULAR_OFFSETS, DK_BATCH_WEIGHTED_OFFSETS)
from params import P_SEP
from model_src.comp_graph.tf_comp_graph import OP2I, ComputeGraph
from model_src.comp_graph.tf_comp_graph_dataloaders import CGRegressDataLoader
from model_src.comp_graph.tf_comp_graph_models import (make_psc_regressor,
                                                       make_cg_regressor)
from model_src.comp_graph.tf_comp_graph_trainer import cifar10_train_test
from model_src.comp_graph.tf_comp_graph_utils import compute_cg_flops, correct_cg_for_downstream
from model_src.autogo_search.auto_mutate import (AccDiffChecker, FlopsDiffChecker,
                                                 RunManager)
from model_src.predictor.gpi_family_data_manager import get_domain_configs
from motifs.motif_structures.motif_block import Mutator
from utils.model_utils import (device, get_activ_by_name, model_load,
                               set_random_seed)
from motifs.motif_structures.utils import cg_to_nx
from test_encoder import load_and_test_encoder
from motifs.motif_structures.motif_process import highlight_node_groups
from encoder.utils import get_node_groups
from encoder.utils import inherit_graph_attributes, get_context_edges


domain_configs = get_domain_configs()


def add_general_params(_parser: PARSER) -> None:
    """
    Add general parameters to the parser
    """
    _parser.add_argument("-model_path", required=True, type=str,
                         help='Path to the model file, e.g., hiaml_best.pkl')
    _parser.add_argument("-model_family", required=True, type=str,
                         help='Family name of the input model, e.g., nb201')
    _parser.add_argument("-graph_encoder_path", required=False, type=str, 
                         default="cache_sentence_piece/h+i+n15+n2+t/h+i+n15+n2+t_encoder_shp.pkl",
                         help='Path to the graph encoder')
    _parser.add_argument("-tokenizer_path", required=False, type=str,
                         default="cache_sentence_piece/h+i+n15+n2+t/models/h+i+n15+n2+t_vsize2000_bpe_shp.model",
                         help='Path to the segment tokenizer')
    _parser.add_argument("-segment_database_path", required=False, type=str,
                         default="cache_sentence_piece/h+i+n15+n2+t/combined_segment_DB_res_ratio.pkl",
                         help='Path to the database of segments')
    _parser.add_argument("-graph_encoder_family", required=False, type=str,
                         default="combined",
                         help='Family name of the graph encoder, e.g., nb201')
    _parser.add_argument("-predictor_path", required=True, type=str,
                         help='Path to the predictor')
    _parser.add_argument("-predictor_type", required=True, type=str,
                         help='Type of the predictor')
    _parser.add_argument("-seed", required=False, default=0, type=int,
                         help='Random seed')
    _parser.add_argument("-output_dir", required=False, type=str, default="outputs/",
                         help='Output directory')
    _parser.add_argument("-logs_dir", required=False, type=str, default="outputs/",
                         help='Log directory')


def add_mutation_params(_parser):
    """
    Add parameters related to the mutation to the parser
    """
    _parser.add_argument("-input_h", required=True, type=int, default=32,
                         help='Height of the model input')
    _parser.add_argument("-input_w", required=True, type=int, default=32,
                         help='Width of the model input')
    _parser.add_argument("-input_c", required=True, type=int, default=3,
                         help='Channel of the model input')
    _parser.add_argument("-top_k", required=False, type=int, default=10,
                         help="The number of top architectures we carried over to the next iteration")
    _parser.add_argument("-epoch", required=False, type=int, default=5,
                         help="The number of epochs we run")
    _parser.add_argument("-n_jobs", required=False, type=int, default=1,
                         help="Multithreading jobs; should not exceed -top_k")
    _parser.add_argument("-max_candidate_per_iter", required=False, type=int, default=10,
                         help="The number of candidate we consider in each search iteration")
    _parser.add_argument("-max_target_per_iter", required=False, type=int, default=100,
                         help="The number of candidate we consider in each search iteration")
    _parser.add_argument("-propagation_type", default='milp', type=str,
                         help="Type of the virtual propogation")
    _parser.add_argument("-mutation_unit", default='segment', type=str,
                         help="Unit of mutation")
    _parser.add_argument("-max_visited_architecture", default=sys.maxsize, type=int,
                         help="Maximum amount of visited architecture")
    _parser.add_argument("-iteration_min_percent", default=0., type=float,
                         help="percentage of least edit distance (#iterations) for a CG to be trained from scratch")
    _parser.add_argument("-num_consecutive_segs", default=3, type=int,
                         help="up to # consecutive segments to be selected as source")
    _parser.add_argument("-segment_sparsity", default=0.5, type=float,
                         help="percent of multi-node segments to prune")

def add_predictor_params(_parser):
    """
    Add parameters related to the GNN predictor
    """
    _parser.add_argument("-predictor_out_embed_size", required=False, default=32, type=int,
                         help="")
    _parser.add_argument("-predictor_shape_embed_size", required=False, default=8, type=int,
                         help="")
    _parser.add_argument("-predictor_kernel_embed_size", required=False, default=8, type=int,
                         help="")
    _parser.add_argument("-predictor_n_unique_kernels", required=False, default=8, type=int,
                         help="")
    _parser.add_argument("-predictor_n_shape_vals", required=False, default=6, type=int,
                         help="")
    _parser.add_argument("-predictor_hidden_size", required=False, default=32, type=int,
                         help="")
    _parser.add_argument("-predictor_out_channels", required=False, default=32, type=int,
                         help="")
    _parser.add_argument("-predictor_gnn_activ", required=False, default="tanh", type=str,
                         help="")
    _parser.add_argument("-predictor_n_gnn_layers", required=False, default=6, type=int,
                         help="")
    _parser.add_argument("-predictor_aggr_method", required=False, default="mean", type=str,
                         help="")
    _parser.add_argument("-predictor_dropout_prob", required=False, default=0., type=float,
                         help="")


def add_flops_params(_parser):
    """
    Add parameters related to the latency checking
    """
    _parser.add_argument("-min_flops_decrease_percent", required=False, default=0, type=int,
                         help="Minimum decrease in the flops after mutation")
    _parser.add_argument("-max_flops_decrease_percent", required=False, default=20, type=int,
                         help="Maximum decrease in the flops after mutation")


def add_training_params(_parser):
    """
    Add parameters related to the model training.
    """
    _parser.add_argument("-train_type", required=False, default="TensorFlow", type=str,
                         help="Only TF supported for code submission")
    _parser.add_argument("-train_dataset", required=False, default="CIFAR10", type=str,
                         help="The dataset to perform training")
    _parser.add_argument("-num_train_cgs", required=False, default=0, type=int,
                         help="Number of mutated CG to be trained. For example, if this parameter is 5, we train"
                              "the top 5 CGs after mutation")
    _parser.add_argument("-num_runs_cg", required=False, default=3, type=int,
                         help="Number of trainings for each CG. For example, if this parameter is 3, we train"
                              "each CG 3 times.")
    _parser.add_argument("-train_input_cg", required=False, action="store_true",
                         help="whether the input CG needs to be trained")
    _parser.add_argument("-train_epochs", required=False, default=200, type=int,
                         help="Number of training epochs for the mutated CG")
    _parser.add_argument("-train_batch_size", required=False, default=256, type=int,
                         help="Batch size for the training")


def cg_regress(cg, model, forward_function_id=1):
    """
    Forward function of the predictor.
    """
    correct_index_list = []
    if type(cg) is list:
        if forward_function_id == 1:
            loader = CGRegressDataLoader(64, [[*cg[i], float(i), float(i)] for i in range(len(cg))], verbose=False)
        elif forward_function_id == 2:
            loader = CGRegressDataLoader(32, [[cg[i], float(i)] for i in range(len(cg))], verbose=False)
        else:
            loader = CGRegressDataLoader(32, [[cg[i], float(i)] for i in range(len(cg))], verbose=False)
    else:
        loader = CGRegressDataLoader(1, [[cg, 1.0]], verbose=False)

    def _batch_fwd_func_1(_model, _batch):
        regular_node_inds = _batch[DK_BATCH_CG_REGULAR_IDX]
        regular_node_shapes = _batch[DK_BATCH_CG_REGULAR_SHAPES]
        weighted_node_inds = _batch[DK_BATCH_CG_WEIGHTED_IDX]
        weighted_node_shapes = _batch[DK_BATCH_CG_WEIGHTED_SHAPES]
        weighted_node_kernels = _batch[DK_BATCH_CG_WEIGHTED_KERNELS]
        weighted_node_bias = _batch[DK_BATCH_CG_WEIGHTED_BIAS]
        edge_tsr_list = _batch[DK_BATCH_EDGE_TSR_LIST]
        batch_last_node_idx_list = _batch[DK_BATCH_LAST_NODE_IDX_LIST]
        batch_reg_node_offset = _batch[DK_BATCH_REG_NODE_OFFSET]
        batch_w_offsets = _batch[DK_BATCH_WEIGHTED_OFFSETS]
        batch_r_offsets = _batch[DK_BATCH_REGULAR_OFFSETS]

        return _model(regular_node_inds, regular_node_shapes, weighted_node_inds, weighted_node_shapes,
                      weighted_node_kernels, weighted_node_bias, edge_tsr_list, batch_last_node_idx_list,
                      batch_reg_node_offset, batch_w_offsets, batch_r_offsets, ext_feat=[0, 0],
                      )

    def _batch_fwd_func_2(_model, _batch):
        nonlocal correct_index_list
        regular_node_inds = _batch[DK_BATCH_CG_REGULAR_IDX]
        regular_node_shapes = _batch[DK_BATCH_CG_REGULAR_SHAPES]
        weighted_node_inds = _batch[DK_BATCH_CG_WEIGHTED_IDX]
        weighted_node_shapes = _batch[DK_BATCH_CG_WEIGHTED_SHAPES]
        weighted_node_kernels = _batch[DK_BATCH_CG_WEIGHTED_KERNELS]
        weighted_node_bias = _batch[DK_BATCH_CG_WEIGHTED_BIAS]
        edge_tsr_list = _batch[DK_BATCH_EDGE_TSR_LIST]
        batch_last_node_idx_list = _batch[DK_BATCH_LAST_NODE_IDX_LIST]
        correct_index_list += _batch["batch_target_tensor"].long().tolist()
        return _model(regular_node_inds, regular_node_shapes, weighted_node_inds, weighted_node_shapes,
                      weighted_node_kernels, weighted_node_bias, edge_tsr_list, batch_last_node_idx_list,
                      ext_feat=[0, 0])

    preds = []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            if forward_function_id == 1:
                batch_vals = _batch_fwd_func_1(model, batch)
            elif forward_function_id == 2:
                batch_vals = _batch_fwd_func_2(model, batch)
            pred = batch_vals.squeeze(1)
            preds.extend(pred.detach().tolist())

    if len(preds) == 1:
        preds = preds[0]

    return preds


def make_gnn_constructor(gnn_type, gnn_args=None):

    def gnn_constructor(in_channels, out_channels):
        return eval("torch_geometric.nn.%s(%d, %d, %s)"
                    % (gnn_type, in_channels, out_channels, gnn_args))

    return gnn_constructor


def load_accuracy_predictor(filename, predictor_type, in_channels=32, out_embed_size=32, hidden_size=32,
                            out_channels=32,
                            n_gnn_layers=6, aggr_method="mean", gnn_activ="tanh", dropout_prob=0., shape_embed_size=8,
                            kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6):
    print(filename)

    gnn_type = "GraphConv"
    gnn_args = ""

    if predictor_type == "PSC_predictor":
        predictor = make_psc_regressor(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=in_channels,
                                       shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                                       hidden_size=hidden_size, out_channels=out_channels,
                                       gnn_constructor=make_gnn_constructor(gnn_type, gnn_args=gnn_args),
                                       return_var=False,
                                       gnn_activ=get_activ_by_name(gnn_activ), n_gnn_layers=n_gnn_layers,
                                       dropout_prob=dropout_prob, aggr_method=aggr_method,
                                       regressor_activ=None).to(device())

        sd = model_load(filename)
        try:
            predictor.load_state_dict(sd["model"], strict=True)
        except RuntimeError:
            state_dict = OrderedDict()
            for n, p in sd["model"].items():
                if "total_ops" not in n and "total_params" not in n:
                    state_dict[n] = p
            predictor.load_state_dict(state_dict, strict=True)

    elif predictor_type == "GNN":
        predictor =  make_cg_regressor(n_unique_labels=len(OP2I().build_from_file()),
                                      out_embed_size=out_embed_size,
                                      shape_embed_size=shape_embed_size,
                                      kernel_embed_size=kernel_embed_size,
                                      n_unique_kernels=n_unique_kernels,
                                      n_shape_vals=n_shape_vals,
                                      hidden_size=hidden_size,
                                      out_channels=out_channels,
                                      gnn_constructor=make_gnn_constructor(gnn_type, gnn_args=gnn_args),
                                      gnn_activ=get_activ_by_name(gnn_activ),
                                      n_gnn_layers=n_gnn_layers,
                                      dropout_prob=dropout_prob,
                                      aggr_method=aggr_method,
                                      regressor_activ=None)
        predictor.load_state_dict(torch.load(filename, map_location='cpu')['model'], strict=True)


    else:
        raise NotImplementedError("%s is not a supported predictor type" % predictor_type)

    predictor.eval()
    predictor = predictor.to(device())

    return predictor


def load_input_model(model_path, input_c, input_h, input_w):
    _, input_filetype = os.path.splitext(model_path)
    if input_filetype == ".pkl":
        with open(model_path, "rb") as f:
            input_model = pickle.load(f)
    elif input_filetype == ".pb":
        input_model = ComputeGraph(C_in=input_c,
                                   H=input_h, W=input_w,
                                   name="UserInputPB",
                                   max_hidden_size=domain_configs["max_hidden_size"],
                                   max_kernel_size=domain_configs["max_kernel_size"],
                                   max_derived_H=domain_configs["max_h"],
                                   max_derived_W=domain_configs["max_w"])
        input_model.build_from_pb(model_path, OP2I().build_from_file(), oov_threshold=0.)
    else:
        raise NotImplementedError("%s is not a supported input CG" % input_filetype)

    return input_model

def plot_mutant_changes(mutant_graph, initial_graph, nt, sp, save_dir='segments', family='', model_type='unigram', encoding='op', s2color=None, am_name=None):
    print('segmenting graph {}'.format(mutant_graph.name))
    initial_node_list, mutant_node_list = list(nx.topological_sort(initial_graph)), list(nx.topological_sort(mutant_graph))
    initial_graph_tokens, mutant_graph_tokens = nt.encode_graph(initial_graph, node_list=initial_node_list), nt.encode_graph(mutant_graph, node_list=mutant_node_list)
    initial_encoded, mutant_encoded = sp.encode_as_pieces(initial_graph_tokens), sp.encode_as_pieces(mutant_graph_tokens)
    
    sum_groups = sum([len(encoded_g) for encoded_g in mutant_encoded])
    if sum_groups == len(mutant_graph_tokens) + 1:
        mutant_encoded = mutant_encoded[1:]
    if s2color is not None:
        color_map = []
        for group in mutant_encoded:
            color_map.append(plt_colors.to_hex(
                (s2color[group][0], s2color[group][1], s2color[group][2], s2color[group][3])))
    else:
        color_map = None
    mutant_node_groups = get_node_groups(mutant_node_list, mutant_encoded)
    initial_node_groups = get_node_groups(initial_node_list, initial_encoded)

    def get_subgraph_and_context_edges(node_group, graph):

        ng = node_group
        _subgraph = graph.subgraph(ng)
        subgraph = _subgraph.copy()
        inherit_graph_attributes(subgraph, graph)

        subgraph_nodes = subgraph.nodes()
        context_edges = get_context_edges(subgraph, graph, subgraph_nodes)

        return subgraph, context_edges

    initial_subgraphs, mutant_subgraphs = [], []
    for ng in initial_node_groups:
        sg, _ = get_subgraph_and_context_edges(ng, initial_graph)
        initial_subgraphs.append(sg)
    for ng in mutant_node_groups:
        sg, _ = get_subgraph_and_context_edges(ng, mutant_graph)
        mutant_subgraphs.append(sg)

    highlight_idx = find_changed_seg_idx(initial_subgraphs, mutant_subgraphs)
    new_node_groups = []
    for idx in highlight_idx:
        new_node_groups.append(mutant_node_groups[idx])
    mutant_node_groups = new_node_groups
    print('#groups:{}, max group length:{}'.format(
        len(mutant_node_groups), max([len(g) for g in mutant_node_groups])))
    if am_name is None:
        fname = '{}_random'.format(family)
    else:
        fname = am_name
    highlight_node_groups(X_s=mutant_graph, node_groups=mutant_node_groups, name=fname,
                          key_set='short', save_path=save_dir, color_map=color_map, use_cg_name=False)


def segment_and_plot_graph(graph, nt, sp, save_dir='segments', family='', model_type='unigram', encoding='op', s2color=None, am_name=None):
    print('segmenting graph {}'.format(graph.name))
    node_list = list(nx.topological_sort(graph))
    graph_tokens = nt.encode_graph(graph, node_list=node_list)
    encoded = sp.encode_as_pieces(graph_tokens)

    sum_groups = sum([len(encoded_g) for encoded_g in encoded])
    if sum_groups == len(graph_tokens) + 1:
        encoded = encoded[1:]

    if s2color is not None:
        color_map = []
        for group in encoded:
            color_map.append(plt_colors.to_hex(
                (s2color[group][0], s2color[group][1], s2color[group][2], s2color[group][3])))
    else:
        color_map = None
    node_groups = get_node_groups(node_list, encoded)
    print('#groups:{}, max group length:{}'.format(
        len(node_groups), max([len(g) for g in node_groups])))
    if am_name is None:
        fname = '{}_random'.format(family)
    else:
        fname = am_name
    highlight_node_groups(X_s=graph, node_groups=node_groups, name=fname,
                          key_set='short', save_path=save_dir, color_map=color_map, use_cg_name=False)


def find_changed_seg_idx(e1, e2):
    i, j = 0, 0
    diff_idx = []
    while i < len(e2) and j < len(e1):
        if not compare_networkx(e1[j], e2[i]):
            indices = find_reconcile(i + 1, j + 1, e1, e2)
            if indices is None:
                k1, k2 = len(e1), len(e2)
            else:
                [k1, k2] = indices
            for non_matching_seg_idx in range(i, k2):
                diff_idx.append(non_matching_seg_idx)
            i, j = k2, k1
        else:
            i += 1
            j += 1
    return diff_idx


def compare_networkx(nx1, nx2):
    if len(nx1.nodes) != len(nx2.nodes) or len(nx1.edges) != len(nx2.edges):
        return False
    nx1_info, nx2_info = [], []
    for _, v in nx1.nodes(data=True):
        info = f"{v['op_type_idx']}_{v['resolution']}"
        nx1_info.append(info)
    for _, v in nx2.nodes(data=True):
        info = f"{v['op_type_idx']}_{v['resolution']}"
        nx2_info.append(info)
    nx1_info.sort()
    nx2_info.sort()
    for i, info in enumerate(nx1_info):
        if info != nx2_info[i]:
            return False
    return True
    

def find_reconcile(i, j, e1, e2):
    for k1 in range(j, len(e1)):
        for k2 in range(i, len(e2)):
            if compare_networkx(e1[k1], e2[k2]):
                return [k1, k2]
    return None


def save_mutation_results(params, filename, top_set, mutator, run_manager, input_model):
    op2i = OP2I().build_from_file()
    print(top_set[0])
    new_top_set = []
    for i in range(len(top_set)):
        try: 
            new_cg = correct_cg_for_downstream(top_set[i][0], op2i=op2i, H=params.input_h, W=params.input_w, C=params.input_c)
        except Exception as e:
            print(f"Top set CG rank {i}, failed to compile into network!")
            new_cg = top_set[i][0]
            print("Exception: ", e)
        new_tuple = (new_cg, top_set[i][1])
        new_top_set.append(new_tuple)
    top_set = new_top_set

    output_dir = os.path.join(params.output_dir, filename)
    logs_dir = os.path.join(params.logs_dir, filename)

    vocab_size = 2000
    sub_str = False
    model_type = "bpe"
    encoding = 'shp'
    sp, nt = load_and_test_encoder(families=['hiaml', 'inception', 'nb101_5k', 'nb201', 'two_path'], vocab_size=vocab_size, sub_str=sub_str, model_type=model_type, encoding=encoding)

    if params.mutation_unit == "op":
        vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
        singles = [token for token in vocabs if len(token) == 1]
        sp.set_vocabulary(singles)

    initial_cg_nx = cg_to_nx(input_model)[0]
    segment_and_plot_graph(initial_cg_nx, nt, sp, save_dir=output_dir, family=params.model_family, model_type=model_type, encoding=encoding, am_name="input_arch")

    for i in range(len(top_set)):
        g = cg_to_nx(top_set[i][0])[0]
        segment_and_plot_graph(g, nt, sp, save_dir=output_dir, family=params.model_family, model_type=model_type, encoding=encoding, am_name=f"rank_{i}")#, s2color=segment_map
        plot_mutant_changes(g, initial_cg_nx, nt, sp, save_dir=output_dir, family=params.model_family, model_type=model_type, encoding=encoding, am_name=f"mutant_{i}")

        with open(output_dir + "/rank_%d.pkl" % i, "wb+") as f:
            pickle.dump(top_set[i][0], f)

    print("Saving results...")
    with open(output_dir + "/mutation_result.txt", "w+") as sys.stdout:
        run_manager.print_results(top_set)
        print("Total time: ", run_manager.time_consumption)
        print("Total calls to MILP: ", mutator.total_milp_calls)
        print(f"MILP Success/Fails: {mutator.milp_dict['success']}/{mutator.milp_dict['fail']}")
        if mutator.milp_dict['success'] > 1:
            print(f"MILP Success Ave Time: {np.mean(mutator.milp_dict['success_time'])} +/- {np.std(mutator.milp_dict['success_time'])}")
        if mutator.milp_dict['fail'] > 1:
            print(f"MILP Fail Ave Time: {np.mean(mutator.milp_dict['fail_time'])} +/- {np.std(mutator.milp_dict['fail_time'])}")
        for key in ["conv", "pool", "concat"]:
            print(f"Average {key} per MILP call: {mutator.milp_dict[key] / max(mutator.total_milp_calls, 1)}")
        print("Total visited architectures: %d" % len(run_manager.visited_architecture_list))
        print("Total visited architectures without repetition: %d" % len(set(run_manager.visited_architecture_list)))

    sys.stdout = sys.__stdout__
    print("Finished saving mutation results...")


def save_train_results(params, filename, train_results, best_str):
    output_dir = os.path.join(params.output_dir, filename)
    with open(output_dir + "/train_result.txt", "w+") as sys.stdout:
        print(train_results)
        print(best_str)

    sys.stdout = sys.__stdout__


def train_model(params, model, CompGraphOutputNet, l2_reg_constant=1e-5, post_trainer=cifar10_train_test):
    def net_maker():
        return CompGraphOutputNet(OP2I().build_from_file(), cg=model, name=model.name,
                                  l2_reg_constant=l2_reg_constant,
                                  disable_bn=False, enable_bias=False)

    _, input_model_test_acc = post_trainer(net_maker,
                                                              num_runs=params.num_runs_cg,
                                                              batch_size=params.train_batch_size,
                                                              epochs=params.train_epochs)

    return input_model_test_acc.result


def main(params, filename):
    output_dir = os.path.join(params.output_dir, filename)
    logs_dir = os.path.join(params.logs_dir, filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    with open(logs_dir + "/params.pkl", "wb+") as f:
        pickle.dump(params, f)

    set_random_seed(params.seed)

    input_model = load_input_model(params.model_path, params.input_c, params.input_h, params.input_w)

    predictor = load_accuracy_predictor(params.predictor_path, params.predictor_type)

    if params.predictor_type == "PSC_predictor":
        forward_function_id = 1
    elif params.predictor_type == "GNN":
        forward_function_id = 2
    else:
        raise NotImplementedError("%s is not a supported predictor type" % params.predictor_type)
    
    perf_checker = AccDiffChecker(partial(cg_regress, model=predictor,
                                          forward_function_id=forward_function_id),
                                  min_increase_percent=0,
                                  predictor_type=params.predictor_type)

    flops_checker = FlopsDiffChecker(OP2I().build_from_file(),
                                     min_decrease_percent=params.min_flops_decrease_percent,
                                     max_decrease_percent=params.max_flops_decrease_percent)

    family = params.model_family if params.model_family in ["hiaml", "inception", "two_path", "nb201", "nb101", "edsr", "generic_noDW"] else "generic"
    mutator = Mutator(family,
                      params.graph_encoder_family,
                      params.graph_encoder_path,
                      params.tokenizer_path,
                      params.segment_database_path,
                      params.max_candidate_per_iter,
                      params.max_target_per_iter,
                      params.num_consecutive_segs, 
                      propagation_type=params.propagation_type,
                      mutation_unit=params.mutation_unit,
                      predictor_type=params.predictor_type,
                      acc_checker=perf_checker,
                      flops_checker=flops_checker,
                      has_budget=params.max_visited_architecture != sys.maxsize,
                      segment_sparsity=params.segment_sparsity)

    run_manager = RunManager(mutator,
                             OP2I().build_from_file(),
                             flops_checker,
                             perf_checker,
                             max_visited_architecture=params.max_visited_architecture,
                             num_iterations=params.epoch,
                             predictor_type=params.predictor_type,
                             n_jobs=params.n_jobs)

    if params.train_type == "TensorFlow":
        from model_src.comp_graph.tf_comp_graph_output import \
            CompGraphOutputNet
    else:
        raise NotImplementedError("%s is not supported" % params.train_type)

    top_set, pair_CG_list = run_manager.run(input_model, top_k=params.top_k)

    save_mutation_results(params, filename, top_set, mutator, run_manager, input_model)

    if params.train_input_cg:
        print("\n Training the input CG \n")
        input_model_dev_acc = train_model(params, input_model, CompGraphOutputNet)
    else:
        input_model_dev_acc = None

    mutated_model_dev_acc_list = []
    if params.num_train_cgs == -1:
        params.num_train_cgs = len(top_set)
    if params.num_train_cgs > 0:
        best_score, best_cg, best_i = -1, None, -1
        i, num_evals = 0, 0
        while num_evals < params.num_train_cgs:
            if i >= len(top_set):
                break
            if top_set[i][-1][-1] < np.ceil(params.epoch * params.iteration_min_percent):
                print("\n Rank %d mutated CG has a lower edit distance than %d; not training \n" % (i, np.ceil(params.epoch * params.iteration_min_percent)))
            else:
                print("\n Training rank %d mutated CG \n" % i)
                try:
                    result = train_model(params, top_set[i][0], CompGraphOutputNet)
                    num_evals += 1
                except Exception as e:
                    print(f"Attempt to train rank {i} failed, setting accs to be -1")
                    print("Exception: ", e)
                    result = [-1] * params.num_runs_cg
                max_score = np.mean(result)
                if max_score > best_score:
                    best_score = max_score
                    best_cg = top_set[i][0]
                    best_i = i
                mutated_model_dev_acc_list.append(result)
            i += 1
        best_flops = compute_cg_flops(best_cg, OP2I().build_from_file(), use_fast_counter=True, div=1e6)
        delta_flops = ((best_flops - flops_checker.baseline_flops) / flops_checker.baseline_flops) * 100
        best_nodes = len(best_cg.nodes)
        delta_nodes = best_nodes - len(input_model.nodes)

        best_train_str = f"Best rank = {best_i}; best acc = {best_score}; best FLOPs = {best_flops}; Delta FLOPs = {delta_flops}; iteration = {top_set[best_i][-1][-1]}; Nodes = {best_nodes}; Delta Nodes = {delta_nodes}"
    else:
        best_train_str = None

    save_train_results(params, filename, [input_model_dev_acc, mutated_model_dev_acc_list], best_train_str)


def parse_arguments():
    """
    Call the parser functions and returned the parsed arguments
    """
    parser = argparse.ArgumentParser()

    add_general_params(parser)
    add_predictor_params(parser)
    add_flops_params(parser)
    add_mutation_params(parser)
    add_training_params(parser)

    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    params = parse_arguments()
    filename = params.model_family + "_{}_epoch_{}_max_target_{}_top_k_{}_{}_date_{}". \
        format(params.mutation_unit,
               params.epoch,
               params.max_target_per_iter,
               params.top_k,
               params.predictor_type,
               datetime.now().strftime("%Y%m%d_%H%M%S"))
    main(params, filename)
