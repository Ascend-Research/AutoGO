import copy
import pickle
import random
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from constants import *
from model_src.comp_graph.tf_comp_graph import ComputeGraph, prune_single_cat_add_mul_nodes_PSC
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from networkx.algorithms import isomorphism
from networkx.algorithms.dag import dag_longest_path_length
from params import *
from tqdm import tqdm


def load_family_graphs_info(family, to_nx=False, k=-1, randomize=False, cg_name=None):

    if family in ['hiaml','two_path','inception']:
        family_manager = FamilyDataManager(log_f=print, families=[family], data_dir=DATA_DIR)
        nets = family_manager.get_family_custom_set(data_file=CG_NAMES[family]+'.pkl' if 'json' not in CG_NAMES[family] else CG_NAMES[family], family=family)
    elif family in ['ofa_mbv3', 'ofa_pn']:
        family_manager = FamilyDataManager(log_f=print, families=[family], data_dir=DATA_DIR)
        nets = family_manager.load_cache_data(family)
    else:
        with open(P_SEP.join([DATA_DIR, CG_NAMES[family]+'.pkl' if cg_name is None else cg_name+'.pkl']), 'rb') as f:
            nets = pickle.load(f)
    print('loaded {} compute graphs from {}'.format(len(nets), P_SEP.join([DATA_DIR, CG_NAMES[family] if cg_name is None else cg_name])))
    
    if k == -1:
        k = len(nets)
    if isinstance(nets[0], dict):
        cgs = [net['compute graph'] for net in nets]
        accs = [net['acc'] for net in nets]
        flops = [net['flops'] for net in nets]
    else:
        cgs = [net[0] for net in nets]
        accs = [net[1] for net in nets]
        flops = []
    s = pd.Series(accs)
    print(s.describe())
    print('extracting the cgs for the first {} nets'.format(k))
    acc_idx = list(np.argsort(accs))
    acc_idx.reverse()
    if randomize:
        random.shuffle(acc_idx)
    sorted_idx = acc_idx[:k]
    cgs = [cgs[i] for i in sorted_idx]
    accs = [accs[i] for i in sorted_idx]
    print('new distribtion:')
    s = pd.Series(accs)
    print(s.describe())
    if to_nx:
        graphs = cg_to_nx(cgs)
    else:
        graphs = cgs
    return graphs, accs, flops

def load_nx_graphs_from_cg_name(graphs_name='', index=None):
    cache_name = graphs_name+'.pkl' if 'json' not in graphs_name else graphs_name
    if any([f in graphs_name for f in ['hiaml', 'two_path','inception','ofa_resnet']]):
        if 'two_path' in graphs_name:
            family = 'two_path'
        elif 'ofa_resnet' in graphs_name:
            family = 'ofa_resnet'
        elif 'hiaml' in graphs_name:
            family = 'hiaml'
        else:
            family = 'inception'
        family_manager = FamilyDataManager(log_f=print, families=[family], data_dir=DATA_DIR)
        nets = family_manager.get_family_custom_set(
            data_file=cache_name, family=family)
    else:
        with open(P_SEP.join([DATA_DIR, cache_name]), 'rb') as f:
            nets = pickle.load(f)
    print('loaded {} compute graphs from {}'.format(len(nets), P_SEP.join([DATA_DIR, graphs_name+'.pkl'])))
    if isinstance(nets[0], dict):
        if index is None:
            cgs = [net['compute graph'] for net in nets]
        else:
            cgs = [nets[index]['compute graph']]
    else:
        if index is None:
            cgs = [net[0] for net in nets]
        else:
            cgs = [nets[index][0]]
    graphs = cg_to_nx(cgs)
    return graphs

def extract_in_resolution(res, add_batch=False):
    if len(res) == 3:
        if add_batch:
            return (1, res[0], res[1], res[2])
        else:
            return res
    if add_batch:
        return (1, res[0], res[2], res[4])
    else:
        return (res[0], res[2], res[4])

def extract_out_resolution(res, add_batch=False):
    if len(res) == 3:
        if add_batch:
            return (1, res[0], res[1], res[2])
        else:
            return res
    if add_batch:
        return (1, res[1], res[3], res[5])
    else:
        return (res[1], res[3], res[5])

def get_input_nodes(nx_dg: nx.DiGraph, ids_only=True):
    in_nodes = []
    for i, n in nx_dg.nodes(data=True):
        if nx_dg.in_degree(i) == 0:
            if ids_only:
                in_nodes.append(i)
            else:
                in_nodes.append(n)
    return in_nodes
    
def get_output_nodes_ids(nx_dg: nx.DiGraph, ids_only=True):
    out_nodes = []
    for i, n in nx_dg.nodes(data=True):
        if nx_dg.out_degree(i) == 0:
            if ids_only:
                out_nodes.append(i)
            else:
                out_nodes.append(n)
    return out_nodes

def get_src_id2dst_ids_dict(graph):
    src_id2dst_ids = defaultdict(set)
    nodes = graph.nodes
    for src, dst in graph.edges:
        src_id2dst_ids[nodes[src]['str_id']].add(nodes[dst]['str_id'])

    return src_id2dst_ids

def is_equal(motif1, motif2, check_resolution=False):
    if check_resolution:
        gm = isomorphism.DiGraphMatcher(
        motif1, motif2, node_match=lambda x, y: x["op_name"] == y["op_name"] and x["resolution"] == y["resolution"])
    else:
        gm = isomorphism.DiGraphMatcher(
        motif1, motif2, node_match=lambda x, y: x["op_name"] == y["op_name"])
    return gm.is_isomorphic()

def get_h(motif):
        h = dag_longest_path_length(motif)
        return h

def nx_to_cg(graphs):
    cgs = []
    if not isinstance(graphs, list):
        graphs = [graphs]

    for graph in graphs:
        cg = ComputeGraph()
        cg.name = graph.name
        cg.input_shape = list(graph.graph['input_shape'])
        cg.edge_pairs = list(graph.edges)
        cg.regular_nodes = [copy.deepcopy(n) for n in graph.graph['regular_nodes']]
        cg.weighted_nodes = [copy.deepcopy(n) for n in graph.graph['weighted_nodes']]
        cg.n_regular_nodes = graph.graph['n_regular_nodes']
        cg.n_weighted_nodes = graph.graph['n_weighted_nodes']
        cg.max_hidden_size = graph.graph['max_hidden_size']  # For normalization
        cg.max_kernel_size = graph.graph['max_kernel_size']
        cg.max_derived_H = graph.graph['max_derived_H']
        cg.max_derived_W = graph.graph['max_derived_W']
        automutate_keys = ['parent', 'depth', 'iter', 'h_depth']
        for key in automutate_keys:
            if key in graph.graph.keys():
                setattr(cg, key, graph.graph[key])
        cg.set_nodes_edge_pairs(cg.nodes, graph.graph['src_id2dst_ids_dict'])
        
        nodes = cg.nodes
        src2dst = cg.src_id2dst_ids_dict
        nodes, src2dst = prune_single_cat_add_mul_nodes_PSC(nodes, src2dst)
        cg.set_nodes_edge_pairs(nodes, src2dst)

        cgs.append(cg)
    return cgs

def cg_to_nx(cgs, family='nb201', relu6=False, keep_name=False):
    
    networkx_graphs = []
    if not isinstance(cgs, list):
        cgs = [cgs]
    if len(cgs) > 1:
        bar = tqdm(
            total=len(cgs), desc="Building nx graphs from comp graphs", ascii=True
        )
    for cg in cgs:
        nx_dg = nx.DiGraph(**cg.__dict__)
        
        for idx, node in enumerate(cg.nodes):
            label = node.label
            m = re.search(r"_\d+$", label)
            if m is not None:
                label = label.split("_")[0]
            if not relu6 and 'relu6' in label:
                label = 'relu'
            if 'add' in label:
                label = 'add'
            if 'fusedbatchnorm' in label:
                label = 'fusedbatchnorm'
            node_dict = node.__dict__
            node_dict['op_name'] = label
            if re.search(r"\D", node.str_id) is not None:
                if keep_name:
                    node_dict['name'] = node.str_id
                node_dict['str_id'] = str(idx)
                cg.nodes[idx].str_id = str(idx)
            nx_dg.add_node(
                idx, **node_dict
            )
        nx_dg.add_edges_from(cg.edge_pairs)
        nx_dg.graph['src_id2dst_ids_dict'] = cg.src_id2dst_ids_dict
        networkx_graphs.append(nx_dg)
        if len(cgs) > 1:
            bar.update(1)
    if len(cgs) > 1:
        bar.close()
    return networkx_graphs
