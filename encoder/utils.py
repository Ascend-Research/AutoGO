from tqdm import tqdm
import networkx as nx
import pickle
import codecs
import random
from constants import *
import itertools
from motifs.motif_structures.utils import *
from params import *
from motifs.motif_structures.utils import extract_in_resolution, get_input_nodes


def get_segment2subgraph(sampled_graphs, nt, sp, family,
                         encoding='op',
                         sub_str=False,
                         model_type='bpe',
                         save=True):

    segment2subgraph = {}

    if len(sampled_graphs) > 1:
        bar = tqdm(total=len(sampled_graphs), desc="Mapping segments to subgraphs", ascii=True)

    for sampled_graph in sampled_graphs:
        node_list = list(nx.topological_sort(sampled_graph))
        g_tokens = nt.encode_graph(sampled_graph, node_list=node_list)
        encoded = sp.encode_as_pieces(g_tokens)
        sum_groups = sum([len(encoded_g) for encoded_g in encoded])
        if sum_groups == len(g_tokens) + 1:
            encoded = encoded[1:]
        sum_groups = sum([len(encoded_g) for encoded_g in encoded])
        if len(g_tokens) != len(node_list):
            print('token list is not the same size as node list')
            continue
        if len(g_tokens) != sum_groups:
            print('the sum of elements in segment groups is not equal to the token list')
            continue
        _node_groups = get_node_groups(node_list, encoded)
        node_groups = _node_groups
        for g_id, ng in enumerate(node_groups):
            _subgraph = sampled_graph.subgraph(ng)
            subgraph = _subgraph.copy()
            inherit_graph_attributes(subgraph, sampled_graph)
            new_subgraph = True
            if encoded[g_id] not in segment2subgraph.keys():
                segment2subgraph[encoded[g_id]] = []
            for sg in segment2subgraph[encoded[g_id]]:
                if is_equal(sg, subgraph, check_resolution=True):
                    new_subgraph = False
                    break
            if new_subgraph:
                segment2subgraph[encoded[g_id]].append(subgraph)
        if len(sampled_graphs) > 1:
            bar.update(1)
    if len(sampled_graphs) > 1:
        bar.close()
    if save:
        path = P_SEP.join([CACHE_SPM_DIR, family])
        with codecs.open(P_SEP.join([path, '{}_segment2subgraph_{}_{}{}.pkl'.format(family, encoding, model_type, '' if not sub_str else '_substr')]), 'wb') as f:
            pickle.dump(dict(segment2subgraph), f)
    return segment2subgraph


def inherit_graph_attributes(subgraph, graph):
    subgraph_str_ids = {n['str_id']: n for _, n in subgraph.nodes(data=True)}
    subgraph.graph['input_shape'] = [extract_in_resolution(
        i['resolution'], add_batch=True) for i in get_input_nodes(subgraph, ids_only=False)]
    subgraph.graph['edge_pairs'] = list(subgraph.edges)
    subgraph.graph['regular_nodes'] = [copy.deepcopy(
        n) for n in graph.graph['regular_nodes'] if n.str_id in subgraph_str_ids]
    subgraph.graph['weighted_nodes'] = [copy.deepcopy(
        n) for n in graph.graph['weighted_nodes'] if n.str_id in subgraph_str_ids]
    subgraph.graph['n_regular_nodes'] = len(subgraph.graph['regular_nodes'])
    subgraph.graph['n_weighted_nodes'] = len(subgraph.graph['weighted_nodes'])

    subgraph.graph['max_hidden_size'] = graph.graph['max_hidden_size']
    subgraph.graph['max_kernel_size'] = graph.graph['max_kernel_size']
    subgraph.graph['max_derived_H'] = graph.graph['max_derived_H']
    subgraph.graph['max_derived_W'] = graph.graph['max_derived_W']
    subgraph.graph['src_id2dst_ids_dict'] = get_src_id2dst_ids_dict(subgraph)


def get_context_edges(subgraph, sampled_graph, subgraph_nodes):
    context_edges = {}
    for n in subgraph_nodes:
        context_edges[n] = [[], []]
        if subgraph.in_degree(n) != sampled_graph.in_degree(n):
            for (s, d) in list(sampled_graph.in_edges(n)):
                if s not in subgraph_nodes:
                    context_edges[n][0].append(s)
        if subgraph.out_degree(n) != sampled_graph.out_degree(n):
            for (s, d) in list(sampled_graph.out_edges(n)):
                if d not in subgraph_nodes:
                    context_edges[n][1].append(d)
        if context_edges[n] == [[], []]:
            del context_edges[n]

    return context_edges


def get_segment2subgraph_list_with_ids(graph, graph_encoder, tokenizer, num_consecutive_segs=1, segment_sparsity=1.):
    """
    Segment the input graph
    """
    segment2subgraph_list = []

    def get_subgraph_and_context_edges(node_group):

        ng = node_group
        _subgraph = graph.subgraph(ng)
        subgraph = _subgraph.copy()
        inherit_graph_attributes(subgraph, graph)

        subgraph_nodes = subgraph.nodes()
        context_edges = get_context_edges(subgraph, graph, subgraph_nodes)

        return subgraph, context_edges

    node_list = list(nx.topological_sort(graph))

    encoded_graph = graph_encoder.encode_graph(graph, node_list=node_list)
    assert len(encoded_graph) == len(node_list), "Number of graph nodes changed after encoding"
    encoded_segments = tokenizer.encode_as_pieces(encoded_graph)

    method = getattr(tokenizer, 'get_piece_size', None)
    if callable(method):
        if segment_sparsity is not None:
            vocabs = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())]
            singles = [token for token in vocabs if len(token) == 1]
            multis = [token for token in vocabs if len(token) > 1]
            new_multis = random.sample(multis, int(len(multis) * segment_sparsity))
            new_vocab = singles + new_multis
            tokenizer.set_vocabulary(new_vocab)
            encoded_segments = tokenizer.encode_as_pieces(encoded_graph)
            tokenizer.reset_vocabulary()
        else:
            encoded_segments = tokenizer.encode_as_pieces(encoded_graph)
        assert "".join(encoded_segments) == encoded_graph, "Graph elements changed after segmentation"

    node_groups = get_node_groups(node_list, encoded_segments)
    
    for g_id, ng in enumerate(node_groups):
        for ncs in range(num_consecutive_segs):
            if g_id == 0 or g_id >= len(node_groups)-ncs-1:
                continue

            cng = list(itertools.chain(*node_groups[g_id:g_id+ncs+1]))
            subgraph = graph.subgraph(cng)
            parent_nodes = [edge[0] for edge in graph.in_edges(cng) if edge[0] not in cng]
            child_nodes = [edge[1] for edge in graph.out_edges(cng) if edge[1] not in cng]

            predecessor_ng = list(itertools.chain(*node_groups[:g_id]))
            predecessor_str = "".join(encoded_segments[:g_id])

            segment_ng = list(itertools.chain(*node_groups[g_id:g_id+ncs+1]))
            segment_str = "".join(encoded_segments[g_id:g_id+ncs+1])

            successor_ng = list(itertools.chain(*node_groups[g_id+ncs+1:]))
            successor_str = "".join(encoded_segments[g_id+ncs+1:])

            predecessor_subgraph, predecessor_ce = get_subgraph_and_context_edges(predecessor_ng)
            segment_subgraph, segment_ce = get_subgraph_and_context_edges(segment_ng)
            if segment_sparsity is None and len(segment_subgraph.nodes) == 1:
                if prune_res_conv(seg_subgraph=segment_subgraph):
                    continue

            successor_subgraph, successor_ce = get_subgraph_and_context_edges(successor_ng)

            segment2subgraph_list.append(
                {'encoded_segment': encoded_segments[g_id], 'subgraph': subgraph, 'parent_nodes': parent_nodes,
                'child_nodes': child_nodes, "predecessor_subgraph": predecessor_subgraph,
                "segment_subgraph": segment_subgraph, "successor_subgraph": successor_subgraph})

    return segment2subgraph_list


def get_psc_subgraphs_dict(sampled_graphs, nt, sp, family,
                           encoding='op',
                           sub_str=False,
                           model_type='bpe',
                           save=True,
                           multi_segs=False,
                           plot=False,
                           save_path=P_SEP.join([PLOT_DIR]),
                           skip_short_segs=False):

    import itertools

    segment2subgraph_dict = {}

    def get_subgraph_and_context_edges(node_group):

        ng = node_group
        _subgraph = sampled_graph.subgraph(ng)
        subgraph = _subgraph.copy()
        inherit_graph_attributes(subgraph, sampled_graph)

        subgraph_nodes = subgraph.nodes()
        context_edges = get_context_edges(subgraph, sampled_graph, subgraph_nodes)

        return subgraph, context_edges

    if len(sampled_graphs) > 1:
        bar = tqdm(total=len(sampled_graphs), desc="Mapping  {} segments to subgraphs for {}".format(
            nt.family, family), ascii=True)

    for sampled_graph in sampled_graphs:
        node_list = list(nx.topological_sort(sampled_graph))
        g_tokens = nt.encode_graph(sampled_graph, node_list=node_list)
        if multi_segs:
            encoded = sp.encode(g_tokens, out_type=str,
                                enable_sampling=True, alpha=0.1, nbest_size=-1)
        else:
            encoded = sp.encode_as_pieces(g_tokens)
        sum_groups = sum([len(encoded_g) for encoded_g in encoded])
        if sum_groups == len(g_tokens) + 1:
            encoded = encoded[1:]
        sum_groups = sum([len(encoded_g) for encoded_g in encoded])
        if len(g_tokens) != len(node_list):
            print('token list is not the same size as node list')
            continue
        if len(g_tokens) != sum_groups:
            print('the sum of elements in segment groups is not equal to the token list')
            print(encoded, g_tokens, len(g_tokens), sum_groups)
            continue
        _node_groups = get_node_groups(node_list, encoded)
        node_groups = _node_groups

        for g_id, ng in enumerate(node_groups):
            predecessor_ng = list(itertools.chain(*node_groups[:g_id]))
            predecessor_str = "".join(encoded[:g_id])

            segment_ng = node_groups[g_id]
            segment_str = encoded[g_id]
            if len(predecessor_ng) == 0:
                predecessor_ng, segment_ng = segment_ng[:1], segment_ng[1:]
                predecessor_str, segment_str = segment_str[:1], segment_str[1:]
            if g_id == len(node_groups) - 1:
                segment_ng, successor_ng = segment_ng[:-1], segment_ng[-1:]
                segment_str, successor_str = segment_str[:-1], segment_str[-1:]
            else:
                successor_ng = itertools.chain(*node_groups[g_id + 1:])
                successor_str = "".join(encoded[g_id + 1:])

            predecessor_subgraph, predecessor_ce = get_subgraph_and_context_edges(predecessor_ng)
            segment_subgraph, segment_ce = get_subgraph_and_context_edges(segment_ng)
            successor_subgraph, successor_ce = get_subgraph_and_context_edges(successor_ng)

            segment_token = segment_str
            if skip_short_segs and len(segment_token) == 1:
                continue
            
            segment_dict = {
                "predecessor_cg": [nx_to_cg(predecessor_subgraph)[0]],
                "segment_cg": [nx_to_cg(segment_subgraph)[0]],
                "successor_cg": [nx_to_cg(successor_subgraph)[0]]
            }
            if segment_token in segment2subgraph_dict.keys():
                for k, v in segment_dict.items():
                    segment2subgraph_dict[segment_token][k].extend(v)
            else:
                segment2subgraph_dict[segment_token] = segment_dict

        if len(sampled_graphs) > 1:
            bar.update(1)
    if len(sampled_graphs) > 1:
        bar.close()
    return segment2subgraph_dict


def prune_res_conv(seg_subgraph):
    [(_, single_node)] = seg_subgraph.nodes(data=True)
    if single_node['op_type_idx'] != 3:
        return False
    res = single_node['resolution']
    if res[-2] == res[-1] and res[0] == res[1]:
        return False
    return True


def get_node_groups(node_list, encoded):
    node_groups = []
    current_position = 0
    for g_id, encoded_g in enumerate(encoded):
        group = []
        for n_id, n in enumerate(encoded_g):
            group.append(node_list[current_position])
            current_position += 1
        node_groups.append(group)
    return node_groups
