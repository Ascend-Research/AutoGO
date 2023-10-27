import copy
import warnings
import numpy as np
import collections


def hash_module(matrix, labeling):
  import hashlib
  vertices = np.shape(matrix)[0]
  in_edges = np.sum(matrix, axis=0).tolist()
  out_edges = np.sum(matrix, axis=1).tolist()

  assert len(in_edges) == len(out_edges) == len(labeling)
  hashes = list(zip(out_edges, in_edges, labeling))
  hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
  for _ in range(vertices):
    new_hashes = []
    for v in range(vertices):
      in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
      out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
      new_hashes.append(hashlib.md5(
          (''.join(sorted(in_neighbors)) + '|' +
           ''.join(sorted(out_neighbors)) + '|' +
           hashes[v]).encode('utf-8')).hexdigest())
    hashes = new_hashes
  fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

  return fingerprint


def edge_pairs_to_edge_list(edge_pairs):
    return [[p[0] for p in edge_pairs], [p[1] for p in edge_pairs]]


def edge_list_to_edge_pairs(edge_list):
    rv = list(zip(edge_list[0], edge_list[1]))
    return sorted(rv)


def edge_list_to_adj_dict(src_inds, dst_inds, ignore_self_edge=True):
    adj_dict = collections.defaultdict(set)
    for src, dst in zip(src_inds, dst_inds):
        if ignore_self_edge and src == dst:
            continue
        adj_dict[src].add(dst)
    return adj_dict


def adj_dict_to_edge_list(adj_dict):
    edge_pairs = []
    for src, dst_inds in adj_dict.items():
        for dst in dst_inds:
            edge_pairs.append((src, dst))
    return edge_pairs_to_edge_list(edge_pairs)


def edge_list_to_adj_mat(src_inds, dst_inds, ignore_self_edge=True):
    edge_pairs = edge_list_to_edge_pairs([src_inds, dst_inds])
    n_nodes = max(max(p) for p in edge_pairs) + 1 
    mat = [[0 for i in range(n_nodes)] for j in range(n_nodes)]
    for src, dst in edge_pairs:
        if ignore_self_edge and src == dst: continue
        mat[src][dst] = 1
    return mat


def edge_list_to_edge_matrix(edge_list, n_nodes):
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for idx in range(len(edge_list[0])):
        matrix[edge_list[0][idx]][edge_list[1][idx]] = 1
    return matrix


def is_complete_dag(num_nodes, edge_list):
    edge_set = {(src, dst) for src, dst in zip(edge_list[0], edge_list[1])}
    for n1 in range(num_nodes):
        for n2 in range(num_nodes):
            if n2 <= n1: continue
            if (n1, n2) not in edge_set:
                return False
    return True


def get_complete_dag_edges(num_nodes):
    edge_pairs = set()
    for n1 in range(num_nodes):
        for n2 in range(num_nodes):
            if n2 <= n1: continue
            edge_pairs.add( (n1, n2) )
    return edge_pairs


def prune_self_edge_from_edge_list(torch_geo_edge_list):
    src_list, dst_list = [], []
    for src, dst in zip(torch_geo_edge_list[0], torch_geo_edge_list[1]):
        if src != dst:
            src_list.append(src)
            dst_list.append(dst)
    return [src_list, dst_list]


def get_output_node_indices(torch_geo_edge_list, output_node_idx=None):
    src_set, dst_set = set(), set()
    for src, dst in zip(torch_geo_edge_list[0], torch_geo_edge_list[1]):
        if src == dst: continue
        if output_node_idx is not None and dst == output_node_idx: continue
        src_set.add(src)
        dst_set.add(dst)
    return dst_set - src_set


def get_input_node_indices(torch_geo_edge_list):
    src_set, dst_set = set(), set()
    for src, dst in zip(torch_geo_edge_list[0], torch_geo_edge_list[1]):
        if src == dst: continue
        src_set.add(src)
        dst_set.add(dst)
    return src_set - dst_set


def input_node_inds_list_to_edge_list(input_node_inds):
    src_list, dst_list = [], []
    unique_edges = set()
    for ni, input_inds in enumerate(input_node_inds):
        for si in input_inds:
            assert si <= ni
            assert (si, ni) not in unique_edges
            unique_edges.add((si, ni))
            src_list.append(si)
            dst_list.append(ni)
    return src_list, dst_list


def get_incoming_nodes_dict(adj_dict):
    rv = collections.defaultdict(set)
    for src, dst_set in adj_dict.items():
        for dst in dst_set:
            rv[dst].add(src)
    return rv


def dfs_path(adj_dict, curr_src=0, seen=None, path=None):
    if seen is None: seen = []
    if path is None: path = [curr_src]
    seen.append(curr_src)
    paths = []
    for t in list(adj_dict[curr_src]):
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(dfs_path(adj_dict, t, seen[:], t_path))
    return paths


def bfs_neighbor_nodes(adj_dict, start, n_hops):
    visited = {start,}
    q = [(start, 0)]
    while len(q) > 0:
        n, hop = q.pop(0)
        if hop < n_hops:
            for neighbor in adj_dict[n]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, hop + 1))
    return list(visited)


def remove_node_in_dag(nodes, edge_list, tgt_idx, verbose=True):
    if verbose and (tgt_idx == 0 or tgt_idx == len(nodes) - 1):
        warnings.warn("Removing the first or last node will cause the loss of edges")

    new_nodes = copy.deepcopy(nodes)
    del new_nodes[tgt_idx]

    parent_indices = set()
    child_new_indices = set()
    for src, dst in zip(edge_list[0], edge_list[1]):
        if src == tgt_idx:
            child_new_indices.add(dst - 1)
        elif dst == tgt_idx:
            parent_indices.add(src)

    new_srcs, new_dsts = [], []
    for src, dst in zip(edge_list[0], edge_list[1]):
        if src != tgt_idx and dst != tgt_idx:
            new_srcs.append(src - 1 if src > tgt_idx else src)
            new_dsts.append(dst - 1 if dst > tgt_idx else dst)
    for src in parent_indices:
        for dst in child_new_indices:
            new_srcs.append(src)
            new_dsts.append(dst)
    return new_nodes, [new_srcs, new_dsts]


def topo_sort_dfs(nodes, adj_dict):
    visited = set()
    results = []
    for node in nodes:
        if node not in visited:
            _topo_sort_dfs(node, adj_dict, visited, results)
    return results


def _topo_sort_dfs(node, adj_dict, visited, results):
    chds = adj_dict[node]
    visited.add(node)
    for chd in chds:
        if chd not in visited:
            _topo_sort_dfs(chd, adj_dict, visited, results)
    results.append(node)


def is_reachable(adj_dict, src, dst):
    visited = set()
    queue = [src]
    visited.add(src)
    while queue:
        n = queue.pop(0)
        if n == dst:
            return True
        for neighbor in adj_dict[n]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return False


def get_reverse_adj_dict(src2dsts, allow_self_edges=False):
    dst2srcs = collections.defaultdict(set)
    for src, dsts in src2dsts.items():
        for dst in dsts:
            if not allow_self_edges:
                assert src != dst, "src: {}, dst: {}".format(src, dst)
            dst2srcs[dst].add(src)
    return dst2srcs


def get_index_based_input_inds(node_ids, src2dsts):
    dst2srcs = get_reverse_adj_dict(src2dsts, allow_self_edges=False)
    node_id2idx = {nid: ni for ni, nid in enumerate(node_ids)}
    graph_input_inds = []
    for ni, node_id in enumerate(node_ids):
        input_ids = dst2srcs[node_id]
        node_input_inds = [node_id2idx[_id] for _id in input_ids]
        node_input_inds.sort()
        assert all(i < ni for i in node_input_inds), "{}, {}".format(ni, node_input_inds)
        graph_input_inds.append(node_input_inds)
    return graph_input_inds


def build_nx_block_graph(op_label_list, edge_lists):
    import networkx as nx
    graph = nx.DiGraph()
    for ni, node_label in enumerate(op_label_list):
        graph.add_node(node_label + "_{}".format(ni))
    for src, dst in zip(edge_lists[0], edge_lists[1]):
        if src == dst: continue
        src_label = op_label_list[src] + "_{}".format(src)
        dst_label = op_label_list[dst] + "_{}".format(dst)
        graph.add_edge(src_label, dst_label)
    return graph


def compute_graph_edit_distance(truth_op_label_list, truth_edge_lists,
                                pred_op_label_list, pred_edge_lists):
    from networkx.algorithms.similarity import optimize_graph_edit_distance
    truth_graph = build_nx_block_graph(truth_op_label_list, truth_edge_lists)
    pred_graph = build_nx_block_graph(pred_op_label_list, pred_edge_lists)
    dist = float("inf")
    for v in optimize_graph_edit_distance(truth_graph, pred_graph):
        dist = min(dist, v)
    return dist
