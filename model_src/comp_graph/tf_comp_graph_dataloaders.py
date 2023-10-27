import os
import math
import torch
import random
import pickle
import collections
from tqdm import tqdm
from constants import *
from utils.misc_utils import UniqueDict
from model_src.comp_graph.tf_comp_graph import ComputeGraph, WeightedNode, RegularNode
from utils.graph_utils import hash_module, edge_list_to_edge_matrix, bfs_neighbor_nodes, \
    edge_list_to_edge_pairs


_nn_graph_to_cg_map = collections.defaultdict(list)


def _get_graph_id(regular_inds, regular_shapes,
                  weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                  edge_pairs,
                  pivot_idx=-1, pivot_name="none", strategy="complete"):
    edge_list = [[t[0] for t in edge_pairs], [t[1] for t in edge_pairs]]
    graph_inds = weighted_node_inds + regular_inds
    graph_shapes = weighted_node_shapes + regular_shapes
    node_features = []
    for ni in range(len(graph_inds)):
        op_idx = str(graph_inds[ni])
        op_shape = str(graph_shapes[ni])
        if ni < len(weighted_node_kernels):
            op_kernel = str(weighted_node_kernels[ni])
        else:
            op_kernel = "0"
        if ni < len(weighted_node_bias):
            op_bias = str(weighted_node_bias[ni])
        else:
            op_bias = "0"
        node_feature = "|".join([op_idx, op_shape, op_kernel, str(op_bias), str(pivot_idx), pivot_name])
        node_features.append(node_feature)
    if strategy == "complete":
        edge_mat = edge_list_to_edge_matrix(edge_list, len(node_features))
        graph_id = hash_module(edge_mat, node_features)
    elif strategy == "simple":
        graph_id = "+".join([str(node_features), str(edge_pairs)])
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))
    return graph_id


class CGRegressDataLoader:

    def __init__(self, batch_size, data, cache_file=None, verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.curr_batch_idx = 0
        self.batches = []
        if cache_file is not None and os.path.isfile(cache_file):
            print("Note: Loading cached batches from {}".format(cache_file))
            print("Input data instances will be ignored!")
            with open(cache_file, "rb") as f:
                self.batches = pickle.load(f)
        else:
            if len(data[0]) == 6:
                self._build_psc_batches_weighted(data)
            elif len(data[0]) == 5:
                self._build_psc_batches(data)
            else:
                self._build_batches(data)
            if cache_file is not None:
                print("Saving batches to cache: {}".format(cache_file))
                with open(cache_file, "wb") as f:
                    pickle.dump(self.batches, f, protocol=4)
        self.n_batches = len(self.batches)

    def _build_batches(self, data):
        from model_src.comp_graph.tf_comp_graph import ComputeGraph
        self.batches = []

        bins = collections.defaultdict(list)
        for item in data:
            if len(item) == 3:
                g, flops, tgt_val = item
            elif len(item) == 2:
                g, tgt_val = item
                flops = 0
            assert isinstance(g, ComputeGraph)
            key = "{}|{}".format(len(g.nodes), g.regular_node_start_idx)
            bins[key].append((g, flops, tgt_val))

        n_batches = 0
        for _, instances in bins.items():
            n_batches += math.ceil(len(instances) / self.batch_size)

        bar = None
        if self.verbose:
            bar = tqdm(total=n_batches, desc="Building batches", ascii=True)
        for k, data_list in bins.items():
            idx = 0
            while idx < len(data_list):
                batch_list = data_list[idx:idx + self.batch_size]
                batch_regular_inds = []
                batch_regular_shapes = []
                batch_weighted_inds = []
                batch_weighted_shapes = []
                batch_weighted_kernels = []
                batch_weighted_bias = []
                batch_edge_list = []
                batch_names = []
                batch_tgt = []
                batch_last_node_idx_list = []
                batch_unique_str_id_set = set()
                batch_flops = []
                for inst in batch_list:
                    g, flops, tgt = inst
                    assert isinstance(g, ComputeGraph)
                    batch_names.append(g.name)
                    regular_node_inds, regular_node_shapes, \
                        weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias, \
                            edge_list = g.get_gnn_features()
                    if len(regular_node_inds) > 0:
                        batch_regular_inds.append(torch.LongTensor(regular_node_inds).unsqueeze(0))
                        batch_regular_shapes.append(torch.FloatTensor(regular_node_shapes).unsqueeze(0))
                    if len(weighted_node_inds) > 0:
                        batch_weighted_inds.append(torch.LongTensor(weighted_node_inds).unsqueeze(0))
                        batch_weighted_shapes.append(torch.FloatTensor(weighted_node_shapes).unsqueeze(0))
                        batch_weighted_kernels.append(torch.LongTensor(weighted_node_kernels).unsqueeze(0))
                        batch_weighted_bias.append(torch.FloatTensor(weighted_node_bias).unsqueeze(0))
                    batch_edge_list.append(torch.LongTensor(edge_list))
                    batch_tgt.append(tgt)
                    prev_len = batch_last_node_idx_list[-1] if len(batch_last_node_idx_list) > 0 else -1
                    batch_last_node_idx_list.append(prev_len + len(weighted_node_inds) + len(regular_node_inds))
                    graph_id = _get_graph_id(regular_node_inds, regular_node_shapes,
                                             weighted_node_inds, weighted_node_shapes, weighted_node_kernels,
                                             weighted_node_bias,
                                             edge_list_to_edge_pairs(edge_list), strategy="simple")
                    batch_unique_str_id_set.add(graph_id)
                    batch_flops.append(torch.Tensor([flops]))
                batch_tgt = torch.FloatTensor(batch_tgt)
                if len(batch_regular_inds) > 0:
                    batch_regular_inds = torch.cat(batch_regular_inds, dim=0)
                    batch_regular_shapes = torch.cat(batch_regular_shapes, dim=0)
                else:
                    batch_regular_inds = None
                    batch_regular_shapes = None
                if len(batch_weighted_inds) > 0:
                    batch_weighted_inds = torch.cat(batch_weighted_inds, dim=0)
                    batch_weighted_shapes = torch.cat(batch_weighted_shapes, dim=0)
                    batch_weighted_kernels = torch.cat(batch_weighted_kernels, dim=0)
                    batch_weighted_bias = torch.cat(batch_weighted_bias, dim=0)
                else:
                    batch_weighted_inds = None
                    batch_weighted_shapes = None
                    batch_weighted_kernels = None
                    batch_weighted_bias = None
                batch_flops = torch.cat(batch_flops, dim=0).unsqueeze(1)
                batch = UniqueDict([
                    (DK_BATCH_SIZE, len(batch_list)),
                    (DK_BATCH_CG_REGULAR_IDX, batch_regular_inds),
                    (DK_BATCH_CG_REGULAR_SHAPES, batch_regular_shapes),
                    (DK_BATCH_CG_WEIGHTED_IDX, batch_weighted_inds),
                    (DK_BATCH_CG_WEIGHTED_SHAPES, batch_weighted_shapes),
                    (DK_BATCH_CG_WEIGHTED_KERNELS, batch_weighted_kernels),
                    (DK_BATCH_CG_WEIGHTED_BIAS, batch_weighted_bias),
                    (DK_BATCH_EDGE_TSR_LIST, batch_edge_list),
                    (DK_BATCH_LAST_NODE_IDX_LIST, batch_last_node_idx_list),
                    (DK_BATCH_UNIQUE_STR_ID_SET, batch_unique_str_id_set),
                    (DK_BATCH_TARGET_TSR, batch_tgt),
                    (DK_BATCH_FLOPS, batch_flops),
                    ("BATCH_NAMES", batch_names),
                ])
                if len(batch_unique_str_id_set) < len(batch_list):
                    print("Collected {} unique features but batch size is {}".format(len(batch_unique_str_id_set),
                                                                                     len(batch_list)))
                idx += self.batch_size
                self.batches.append(batch)
                if bar is not None: bar.update(1)
        if bar is not None: bar.close()
        self.shuffle()

    def _build_psc_batches(self, data):
        self.batches = []

        bins = collections.defaultdict(list)
        for g_p, g_s, g_c, mean_tgt, dev_tgt in data:
            assert isinstance(g_p, ComputeGraph) and isinstance(g_s, ComputeGraph) and isinstance(g_c, ComputeGraph)
            overall_n_nodes = len(g_p.nodes) + len(g_s.nodes) + len(g_c.nodes)
            overall_reg_node_start_idx = g_p.regular_node_start_idx + g_s.regular_node_start_idx + g_c.regular_node_start_idx
            key = "{}|{}".format(overall_n_nodes, overall_reg_node_start_idx)
            bins[key].append((g_p, g_s, g_c, mean_tgt, dev_tgt))

        n_batches = 0
        for _, instances in bins.items():
            n_batches += math.ceil(len(instances) / self.batch_size)

        bar = None
        if self.verbose:
            bar = tqdm(total=n_batches, desc="Building batches", ascii=True)
        for k, data_list in bins.items():
            idx = 0
            while idx < len(data_list):
                batch_list = data_list[idx:idx + self.batch_size]
                batch_regular_inds = [[], [], []]
                batch_regular_shapes = [[], [], []]
                batch_weighted_inds = [[], [], []]
                batch_weighted_shapes = [[], [], []]
                batch_weighted_kernels = [[], [], []]
                batch_weighted_bias = [[], [], []]
                batch_edge_list = [[], [], []] 
                #batch_names = []  # Remove for now
                batch_tgt = [[], []]  # Mean and S.Dev
                batch_last_node_idx_list = [[], [], []]
                batch_reg_node_offset = int(k[k.find("|") + 1:])
                batch_w_psc_offset_list = [[], []]
                batch_r_psc_offset_list = [[], []]
                batch_unique_str_id_set = set()

                for inst in batch_list:
                    g_p, g_s, g_c, mean_tgt, dev_tgt = inst
                    assert isinstance(g_p, ComputeGraph) and isinstance(g_s, ComputeGraph) and isinstance(g_c, ComputeGraph)
                    for i, g_i in enumerate([g_p, g_s, g_c]):
                        regular_node_inds, regular_node_shapes, weighted_node_inds, \
                            weighted_node_shapes, weighted_node_kernels, weighted_node_bias, \
                                edge_list = g_i.get_gnn_features()
                        if len(regular_node_inds) > 0:
                            batch_regular_inds[i].append(torch.LongTensor(regular_node_inds).unsqueeze(0))
                            batch_regular_shapes[i].append(torch.FloatTensor(regular_node_shapes).unsqueeze(0))
                        else:
                            batch_regular_inds[i].append(None)
                            batch_regular_shapes[i].append(None)
                        if len(weighted_node_inds) > 0:
                            batch_weighted_inds[i].append(torch.LongTensor(weighted_node_inds).unsqueeze(0))
                            batch_weighted_shapes[i].append(torch.FloatTensor(weighted_node_shapes).unsqueeze(0))
                            batch_weighted_kernels[i].append(torch.LongTensor(weighted_node_kernels).unsqueeze(0))
                            batch_weighted_bias[i].append(torch.FloatTensor(weighted_node_bias).unsqueeze(0))
                        else:
                            batch_weighted_inds[i].append(None)
                            batch_weighted_shapes[i].append(None)
                            batch_weighted_kernels[i].append(None)
                            batch_weighted_bias[i].append(None)
                        batch_edge_list[i].append(torch.LongTensor(edge_list))
                        prev_len = batch_last_node_idx_list[i][-1] if len(batch_last_node_idx_list[i]) > 0 else -1
                        batch_last_node_idx_list[i].append(prev_len + len(weighted_node_inds) + len(regular_node_inds))
                        if i == 0:
                            batch_tgt[0].append(mean_tgt)
                            batch_tgt[1].append(dev_tgt)
                            batch_w_psc_offset_list[0].append(len(weighted_node_inds))
                            batch_r_psc_offset_list[0].append(len(regular_node_inds))

                            graph_id = _get_graph_id(regular_node_inds, regular_node_shapes,
                                                     weighted_node_inds, weighted_node_shapes, weighted_node_kernels,
                                                     weighted_node_bias,
                                                     edge_list_to_edge_pairs(edge_list), strategy="simple")
                            batch_unique_str_id_set.add(graph_id)
                        elif i == 1:
                            batch_w_psc_offset_list[1].append(batch_w_psc_offset_list[0][-1] + len(weighted_node_inds))
                            batch_r_psc_offset_list[1].append(batch_r_psc_offset_list[0][-1] + len(regular_node_inds))
                if batch_tgt[1][0] is not None:
                    batch_tgt = torch.cat([torch.FloatTensor(batch_tgt[0]).unsqueeze(1), torch.FloatTensor(batch_tgt[1]).unsqueeze(1)], axis=1)
                else:
                    batch_tgt = torch.FloatTensor(batch_tgt[0])

                batch_regular_inds = self._merge_psc(batch_regular_inds)
                batch_regular_shapes = self._merge_psc(batch_regular_shapes)

                if batch_reg_node_offset > 0:
                    batch_weighted_inds = self._merge_psc(batch_weighted_inds)
                    batch_weighted_shapes = self._merge_psc(batch_weighted_shapes)
                    batch_weighted_kernels = self._merge_psc(batch_weighted_kernels)
                    batch_weighted_bias = self._merge_psc(batch_weighted_bias)
                else:
                    batch_weighted_inds = None
                    batch_weighted_shapes = None
                    batch_weighted_kernels = None
                    batch_weighted_bias = None

                batch = UniqueDict([
                    (DK_BATCH_SIZE, len(batch_list)),
                    (DK_BATCH_CG_REGULAR_IDX, batch_regular_inds),
                    (DK_BATCH_CG_REGULAR_SHAPES, batch_regular_shapes),
                    (DK_BATCH_CG_WEIGHTED_IDX, batch_weighted_inds),
                    (DK_BATCH_CG_WEIGHTED_SHAPES, batch_weighted_shapes),
                    (DK_BATCH_CG_WEIGHTED_KERNELS, batch_weighted_kernels),
                    (DK_BATCH_CG_WEIGHTED_BIAS, batch_weighted_bias),
                    (DK_BATCH_EDGE_TSR_LIST, batch_edge_list),
                    (DK_BATCH_LAST_NODE_IDX_LIST, batch_last_node_idx_list),
                    (DK_BATCH_UNIQUE_STR_ID_SET, batch_unique_str_id_set),
                    (DK_BATCH_REG_NODE_OFFSET, batch_reg_node_offset),
                    (DK_BATCH_WEIGHTED_OFFSETS, batch_w_psc_offset_list),
                    (DK_BATCH_REGULAR_OFFSETS, batch_r_psc_offset_list),
                    (DK_BATCH_TARGET_TSR, batch_tgt),
                ])
                idx += self.batch_size
                self.batches.append(batch)
                if bar is not None: bar.update(1)
        if bar is not None: bar.close()
        self.shuffle()

    def _merge_psc(self, tensor_list_of_lists):
        psc_list = []
        for i in range(len(tensor_list_of_lists[0])):
            concat_list = []
            for j in range(3):
                if tensor_list_of_lists[j][i] is not None:
                    concat_list.append(tensor_list_of_lists[j][i])
            if len(concat_list) > 0:
                psc_list.append(
                    torch.cat(concat_list, dim=1)
                )
        return torch.cat(psc_list, dim=0)

    def shuffle(self):
        random.shuffle(self.batches)

    def __iter__(self):
        self.shuffle()
        self.curr_batch_idx = 0
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        if self.curr_batch_idx >= len(self.batches):
            self.shuffle()
            self.curr_batch_idx = 0
            raise StopIteration()
        next_batch = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1
        return next_batch

    def get_overlapping_data_count(self, loader):
        if not isinstance(loader, CGRegressDataLoader):
            print("Type mismatch, no overlaps by default")
            return 0
        n_unique_overlaps = 0
        my_data = set()
        for batch in self:
            batch_unique_str_id_set = batch[DK_BATCH_UNIQUE_STR_ID_SET]
            for str_id in batch_unique_str_id_set:
                my_data.add(str_id)
        for batch in loader:
            batch_unique_str_id_set = batch[DK_BATCH_UNIQUE_STR_ID_SET]
            for str_id in batch_unique_str_id_set:
                if str_id in my_data:
                    n_unique_overlaps += 1
        return n_unique_overlaps


def _get_op_nn_graphs(cg, target_map, n_hops=3,
                      include_op_keywords=None,
                      exclude_op_keywords=None,
                      use_bi_dir_edges=False):
    nn_graphs = []
    targets = []
    nodes = cg.nodes
    adj_dict = collections.defaultdict(set)
    src2dst_ids = collections.defaultdict(set)
    for si, di in cg.edge_pairs:
        sid, did = nodes[si].str_id, nodes[di].str_id
        adj_dict[sid].add(did)
        adj_dict[did].add(sid)
        src2dst_ids[sid].add(did)

    name2node = {}
    id2node = {}
    for n in nodes:
        name = n.str_id.split("|")[-1]
        name2node[name] = n
        id2node[n.str_id] = n

    nn_data = []
    for t_name, val in target_map.items():
        t_name = t_name.strip().lower()
        op_label = t_name.split("/")[-1]
        if include_op_keywords is not None and \
            all(w not in op_label for w in include_op_keywords):
            continue
        if exclude_op_keywords is not None and \
            any(w in op_label for w in exclude_op_keywords):
            continue
        if t_name in name2node:
            pivot = name2node[t_name]
            neighbor_ids = bfs_neighbor_nodes(adj_dict, pivot.str_id, n_hops)
            assert len(neighbor_ids) > 1
            nn_data.append((neighbor_ids, pivot))
            targets.append(val)

    for gi, (neighbor_ids, pivot) in enumerate(nn_data):
        nn_src2dst_ids = collections.defaultdict(set)
        for sid in neighbor_ids:
            for nid in src2dst_ids[sid]:
                if nid in neighbor_ids:
                    nn_src2dst_ids[sid].add(nid)
        nn_nodes = [id2node[_id] for _id in neighbor_ids]
        nn_weighted_nodes = [n for n in nn_nodes if isinstance(n, WeightedNode)]
        nn_regular_nodes = [n for n in nn_nodes if isinstance(n, RegularNode)]
        nn_weighted_nodes.sort(key=lambda _n: (_n.op_type_idx, _n.str_id))
        nn_regular_nodes.sort(key=lambda _n: (_n.op_type_idx, _n.str_id))
        nn_nodes = nn_weighted_nodes + nn_regular_nodes

        regular_inds = [n.op_type_idx for n in nn_regular_nodes]
        regular_shapes = [list(n.resolution) for n in nn_regular_nodes]
        for lv in regular_shapes:
            lv[0] = float(lv[0]) / cg.max_derived_H
            lv[1] = float(lv[1]) / cg.max_derived_H
            lv[2] = float(lv[2]) / cg.max_derived_W
            lv[3] = float(lv[3]) / cg.max_derived_W
            lv[4] = float(lv[4]) / cg.max_hidden_size
            lv[5] = float(lv[5]) / cg.max_hidden_size
            assert all(0 <= v <= 1 for v in lv)
        weighted_node_inds = [n.op_type_idx for n in nn_weighted_nodes]
        weighted_node_hw = [list(n.resolution[:-2]) for n in nn_weighted_nodes]
        weighted_node_channels = [n.shape[:2] for n in nn_weighted_nodes]
        weighted_node_bias = [[0, 1] if n.metadata is not None and \
                                        "use_bias" in n.metadata and \
                                        n.metadata["use_bias"]
                              else [1, 0] for n in nn_weighted_nodes]
        weighted_node_shapes = []
        for hw, ch in zip(weighted_node_hw, weighted_node_channels):
            hw[0] = float(hw[0]) / cg.max_derived_H
            hw[1] = float(hw[1]) / cg.max_derived_H
            hw[2] = float(hw[2]) / cg.max_derived_W
            hw[3] = float(hw[3]) / cg.max_derived_W
            lv = hw + ch
            weighted_node_shapes.append(lv)
            assert all(0 <= v <= 1 for v in lv)
        weighted_node_kernels = [n.shape[2:] for n in nn_weighted_nodes]
        pivot_idx = None
        for i, n in enumerate(nn_nodes):
            if n.str_id == pivot.str_id:
                pivot_idx = i
                break
        assert pivot_idx is not None

        edge_pairs = []
        node_id2idx = {n.str_id: i for i, n in enumerate(nn_nodes)}
        for src_id, dst_ids in nn_src2dst_ids.items():
            for dst_id in list(dst_ids):
                edge_pairs.append( (node_id2idx[src_id], node_id2idx[dst_id]) )
                if use_bi_dir_edges:
                    edge_pairs.append( (node_id2idx[dst_id], node_id2idx[src_id]) )
        pivot_name = pivot.str_id.split("|")[-1]
        graph_id = _get_graph_id(regular_inds, regular_shapes,
                                 weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                                 sorted(edge_pairs),
                                 pivot_idx=pivot_idx, pivot_name=pivot_name, strategy="simple")
        nn_graphs.append( (regular_inds, regular_shapes,
                           weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                           edge_pairs, pivot_idx, pivot_name, graph_id) )
        _nn_graph_to_cg_map[graph_id].append( (cg, targets[gi]) )
        
    assert len(nn_graphs) == len(targets) > 0
    return nn_graphs, targets
