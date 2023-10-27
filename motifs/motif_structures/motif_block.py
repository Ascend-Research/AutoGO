import pickle
import pandas as pd
import networkx as nx
from motifs.motif_structures.utils import nx_to_cg
import time
import copy
import random
import sentencepiece as spm
from typing import List
from model_src.comp_graph.tf_comp_graph import WeightedNode, RegularNode
from constants import OPS
from params import *
from MILP.milp_model import resolution_propagate
from encoder.utils import *
from .utils import *
from model_src.comp_graph.tf_comp_graph import ComputeGraph


def auto_set_nodes(nx_dg: nx.DiGraph, reset_ids=False):
    weighted, regular = [], []
    updated_nx_dg = copy.deepcopy(nx_dg)
    for node in updated_nx_dg.nodes:
        if reset_ids:
            updated_nx_dg.nodes[node]['str_id'] = str(node)
        if updated_nx_dg.nodes[node]['type_idx']:
            wn = WeightedNode(str_id=updated_nx_dg.nodes[node]['str_id'], shape=updated_nx_dg.nodes[node]['shape'],
                              op_type_idx=updated_nx_dg.nodes[node]['op_type_idx'],
                              label=updated_nx_dg.nodes[node]['label'])
            if 'resolution' in updated_nx_dg.nodes[node].keys():
                wn.resolution = updated_nx_dg.nodes[node]['resolution']
            if 'strides' in updated_nx_dg.nodes[node].keys():
                wn.strides = updated_nx_dg.nodes[node]['strides']
            if 'metadata' in updated_nx_dg.nodes[node].keys():
                wn.metadata = updated_nx_dg.nodes[node]['metadata']
            weighted.append(wn)
        else:
            rn = RegularNode(str_id=updated_nx_dg.nodes[node]['str_id'], op_type_idx=updated_nx_dg.nodes[node]
            ['op_type_idx'], label=updated_nx_dg.nodes[node]['label'])
            if 'resolution' in updated_nx_dg.nodes[node].keys():
                rn.resolution = updated_nx_dg.nodes[node]['resolution']
            if 'strides' in updated_nx_dg.nodes[node].keys():
                rn.strides = updated_nx_dg.nodes[node]['strides']
            if 'metadata' in updated_nx_dg.nodes[node].keys():
                rn.metadata = updated_nx_dg.nodes[node]['metadata']
            regular.append(rn)
    updated_nx_dg.graph['weighted_nodes'] = weighted
    updated_nx_dg.graph['regular_nodes'] = regular
    updated_nx_dg.graph['n_regular_nodes'] = len(regular)
    updated_nx_dg.graph['n_weighted_nodes'] = len(weighted)
    return updated_nx_dg


class Mutator:
    def __init__(self,
                 model_family,
                 graph_encoder_family,
                 graph_encoder_path,
                 tokenizer_path,
                 segment_database_path,
                 max_candidate_per_iter,
                 max_target_per_iter,
                 num_consecutive_segs=1,
                 max_h=10,
                 propagation_type='all_nodes',
                 predictor_type="k_adaptor",
                 num_classes=10,
                 invalid_ops=[],
                 mutation_unit='segment',
                 acc_checker=None,
                 flops_checker=None,
                 has_budget=False,
                 segment_sparsity=0.,
                 milp_verbose=False):

        self.model_family = model_family
        self.graph_encoder_family = graph_encoder_family

        with open(graph_encoder_path, "rb") as f:
            self.graph_encoder = pickle.load(f)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        with open(segment_database_path, "rb") as f:
            self.segment_database = pickle.load(f)
        
        if mutation_unit != "op":
            self.segment_sparsity = 1.0 - segment_sparsity
        else:
            self.segment_sparsity = None

        self.max_h = max_h
        self.max_candidate_per_iter = max_candidate_per_iter
        self.max_target_per_iter = max_target_per_iter
        self.num_consecutive_segs = num_consecutive_segs

        self.propagation_type = propagation_type
        self.num_classes = num_classes
        self.invalid_ops = invalid_ops
        self.mutation_unit = mutation_unit
        self.log_f = print
        self.predictor_type = predictor_type

        self.total_milp_calls = 0
        self.milp_dict = {'success': 0,
                          'success_time': [],
                          'fail': 0,
                          'fail_time': [],
                          'conv': 0,
                          'pool': 0,
                          'concat': 0}
        self.total_input_cg_list = []
        self.total_candidate_list = []
        self.total_target_list = []
        self.total_mutated_arch_list = []
        self.total_PSC_list = []
        self.baseline_acc_list = []
        self.total_PSC_acc_list = []

        self.acc_checker = acc_checker
        self.flops_checker = flops_checker

        self.has_budget = has_budget

        if self.mutation_unit == 'segment' or self.mutation_unit == "op":
            def get_literal_res(res_str):
                from ast import literal_eval
                res = literal_eval(res_str)
                res = [tuple(r) for r in res]
                return tuple(res)

            self.segment_database['res_in_tuple'] = self.segment_database['res_in'].map(
                lambda row: get_literal_res(row))
            self.segment_database['res_out_tuple'] = self.segment_database['res_out'].map(
                lambda row: get_literal_res(row))
            self.groups = self.segment_database.groupby(['res_in_tuple', 'res_out_tuple'])

            self.segment_database_dict = {}
            for resolutions, group in self.groups:
                current_in_resolution_key = sorted([self.extract_in_resolution(res) for res in resolutions[0]])
                current_out_resolution_key = sorted([self.extract_out_resolution(res) for res in resolutions[1]])
                resolution_key = (tuple(current_in_resolution_key), tuple(current_out_resolution_key))
                if resolution_key in self.segment_database_dict.keys():
                    self.segment_database_dict[resolution_key] = \
                        pd.concat([self.segment_database_dict[resolution_key], group])
                else:
                    self.segment_database_dict[resolution_key] = group

                num_input_output_key = (len(resolutions[0]), len(resolutions[1]))
                if num_input_output_key in self.segment_database_dict.keys():
                    self.segment_database_dict[num_input_output_key] = \
                        pd.concat([self.segment_database_dict[num_input_output_key], group])
                else:
                    self.segment_database_dict[num_input_output_key] = group
            for key in self.segment_database_dict.keys():
                self.segment_database_dict[key] = self.segment_database_dict[key].to_dict('records')

        else:
            raise NotImplementedError("%s is not supported" % self.mutation_unit)
        self.milp_verbose = milp_verbose

    @staticmethod
    def extract_out_resolution(res):
        return res[1], res[3], res[5]

    @staticmethod
    def extract_in_resolution(res):
        return res[0], res[2], res[4]

    @staticmethod
    def validate_arch(arch: nx.DiGraph):
        if len(list(nx.simple_cycles(arch))) != 0:
            return False
        return True

    @staticmethod
    def get_subgraph_resolutions(sg: nx.DiGraph, unique=False):
        in_nodes, out_nodes = [], []
        for i, n in sg.nodes(data=True):
            if sg.in_degree(i) == 0:
                in_nodes.append(n)
            if sg.out_degree(i) == 0:
                out_nodes.append(n)
        if unique:
            _res_in, _res_out = [], []
            for n in in_nodes:
                if n['resolution'] not in _res_in:
                    _res_in.append(n['resolution'])
            for n in out_nodes:
                if n['resolution'] not in _res_out:
                    _res_out.append(n['resolution'])
        else:
            _res_in = [(n['resolution']) for n in in_nodes]
            _res_out = [(n['resolution']) for n in out_nodes]
        return {'in': _res_in, 'out': _res_out}

    def validate_segment(self, motif: nx.DiGraph):
        for node_id in motif.nodes:
            if motif.nodes[node_id]['op_name'] in ['input', 'output', 'mean', 'matmul']:
                return False
            if self.graph_encoder_family == "combined":
                if motif.nodes[node_id]['op_name'] not in \
                        OPS[self.model_family].intersection(
                            OPS["hiaml"].union(OPS["two_path"], OPS["inception"], OPS["nb201"], OPS["nb101"])):
                    return False
            elif motif.nodes[node_id]['op_name'] not in \
                    OPS[self.model_family].intersection(OPS[self.graph_encoder_family]):
                return False
        return True

    def find_good_match_res(self, source_segments_parent_resolutions,
                            source_segment_child_resolutions,
                            candidate):

        good_target_list = []

        source_blocks_resolutions_in = [self.extract_out_resolution(
            source_segment_parent_resolution) for source_segment_parent_resolution in
            source_segments_parent_resolutions]
        source_blocks_resolutions_out = [self.extract_in_resolution(
            source_segment_child_resolution) for source_segment_child_resolution in
            source_segment_child_resolutions]

        resolution_key = (tuple(source_blocks_resolutions_in), tuple(source_blocks_resolutions_out))
        dimension_key = (len(source_blocks_resolutions_in), len(source_blocks_resolutions_out))

        already_selected_segments = []
        
        if resolution_key in self.segment_database_dict.keys():
            target_list = self.segment_database_dict[resolution_key]
            already_selected_segments = [(target['seg_str'], target['num_inputs'], target['num_outputs']) for target in target_list]

            for transformed_target in target_list:
                input_node = [node for node in transformed_target['subgraph'].nodes
                              if transformed_target['subgraph'].in_degree(node) == 0]
                input_resolution = [extract_in_resolution(transformed_target['subgraph']._node[x]["resolution"])
                                    for x in input_node]
                if sorted(source_blocks_resolutions_in) != sorted(input_resolution):
                    transformed_target['subgraph'] = nx_to_cg(transformed_target['subgraph'])
                    transformed_target['subgraph'] = cg_to_nx(transformed_target['subgraph'])[0]

            good_target_list += [target for target in target_list if self.validate_segment(target["subgraph"])]

        random.shuffle(good_target_list)
        good_target_list = good_target_list[:self.max_target_per_iter // 2]
        num_needed_target = max(0, self.max_target_per_iter - len(good_target_list))

        if dimension_key in self.segment_database_dict.keys() \
                and self.mutation_unit == "segment" and num_needed_target > 0:

            target_list_ = self.segment_database_dict[dimension_key]
            target_list = []
            for target in target_list_:
                if (target['seg_str'], target['num_inputs'], target['num_outputs']) not in already_selected_segments:
                    already_selected_segments.append((target['seg_str'], target['num_inputs'], target['num_outputs']))
                    target_list.append(target)
            target_list = [target for target in target_list if self.validate_segment(target["subgraph"])]
            target_list = [target for target in target_list if
                           len(target["subgraph"].nodes) > 1 and len(target["subgraph"].edges) > 1]

            if self.predictor_type == "PSC_predictor":
                PSC_list = []
                for element in target_list:
                    arch = element["subgraph"]
                    for node in arch.nodes:
                        arch._node[node]["str_id"] = str(node)
                    arch.graph['src_id2dst_ids_dict'] = get_src_id2dst_ids_dict(arch)
                    arch = auto_set_nodes(arch)
                    PSC_list.append((candidate["predecessor_subgraph"], arch, candidate["successor_subgraph"]))
                
                PSC_list = [tuple(nx_to_cg(list(element))) for element in PSC_list]
                acc_list = self.acc_checker._get_acc(PSC_list)
                top_index = sorted(range(len(acc_list)), key=lambda i: acc_list[i])[-min(num_needed_target, len(acc_list)):]
                target_list = [target_list[i] for i in top_index]
            else:
                target_list = random.sample(target_list, min(len(target_list), num_needed_target))

            if self.propagation_type == "milp":
                target_list = copy.deepcopy(target_list)
                count_success = 0
                for target in target_list:
                    self.total_milp_calls += 1
                    n_conv, n_pool, n_concat = parse_subgraph_res_nodes(target['subgraph'])
                    self.milp_dict['conv'] += n_conv
                    self.milp_dict['pool'] += n_pool
                    self.milp_dict['concat'] += n_concat
                    transformed_target = self.get_target_with_correct_resolution(target,
                                                                                 source_blocks_resolutions_in,
                                                                                 source_blocks_resolutions_out)

                    if transformed_target['subgraph'] is not None:
                        count_success += 1
                        good_target_list.append(transformed_target)
                        input_node = [node for node in transformed_target['subgraph'].nodes
                                      if transformed_target['subgraph'].in_degree(node) == 0]
                        input_resolution = [extract_in_resolution(transformed_target['subgraph']._node[x]["resolution"])
                                            for x in input_node]
                        assert sorted(source_blocks_resolutions_in) == sorted(input_resolution), "Unsuccessful"
                if self.milp_verbose:
                    print(f"Successful MILP calls: {count_success}/{len(target_list)}")
            else:
                raise NotImplementedError("%s is not supported" % self.propagation_type)

        if self.mutation_unit == "op":
            good_target_list = [item for item in good_target_list if len(item["subgraph"]) == 1]

        return good_target_list

    def get_target_with_correct_resolution(self, target, source_blocks_resolutions_in, source_blocks_resolutions_out):
        target_subgraph = target["subgraph"]
        input_ids = [i for i, n in target_subgraph.nodes(data=True) if target_subgraph.in_degree(i) == 0]
        output_ids = [i for i, n in target_subgraph.nodes(data=True) if target_subgraph.out_degree(i) == 0]

        random.shuffle(input_ids)
        random.shuffle(output_ids)

        input2res, output2res = {}, {}
        for value, key in enumerate(input_ids):
            target_subgraph.nodes[key]['resolution'] = (source_blocks_resolutions_in[value][0],
                                                        target_subgraph.nodes[key]['resolution'][1],
                                                        source_blocks_resolutions_in[value][1],
                                                        target_subgraph.nodes[key]['resolution'][3],
                                                        source_blocks_resolutions_in[value][2],
                                                        target_subgraph.nodes[key]['resolution'][5])

        for value, key in enumerate(output_ids):
            output2res[key] = source_blocks_resolutions_out[value]
        input_2 = []
        for i, n in target_subgraph.nodes(data=True):
            if n['op_name'] == 'concat' and target_subgraph.in_degree(i) == 0:
                target_subgraph.nodes[i]['op_name'] = 'input2'
                input_2.append(i)

        target_subgraph, success = self.virtual_forward_propagation(target_subgraph, input_ids=input_ids,
                                                                    return_none=True)

        for i in input_2:
            if target_subgraph.nodes[i]['op_name'] == 'input2' and target_subgraph.in_degree(i) == 0:
                target_subgraph.nodes[i]['op_name'] = 'concat'
            else:
                raise Exception('Invalid transformation of input concat node')

        if success:
            start = time.time()
            _, target_subgraph, _ = resolution_propagate(target_subgraph, output2res)
            milp_time = time.time() - start
            if target_subgraph is None:
                self.milp_dict['fail'] += 1
                self.milp_dict['fail_time'].append(milp_time)
            else:
                self.milp_dict['success'] += 1
                self.milp_dict['success_time'].append(milp_time)
        else:
            target_subgraph = None
        target["subgraph"] = target_subgraph

        return target

    def virtual_forward_propagation(self, nx_dg: nx.DiGraph, input_ids=None, return_none=False):
        impact_ops = ['conv2d', 'avgpool',
                      'maxpool', 'mean', 'matmul', 'concat', 'depthwise']
        fixed_nx_dg = copy.deepcopy(nx_dg)

        if input_ids is None:
            input_ids = []
            for i, node in nx_dg.nodes(data=True):
                if node['op_name'] == 'input':
                    input_ids.append(i)
                    break
        else:
            for input_id in input_ids:
                in_node = fixed_nx_dg.nodes[input_id]
                Ho = in_node['resolution'][0]
                Wo = in_node['resolution'][2]
                Co = in_node['resolution'][4]
                if in_node['op_name'] == 'concat':
                    concat_parents = [
                        v for v in fixed_nx_dg.predecessors(input_id)]
                    if len(concat_parents) > 0:
                        Co = sum([fixed_nx_dg.nodes[v]['resolution'][-1]
                                  for v in concat_parents])
                elif in_node['op_name'] in ['conv2d', 'avgpool', 'maxpool', 'depthwise']:
                    if in_node['metadata']['padding'] != 'same':
                        raise Exception('Invalid Padding')
                    else:
                        Ho /= in_node['strides'][1]
                        Wo /= in_node['strides'][2]
                        try:
                            Ho = int(Ho)
                            Wo = int(Wo)
                        except:
                            raise Exception('Invalid H/W')
                        if Ho < 1 or Wo < 1:
                            success = False
                            self.log_f(
                                'Resolution propagation was unsuccessful: Invalid H/W, reverting the mutation...')
                            break
                elif in_node['op_name'] == 'mean':
                    Ho = 1
                    Wo = 1
                elif in_node['op_name'] == 'matmul':
                    Co = self.num_classes
                fixed_nx_dg.nodes[input_id]['resolution'] = (
                    in_node['resolution'][0], Ho, in_node['resolution'][2], Wo, in_node['resolution'][4], Co)

        success = True
        fixed_tuples = []
        fixed = []
        children = []

        for input_id in input_ids:
            i = input_id
            visited_tuples = []
            visited = []

            successors = [(i, s) for s in fixed_nx_dg.successors(i)]
            children.extend(successors)
            visited_tuples.extend(successors)

            while len(children) > 0:
                p, ch = children.pop(0)

                parent = fixed_nx_dg.nodes[p]
                child = fixed_nx_dg.nodes[ch]

                expected_input = extract_out_resolution(
                    parent['resolution'])
                actual_input = extract_in_resolution(
                    child['resolution'])

                if child['op_name'] == 'concat':
                    concat_parents = [v for v in fixed_nx_dg.predecessors(ch)]
                    if len(concat_parents) > 0:
                        C = sum([fixed_nx_dg.nodes[v]['resolution'][-1]
                                 for v in concat_parents])
                    expected_HW = extract_out_resolution(
                        fixed_nx_dg.nodes[concat_parents[0]]['resolution'])[:-1]

                if (expected_input != actual_input and child['op_name'] != 'concat') or \
                        (expected_input[:-1] != actual_input[:-1] and child['op_name'] == 'concat'):
                    if ch in fixed or ch in visited:
                        success = False
                        if self.milp_verbose:
                            self.log_f(
                                'Resolution propagation was unsuccessful, reverting the mutation...')
                        break

                    H = parent['resolution'][1]
                    W = parent['resolution'][3]
                    if child['op_name'] != 'concat':
                        C = parent['resolution'][5]

                    if child['op_name'] in impact_ops:

                        if child['op_name'] in ['conv2d', 'avgpool', 'maxpool', 'depthwise']:
                            if child['metadata']['padding'] != 'same':
                                raise Exception('Invalid Padding')
                            else:
                                H /= child['strides'][1]
                                W /= child['strides'][2]
                                try:
                                    H = int(H)
                                    W = int(W)
                                except:
                                    raise Exception('Invalid H/W')
                                if H < 1 or W < 1:
                                    success = False
                                    self.log_f(
                                        'Resolution propagation was unsuccessful: Invalid H/W, reverting the mutation...')
                                    break
                        elif child['op_name'] == 'mean':
                            H = 1
                            W = 1
                        elif child['op_name'] == 'matmul':
                            C = self.num_classes
                    if child['op_name'] == 'concat':
                        fixed_nx_dg.nodes[ch]['resolution'] = (
                            expected_HW[0], expected_HW[0], expected_HW[1], expected_HW[1], C, C)
                    else:
                        fixed_nx_dg.nodes[ch]['resolution'] = (
                            expected_input[0], H, expected_input[1], W, expected_input[2], C)
                for gp in fixed_nx_dg.successors(ch):
                    tuple_ = (ch, gp)
                    if tuple_ not in visited_tuples:
                        children.append(tuple_)
                        visited_tuples.append(tuple_)

                fixed.append(ch)
                visited.append(ch)
                fixed_tuples.append((p, ch))

        fixed_nx_dg = auto_set_nodes(
            fixed_nx_dg, reset_ids=False)
        if return_none and not success:
            fixed_nx_dg = None
        return fixed_nx_dg, success

    def replace_segment(self, graph, candidate_list: List[nx.DiGraph], target_segment_list: List[List[nx.DiGraph]]):
        """
        Find new architectures by replacing each source candidate segment with the target segment
        """
        if len(candidate_list) == 0 or len(target_segment_list) == 0:
            return [], []

        temp_zip = list(zip(candidate_list, target_segment_list))
        random.shuffle(temp_zip)
        candidate_list, target_segment_list = zip(*temp_zip)

        mutated_architecture_list = []
        mutated_architecture_PSC_list = []
        for i in range(min(self.max_candidate_per_iter, len(candidate_list))):
            source_candidate = candidate_list[i]
            candidate_subgraph = source_candidate["subgraph"]
            candidate_parent_nodes = source_candidate["parent_nodes"]
            candidate_child_nodes = source_candidate["child_nodes"]
            target_segments = target_segment_list[i]
            random.shuffle(target_segments)

            success = True
            for j in range(min(self.max_target_per_iter, len(target_segments))):
                target = target_segments[j]["subgraph"]
                mutated_architecture = copy.deepcopy(graph)

                max_node_id = max(graph.nodes)
                renaming_dict = {id: id + max_node_id + 1 for id in target.nodes}
                target = nx.relabel_nodes(target, renaming_dict)

                mutated_architecture = nx.compose(mutated_architecture, target)

                target_input_node = [node for node in target.nodes if target.in_degree(node) == 0]
                target_output_node = [node for node in target.nodes if target.out_degree(node) == 0]
                random.shuffle(target_input_node)
                random.shuffle(target_output_node)

                used_target_nodes = []
                for node in candidate_parent_nodes:
                    suitable_target_input_node_list = [target_node for target_node in target_input_node
                                                       if
                                                       target_node not in used_target_nodes and self.extract_in_resolution(
                                                           mutated_architecture._node[target_node]["resolution"]) ==
                                                       self.extract_out_resolution(
                                                           mutated_architecture._node[node]["resolution"])]

                    if len(suitable_target_input_node_list) == 0:
                        mutated_architecture = copy.deepcopy(graph)
                        success = False
                        break

                    chosen_input_target_node = random.choice(suitable_target_input_node_list)
                    used_target_nodes.append(chosen_input_target_node)
                    mutated_architecture.add_edge(node, chosen_input_target_node)

                used_target_nodes = []
                for node in candidate_child_nodes:

                    if not success:
                        break

                    suitable_target_output_node_list = [target_node for target_node in target_output_node
                                                        if
                                                        target_node not in used_target_nodes and self.extract_out_resolution(
                                                            mutated_architecture._node[target_node]["resolution"]) ==
                                                        self.extract_in_resolution(
                                                            mutated_architecture._node[node]["resolution"])]

                    if len(suitable_target_output_node_list) == 0:
                        mutated_architecture = copy.deepcopy(graph)
                        success = False
                        break

                    chosen_output_target_node = random.choice(suitable_target_output_node_list)
                    used_target_nodes.append(chosen_output_target_node)
                    mutated_architecture.add_edge(chosen_output_target_node, node)

                if success:
                    mutated_architecture.remove_nodes_from(candidate_subgraph.nodes)
                    mutated_architecture.graph["input_shape"] = mutated_architecture.graph["input_shape"][0]
                    mutated_architecture_list.append(mutated_architecture)
                    mutated_architecture_PSC_list.append((source_candidate["predecessor_subgraph"],
                                                          candidate_subgraph,
                                                          target,
                                                          source_candidate["successor_subgraph"],))
                    if len([x for x in mutated_architecture.nodes() if mutated_architecture.out_degree(x) == 0]) > 1:
                        raise NotImplementedError

        return mutated_architecture_list, mutated_architecture_PSC_list

    def get_bad_candidate(self, candidate_list):
        """
        Find the bad candidates
        """
        bad_candidate_list = []
        for s in candidate_list:
            if not self.validate_segment(s['subgraph']) or \
                    (len(s['subgraph']) < 1 and self.mutation_unit == "segment" and "resnet" not in self.model_family):
                bad_candidate_list.append(s)

        return bad_candidate_list

    def get_candidate_list(self, cg, perform_filtering=True):
        if self.mutation_unit == 'segment':
            found_candidate_list = get_segment2subgraph_list_with_ids(cg, self.graph_encoder, self.tokenizer, self.num_consecutive_segs, self.segment_sparsity)
        elif self.mutation_unit == 'op':
            class SingleTokenizer:
                def __init__(self):
                    pass

                def encode_as_pieces(self, text):
                    return [x for x in text]

            found_candidate_list = get_segment2subgraph_list_with_ids(cg, self.graph_encoder, SingleTokenizer(), segment_sparsity=self.segment_sparsity)
        else:
            raise NotImplementedError("%s is not supported" % self.mutation_unit)

        bad_candidate_list = self.get_bad_candidate(found_candidate_list)
        candidate_list = [x for x in found_candidate_list if x not in bad_candidate_list]

        if not perform_filtering:
            return candidate_list

        flops_list = []

        for i in range(len(candidate_list)):
            arch = candidate_list[i]["segment_subgraph"]
            for node in arch.nodes:
                arch._node[node]["str_id"] = str(node)
            arch.graph['src_id2dst_ids_dict'] = get_src_id2dst_ids_dict(arch)
            arch = auto_set_nodes(arch)
            candidate_list[i]["segment_subgraph"] = arch

            arch = candidate_list[i]["subgraph"]
            for node in arch.nodes:
                arch._node[node]["str_id"] = str(node)
            arch.graph['src_id2dst_ids_dict'] = get_src_id2dst_ids_dict(arch)
            arch = auto_set_nodes(arch)
            candidate_list[i]["subgraph"] = arch

            try:
                flops = self.flops_checker.get_flops(nx_to_cg(arch)[0])
            except:
                flops = 0
            flops_list.append(flops)

        top_flops_index = sorted(range(len(flops_list)), key=lambda i: flops_list[i], reverse=True)
        flops_indices = top_flops_index[-1 * self.max_candidate_per_iter// 2:]
        top_flops_candidate_list = [candidate_list[index] for index in
                                    flops_indices]

        candidate_list = [element for i, element in enumerate(candidate_list) if i not in flops_indices]

        random.shuffle(candidate_list)
        candidate_list = top_flops_candidate_list + \
                        candidate_list[:self.max_candidate_per_iter // 2]

        return candidate_list

    def get_target_list(self, cg, candidate_list):
        parent_resolution_list = [sorted([list(cg._node[parent]["resolution"])
                                          for parent in candidate["parent_nodes"]]) for candidate in candidate_list]

        child_resolution_list = [sorted([list(cg._node[child]["resolution"])
                                         for child in candidate["child_nodes"]]) for candidate in candidate_list]

        target_list = []
        for i in range(len(candidate_list)):
            if self.propagation_type in ['grid', 'milp', 'test']:
                matching_targets = self.find_good_match_res(parent_resolution_list[i],
                                                            child_resolution_list[i],
                                                            candidate_list[i])
            else:
                raise NotImplementedError("%s is not supported" % self.propagation_type)
            target_list.append(copy.deepcopy(matching_targets))

        return target_list

    def mutate(self, cg: ComputeGraph) -> (bool, list):
        cg = cg_to_nx([cg], self.model_family)[0]

        candidate_list = self.get_candidate_list(cg)
        target_list = self.get_target_list(cg, candidate_list)
        mutated_architectures, mutated_architecture_PSC_list = self.replace_segment(cg, candidate_list, target_list)

        self.total_input_cg_list.append(cg)
        self.total_candidate_list.append(candidate_list)
        self.total_target_list.append(target_list)
        self.total_mutated_arch_list.append(mutated_architectures)

        valid_mutated_architectures = [arch for arch in mutated_architectures if self.validate_arch(arch)]
        valid_mutated_architecture_index = [i for i in range(len(mutated_architectures)) if
                                            mutated_architectures[i] in valid_mutated_architectures]
        valid_mutated_architecture_PSC_list = [mutated_architecture_PSC_list[i]
                                               for i in valid_mutated_architecture_index]

        for arch in valid_mutated_architectures:
            for node in arch.nodes:
                arch._node[node]["str_id"] = str(node)
            arch.graph['src_id2dst_ids_dict'] = get_src_id2dst_ids_dict(arch)

        valid_mutated_architectures = [auto_set_nodes(arch) for arch in valid_mutated_architectures]

        return valid_mutated_architectures, valid_mutated_architecture_PSC_list

def parse_subgraph_res_nodes(subgraph):
    num_conv, num_pool, num_concat = 0, 0, 0
    for _, v in subgraph.nodes(data=True):
        if "conv" in v['label'].lower() or "depth" in v['label'].lower():
            num_conv += 1
        elif "pool" in v['label'].lower():
            num_pool += 1
        elif "cat" in v['label'].lower():
            num_concat += 1
    return num_conv, num_pool, num_concat
