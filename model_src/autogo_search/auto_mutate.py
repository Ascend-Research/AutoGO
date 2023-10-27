import copy
import time
import oapackage
from model_src.comp_graph.tf_comp_graph import ComputeGraph
from model_src.comp_graph.tf_comp_graph_utils import compute_cg_flops
from motifs.motif_structures.utils import cg_to_nx, nx_to_cg
from params import *
from tqdm import tqdm
from utils.nas_utils import grid_random_sampling, rank_sort
from joblib import delayed, Parallel


class FlopsDiffChecker:
    """
    Checking if the flops of a input CG is accepted
    """

    def __init__(self, op2i,
                 max_decrease_percent=40.0,
                 min_decrease_percent=0.0):
        self.op2i = op2i
        self.baseline_cg = None
        self.baseline_flops = None
        # Maximum decrease in the flops
        self.max_decrease_percent = max_decrease_percent
        # Minimum decrease in the flops
        self.min_decrease_percent = min_decrease_percent

    def get_flops(self, cg: ComputeGraph):
        """
        Return the flops of the input CG
        """
        return compute_cg_flops(cg, self.op2i, use_fast_counter=True)

    def get_baseline(self):
        """
        Return the flops of baseline CG
        """
        return self.baseline_flops

    def set_baseline(self, cg: ComputeGraph):
        """
        Set the flops baseline to be the flops of user input CG
        """
        self.baseline_cg = cg
        self.baseline_flops = self.get_flops(cg)

    def check(self, cg: ComputeGraph):
        """
        Check whether the flops of input CG is lower than the baseline
        """
        cand_flops = self.get_flops(cg)

        delta_flops = ((self.baseline_flops - cand_flops) / self.baseline_flops) * 100

        if self.max_decrease_percent > delta_flops > self.min_decrease_percent:
            return cand_flops, True

        return cand_flops, False


class AccDiffChecker:
    """
    Checking if the accuracy of an input CG is accepted
    """

    def __init__(self, predictor, predictor_type="k_adaptor",
                 min_increase_percent=0.02):

        self.predictor = predictor
        self.predictor_type = predictor_type
        self.baseline_cg = None
        self.baseline_acc = None

        self.min_increase_percent = min_increase_percent

        self.num_passed_CG = 0

    def _get_acc(self, cg: ComputeGraph, count=True):
        """
        Check the accuracy of the input CG
        """
        if cg == []:
            return []

        result = self.predictor(cg)
        if type(result) != list:
            result = [result]
        if count:
            self.num_passed_CG += len(result)
        return result

    def get_baseline(self):
        """
        Check the baseline accuracy
        """
        return self.baseline_acc

    def set_baseline(self, cg: ComputeGraph, PSC_sampling_times=10, get_candidate_function=None):
        """
        Set the accuracy baseline to be the flops of user input CG
        """
        self.baseline_cg = cg

        if self.predictor_type == "PSC_predictor":
            assert get_candidate_function is not None, "PSC predictor requires get_candidate_function"
            candidate_list = get_candidate_function(cg_to_nx(cg)[0], perform_filtering=False)
            acc_list = []
            for candidate in candidate_list:
                current_PSC = [candidate["predecessor_subgraph"],
                               candidate["segment_subgraph"],
                               candidate["successor_subgraph"]]
                current_acc = self._get_acc([nx_to_cg(current_PSC)], count=False)
                acc_list.append(current_acc[0])

            self.baseline_acc = sum(acc_list) / len(acc_list)
        else:
            self.baseline_acc = self._get_acc(cg)[0]

    def check(self, cg: ComputeGraph):
        """
        Check whether the accuracy of input CG is lower than the baseline
        """
        cand_acc = self._get_acc(cg)
        if len(cand_acc) == 1:
            cand_acc = cand_acc[0]

        if type(cand_acc) == list:
            delta_acc = [((cand_acc_i - self.baseline_acc) / abs(self.baseline_acc)) * 100 for cand_acc_i in cand_acc]
            check_result = [delta_acc_i > self.min_increase_percent
                            for delta_acc_i in delta_acc]
            check_result = [bool(check_result_i) for check_result_i in check_result]
            return cand_acc, check_result

        delta_acc = ((cand_acc - self.baseline_acc) / abs(self.baseline_acc)) * 100

        if delta_acc > self.min_increase_percent:
            return cand_acc, True
        return cand_acc, False


class RunManager:

    def __init__(self, mutator, op2i, flops_checker: FlopsDiffChecker, acc_checker: AccDiffChecker,
                 max_visited_architecture, num_iterations=5, log_f=print, predictor_type="PSC_predictor", k_select="rank_sort", n_jobs=1):
        self.op2i = op2i
        self.mutator = mutator
        self.flops_checker = flops_checker
        self.acc_checker = acc_checker
        self.num_iterations = num_iterations
        self.visited_architecture_list = []
        self.visited_performance_list = []
        self.visited_iteration_list = []
        self.time_consumption = None
        self.predictor_type = predictor_type
        self.max_visited_architecture = max_visited_architecture
        self.k_select = eval(k_select)
        self.n_jobs = n_jobs

    def print_results(self, top_set):
        """
        Print the performance of found architectures
        """
        best_archs = [top[0] for top in top_set]
        top_set = [top[1] for top in top_set]
        print(top_set)

        baseline_nodes = len(self.visited_architecture_list[0].nodes)
        print("User cg perfs: Accuracy {}, Flops {}, Nodes {}".format(self.acc_checker.get_baseline(),
                                                            self.flops_checker.get_baseline(), baseline_nodes))
        print("Showing up to {} current best archs, sorted by prediccted accuracy".format(len(top_set)))

        for ri, (acc, flops, iteration) in enumerate(top_set):
            arch_nodes = len(best_archs[ri].nodes)
            node_diff = arch_nodes - baseline_nodes
            print("Rank %d: Accuracy %.4f, Flops %.4f, #Nodes %d, Iteration %d, Delta Accuracy %.4f%%, Delta Flops %.4f%%, Delta Nodes %d" %
                  (ri, acc, flops, arch_nodes, iteration,
                   100 * (acc - self.acc_checker.get_baseline()) / abs(self.acc_checker.get_baseline()),
                   100 * (flops - self.flops_checker.get_baseline()) / self.flops_checker.get_baseline(),
                   node_diff))

    def filter_repeated_pareto_result(self, pareto):
        filtered_index_list = []
        pareto_performance_list = []
        for index in pareto.allindices():
            pareto_index_performance = self.visited_performance_list[index]
            if pareto_index_performance not in pareto_performance_list: 
                pareto_performance_list.append(pareto_index_performance)
                filtered_index_list.append(index)

        return filtered_index_list

    def run(self, input_cg, top_k=10):
        """
        Run the mutation process
        """
        start_time = time.time()

        self.acc_checker.set_baseline(input_cg, get_candidate_function=self.mutator.get_candidate_list)
        self.flops_checker.set_baseline(input_cg)

        init_perf = (self.acc_checker.get_baseline(), self.flops_checker.get_baseline(), 0)

        pareto = oapackage.ParetoDoubleLong()

        self.visited_architecture_list.append(input_cg)
        self.visited_performance_list.append(init_perf)

        pareto.addvalue(oapackage.doubleVector((self.acc_checker.get_baseline(),
                                                -1 * self.flops_checker.get_baseline())), 0)
        
        pair_CG_list = []

        for _ in tqdm(range(self.num_iterations)):

            if self.acc_checker.num_passed_CG > self.max_visited_architecture:
                print("Surpassed max_visited_architectures at start of for-loop; breaking")
                break

            filtered_pareto = self.filter_repeated_pareto_result(pareto)

            _top_set_index = grid_random_sampling(top_k, [self.visited_performance_list[i][:-1] for i in filtered_pareto])
            top_set_index = [filtered_pareto[i] for i in _top_set_index]
            top_set = copy.deepcopy([self.visited_architecture_list[i] for i in top_set_index])

            if _ > 0:
                deficit = top_k - len(top_set)
                p_size = len(pareto.allindices())
                print(f"Iter {_}; top-k = {top_k}; #pareto = {p_size}; Pareto deficit = {deficit}")

            if len(top_set) < top_k and _ != 0:
                _top_set_index = self.k_select(min(top_k - len(top_set), len(top_cg_list)),
                                                      [(top_acc_list[i], top_flops_list[i])
                                                       for i in range(len(top_cg_list))])
                for i in _top_set_index:
                    top_set.append(top_cg_list[i])

            _top_arch_list = Parallel(n_jobs=self.n_jobs)(delayed(self.mutator.mutate)(arch) for arch in top_set)

            if self.acc_checker.num_passed_CG > self.max_visited_architecture:
                print("Surpassed max_visited_architectures while analyzing top-k; breaking")
                break

            top_arch_list = [item for sublist in _top_arch_list for item in sublist[0]]
            top_arch_PSC_list = [item for sublist in _top_arch_list for item in sublist[1]]

            top_cg_list = nx_to_cg(top_arch_list)
            top_arch_PSC_list = [tuple(nx_to_cg(list(element))) for element in top_arch_PSC_list]

            top_cg_list_index = list({str(top_cg_list[i]).split("str_id")[-1]: i for i in
                                      range(len(top_cg_list))}.values())
            top_cg_list = [top_cg_list[i] for i in top_cg_list_index]
            top_arch_PSC_list = [top_arch_PSC_list[i] for i in top_cg_list_index]

            identical_mutation_index_list = [i for i in range(len(top_arch_PSC_list)) if
                                             str(top_arch_PSC_list[i][1]).split("str_id")[-1] ==
                                             str(top_arch_PSC_list[i][2]).split("str_id")[-1]]
            for index in sorted(identical_mutation_index_list, reverse=True):
                del top_arch_PSC_list[index]
                del top_cg_list[index]

            self.mutator.total_PSC_list.append(top_arch_PSC_list)

            rearranged_baseline_arch_PSC_list = [element[0:2] + element[-1:] for element in top_arch_PSC_list]
            rearranged_top_arch_PSC_list = [element[:1] + element[2:] for element in top_arch_PSC_list]

            if self.predictor_type == "PSC_predictor":
                baseline_acc_list = self.acc_checker._get_acc(rearranged_baseline_arch_PSC_list, count=False)
                top_acc_list = self.acc_checker._get_acc(rearranged_top_arch_PSC_list, count=False)
            elif self.predictor_type == "GNN":
                baseline_acc_list = None
                top_acc_list = self.acc_checker._get_acc(top_cg_list)
            else:
                raise NotImplementedError("%s is not supported" % self.predictor_type)

            self.mutator.baseline_acc_list.append(baseline_acc_list)
            self.mutator.total_PSC_acc_list.append(top_acc_list)
            top_flops_list = [self.flops_checker.get_flops(element) for element in top_cg_list]

            for i in range(len(top_cg_list)):
                arch_performance = (top_acc_list[i], top_flops_list[i], _ + 1)

                if self.predictor_type == "PSC_predictor":
                    if (not self.flops_checker.check(top_cg_list[i])[1]) or \
                            (not self.acc_checker.check([list(rearranged_top_arch_PSC_list[i])])[1]):
                        continue
                elif self.predictor_type == "GNN":
                    if (not self.flops_checker.check(top_cg_list[i])[1]) or \
                            (not self.acc_checker.check(top_cg_list[i])[1]):
                        continue
                pareto.addvalue(oapackage.doubleVector((top_acc_list[i], -1 * top_flops_list[i])),
                                len(self.visited_architecture_list))

                self.visited_architecture_list.append(top_cg_list[i])
                self.visited_performance_list.append(arch_performance)

        top_set = []
        for i in self.filter_repeated_pareto_result(pareto):
            arch = self.visited_architecture_list[i]
            performance = self.visited_performance_list[i]
            if arch == input_cg:
                continue
            top_set.append((arch, performance))

        filtered_top_set = [element for element in top_set
                            if self.flops_checker.check(element[0])[1]]

        filtered_top_set = sorted(filtered_top_set, key=lambda x: x[1][0], reverse=True)
        top_set = sorted(top_set, key=lambda x: x[1][0], reverse=True)

        top_set = filtered_top_set

        self.print_results(top_set)
        self.time_consumption = time.time() - start_time

        return top_set, pair_CG_list
