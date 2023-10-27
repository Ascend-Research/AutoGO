import os
import json
import pickle
import random
from tqdm import tqdm
from params import P_SEP, CACHE_DIR, DATA_DIR
from model_src.comp_graph.tf_comp_graph import ComputeGraph, load_from_state_dict
from utils.misc_utils import RunningStatMeter
from model_src.comp_graph.tf_comp_graph_splits import CGSplitManager


_DOMAIN_CONFIGS = {
    "classification": {
        "c_in": 3,
        "max_h": 256,
        "max_w": 256,
        "max_kernel_size": 7,
        "max_hidden_size": 512,
    },
}


_FAMILY2MAX_HW = {
    "ofa": 224,
    "nb101": 32,
    "nb301": 32,
    "nb201c10": 32,
    "nb201c100": 32,
    "nb201imgnet": 32,
}


def get_domain_configs(domain="classification"):
    return _DOMAIN_CONFIGS[domain]


def reset_cg_max_HW(input_file, output_file,
                    max_H, max_W,
                    log_f=print):
    log_f("Loading cache from: {}".format(input_file))
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    new_data = []
    for di, d in enumerate(data):
        cg = d["compute graph"]
        assert isinstance(cg, ComputeGraph)
        if di == 0:
            log_f("Resetting max H from {} to {}".format(cg.max_derived_H, max_H))
            log_f("Resetting max W from {} to {}".format(cg.max_derived_W, max_W))
        cg.max_derived_H = max_H
        cg.max_derived_W = max_W
        new_data.append(d)
    log_f("Writing {} compute graph data to {}".format(len(new_data), output_file))
    with open(output_file, "wb") as f:
        pickle.dump(new_data, f, protocol=4)

class FamilyDataManager:
    def __init__(self, families=("nb101",),
                 family2args=None,
                 cache_dir=CACHE_DIR, data_dir=DATA_DIR,
                 log_f=print):
        self.log_f = log_f
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.families = families
        self.family2args = family2args
        self.validate_cache()

    def get_cache_file_path(self, family):
        if family.startswith("E-"):
            return self.cache_dir + "/gpi_comp_graph_cache_{}.pkl".format(family)
        return P_SEP.join([self.cache_dir, "gpi_{}_comp_graph_cache.pkl".format(family)])

    def validate_cache(self):
        for f in self.families:

            if f.lower() == "hiaml" or f.lower() == "two_path" or \
                    f.lower() == "inception":
                continue

            if f.lower() == "ofa_mbv3" or f.lower() == "ofa_pn":
                f = "ofa"

            cache_file = self.get_cache_file_path(f)
            if not os.path.isfile(cache_file):
                assert False, f"Cannot find this cache: {cache_file}"

        self.log_f("Cache validated for {}".format(self.families))

    def load_cache_data(self, family):
        if family.lower() == "hiaml" or family.lower() == "two_path":
            d = self.get_gpi_custom_set(family_name=family.lower(),
                                        perf_diff_threshold=0,
                                        target_round_n=None,
                                        verbose=False)
            data = [{"compute graph": t[0], "flops": t[1], "acc": t[-1]} for t in d]
        elif family.lower() == "inception":
            d = self.get_inception_custom_set(perf_diff_threshold=0,
                                              target_round_n=None,
                                              verbose=False)
            data = [{"compute graph": t[0], "flops": t[1], "acc": t[-1]} for t in d]
        elif family.lower() == "ofa_mbv3":
            cache_file = self.get_cache_file_path("ofa")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            subset = []
            for d in data:
                if "mbv3" in d["compute graph"].name.lower():
                    subset.append(d)
            assert len(subset) > 0, "Found empty subset for ofa_mbv3"
            return subset
        elif family.lower() == "ofa_pn":
            cache_file = self.get_cache_file_path("ofa")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            subset = []
            for d in data:
                if "pn" in d["compute graph"].name.lower():
                    subset.append(d)
            assert len(subset) > 0, "Found empty subset for ofa_pn"
            return subset
        else:
            cache_file = self.get_cache_file_path(family)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
        return data

    @staticmethod
    def override_cg_max_attrs(data, max_H=None, max_W=None,
                              max_hidden=None, max_kernel=None):
        for d in data:
            cg = d["compute graph"]
            assert isinstance(cg, ComputeGraph)
            if max_H is not None:
                cg.max_derived_H = max_H
            if max_W is not None:
                cg.max_derived_W = max_W
            if max_hidden is not None:
                cg.max_hidden_size = max_hidden
            if max_kernel is not None:
                cg.max_kernel_size = max_kernel

    def get_src_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    group_by_family=False, shuffle=False,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    verbose=True,
                                    rescale_perf=False):
        family2data = {}
        for f in self.families:
            if verbose:
                self.log_f("Loading {} cache data...".format(f))
            fd = self.load_cache_data(f)              

            if verbose:
                self.log_f("Specified normalize_HW_per_family={}".format(normalize_HW_per_family))
            if normalize_HW_per_family:
                self.override_cg_max_attrs(fd,
                                           max_H=_FAMILY2MAX_HW[f],
                                           max_W=_FAMILY2MAX_HW[f])
            if max_hidden_size is not None:
                if verbose:
                    self.log_f("Override max_hidden_size to {}".format(max_hidden_size))
                self.override_cg_max_attrs(fd, max_hidden=max_hidden_size)
            if max_kernel_size is not None:
                if verbose:
                    self.log_f("Override max_kernel_size to {}".format(max_kernel_size))
                self.override_cg_max_attrs(fd, max_kernel=max_kernel_size)

            if shuffle:
                random.shuffle(fd)

            ratio_manager = CGSplitManager(f, log_f=self.log_f)
            train_data, dev_data, test_data = ratio_manager.split_data(fd, dev_ratio, test_ratio)

            if rescale_perf:
                from utils.data_utils import Normalizer
                normalizer = Normalizer(train_data)
                normalizer.transform(train_data)
                normalizer.transform(dev_data)
                normalizer.transform(test_data)

            if dev_ratio < 1e-5:
                train_data += dev_data
                self.log_f("Dev ratio: {} too small, will add dev data to train data".format(dev_ratio))
            if test_ratio < 1e-5:
                train_data += test_data
                self.log_f("Test ratio: {} too small, will add test data to train data".format(test_ratio))
            family2data[f] = (train_data, dev_data, test_data)
            if verbose:
                self.log_f("Family {} train size: {}".format(f, len(train_data)))
                self.log_f("Family {} dev size: {}".format(f, len(dev_data)))
                self.log_f("Family {} test size: {}".format(f, len(test_data)))
        if group_by_family:
            return family2data
        else:
            train_set, dev_set, test_set = [], [], []
            for f, (train, dev, test) in family2data.items():
                train_set.extend(train)
                dev_set.extend(dev)
                test_set.extend(test)
            random.shuffle(train_set)
            random.shuffle(dev_set)
            random.shuffle(test_set)
            if verbose:
                self.log_f("Combined train size: {}".format(len(train_set)))
                self.log_f("Combined dev size: {}".format(len(dev_set)))
                self.log_f("Combined test size: {}".format(len(test_set)))
            return train_set, dev_set, test_set

    def get_flops(self, cg_dict):
        if self.missing_flops == False and not "flops" in cg_dict:
            cg_name = cg_dict["compute graph"].name if "compute graph" in cg_dict else cg_dict["cg"]["name"]
            self.missing_flops = True

        flops = cg_dict.get('flops', 0)
        if flops > 1e6:
            return flops / 1e9
        return flops

    def get_regress_train_dev_test_sets(self, dev_ratio, test_ratio,
                                        normalize_target=False,
                                        normalize_max=None,
                                        group_by_family=False,
                                        normalize_HW_per_family=False,
                                        max_hidden_size=None, max_kernel_size=None,
                                        shuffle=False, perf_key="acc",
                                        verbose=True):
        self.missing_flops = False
    
        if group_by_family:
            family2data = self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                           group_by_family=group_by_family,
                                                           normalize_HW_per_family=normalize_HW_per_family,
                                                           max_hidden_size=max_hidden_size,
                                                           max_kernel_size=max_kernel_size,
                                                           shuffle=shuffle, verbose=verbose)
            tgt_meter = RunningStatMeter()
            rv = {}
            for f, (train_data, dev_data, test_data) in family2data.items():

                fam_tgt_meter = RunningStatMeter()
                train_set, dev_set, test_set = [], [], []
                for d in train_data:
                    train_set.append([d["compute graph"], self.get_flops(d), d[perf_key]])
                    tgt_meter.update(d[perf_key])
                    fam_tgt_meter.update(d[perf_key])
                for d in dev_data:
                    dev_set.append([d["compute graph"], self.get_flops(d), d[perf_key]])
                    tgt_meter.update(d[perf_key])
                    fam_tgt_meter.update(d[perf_key])
                for d in test_data:
                    test_set.append([d["compute graph"], self.get_flops(d), d[perf_key]])
                rv[f] = (train_set, dev_set, test_set)
                if verbose:
                    self.log_f("Max {} target value: {}".format(f, fam_tgt_meter.max))
                    self.log_f("Min {} target value: {}".format(f, fam_tgt_meter.min))
                    self.log_f("Avg {} target value: {}".format(f, fam_tgt_meter.avg))

            if verbose:
                self.log_f("Max global target value: {}".format(tgt_meter.max))
                self.log_f("Min global target value: {}".format(tgt_meter.min))
                self.log_f("Avg global target value: {}".format(tgt_meter.avg))

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for _, (train_set, dev_set, test_set) in rv.items():
                    for t in train_set:
                        t[-1] /= tgt_meter.max
                    for t in dev_set:
                        t[-1] /= tgt_meter.max
                    for t in test_set:
                        t[-1] /= tgt_meter.max
                        if normalize_max is not None:
                            t[-1] = min(t[-1], normalize_max)

            return rv

        else:
            train_dicts, dev_dicts, test_dicts = \
                self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                 group_by_family=group_by_family,
                                                 shuffle=shuffle, verbose=verbose)
            tgt_meter = RunningStatMeter()
            train_set, dev_set, test_set = [], [], []
            for d in train_dicts:
                train_set.append([d["compute graph"], self.get_flops(d), d[perf_key]])
                tgt_meter.update(d[perf_key])
            for d in dev_dicts:
                dev_set.append([d["compute graph"], self.get_flops(d), d[perf_key]])
                tgt_meter.update(d[perf_key])
            for d in test_dicts:
                test_set.append([d["compute graph"], self.get_flops(d), d[perf_key]])

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for t in train_set:
                    t[-1] /= tgt_meter.max
                for t in dev_set:
                    t[-1] /= tgt_meter.max
                for t in test_set:
                    t[-1] /= tgt_meter.max
                    if normalize_max is not None:
                        t[-1] = min(t[-1], normalize_max)
            if verbose:
                self.log_f("Max target value: {}".format(tgt_meter.max))
                self.log_f("Min target value: {}".format(tgt_meter.min))
                self.log_f("Avg target value: {}".format(tgt_meter.avg))

            return train_set, dev_set, test_set

    def get_psc_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    normalize_target=False,
                                    group_by_family=False,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    shuffle=False, perf_keys=["acc_mean", "acc_var"],
                                    use_var=False,
                                    verbose=True,
                                    rescale_perf=True):
    
        if group_by_family:
            family2data = self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                           group_by_family=group_by_family,
                                                           normalize_HW_per_family=normalize_HW_per_family,
                                                           max_hidden_size=max_hidden_size,
                                                           max_kernel_size=max_kernel_size,
                                                           shuffle=shuffle, verbose=verbose,
                                                           rescale_perf=rescale_perf)
            mean_meter, dev_meter = RunningStatMeter(), RunningStatMeter()
            rv = {}
            for f, (train_data, dev_data, test_data) in family2data.items():

                fam_mean_meter, fam_dev_meter = RunningStatMeter(), RunningStatMeter()
                train_set, dev_set, test_set = [], [], []
                for d in train_data:
                    mean = d[perf_keys[0]]
                    dev = d[perf_keys[1]] if use_var else None
                    train_set.append([d["predecessor_cg"], d["segment_cg"], d["successor_cg"], mean, dev])
                    mean_meter.update(mean)
                    fam_mean_meter.update(mean)
                    if use_var:
                        dev_meter.update(dev)
                        fam_dev_meter.update(dev)
                for d in dev_data:
                    mean = d[perf_keys[0]]
                    dev = d[perf_keys[1]] if use_var else None
                    dev_set.append([d["predecessor_cg"], d["segment_cg"], d["successor_cg"], mean, dev])
                    mean_meter.update(mean)
                    fam_mean_meter.update(mean)
                    if use_var:
                        dev_meter.update(dev)
                        fam_dev_meter.update(dev)
                for d in test_data:
                    dev = d[perf_keys[1]] if use_var else None
                    test_set.append([d["predecessor_cg"], d["segment_cg"], d["successor_cg"], d[perf_keys[0]], dev])
                rv[f] = (train_set, dev_set, test_set)
                if verbose:
                    self.log_f("Max {} mean target value: {}".format(f, fam_mean_meter.max))
                    self.log_f("Min {} mean target value: {}".format(f, fam_mean_meter.min))
                    self.log_f("Avg {} mean target value: {}".format(f, fam_mean_meter.avg))

                    self.log_f("Max {} s.dev target value: {}".format(f, fam_dev_meter.max))
                    self.log_f("Min {} s.dev target value: {}".format(f, fam_dev_meter.min))
                    self.log_f("Avg {} s.dev target value: {}".format(f, fam_dev_meter.avg))

            if verbose:
                self.log_f("Max global mean target value: {}".format(mean_meter.max))
                self.log_f("Min global mean target value: {}".format(mean_meter.min))
                self.log_f("Avg global mean target value: {}".format(mean_meter.avg))

                self.log_f("Max global s.dev target value: {}".format(dev_meter.max))
                self.log_f("Min global s.dev target value: {}".format(dev_meter.min))
                self.log_f("Avg global s.dev target value: {}".format(dev_meter.avg))

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for _, (train_set, dev_set, test_set) in rv.items():
                    for t in train_set:
                        t[-2] /= mean_meter.max
                        t[-1] /= dev_meter.max
                    for t in dev_set:
                        t[-2] /= mean_meter.max
                        t[-1] /= dev_meter.max
                    for t in test_set:
                        t[-2] /= mean_meter.max
                        t[-1] /= dev_meter.max

            return rv

        else:
            train_dicts, dev_dicts, test_dicts = \
                self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                 group_by_family=group_by_family,
                                                 shuffle=shuffle, verbose=verbose)
            mean_meter, dev_meter = RunningStatMeter(), RunningStatMeter()
            train_set, dev_set, test_set = [], [], []
            for d in train_dicts:
                mean = d[perf_keys[0]]
                dev = d[perf_keys[1]] if use_var else None
                train_set.append([d["predecessor_cg"], d["segment_cg"], d["successor_cg"], mean, dev])
                mean_meter.update(mean)
                if use_var:
                    dev_meter.update(dev)
            for d in dev_dicts:
                mean = d[perf_keys[0]]
                dev = d[perf_keys[1]] if use_var else None
                dev_set.append([d["predecessor_cg"], d["segment_cg"], d["successor_cg"], mean, dev])
                mean_meter.update(mean)
                if use_var:
                    dev_meter.update(dev)
            for d in test_dicts:
                dev = d[perf_keys[1]] if use_var else None
                test_set.append([d["predecessor_cg"], d["segment_cg"], d["successor_cg"], d[perf_keys[0]], dev])

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for t in train_set:
                    t[-2] /= mean_meter.max
                    t[-1] /= dev_meter.max
                for t in dev_set:
                    t[-2] /= mean_meter.max
                    t[-1] /= dev_meter.max
                for t in test_set:
                    t[-2] /= mean_meter.max
                    t[-1] /= dev_meter.max
            if verbose:
                self.log_f("Max mean target value: {}".format(mean_meter.max))
                self.log_f("Min mean target value: {}".format(mean_meter.min))
                self.log_f("Avg mean target value: {}".format(mean_meter.avg))

                self.log_f("Max s.dev target value: {}".format(dev_meter.max))
                self.log_f("Min s.dev target value: {}".format(dev_meter.min))
                self.log_f("Avg s.dev target value: {}".format(dev_meter.avg))

            return train_set, dev_set, test_set

    def get_gpi_custom_set(self, family_name="hiaml", dataset="cifar10",
                           max_hidden_size=None, max_kernel_size=None,
                           perf_diff_threshold=2e-4, target_round_n=None,
                           verbose=True):
        if verbose:
            self.log_f("Loading {} data...".format(family_name))

        data_file = P_SEP.join([self.data_dir, "gpi_test_{}_{}_labelled_cg_data.json".format(family_name, dataset)])

        with open(data_file, "r") as f:
            data = json.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)
        
        self.missing_flops = False
        for k, v in data.items():
            cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                              max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                              max_derived_H=max_derived_H, max_derived_W=max_derived_W)
            cg = load_from_state_dict(cg, v["cg"])
            acc = v["max_perf"] / 100.
            flops = self.get_flops(v)
            rv.append((cg, flops, acc))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[-1], reverse=True)
            pruned_indices = set()
            for i, (g, _, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][-1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for {}: {}".format(family_name, tgt_meter.max))
            self.log_f("Min final target value for {}: {}".format(family_name, tgt_meter.min))
            self.log_f("Avg final target value for {}: {}".format(family_name, tgt_meter.avg))
            self.log_f("Loaded {} {} instances".format(len(rv), family_name))

        return rv

    def get_inception_custom_set(self, dataset="cifar10",
                                 max_hidden_size=None, max_kernel_size=None,
                                 perf_diff_threshold=None, target_round_n=None,
                                 verbose=True):

        data_file = P_SEP.join([self.data_dir, "inception_{}_labelled_cg_data.json".format(dataset)])

        with open(data_file, "r") as f:
            data = json.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)

        self.missing_flops = False
        for k, v in data.items():
            cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                              max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                              max_derived_H=max_derived_H, max_derived_W=max_derived_W)
            cg = load_from_state_dict(cg, v["cg"])
            acc = v["max_perf"] / 100.
            flops = self.get_flops(v)
            rv.append((cg, flops, acc))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[-1], reverse=True)
            pruned_indices = set()
            for i, (g, _, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][-1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for inception: {}".format(tgt_meter.max))
            self.log_f("Min final target value for inception: {}".format(tgt_meter.min))
            self.log_f("Avg final target value for inception: {}".format(tgt_meter.avg))
            self.log_f("Loaded {} inception instances".format(len(rv)))

        return rv

    def get_family_custom_set(self, dataset="cifar10",
                                 max_hidden_size=None, max_kernel_size=None,
                                 perf_diff_threshold=None, target_round_n=None,
                                 verbose=True, family='inception', data_file='inception_cifar10_labelled_cg_data.json'):

        data_file = P_SEP.join([self.data_dir, data_file])

        if 'json' in data_file:
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)
        
        if type(data) == dict:
            for k, v in data.items():
                cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                                max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                                max_derived_H=max_derived_H, max_derived_W=max_derived_W)
                cg = load_from_state_dict(cg, v["cg"])
                acc = v["max_perf"] / 100.
                rv.append((cg, acc))
                if bar is not None:
                    bar.update(1)
        else:
            for v in data:
                cg = v['compute graph']
                acc = v["acc"]
                rv.append((cg, acc))
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for {}: {}".format(family, tgt_meter.max))
            self.log_f("Min final target value for {}: {}".format(family, tgt_meter.min))
            self.log_f("Avg final target value for {}: {}".format(family, tgt_meter.avg))
            self.log_f("Loaded {} {} instances".format(len(rv), family))

        return rv
