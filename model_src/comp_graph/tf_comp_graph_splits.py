import pickle
import os
import random
from constants import *
from params import *
from copy import deepcopy


class CGSplitManager:
    def __init__(self, family, log_f=print):
        self.family = family
        self.log_f = log_f
        self.cache_name = P_SEP.join([CACHE_DIR, f"gpi_{family}_split_idx.pkl"])
        if self._check_for_index_file():
            self.log_f(f"Found index file for cache {family}.")
            with open(self.cache_name, "rb") as f:
                self.cache = pickle.load(f)
        else:
            self.log_f(f"Unable to find index file for cache {family}; will save indices to {self.cache_name}.")
            self.cache = {}

    def _check_for_index_file(self):
        return os.path.exists(self.cache_name)

    def _save(self, ):
        with open(self.cache_name, "wb") as f:
            self.log_f("Saving cache")
            pickle.dump(self.cache, f, protocol=4)

    def _ratio_key(self, dev, test):
        return f"dev{str(dev)}_test{str(test)}"

    def ratios_to_indices(self, dev_ratio, test_ratio):
        ratio_key = self._ratio_key(dev=dev_ratio, test=test_ratio)
        if ratio_key in self.cache.keys(): 
            self.log_f(f"Found ratios for key {ratio_key}")
            return self.cache[ratio_key]
        else: 
            self.log_f(f"Unable to find ratios for key {ratio_key}")
            return None

    def _generate_ratios(self, dev_ratio, test_ratio, data):
        ratio_key = self._ratio_key(dev=dev_ratio, test=test_ratio)
        assert ratio_key not in self.cache.keys()
        cg_names = self._get_unique_names(data)
        cg_names.sort()
        random.shuffle(cg_names)
        dev_size = max(int(dev_ratio * len(cg_names)), 1)
        test_size = max(int(test_ratio * len(cg_names)), 1)
        dev_idx = cg_names[:dev_size]
        test_idx = cg_names[dev_size:dev_size + test_size]
        train_idx = cg_names[dev_size + test_size:]

        dev_idx.sort()
        test_idx.sort()
        train_idx.sort()

        self.cache[ratio_key] = {
            "train": train_idx,
            "dev": dev_idx,
            "test": test_idx
        }
        self._save()
        return self.cache[ratio_key]
        
    def _get_unique_names(self, data):
        cg_key = "compute graph"
        cg_key = cg_key if cg_key in data[0].keys() else "segment_cg"

        cg_names = set()
        for entry in data:
            cg_names.add(entry[cg_key].name)
        return list(cg_names)

    def split_data(self, data, dev, test):
        r_dict = self.ratios_to_indices(dev, test)
        if r_dict is None:
            r_dict = self._generate_ratios(dev, test, data)

        ratio_dict = deepcopy(r_dict)
        cg_key = "compute graph"
        cg_key = cg_key if cg_key in data[0].keys() else "segment_cg"
        data.sort(key=lambda x: x[cg_key].name)

        train, dev, test = [], [], []
        while len(data) > 0:
            entry = [data.pop()]
            if "segment" in cg_key:
                while len(data) > 0 and data[-1][cg_key].name == entry[0][cg_key].name:
                    entry.append(data.pop())
            if len(ratio_dict['train']) > 0 and entry[0][cg_key].name == ratio_dict['train'][-1]:
                train += entry
                ratio_dict['train'].pop()
            elif len(ratio_dict['dev']) > 0 and entry[0][cg_key].name == ratio_dict['dev'][-1]:
                dev += entry
                ratio_dict['dev'].pop()
            elif len(ratio_dict['test']) > 0 and entry[0][cg_key].name == ratio_dict['test'][-1]:
                test += entry
                ratio_dict['test'].pop()
            else:
                raise ValueError
        return train, dev, test


def test_part_folds(test_data, num_folds=10, log_f=print):
    sample_dict = {}
    num_dict = {}
    for entry in test_data:
        entry_name = entry[0].name
        if entry_name in sample_dict.keys():
            sample_dict[entry_name].append(entry)
            num_dict[entry_name] += 1
        else:
            sample_dict[entry_name] = [entry]
            num_dict[entry_name] = 1
    num_max = max(num_dict.values())
    if num_max == 1:
        log_f("Max number of repeats is 1; just returning original test partition")
        return [test_data]
    num_min = min(num_dict.values())
    log_f(f"Adjusting num_folds from {num_folds} to accomodate min={num_min} and max={num_max}:")
    num_folds = max(min(num_folds, num_max), num_min)
    log_f(f"Adjustment num_folds={num_folds}")
    test_parts = [[] for _ in range(num_folds)]

    for name, psc_list in sample_dict.items():
        for fold in range(num_folds):
            idx = fold % len(psc_list)
            test_parts[fold].append(psc_list[idx])

    return test_parts
