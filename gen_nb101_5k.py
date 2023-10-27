import pickle

with open("data/gpi_nb101_comp_graph_cache.pkl", "rb") as f:
    cgs = pickle.load(f)

with open("data/nb101_5k_name_list.pkl", "rb") as f:
    name_list = pickle.load(f)

filtered_cgs = [cg for cg in cgs if cg['compute graph'].name in name_list]

with open("data/gpi_nb101_5k_comp_graph_cache.pkl", "wb") as f:
    pickle.dump(filtered_cgs, f, protocol=4)