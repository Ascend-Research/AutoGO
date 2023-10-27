import pickle
import copy
from params import P_SEP
from model_src.comp_graph.tf_comp_graph import ComputeGraph, RegularNode, OP2I
from model_src.comp_graph.tf_comp_graph_utils import post_prune_nodes_by_keywords
from model_src.comp_graph.tf_comp_graph_utils import get_topo_sorted_nodes, post_prune_dilation
from utils.graph_utils import get_reverse_adj_dict, get_index_based_input_inds
from model_src.predictor.gpi_family_data_manager import get_domain_configs
from model_src.comp_graph.tf_comp_graph_output import CompGraphOutputNet as TFNet
from model_src.comp_graph.tf_comp_graph_output_torch import CompGraphOutputNet as TorchNet
from model_src.comp_graph.semseg_net_torch import CompGraphOutputNet as SegNet


# NOTE: These are important for changing a classification CG into an HPE/SemSeg CG.
# Key is the name of the pkl file of the original architecture - we provide code for ResNet-50/101 and VGG-16 BN.
# First entry is the output channels for HPE.
# Second entry (list) is the output channels for the main and auxiliary heads present in the semseg repo
# E.g., https://github.com/hszhao/semseg/blob/master/model/pspnet.py#L72
# The forward-function for the semantic segmentation network returns both outputs.
# The last entry, an operation name, is for parsing the network to determine where to 
# make a new auxiliary connection for the semantic segmentation auxiliary head.
MASTER_CG_ARG_DICT = {
    "resnet50.pkl": [2048, [2048, 1024], "relu"],
    "resnet101.pkl": [2048, [2048, 1024], "relu"],
    "vgg16_bn.pkl": [512, [512, 256], "maxpool"],
}

# net_file: Can be .pb file or compute graph saved in ".pkl" form (e.g., load pickle, loaded object is CG)
# name: Name of the new CG.
# Net: If true, will return the torch.nn.Module object
# op2i: From model_src/comp_graph/tf_comp_graph
def cg_class(net_file, name="MyClassificationCG", net=False, op2i=None):

    if op2i is None:
        op2i = OP2I().build_from_file()

    if net_file[-3:] == ".pb":
        cg = ComputeGraph(name=name, C_in=3,
                          H=224, W=224)  # Always use ImageNet to start for dilation rules down the road.
        cg.build_from_pb(net_file, op2i, oov_threshold=0.)
    else:
        with open(net_file, "rb") as f:
            cg = pickle.load(f)

    cg = post_prune_nodes_by_keywords(["paddings"], cg)

    if net:
        return TorchNet(op2i, cg, squeeze_output=True)
    else:
        return cg


# Function for HPE networks
# HPE conversion isn't that complex
# We just cut off the classification head
# HPE code then attaches its head to the output
# NOTE: 'config_name' should be one of the keys in MASTER_CG_ARG_DICT
def cg_hpe(net_file, config_name=None, name="MyComputeGraph", net=False, op2i=None):
    if op2i is None:
        op2i = OP2I().build_from_file()
    target_h, target_w, target_c = 224, 224, 3  # Always use ImageNet size for consistency
    domain_configs = get_domain_configs()

    # Get the classification CG with pruned padding nodes
    cg = cg_class(net_file=net_file, name=name, net=False, op2i=op2i)
    
    # Process to remove classification head, leaving only a stub for the last latent tensor
    nodes = cg.nodes
    src2dst_ids = cg.src_id2dst_ids_dict
    nodes = get_topo_sorted_nodes(nodes, get_reverse_adj_dict(src2dst_ids))
    node_input_inds = get_index_based_input_inds([n.str_id for n in nodes], src2dst_ids)

    # Look for the last node where the output HW are greater than 1.
    # That rule defines where we cut off - usually this node precedes some reshape, mean or global pool op.
    last_node = None
    for i, node in enumerate(nodes[::-1]):
        if node.resolution[1] > 1 and node.resolution[3] > 1:
            last_node = node
            last_node_i = len(nodes) - i
            break
    assert last_node is not None, "Unable to locate a last node with HW > 1"

    nodes = nodes[:last_node_i]
    node_input_inds = node_input_inds[:last_node_i]

    # For simplicity and to avoid headaches down the road loading new model dicts we re-create the CG.
    def model_maker():
        _model = TFNet(op2i=op2i, name=cg.name, squeeze_output=False,
                                    topo_nodes=nodes, net_input_inds=node_input_inds)
        return lambda _x, training: _model.call(_x, training=training)

    new_cg = ComputeGraph(C_in=target_c,
                          H=target_h, W=target_w,
                          name=cg.name,
                          max_hidden_size=domain_configs["max_hidden_size"],
                          max_kernel_size=domain_configs["max_kernel_size"],
                          max_derived_H=domain_configs["max_h"],
                          max_derived_W=domain_configs["max_w"])
    new_cg = post_prune_dilation(new_cg, keep_dil_info=True)
    new_cg.build_from_model_maker(model_maker=model_maker,
                                    op2idx=op2i, oov_threshold=0.)

    if config_name is None:
        config_name = net_file.split(P_SEP)[-1]
    if net:
        net_params = MASTER_CG_ARG_DICT[config_name]
        return TorchNet(op2i, new_cg, squeeze_output=False), net_params[0]
    else:
        return new_cg

# Semantic Segmentation function
# This is the most complex model as it requires us to append an additional head onto the network.
# This requires locating the node where that head's edge should stem from, and appending it.
# This is tricky as CGs have no hierarchy of "stages" for each resolution we can simply index
# They are graphs, so we must do graph traversal.
def cg_seg(net_file, config_name=None, name="MyComputeGraph", net=False):
    op2i = OP2I().build_from_file()

    if config_name is None:
        config_name = net_file.split(P_SEP)[-1]

    net_params = MASTER_CG_ARG_DICT[config_name]
    aux_res = net_params[1][::-1]
    aux_op = net_params[2]

    new_cg = cg_hpe(net_file=net_file, name=name, net=False, op2i=op2i)

    sorted_reg_list = copy.deepcopy(new_cg.nodes)
    sorted_reg_list.sort(key = lambda x: x.resolution[-1])
    for node in sorted_reg_list:
        if node.label in aux_op and node.resolution[-1] == aux_res[0]:
            node2change_id = node.str_id
            new_res = copy.deepcopy(node.resolution)
        elif node.label in aux_op and node.resolution[-1] == aux_res[1]:
            break
    new_output = RegularNode("new_output", "output_aux", op_type_idx=2)
    new_output_idx = len(new_cg.nodes)
    new_output.resolution = new_res
    new_cg.regular_nodes.append(new_output)

    for edge in new_cg.edge_pairs:
        if new_cg.nodes[edge[0]].str_id == node2change_id:
            new_cg.edge_pairs.append([edge[0], new_output_idx])
            break
    if net:
        # NOTE: SegNet code differs from the regular CG in that it will adjust the dilation parameter of later convs.
        # To preserve HW of tensors in the later stage of the network.
        # Just like PSPNet
        # https://github.com/hszhao/semseg/blob/master/model/pspnet.py#L49
        return SegNet(op2i, new_cg, squeeze_output=False), net_params
    else:
        return new_cg
