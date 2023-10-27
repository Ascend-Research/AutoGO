from params import *
from test_encoder import load_and_test_encoder, create_encoder_for_families
from encoder.utils import inherit_graph_attributes, get_context_edges


def get_subgraph_and_context_edges(sampled_graph, node_group):

    ng = node_group
    _subgraph = sampled_graph.subgraph(ng)
    subgraph = _subgraph.copy()
    inherit_graph_attributes(subgraph, sampled_graph)
    
    subgraph_nodes = subgraph.nodes()
    context_edges = get_context_edges(subgraph, sampled_graph, subgraph_nodes)

    return subgraph, context_edges

def find_in_nodes(cg_edges, ng):
    in_nodes = []
    num_inputs = 0
    for node in ng:
        flag_firstNode = True
        for edge in cg_edges:
            if edge[1] == node:
                flag_firstNode = False
                if edge[0] not in ng and edge[0] not in in_nodes: 
                    num_inputs += 1
                    if node not in in_nodes:
                        in_nodes.append(node)
        if flag_firstNode:
            in_nodes.append(node)
    return in_nodes, num_inputs

def find_out_nodes(cg_edges, ng):
    out_nodes = []
    num_outputs = 0
    for node in ng:
        flag_lastNode = True
        for edge in cg_edges:
            if edge[0] == node:
                flag_lastNode = False
                if  edge[1] not in ng and edge[1] not in out_nodes:
                    num_outputs += 1
                    if node not in out_nodes:
                        out_nodes.append(node)
        if flag_lastNode:
            out_nodes.append(node)
    return out_nodes, num_outputs

parser = argparse.ArgumentParser()
parser.add_argument("-families", type=str, required=False, default="hiaml+two_path+nb201+nb101_5k+inception", help="separate families by +")
parser.add_argument("-encoding", type=str, required=False, default="shp")
parser.add_argument("-vocab_size", type=int, required=False, default=2000)
params = parser.parse_args()

encoding = params.encoding
vocab_size = params.vocab_size
model_type = "bpe"
sub_str = False

families = list(v for v in set(params.families.split("+")) if len(v) > 0)
families.sort()

nt = create_encoder_for_families(families=families, encoding=encoding)

sp, nt = load_and_test_encoder(families=families, vocab_size=vocab_size, sub_str=sub_str, model_type=model_type, encoding=encoding)
print('encoder vocab size:{}'.format(len(nt.op2vocab)))
