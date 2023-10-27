from tqdm import tqdm
import networkx as nx
from motifs.motif_structures.utils import *

class Network_Encoder:
    def __init__(self, families=None, encoding='op'):
        self.families = families
        self.op2vocab = {}
        self.vocab2op = {}
        self.next_char_index = 0
        self.encoding = encoding
        with open("cache_sentence_piece/common_chinese_char.txt") as f:
            temp = f.readlines()
        self.chinese_char_list = temp[0].split()

    def bfs_visit(self, g:nx.DiGraph):
        i = -1
        for i, node in g.nodes(data=True):
            if node['op_name'] == 'input':
                break
        children = [ch for ch in g.successors(i)]
        visited = [i] + [ch for ch in g.successors(i)]
        while len(children) > 0:
            ch = children.pop(0)
            for gc in g.successors(ch):
                if gc not in visited:
                    children.append(gc)
                    visited.append(gc)
        return visited

    def encode_node(self, g, node_id):
        node = g.nodes[node_id]
        node_token = node['op_name']
        if 'encoding' not in self.__dict__:
            encoding = 'op'
        else:
            encoding = self.encoding
        if encoding == 'res':
            node_token += "," + ",".join([str(r) for r in node['resolution']])
        elif encoding == 'res_ratio':
            r = node['resolution']
            ratios = [1, r[1]/r[0], 1, r[3]/r[2], 1, r[5]/r[4]]
            node_token += "," + ",".join([str(float(r)) for r in ratios])
        elif 'shp' in encoding:
            if 'shape' in node.keys():
                node_token += "," + str(node['shape'][3])
        elif 'chn' in encoding :
            if 'shape' in node.keys():
                node_token += "," + str(node['shape'][1])

        if "nio" not in encoding:
            in_edges = [e for e in g.edges if e[1] == node_id]
            out_edges = [e for e in g.edges if e[0] == node_id]
            node_token += ',in,'
            for in_edge in in_edges:
                node_token += g.nodes[in_edge[0]]['op_name'] + ','
            node_token += ',out,'
            for out_edge in out_edges:
                node_token += g.nodes[out_edge[1]]['op_name'] + ','

        if node_token not in self.op2vocab.keys():
            self.next_char_index += 1
            new_vocab = self.chinese_char_list[self.next_char_index]
            self.op2vocab[node_token] = new_vocab
            self.vocab2op[new_vocab] = node_token

        return self.op2vocab[node_token]
    
    def encode_graph(self, g, node_list=None):
        g_tokenized = ''
        if node_list is None:
            node_list = list(nx.topological_sort(g))
        for node_id in node_list:
            g_tokenized += self.encode_node(g, node_id)
        return g_tokenized
    
    def encode_all_graphs(self):
        graph_tokens = []
        graphs_tokens_str = ""
        graph_tokens_str_dict = {}
        all_graphs = []
        for family in self.families:
            graphs, _, _ = load_family_graphs_info(family, to_nx=True, randomize=False)
            all_graphs += graphs
        
        bar = tqdm(total=len(all_graphs), desc="Tokenizing compute graphs", ascii=True )
        for idx, g in enumerate(all_graphs):
            bar.update(1)
            g_token = self.encode_graph(g)
            graph_tokens.append(g_token)
            graphs_tokens_str += '<s>' + g_token + '</s>\n'
            if g_token not in graph_tokens_str_dict.keys():
                graph_tokens_str_dict[g_token] = [idx]
            else:
                graph_tokens_str_dict[g_token].append(idx)
        bar.close()
        return graph_tokens, graphs_tokens_str

    def decode_graphs_utf8(self, tokens):
        tokens_str = tokens.decode('utf8')
        
        graphs_nodes = []

        for token_str in tokens_str.split('\n'):
            graph_nodes = self.decode_graph(token_str)
            graphs_nodes.append(graph_nodes)

        return graphs_nodes

    def decode_graph(self, token_str):
        graph_nodes = []

        token_str = token_str.replace('<s>', '')
        token_str = token_str.replace('</s>', '')
        token_str = token_str.replace('\n', '')
        
        for ch in token_str:
            try:
                graph_nodes.append(self.vocab2op[ch])
            except KeyError as e:
                print('detokenizing a graph with the wrong vocab dictionary, load the correct vocab')
        
        return graph_nodes
