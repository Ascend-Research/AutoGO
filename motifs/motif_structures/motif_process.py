import matplotlib.pyplot as plt
from .utils import *
from pathlib import Path
from params import *

plt.style.use("ggplot")
colors = {'input': 'red', 'avgpool':'green', 'add':'blue', 'conv2d':'yellow', 'relu':'orange', 'matmul':'brown', 'identity':'pink', 'fusedbatchnorm':'grey','fusedbatchnormv3':'grey', 'mean':'purple', 'output':'turquoise', 'concat':'magenta', 'zero':'lime', 'maxpool':'black', 'depthwise':'moccasin', 'batchtospacend': 'yellow', 'spacetobatchnd': 'yellow', 'mul':'lime'}


def highlight_node_groups(X_s, node_groups, name, verbos=True, key_set='all', save_path=P_SEP.join([PLOT_DIR, "highlighted_groups"]), color_map=None, graph_logger=None, use_cg_name=True):
    plt.style.use('ggplot')
    if color_map is None:
        colors = []
        for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
            if color['color'] == "#777777":
                continue
            colors.append(color['color'])
    if use_cg_name:
        graph_name = X_s.name
        digraph_name = "{}_{}".format(graph_name, name)
    else:
        digraph_name = name
    from graphviz import Digraph
    groups_flatten = [n for g in node_groups for n in g]
    dot_motifs = Digraph(
        filename=digraph_name,
        graph_attr=dict(size="12,12"),
    )
    norm_node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
        color="grey",
    )
    for g_id, node_group in enumerate(node_groups):
        if color_map is not None:
            color = color_map[g_id]
        else:
            color = colors[g_id%len(colors)]
        special_node_attr = dict(
            style="filled",
            shape="box",
            align="left",
            fontsize="12",
            ranksep="0.1",
            height="0.2",
            color=color,
        )
      
        for idx, n in X_s.nodes(data=True):
            string = "id=" + str(idx) + "\n"
            for key in n.keys():
                if key_set == 'all':
                    string += "{}: {}".format(key, n[key]) + "\n"
                elif key_set == 'short':
                    if key in ['op_name', 'resolution']:
                        string += "{}: {}".format(key, n[key]) + "\n"
            if idx in node_group:
                dot_motifs.node(str(idx), string, _attributes=special_node_attr)
            elif idx not in groups_flatten:
                dot_motifs.node(str(idx), string, _attributes=norm_node_attr)
        
    for src_i, dst_i in list(X_s.edges):
        dot_motifs.edge(str(src_i), str(dst_i))

    num_rows = len(dot_motifs.body)
    content_size = num_rows * 0.5
    size = max(0.5, content_size)
    size_str = str(size) + "," + str(size)
    dot_motifs.graph_attr.update(size=size_str)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    if graph_logger is not None:
        graph_logger.add(dot_motifs)
    else:
        print('saving to {}'.format(save_path))
        dot_motifs.render(
                view=False,
                directory=save_path,
                format="png",
                cleanup=True,
        )
    return True
