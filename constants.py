OOV_TOKEN = "<OOV>"

abbrv_families_names = {
    'hiaml': 'h',
    'two_path': 't',
    'nb101_5k': 'n15',
    'nb201': 'n2',
    'nb201c10': 'n2',
    'inception': 'i',
    }

# DK: string constants used to index batch dict produced by custom data loaders
DK_BATCH_SIZE = "batch_size"
DK_BATCH_TARGET_TSR = "batch_target_tensor"
DK_BATCH_UNIQUE_STR_ID_SET = "batch_unique_id_set"
DK_BATCH_EDGE_TSR_LIST = "batch_edge_tsr_list"
DK_BATCH_TARGET_EDGE_TYPE_TSR = "batch_target_edge_type_tensor"
DK_BATCH_LAST_NODE_IDX_LIST = "batch_last_node_idx_list"
DK_BATCH_CG_REGULAR_IDX = "batch_cg_regular_idx"
DK_BATCH_CG_REGULAR_SHAPES = "batch_cg_regular_shapes"
DK_BATCH_CG_WEIGHTED_IDX = "batch_cg_weighted_idx"
DK_BATCH_CG_WEIGHTED_SHAPES = "batch_cg_weighted_shapes"
DK_BATCH_CG_WEIGHTED_KERNELS = "batch_cg_weighted_kernels"
DK_BATCH_CG_WEIGHTED_BIAS = "batch_cg_weighted_bias"
DK_BATCH_FLOPS = "batch_flops"
DK_BATCH_REG_NODE_OFFSET = "batch_reg_node_offset"
DK_BATCH_WEIGHTED_OFFSETS = "batch_w_offsets"
DK_BATCH_REGULAR_OFFSETS = "batch_r_offsets"


# CHKPT: checkpoint dict keys
CHKPT_COMPLETED_EPOCHS = "completed_epochs"
CHKPT_MODEL = "model"
CHKPT_OPTIMIZER = "optimizer"
CHKPT_METADATA = "metadata"
CHKPT_PARAMS = "params"
CHKPT_BEST_EVAL_RESULT = "best_eval_result"
CHKPT_BEST_EVAL_EPOCH = "best_eval_epoch"
CHKPT_PAST_EVAL_RESULTS = "past_eval_results"
CHKPT_ITERATION = "iteration"
CHKPT_BEST_EVAL_ITERATION = "best_eval_iteration"
CHKPT_ACTOR_CRITIC = "actor_critic"
CHKPT_ENV_TRAJECTORY = "env_trajectory"
CHKPT_ENV_STEP_REWARD_MODULE = "env_step_reward_module"
CHKPT_ENV_FINAL_REWARD_MODULE = "env_final_reward_module"
CHKPT_PARETO_FRONT = "pareto_front"
CHKPT_EVALUATOR = "evaluator"
CHKPT_GNN_PERF_PREDICTOR = "gnn_perf_predictor"
CHKPT_ENCODER = "encoder"
CHKPT_DECODER = "decoder"

CG_NAMES = {
    'nb101': 'gpi_nb101_comp_graph_cache',
    'nb101_5k':'gpi_nb101_5k_comp_graph_cache',
    'nb201': 'gpi_nb201c10_comp_graph_cache',
    'nb201c10': 'gpi_nb201c10_comp_graph_cache',
    'hiaml': 'gpi_test_hiaml_cifar10_labelled_cg_data.json',
    'inception': 'inception_cifar10_labelled_cg_data.json',
    'two_path':'gpi_test_two_path_cifar10_labelled_cg_data.json',
}

CG_NAMES_ALL = {
    'nb201': 'gpi_nb201c10_comp_graph_cache_all',
    'hiaml': 'gpi_hiaml_comp_graph_cache',
    'nb101': 'gpi_nb101c10_comp_graph_cache',
    'two_path': 'gpi_two_path_cifar10_comp_graph_sample=1k',
    'inception': 'gpi_test_two_path_cifar10_labelled_cg_data.json',
}

SEEDS_RAW = [12345,
             1,
             2,
             3,
             4,
             5,
             6,
             7,
             8,
             9]

OPS = {'hiaml': {'avgpool', 'conv2d', 'relu', 'add', 'fusedbatchnorm', 'matmul', 'identity', 'mean'},
'nb201':{'avgpool', 'conv2d', 'relu', 'add', 'fusedbatchnorm', 'matmul', 'identity', 'mean', 'zero'},
'nb101': {'avgpool','maxpool', 'conv2d', 'relu', 'add', 'fusedbatchnorm', 'matmul', 'identity', 'mean', 'concat'},
'inception': {'add',
 'avgpool',
 'concat',
 'conv2d',
 'depthwise',
 'fusedbatchnorm',
 'identity',
 'matmul',
 'maxpool',
 'mean',
 'relu'},
'two_path': {'add',
 'avgpool',
 'concat',
 'conv2d',
 'fusedbatchnorm',
 'identity',
 'matmul',
 'maxpool',
 'mean',
 'relu'},
 'generic': {'avgpool', 'maxpool', 'conv2d', 'relu', 'add', 'concat', 'fusedbatchnorm', 'matmul', 'identity', 'mean', 'depthwise'},
 'generic_noDW': {'avgpool', 'maxpool', 'conv2d', 'relu', 'add', 'concat', 'fusedbatchnorm', 'matmul', 'identity', 'mean'},
 'edsr': {'conv2d', 'relu', 'add', 'concat', 'matmul', 'identity', 'mean'}}
