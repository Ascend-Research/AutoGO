from pyomo.environ import *
import networkx as nx
from params import *
import copy
from functools import reduce 


def create_model_from_digraph(G,target_outputs,a=1,b=1,c=1,d=1,e=1):

    model = ConcreteModel()

    node_names = list(G.nodes)

    model.Nodes = Set(initialize=node_names)
    model.Arcs = Set(initialize=list(G.edges))

    inputs = [k for k in node_names if G.in_degree[k]==0]
    model.inputs = Set(initialize=inputs)

    outputs = [k for k in node_names if G.out_degree[k]==0] 
    model.outputs = Set(initialize=outputs)
    model.flows = Var(model.Arcs, range(3), within=NonNegativeReals) 
    model.node_in_flows = Var(model.Nodes, range(3), within=Integers, bounds=(2,10000))
    model.node_out_flows = Var(model.Nodes,  range(3), within=Integers, bounds=(2,10000))

    idx_bin = []
    val_bin = {}
    for n in node_names:
        if G.nodes[n]['op_name'] in ['conv2d','avgpool','maxpool']:
            aux_mult = [1,2,4]
            for k in [0,1]:
                for t in range(len(aux_mult)):
                    idx_bin += [(n,k,t)]
                    val_bin[(n,k,t)]=int(G.nodes[n]['res_out'][k]/aux_mult[t])
            aux_mult = [0.5,1,2]
            for t in range(len(aux_mult)):
                idx_bin += [(n,2,t)]
                val_bin[(n,2,t)]=int(G.nodes[n]['res_out'][2]/aux_mult[t])

    model.bin_var_set = Set(initialize=idx_bin)
    model.bin_vars = Var(model.bin_var_set,within=Binary)

    idx_change_size, idx_keep_size, idx_concat = [],[],[]

    idx_conv2d, idx_pooling = [],[]

    pairs_in, pairs_out=[],[]
    concat_set=[]
    for n in node_names:
        name = G.nodes[n]['op_name']
        if name in ['conv2d','avgpool','maxpool']:
            idx_change_size += [n]
            if name == 'conv2d':
                idx_conv2d.append(n)
            else:
                idx_pooling.append(n)
        elif name=='concat':
            idx_concat += [n] 
        else:
            idx_keep_size += [n] 

        if not name=='concat':
            pairs_in+=[((n1,n),k) for n1 in G.predecessors(n) for k in [0,1,2]]
        else:
            check = 0
            for k1 in range(3):
                if not (G.nodes[n]['res_in'][k1]==G.nodes[n]['res_out'][k1]):            
                    k=k1
                    check+=1
                pairs_in+=[((p,n),k1) for p in G.predecessors(n)]
            if check == 1 or check == 0:
                return None
            concat_set.append((n,k))    
        
        pairs_out+=[((n,n1),k) for n1 in G.successors(n) for k in [0,1,2]]

    model.idx_change_size = Set(initialize=idx_change_size)

    def in_ineq_(model,n1,n2,k):
        return model.flows[((n1,n2),k)] == model.node_in_flows[n2,k]
    model.indeg_ineq_set = Set(initialize=pairs_in)    
    model.indeg_ineq=Constraint(model.indeg_ineq_set,rule=in_ineq_)

    def out_ineq_(model,n1,n2,k):
        return model.flows[((n1,n2),k)] == model.node_out_flows[n1,k]
    model.outdeg_ineq_set = Set(initialize=pairs_out) 
    model.outdeg_ineq=Constraint(model.outdeg_ineq_set,rule=out_ineq_)

    def keep_size_const_(model,n, k):
        return model.node_out_flows[n,k] == model.node_in_flows[n,k]
    model.pass_through = Constraint(idx_keep_size,[0,1,2],rule=keep_size_const_)

    model.pass_through_channels_pooling = Constraint(idx_pooling,[2],rule=keep_size_const_)
    
    def int_inq_(model, i, k):
        return model.node_in_flows[(i,k)] >= model.node_out_flows[(i,k)]
    model.internal_inequalities = Constraint(idx_change_size,[0,1],rule=int_inq_)

    def in_ineq_concat_(model,n2,k):
        return sum(model.flows[((n1,n2),k)] for n1 in G.predecessors(n2)) == model.node_out_flows[n2,k]

    model.concat_set = Set(initialize=concat_set)
    model.indeg_concat_ineq=Constraint(model.concat_set,rule=in_ineq_concat_)
    
    idx_keep_size_const_concat=[]
    for n,k in concat_set:
        idx_keep_size_const_concat+=[(n,k1) for k1 in range(3) if k1!=k]
    model.pass_through_concat = Constraint(idx_keep_size_const_concat,rule=keep_size_const_)

    def bin_equation_(model, n, k):
        return model.node_out_flows[(n,k)] == sum( model.bin_vars[(n,k,t)] * val_bin[(n,k,t)] for t in range(5) if (n,k,t) in idx_bin ) 
    model.bin_const = Constraint(idx_change_size,[0,1,2], rule = bin_equation_)

    def bin_1_in_s_(model, n, k):
        return 1 == sum( model.bin_vars[(n,k,t)] for t in range(5) if (n,k,t) in idx_bin) 
    model.bin_1_in_s = Constraint(idx_change_size,[0,1,2], rule = bin_1_in_s_)

    def fix_inputs_(model,n,k):
        return model.node_in_flows[(n,k)] == G.nodes[n]['res_in'][k]
    model.fix_inputs = Constraint(model.inputs,[0,1,2], rule=fix_inputs_)

    def fix_outputs_(model,n,k):
        return model.node_out_flows[(n,k)] == target_outputs[n][k]
    model.fix_outputs = Constraint(model.outputs,[0,1,2], rule=fix_outputs_)

    model.C_abs_diff = Var(idx_change_size,within=NonNegativeReals)

    def abs_lb_(model,n):
        return model.node_out_flows[(n,2)]-G.nodes[n]['res_out'][2]  >= -model.C_abs_diff[n]
    model.C_abs_lb = Constraint(idx_change_size,rule=abs_lb_)


    model.HWC_abs_diff = Var(idx_change_size,within=NonNegativeReals)
    def abs_hwc_up_(model,n):
        return sum( model.node_out_flows[(n,k)]-G.nodes[n]['res_out'][k] for k in range(3))  <= model.HWC_abs_diff[n]
    model.HWC_abs_up = Constraint(idx_change_size,rule=abs_hwc_up_)

    def abs_hwc_lb_(model,n):
        return sum( model.node_out_flows[(n,k)]-G.nodes[n]['res_out'][k] for k in range(3) ) >= -model.HWC_abs_diff[n]
    model.HWC_abs_lb = Constraint(idx_change_size,rule=abs_hwc_lb_)

    weight_hout = {}
    for n in node_names:
        weight_hout[n] = len(nx.descendants(G,n))

    def obj_channels(model):
        t_abs = sum(model.C_abs_diff[n]*weight_hout[n] for n in idx_change_size)
        return t_abs

    def obj_hwc(model):
        t_out = sum(model.HWC_abs_diff[n]*weight_hout[n] for n in idx_change_size)
        return t_out

    def obj_change_size(model):
        t_out = sum((model.node_in_flows[(i,k)]-model.node_out_flows[(i,k)])*weight_hout[i] for i in idx_change_size for k in range(2))
        return t_out

    def obj_keep_sizes(model):
        t_out = sum((model.node_in_flows[(i,k)])*weight_hout[i] for i in idx_keep_size for k in range(3))
        return t_out

    def obj_concat(model):
        t_out = sum((model.node_out_flows[(i,k)])*weight_hout[i] for i in idx_concat for k in range(3))
        return t_out

    def objective_function(model):
        ret = a*obj_channels(model) + b*obj_hwc(model)+c*obj_change_size(model) + d*obj_keep_sizes(model) + e*obj_concat(model)
        return  ret
    model.total = Objective(rule=objective_function, sense=minimize)

    return model

def resolution_propagate(D,target_outputs,a=1,b=1,c=1,d=1,e=1):
    for _, node in D.nodes(data=True):
        node['res_in']=tuple([node['resolution'][i] for i in [0,2,4]])
        node['res_out']=tuple([node['resolution'][i] for i in [1,3,5]])

    model = create_model_from_digraph(D, target_outputs,a,b,c,d,e)

    if model is None:
        return None, None, model

    results = solve_model(model)

    termination_cond = str(list(results['Solver'])[0]['Termination condition'])

    if not termination_cond == 'infeasible':

        out_dg = get_digraph_result(model,D)

    else:

        out_dg =None
     
    return termination_cond, out_dg, model


def get_digraph_result(model,digraph):

    ret = copy.deepcopy(digraph)

    for n in ret.nodes:

        ret.nodes[n]['res_in'] = tuple([int(model.node_in_flows[(n,t)].value) for t in range(3)])

        ret.nodes[n]['res_out'] = tuple([int(model.node_out_flows[(n,t)].value) for t in range(3)])

        aux = [[int(model.node_in_flows[(n,t)].value),int(model.node_out_flows[(n,t)].value)] for t in range(3)]

        ret.nodes[n]['resolution'] = reduce(lambda l,m: l+m,aux)
        ret.nodes[n]['resolution'] = tuple(ret.nodes[n]['resolution'])

        if ret.nodes[n]['op_name'] in ['conv2d','maxpool','avgpool']:

            ret.nodes[n]['strides'] = list(ret.nodes[n]['strides'])

            ret.nodes[n]['strides'][1] = ret.nodes[n]['res_in'][0] / ret.nodes[n]['res_out'][0]

            ret.nodes[n]['strides'][2] = ret.nodes[n]['res_in'][1] / ret.nodes[n]['res_out'][1]

            ret.nodes[n]['strides'] = tuple(ret.nodes[n]['strides'])

    return ret

def pyomo_postprocess(model):
  model.node_out_flows.display()

def solve_model(model, verbose=False, solver="glpk"):
    opt = SolverFactory(solver)
    results = opt.solve(model)

    if verbose:
        results.write()
        print("\nDisplaying Solution\n" + '-'*60)
        pyomo_postprocess( model)
    
    return results
