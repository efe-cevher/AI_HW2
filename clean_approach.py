import numpy as np
import copy, itertools

data = np.load('data1.npy')
data_size = len(data[0])
bayes_net = dict()

def create_cond_table(var0_index, var1_index):
    var0_true_count = 0
    var1_true_count = [0, 0]
    cond_probs = dict()

    for i in range(data_size):
        if data[var0_index][i]:
            if data[var1_index][i]:
                var1_true_count[0] +=1
            var0_true_count += 1
            
        else:
            if data[var1_index][i]:
                var1_true_count[1] +=1
        
    cond_probs[(True,)] = var1_true_count[0]/var0_true_count
    cond_probs[(False,)] = var1_true_count[1]/(data_size - var0_true_count)

    return cond_probs

def create_cond_table2(var0_index, var1_index, var2_index):
    cond_sizes = [0, 0, 0, 0]
    var2_true_count = [0, 0, 0 ,0]
    cond_probs = dict()

    for i in range(data_size):
        if data[var0_index][i]:
            if data[var1_index][i]:
                if data[var2_index][i]:
                    var2_true_count[0] +=1
                cond_sizes[0] += 1
            else:
                if data[var2_index][i]:
                    var2_true_count[1] +=1
                cond_sizes[1] += 1
        else:
            if data[var1_index][i]:
                if data[var2_index][i]:
                    var2_true_count[2] +=1
                cond_sizes[2] += 1
            else:
                if data[var2_index][i]:
                    var2_true_count[3] +=1
                cond_sizes[3] += 1

    cond_probs[(True, True)] = var2_true_count[0]/cond_sizes[0]
    cond_probs[(True, False)] = var2_true_count[1]/cond_sizes[1]
    cond_probs[(False, True)] = var2_true_count[2]/cond_sizes[2]
    cond_probs[(False, False)] = var2_true_count[3]/cond_sizes[3]

    return cond_probs

def func(node, parents):
    l=[True,False]
    combinations = list(itertools.product(l,repeat=len(parents)))
    combinations_size = len(combinations)

    cond_sizes = [0] * combinations_size
    node_true_count = [0] * combinations_size
    cond_probs = dict()

    for i in range(data_size):

        parents_cond = tuple(data[p][i] for p in parents)

        if parents_cond in combinations:
            comb_index = combinations.index(parents_cond)
            cond_sizes[comb_index] += 1
            if data[node][i]:
                node_true_count[comb_index] += 1
            
    for i in range(combinations_size):
        cond_probs[combinations[i]] = node_true_count[i] / cond_sizes[i]

    return cond_probs

def init_baynet():
    true_count = 0
    for val in data[0]:
        if val:
            true_count += 1
    prob_A = (true_count / data_size)
    bayes_net["A"] = {"children": ["C", "D"], "parents": [], "prob": prob_A, "condit_prob": {} }

    true_count = 0
    for val in data[1]:
        if val:
            true_count += 1
    prob_B = (true_count / data_size)
    bayes_net["B"] = {"children": ["D"], "parents": [], "prob": prob_B, "condit_prob": {} }

    c_cond_probs = create_cond_table(2, [0])
    bayes_net["C"] = {"children": ["E", "F"], "parents": ["A"], "prob": -1, "condit_prob": c_cond_probs }

    d_cond_probs = create_cond_table(3, [0, 1])
    bayes_net["D"] = {"children": ["G"], "parents": ["A", "B"], "prob": -1, "condit_prob": d_cond_probs }

    e_cond_probs = create_cond_table(4, [2])
    bayes_net["E"] = {"children": [], "parents": ["C"], "prob": -1, "condit_prob": e_cond_probs }

    f_cond_probs = create_cond_table(5, [2])
    bayes_net["F"] = {"children": [], "parents": ["C"], "prob": -1, "condit_prob": f_cond_probs }

    g_cond_probs = create_cond_table(6, [3])
    bayes_net["G"] = {"children": [], "parents": ["D"], "prob": -1, "condit_prob": g_cond_probs }

def normalize(distribution):
    return tuple(val * 1 / (sum(distribution)) for val in distribution)

def toposort():
    variables = list(bayes_net.keys())
    print(variables)
    s = set()
    l = []
    while len(s) < len(variables):
        for v in variables:
            if v not in s and all(x in s for x in bayes_net[v]['parents']):
                s.add(v)
                l.append(v)
    print(l)
    return l

def querygiven(Y, e):
    # Y has no parents
    if bayes_net[Y]['prob'] != -1:
        if e[Y]:
            prob = bayes_net[Y]['prob']
        else:
            prob = 1 - bayes_net[Y]['prob']

    # Y has at least 1 parent
    else:
        # get the value of parents of Y
        parents = tuple(e[p] for p in bayes_net[Y]['parents'])
        
        # query for prob of Y = y
        
        prob = bayes_net[Y]['condit_prob'][parents] if e[Y] else 1 - bayes_net[Y]['condit_prob'][parents]
    return prob

def enum_all(variables, e):

    if len(variables) == 0:
        return 1.0
    Y = variables[0]
    if Y in e:
        ret = querygiven(Y, e) * enum_all(variables[1:], e)
    else:
        probs = []
        e2 = copy.deepcopy(e)
        for y in [True, False]:
            e2[Y] = y
            probs.append(querygiven(Y, e2) * enum_all(variables[1:], e2))
        ret = sum(probs) 

    return ret

def enum_ask(X, e):
    dist= []
    l=[True,False]
    combinations = [list(i) for i in itertools.product(l,repeat=len(X))]

    X_list = list(X.keys())
    X_values = list(X.values())

    target = combinations.index(X_values)
    
    for c in combinations:

        e = copy.deepcopy(e)

        for i in range(len(X_list)):
            e[X_list[i]] = c[i]

        variables = toposort()

        dist.append(enum_all(variables, e))

    normalized = normalize(dist)
    return normalized[target]

def calc_with_data(query_dict, evidence_dict):
    data_traversed = []
    for i in range(len(data[0])):
        values = []
        for j in range(len(data)):
            values.append(data[j][i])
        data_traversed.append(values)

    eviences_int = {}
    for evidence in evidence_dict:
        int_val = int(evidence, 32) - 10
        eviences_int[int_val] = evidence

    queries_int = {}
    for query in query_dict:
        int_val = int(query, 32) - 10
        queries_int[int_val] = query

    pool = []

    for line in data_traversed:
        satisfied = True
        for evidence in eviences_int:
            if not line[evidence] == evidence_dict[eviences_int[evidence]]:
                satisfied = False
                break
        if satisfied:
            pool.append(line)
            
    success_count = 0

    for line in pool:
        satisfied = True
        for query in queries_int:
            if not line[query] == query_dict[queries_int[query]]:
                satisfied = False
                break
        if satisfied:
            success_count += 1

    evidence_total = len(pool)

    return success_count / evidence_total


init_baynet()

#UI
query_in = input("Please give query variables: ")
evidence_in = input("Please give evidence variables: ")

def parse_variables(input):
    variables = dict()
    for elem in input.split(" "):
        if len(elem) > 0 and len(elem) < 3:
            if elem[0] == "n":
                variables[elem[1]] = False
            else:
                variables[elem[0]] = True
    return variables

queries = parse_variables(query_in.upper())
evidences = parse_variables(evidence_in.upper())

if len(queries) == 0:
    print("\nQuery side cannot be empty")
    exit()

variable_set = {"A", "B", "C", "D", "E", "F", "G"}

query_set = set(queries.keys())
evidence_set = set(evidences.keys())

if not set.union(query_set, evidence_set).issubset(variable_set):
    print("\nA given variable does not exist")
    exit()

data_result = calc_with_data(queries, evidences)
inferenece_result = enum_ask(queries, evidences)

print("The probability calculated by inference is " + str(inferenece_result))
print("The probability calculated from data is " + str(data_result))