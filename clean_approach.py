import numpy as np
import copy, itertools

data = np.load('data 1.npy')
data_size = len(data[0])
bayes_net = dict()

def create_condtable(node, parents):
# Calculate conditional probabilites for permutations of the parent nodes from data
    l=[True,False]
    permutations = list(itertools.product(l,repeat=len(parents))) #Generate True False permutations
    permutations_size = len(permutations)

    cond_sizes = [0] * permutations_size
    node_true_count = [0] * permutations_size
    cond_probs = dict()

    for i in range(data_size):

        parents_cond = tuple(data[p][i] for p in parents) #Get current state of parents

        perm_index = permutations.index(parents_cond)
        cond_sizes[perm_index] += 1

        if data[node][i]:
            node_true_count[perm_index] += 1
            
    for i in range(permutations_size):
        cond_probs[permutations[i]] = node_true_count[i] / cond_sizes[i]

    return cond_probs

def init_baynet():
# Initialize bayesian network structure from the given data
    true_count = 0
    for val in data[0]:
        if val: true_count += 1

    prob_A = true_count / data_size
    bayes_net["A"] = {"children": ["C", "D"], "parents": [], "prob": prob_A, "condit_prob": {} }

    true_count = 0
    for val in data[1]:
        if val: true_count += 1

    prob_B = true_count / data_size
    bayes_net["B"] = {"children": ["D"], "parents": [], "prob": prob_B, "condit_prob": {} }

    c_cond_probs = create_condtable(2, [0])
    bayes_net["C"] = {"children": ["E", "F"], "parents": ["A"], "prob": -1, "condit_prob": c_cond_probs }

    d_cond_probs = create_condtable(3, [0, 1])

    bayes_net["D"] = {"children": ["G"], "parents": ["A", "B"], "prob": -1, "condit_prob": d_cond_probs }

    e_cond_probs = create_condtable(4, [2])
    bayes_net["E"] = {"children": [], "parents": ["C"], "prob": -1, "condit_prob": e_cond_probs }

    f_cond_probs = create_condtable(5, [2])
    bayes_net["F"] = {"children": [], "parents": ["C"], "prob": -1, "condit_prob": f_cond_probs }

    g_cond_probs = create_condtable(6, [3])
    bayes_net["G"] = {"children": [], "parents": ["D"], "prob": -1, "condit_prob": g_cond_probs }

def normalize(distribution):
# Normalize probabilities (add up to 1.0)
    return tuple(val / (sum(distribution)) for val in distribution)

def query(Y, evidence_dict):
    # No parents
    if bayes_net[Y]["prob"] != -1:
        if evidence_dict[Y]:
            prob = bayes_net[Y]["prob"]
        else:
            prob = 1 - bayes_net[Y]["prob"]
    else:
        # Get the parent values of Y
        parent_vals = tuple(evidence_dict[parent] for parent in bayes_net[Y]["parents"])
        # Probability of Y = y
        if evidence_dict[Y]:
            prob = bayes_net[Y]["condit_prob"][parent_vals]  
        else:
            prob = 1 - bayes_net[Y]["condit_prob"][parent_vals]
    return prob

def enum_all(variables, evidence_dict):
# Enumarate over all the given variables recursively
    if len(variables) == 0:
        return 1.0

    Y = variables[0]
    if Y in evidence_dict:
        result = query(Y, evidence_dict) * enum_all(variables[1:], evidence_dict)

    else:
        probabilities = []
        evidence_copy = copy.deepcopy(evidence_dict)

        for y in [True, False]:
            evidence_copy[Y] = y
            probabilities.append(query(Y, evidence_copy) * enum_all(variables[1:], evidence_copy))

        result = sum(probabilities) 

    return result

def enum_ask(query_dict, evidence_dict):
# Calculate the probability of given query with enumeration
    distribution = []
    l=[True,False]
    permutations = [list(i) for i in itertools.product(l,repeat=len(query_dict))] #Generate True False permutations

    query_vars = list(query_dict.keys())
    query_values = list(query_dict.values())

    target = permutations.index(query_values)
    
    for perm in permutations:

        evidence_copy = copy.deepcopy(evidence_dict)

        for i in range(len(query_vars)):
            evidence_copy[query_vars[i]] = perm[i]

        variables = list(bayes_net.keys())
        distribution.append(enum_all(variables, evidence_copy))

    normalized = normalize(distribution)
    return normalized[target]

def calc_with_data(query_dict, evidence_dict):
# Calculate the probability of given query by only using the data
    data_traversed = [] 
    for i in range(len(data[0])):
        values = []
        for j in range(len(data)):
            values.append(data[j][i])
        data_traversed.append(values)

    eviences_int = {} # Evidence variables dict index values as keys
    for evidence in evidence_dict:
        int_val = int(evidence, 32) - 10
        eviences_int[int_val] = evidence

    queries_int = {} # Query variables dict index values as keys
    for query in query_dict:
        int_val = int(query, 32) - 10
        queries_int[int_val] = query

    pool = [] # Part of the data where given evidences occur

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

def parse_variables(input):
# Parse the user input seperated with spaces
    variables = dict()
    for elem in input.split(" "):
        if len(elem) > 0 and len(elem) < 3:
            if elem[0] == "n":
                if elem[1] in variables:
                    return None
                variables[elem[1]] = False
            else:
                if elem[0] in variables:
                    return None
                variables[elem[0]] = True
    return variables


print("Please wait, initializing the bayesian network from the data...")
init_baynet()

while True:
# Main loop
    
    query_in = input("\nPlease give query variables: ")
    evidence_in = input("Please give evidence variables: ")

    queries = parse_variables(query_in)
    evidences = parse_variables(evidence_in)

    if queries == None:
        print("\nRepeating element on the query side")
        continue

    if evidences == None:
        print("\nRepeating element on the evidence side")
        continue

    if len(queries) == 0:
        print("\nQuery side cannot be empty")
        continue

    variable_set = {"A", "B", "C", "D", "E", "F", "G"}
    query_variables = list(queries.keys())
    query_set = set(query_variables)

    evidence_variables = list(evidences.keys())
    evidence_set = set(evidence_variables)

    if not set.union(query_set, evidence_set).issubset(variable_set):
        print("\nGiven variable does not exist")
        continue

    legal = True
    for var in evidence_set:
        if var in query_set:
            print("\nRepeating variable on two sides")
            legal = False

    if not legal:
        continue

    data_result = calc_with_data(queries, evidences)
    inferenece_result = enum_ask(queries, evidences)

    print("\nThe probability calculated by inference is " + str(inferenece_result))
    print("The probability calculated from data is " + str(data_result) + "\n")