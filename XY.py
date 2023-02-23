import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize
import networkx as nx
import itertools
from numpy import savetxt
from numba import jit, cuda

# Initial variables
depth = 4
nr_qubits = 4
h = 1
beta_list = [round(1/k,2) for k in range(2,1,-1)]+[k for k in range(1,2)]
T = [1/beta for beta in beta_list]
E = []
C = []
M = []

interaction_graph = nx.grid_graph((2,2), periodic=True)
dev = qml.device("default.qubit", wires=nr_qubits)


#Hamiltonian

def create_hamiltonian_matrix(n, graph):
    matrix = np.zeros((2 ** n, 2 ** n))

    for i in graph.edges:
        x = y = z = 1
        for j in range(0, n):
            z = np.kron(z, qml.matrix(qml.PauliZ)(0))           
            if j == i[0] or j == i[1]:
                x = np.kron(x, qml.matrix(qml.PauliX)(0))
                y = np.kron(y, qml.matrix(qml.PauliY)(0))
                
            else:
                x = np.kron(x, np.identity(2))
                y = np.kron(y, np.identity(2))
                
                
        matrix = np.add(matrix, np.add(np.add(x, y),h*z))

    return matrix


ham_matrix = create_hamiltonian_matrix(nr_qubits, interaction_graph)



# Utilty Functions


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T



def convert_list(params):

    # Separates the list of parameters
    dist_params = params[0:nr_qubits]
    ansatz_params_1 = params[nr_qubits : ((depth + 1) * nr_qubits)]
    ansatz_params_2 = params[((depth + 1) * nr_qubits) :]

    coupling = np.split(ansatz_params_1, depth)

    # Partitions the parameters into multiple lists
    split = np.split(ansatz_params_2, depth)
    rotation = []
    for s in split:
        rotation.append(np.split(s, 3))

    ansatz_params = [rotation, coupling]

    return [dist_params, ansatz_params]


def calculate_entropy(distribution):

    total_entropy = 0
    for d in distribution:
        total_entropy += -1 * d[0] * np.log(d[0]) + -1 * d[1] * np.log(d[1])

    # Returns an array of the entropy values of the different initial density matrices

    return total_entropy


# Ansatz

def single_rotation(phi_params, qubits):

    rotations = ["Z", "Y", "X"]
    for i in range(0, len(rotations)):
        qml.AngleEmbedding(phi_params[i], wires=qubits, rotation=rotations[i])



@qml.qnode(dev)
def quantum_circuit(rotation_params, coupling_params, sample=None):

    # Prepares the initial basis state corresponding to the sample
    qml.BasisStatePreparation(sample, wires=range(nr_qubits))

    # Prepares the variational ansatz for the circuit
    for i in range(0, depth):
        single_rotation(rotation_params[i], range(nr_qubits))
        qml.broadcast(
            unitary=qml.CRX,
            pattern="ring",
            wires=range(nr_qubits),
            parameters=coupling_params[i]
        )

    # Calculates the expectation value of the Hamiltonian with respect to the prepared states
    return qml.expval(qml.Hermitian(ham_matrix, wires=range(nr_qubits)))

qnode = qml.QNode(quantum_circuit, dev)



@jit(target_backend='cuda')      
def exact_cost(params,beta):

    global iterations

    # Transforms the parameter list
    parameters = convert_list(params)
    dist_params = parameters[0]
    ansatz_params = parameters[1]

    # Creates the probability distribution
    distribution = prob_dist(dist_params)

    # Generates a list of all computational basis states of our qubit system
    combos = itertools.product([0, 1], repeat=nr_qubits)
    s = [list(c) for c in combos]

    # Passes each basis state through the variational circuit and multiplies
    # the calculated energy EV with the associated probability from the distribution
    cost = 0
    for i in s:
        result = quantum_circuit(ansatz_params[0], ansatz_params[1], sample=i)
        for j in range(0, len(i)):
            result *= distribution[j][i[j]]
        cost += result
       

    # Calculates the entropy and the final cost function
    entropy = calculate_entropy(distribution)
    final_cost = beta * cost - entropy

    return final_cost, cost


def cost_execution(params,beta):

    global iterations

    cost, _ = exact_cost(params,beta)
   
    cost_list.append(cost)
    if iterations % 50 == 0:
        print("Cost at Step {}: {}".format(iterations, cost))

    iterations += 1
    return cost

beta_list = [round(1/k,2) for k in range(10,1,-1)]+[k for k in range(1,11)]
#beta = 1/10 1/9.. 1 2 3.. 10


def prepare_state(params, device):

    # Initializes the density matrix

    final_density_matrix = np.zeros((2 ** nr_qubits, 2 ** nr_qubits))

    # Prepares the optimal parameters, creates the distribution and the bitstrings
    parameters = convert_list(params)
    dist_params = parameters[0]
    unitary_params = parameters[1]

    distribution = prob_dist(dist_params)

    combos = itertools.product([0, 1], repeat=nr_qubits)
    s = [list(c) for c in combos]

    # Runs the circuit in the case of the optimal parameters, for each bitstring,
    # and adds the result to the final density matrix

    for i in s:
        quantum_circuit(unitary_params[0], unitary_params[1], sample=i)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]]) * state
        final_density_matrix = np.add(final_density_matrix, np.outer(state, np.conj(state)))

    return final_density_matrix

for beta in beta_list:
    print(f'Starting beta={beta} training')
    iterations = 0
    cost_list = []

    number = nr_qubits * (1 + depth * 4)
    params = [np.random.randint(-300, 300) / 100 for i in range(0, number)]
    out = minimize(cost_execution, x0=params, args=(beta),method="COBYLA", options={"maxiter": 1600})
    out_params = out["x"]
    prep_density_matrix = prepare_state(out_params, dev)
    savetxt(f'beta_is_{beta}.csv', cost_list, delimiter=',')
    savetxt(f'beta_is_{beta}_params.csv', out_params, delimiter=',')
    params = np.array(np.loadtxt(f'beta_is_{beta}_params.csv', dtype=float))
    cost, energy = exact_cost(params, beta)
    variance = np.abs(np.trace(np.matmul(np.linalg.matrix_power(ham_matrix,2),prep_density_matrix)-np.matmul(prep_density_matrix, ham_matrix)))
    E.append(energy)
    C.append((beta/nr_qubits)**2*variance)
    
    
    z_matrix = 1
    for i in range(np.shape(prep_density_matrix)[0]):
        z_matrix = np.kron(z_matrix, qml.matrix(qml.PauliZ(0)))
    
    
    magnetization = np.abs(np.trace(np.matmul(prep_density_matrix, z_matrix)))
    
    M.append(magnetization)
    
    



plt.plot(T,E)
plt.xlabel('T')
plt.ylabel('E')

plt.plot(T,C)
plt.xlabel('T')
plt.ylabel('C')

plt.show()
    
    
    








