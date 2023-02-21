import pennylane as qml
from pennylane import numpy as np
import networkx as nx

# Intial Variables
n_qubits = 8


G = nx.grid_2d_graph(3, 3) # How to make the grid periodic?

dev = qml.device("default.qubits", wires=n_qubits)




# Mathematical Functions

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def prob_dist(params):
    return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T



# Initialize Grid
def intiialize_grid(shape, n):   
    grid = np.array(range(n))
    grid = grid.reshape(shape)
    
    return grid
            
    
# Thermodynamic Functions 
   
def hamiltonian(n, graph):
    
    matrix = np.zeros((2 ** n, 2 ** n))

    for i in graph.edges:
        x = y = z = 1
        for j in range(0, n):
            if j == i[0] or j == i[1]:
                x = np.kron(x, qml.matrix(qml.PauliX)(0))
                y = np.kron(y, qml.matrix(qml.PauliY)(0))
                z = np.kron(z, qml.matrix(qml.PauliZ)(0))
            else:
                x = np.kron(x, np.identity(2))
                y = np.kron(y, np.identity(2))
                z = np.kron(z, np.identity(2))

        matrix = np.add(matrix, np.add(x, np.add(y, z)))

    return matrix
            
            


def trotterize_partition(m, beta, n, graph):
    
    exp_H = np.exp((-beta/m)*hamiltonian(n, graph))
    
    Z = np.trace(np.lingalg.matrix_power(exp_H, m))
    
      
    
    return Z

        

        
        
        
            
            
 
  
    
    
    
    
