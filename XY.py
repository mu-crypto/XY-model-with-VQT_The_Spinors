import pennylane as qml
from pennylane import numpy as np


n= 8

def hamiltonian(i, j):

    
    
    matrix = np.zeros((2 ** n, 2 ** n))
    
    x=y=1

    
    for k in range(n):
        if k == i or k == j:
            x = np.kron(x, qml.matrix(qml.PauliX)(0))
            y = np.kron(y, qml.matrix(qml.PauliY)(0))
        else:
            x = np.kron(x, np.identity(2))
            y = np.kron(y, np.identity(2))
    matrix = np.add(matrix, np.add(x, y))
    
    return matrix
    
            
            


def trotterize_partition(m, beta):
    
    
    matrix = 1
    for i in range(n):
        for j in range(i+1, n):
            exp_ham = np.exp(-beta/m*hamiltonian(i,j))
            np.matmul(matrix, exp_ham)
            
    Z = np.trace(np.linalg.matrix_power(matrix, m))
    
    return Z
        
            
        
        
        
            
            
 
  
    
    
    
    