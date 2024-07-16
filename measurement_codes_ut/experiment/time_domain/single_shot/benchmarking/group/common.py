import numpy as np


complex_type = np.complex128

I = np.eye(2,dtype=complex_type)
X = np.array([[0,1],[1,0]],dtype=complex_type)
Z = np.array([[1,0],[0,-1]],dtype=complex_type)
Y = 1.j*X@Z

H = np.ones((2,2),dtype=complex_type )
H[1,1]=-1
H/=np.sqrt(2)

S = np.eye(2,dtype=complex_type)
S[1,1] = 1.j

CZ = np.eye(4,dtype=complex_type)
CZ[3,3] = -1

def pauli_exp(pauli : np.ndarray,angle : float) -> np.ndarray:
    return np.cos(angle/2)*np.eye(2) + 1.j*np.sin(angle/2)*pauli

def list_product(list1 : list,list2 : list) -> list:
    list3 = []
    for item1 in list1:
        for item2 in list2:
            list3.append(item1@item2)
    return list3
