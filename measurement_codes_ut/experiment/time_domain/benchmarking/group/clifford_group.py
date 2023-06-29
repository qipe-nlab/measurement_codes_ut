
import numpy as np
import itertools
from .group_base import GroupBase
from .common import I,X,Y,Z,H,S,CZ, list_product

"""Reference
Unique decomposition https://arxiv.org/abs/1310.6813
"""

class CliffordGroup(GroupBase):
    def __init__(self, num_qubit : int) -> None:
        """Constructor of CliffordGroup class
        
        Arguments:
            num_qubit {int} -- number of qubits

        """
        self.num_qubit = num_qubit
        self.name = "Clifford"

        IH = np.kron(I,H)
        HI = np.kron(H,I)
        HH = np.kron(H,H)
        SI = np.kron(S,I)
        IS = np.kron(I,S)

        A1 = I
        A2 = H
        A3 = H@S@H
        B1 = IH@CZ@HH@CZ@HH@CZ
        B2 = CZ@HH@CZ
        B3 = HI@SI@CZ@HH@CZ
        B4 = HI@CZ@HH@CZ
        C1 = I
        C2 = H@S@S@H
        D1 = CZ@HH@CZ@HH@CZ@IH
        D2 = HI@CZ@HH@CZ@IH
        D3 = HH@IS@CZ@HH@CZ@IH
        D4 = HH@CZ@HH@CZ@IH
        E1 = I
        E2 = S
        E3 = S@S
        E4 = S@S@S
        A = [A1,A2,A3]
        B = [B1,B2,B3,B4]
        C = [C1,C2]
        D = [D1,D2,D3,D4]
        E = [E1,E2,E3,E4]

        L = []
        M = []
        for ind_qubit in range(num_qubit):
            if ind_qubit == 0:
                Lc = C
                Mc = E
                Llast = []
            else:
                Llast = [np.kron(item,I) for item in L[-1]]
            shiftI = np.eye(2**ind_qubit)
            Al = [np.kron(shiftI,item) for item in A]
            L.append( Llast + list_product(Al,Lc) )
            M.append(Mc)
            if ind_qubit+1 < num_qubit:
                shiftI = np.eye(2**ind_qubit)
                LcL = [np.kron(shiftI,item) for item in B]
                LcR = [np.kron(item,I) for item in Lc]
                McL = [np.kron(item,shiftI) for item in D]
                McR = [np.kron(I,item) for item in Mc]
                Lc = list_product(LcL,LcR)
                Mc = list_product(McL,McR)

        N = []
        for ind_qubit in range(num_qubit):
            N.append(list_product(L[ind_qubit],M[ind_qubit]))
        
        Nc = N[0]
        for ind_qubit in range(1,num_qubit):
            temp = []
            for item1 in N[ind_qubit]:
                for item2 in Nc:
                    temp.append(item1@np.kron(item2,I))
            Nc = temp
        self.element = np.array(Nc)
    
def test_clifford():
    """test function for Clifford class    
    """
    def order(num_qubit : int) -> int:
        a=1
        for ind in range(1,num_qubit+1):
            a*=(2*(4**ind-1)*(4**ind))
        return a

    num_qubit = 1
    cg = CliffordGroup(num_qubit)
    cg.sample(10)
    cg._check_is_group()
    assert(order(num_qubit)==len(cg.element))

    num_qubit = 2
    cg = CliffordGroup(num_qubit)
    cg.sample(10)
    #cg._check_is_group() # too heavy
    assert(order(num_qubit)==len(cg.element))

if __name__ == "__main__":
    test_clifford()
