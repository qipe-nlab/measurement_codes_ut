import numpy as np
from .group_base import GroupBase
from .common import I,X,Y,Z,pauli_exp

"""Reference
icosahedral RB https://arxiv.org/abs/1406.3364
unitary reflection group http://www.math.ucsd.edu/~nwallach/shephard-todd.pdf
"""


class IcosahedralGroup(GroupBase):
    def __init__(self):
        """Constructor of IcosahedralGroup class
        """
        self.num_qubit = 1
        self.name = "Icosahedral"

        phi = np.arctan([ (1+np.sqrt(5.))/2. ])[0]
        Pauli = [X,Y,Z]

        # identity
        element = [I]

        # vertices - 2pi/5, 4pi/5
        for ind in range(3):
            for a1 in [phi,-phi]:
                for a2 in [np.pi*2/5, np.pi*4/5,-np.pi*2/5, -np.pi*4/5]:
                    op = pauli_exp(Pauli[(ind+1)%3], a1) \
                        @ pauli_exp(Pauli[ind], a2) \
                        @ pauli_exp(Pauli[(ind+1)%3],-a1)
                    element.append(op)
        # edges - pi
        p1 = pauli_exp(X,np.pi)
        p2 = pauli_exp(Y,np.pi)
        p3 = pauli_exp(Z,np.pi)
        for a1 in [0, np.pi*2/5, -np.pi*2/5, np.pi*4/5, -np.pi*4/5]:
            for p in [p1,p2,p3]:
                op = pauli_exp(X,phi) \
                    @ pauli_exp(Z,a1) \
                    @ pauli_exp(X,-phi) \
                    @ p \
                    @ pauli_exp(X,phi) \
                    @ pauli_exp(Z,-a1) \
                    @ pauli_exp(X,-phi)
                element.append(op)

        # faces
        p1 = pauli_exp(X,-np.pi/2)@ pauli_exp(Y,-np.pi/2)
        p2 = pauli_exp(Y,np.pi/2)@ pauli_exp(X,np.pi/2)
        for a1 in [0,-np.pi*2/5, -np.pi*4/5, np.pi*2/5]:
            for p in [p1,p2]:
                op = pauli_exp(X,phi) \
                    @ pauli_exp(Z,a1) \
                    @ pauli_exp(X,-phi) \
                    @ p \
                    @ pauli_exp(X,phi) \
                    @ pauli_exp(Z,-a1) \
                    @pauli_exp(X,-phi)
                element.append(op)

        p1 = pauli_exp(X,-np.pi/2)@ pauli_exp(Y,np.pi/2)
        p2 = pauli_exp(Y,-np.pi/2)@ pauli_exp(X,np.pi/2)
        element.append(p1)
        element.append(p2)

        p1 = pauli_exp(X,np.pi/2)@ pauli_exp(Y,np.pi/2)
        p2 = pauli_exp(Y,-np.pi/2)@ pauli_exp(X,-np.pi/2)
        for a1 in [0, -np.pi*4/5, np.pi*4/5, np.pi*2/5]:
            for p in [p1,p2]:
                op = pauli_exp(X,phi) \
                    @ pauli_exp(Z,a1) \
                    @ pauli_exp(X,-phi) \
                    @ p \
                    @ pauli_exp(X,phi) \
                    @ pauli_exp(Z,-a1) \
                    @pauli_exp(X,-phi)
                element.append(op)

        element.append( pauli_exp(X,np.pi/2)@ pauli_exp(Y,-np.pi/2) )
        element.append( pauli_exp(Y,np.pi/2)@ pauli_exp(X,-np.pi/2) )

        self.element = np.array(element)

def test_icosahedral():
    """test function for icosahedral
    """
    ig = IcosahedralGroup()
    Icosahedral_order = 60
    ig._check_is_group()
    ig.sample(10)
    assert(len(ig.element) == Icosahedral_order)

if __name__ == "__main__":
    test_icosahedral()