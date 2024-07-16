
import numpy as np
from .group_base import GroupBase
from scipy.stats import unitary_group

class UnitaryGroup(GroupBase):
    def __init__(self, num_qubit : int) -> None:
        """Constructor of UnitaryGroup class
        
        Arguments:
            num_qubit {int} -- number of qubits

        """
        self.num_qubit = num_qubit
        self.name = "Unitary"

    def _check_is_group(self) -> None:
        """test function to check the element list consists a group
        """
        pass

    def sample(self, count, seed=0):
        """randomly choose <code>count</code> of elements
        
        Arguments:
            count {int} -- number of samples
        
        Returns:
            list -- list of chosen elements
        """
        np.random.seed(seed)
        dim = 2**self.num_qubit

        element = []
        for _ in range(count):
            gate = unitary_group.rvs(dim)
            element.append(gate)
        return np.array(element)
    
def test_unitary():
    """test function for UnitaryGroup class    
    """
    num_qubit = 2
    dim = 2**num_qubit
    large_I = np.eye(dim)
    ug = UnitaryGroup(num_qubit)
    for u in ug.sample(100):
        assert(np.allclose(u@u.T.conj(), large_I))

if __name__ == "__main__":
    test_unitary()
