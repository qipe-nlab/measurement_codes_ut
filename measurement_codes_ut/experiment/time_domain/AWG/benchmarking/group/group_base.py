
import numpy as np

class GroupBase():
    def __init__(self):
        self.element = None
        raise NotImplementedError("This is abstract class")

    def _check_is_group(self) -> None:
        """test function to check the element list consists a group
        """

        cnt=0
        for ind1 in range(len(self.element)):
            e1 = self.element[ind1]
            for ind2 in range(len(self.element)):
                e2 = self.element[ind2]
                e = e1@e2
                flag=False
                for e3 in self.element:
                    ee = e@e3.T.conjugate()
                    if( np.abs(ee[0,0]-ee[1,1]) < 1e-10 and np.abs(ee[0,1])<1e-10 and np.abs(ee[1,0])<1e-10 ):
                        flag = True
                if(not flag):
                    print("* error at : ", ind1,ind2)
                    cnt+=1
        assert(cnt==0)

    def sample(self, count, seed=0):
        """randomly choose <code>count</code> of elements
        
        Arguments:
            count {int} -- number of samples
        
        Returns:
            list -- list of chosen elements
        """
        np.random.seed(seed)
        return self.element[np.random.choice(self.element.shape[0],size=count,replace=True)]
