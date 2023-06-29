import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

def tensor(gate):
    out = gate[0]
    for g in gate[1:]:
        out = np.kron(out,g)
    return out

class ShapedDataset:
    def __init__(
        self,
        data_dict,
        ):
        self.keys = list(data_dict.keys())
        self.shots = data_dict[self.keys[0]].shape[-2]
        self.shape = data_dict[self.keys[0]].shape[:-2]
        shaped_value = []
        for key in self.keys:
            shaped_value.append(data_dict[key].reshape(int(np.prod(self.shape)), self.shots, 2))
        shaped_value = np.array(shaped_value).transpose(1,0,2,3)
        self.data = []
        for population in shaped_value:
            self.data.append(dict(zip(self.keys, population)))
            
    def get_histogram(self, histogramer, keys=None, mitigation=False):
        self.histogram = []
        for data_dict in self.data:
            self.histogram.append(histogramer.get_histogram(data_dict, keys, mitigation))
            
    def get_pauli(self, histogramer, keys=None, mitigation=False):
        self.get_histogram(histogramer, keys, mitigation)
        self.pauli = {}
        for i, qubit in enumerate(self.keys):
            pauli = []
            for histogram in self.histogram:
                expv = 0.
                for key, population in histogram.items():
                    if key[i] == "1":
                        expv -= population
                    else:
                        expv += population
                expv /= sum(histogram.values())
                pauli.append(expv)
            self.pauli[qubit] = np.array(pauli)
            
class Projector:
    def __init__(
        self,
        model = LogisticRegression,
        const = {'C':np.logspace(-5, 5, 11)},
        options = {"solver":"lbfgs", "multi_class":"auto"}
        ):
        self.model = model
        self.const = const
        self.options = options
        self.cls = None
        self.train_data = None
        self.label = ["0","1"]
        
    def train(self, train_data):
        classifer = self.model()
        for key in self.options.keys():
            classifer.__dict__[key] = self.options[key]
        pipeline = Pipeline([('standard_scaler', StandardScaler()),(self.model.__name__, classifer)])
        params = {}
        for i,j in zip(self.const.keys(),self.const.values()):
            params.update({self.model.__name__ + '__' + i : j})
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
        self.cls = GridSearchCV(pipeline, param_grid=params, cv=cv)
        self.cls.fit(train_data['data'], train_data['label'])
        self.train_data = train_data
        self.analyze(self.train_data)
        
    def analyze(self, test_data):
        self.fidelity = accuracy_score(test_data['label'], self.cls.predict(test_data['data']))
        conf_mat = confusion_matrix(test_data['label'], self.cls.predict(test_data['data']))
        self.shots = conf_mat.sum(axis=1)[0]
        self.conf_mat = conf_mat/self.shots
        
    def predict(self, data):
        key = self.cls.predict(data)
        return key
    
class Histogramer:
    def __init__(self, proj_dict):
        self.projector = proj_dict
        self.conf_mat = tensor([proj.conf_mat for proj in proj_dict.values()])
        self.conf_inv = la.pinv(self.conf_mat)
        
    def get_histogram(self, data_dict, keys=None, mitigation='least_squares'):
        if keys is None:
            keys = self.projector.keys()
            
        self.n = len(keys)
        self.hist_key = ["".join(i) for i in itertools.product(["0","1"],repeat=self.n)]
            
        labels = []
        for qubit_name in keys:
            projector = self.projector[qubit_name]
            data = data_dict[qubit_name]
            label = projector.predict(data)
            labels.append(label)
        labels = np.array(labels).T
        events = ["".join(event_label) for event_label in labels]
        histogram = {}
        for key in self.hist_key:
            histogram[key] = events.count(key)
            
        if mitigation == 'pseudo_inverse':
            population = list(histogram.values())
            population = np.dot(self.conf_inv, population)
            histogram = dict(zip(self.hist_key, population))
            
        elif mitigation == 'least_squares':
            population = list(histogram.values())
            shots = sum(population)
            
            def fun(x):
                return sum((population - np.dot(self.conf_mat, x))**2)
            
            x0 = np.random.rand(len(population))
            x0 = x0 / sum(x0)
            cons = ({'type': 'eq', 'fun': lambda x: shots - sum(x)})
            bnds = tuple((0, shots) for x in x0)
            res = opt.minimize(fun, x0, method='SLSQP', constraints=cons, bounds=bnds, tol=1e-6)
            population = res.x
            histogram = dict(zip(self.hist_key, population))
        else:
            pass
        return histogram

class MultiProjector(Histogramer):
    def __init__(self, proj_dict):
        super().__init__(proj_dict)
        
    def get_pauli_dict(self, data_dict, mitigation="least_squares"):
        sd = ShapedDateset(data_dict)
        sd.get_pauli(self, mitigation=mitigation)
        paulis = {}
        for qubit_name, pauli in sd.pauli.items():
            paulis[qubit_name] = np.array(pauli).reshape(sd.shape)
        return paulis