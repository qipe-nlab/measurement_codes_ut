import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics         import accuracy_score, confusion_matrix

def my_prod(shape):
    if shape is ():
        return 1
    else:
        return np.prod(shape)

def tensor(gate):
    out = gate[0]
    for g in gate[1:]:
        out = np.kron(out,g)
    return out

def extract_last(data):
    shape       = data.shape[:-1]
    shaped_data = data.reshape([my_prod(data.shape[:-1]),data.shape[-1]])
    return shaped_data, shape

def extract_first(data):
    shape       = data.shape[1:]
    shaped_data = data.reshape([data.shape[0],my_prod(data.shape[1:])])
    return shaped_data, shape

def show_confusion_matrix(confusion_matrix,qubit=True):
    plt.figure(figsize=(6,5))
    plt.imshow(confusion_matrix)
    ys, xs = np.meshgrid(range(confusion_matrix.shape[0]),range(confusion_matrix.shape[1]),indexing="ij")
    for (x,y,val) in zip(xs.flatten(), ys.flatten(), confusion_matrix.flatten()):
        plt.text(x,y,"{0:.2f}".format(val), horizontalalignment="center",verticalalignment="center",color="w")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if qubit:
        n = int(np.log2(confusion_matrix.shape[0]))
        plt.xticks(range(2**n),[''.join(i) for i in list(itertools.product(['0','1'],repeat=n))])
        plt.yticks(range(2**n),[''.join(i) for i in list(itertools.product(['0','1'],repeat=n))])
    plt.colorbar()
    plt.clim(0,1)
    plt.show()

class Projecter(object):
    def __init__(self,
        model   = LogisticRegression,
        const   = {'C':np.logspace(-5, 5, 11)},
        options = {"solver":"lbfgs", "multi_class":"auto"}
        ):
        
        self.model      = model
        self.const      = const
        self.options    = options
        self.cls        = None
        self.train_data = None
        self.label      = ["0","1"]
    
    def train(self,train_data):
        classifer       = self.model()
        for key in self.options.keys():
            classifer.__dict__[key] = self.options[key]
        pipeline        = Pipeline([('standard_scaler', StandardScaler()),(self.model.__name__, classifer)])
        params          = {}
        for i,j in zip(self.const.keys(),self.const.values()):
            params.update({self.model.__name__ + '__' + i : j})
        cv              = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
        cls             = GridSearchCV(pipeline, param_grid=params, cv=cv)
        cls.fit(train_data['data'], train_data['label'])
        self.cls        = cls
        self.train_data = train_data
        self.analyze(self.train_data)

    def analyze(self,test_data):
        fidelity                = accuracy_score(test_data['label'], self.cls.predict(test_data['data']))
        confusion_mat           = confusion_matrix(test_data['label'], self.cls.predict(test_data['data']))
        self.fidelity           = fidelity
        self.shots              = confusion_mat.sum(axis=1)[0]
        self.confusion_matrix   = confusion_mat/self.shots
        self.confusion_inv      = np.linalg.inv(self.confusion_matrix)
            
    def get_label(self,data):
        shaped_data, shape      = extract_last(data)
        shaped_label            = self.cls.predict(shaped_data)
        label                   = shaped_label.reshape(shape)
        return label

    def get_histogram(self,data,mittigation=True):
        label                   = self.get_label(data)
        shaped_label, shape     = extract_last(label)
        shaped_histogram        = np.array([[i.count(j) for j in self.label] for i in shaped_label.tolist()])
        if mittigation:
            shaped_histogram    = np.einsum('ij,ki->kj',self.confusion_inv, shaped_histogram)
        if shape is ():
            return shaped_histogram
        else:
            histogram               = shaped_histogram.reshape(shape + [len(self.label)])
            return histogram

class MultiProjector:
    def __init__(self,projector_dict):

        self.keys               = []
        self.projector          = {}
        confusion_matrix_list   = []
        for k,v in projector_dict.items():
            self.keys.append(k)
            self.projector[k] = v
            confusion_matrix_list.append(v.confusion_matrix)
        self.confusion_matrix   = tensor(confusion_matrix_list)
        self.confusion_inv      = np.linalg.inv(self.confusion_matrix)
        self.label              = ["".join(i) for i in itertools.product(["0","1"],repeat=len(self.keys))]
        self.population         = np.array(list(itertools.product(range(2),repeat=len(self.keys))),dtype=np.float64)
        show_confusion_matrix(self.confusion_matrix)

    def get_histogram_dict(self,data_dict,mittigation=True):
        label               = np.array([self.projector[k].get_label(data_dict[k]) for k in self.keys])
        shaped_label, shape = extract_first(label)
        label               = np.array(["".join(i) for i in shaped_label.T]).reshape(shape)
        shaped_label, shape = extract_last(label)
        shaped_histogram    = np.array([[i.count(j) for j in self.label] for i in shaped_label.tolist()])
        if mittigation:
            shaped_histogram    = np.einsum('ij,ki->kj',self.confusion_inv, shaped_histogram)
        if shape is ():
            return np.array([dict(zip(self.label,i)) for i in shaped_histogram])
        else:
            return np.array([dict(zip(self.label,i)) for i in shaped_histogram]).reshape(shape)

    def get_pauli_dict(self, data_dict, mittigation=True):
        label               = np.array([self.projector[k].get_label(data_dict[k]) for k in self.keys])
        shaped_label, shape = extract_first(label)
        label               = np.array(["".join(i) for i in shaped_label.T]).reshape(shape)
        shaped_label, shape = extract_last(label)
        shaped_histogram    = np.array([[i.count(j) for j in self.label] for i in shaped_label.tolist()])
        shot                = shaped_histogram.sum(axis=1)
        if mittigation:
            shaped_histogram    = np.einsum('ij,ki->kj',self.confusion_inv, shaped_histogram)
        for i in range(np.prod(shape)):
            shaped_histogram[i] /= shot[i]
        pauli               = 1 - 2*np.einsum('ij,ki->jk',self.population,shaped_histogram).reshape([len(self.keys)]+list(shape))
        pauli_dict          = dict(zip(self.keys,pauli))
        return pauli_dict