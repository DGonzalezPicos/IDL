import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class XorNet(object):
    def __init__(self,
                 settings=None, **kwargs):

        ### META
        self.rng = np.random.default_rng()

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 0.1,
            "epochs": 50000,
            "weight_scale": 1
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = self.settings["epochs"]

        ### MAIN MCP
        # set initial weights, if they are initialized at w_i!=0 the first evaluation already steps them differently
        self.weights = self.initialize_weights()
        self.activation_func = [self.__sigmoid,self.__sigmoid]
        self.activation_func_prime = [self.__sigmoid_prime,self.__sigmoid_prime]
    def initialize_weights(self):
        _weights=[]
        _weights.append(self.rng.uniform(-1, 1, size=(3,2)) * self.settings["weight_scale"])
        _weights.append(self.rng.uniform(-1, 1, size=(3,1)) * self.settings["weight_scale"])
        
        return _weights
        
    def xor_net(self,X,weights=None):
        if isinstance(weights,type(None)):
            _weights=self.weights
        else:
            _weights=weights
        Z=[]
        for n in range(len(_weights)):
            _X=self.__append_bias(X)
            Z.append(np.dot(_weights[n].T,_X))
            y = self.activation_func[n](Z[-1])
            X=y
        return y, Z
        
    
    def mse(self,X,y,**kwargs):
        y_predicted,z=self.xor_net(X,**kwargs)
        err=np.sqrt(np.sum(np.power(y-y_predicted,2)))
        return err

    def grdmse(self,X,y,**kwargs):        
        y_predicted,Z=self.xor_net(X,**kwargs)  
        delta=[]
        grad=[]
        
        delta.insert(0,(y_predicted-y)*self.activation_func_prime[-1](Z[-1]))
        grad.insert(0,np.dot(self.__append_bias(self.activation_func[-1](Z[-2])),delta[0].T))

        delta.insert(0,(self.weights[-1][1:,:]*delta[-1])*self.activation_func_prime[-2](Z[-2]))
        grad.insert(0,np.dot(self.__append_bias(self.activation_func[-2](X)),delta[0].T))
        
        return grad
    
    def fit(self,**kwargs):
        X=np.array([[0,0,1,1],[0,1,0,1]])
        Y=np.array([0,1,1,0])
        err=np.zeros(self.epochs)
        missclassified=np.zeros(self.epochs)
        for t in np.arange(self.epochs):
            for i in range(len(Y)):
                x=X[:,[i]]
                y=Y[i]
                y_predicted,Z=self.xor_net(x,**kwargs)
                grad=self.grdmse(x,y)
                for n in range(len(self.weights)):
                    self.weights[n]-=self.learning_rate*grad[n]
            err[t]=self.mse(X,Y)
            missclassified[t]=np.count_nonzero(self.xor_net(X)[0]!=Y)
        return err, missclassified
    @staticmethod   
    def __append_bias(X) :
         return np.concatenate((np.ones((1,X.shape[1])),X))
     
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_prime(x):
        x = 1 / (1 + np.exp(-x))
        return x * (1. - x)

def get_data():
    import pandas as pd

    df_in = pd.read_csv('train_in.csv', header=None)
    train_in = df_in.to_numpy()

    df_out = pd.read_csv('train_out.csv', header=None)
    train_out = df_out.to_numpy()

    df_in = pd.read_csv('test_in.csv', header=None)
    test_in = df_in.to_numpy()

    df_out = pd.read_csv('test_out.csv', header=None)
    test_out = df_out.to_numpy()

    return train_in, train_out, test_in, test_out


if __name__ == '__main__':
    ### input will be [0,1 ; 0,1]
    ### output should be prediction either logits or one hot
    net=XorNet(); err,missclassified=net.fit()
    
    fig,ax=plt.subplots()
    ax2=ax.twinx()
    ax.plot(err,'r')
    ax.set_ylabel("Error",c='r')
    ax2.plot(missclassified,'b')
    ax2.set_ylabel("Missclassified items",c='b')