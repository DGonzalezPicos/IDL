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
            "learning_rate": 1,
            "epochs": 5000,
            "weight_scale": 1
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = self.settings["epochs"]
        
        #Fit results
        self.epochs_needed = np.inf
        self.has_converged = False
        self.errors=None
        self.missclassified_items=None

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
            missclassified[t]=np.count_nonzero(np.round(self.xor_net(X)[0])!=Y)
            
            if (not self.has_converged) & (missclassified[t]==0):
                self.has_converged=True
                self.epochs_needed=t
        self.errors=err
        self.missclassified_items=missclassified
    
    def plot_fit(self):
        fig,ax=plt.subplots()
        
        if self.has_converged:
            ax.set_title('Convereged in {} epochs'.format(self.epochs_needed))
        else:
            ax.set_title('Did not converge in {} epochs'.format(self.epochs))
            
        ax2=ax.twinx()
        ax.plot(self.errors,'r')
        ax.set_ylabel("Error",c='r')
        ax2.plot(self.missclassified_items,'b')
        ax2.set_ylabel("Missclassified items",c='b')
        
        return fig, ax, ax2
        
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
    #Default network:
    net=XorNet()
    net.fit()
    net.plot_fit()
    
    #Different learing rates 
    net10=XorNet(settings={'epochs':10000,'learning_rate':10})
    net10.fit()
    net10.plot_fit()
    
    net01=XorNet(settings={'epochs':10000,'learning_rate':0.1})
    net01.fit()
    net01.plot_fit()
    
    #Different activation functions
    tanh=lambda x: np.tanh(x)
    tanh_prime=lambda x: 1-np.power(np.tanh(x),2)
    
    relu=lambda x: np.maximum(0,x)
    relu_prime=lambda x: np.heaviside(x,1)
    
    net_tan=XorNet(settings={'epochs':50000,'learning_rate':0.1}) #NEED LESS L.R.!
    net_tan.activation_func = [tanh,tanh]
    net_tan.activation_func_prime = [tanh_prime,tanh_prime]
    net_tan.fit()
    net_tan.plot_fit()
    
    net_relu=XorNet(settings={'epochs':10000,'learning_rate':0.1}) #NEED LESS L.R.!
    net_relu.activation_func = [tanh,tanh]
    net_relu.activation_func_prime = [tanh_prime,tanh_prime]
    net_relu.fit()
    net_relu.plot_fit()
    
    #Random weights
    n_experiments=50
    max_tries=10000
    
    tries_needed=np.ones(n_experiments)*np.NAN
    
    X=np.array([[0,0,1,1],[0,1,0,1]])
    Y=np.array([0,1,1,0])
    #A posteriori, known the order of magintude of the correct weights
    for n in range(n_experiments):
        for i in range(max_tries):
            net=XorNet({"weight_scale": 20})
            if np.all(np.round(net.xor_net(X)[0])==Y):
                tries_needed[n]=i+1
                break
    print('Random tries average: {:.0f}'.format(np.mean(tries_needed)))
    print('Random tries deviation: {:.0f}'.format(np.std(tries_needed)/len(tries_needed)))
            
            
            