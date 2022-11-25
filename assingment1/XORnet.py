import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class XorNet(object):
    def __init__(self,
                 settings=None, weights=None, **kwargs):

        ### META
        self.rng = np.random.default_rng()

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 1e-2,
            "epochs": 50,
            "weight_scale": 1e-2
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = self.settings["epochs"]

        ### MAIN MCP
        # set initial weights, if they are initialized at w_i!=0 the first evaluation already steps them differently
        # allow `weights` to be passed directly or fall back to default i.e. lazy-mode
        self.weights = self.rng.uniform(-1, 1, size=(3,3)) * self.settings["weight_scale"]
        self.bias = np.zeros(shape=(2))
        self.activation_func = self.__sigmoid
        
    def __forward(self):
        ## TO-DO: fix weight indices
        
        z1 = np.dot(self.weights[:,:2], self.x)
        a1 = self.activation_func(z1)
        
        z2 = np.dot(self.weights[-1], a1)
        a2 = self.activation_func(z2)
        
        return (a1, a2)
    
    def __backward(self, a1, a2):
        ## TO-DO: fix weight indices
        dz2 = a2 - self.y
        dw2 = np.dot(dz2, a1.T)/self.m
        
        dz1 = np.dot(self.weights[-1].T,dz2) * a1*(1-a1)
        dw1 = np.dot(dz1,self.x.T)/self.m
        dW = np.append(dw1, dw2).reshape(self.weights.shape)
        return dW
    
    def __update(self, dW):
        self.weights -= self.learning_rate * dW
        return self
    
    def loss(self, pred):
        '''
        Compute the **loss function** to estimate how well the network is performing.
        The formula is the one for *binary cross-centropy*

        Parameters
        ----------
        prediction : float
            network output after each forward propagation. Last element  of the `A ` vector

        Returns
        -------
        value of loss function (float).

        '''
        return (-(1/self.m)*np.sum(self.y*np.log(pred)+(1-self.y)*np.log(1-pred)))

    def run_network(self, X):
        # input vector X is decomposed in (x, y)
        # where `x` are the input values and `y` the expected value
        self.x = self.__append_bias(X[0])
        self.y = X[1]
        self.m = self.x.shape[1]
        
        # save `loss` at each epoch
        self.loss_vec = np.zeros(self.epochs)
        
        # start the network:
            ## 1. Forward propagation (estimate output)
            ## 2. Compute Loss function value
            ## 3. Backward propagation (adjust weights)
        for e in range(self.epochs):
            a1, a2 = self.__forward()
            self.loss_vec[e] = self.loss(a2)
            dW = self.__backward(a1, a2)
            self.__update(dW)
        return self

        

    def mse(self):
        pass

    def grdmse(self):
        pass

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_prime(x):
        x = 1 / (1 + np.exp(-x))
        return x * (1. - x)
    
    @staticmethod
    def __append_bias(X):
        bias = np.ones(shape=(len(X), 1))
        return np.concatenate((X, bias), axis=1)
    
    def plot_loss(self, ax=None, **kwargs):
        '''plot loss function value vs. **epochs**'''
        ax = ax or plt.gca()
        ax.plot(np.arange(0, self.epochs), self.loss_vec, **kwargs)
        ax.set(xlabel='Epochs', ylabel='Loss function')
        return None


def XOR(x):
    return int(bool(x[0]) != bool(x[1]))


if __name__ == '__main__':
    ### input will be [0,1 ; 0,1]    
    # XOR inputs
    x=np.array([[0,0],[0,1]])
    # XOR outputs
    y = [XOR(i) for i in x]
    # input vector X
    X = [x, y]

    xor = XorNet()
    xor.epochs = int(20e3)
    xor.run_network(X)

    xor.plot_loss()


