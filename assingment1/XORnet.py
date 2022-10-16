import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class XorNet(object):
    def __init__(self,
                 settings=None,a_fun_and_der=False, **kwargs):

        ### META
        self.rng = np.random.default_rng(**kwargs)

        ### DEFAULT VALUES
        default_settings = {
            "learning_rate": 1,
            "epochs": 5000,
            "weight_scale": 1,
            "weight_deviation": 1,
            "weight_initialization": 'uniform'
        }
        if isinstance(settings, dict):
            self.settings = {**default_settings, **settings}
        else:
            self.settings = default_settings

        self.learning_rate = self.settings["learning_rate"]
        self.epochs = self.settings["epochs"]
        
        #Fit results
        self.epochs_needed = np.nan
        self.has_converged = False
        self.errors=None
        self.missclassified_items=None

        ### MAIN MCP
        # set initial weights, if they are initialized at w_i!=0 the first evaluation already steps them differently
        self.weights = self.initialize_weights()
        if type(a_fun_and_der)==tuple:
            self.activation_func=a_fun_and_der[0]
            self.activation_func_prime=a_fun_and_der[1]
        else:
            self.activation_func = [self.__sigmoid,self.__sigmoid]
            self.activation_func_prime = [self.__sigmoid_prime,self.__sigmoid_prime]
            
    def initialize_weights(self):
        w_i=self.settings['weight_initialization']
        if w_i == 'uniform':
            initializer=lambda s: self.rng.uniform(-1, 1, size=s)
        elif w_i == 'normal':
            initializer=lambda s: self.rng.normal(loc=0,scale=self.settings['weight_deviation'],size=s)
        elif w_i == 'constant':
            initializer=lambda s: np.ones(s)  
        elif w_i == 'glorot':
            def initializer(s):
                fan_in=s[0]*s[1]
                fan_out=s[1]
                return self.rng.normal(loc=0,scale=np.sqrt(2/(fan_in+fan_out)),size=s)
        else:
            raise KeyError(w_i)
        _weights=[]
        _weights.append(initializer((3,2)) * self.settings["weight_scale"])
        _weights.append(initializer((3,1)) * self.settings["weight_scale"])
        
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
    
    def fit(self,stop_when_converge=False,**kwargs):
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
                if stop_when_converge:
                    break
        self.errors=err
        self.missclassified_items=missclassified
        
    
    def plot_fit(self,aditional_title=None):
        
        fig,ax=plt.subplots()
        
        ax2=ax.twinx()
        
        s='$\eta$={:.2f}\tinitialization={}'.format(self.learning_rate,self.settings['weight_initialization'])
        if type(aditional_title)==str:
            s=aditional_title+'\t'+s
        fig.suptitle(s)
        
        if self.has_converged:
            ax.set_title('Convereged in {} epochs'.format(self.epochs_needed))
        else:
            ax.set_title('Did not converge in {} epochs'.format(self.epochs))  
        
        ax.plot(self.errors,'r')
        ax.set_ylabel("Error",c='r')
        ax.set_ylim(bottom=0 )
        ax2.plot(self.missclassified_items,'b')
        ax2.set_ylabel("Missclassified items",c='b')
        ax2.set_yticks([0,1,2,3,4])
        ax2.set_ylim([-0.2,4.2])
        
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
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

def params_survey(n_grid=10, n_samples=25, max_epochs=1000):

    # gridsearch
    l = np.geomspace(1e-1, 1e1, n_grid)
    s = np.geomspace(1e-1, 1e1, n_grid)
    ll, ss = np.meshgrid(l, s)

    mean_epochs = np.zeros(ll.size)
    std_epochs = np.zeros(ll.size)
    not_converged = np.zeros(ll.size)

    for i, (lr, w_scale) in tqdm(enumerate(zip(ll.flatten(), ss.flatten())), leave=False):
        settings = {
            "learning_rate": lr,
            "weight_scale": w_scale,
            "epochs":max_epochs
        }
        
        epochs = np.full(n_samples,np.nan)
        for j in tqdm(range(n_samples), leave=False):
            net = XorNet(settings=settings)
            net.fit(stop_when_converge=True)

            epochs[j]=net.epochs_needed
        
        mean_epochs[i]=np.nanmean(epochs)
        std_epochs[i]=np.nanstd(epochs)
        not_converged[i]=np.count_nonzero(np.isnan(epochs))/n_samples
        
    df_mean = pd.DataFrame(data=mean_epochs.reshape(ll.shape), columns=l, index=s)
    df_mean.to_csv("XOR_net_mean.csv")
    df_std = pd.DataFrame(data=std_epochs.reshape(ll.shape), columns=l, index=s)
    df_std.to_csv("XOR_net_std.csv")
    df_not_converged = pd.DataFrame(data=not_converged.reshape(ll.shape), columns=l, index=s)
    df_not_converged.to_csv("XOR_net_not_converged.csv")

    plot_results(settings['epochs'])
    
    
def plot_results(n_epochs):
    df_mean = pd.read_csv("XOR_net_mean.csv",index_col=0)
    
    fig,ax=plt.subplots()
    ax.set_facecolor('black')
    ax.set_title('Average epochs for convergence')
    sns.heatmap(df_mean,annot=True,fmt='g',cmap='Blues')
    plt.show()
    
    df_std = pd.read_csv("XOR_net_std.csv",index_col=0)
    
    fig,ax=plt.subplots()
    ax.set_facecolor('black')
    ax.set_title('std in convergence epochs')
    sns.heatmap(df_std,annot=True,fmt='g',cmap='Blues')
    plt.show()
    
    df_nc = pd.read_csv("XOR_net_not_converged.csv",index_col=0)

    fig,ax=plt.subplots()
    ax.set_title('Samples ratio not converged in less than '+str(n_epochs)+' epochs')
    sns.heatmap(df_nc,annot=True,fmt='.0%',cbar=False,cmap='Blues')

def XorNet_survey(n_samples=5,description=False,**kwargs):
    epochs = np.zeros(n_samples)
    not_converged = np.zeros(n_samples)
    for i in tqdm(range(n_samples), leave=False):
        net=XorNet(**kwargs)
        net.fit(stop_when_converge=True)

        epochs[i]=net.epochs_needed
    mean_epochs=np.nanmean(epochs)
    std_epochs=np.nanstd(epochs)
    not_converged=np.count_nonzero(np.isnan(epochs))/n_samples
    
    print('---------------------------')
    if description:
        print(description)
    s='l_rate={:.2f}\tinitialization={}'.format(net.learning_rate,net.settings['weight_initialization'])
    print(s)
    print('Epochs: {:.0f}\tstd: {:0.0f}\nNot converged in epochs<{:.0f}: {:.0%}'
          .format(mean_epochs,std_epochs,net.settings['epochs'],not_converged))
    return mean_epochs,std_epochs,not_converged

if __name__ == '__main__':
    
    #Sample the default network:
    params_survey(n_grid=3, n_samples=5,max_epochs=500000)  
    
    #Default network:
    net=XorNet(seed=2022)
    net.fit()
    net.plot_fit()
    
    #Different initialization funcitons
    net=XorNet_survey(n_samples=50,settings={'weight_initialization':'normal'})
    
    net=XorNet_survey(n_samples=50,settings={'weight_initialization':'glorot'})
    
    net=XorNet_survey(n_samples=50,settings={'weight_initialization':'normal','weight_deviation':10})
    
    net=XorNet_survey(n_samples=50,settings={'weight_initialization':'constant','epochs':10000})
    
    #Different learing rates 
    net10=XorNet_survey(n_samples=50,settings={'epochs':10000,'learning_rate':10})
    
    net01=XorNet_survey(n_samples=50,settings={'epochs':10000,'learning_rate':0.1})
    
    #Different activation functions
    tanh=lambda x: np.tanh(x)
    tanh_prime=lambda x: 1-np.power(np.tanh(x),2)
    
    relu=lambda x: np.maximum(0,x)
    relu_prime=lambda x: np.heaviside(x,1)
    
    
    net_tanh=XorNet_survey(n_samples=50, description='Tanh activation function',
                          settings={'epochs':10000,'learning_rate':0.1},
                          a_fun_and_der=([tanh,tanh],[tanh_prime,tanh_prime]))
    
    net_relu=XorNet_survey(n_samples=50, description='ReLU activation function',
                          settings={'epochs':10000,'learning_rate':0.1},
                          a_fun_and_der=([relu,relu],[relu_prime,relu_prime]))
    
    net_hybrid=XorNet_survey(n_samples=50, description='ReLU and sigmoid activation functions',
                          settings={'epochs':10000,'learning_rate':0.1},
                          a_fun_and_der=([relu,sigmoid],[relu_prime,sigmoid_prime]))
    net_hybrid=XorNet_survey(n_samples=50,description='ReLU and sigmoid activation functions',
                          settings={'epochs':10000,'learning_rate':1},
                          a_fun_and_der=([relu,sigmoid],[relu_prime,sigmoid_prime]))
    #Random weights
    n_experiments=1000
    max_tries=10000
    
    tries_needed=np.ones(n_experiments)*np.nan
    errors=np.ones(n_experiments)*np.nan
    
    X=np.array([[0,0,1,1],[0,1,0,1]])
    Y=np.array([0,1,1,0])
    #A posteriori, known the order of magintude of the correct weights
    for n in tqdm(range(n_experiments),leave=False):
        for i in range(max_tries):
            net=XorNet({"weight_scale": 20})
            if np.all(np.round(net.xor_net(X)[0])==Y):
                tries_needed[n]=i+1
                errors[n]=net.mse(X, Y)
                break
    print('Random tries average: {:.0f}'.format(np.nanmean(tries_needed)))
    print('MSE average: {:.4f}'.format(np.nanmean(errors)))
    print('Random tries deviation: {:.0f}'.format(np.nanstd(tries_needed)))
    print('Times not found in {} tries: {:.0f}%'
          .format(max_tries,np.count_nonzero(np.isnan(tries_needed))/n_experiments*100))           
            
            