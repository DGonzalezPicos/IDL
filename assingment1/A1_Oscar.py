import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Read test datasets

df_in=pd.read_csv('train_in.csv',header=None)
train_in=df_in.to_numpy()

df_out=pd.read_csv('train_out.csv',header=None)
train_out=df_out.to_numpy()

def show_data(data,index):
    #Reshape the image and plot it
    img=data[index,:].reshape((16,16))
    plt.imshow(img,cmap='gray')

#Image test    
#show_data(train_in,80)

#Rearrange data depending on 'train_out' value
data=[None for i in range(10)]
for n in range(10):
    data[n]=train_in[np.where(train_out[:,0]==n)[0],:]
    
#show_data(data[9],9)

#Center of each class
centers=[data[n].mean(axis=0) for n in range(10)]


#Read test datasets
df_in=pd.read_csv('test_in.csv',header=None)
test_in=df_in.to_numpy()

df_out=pd.read_csv('test_out.csv',header=None)
test_out=df_out.to_numpy()

#Euclidean n-dimensional distance
def dist(a,b):
    return np.sqrt(np.sum((a-b)**2,axis=1))


#Test the result of the algorithm

fails=np.array([0 for i in range(10)]) #List of the number of misclassified items
aims=np.array([0 for i in range(10)]) #List of the number of correctly classified items

centers_array=np.array(centers) #Build array from list to ease iteration
fails_map=np.zeros((10,10)) #10x10 map of expected value vs classified value

def classifier(data):
    #Defines the algorithm that classifies the data 
    return np.argmin(dist(centers_array,data))

#Test the algorithm and store the results in the corresponding variables
for i in range(test_in.shape[0]):
    n_pred=classifier(test_in[i,:])
    n_real=test_out[i,0]
    
    if n_pred==n_real:
        aims[n_real]+=1
    else:
        fails[n_real]+=1
        fails_map[n_real,n_pred]+=1

#Calculate the number of correct classifications related to the total number of items
accuracy=aims/(aims+fails)

#Show the results
print('n (real) \t Precission')
for i in range(accuracy.shape[0]):
    print('{:}\t\t\t{:.0f}%'.format(i,accuracy[i]*100))

