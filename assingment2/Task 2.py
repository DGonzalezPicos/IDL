#!/usr/bin/env python
# coding: utf-8

# # Task 2: CNN for telling the time
# ---
# *Author: Lukas Welzel for Group 24 (?)*
# 
# ### Index:
# - [Imports](#imports)
# - [Main Content](#main-content) (xN)
#   - [Overview](#overview)
#   - [Loading Data](#loading-data)
#   - [Selecting Model](#selecting-model)
#   - [Experiments](#experiments)
# - [Citations](#citations)
# - [Footer](#footer)

# In[58]:


# imports
# use environment_TTT.yaml an related IDL kernel
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from TTTcnn import *
from default_CNNs import *
# making sure everything looks as it should
print("TF version: ", tf.__version__)
print(f"Available devices:")
[print("\t", t) for t in tf.config.list_physical_devices('GPU')]; # at least one GPU
# print(single_head_classification())


# ## Main Content
# ### Overview
# In TTTcnn.py we defined a CNN by sublassing Keras' Model class. This "TellTheTimeCNN" class will be the main focus of this task. The main changes from the base Model is the setup that is now encapsulated and can be interacted with using the "settings" argument. TTTcnn.py also contains some utility functions for data loading.
# 
# ### Loading Data

# In[3]:


x_train, base_y_train, x_test, base_y_test = get_data()


# ### Selecting Model
# Instances of TellTheTimeCNN have a full set of default settings, namely:
# 
# ---
# <blockquote>
#     <p>"learning_rate":       1e-2 </p>
#     <p>"epochs":              50 </p>
#     <p>"batch_size":          128 </p>
#     <p>"encoding":            "decimal" </p>
#     <p>"type":                ["regression"],  # classification, regression (can be sequence) </p>
#     <p>"actfn_normalization": ["tanh"],  # must be sequence if type is sequence </p>
#     <p>"loss":                ["mse_sincos"]  # must be sequence if type is sequence </p>
# </blockquote>
# 
# ---
# 
# As is obvious from the last few lines the model is set up to support multiple heads.

# In[4]:


# setting up a default model without a rng seed
default_model = TellTheTimeCNN()


# In[5]:


# the default model uses decimal encoding which we will now use
# encoding of data can be done via the encode_y function of any TellTheTimeCNN instance.
y_train, y_test = default_model.encode_y(base_y_train), default_model.encode_y(base_y_test)
try:
    print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", y_train.shape)
except AttributeError:
    print("Encoding from hh,mm -> f: ", base_y_train.shape, " -> ", len(y_train))


# In[7]:


# default_model.train(x_train, y_train)
# default_model.test(x_test, y_test)


# ### Experiments

# Since we had large issues with training stability for both classification and regression we implement an early stopping method and learning rate schedule to improe stability and not waste training time on "lost" runs where the losses explode. We found that, optimizer making use of momentum where needed for timely convergence, however we note that simple gradient descent optimizers might have improved the stability of the training. In practice however, the architecures that were tried did not find good optima with simple optimizers and failed to converge no matter the learning rate. To further improve staiblity we attempted to implement a learning rate scheduler in combination with a function to reduce the learning rate when the validation loss plateaus. Sadly, TF2.6.0 still contains a 2 year old bug that makes this combination impossible, see e.g. https://github.com/tensorflow/tensorflow/issues/41639.

# In[64]:


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
n_runs = 3
n_epochs = 50


# In[9]:


# Data visualization
fig, (ax, ax4, ax2, ax3) = plt.subplots(1, 4, constrained_layout=True, figsize=(24*0.6, 6*0.6))
ax.hist(y_train, bins=720, label="Y training (decimal)")
ax.hist(y_test, bins=720, label="Y testing (decimal)")
ax.set_xlabel("Decimal encoding (hh:mm -> (0,1] f)")
ax.set_ylabel("Counts")
ax.legend()
ax.set_title("Decimal encoding distribution (100 bins)")

hist = ax2.hist2d(base_y_train[:, 0], base_y_train[:, 1], vmin=0., bins=[12, 60], label="Y training (hh:mm)")
fig.colorbar(hist[3], ax=ax2,label="Counts")
ax2.set_xlabel("Hours")
ax2.set_ylabel("Minutes")
ax2.set_title("Training labels (hh:mm)")

hist = ax3.hist2d(base_y_test[:, 0], base_y_test[:, 1], vmin=0., bins=[12, 60], label="Y testing (hh:mm)")
fig.colorbar(hist[3], ax=ax3, label="Counts")
ax3.set_xlabel("Hours")
ax3.set_ylabel("Minutes")
ax3.set_title("Testing labels (hh:mm)")

ax4.hist(base_y_train[:, 1], bins=60, label="Y training (minutes)")
ax4.set_xlabel("Minutes")
ax4.set_ylabel("Counts")
ax4.legend()
ax4.set_title("Minutes distribution (60 bins)")

fig.suptitle("Learnin/testing labels distributions")

plt.show()


# In[91]:


# Classification
def prep_class_1head(settings, n_classes):
    class_head1_model = TellTheTimeCNN(settings=settings)
    y_train = class_head1_model.encode_y(base_y_train, n_classes=n_classes)
    y_test = class_head1_model.encode_y(base_y_test, n_classes=n_classes)
    return class_head1_model, y_train, y_test

def train_this(settings, n_runs=3, n_epochs=25, n_classes=72):
    data = []
    for i in range(n_runs):
        model, y_train, y_test = prep_class_1head(settings, n_classes)
        history = model.train(x_train, y_train, validation_data=(x_test, y_test),
                              epochs=n_epochs)
        # test = default_model.test(x_test, ytest)
        data.append(history.history)
    return data # , test


# In[80]:


def plot_history(hist):
    fig, (ax1) = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 6))

    ax1.set_title(f"Loss")
    for el in hist:
        ax1.plot(12. * 60. * np.array(el["loss"]), label="Training")
        ax1.plot(12. * 60. * np.array(el["val_loss"]), label="Validation")
        # ax1.scatter(len(el["loss"]) - 1, 12. * 60. * test['loss'], label="Test", c="black")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Error/Loss [minutes]")
        ax1.legend()
    
    plt.show()


# In[72]:


# SINGLE HEAD CLASSIFICATION MODELS
from default_CNNs import single_head_classification
from TTTcnn import *

# settings 72 classes, single head
settings = single_head_classification(72)
c72_result = train_this(settings)

# settings 720 classes, single head, you really should not test this
# settings = {**default_classification_settings, **{"n_classes": [720]}}
# class720_head1_model, y_train, y_test = prep_class_1head(settings, 720)


# In[92]:


# settings 24 classes, single head
settings = single_head_classification(24)
c24_result = train_this(settings, n_classes=24)


# In[93]:


# settings 12 classes, single head
settings = single_head_classification(12)
c12_result = train_this(settings, n_classes=12)


# In[94]:


def plot_history(hist, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))

    ax1.set_title(f"Loss")
    ax2.set_title(f"Accuracy")
    for el in hist:
        ax1.plot(np.array(el["loss"]), label="Training")
        ax1.plot(np.array(el["val_loss"]), label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Error/Loss")
        ax1.legend()
        
    
        ax2.plot(el["accuracy"], label="Training")
        ax2.plot(el["val_accuracy"], label="Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
    
    fig.suptitle(title)
    
    plt.show()

    
plot_history(c72_result, "72 classes")
plot_history(c24_result, "24 classes")
plot_history(c12_result, "12 classes")
# print(c72_result[0])
#print(c24_result) 
#print(c12_result)


# In[104]:


def plot_comb_history(hists, title=""):
    fig, (ax1) = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    
    colors = ["blue", "orange"]
    lss = ["solid", "dashed", "dotted"]
    labels = ["72", "24", "12"]

    ax1.set_title(f"Loss")
    ax2.set_title(f"Accuracy")
    for hist, ls, label in zip(hists, lss, labels):
        l = np.min([len(el["loss"]) for el in hist])
        losses = np.array([el["loss"][:l] for el in hist])
        accuracy = np.array([el["accuracy"][:l] for el in hist])
        
        val_losses = np.array([el["val_loss"][:l] for el in hist])
        val_accuracy = np.array([el["val_accuracy"][:l] for el in hist])
        
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        
        mean_val_loss = np.mean(val_losses, axis=0)
        std_val_loss = np.std(val_losses, axis=0)
        
        mean_accuracy = np.mean(losses, axis=0)
        std_accuracy = np.std(losses, axis=0)
        
        mean_val_accuracy = np.mean(val_losses, axis=0)
        std_val_accuracy = np.std(val_losses, axis=0)
        
        x = np.arange(len(mean_loss))
        ax1.plot(mean_loss, label=f"Training {label}", ls=ls, c="blue")
        ax1.fill_between(x,
                        mean_loss + std_loss,
                        mean_loss - std_loss, color="blue", alpha=0.2)
        ax1.plot(mean_val_loss, label=f"Validation {label}", ls=ls, c="orange")
        ax1.fill_between(x,
                mean_val_loss + std_val_loss,
                mean_val_loss - std_val_loss, color="orange", alpha=0.2)

        
    
        ax2.plot(mean_accuracy, label=f"Training {label}", ls=ls, c="blue")
        ax2.fill_between(x,
                mean_accuracy + std_accuracy,
                mean_accuracy - std_accuracy, color="blue", alpha=0.2)
        ax2.plot(mean_val_accuracy, label=f"Validation {label}", ls=ls, c="orange")
        ax2.fill_between(x,
                         mean_val_accuracy + std_val_accuracy,
                         mean_val_accuracy - std_val_accuracy, color="orange", alpha=0.2)
        
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error/Loss")
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    fig.suptitle(title)
    
    plt.show()

    
plot_comb_history([c72_result, c24_result, c12_result], "Single head classification")


# ### Regression
# We had significant issues with regression. Almost no architecture worked well, and those that worked still had large errors.

# In[98]:


# Regression
def prep_reg_1head(settings):
    model = TellTheTimeCNN(settings=settings)
    y_train = model.encode_y(base_y_train)
    y_test = model.encode_y(base_y_test)
    return model, y_train, y_test

def train_this(settings, n_runs=3, n_epochs=25):
    data = []
    for i in range(n_runs):
        model, y_train, y_test = prep_reg_1head(settings)
        history = model.train(x_train, y_train, validation_data=(x_test, y_test),
                              epochs=n_epochs)
        # test = default_model.test(x_test, ytest)
        data.append(history.history)
    return data # , test

from default_CNNs import single_large_head_regression, single_head_regression
from TTTcnn import *


# In[99]:


settings = single_large_head_regression()
r_dec_large_result = train_this(settings)


# In[100]:


settings = single_head_regression()
r_dec_result = train_this(settings)


# In[113]:


def plot_comb_reg_history(hists, title=""):
    fig, (ax1) = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    
    colors = ["blue", "orange"]
    lss = ["solid", "dashed", "dotted"]
    labels = ["[0.0, 12.0)", "[0.0, 1.0)"]

    # ax1.set_title(f"Loss")
    # ax2.set_title(f"Accuracy")
    for hist, ls, label in zip(hists, lss, labels):
        l = np.min([len(el["loss"]) for el in hist])
        losses = np.array([el["loss"][:l] for el in hist])
        # accuracy = np.array([el["accuracy"][:l] for el in hist])
        
        val_losses = np.array([el["val_loss"][:l] for el in hist])
        #val_accuracy = np.array([el["val_accuracy"][:l] for el in hist])
        
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        
        mean_val_loss = np.mean(val_losses, axis=0)
        std_val_loss = np.std(val_losses, axis=0)
        
        #mean_accuracy = np.mean(losses, axis=0)
        #std_accuracy = np.std(losses, axis=0)
        
        #mean_val_accuracy = np.mean(val_losses, axis=0)
        #std_val_accuracy = np.std(val_losses, axis=0)
        
        x = np.arange(len(mean_loss))
        ax1.plot(mean_loss, label=f"Training {label}", ls=ls, c="blue")
        ax1.fill_between(x,
                        mean_loss + std_loss,
                        mean_loss - std_loss, color="blue", alpha=0.2)
        ax1.plot(mean_val_loss, label=f"Validation {label}", ls=ls, c="orange")
        ax1.fill_between(x,
                mean_val_loss + std_val_loss,
                mean_val_loss - std_val_loss, color="orange", alpha=0.2)

        
    
#         ax2.plot(mean_accuracy, label=f"Training {label}", ls=ls, c="blue")
#         ax2.fill_between(x,
#                 mean_accuracy + std_accuracy,
#                 mean_accuracy - std_accuracy, color="blue", alpha=0.2)
#         ax2.plot(mean_val_accuracy, label=f"Validation {label}", ls=ls, c="orange")
#         ax2.fill_between(x,
#                          mean_val_accuracy + std_val_accuracy,
#                          mean_val_accuracy - std_val_accuracy, color="orange", alpha=0.2)
        
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error/Loss")
    ax1.legend()
    ax1.set_yscale("log")

#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Accuracy")
#     ax2.legend()
    
    fig.suptitle(title)
    
    plt.show()

    
plot_comb_reg_history([r_dec_large_result, r_dec_result], "Single head regression")


# In[ ]:





# In[117]:


# Multihead

# Regression
def prep_reg_1head(settings):
    model = TellTheTimeCNN(settings=settings)
    y_train = model.encode_y(base_y_train)
    y_test = model.encode_y(base_y_test)
    return model, y_train, y_test

def train_this(settings, n_runs=3, n_epochs=25):
    data = []
    for i in range(n_runs):
        model, y_train, y_test = prep_reg_1head(settings)
        history = model.train(x_train, y_train, validation_data=(x_test, y_test),
                              epochs=n_epochs)
        # test = default_model.test(x_test, ytest)
        data.append(history.history)
    return data # , test

from default_CNNs import big_double_head_regression
from TTTcnn import *


# In[118]:


settings = big_double_head_regression()
r_cossin_2head_result = train_this(settings)


# In[119]:


def plot_comb_reg_history(hists, title=""):
    fig, (ax1) = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    
    colors = ["blue", "orange"]
    lss = ["solid", "dashed", "dotted"]
    labels = ["[0.0, 12.0)", "[0.0, 1.0)", "multi-head (sin-cos)"]

    # ax1.set_title(f"Loss")
    # ax2.set_title(f"Accuracy")
    for hist, ls, label in zip(hists, lss, labels):
        l = np.min([len(el["loss"]) for el in hist])
        losses = np.array([el["loss"][:l] for el in hist])
        # accuracy = np.array([el["accuracy"][:l] for el in hist])
        
        val_losses = np.array([el["val_loss"][:l] for el in hist])
        #val_accuracy = np.array([el["val_accuracy"][:l] for el in hist])
        
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        
        mean_val_loss = np.mean(val_losses, axis=0)
        std_val_loss = np.std(val_losses, axis=0)
        
        #mean_accuracy = np.mean(losses, axis=0)
        #std_accuracy = np.std(losses, axis=0)
        
        #mean_val_accuracy = np.mean(val_losses, axis=0)
        #std_val_accuracy = np.std(val_losses, axis=0)
        
        x = np.arange(len(mean_loss))
        ax1.plot(mean_loss, label=f"Training {label}", ls=ls, c="blue")
        ax1.fill_between(x,
                        mean_loss + std_loss,
                        mean_loss - std_loss, color="blue", alpha=0.2)
        ax1.plot(mean_val_loss, label=f"Validation {label}", ls=ls, c="orange")
        ax1.fill_between(x,
                mean_val_loss + std_val_loss,
                mean_val_loss - std_val_loss, color="orange", alpha=0.2)

        
    
#         ax2.plot(mean_accuracy, label=f"Training {label}", ls=ls, c="blue")
#         ax2.fill_between(x,
#                 mean_accuracy + std_accuracy,
#                 mean_accuracy - std_accuracy, color="blue", alpha=0.2)
#         ax2.plot(mean_val_accuracy, label=f"Validation {label}", ls=ls, c="orange")
#         ax2.fill_between(x,
#                          mean_val_accuracy + std_val_accuracy,
#                          mean_val_accuracy - std_val_accuracy, color="orange", alpha=0.2)
        
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error/Loss")
    ax1.legend()
    ax1.set_yscale("log")

#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Accuracy")
#     ax2.legend()
    
    fig.suptitle(title)
    
    plt.show()

    
plot_comb_reg_history([r_dec_large_result, r_dec_result, r_cossin_2head_result], "Regression")


# In[ ]:




