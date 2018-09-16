
# coding: utf-8

# In[5]:


import pickle
import matplotlib.pyplot as plt


# In[6]:




pickle_in = open("hist_relu.pickle",'rb')
relu_hist = pickle.load(pickle_in)

pickle_in = open("hist_elu.pickle",'rb')
elu_hist = pickle.load(pickle_in)

pickle_in = open("hist_leakyrelu.pickle",'rb')
leaky_hist = pickle.load(pickle_in)


pickle_in = open("hist_leakyrelu.pickle",'rb')
leaky_hist = pickle.load(pickle_in)

pickle_in = open("hist_leakyrelu_batchnorm.pickle",'rb')
batch_leaky_hist = pickle.load(pickle_in)

pickle_in = open("hist_relu_batchnorm.pickle",'rb')
batch_relu_hist = pickle.load(pickle_in)


# In[7]:


plt.figure(figsize=(12,12))
plt.plot(relu_hist['loss'])
plt.plot(elu_hist['loss'])
plt.plot(leaky_hist['loss'])
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.legend(['relu','elu','leaky'],loc = 'upper left')
plt.savefig("train_loss")
plt.show()


# In[8]:


plt.figure(figsize=(12,12))
plt.plot(relu_hist['acc'])
plt.plot(elu_hist['acc'])
plt.plot(leaky_hist['acc'])
plt.xlabel('epoch')
plt.ylabel(' train accuracy')
plt.legend(['relu','elu','leaky'],loc = 'upper left')
plt.savefig("train_accuracy")
plt.show()


# In[9]:


plt.figure(figsize=(12,12))
plt.plot(relu_hist['val_acc'])
plt.plot(elu_hist['val_acc'])
plt.plot(leaky_hist['val_acc'])
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.legend(['relu','elu','leaky'],loc = 'upper left')
plt.savefig("test_accuracy")
plt.show()


# In[10]:


plt.figure(figsize=(12,12))
plt.plot(relu_hist['val_loss'])
plt.plot(elu_hist['val_loss'])
plt.plot(leaky_hist['val_loss'])
plt.xlabel('epoch')
plt.ylabel('test loss')
plt.legend(['relu','elu','leaky'],loc = 'upper left')
plt.savefig("test_loss")
plt.show()


# In[11]:


plt.figure(figsize=(12,12))
plt.plot(batch_relu_hist['loss'])
plt.plot(elu_hist['loss'])
plt.plot(batch_leaky_hist['loss'])
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.legend(['b_relu','elu','b_leaky'],loc = 'upper left')
plt.savefig("train_loss_batch")
plt.show()


# In[12]:


plt.figure(figsize=(12,12))
plt.plot(batch_relu_hist['acc'])
plt.plot(elu_hist['acc'])
plt.plot(batch_leaky_hist['acc'])
plt.xlabel('epoch')
plt.ylabel('train_accuracy')
plt.legend(['b_relu','elu','b_leaky'],loc = 'upper left')
plt.savefig("train_accuracy_batch")
plt.show()


# In[13]:


plt.figure(figsize=(12,12))
plt.plot(batch_relu_hist['val_acc'])
plt.plot(elu_hist['val_acc'])
plt.plot(batch_leaky_hist['val_acc'])
plt.xlabel('epoch')
plt.ylabel('val_accuracy')
plt.legend(['b_relu','elu','b_leaky'],loc = 'upper left')
plt.savefig("test_accuracy_batch")
plt.show()


# In[14]:


plt.figure(figsize=(12,12))
plt.plot(batch_relu_hist['val_loss'])
plt.plot(elu_hist['val_loss'])
plt.plot(batch_leaky_hist['val_loss'])
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.legend(['b_relu','elu','b_leaky'],loc = 'upper left')
plt.savefig("test_loss_batch")
plt.show()


# In[28]:





# In[ ]:




