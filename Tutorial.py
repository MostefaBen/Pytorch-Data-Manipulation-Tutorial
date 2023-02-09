#!/usr/bin/env python
# coding: utf-8

# In[16]:


# import Libraries
import torch
import numpy as np


# In[76]:


# creating a tensor (as a vector) prepopulated with values
X_data = torch.arange(10, dtype=torch.float32)
X_data


# In[4]:


# total number of element in a tensor
X_data.numel()


# In[5]:


# shape of each axis
X_data.shape


# In[78]:


# changing the shape of a tensor
X_data = X_data.reshape(2, 5)
X_data


# In[74]:


# shape of each axis
X_data.shape


# In[79]:


# creating a tensor with the same shape
Z = torch.zeros_like(X_data)
Z


# In[10]:


# creating a tensor of zeros
A_data = torch.zeros((3, 4, 5))
A_data


# In[11]:


# creating a tensor of ones
B_data = torch.ones((3, 4, 5))
B_data


# In[13]:


#  creating a tensor prepopulated with element from a standard Gaussian (normal) distribution (i.e, Mean 0 and Std 1)
C_data = torch.randn(4, 4)
C_data


# In[14]:


# Python matrix to tensor 
D_data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
D_data


# In[ ]:


# Numpy matrix to tensor


# In[32]:


D_data = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
D_data


# In[18]:


# Access tensor elements by indexing
# the last row
D_data[-1]


# In[19]:


# Access tensor elements by indexing
# the last  column
D_data[:, -1]


# In[21]:


# changing element of a tensor
D_data[1,2] = 20
D_data


# In[22]:


# changing multiple elements of a tensor
D_data[2:,:] = 30
D_data


# In[29]:


# Operations on tensor(+, *, /, -, **)
X = torch.tensor([1, 2, 3, 4])
Y = torch.tensor([5, 6, 7, 8])
X+Y, X*Y, X/Y, X-Y, X**Y


# In[27]:


# or any math elementwise operations
torch.exp(X_data)


# In[82]:


# tensor concatenation
X = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
Y = torch.tensor([[1, 0, 3], [4, 5, 7], [7, 8, 9]])
torch.cat([X, Y], dim=0), torch.cat([X, Y], dim=1)


# In[64]:


# constructing a tensor with logical statements
X == Y


# In[66]:


# Summing all tensor's elements
X, X.sum()


# In[67]:


# broadcasting mechanism on tensors
C = torch.arange(6).reshape((6, 1))
D = torch.arange(4).reshape((1, 4))
C, D


# In[69]:


C + D, (C + D).shape


# In[83]:


# Saving memory by asigning to the same memory location instead of refrencing to a new place
before  = id(X)
print("before id(X): ", id(X))
X = X + Y
id(X) == before
print("After id(X): ", id(X))


# In[84]:


# performing in place operations to save memory
before  = id(X)
print("before id(X): ", id(X))
X[:] = X + Y
id(X) == before
print("After id(X): ", id(X))


# In[97]:


# Converting to a Numpy tensor, or vice versa
A = X.numpy()             # from tensor to numpy 
print(type(X), type(A))   
B = torch.tensor(A)       # from numpy to tensor
print(type(B))


# In[108]:


# from tensor to Python scalar
# only one element tensors can be converted to Python scalars !
F = torch.tensor([84.1256])
F, F.item(), float(F), int(F)

