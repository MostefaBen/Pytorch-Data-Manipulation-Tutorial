```python
# import Libraries
import torch
import numpy as np
```


```python
# creating a tensor (as a vector) prepopulated with values
X_data = torch.arange(10, dtype=torch.float32)
X_data
```




    tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])




```python
# total number of element in a tensor
X_data.numel()
```




    10




```python
# shape of each axis
X_data.shape
```




    torch.Size([10])




```python
# changing the shape of a tensor
X_data = X_data.reshape(2, 5)
X_data
```




    tensor([[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]])




```python
# shape of each axis
X_data.shape
```




    torch.Size([2, 5])




```python
# creating a tensor with the same shape
Z = torch.zeros_like(X_data)
Z
```




    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])




```python
# creating a tensor of zeros
A_data = torch.zeros((3, 4, 5))
A_data
```




    tensor([[[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]])




```python
# creating a tensor of ones
B_data = torch.ones((3, 4, 5))
B_data
```




    tensor([[[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]],
    
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]]])




```python
#  creating a tensor prepopulated with element from a standard Gaussian (normal) distribution (i.e, Mean 0 and Std 1)
C_data = torch.randn(4, 4)
C_data
```




    tensor([[-0.1987, -0.4383,  1.5985,  0.2409],
            [ 1.3931,  0.4878,  0.8661,  0.1273],
            [-0.7192, -0.3716, -0.0618,  1.2327],
            [-1.1084, -0.0987,  0.5090, -0.0410]])




```python
# Python matrix to tensor 
D_data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
D_data
```




    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])




```python
# Numpy matrix to tensor
```


```python
D_data = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
D_data
```




    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=torch.int32)




```python
# Access tensor elements by indexing
# the last row
D_data[-1]
```




    tensor([7, 8, 9], dtype=torch.int32)




```python
# Access tensor elements by indexing
# the last  column
D_data[:, -1]
```




    tensor([3, 6, 9], dtype=torch.int32)




```python
# changing element of a tensor
D_data[1,2] = 20
D_data
```




    tensor([[ 1,  2,  3],
            [ 4,  5, 20],
            [ 7,  8,  9]], dtype=torch.int32)




```python
# changing multiple elements of a tensor
D_data[2:,:] = 30
D_data
```




    tensor([[ 1,  2,  3],
            [ 4,  5, 20],
            [30, 30, 30]], dtype=torch.int32)




```python
# Operations on tensor(+, *, /, -, **)
X = torch.tensor([1, 2, 3, 4])
Y = torch.tensor([5, 6, 7, 8])
X+Y, X*Y, X/Y, X-Y, X**Y
```




    (tensor([ 6,  8, 10, 12]),
     tensor([ 5, 12, 21, 32]),
     tensor([0.2000, 0.3333, 0.4286, 0.5000]),
     tensor([-4, -4, -4, -4]),
     tensor([    1,    64,  2187, 65536]))




```python
# or any math elementwise operations
torch.exp(X_data)
```




    tensor([[1.0000e+00, 2.7183e+00, 7.3891e+00, 2.0086e+01, 5.4598e+01],
            [1.4841e+02, 4.0343e+02, 1.0966e+03, 2.9810e+03, 8.1031e+03]])




```python
# tensor concatenation
X = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
Y = torch.tensor([[1, 0, 3], [4, 5, 7], [7, 8, 9]])
torch.cat([X, Y], dim=0), torch.cat([X, Y], dim=1)
```




    (tensor([[1., 2., 3.],
             [4., 5., 6.],
             [7., 8., 9.],
             [1., 0., 3.],
             [4., 5., 7.],
             [7., 8., 9.]]),
     tensor([[1., 2., 3., 1., 0., 3.],
             [4., 5., 6., 4., 5., 7.],
             [7., 8., 9., 7., 8., 9.]]))




```python
# constructing a tensor with logical statements
X == Y
```




    tensor([[ True, False,  True],
            [ True,  True, False],
            [ True,  True,  True]])




```python
# Summing all tensor's elements
X, X.sum()
```




    (tensor([[1., 2., 3.],
             [4., 5., 6.],
             [7., 8., 9.]]),
     tensor(45.))




```python
# broadcasting mechanism on tensors
C = torch.arange(6).reshape((6, 1))
D = torch.arange(4).reshape((1, 4))
C, D
```




    (tensor([[0],
             [1],
             [2],
             [3],
             [4],
             [5]]),
     tensor([[0, 1, 2, 3]]))




```python
C + D, (C + D).shape
```




    (tensor([[0, 1, 2, 3],
             [1, 2, 3, 4],
             [2, 3, 4, 5],
             [3, 4, 5, 6],
             [4, 5, 6, 7],
             [5, 6, 7, 8]]),
     torch.Size([6, 4]))




```python
# Saving memory by asigning to the same memory location instead of refrencing to a new place
before  = id(X)
print("before id(X): ", id(X))
X = X + Y
id(X) == before
print("After id(X): ", id(X))
```

    before id(X):  198044864
    After id(X):  197988000
    


```python
# performing in place operations to save memory
before  = id(X)
print("before id(X): ", id(X))
X[:] = X + Y
id(X) == before
print("After id(X): ", id(X))
```

    before id(X):  197988000
    After id(X):  197988000
    


```python
# Converting to a Numpy tensor, or vice versa
A = X.numpy()             # from tensor to numpy 
print(type(X), type(A))   
B = torch.tensor(A)       # from numpy to tensor
print(type(B))
```

    <class 'torch.Tensor'> <class 'numpy.ndarray'>
    <class 'torch.Tensor'>
    


```python
# from tensor to Python scalar
# only one element tensors can be converted to Python scalars !
F = torch.tensor([84.1256])
F, F.item(), float(F), int(F)
```




    (tensor([84.1256]), 84.12560272216797, 84.12560272216797, 84)


