#!/usr/bin/env python
# coding: utf-8

# In[117]:


from braket.circuits import Circuit
import numpy as np
from braket.aws import AwsDevice
import random


# In[71]:


# set up the on-demand simulator SV1
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")


# In[72]:


x = [ [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0, 0 ] ]


# In[73]:


def MC(x,q,p):
    crt = Circuit()
    for i in range(len(x)):
        theta = np.pi*p[i]*x[i]
        unitary_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
       # unitary_gate = UnitaryGate(unitary_matrix)
        crt.unitary(matrix = unitary_matrix,targets = [0])
    result = device.run(crt, shots=1000).result()
    unitary = crt.as_unitary() 
    #print(crt)
    return unitary


# In[96]:


p1 = []
Loss = []
for i in range(7):
    p1 .append(random.random())                                              
                          


# In[98]:


p1


# In[99]:


count = 0
uni = []
j = 1
for i in range(len(x)):
    unitary = (MC(x[i],count,p1))
    unitary = unitary.flatten()
    uni.append(unitary)
#    count+=1


# In[93]:


Loss


# In[116]:


Lmin = []
Loss = []
for m in range(10):
    for i in range(len(x)):
        L = 0 
        for j in range(len(x)):
            if i !=j:
                D = np.arccos(np.inner(uni[i],uni[j]))
                d = np.dot(x[i],x[j])
                L +=np.abs(D-d)
                #Loss.append(L)
        Lmin.append(L)
        Loss.append(min(Lmin))
min(Loss)


# In[82]:


Loss = []


# In[87]:


Loss.append(min(Lmin))


# In[88]:


Loss


# In[68]:


#finding the best set of parameters
index = Lmin.index(min(Lmin))
Pt = p1[index]


# In[69]:


Pt


# In[78]:


len(x)


# In[ ]:


#Old data mapper
def MC(x,q,p,crt):
    for i in range(len(x)):
        theta = 2*p[i]*x[i]
        unitary_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
       # unitary_gate = UnitaryGate(unitary_matrix)
        crt.unitary(matrix = unitary_matrix,targets = [q])
        


# In[ ]:


#Finding rho0 and rho1


# In[ ]:


def crtcreation()


# In[124]:


crt = Circuit()


# In[125]:


for i in range(7):
    theta = 2
    unitary_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
       # unitary_gate = UnitaryGate(unitary_matrix)
    crt.unitary(matrix = unitary_matrix,targets = [0])
    


# In[126]:


print(crt)


# In[ ]:




