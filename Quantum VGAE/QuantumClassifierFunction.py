#!/usr/bin/env python
# coding: utf-8

# In[3]:


from braket.circuits import Circuit
import numpy as np
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from braket.aws import AwsDevice
import random


# In[2]:


def quantumclassifier(x,p,device):
    crt = Circuit()
    for i in range(len(x)):
        crt.h(i)
    k=0
    for i in x:
        for j in range(len(i)):
       # val = random.uniform(np.pi/2,np.pi)
            theta =  2*p[j]* i[j]
            unitary_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
            crt.unitary(matrix = unitary_matrix,targets=[k])
        
        
        k +=1    
    i = 0
    j = 1

    while i<=len(x)-1:
    
        if i==j:
        
            i+=1
        if j<=len(x)-1:
        
            crt.cnot(j,i)
            i+=1
            j+=1
        
        else:
            j =0
        
    for i in range(len(x)):
        crt.cnot(i,len(x))
    c = crt.probability(target=[len(x)])
    print(crt)
    result = device.run(crt, shots=1000).result()
    state = result.values[0]
    label = 0
    if state[0]>state[1]:
        label = 0
    else:
        label = 1
    return


# In[4]:


x = [ [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0, 0 ] ]


# In[5]:


p = [2.4561690554930085,
 2.8776545409863896,
 1.9523369867805616,
 1.9015418386858185,
 2.1884877038557176,
 2.591965850468072,
 1.9551279050960018]


# In[6]:


# set up the on-demand simulator SV1
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")


# In[7]:


label = quantumclassifier(x,p,device)


# In[8]:


label


# In[ ]:




