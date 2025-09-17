#!/usr/bin/env python
# coding: utf-8

# In[499]:


from braket.circuits import Circuit
import numpy as np
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from braket.aws import AwsDevice
import random


# In[500]:


crt = Circuit()


# In[501]:


x = [ [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 1, 0, 0, 0, 0, 0, 0 ], [ 0, 1, 0, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0, 0 ], [ 0, 0, 1, 0, 0, 0, 0 ] ]


# In[502]:


p =[2.4221162432813474,
 2.604542405845682,
 2.134080639199246,
 2.3273229395469537,
 2.7858851749280262,
 2.3401995813971266,
 2.007709970097028]


# In[503]:


for i in range(len(x)):
    crt.h(i)


# In[504]:


k=0
for i in x:
    for j in range(len(i)):
       # val = random.uniform(np.pi/2,np.pi)
        theta =  2*p[j]* i[j]
        unitary_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        crt.unitary(matrix = unitary_matrix,targets=[k])
        
        
    k +=1    
    


# In[505]:


print(crt)


# In[506]:


from braket.circuits import Observable


# In[507]:


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


# In[508]:


print(crt)


# In[509]:


for i in range(len(x)):
    crt.cnot(i,len(x))
    


# In[512]:


theta = np.pi
crt.rx(len(x),theta)

c = crt.probability(target=[len(x)])


# In[513]:


print(crt)


# In[492]:


# set up the on-demand simulator SV1
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")


# In[493]:


result = device.run(crt, shots=1000).result()


# In[494]:


state = result.values[0]


# In[498]:


state[0]


# In[496]:


if state[0]>state[1]:
    label = 0
else:
    label = 1


# In[497]:


label


# In[ ]:




