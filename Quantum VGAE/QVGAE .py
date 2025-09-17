#!/usr/bin/env python
# coding: utf-8

# In[33]:


from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
import pandas as pd
import numpy as np
from braket.circuits import Observable


# In[17]:


device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")


# In[3]:


device = LocalSimulator()


# In[3]:


import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA

# Fetch data from yfinance
tickers = ['AAPL']
start_date = '2023-12-01'
end_date = '2024-01-01'

data = yf.download(tickers, start=start_date, end=end_date,interval = '1d')
# The data will have a multi-level column structure with 'Adj Close', 'Close', etc.

# Flatten the multi-level columns
data.columns = ['_'.join(col).strip() for col in data.columns.values]

# Drop any columns with NaN values (optional)
data.dropna(axis=1, inplace=True)

# Initialize PCA
n_components = 2  # Number of principal components
pca = PCA(n_components=n_components)

# Fit and transform the data
principal_components = pca.fit_transform(data)

# Create a DataFrame with the principal components
df_pca = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(n_components)], index=data.index)

# Output the resulting DataFrame with principal components
#print(df_pca.head())

# Convert DataFrame to list of lists
p = df_pca.values.tolist()

# Output the list of principal components
#print(principal_components_list)


# In[4]:


tickers = ['AAPL']
start_date = '2023-12-01'
end_date = '2024-01-01'

data = yf.download(tickers, start=start_date, end=end_date,interval = '1d')
# The data will have a multi-level column structure with 'Adj Close', 'Close', etc.

# Extract the closing prices
closing_prices = data['Close']

# Convert the DataFrame of closing prices to a list of lists
adj = closing_prices.values.tolist()


# In[5]:


target_date = '2024-01-03'  # Replace with your desired date

# Fetch historical data for the target date
d = yf.download(tickers, start='2024-01-02', end=target_date, interval='1d')

# Extract the closing price for the target date
closing_price = data['Close'].iloc[0]


# In[6]:


#Circuit creation
def design_cir(theta, X, A):
    P = theta
    crt = Circuit()
    unitary_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    for i in range(len(X)):
        crt.h(i)
        crt.unitary(matrix = unitary_matrix,targets = [i])
    #Rotations
    for i in range(len(X)):
        theta = X[i]*P
        for j in range(len(X[0])):
            crt.ry(i,theta[j])
    #Entanglement
    for i in range(len(X)):
        for j in range(len(X)):
            if i!=j:
                crt.cnot(i,j)
    #Observables
    for i in range(len(X)):
        crt.probability(target = i)
        crt.expectation(observable=Observable.Z(), target=i)
        crt.variance(observable=Observable.Z(), target=i)
    return crt    


# In[12]:


#Circuit execution
def exec_cir(crt,A):
    result = device.run(crt,shots = 10000).result()
    mean = []
    cov_mat = []
    i = 1
    j = 2
    while i< len(result.values):
        mean.append(result.values[i])
        cov_mat.append(result.values[j])
        i+=3
        j+=3
    #print(mean, cov_mat)
    mean = np.array(mean)
    cov_mat = np.array(cov_mat)
    variance = cov_mat
    sigma = np.zeros(shape =(len(p),len(p)))
    np.fill_diagonal(sigma,cov_mat)
    #cov_mat = [[cov_mat[0],0,0,0,0],[0,cov_mat[1],0,0,0],[0,0,cov_mat[2],0,0],[0,0,0,cov_mat[3],0],[0,0,0,0,cov_mat[4]]]
    cov_mat = np.array(sigma)
    #Sampling
    Z=[]
    for i in range(len(A)):
        ep = np.random.normal(0,1)
        cov_mat = cov_mat*ep
        Z.append(np.random.multivariate_normal(mean, cov_mat,size =1))
    Z = np.array(Z)
    Z = Z.reshape(len(p),len(p))
        
        
    return Z, mean, variance


# In[13]:


#decoder
import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to calculate p(A | Z)
def p_A_given_Z(A, Z,epsilon=1e-7):
    N = A.shape[0]
    log_prob = 0
    auc = []
    auc1= []
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                z_i = Z[i, :]
                z_j = Z[j, :]
                dot_product = np.dot(z_i, z_j)
                prob = sigmoid(dot_product)
                log_prob += np.log(prob+epsilon)
                auc.append(prob)
            else:
                z_i = Z[i, :]
                z_j = Z[j, :]
                dot_product = np.dot(z_i, z_j)
                prob =sigmoid(dot_product)
                log_prob += np.log(1 - prob+epsilon)
                auc.append(1-prob)
        #auc1.append(auc)        
    print("Probability: ",prob)
    return log_prob, prob, auc

# Example data
# adj_matrix = np.array([
#     [0, 1, 0],
#     [1, 0, 1],
#     [0, 1, 0]
# ])



# In[14]:


def binary_cross_entropy_with_logits(logits, targets):
    logits_clipped = np.clip(logits, 1e-7, 1 - 1e-7)  # Avoid log(0)
    loss = -np.sum(targets * np.log(logits_clipped) + (1 - targets) * np.log(1 - logits_clipped))
    return loss

def log_p_A_given_Z(A, Z):
    logits = np.dot(Z, Z.T)  # Inner product of Z
    log_p_A_given_Z = -binary_cross_entropy_with_logits(sigmoid(logits), A)
    return log_p_A_given_Z


def kl_divergence_q_p(mu, logvar):
    mu = np.array(mu)
    logvar = np.array(logvar)
    kl_div = 0.5 * np.sum(np.exp(logvar) + mu**2 - 1 - logvar)
    return kl_div

def calculate_L(A, X,Z,mu,logvar):
    #mu = [0.6152, 0.027, 0.0044, 0.149, -0.0486]
    #logvar = [0.6215289599999998, 0.9992709999999997, 0.9999806400000003, 0.9777990000000003, 0.99763804]#encoder(X, A, input_dim, hidden_dim, latent_dim)
    
    # Print the values of mu and logvar
    #print("mu:", mu)
    #print("logvar:", logvar)
    log_p_A = log_p_A_given_Z(A, Z)
    kl_div = kl_divergence_q_p(mu, logvar)
    L = log_p_A - kl_div
    #print("Loss: ",L)
    return L


# In[15]:


#Parameter update
def para_update(theta,l):
    alpha = 0.001
    theta = theta+ alpha*l
    print("Parameter updated:",theta)
    return theta


# In[94]:


theta = -12.643706170072582
for i in range(1):
    crt = design_cir(theta,X,adj_matrix)
    Z, mean,var = exec_cir(crt,adj_matrix)
    log_likelihood,prob,auc = p_A_given_Z(adj_matrix, Z)
    l = calculate_L(adj_matrix,X,Z,mean,var)
    theta = para_update(theta, l)


# In[96]:


#AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
adj = adj_matrix.flatten()
# Example ground truth labels and predicted probabilities
y_true = adj#np.array([[0,0,1,0,1],[1,1,0,1,1],[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,1,0,1,1],[0,1,0,0,1],[1,0,1,0,0],[1,0,1,0,0]])
y_scores = np.array(auc)
auc1 = roc_auc_score(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)
print("AUC:", auc1)
print('AP',ap_score)


# In[97]:


y_scores


# In[89]:


for i in range(len(y_scores)):
    if y_scores[i]>0.7:
        y_scores[i] = 1
    else:
        y_scores[i] = 0
        


# In[90]:


y_scores


# In[81]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[91]:


precision = precision_score(y_true, y_scores)
recall = recall_score(y_true,y_scores)


# In[92]:


precision


# In[93]:


recall


# In[ ]:


import numpy as np

# Function to calculate the probability density of a multivariate normal distribution
def multivariate_normal_density(z, mu, sigma):
    mu = np.array(mu)
    sigma = np.array(sigma)
    
    size = len(mu)
    if size != len(sigma):
        raise ValueError("The length of mean vector and standard deviation vector must be the same")
    
    # Calculate the determinant of the covariance matrix
    det_sigma = np.prod(sigma**2)
    
    # Calculate the inverse of the covariance matrix
    inv_sigma = np.diag(1 / sigma**2)
    
    # Calculate the normalization constant
    norm_const = 1.0 / (np.sqrt((2 * np.pi)**size * det_sigma))
    
    # Calculate the exponent
    z_mu = z - mu
    result = np.exp(-0.5 * np.dot(np.dot(z_mu.T, inv_sigma), z_mu))
    
    return norm_const * result

# Function to calculate q(Z | X, A)
def q_Z_given_X_A(Z, mu, sigma):
    N = Z.shape[0]
    total_log_prob = 0
    
    for i in range(N):
        z_i = Z[i, :]
        mu_i = mu[i]
        sigma_i = sigma[i]
        prob_density = multivariate_normal_density(z_i, mu_i, sigma_i)
        total_log_prob += np.log(prob_density)
    
    return total_log_prob

# Example data
latent_matrix = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

mu = [
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0]
]

sigma = [
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0]
]

# Calculate q(Z | X, A)
log_prob_q = q_Z_given_X_A(latent_matrix, mu, sigma)
print("Log Probability q(Z | X, A):", log_prob_q)


# In[ ]:




