#!/usr/bin/env python
# coding: utf-8

# In[55]:


from scipy.io import loadmat
import numpy as np
import copy
from numpy import linalg as LA
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import model_selection
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift


# In[135]:

'''自定归一化'''
def norm(X):
    re2_X = X
    n, p = X.shape
    re3_X = []
    for i in range(0, n):
        re3_Xj = []
        for j in range(0, p):
            re3_Xj.append((re2_X[i][j] - min(re2_X[i]))/(max(re2_X[i]) - min(re2_X[i])))
        re3_X.append(re3_Xj)
    re3_X = np.array(re3_X)
    return re3_X

def D1(X):
    n, p = X.shape
    D = np.ones((n, p - 1))
    for i in range(n):
        D[i] = np.diff(X[i])
        
    return D

def D2(X):
    n, p = X.shape
    D = np.ones((n, p - 2))
    for i in range(n):
        D[i] = np.diff(np.diff(X[i]))
        
    return D

def msc(X):

    me = np.mean(X, axis = 0)
    [m,p] = np.shape(X)
    X_msc = np.zeros((m,p))
    for i in range(m):
        poly = np.polyfit(me, X[i],1) 
        j = 0
        X_msc[i] = (X[i]-poly[1])/poly[0]
    
    return X_msc

'''自写SG平滑算法'''
def SGSDW_weight(k, window_size): #k为k阶拟合;window_size为窗口大小，限制为奇数
    j = window_size // 2
    m = np.array([[a**b for a in range(-j, j + 1)] for b in range(k + 1)]).T
    weight = LA.inv(m.T @ m)@ m.T
    
    return (m @ weight)[j]

def SGSDW(k, window_size, X1): #X的形状是 样本x光谱长度
    assert window_size % 2 != 0
    j = window_size // 2
    weight = SGSDW_weight(k, window_size)
    X = copy.deepcopy(X1)
    for i in range(X.shape[0]):
        x = X[i]
        x_m = np.expand_dims(x, axis = 0)
        for a in range(-j, j + 1):
            if a < 0:
                x_l = shift(x, -a, cval=x[0])
                x_m = np.insert(x_m, a + j, x_l, axis = 0)
            elif a > 0:
                x_l = shift(x, -a, cval=x[-1])
                x_m = np.insert(x_m, a + j, x_l, axis = 0)
        X[i] = weight @ x_m

    return X
        
    
def evaluation(water_train, water_hat1, water_test, water_hat2):
    r2 = r2_score(water_train, water_hat1)
    q2 = r2_score(water_test, water_hat2)
    rmsec = np.sqrt(mean_squared_error(water_train, water_hat1))
    rmsep = np.sqrt(mean_squared_error(water_test, water_hat2))
    print(f'r2:{r2}\nq2:{q2}\nrmsec:{rmsec}\nrmsep:{rmsep}\n')
    
    return r2, q2, rmsec, rmsep


# In[3]:


corn = loadmat('./data/corn.mat')

X = corn['mp5']
water = corn['water']
pro = corn['pro']
oil = corn['oil']
starch = corn['water']

plt.figure()
plt.plot(X.T)

X = D2(X)

X = SGSDW(2,7,X) #k阶近似选择2，窗口大小选择7


plt.figure()
plt.plot(X.T)


# In[4]:


X_train, X_test, water_train, water_test = train_test_split(X, water, test_size=0.2, random_state=42)

def evaluation(water_train, water_hat1, water_test, water_hat2):
    r2 = r2_score(water_train, water_hat1)
    q2 = r2_score(water_test, water_hat2)
    rmsec = np.sqrt(mean_squared_error(water_train, water_hat1))
    rmsep = np.sqrt(mean_squared_error(water_test, water_hat2))
    print(f'r2:{r2}\nq2:{q2}\nrmsec:{rmsec}\nrmsep:{rmsep}\n')
    
    return r2, q2, rmsec, rmsep


# In[5]:


kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

mse = []

for i in np.arange(1, 20):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, X_train, water_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(-score)

optimal_pc = np.argmin(mse) + 1
plt.figure()
plt.plot(np.arange(1, 20), np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Water in corn')
plt.xlim(xmin=-1)


# In[6]:


pls2 = PLSRegression(n_components=optimal_pc)
pls2.fit(X_train, water_train)
water_hat1 = pls2.predict(X_train)
water_hat2 = pls2.predict(X_test)

r2, q2, rmsec, rmsep = evaluation(water_train, water_hat1, water_test, water_hat2)

plt.figure()
plt.plot(water_train, water_hat1, 'ko', 
         label=f"$R^2$={r2:.2f}, RMSEC={rmsec:.2f}")
plt.plot(water_test, water_hat2, 'bx', 
         label=f"$Q^2$={q2:.2f}, RMSEP={rmsep:.2f}")
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual v.s. Predicted')
plt.legend()
ax = plt.gca()
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")




