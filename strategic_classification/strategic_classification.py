
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import random
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import sys
import pickle
import matplotlib.pyplot as plt
import os
from scipy import stats
import csv
import statistics

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

XDIM = 11

# load data
url = 'data/credit_processed.csv'
df = pd.read_csv(url)
df["NoDefaultNextMonth"].replace({0: -1}, inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

df = df.drop(['Married', 'Single', 'Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60'], axis = 1)

scaler = StandardScaler()
df.loc[:, df.columns != "NoDefaultNextMonth"] = scaler.fit_transform(df.drop("NoDefaultNextMonth", axis=1))

fraud_df = df.loc[df["NoDefaultNextMonth"] == -1]
non_fraud_df = df.loc[df["NoDefaultNextMonth"] == 1][:6636]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
df = normal_distributed_df.sample(frac=1).reset_index(drop=True)

Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values

X_all, Y_all = torch.from_numpy(X), torch.from_numpy(Y)

# split test data and train data
Y_all=np.where(Y_all == -1, 0, Y_all)
X_test=X_all[1:1000,:]
Y_test=Y_all[1:1000]
X=X_all[1000:,:]
Y=Y_all[1000:]

def select_random_rows(X, Y, A):
    m = X.shape[0]

    # Randomly select A indices
    random_indices = np.random.choice(m, A, replace=True)

    # Extract corresponding rows
    X_A = X[random_indices, :]
    Y_A = Y[random_indices]

    return X_A, Y_A



#Oracle to obtain features by optimal response
def xi_opt_reaction(xi_true, w, b, gamma):
    # Ensure inputs are numpy arrays
    xi_true = np.asarray(xi_true).reshape(n-1)
    w = np.asarray(w).reshape(n-1)

    # Check if w is not zero
    if np.isclose(np.linalg.norm(w), 0):
        raise ValueError("The vector w must not be a zero vector.")

    # Calculate the optimal z*

    w_dot_xi_true = np.dot(w, xi_true)

    if w_dot_xi_true + b <0:
        w_dot_w = np.dot(w, w)

        xi_accept = xi_true - ((w_dot_xi_true + b) / w_dot_w) * w

        cost=np.linalg.norm(xi_true-xi_accept, ord=2)**2
        if cost < 2/gamma:
            xi_star = xi_accept
            react_flag=1
        else:
            xi_star = xi_true
            react_flag=0
    else:
        xi_star = xi_true
        react_flag=0

    return xi_star, react_flag



#Oracle to compute thew function value of f for given xi_react and xi_L
def f_calc(xi_react, xi_L, w, b):

    z = xi_react @ w + b
    h = 1 / (1 + np.exp(-z))

    loss = - (xi_L * np.log(h + 1e-8) + (1 - xi_L) * np.log(1 - h + 1e-8))
    return loss



#Oracle to calculate errors
def Error_calc(x,X,Y,gamma_cost):
    n = x.shape[0]
    m = X.shape[0]
    f_value_sum=0
    for i in range(m):
        xi_true,xi_L=X[i,:], Y[i]
        #Obtain features by optimal response
        xi_opt, flag=xi_opt_reaction(xi_true, x[0:n-1], x[n-1],gamma_cost)
        #Obtain the function value
        f_value = f_calc(xi_opt,xi_L, x[0:n-1], x[n-1])
        f_value_sum+=f_value
    return (f_value_sum/m)[0]


#Oracle to calculate AUC
def AUC_calc(x,X_test,Y_test,gamma_cost):
    n = x.shape[0]
    m = X_test.shape[0]
    y_score=[]
    for i in range(m):
        xi_true=X_test[i,:]
        #Obtain features by optimal response
        xi_opt, flag=xi_opt_reaction(xi_true, x[0:n-1], x[n-1],gamma_cost)
        y_score.append(xi_opt @ x[0:n-1]+ x[n-1])
    return roc_auc_score(Y_test, y_score)


#Oracle to calculate test_accuracy
def accuracy_calc(x,X_test,Y_test,gamma_cost):
    n = x.shape[0]
    m = X_test.shape[0]
    y_pred=[]
    for i in range(m):
        xi_true=X_test[i,:]
        #Obtain features by optimal response
        xi_opt, flag=xi_opt_reaction(xi_true, x[0:n-1], x[n-1],gamma_cost)
        result = 1 if xi_opt @ x[0:n-1] + x[n-1] >= 0 else 0
        y_pred.append(result)
    return accuracy_score(Y_test, y_pred)


#Oracle to compute the partial gradient information of function F
def f_glad(x,sample_num):
    n = x.shape[0]
    f_glad_sum=0
    for i in range(sample_num):
        xi_true,xi_L=select_random_rows(X, Y, 1)
        #Obtain features by optimal response
        xi_react, flag=xi_opt_reaction(xi_true, x[0:n-1], x[n-1],gamma_cost)
        #Obtain the partial gradient information
        w_xi_p_b= np.dot(x[0:n-1].reshape(n-1),xi_react)+x[n-1]
        sigmoid_val=1/(1+np.exp(-w_xi_p_b))
        w_glad = (xi_L-sigmoid_val) * xi_react
        b_glad = (xi_L-sigmoid_val)
        f_glad = np.concatenate([w_glad, b_glad]).reshape(n,1)
        f_glad_sum+=f_glad
    return f_glad_sum/sample_num


#Oracle to compute the gradient of function f for given x and xi
def f_glad_given_xi(x,xi_react_past,xi_L_past):
    n = x.shape[0]
    w_xi_p_b= np.dot(x[0:n-1].reshape(n-1),xi_react_past)+x[n-1]
    sigmoid_val=1/(1+np.exp(-w_xi_p_b))
    w_glad = (xi_L_past-sigmoid_val) * xi_react_past
    b_glad = (xi_L_past-sigmoid_val)
    f_glad = np.concatenate([w_glad, b_glad]).reshape(n,1)
    return f_glad



# Common settings
n=XDIM +1
## the initial point for each method
initial_point=1
x_initial_point=initial_point*np.ones([n,1]).reshape(n,1)
## sample size at each iteration
m_k_initial=100
m_k_increase_rate=2
#Maximum number of samples for each method, which is used as the termination condition.
#max_m_k_sum=20000
max_m_k_sum=int(sys.argv[1])
gamma_cost=float(sys.argv[2])

# settings for each method
##GZO-NS and GZO HS
mu_0_guid=10
mu_min_guid=0.1
beta_0_guid=1
alpha_k_damping_factor=0.98

alpha_0_guide=0.0
alpha_0_guide_hist=alpha_0_guide
gamma_mu=0.95
gamma_beta=0.95

## window_size for GZO HS
window_hist=1

## the setting of TG and ZO-OGVR
mu_0_ZO_TG_OGVR=10
mu_min_ZO_TG_OGVR=0.1
beta_0_ZO_TG_OGVR=1/np.sqrt(n)

## the setting of ZO-OGVR
L_xi_alpha_sq_devided_sigma_sq_GZO_OGVR=0.1
s_max=10
sample_num_for_c_0=20 # samples to calculate c_0
beta_0_OGVR=1/np.sqrt(n)

#the setting of the ZO-OG method
mu_0_ZO_OG=0.1
beta_0_ZO_OG=0.05


#the number of simulations
num_sim=int(sys.argv[3])

#lists to store the results of all method

# Save the evaluation values
GZO_NS_test_result = []
GZO_HS_test_result = []
ZO_TG_test_result = []
ZO_OG_test_result = []
ZO_OGVR_test_result = []

GZO_NS_train_result = []
GZO_HS_train_result = []
ZO_TG_train_result = []
ZO_OG_train_result = []
ZO_OGVR_train_result = []

GZO_NS_auc_result = []
GZO_HS_auc_result = []
ZO_TG_auc_result = []
ZO_OG_auc_result = []
ZO_OGVR_auc_result = []

GZO_NS_acc_result = []
GZO_HS_acc_result = []
ZO_TG_acc_result = []
ZO_OG_acc_result = []
ZO_OGVR_acc_result = []


for sim_iter in range(num_sim):
    #---------------------------
    #---------------------------
    #GZO-NS
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_initial_point
    ##\mu_0
    mu_k=mu_0_guid
    ##\beta
    beta=beta_0_guid
    ##\alpha
    alpha_k=alpha_0_guide

    #----store the values----
    ##the elapsed time (Time to compute the matrics is excluded.)
    GZO_NS_time_vec=[0]
    ##metrics
    start_time2=time.time()
    GZO_NS_train_error_vec=[Error_calc(x_k,X,Y,gamma_cost)]
    GZO_NS_test_error_vec=[Error_calc(x_k,X_test,Y_test,gamma_cost)]
    GZO_NS_AUC_vec=[AUC_calc(x_k,X_test,Y_test,gamma_cost)]
    GZO_NS_accuracy_vec=[accuracy_calc(x_k,X_test,Y_test,gamma_cost)]
    overhead_time+=start_time2-time.time()
    ##the total number of samples
    GZO_NS_m_k_vec=[0]
    #------------------------

    iter=0
    m_k_sum=0
    while 1:

        #minibatch-size
        m_k=m_k_initial+iter*m_k_increase_rate

        # lines 6-12: calculate partial gradient infomation
        sumple_num=m_k
        p_grad=f_glad(x_k,sumple_num)
        p_grad=p_grad/np.linalg.norm(p_grad)

        #line 3: generate u_k
        noise_guide_gradient=np.random.normal()*p_grad
        u_k=np.sqrt(alpha_k/n)*np.random.normal(size=n).reshape(n,1) + np.sqrt(1-alpha_k) * noise_guide_gradient

        #store the cumulative number of samples
        m_k_sum+=sumple_num

        #line 4: calculate g_k
        ##first term in g_k
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum_1=0
        num_flag=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_u_k[0:n-1], x_mu_u_k[n-1],gamma_cost)
            num_flag+=flag
            f_value = f_calc(xi_opt,xi_L,x_mu_u_k[0:n-1],x_mu_u_k[n-1])
            f_value_sum_1+=f_value

        ##second term in g_k
        x_mu_m_u_k=x_k-mu_k*u_k
        f_value_sum_2=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_m_u_k[0:n-1], x_mu_m_u_k[n-1],gamma_cost)
            f_value = f_calc(xi_opt,xi_L,x_mu_m_u_k[0:n-1],x_mu_m_u_k[n-1])
            f_value_sum_2+=f_value

        f_value_1_mean=f_value_sum_1/m_k
        f_value_2_mean=f_value_sum_2/m_k
        g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

        #store the cumulative number of samples
        m_k_sum+=2*m_k

        #line 5: Updates the iterate
        x_k=x_k-beta*g_k

        #line 13: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_guid])

        #line 14: Compute \alpha_k
        alpha_k= 1-alpha_k_damping_factor*(1-alpha_k)

        iter+=1

        #----store the values----
        ##metrics
        start_time2=time.time()
        GZO_NS_train_error_vec.append(Error_calc(x_k,X,Y,gamma_cost))
        GZO_NS_test_error_vec.append(Error_calc(x_k,X_test,Y_test,gamma_cost))
        GZO_NS_AUC_vec.append(AUC_calc(x_k,X_test,Y_test,gamma_cost))
        GZO_NS_accuracy_vec.append(accuracy_calc(x_k,X_test,Y_test,gamma_cost))
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        ##the elapsed time (Time to compute the matrics is excluded.)
        GZO_NS_time_vec.append(time_end)
        ##the total number of samples
        GZO_NS_m_k_vec.append(m_k_sum)
        #------------------------

        #termination condition
        if m_k_sum>max_m_k_sum:
            break

        #set beta
        beta=beta*gamma_beta

    print("GZO-NS end", "now", sim_iter+1, "max", num_sim)

    #---------------------------
    #---------------------------
    #GZO-HS
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_initial_point
    ##\mu_0
    mu_k=mu_0_guid
    ##\beta
    beta=beta_0_guid
    ##\alpha
    alpha_k=alpha_0_guide_hist

    #----store the values----
    ##the elapsed time (Time to compute the matrics is excluded.)
    GZO_HS_time_vec=[0]
    ##metrics
    start_time2=time.time()
    GZO_HS_train_error_vec=[Error_calc(x_k,X,Y,gamma_cost)]
    GZO_HS_test_error_vec=[Error_calc(x_k,X_test,Y_test,gamma_cost)]
    GZO_HS_AUC_vec=[AUC_calc(x_k,X_test,Y_test,gamma_cost)]
    GZO_HS_accuracy_vec=[accuracy_calc(x_k,X_test,Y_test,gamma_cost)]
    overhead_time+=start_time2-time.time()
    ##the total number of samples
    GZO_HS_m_k_vec=[0]
    #------------------------

    #the list of the list to store histrical samples
    xi_list_of_list=[]
    #partial gradient information
    p_grad=0

    iter=0
    m_k_sum=0
    while 1:

        #minibatch-size
        m_k=m_k_initial+iter*m_k_increase_rate

        #Steps 2 and 3: calculate g_k
        noise_guide_gradient=np.random.normal()*p_grad
        u_k=np.sqrt(alpha_k/n)*np.random.normal(size=n).reshape(n,1) + np.sqrt(1-alpha_k) * noise_guide_gradient

        #the list to store samples at current iteration
        xi_list=[]

        #line 4: calculate g_k
        ##first term in g_k
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum_1=0
        num_flag=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_u_k[0:n-1], x_mu_u_k[n-1],gamma_cost)
            num_flag+=flag
            f_value = f_calc(xi_opt,xi_L,x_mu_u_k[0:n-1],x_mu_u_k[n-1])
            f_value_sum_1+=f_value
            xi_list.append([xi_opt,xi_L])

        ##second term in g_k
        x_mu_m_u_k=x_k-mu_k*u_k
        f_value_sum_2=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_m_u_k[0:n-1], x_mu_m_u_k[n-1],gamma_cost)
            f_value = f_calc(xi_opt,xi_L,x_mu_m_u_k[0:n-1],x_mu_m_u_k[n-1])
            f_value_sum_2+=f_value
            xi_list.append([xi_opt,xi_L])

        f_value_1_mean=f_value_sum_1/m_k
        f_value_2_mean=f_value_sum_2/m_k
        g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

        #store samples at current iteration
        xi_list_of_list.append(xi_list)

        #store the cumulative number of samples
        m_k_sum+=2*m_k

        #line 5: Updates the iterate
        x_k=x_k-beta*g_k

        # Compute hist grad
        p_grad=0
        window_size=min(window_hist, iter)
        for i in range(window_size):
            tmp=0
            for xi in xi_list_of_list[iter-1-i]:
                tmp += f_glad_given_xi(x_k,xi[0],xi[1])
            p_grad+=tmp/len(xi_list_of_list[iter-1-i])

        if np.linalg.norm(p_grad) > 0.00001:
            p_grad=p_grad/np.linalg.norm(p_grad)

        #line 13: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_guid])

        #line 14: Compute \alpha_k
        alpha_k= 1- alpha_k_damping_factor*(1-alpha_k)

        iter+=1

        #----store the values----
        ##metrics
        start_time2=time.time()
        GZO_HS_train_error_vec.append(Error_calc(x_k,X,Y,gamma_cost))
        GZO_HS_test_error_vec.append(Error_calc(x_k,X_test,Y_test,gamma_cost))
        GZO_HS_AUC_vec.append(AUC_calc(x_k,X_test,Y_test,gamma_cost))
        GZO_HS_accuracy_vec.append(accuracy_calc(x_k,X_test,Y_test,gamma_cost))
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        ##the elapsed time (Time to compute the matrics is excluded.)
        GZO_HS_time_vec.append(time_end)
        ##the total number of samples
        GZO_HS_m_k_vec.append(m_k_sum)
        #------------------------

        #termination condition
        if m_k_sum>max_m_k_sum:
            break

        #set beta
        beta=beta*gamma_beta

    print("GZO-HS end", "now", sim_iter+1, "max", num_sim)

    #---------------------------
    #---------------------------
    #ZO-TG
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_initial_point
    ##\mu_0
    mu_k=mu_0_ZO_TG_OGVR
    ##\beta
    beta=beta_0_ZO_TG_OGVR

    #----store the values----
    ##the elapsed time (Time to compute the matrics is excluded.)
    ZO_TG_time_vec=[time.time()-start_time]
    ##metrics
    start_time2=time.time()
    ZO_TG_train_error_vec=[Error_calc(x_k,X,Y,gamma_cost)]
    ZO_TG_test_error_vec=[Error_calc(x_k,X_test,Y_test,gamma_cost)]
    ZO_TG_AUC_vec=[AUC_calc(x_k,X_test,Y_test,gamma_cost)]
    ZO_TG_accuracy_vec=[accuracy_calc(x_k,X_test,Y_test,gamma_cost)]
    overhead_time+=start_time2-time.time()
    ##the total number of samples
    ZO_TG_m_k_vec=[0]
    #------------------------

    iter=0
    m_k_sum=0
    while 1:

        #minibatch-size
        m_k=m_k_initial+iter*m_k_increase_rate

        #Steps 2 and 3: calculate g_k
        #sample u_k
        u_k=np.random.normal(size=n).reshape(n,1)
        #calculate g_k
        ##first term in g_k
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum_1=0
        num_flag=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_u_k[0:n-1], x_mu_u_k[n-1],gamma_cost)
            num_flag+=flag
            f_value = f_calc(xi_opt,xi_L,x_mu_u_k[0:n-1],x_mu_u_k[n-1])
            f_value_sum_1+=f_value
        ##second term in g_k
        x_mu_m_u_k=x_k-mu_k*u_k
        f_value_sum_2=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_m_u_k[0:n-1], x_mu_m_u_k[n-1],gamma_cost)
            f_value = f_calc(xi_opt,xi_L,x_mu_m_u_k[0:n-1],x_mu_m_u_k[n-1])
            f_value_sum_2+=f_value

        f_value_1_mean=f_value_sum_1/m_k
        f_value_2_mean=f_value_sum_2/m_k
        g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

        #store the cumulative number of samples
        m_k_sum+=2*m_k

        #Step 4: Updates the iterate
        x_k=x_k-beta*g_k

        # Step 5: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_ZO_TG_OGVR])

        iter+=1

        #----store the values----
        ##metrics
        start_time2=time.time()
        ZO_TG_train_error_vec.append(Error_calc(x_k,X,Y,gamma_cost))
        ZO_TG_test_error_vec.append(Error_calc(x_k,X_test,Y_test,gamma_cost))
        ZO_TG_AUC_vec.append(AUC_calc(x_k,X_test,Y_test,gamma_cost))
        ZO_TG_accuracy_vec.append(accuracy_calc(x_k,X_test,Y_test,gamma_cost))
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        ##the elapsed time (Time to compute the matrics is excluded.)
        ZO_TG_time_vec.append(time_end)
        ##the total number of samples
        ZO_TG_m_k_vec.append(m_k_sum)
        #------------------------


        #termination condition
        if m_k_sum>max_m_k_sum:
            break

        #set beta
        beta=beta*gamma_beta

    print("ZO-TG end", "now", sim_iter+1, "max", num_sim)

    #---------------------------
    #---------------------------
    #ZO-OG

    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_initial_point
    ##\mu_0
    mu_k=mu_0_ZO_OG
    ##beta
    beta=beta_0_ZO_OG

    #----store the values----
    ##the elapsed time (Time to compute the matrics is excluded.)
    ZO_OG_time_v=[0]
    ##metrics
    start_time2=time.time()
    ZO_OG_train_error_vec=[Error_calc(x_k,X,Y,gamma_cost)]
    ZO_OG_test_error_vec=[Error_calc(x_k,X_test,Y_test,gamma_cost)]
    ZO_OG_AUC_vec=[AUC_calc(x_k,X_test,Y_test,gamma_cost)]
    ZO_OG_accuracy_vec=[accuracy_calc(x_k,X_test,Y_test,gamma_cost)]
    overhead_time+=start_time2-time.time()
    ##the total number of samples
    ZO_OG_m_k_vec=[0]
    #------------------------

    iter=0
    m_k_sum=0
    while 1:
        #minibatch-size
        m_k=m_k_initial+iter*m_k_increase_rate

        #calculate g_k
        u_k=np.random.normal(size=n).reshape(n,1)
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum=0
        num_flag=0
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_u_k[0:n-1], x_mu_u_k[n-1],gamma_cost)
            num_flag+=flag
            f_value = f_calc(xi_opt,xi_L,x_mu_u_k[0:n-1],x_mu_u_k[n-1])
            f_value_sum+=f_value
        f_value_mean=f_value_sum/m_k
        g_k=f_value_mean/mu_k*u_k

        #store the cumulative number of samples
        m_k_sum+=m_k

        #Updates the iterate
        x_k=x_k-beta*g_k

        #----store the values----
        ##metrics
        start_time2=time.time()
        ZO_OG_train_error_vec.append(Error_calc(x_k,X,Y,gamma_cost))
        ZO_OG_test_error_vec.append(Error_calc(x_k,X_test,Y_test,gamma_cost))
        ZO_OG_AUC_vec.append(AUC_calc(x_k,X_test,Y_test,gamma_cost))
        ZO_OG_accuracy_vec.append(accuracy_calc(x_k,X_test,Y_test,gamma_cost))
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        ##the elapsed time (Time to compute the matrics is excluded.)
        ZO_OG_time_v.append(time_end)
        ##the total number of samples
        ZO_OG_m_k_vec.append(m_k_sum)
        #------------------------

        #termination condition
        if m_k_sum>max_m_k_sum:
            break

        #Set beta
        beta=beta*gamma_beta

        iter+=1

    print("ZO-OG end", "now", sim_iter+1, "max", num_sim)

    #---------------------------
    #---------------------------
    #ZO OGVR
    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_initial_point
    ##\mu_0
    mu_k=mu_0_ZO_TG_OGVR
    ##\beta
    beta=beta_0_ZO_TG_OGVR

    ## calculate c_0
    random_num=sample_num_for_c_0
    pre_sample_list=[]
    for k in range(random_num):
        xi_true,xi_L=select_random_rows(X, Y, 1)
        xi_opt, flag=xi_opt_reaction(xi_true, x_k[0:n-1], x_k[n-1],gamma_cost)
        num_flag+=flag
        f_value = f_calc(xi_opt,xi_L,x_k[0:n-1],x_k[n-1])
        f_value_sum+=f_value
        pre_sample_list.append([xi_opt,xi_L])
    c_value=f_value_sum/random_num


    #----store the values----
    ##the elapsed time (Time to compute the matrics is excluded.)
    ZO_OGVR_time_vec=[time.time()-start_time]
    ##metrics
    start_time2=time.time()
    ZO_OGVR_train_error_vec=[Error_calc(x_k,X,Y,gamma_cost)]
    ZO_OGVR_test_error_vec=[Error_calc(x_k,X_test,Y_test,gamma_cost)]
    ZO_OGVR_AUC_vec=[AUC_calc(x_k,X_test,Y_test,gamma_cost)]
    ZO_OGVR_accuracy_vec=[accuracy_calc(x_k,X_test,Y_test,gamma_cost)]
    overhead_time+=start_time2-time.time()
    ##the total number of samples
    ZO_OGVR_m_k_vec=[random_num]
    #------------------------

    #List stores x_k + \mu_k u_k
    x_mu_u_k_list = [x_k]
    #List stores m_k
    m_k_list = [random_num]
    #the list of the list to store histrical samples
    xi_list_of_list=[pre_sample_list]

    iter=0
    m_k_sum=random_num
    while 1:

        #minibatch-size
        m_k=m_k_initial+iter*m_k_increase_rate

        #set beta
        beta=beta*gamma_beta

        #Steps 2 and 3: calculate g_k
        #sample u_k
        u_k=np.random.normal(size=n).reshape(n,1)

        x_mu_u_k=x_k+mu_k*u_k
        x_mu_u_k_list.append(x_mu_u_k)

        #calculate g_k
        f_value_sum=0
        num_flag=0
        xi_list=[]
        for k in range(m_k):
            xi_true,xi_L=select_random_rows(X, Y, 1)
            xi_opt, flag=xi_opt_reaction(xi_true, x_mu_u_k[0:n-1], x_mu_u_k[n-1],gamma_cost)
            num_flag+=flag
            f_value = f_calc(xi_opt,xi_L,x_mu_u_k[0:n-1],x_mu_u_k[n-1])
            f_value_sum+=f_value
            xi_list.append([xi_opt,xi_L])
        xi_list_of_list.append(xi_list)
        f_value_mean=f_value_sum/m_k
        #the gradient with variance reduction parameter c_value
        g_k=(f_value_mean-c_value)/mu_k*u_k

        #store samples at current iteration
        m_k_sum+=m_k

        #Step 4: Updates the iterate
        x_k=x_k-beta*g_k

        # Step 5: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_ZO_TG_OGVR])

        iter+=1
        # Step 6: Compute s
        s = min(s_max, iter)

        # Step 7: Compute a_i
        b_i=np.zeros(s)
        for i in range(s):
            b_i[s-1-i] = L_xi_alpha_sq_devided_sigma_sq_GZO_OGVR*np.linalg.norm(x_k - x_mu_u_k_list[iter-1-i])**2 + 1/m_k_list[iter-1-i]
        b_inv_sum=0
        for i in range(s):
            b_inv_sum+=1/b_i[i]
        a_i=np.zeros(s)
        for i in range(s):
            a_i[i] = 1/(b_i[i]*b_inv_sum)

        # Step 8: Compute c_k+1
        c_value=0
        for i in range(s):
            tmp=0
            for xi in xi_list_of_list[iter-1-i]:
                tmp += f_calc(xi[0],xi[1],x_k[0:n-1],x_k[n-1])
            c_value+=a_i[s-1-i]*tmp/len(xi_list_of_list[iter-1-i])

        #----store the values----
        ##metrics
        start_time2=time.time()
        ZO_OGVR_train_error_vec.append(Error_calc(x_k,X,Y,gamma_cost))
        ZO_OGVR_test_error_vec.append(Error_calc(x_k,X_test,Y_test,gamma_cost))
        ZO_OGVR_AUC_vec.append(AUC_calc(x_k,X_test,Y_test,gamma_cost))
        ZO_OGVR_accuracy_vec.append(accuracy_calc(x_k,X_test,Y_test,gamma_cost))
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        ##the elapsed time (Time to compute the matrics is excluded.)
        ZO_OGVR_time_vec.append(time_end)
        ##the total number of samples
        m_k_list.append(m_k)
        ZO_OGVR_m_k_vec.append(m_k_sum)

        #termination condition
        if m_k_sum>max_m_k_sum:
            break

    print("ZO-OGVR end", "now", sim_iter+1, "max", num_sim)

    # Save the evaluation values
    GZO_NS_test_result.append(GZO_NS_test_error_vec[-1])
    GZO_HS_test_result.append(GZO_HS_test_error_vec[-1])
    ZO_TG_test_result.append(ZO_TG_test_error_vec[-1])
    ZO_OG_test_result.append(ZO_OG_test_error_vec[-1])
    ZO_OGVR_test_result.append(ZO_OGVR_test_error_vec[-1])

    GZO_NS_train_result.append(GZO_NS_train_error_vec[-1])
    GZO_HS_train_result.append(GZO_HS_train_error_vec[-1])
    ZO_TG_train_result.append(ZO_TG_train_error_vec[-1])
    ZO_OG_train_result.append(ZO_OG_train_error_vec[-1])
    ZO_OGVR_train_result.append(ZO_OGVR_train_error_vec[-1])

    GZO_NS_auc_result.append(GZO_NS_AUC_vec[-1])
    GZO_HS_auc_result.append(GZO_HS_AUC_vec[-1])
    ZO_TG_auc_result.append(ZO_TG_AUC_vec[-1])
    ZO_OG_auc_result.append(ZO_OG_AUC_vec[-1])
    ZO_OGVR_auc_result.append(ZO_OGVR_AUC_vec[-1])

    GZO_NS_acc_result.append(GZO_NS_accuracy_vec[-1])
    GZO_HS_acc_result.append(GZO_HS_accuracy_vec[-1])
    ZO_TG_acc_result.append(ZO_TG_accuracy_vec[-1])
    ZO_OG_acc_result.append(ZO_OG_accuracy_vec[-1])
    ZO_OGVR_acc_result.append(ZO_OGVR_accuracy_vec[-1])

# Function to calculate p-values for all metrics
def calculate_p_values(results1, results2):
    return stats.ttest_rel(results1, results2)[1]

# Calculate p-values for all metrics
p_values = {}
metrics = ['test', 'train', 'auc', 'acc']
results_dict = {
    'test': [GZO_NS_test_result, GZO_HS_test_result, ZO_TG_test_result, ZO_OG_test_result, ZO_OGVR_test_result],
    'train': [GZO_NS_train_result, GZO_HS_train_result, ZO_TG_train_result, ZO_OG_train_result, ZO_OGVR_train_result],
    'auc': [GZO_NS_auc_result, GZO_HS_auc_result, ZO_TG_auc_result, ZO_OG_auc_result, ZO_OGVR_auc_result],
    'acc': [GZO_NS_acc_result, GZO_HS_acc_result, ZO_TG_acc_result, ZO_OG_acc_result, ZO_OGVR_acc_result],
}

for metric in metrics:
    results = results_dict[metric]
    p_values[metric] = {
        'GZO_NS_GZO_HS': calculate_p_values(results[0], results[1]),
        'GZO_NS_ZO_TG': calculate_p_values(results[0], results[2]),
        'GZO_NS_ZO_OG': calculate_p_values(results[0], results[3]),
        'GZO_NS_ZO_OGVR': calculate_p_values(results[0], results[4]),
        'GZO_HS_ZO_TG': calculate_p_values(results[1], results[2]),
        'GZO_HS_ZO_OG': calculate_p_values(results[1], results[3]),
        'GZO_HS_ZO_OGVR': calculate_p_values(results[1], results[4]),
    }

# Output results to a CSV file
folder_name = "_".join(sys.argv[1:])

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, 'results.csv')
with open(file_path, mode='w', newline='') as file_tmp:
    writer = csv.writer(file_tmp)
    # Write headers
    writer.writerow(['Metric', 'Method', 'Objective Value', 'Std Value', 'p-value GZO_NS', 'p-value GZO_HS'])

    # Write results for each metric
    for metric in metrics:
        results = results_dict[metric]
        p_vals = p_values[metric]
        writer.writerow([metric, 'GZO_NS', statistics.mean(results[0]), statistics.stdev(results[0]), 'None', p_vals['GZO_NS_GZO_HS']])
        writer.writerow([metric, 'GZO_HS', statistics.mean(results[1]), statistics.stdev(results[1]), p_vals['GZO_NS_GZO_HS'], 'None'])
        writer.writerow([metric, 'ZO_TG', statistics.mean(results[2]), statistics.stdev(results[2]), p_vals['GZO_NS_ZO_TG'], p_vals['GZO_HS_ZO_TG']])
        writer.writerow([metric, 'ZO_OG', statistics.mean(results[3]), statistics.stdev(results[3]), p_vals['GZO_NS_ZO_OG'], p_vals['GZO_HS_ZO_OG']])
        writer.writerow([metric, 'ZO_OGVR', statistics.mean(results[4]), statistics.stdev(results[4]), p_vals['GZO_NS_ZO_OGVR'], p_vals['GZO_HS_ZO_OGVR']])
