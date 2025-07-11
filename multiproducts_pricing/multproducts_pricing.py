from scipy.stats import bernoulli
import numpy as np
import time
import math
import csv
from scipy import stats
import statistics
import sys
import pickle
import os
import matplotlib.pyplot as plt

def index_realization_vectorized(cumulate_list, num_rands):
    #This function returns the indices where random values fall within the cumulative distribution.
    return np.searchsorted(cumulate_list, num_rands)

def cost_cal(xi):
    # This function calculates the total cost based on the input demand vector xi.
    cost=np.zeros(n)
    for i in range(n):
        if xi[i]<L_v[i]:
            cost[i]=a_v[i]*xi[i]
        elif xi[i]<U_v[i]:
            cost[i]=b_v[i]*(xi[i]-L_v[i])+a_v[i]*L_v[i]
        else:
            cost[i]=c_v[i]*(xi[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
    return np.sum(cost)

def one_realized_f(x):
    #This function computes the realized profit in a single trial for a given price vector x.
    price_v=x.reshape(n,1)
    exp_value=np.exp(gamma_v*(alpha_v-price_v))
    sum_exp_value=a_0+np.sum(exp_value)
    purchase_prob=exp_value/sum_exp_value
    cumulate_list = np.cumsum(purchase_prob)
    num_rands = np.random.rand(m)
    indices = index_realization_vectorized(cumulate_list, num_rands)
    sum_sold = np.bincount(indices, minlength=n+1)
    cost=np.zeros(n)
    for i in range(n):
        if sum_sold[i]<L_v[i]:
            cost[i]=a_v[i]*sum_sold[i]
        elif sum_sold[i]<U_v[i]:
            cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
        else:
            cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
    return -np.sum(price_v*sum_sold[:-1].reshape(n,1))+np.sum(cost)

def expected_f(x):
    #This function calculates the average realized profit in ``metric_iter'' trials for a given price vector x.
    price_v=x.reshape(n,1)
    exp_value=np.exp(gamma_v*(alpha_v-price_v))
    sum_exp_value=a_0+np.sum(exp_value)
    purchase_prob=exp_value/sum_exp_value

    revenue_v=[]
    for k in range(metric_iter):
        cumulate_list = np.cumsum(purchase_prob)
        num_rands = np.random.rand(m)
        indices = index_realization_vectorized(cumulate_list, num_rands)
        sum_sold = np.bincount(indices, minlength=n+1)

        cost=np.zeros(n)
        for i in range(n):
            if sum_sold[i]<L_v[i]:
                cost[i]=a_v[i]*sum_sold[i]
            elif sum_sold[i]<U_v[i]:
                cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
            else:
                cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
            #debug
            if cost[i]<0:
                error
        revenue_v.append(np.sum(price_v*sum_sold[:-1].reshape(n,1))-np.sum(cost))
    return -np.average(revenue_v)

def expected_f_sample(x,random_num):
    #This function calculates the average realized profit in ``random_num'' trials for a given price vector x.
    price_v=x.reshape(n,1)
    exp_value=np.exp(gamma_v*(alpha_v-price_v))
    sum_exp_value=a_0+np.sum(exp_value)
    purchase_prob=exp_value/sum_exp_value

    revenue_v=[]
    sample_list=[]
    for k in range(random_num):
        cumulate_list = np.cumsum(purchase_prob)
        num_rands = np.random.rand(m)
        indices = index_realization_vectorized(cumulate_list, num_rands)
        sum_sold = np.bincount(indices, minlength=n+1)

        cost=np.zeros(n)
        for i in range(n):
            if sum_sold[i]<L_v[i]:
                cost[i]=a_v[i]*sum_sold[i]
            elif sum_sold[i]<U_v[i]:
                cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
            else:
                cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
        revenue_v.append(np.sum(price_v*sum_sold[:-1].reshape(n,1))-np.sum(cost))
        sample_list.append(sum_sold)
    return -np.average(revenue_v),sample_list

def f_given(x,sum_sold):
    #This function calculates the realized profit given a price vector and a demand vector.
    price_v=x.reshape(n,1)
    cost=np.zeros(n)
    for i in range(n):
        if sum_sold[i]<L_v[i]:
            cost[i]=a_v[i]*sum_sold[i]
        elif sum_sold[i]<U_v[i]:
            cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
        else:
            cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
        #debug
        if cost[i]<0:
            error
    return -np.sum(price_v*sum_sold[:-1].reshape(n,1))+np.sum(cost)

# Common settings
#the number of simulations
num_sim=int(sys.argv[1])
#problem size
n=10
m=40
#Number of samples to calculate the metric defined in Section 6.2 od our paper.
metric_iter=1000
#Maximum number of samples for each method, which is used as the termination condition.
max_m_k_sum=5000

m_k_initial=30
m_k_increase_rate=2

# settings for each method
##GZO-NS and GZO HS
beta_0_guid=0.01
mu_0_guid=0.19
mu_min_guid=0.0001
alpha_k_damping_factor=0.98
alpha_0_guide=0.0
alpha_0_guide_hist=alpha_0_guide
gamma_mu=0.95
gamma_beta=0.95

## window_size for GZO HS
window_hist=1

## the setting of TG and ZO-OGVR
mu_0_ZO_TG_OGVR=0.19
mu_min_TG_OGVR=0.0001
beta_0_ZO_TG=0.01/np.sqrt(n)

## the setting of ZO-OGVR
beta_0_OGVR=0.001
L_xi_alpha_sq_devided_sigma_sq=0.1
s_max=10
initial_samples=20 # samples to calculate c_0

#the setting of the ZO-OG method
mu_0_ZO_OG=0.001
beta_0_ZO_OG=0.00001

#the setting of the random seed
np.random.seed(2024)

for date_ID in [8, 12, 21, 25, 29, 32, 38, 49]:
    #used data
    date_ID=f'{date_ID:02}'

    #Problem setting
    usedata='data/2022_%s.csv' %date_ID
    co_a_0=0.1

    #the initial point for each method
    initial_point=0.5
    x_initial=initial_point*np.ones([n,1]).reshape(n,1)

    #lists to store the results of all method
    GZO_NS_f_result=[]
    GZO_HS_f_result=[]
    ZO_TG_f_result=[]
    ZO_OG_f_result=[]
    ZO_OGVR_f_result=[]

    GZO_NS_time_result=[]
    GZO_HS_time_result=[]
    ZO_TG_time_result=[]
    ZO_OG_time_result=[]
    ZO_OGVR_time_result=[]


    for exp_iter in range(num_sim):
        #Read the actual price data.
        f = open(usedata, encoding="utf-8_sig")
        areas = f.read().split()
        f.close()
        alpha_v=np.array([int(s) for s in areas])[0:n].reshape(n,1)

        #Normalize the price data.
        alpha_v=alpha_v/np.max(alpha_v)

        #Set the parameters of the problem.
        gamma_fac=1.0
        gamma_v=gamma_fac*math.pi/(0.5*np.sqrt(6)*alpha_v)
        w_v=alpha_v*(0.25+0.25*np.random.rand(n,1))
        a_v=2.0*w_v
        b_v=w_v
        c_v=3.0*w_v
        a_0=co_a_0*n
        L_v=m/n*0.5*np.ones([n,1])
        U_v=m/n*1.5*np.ones([n,1])

        #Guided two-point method
        start_time=time.time()
        overhead_time=0

        #Preparation of inputs
        ##x_0
        x_k=x_initial
        ##\mu_0
        mu_k=mu_0_guid
        ##\beta
        beta=beta_0_guid

        #store the elapsed time, the objective value, and the total number of samples
        ##the elapsed time (Time to compute the objective value is excluded.)
        GZO_NS_time_vec=[0]
        ##the objective value
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        GZO_NS_f_vec=[f_end]
        ##the total number of samples
        GZO_NS_m_k_vec=[0]

        alpha_k=alpha_0_guide

        iter=0
        m_k_sum=0
        while 1:

            #set m_k
            m_k=m_k_initial+iter*m_k_increase_rate

            #set beta
            beta=beta*gamma_beta

            #Steps 2 and 3: calculate g_k
            #sample u_k

            #guide gradient
            sum_grad=0
            sumple_num=m_k

            for k in range(sumple_num):
                exp_value=np.exp(gamma_v*(alpha_v-x_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                sum_grad+=-sum_sold.reshape(n+1,1)[0:n]
            m_grad=(1/sumple_num)*sum_grad

            m_grad=m_grad/np.linalg.norm(m_grad)

            noise_guide_gradient=np.random.normal()*m_grad
            u_k=np.sqrt(alpha_k/n)*np.random.normal(size=n).reshape(n,1) + np.sqrt(1-alpha_k) * noise_guide_gradient

            m_k_sum+=sumple_num

            #sample \xi_k^{1,j}, \xi_k^{2,j}, and calculate g_k
            x_mu_u_k=x_k+mu_k*u_k
            f_value_sum_1=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                f_value_sum_1+=f_given(x_mu_u_k,sum_sold)

            x_mu_m_u_k=x_k-mu_k*u_k
            f_value_sum_2=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_m_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                f_value_sum_2+=f_given(x_mu_m_u_k,sum_sold)

            f_value_1_mean=f_value_sum_1/m_k
            f_value_2_mean=f_value_sum_2/m_k
            g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

            m_k_sum+=2*m_k

            #Step 4: Updates the iterate
            x_k=x_k-beta*g_k

            # Step 5: Compute mu_k
            mu_k=np.max([mu_k*gamma_mu,mu_min_guid])
            alpha_k= 1- alpha_k_damping_factor*(1-alpha_k)

            iter+=1

            #Store the elapsed time, the objective value, and the total number of samples
            start_time2=time.time()
            f_end=expected_f(x_k)
            overhead_time+=start_time2-time.time()
            time_end=time.time()-start_time-overhead_time
            GZO_NS_f_vec.append(f_end)
            GZO_NS_time_vec.append(time_end)
            GZO_NS_m_k_vec.append(m_k_sum)

            #termination condition
            if m_k_sum>max_m_k_sum:
                break



        #Guided two-point method with historical sample
        start_time=time.time()
        overhead_time=0

        #Preparation of inputs
        ##x_0
        x_k=x_initial
        ##\mu_0
        mu_k=mu_0_guid
        ##\beta
        beta=beta_0_guid

        #store the elapsed time, the objective value, and the total number of samples
        ##the elapsed time (Time to compute the objective value is excluded.)
        GZO_HS_time_vec=[0]
        ##the objective value
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        GZO_HS_f_vec=[f_end]
        ##the total number of samples
        m_k_sum_2_v_hist=[0]
        alpha_k=alpha_0_guide_hist

        xi_list_of_list=[]
        m_grad=0

        iter=0
        m_k_sum=0
        while 1:

            #set m_k
            m_k=m_k_initial+iter*m_k_increase_rate

            #set beta
            beta=beta*gamma_beta

            #Steps 2 and 3: calculate g_k
            #sample u_k

            #guide gradient
            sum_grad=0
            sumple_num=m_k

            xi_list=[]

            noise_guide_gradient=np.random.normal()*m_grad
            u_k=np.sqrt(alpha_k/n)*np.random.normal(size=n).reshape(n,1) + np.sqrt(1-alpha_k) * noise_guide_gradient


            #sample \xi_k^{1,j}, \xi_k^{2,j}, and calculate g_k
            x_mu_u_k=x_k+mu_k*u_k
            f_value_sum_1=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                xi_list.append(sum_sold)
                f_value_sum_1+=f_given(x_mu_u_k,sum_sold)

            x_mu_m_u_k=x_k-mu_k*u_k
            f_value_sum_2=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_m_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                xi_list.append(sum_sold)
                f_value_sum_2+=f_given(x_mu_m_u_k,sum_sold)

            f_value_1_mean=f_value_sum_1/m_k
            f_value_2_mean=f_value_sum_2/m_k
            g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

            xi_list_of_list.append(xi_list)

            m_k_sum+=2*sumple_num

            #Step 4: Updates the iterate
            x_k=x_k-beta*g_k

            # Step 8: Compute hist grad
            m_grad=0
            window_size=min(window_hist, iter)
            for i in range(window_size):
                tmp=0
                for xi in xi_list_of_list[iter-1-i]:
                    tmp += -xi[0:n].reshape(n,1)
                m_grad+=tmp/len(xi_list_of_list[iter-1-i])
            m_grad=m_grad/window_hist

            if np.linalg.norm(m_grad) > 0.00001:
                m_grad=m_grad/np.linalg.norm(m_grad)

            # Step 5: Compute mu_k
            mu_k=np.max([mu_k*gamma_mu,mu_min_guid])
            alpha_k= 1- alpha_k_damping_factor*(1-alpha_k)

            iter+=1

            #Store the elapsed time, the objective value, and the total number of samples
            start_time2=time.time()
            f_end=expected_f(x_k)
            overhead_time+=start_time2-time.time()
            time_end=time.time()-start_time-overhead_time
            GZO_HS_f_vec.append(f_end)
            GZO_HS_time_vec.append(time_end)
            m_k_sum_2_v_hist.append(m_k_sum)

            #termination condition
            if m_k_sum>max_m_k_sum:
                break


        #No guided two-point method
        start_time=time.time()
        overhead_time=0

        #Preparation of inputs
        ##x_0
        x_k=x_initial
        ##\mu_0
        mu_k=mu_0_ZO_TG_OGVR
        ##\beta
        beta=beta_0_ZO_TG

        #store the elapsed time, the objective value, and the total number of samples
        ##the elapsed time (Time to compute the objective value is excluded.)
        ZO_TG_time_vec=[time.time()-start_time]
        ##the objective value
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        ZO_TG_f_vec=[f_end]
        ##the total number of samples
        m_k_sum_2_v=[0]

        iter=0
        m_k_sum=0
        while 1:

            #set m_k
            m_k=m_k_initial+iter*m_k_increase_rate

            #set beta
            beta=beta*gamma_beta

            #Steps 2 and 3: calculate g_k
            #sample u_k
            u_k=np.random.normal(size=n).reshape(n,1)
            #sample \xi_k^{1,j}, \xi_k^{2,j}, and calculate g_k
            x_mu_u_k=x_k+mu_k*u_k
            f_value_sum_1=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                f_value_sum_1+=f_given(x_mu_u_k,sum_sold)

            x_mu_m_u_k=x_k-mu_k*u_k
            f_value_sum_2=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_m_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                f_value_sum_2+=f_given(x_mu_m_u_k,sum_sold)

            f_value_1_mean=f_value_sum_1/m_k
            f_value_2_mean=f_value_sum_2/m_k
            g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k


            m_k_sum+=2*m_k

            #Step 4: Updates the iterate
            x_k=x_k-beta*g_k

            # Step 5: Compute mu_k
            mu_k=np.max([mu_k*gamma_mu,mu_min_TG_OGVR])

            iter+=1

            #Store the elapsed time, the objective value, and the total number of samples
            start_time2=time.time()
            f_end=expected_f(x_k)
            overhead_time+=start_time2-time.time()
            time_end=time.time()-start_time-overhead_time
            ZO_TG_f_vec.append(f_end)
            ZO_TG_time_vec.append(time_end)
            m_k_sum_2_v.append(m_k_sum)

            #termination condition
            if m_k_sum>max_m_k_sum:
                break

        #NO guided one-point ZO
        #variables for measuring time
        start_time=time.time()
        overhead_time=0

        #Preparation of inputs
        ##x_0
        x_k=x_initial
        ##\mu_0
        mu_k=mu_0_ZO_OG
        ##beta
        beta=beta_0_ZO_OG

        #store the elapsed time, the objective value, and the total number of samples
        ##the elapsed time (Time to compute the objective value is excluded.)
        ZO_OG_time_vec=[0]
        start_time2=time.time()
        tmp=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        ##objective value
        ZO_OG_f_vec=[tmp]
        ##the total number of samples
        CZO_m_k_sum_v=[0]

        iter=0
        m_k_sum=0
        while 1:
            #Set m_k
            m_k=m_k_initial+iter*m_k_increase_rate

            #Set beta
            beta=beta*gamma_beta

            #Calculate a gradient
            u_k=np.random.normal(size=n).reshape(n,1)
            x_mu_u_k=x_k+mu_k*u_k
            f_value_sum=0
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                f_value_sum+=f_given(x_mu_u_k,sum_sold)
            f_value_mean=f_value_sum/m_k
            g_k=f_value_mean/mu_k*u_k

            m_k_sum+=m_k

            #Updates the iterate
            x_k=x_k-beta*g_k

            #Store the elapsed time, the objective value, and the total number of samples
            #the elapsed time (Time to compute the objective value is excluded.)
            start_time2=time.time()
            f_end=expected_f(x_k)
            overhead_time+=start_time2-time.time()
            time_end=time.time()-start_time-overhead_time
            ZO_OG_f_vec.append(f_end)
            ZO_OG_time_vec.append(time_end)
            CZO_m_k_sum_v.append(m_k_sum)

            #termination condition
            if m_k_sum>max_m_k_sum:
                break

            iter+=1

        #ZO OGVR
        #variables for measuring time
        start_time=time.time()
        overhead_time=0

        #Preparation of inputs
        ##x_0
        x_k=x_initial
        ##c_0
        random_num=initial_samples
        c_value,pre_sample_list=expected_f_sample(x_k,random_num)
        ##\mu_0
        mu_k=mu_0_ZO_TG_OGVR
        ##\beta
        beta=beta_0_OGVR

        #store the elapsed time, the objective value, and the total number of samples
        ##the elapsed time (Time to compute the objective value is excluded.)
        ZO_OGVR_time_vec=[time.time()-start_time]
        ##the objective value
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        ZO_OGVR_f_vec=[f_end]
        ##the total number of samples
        m_k_sum_v=[random_num]

        #List stores x_k + \mu_k u_k
        x_mu_u_k_list = [x_k]
        #List stores m_k
        m_k_list = [random_num]
        #List stores xi_k^j
        xi_list_of_list=[pre_sample_list]

        iter=0
        m_k_sum=random_num
        while 1:

            #set m_k
            m_k=m_k_initial+iter*m_k_increase_rate


            #set beta
            beta=beta*gamma_beta

            #Steps 2 and 3: calculate g_k
            #sample u_k
            u_k=np.random.normal(size=n).reshape(n,1)

            x_mu_u_k=x_k+mu_k*u_k
            x_mu_u_k_list.append(x_mu_u_k)

            #sample \xi_k^j and calculate g_k
            f_value_sum=0
            xi_list=[]
            for k in range(m_k):
                exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
                sum_exp_value=a_0+np.sum(exp_value)
                px=exp_value/sum_exp_value
                px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
                cumulate_list = np.cumsum(px)
                num_rands = np.random.rand(m)
                indices = index_realization_vectorized(cumulate_list, num_rands)
                sum_sold = np.bincount(indices, minlength=n+1)
                xi_list.append(sum_sold)
                f_value_sum+=f_given(x_mu_u_k,sum_sold)
            xi_list_of_list.append(xi_list)
            f_value_mean=f_value_sum/m_k
            g_k=(f_value_mean-c_value)/mu_k*u_k

            m_k_sum+=m_k

            #Step 4: Updates the iterate
            x_k=x_k-beta*g_k

            # Step 5: Compute mu_k
            mu_k=np.max([mu_k*gamma_mu,mu_min_TG_OGVR])

            iter+=1
            # Step 6: Compute s
            s = min(s_max, iter)

            # Step 7: Compute a_i
            b_i=np.zeros(s)
            for i in range(s):
                b_i[s-1-i] = L_xi_alpha_sq_devided_sigma_sq*np.linalg.norm(x_k - x_mu_u_k_list[iter-1-i])**2 + 1/m_k_list[iter-1-i]
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
                    tmp += f_given(x_k,xi)
                c_value+=a_i[s-1-i]*tmp/len(xi_list_of_list[iter-1-i])

            #Store the elapsed time, the objective value, and the total number of samples
            start_time2=time.time()
            f_end=expected_f(x_k)
            overhead_time+=start_time2-time.time()
            time_end=time.time()-start_time-overhead_time
            ZO_OGVR_f_vec.append(f_end)
            ZO_OGVR_time_vec.append(time_end)
            m_k_list.append(m_k)
            m_k_sum_v.append(m_k_sum)

            #termination condition
            if m_k_sum>max_m_k_sum:
                break

        #Save the evaluation value
        GZO_NS_f_result.append(GZO_NS_f_vec[-1])
        GZO_NS_time_result.append(GZO_NS_time_vec[-1])

        GZO_HS_f_result.append(GZO_HS_f_vec[-1])
        GZO_HS_time_result.append(GZO_HS_time_vec[-1])

        ZO_TG_f_result.append(ZO_TG_f_vec[-1])
        ZO_TG_time_result.append(ZO_TG_time_vec[-1])

        ZO_OG_f_result.append(ZO_OG_f_vec[-1])
        ZO_OG_time_result.append(ZO_OG_time_vec[-1])

        ZO_OGVR_f_result.append(ZO_OGVR_f_vec[-1])
        ZO_OGVR_time_result.append(ZO_OGVR_time_vec[-1])

    #Calculate p-value of the proposed methods for the CZO (minibatch / batch size 1) method
    Pval_GZO_NS_and_ZO_TG=stats.ttest_rel(GZO_NS_f_result,ZO_TG_f_result)[1]
    Pval_GZO_NS_and_ZO_OG=stats.ttest_rel(GZO_NS_f_result,ZO_OG_f_result)[1]
    Pval_GZO_NS_and_ZO_OGVR=stats.ttest_rel(GZO_NS_f_result,ZO_OGVR_f_result)[1]
    Pval_GZO_HS_and_ZO_TG=stats.ttest_rel(GZO_HS_f_result,ZO_TG_f_result)[1]
    Pval_GZO_HS_and_ZO_OG=stats.ttest_rel(GZO_HS_f_result,ZO_OG_f_result)[1]
    Pval_GZO_HS_and_ZO_OGVR=stats.ttest_rel(GZO_HS_f_result,ZO_OGVR_f_result)[1]

    #Outputs results
    folder_name = f"{date_ID}_{num_sim}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_path = os.path.join(folder_name, 'results.csv')
    with open(file_path, mode='w') as file_tmp:
        writer = csv.writer(file_tmp)
        writer.writerow(['Method','objective value','std_value','computation time (seconds)','std_time','p-value for GZO_NS','p-value for GZO_HS'])
        writer.writerow(['GZO_NS',statistics.mean(GZO_NS_f_result),statistics.stdev(GZO_NS_f_result),statistics.mean(GZO_NS_time_result),statistics.stdev(GZO_NS_time_result),0,0])
        writer.writerow(['GZO_HS',statistics.mean(GZO_HS_f_result),statistics.stdev(GZO_HS_f_result),statistics.mean(GZO_HS_time_result),statistics.stdev(GZO_HS_time_result),0,0])
        writer.writerow(['ZO_TG',statistics.mean(ZO_TG_f_result),statistics.stdev(ZO_TG_f_result),statistics.mean(ZO_TG_time_result),statistics.stdev(ZO_TG_time_result),Pval_GZO_NS_and_ZO_TG,Pval_GZO_HS_and_ZO_TG])
        writer.writerow(['ZO_OG',statistics.mean(ZO_OG_f_result),statistics.stdev(ZO_OG_f_result),statistics.mean(ZO_OG_time_result),statistics.stdev(ZO_OG_time_result),Pval_GZO_NS_and_ZO_OG,Pval_GZO_HS_and_ZO_OG])
        writer.writerow(['ZO_OGVR',statistics.mean(ZO_OGVR_f_result),statistics.stdev(ZO_OGVR_f_result),statistics.mean(ZO_OGVR_time_result),statistics.stdev(ZO_OGVR_time_result),Pval_GZO_NS_and_ZO_OGVR,Pval_GZO_HS_and_ZO_OGVR])
