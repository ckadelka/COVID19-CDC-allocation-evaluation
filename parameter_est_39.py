#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:33:04 2021

@author: rafiul
"""

import sys
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
#from geneticalgorithm import geneticalgorithm as ga
#from geneticalgorithm_pronto import geneticalgorithm as ga
from ga import ga
import get_cfr00
import get_yll_and_mean_age
import get_vaccination_rates_new
import get_contact_matrix
import get_cases_deaths_recovered_from_JHU_data as get_JHU_data
#from datetime import datetime
#
#def days_between(d1, d2):
#    d1 = datetime.strptime(d1, "%Y-%m-%d")
#    d2 = datetime.strptime(d2, "%Y-%m-%d")
#    return abs((d2 - d1).days)
##from matplotlib import cm

version='39'
output_folder = 'results/version%s/' % version

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 4#random.randint(0,100)

if len(sys.argv)>2:
    nsim = int(sys.argv[2])
else:
    nsim = 10

if len(sys.argv)>3:
    niter = int(sys.argv[3])
else:
    niter = 20

if len(sys.argv)>4:
    npop = int(sys.argv[4])
else:
    npop = 200#1000


######
#  Global Variables
####### 

## model parameters

Ngp=17 #number of groups
Ncomp=20 #number of compartments per group
gp_in_agegp = np.array([0,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3]) #age of each group
vacopt = np.array([4,2,1,3,2,2,1,4,3,3,1,3,3,3,3,4,2])-1
T=365
dt=1/2 #needs to be 1/natural number so there is an exact match between modeled case and death count and observed daily death count

######
#  Global Variables
####### 

## fixed parameter choices

mean_4agegp = get_yll_and_mean_age.mean_4agegp #from SSA
mean_age_in_agegp = mean_4agegp[gp_in_agegp]

yll_4agegp = get_yll_and_mean_age.yll_4agegp #from SSA

hesitancy=0.3 #based on most recent estimate 
option_vaccine_projection = 2 #2 = linearly decreasing daily vaccinations, 1= constant at level of May 4, 2021 until all willing individuals are vaccinated
number_of_daily_vaccinations = np.array(list(map(lambda x: get_vaccination_rates_new.vaccinations_on_day_t(x,hesitancy,option_vaccine_projection),range(500))),dtype=np.float64)

Nsize= get_cfr00.Nsize #from US 2019 census data
CFR = get_cfr00.CFR 

mu_E = 1/3.7 #https://bmjopen.bmj.com/content/10/8/e039652.abstract
mu_P = 1/2.1 #from https://www.nature.com/articles/s41591-020-0962-9.pdf
mu_C = 1/2.7227510411724856 #fitted from N=22507139 available CDC cases as of April 9, 2021
mu_Q = 1/(22-1/mu_C) #22 days yield best overlay of case and death counts using John Hopkins data 
mu_A = 1/5 #from https://www.nature.com/articles/s41591-020-0962-9.pdf, (1/mu_P+1/mu_C)

def q_based_on_q17(q17,weighted_mean_q = 0.7):
   q=np.zeros(Ngp,dtype=np.float64)
   param=0.02
   N_4 = np.asarray(get_contact_matrix.N_4,dtype=np.float64)
   q_4 = 1-param*(mean_4agegp[-1]-mean_4agegp)
   mean_q = np.dot(N_4,q_4)/np.sum(N_4)
   q_4 = q_4 + (weighted_mean_q-mean_q)
   param = param*(q17-weighted_mean_q)/(q_4[3]-weighted_mean_q)
   q_4 = 1-param*(mean_4agegp[-1]-mean_4agegp)
   mean_q = np.dot(N_4,q_4)/np.sum(N_4)
   q_4 = q_4 + (weighted_mean_q-mean_q)
   for i in range(Ngp):
       q[i] = q_4[gp_in_agegp[i]]
   return  q 

#q=0.7*np.ones(17,dtype=np.float64)
q17=0.85
q = q_based_on_q17(q17)

contact_matrix = get_contact_matrix.contact_matrix #from 2017 paper and adaptations
#contact_matrix = get_contact_matrix.old_contact_matrix #from 2017 paper and adaptations

cases_prop=get_cfr00.cases_prop
death_prop = get_cfr00.death_prop

index_2020_10_13 = get_JHU_data.index_2020_10_13+62
daily_cases = get_JHU_data.daily_cases
filtered_daily_cases = get_JHU_data.filtered_daily_cases
filtered_daily_deaths = get_JHU_data.filtered_daily_deaths
cumdeaths = get_JHU_data.cumdeaths
cumcases = get_JHU_data.cumcases
max_days_back = 40
dates = get_JHU_data.dates

last_day_for_fit = '2021-04-28'# if SLURM_ID%2==0 else '2021-04-09'
dates_index = list(dates).index(last_day_for_fit)
filtered_daily_cases = filtered_daily_cases[:dates_index]
daily_cases = daily_cases[:dates_index+1]
cumcases = cumcases[:dates_index+1]
cumdeaths = cumdeaths[:dates_index+1]
dates = dates[:dates_index+1]

import pandas as pd
#A = pd.read_excel('variants.xlsx')
A = pd.read_csv('variants.csv')
beta_multiplier = []
relative_transmissibility = np.array(A['transmissibility'])/100
dates_variants = [str(el)[:10] for el in A.columns[2:]]
dates_variants_index = [list(dates).index(d)-7 for d in dates_variants]
A_data = np.array(A.iloc[:,2:])

def function(x,x0,k):
    return 1+0.5/(1+np.exp(-k*(x-x0)))

from scipy.optimize import curve_fit
params, covs = curve_fit(function, range(6),np.dot(relative_transmissibility,A_data))
xs = np.arange(-dates_variants_index[0]/14,100,1/14)
#
#f,ax = plt.subplots()
#ax.plot(dates_variants_index,np.dot(relative_transmissibility,A_data),'x')
#x0 = np.arange(3,5,0.1)
#ax.plot(dates_variants_index[0]+14*xs,function(xs,params[0],params[1]))

overall_transmissibility = function(xs,params[0],params[1])
overall_transmissibility = overall_transmissibility[index_2020_10_13:]
#overall_transmissibility = (overall_transmissibility-1)*10+1

## fitted parameter choices 

#beta
#midc
#exponent

## variable parameter choices
f_A = 0.75 #CDC best estimate
f_V=0.5 #big unknown, we'll vary it from 0 to 1
q17 = 0.85

dummy = SLURM_ID%9
if dummy<3:
    f_A = np.array([0.25,0.75,1])[dummy%3]
elif dummy<6:
    f_V = np.array([0,0.5,1])[dummy%3]
elif dummy<9:
    q17 = np.array([0.7,0.85,1])[dummy%3]
q = q_based_on_q17(q17)

ve=0.9
sigma = 0.7* np.ones(17,dtype=np.float64)#np.array([0,1-np.sqrt(1-ve),ve])[index3] * np.ones(17,dtype=np.float64)
delta = 1-(1-ve)/(1-sigma)#np.array([ve,1-np.sqrt(1-ve),0])[index3] * np.ones(17,dtype=np.float64)

## calculated values based on choice of mu_C

active_filtered_cases = np.zeros(len(filtered_daily_cases)+1,dtype=np.float64)
active_filtered_cases[0]=daily_cases[0]
for i in np.arange(len(filtered_daily_cases)):
    dummy=np.arange(min(max_days_back,i+1))
    active_filtered_cases[i+1] = np.dot(filtered_daily_cases[i-dummy],(1-mu_C)**dummy)

inferred_recovered = cumcases-cumdeaths-active_filtered_cases-mu_C/mu_Q*active_filtered_cases

param = np.zeros(4,dtype=np.float64)

######
#  Functions 
#######

@jit(nopython=True)
def get_initial_values(Nsize,mu_A,mu_C,mu_Q,q,START_DATE_INDEX=index_2020_10_13,hesitancy=0.3):
    cases = active_filtered_cases[START_DATE_INDEX]#(cumcases[START_DATE_INDEX]-cumcases[START_DATE_INDEX-1])
    #cases = 1/mu_C*(cumcases[START_DATE_INDEX]-cumcases[START_DATE_INDEX-1])
    death = cumdeaths[START_DATE_INDEX]
    recover = inferred_recovered[START_DATE_INDEX]
    
    total_nr_recovered_by_age = recover*cases_prop-death*death_prop#np.array([recover*cases_prop[i]-death*death_prop[i] for i in range(4)])
    total_nr_recovered_by_age = total_nr_recovered_by_age/np.sum(total_nr_recovered_by_age)*recover
    q_small = np.array([q[0],q[1],q[9],q[13]],dtype=np.float64) #q[0,1,9,13]#[0.25,0.42,0.66,0.69]
    
    total_nr_recovered_or_dead_by_age = total_nr_recovered_by_age+death*death_prop#np.array([(total_nr_recovered_by_age[i]+death*death_prop[i]) for i in range(4)])
    total_nr_infections_by_age = 1/q_small*total_nr_recovered_or_dead_by_age#np.array([1/q_small[i]*total_nr_recovered_or_dead_by_age[i] for i in range(4)])
    total_nr_asymptomatic_recovered_by_age = (1-q_small)*total_nr_infections_by_age#np.array([(1-q_small[i])*total_nr_infections_by_age[i] for i in range(4)])
    
    recover_asympto_prop = recover*cases_prop/q_small*(1-q_small) #np.array([recover*cases_prop[i]/q_small[i]*(1-q_small[i]) for i in range(4)])
    recover_asympto_prop = recover_asympto_prop/np.sum(recover_asympto_prop)
    
    gp_in_agegp = np.array([0,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3],dtype=np.int32)
    
    initial_values=np.zeros(Ncomp*Ngp,dtype=np.float64)#np.array(323,dtype=np.float64)#
    #the last element keeps track of the current phase
    
    for i in range(Ngp):
        j=i*Ncomp
        
        gp_proportion_amoung_agegp = Nsize[i]/np.dot(np.asarray(gp_in_agegp==gp_in_agegp[i],dtype=np.float64),Nsize)
        gp_proportion_amoung_agegp_times_CFR = CFR[i]*Nsize[i]/np.dot(np.multiply(CFR,np.asarray(gp_in_agegp==gp_in_agegp[i],dtype=np.float64)),Nsize)
        gp_proportion_amoung_agegp_times_1_minus_CFR = (1-CFR[i])*Nsize[i]/np.dot(np.multiply(1-CFR[i],np.asarray(gp_in_agegp==gp_in_agegp[i],dtype=np.float64)),Nsize)
        
        initial_values[j+15] = cases*(cases_prop[gp_in_agegp[i]])*gp_proportion_amoung_agegp  #clinical
        initial_values[j+16] = 0. #clinical -- vaccinated
        initial_values[j+17] = death*(death_prop[gp_in_agegp[i]])*gp_proportion_amoung_agegp_times_CFR #dead
        initial_values[j+18] = total_nr_recovered_by_age[gp_in_agegp[i]]*gp_proportion_amoung_agegp_times_1_minus_CFR #recov
        
        initial_values[j+19] = mu_C/mu_Q*(initial_values[j+15]+initial_values[j+16])
        
        initial_values[j+12]= mu_C/mu_P*(1-hesitancy)*initial_values[j+15] #pre-clinical, willing to vaccinate
        initial_values[j+13]= mu_C/mu_P*(hesitancy)*initial_values[j+15] #pre-clinical, not willing to vaccinate
        initial_values[j+14]=0. #pre-clinical, vaccinated
        
        initial_values[j+3] = 1/q[i]*mu_C/mu_E*(1-hesitancy) *initial_values[j+15] #expo, willing to vaccinate
        initial_values[j+4] = 1/q[i]*mu_C/mu_E*hesitancy *initial_values[j+15] #expo, not willing to vaccinate
        initial_values[j+5] =0. #expo, vaccinated
        
        initial_values[j+6] = 1/q[i]*(1-q[i])*mu_C/mu_A*(1-hesitancy) *initial_values[j+15]  #asympto, willing to vaccinate
        initial_values[j+7] = 1/q[i]*(1-q[i])*mu_C/mu_A*hesitancy *initial_values[j+15]  #asympto, not willing to vaccinate
        initial_values[j+8] = 0. #asympto, vaccinated
        
        initial_values[j+9] =(1-hesitancy)*total_nr_asymptomatic_recovered_by_age[gp_in_agegp[i]]*gp_proportion_amoung_agegp #asymptomatic recovered, willing to vaccinate
        initial_values[j+10]=hesitancy*total_nr_asymptomatic_recovered_by_age[gp_in_agegp[i]]*gp_proportion_amoung_agegp #asymptomatic recovered, not willing to vaccinate
        initial_values[j+11]=0. #asymptomatic recovered, vaccinated
        
        total_susceptible= Nsize[i] - np.sum(initial_values[(j+3):(j+20)])
        
        initial_values[j]= (1-hesitancy)*total_susceptible #susceptible, willing to vaccinate
        initial_values[j+1] = hesitancy*total_susceptible #susceptible, not willing to vaccinate
        initial_values[j+2] = 0. #susceptible, vaccinated
    
    return initial_values

@jit(nopython=True) 
def parm_to_beta(param):
   beta=np.zeros(Ngp,dtype=np.float64)
   for i in range(Ngp):
       beta[i] = param[0] + mean_age_in_agegp[i] * param[1]
   return  beta 

@jit(nopython=True) 
def parm_to_q(param):
   q=np.zeros(Ngp,dtype=np.float64)
   N_4 = np.asarray(get_contact_matrix.N_4,dtype=np.float64)
   q_4 = 1-param*(mean_4agegp[-1]-mean_4agegp)
   mean_q = np.dot(N_4,q_4)/np.sum(N_4)
   q_4 = q_4 + (0.7-mean_q)
   for i in range(Ngp):
       q[i] = q_4[gp_in_agegp[i]]
   return  q 

@jit(nopython=True)
def mat_vecmul(matrix1,vector):
    rvector = np.zeros(matrix1.shape[0],dtype=np.float64)
    for i in range(matrix1.shape[0]):
        for k in range(matrix1.shape[0]):
            rvector[i] += matrix1[i][k] * vector[k]
    return rvector


@jit(nopython=True)#parallel=True)
def SYS_ODE_VAX_RK4(X,t,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations): 
    #mu_A=1/(1/mu_P+1/mu_C) 
    
    f = np.zeros(Ngp,dtype=np.float64)
    for i in range(Ngp):
        f[i] = (f_A*(X[6::Ncomp][i]+X[7::Ncomp][i]+f_V*X[8::Ncomp][i]) + X[12::Ncomp][i]+X[13::Ncomp][i]+f_V*X[14::Ncomp][i] + X[15::Ncomp][i]+f_V*X[16::Ncomp][i])/Nsize[i]
    
    dummy = mat_vecmul(contact_matrix,f)
    F = np.multiply(beta,dummy)
    
    daily_doses = number_of_daily_vaccinations[int(t)]
    
#    nr_of_ppl_to_be_vaccinated = X[::Ncomp]+X[3::Ncomp]+X[6::Ncomp]+X[9::Ncomp]+X[12::Ncomp]
#    in_phase_0 = np.dot(np.asarray(vacopt==0,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
#    phase=0
#    if in_phase_0>0.:
#        nu = min(daily_doses/in_phase_0,1.)*np.asarray(vacopt==phase,dtype=np.float64)
#    else:
#        nu = np.zeros(Ngp,dtype=np.float64)
#    if in_phase_0<daily_doses:
#        phase+=1
#        in_phase_1 = np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
#        if in_phase_1>0.:
#            nu[vacopt==phase] = min((daily_doses-in_phase_0)/in_phase_1,1.)
#        if in_phase_0+in_phase_1<daily_doses:
#            phase+=1
#            in_phase_2 = np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
#            if in_phase_2>0.:
#                nu[vacopt==phase] = min((daily_doses-in_phase_0-in_phase_1)/in_phase_2,1.)
#            if in_phase_0+in_phase_1+in_phase_2<daily_doses:
#                phase+=1
#                in_phase_3 = np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
#                if in_phase_3>0.:
#                    nu[vacopt==phase] = min((daily_doses-in_phase_0-in_phase_1-in_phase_2)/in_phase_3,1.)



    #check what the current phase is
    nr_of_ppl_vaccinated_and_not_yet_sympomatic = X[2::Ncomp]+X[5::Ncomp]+X[8::Ncomp]+X[11::Ncomp]+X[14::Ncomp]    
    if np.dot(np.asarray(vacopt==3,dtype=np.float64),nr_of_ppl_vaccinated_and_not_yet_sympomatic)>0:
        current_phase = 3
    elif np.dot(np.asarray(vacopt==2,dtype=np.float64),nr_of_ppl_vaccinated_and_not_yet_sympomatic)>0:
        current_phase = 2
    elif np.dot(np.asarray(vacopt==1,dtype=np.float64),nr_of_ppl_vaccinated_and_not_yet_sympomatic)>0:
        current_phase = 1
    else:
        current_phase = 0


    nr_of_ppl_to_be_vaccinated = X[::Ncomp]+X[3::Ncomp]+X[6::Ncomp]+X[9::Ncomp]+X[12::Ncomp]
    in_phase_0 = np.dot(np.asarray(vacopt==0,dtype=np.float64),nr_of_ppl_to_be_vaccinated)    
    phase=0
    if in_phase_0>0.:
        nu = min(daily_doses/in_phase_0,1.)*np.asarray(vacopt==phase,dtype=np.float64)
    else:
        nu = np.zeros(Ngp,dtype=np.float64)
    if in_phase_0<daily_doses or current_phase > 0:
        phase+=1
        in_phase_1 = np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
        if in_phase_1>0.:
            nu[vacopt==phase] = min((daily_doses-in_phase_0)/in_phase_1,1.)
        if in_phase_0+in_phase_1<daily_doses  or current_phase > 1:
            phase+=1
            in_phase_2 = np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
            if in_phase_2>0.:
                nu[vacopt==phase] = min((daily_doses-in_phase_0-in_phase_1)/in_phase_2,1.)
            if in_phase_0+in_phase_1+in_phase_2<daily_doses  or current_phase > 2:
                phase+=1
                in_phase_3 = np.dot(np.asarray(vacopt==phase,dtype=np.float64),nr_of_ppl_to_be_vaccinated)
                if in_phase_3>0.:
                    nu[vacopt==phase] = min((daily_doses-in_phase_0-in_phase_1-in_phase_2)/in_phase_3,1.)

    Xprime = np.zeros(Ngp*Ncomp,dtype=np.float64)
    
    total_C = np.sum(X[15::Ncomp]) + np.sum(X[16::Ncomp]) + np.sum(X[19::Ncomp])
    r = 1/(1+(midc/np.log10(total_C+1))**exponent)
    rv = r
    
    variant_transmissibility = overall_transmissibility[int(t)-0]
    
    for i in range(Ngp):
        j=i*Ncomp
        
        Xprime[j]=-(1-r)*X[j]*F[i]*variant_transmissibility - nu[i]*X[j] #susceptible, willing to vaccinate
        Xprime[j+1] =-(1-r)*X[j+1]*F[i]*variant_transmissibility  #susceptible, not willing to vaccinate
        Xprime[j+2] =-(1-rv)*(1-sigma[i])*X[j+2]*F[i]*variant_transmissibility + nu[i]*X[j] #susceptible, vaccinated
        
        Xprime[j+3] =(1-r)*X[j]*F[i]*variant_transmissibility-mu_E*X[j+3] - nu[i]*X[j+3] #expo, willing to vaccinate
        Xprime[j+4] =(1-r)*X[j+1]*F[i]*variant_transmissibility-mu_E*X[j+4] #expo, not willing to vaccinate
        Xprime[j+5] =(1-rv)*(1-sigma[i])*X[j+2]*F[i]*variant_transmissibility-mu_E*X[j+5] + nu[i]*X[j+3] #expo, vaccinated
        
        Xprime[j+6] =mu_E * (1-q[i]) * X[j+3] - mu_A * X[j+6] - nu[i]*X[j+6] #asympto, willing to vaccinate
        Xprime[j+7] =mu_E * (1-q[i]) * X[j+4] - mu_A * X[j+7] #asympto, not willing to vaccinate
        Xprime[j+8] =mu_E * (1-q[i]*(1-delta[i])) * X[j+5] - mu_A * X[j+8] + nu[i]*X[j+6] #asympto, vaccinated
        
        Xprime[j+9] =mu_A * X[j+6] - nu[i] * X[j+9] #asymptomatic recovered, willing to vaccinate
        Xprime[j+10]=mu_A * X[j+7] #asymptomatic recovered, not willing to vaccinate
        Xprime[j+11]=mu_A * X[j+8] + nu[i] * X[j+9] #asymptomatic recovered, vaccinated
        
        Xprime[j+12]=mu_E*q[i]*X[j+3] - mu_P*X[j+12] - nu[i]*X[j+12] #pre-clinical, willing to vaccinate
        Xprime[j+13]=mu_E*q[i]*X[j+4] - mu_P*X[j+13] #pre-clinical, not willing to vaccinate
        Xprime[j+14]=mu_E*q[i]*(1-delta[i])*X[j+5] - mu_P*X[j+14] + nu[i]*X[j+12] #pre-clinical, vaccinated
        
        Xprime[j+15]=mu_P*(X[j+12]+X[j+13]) - mu_C*X[j+15] #clinical, not vaccinated
        Xprime[j+16]=mu_P*X[j+14] - mu_C*X[j+16] #clinical, vaccinated
        
        Xprime[j+19]=mu_C*(X[j+15]+X[j+16]) - mu_Q*X[j+19] #clinical but no longer spreading, Q
        
        Xprime[j+17]=CFR[i]*mu_Q*(X[j+19]) #dead
        Xprime[j+18]=(1-CFR[i])*mu_Q*(X[j+19]) #recov
    return Xprime

@jit(nopython=True)#(nopython=True)
def RK4(func, X0, ts,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations): 
    """
    Runge Kutta 4 solver.
    """
    dt = ts[1] - ts[0]
    nt = len(ts)
    X  = np.zeros((nt, X0.shape[0]),dtype=np.float64)
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], ts[i],vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
        k2 = func(X[i] + dt/2. * k1, ts[i] + dt/2.,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
        k3 = func(X[i] + dt/2. * k2, ts[i] + dt/2.,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
        k4 = func(X[i] + dt    * k3, ts[i] + dt,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

#@jit('float64(float64[:])',nopython=True) 
@jit(nopython=True) 
def model_evaluation_param_est(param):
    beta = parm_to_beta(param[0:2])
    midc=param[2]
    exponent=param[3]**2
    
    initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q,hesitancy=hesitancy)
    Y=RK4(SYS_ODE_VAX_RK4,initial_values,ts,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
    
    deaths = Y[:,17::Ncomp][::2,:] 
    cum_deaths_model = np.sum(deaths,1) 
    cum_deaths_observed = cumdeaths[index_2020_10_13:] 
    
    cum_cases_model = np.sum((Y[:,15::Ncomp] + Y[:,16::Ncomp] + Y[:,17::Ncomp] + Y[:,18::Ncomp] +Y[:,19::Ncomp])[::2,:],1)
    cum_cases_observed = cumcases[index_2020_10_13:] 
    
    length = min(len(cum_deaths_model),len(cum_deaths_observed))
    cum_deaths_model_length_of_cum_deaths_observed = cum_deaths_model[:length]
    cum_deaths_observed=cum_deaths_observed[:length]
    cum_cases_model_length_of_cum_deaths_observed = cum_cases_model[:length]
    cum_cases_observed=cum_cases_observed[:length]
    
    reduction_factor = 0.02 #1/(cum_cases_model_length_of_cum_deaths_observed[-1]/cum_deaths_model_length_of_cum_deaths_observed[-1])
    
    cum_cases_observed =reduction_factor*cum_cases_observed
    cum_cases_model_length_of_cum_deaths_observed = reduction_factor*cum_cases_model_length_of_cum_deaths_observed
    
    weights = np.zeros(len(cum_deaths_observed))
    #weights[(len(cum_deaths_observed)-60):] = np.arange(60)+1
    #weights = np.sqrt(np.arange(len(cum_deaths_observed))+1.)
    weights = (np.arange(len(cum_deaths_observed))+1.)
    wSSE_deaths = np.dot(np.square(weights),np.square(np.subtract(cum_deaths_observed,cum_deaths_model_length_of_cum_deaths_observed)))
    wSSE_cases = np.dot(np.square(weights),np.square(np.subtract(cum_cases_observed,cum_cases_model_length_of_cum_deaths_observed)))
    wSSE = wSSE_deaths + wSSE_cases
    
    if np.isnan(wSSE):
        wSSE = 1.122e24
    
    # MSE = np.square(np.subtract(cum_deaths_observed,cum_deaths_model_length_of_cum_deaths_observed)).mean() 
    return wSSE


@jit(nopython=True) 
def model_evaluation_param_est_daily_fit(param):
    beta = parm_to_beta(param[0:2])
    midc=param[2]
    exponent=param[3]**2
    
    initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q,hesitancy=hesitancy)
    Y=RK4(SYS_ODE_VAX_RK4,initial_values,ts,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
    
    deaths = Y[:,17::Ncomp][::2,:] 
    cum_deaths_model = np.sum(deaths,1)     
    daily_deaths_model = cum_deaths_model[1:]-cum_deaths_model[:-1]
    
    cum_cases_model = np.sum((Y[:,15::Ncomp] + Y[:,16::Ncomp] + Y[:,17::Ncomp] + Y[:,18::Ncomp] +Y[:,19::Ncomp])[::2,:],1)
    daily_cases_model = cum_cases_model[1:]-cum_cases_model[:-1]
    
    daily_deaths_observed = filtered_daily_deaths[index_2020_10_13-1:] 
    daily_cases_observed = filtered_daily_cases[index_2020_10_13-1:] 
    
    length = min(len(daily_deaths_model),len(daily_deaths_observed))
    daily_deaths_model_length_of_daily_deaths_observed = daily_deaths_model[:length]
    daily_deaths_observed=daily_deaths_observed[:length]
    daily_cases_model_length_of_daily_deaths_observed = daily_cases_model[:length]
    daily_cases_observed=daily_cases_observed[:length]
    
    reduction_factor = 0.02 #1/(cum_cases_model_length_of_cum_deaths_observed[-1]/cum_deaths_model_length_of_cum_deaths_observed[-1])
    
    daily_cases_observed = reduction_factor*daily_cases_observed
    daily_cases_model_length_of_daily_deaths_observed = reduction_factor*daily_cases_model_length_of_daily_deaths_observed
    
    
    
    #weights = np.ones(length)
    #weights[(len(cum_deaths_observed)-60):] = np.arange(60)+1
    #weights = np.sqrt(np.arange(len(cum_deaths_observed))+1.)
    weights = np.arange(len(daily_cases_observed))+1.
    wSSE_deaths = np.dot(np.square(weights),np.square(np.subtract(daily_deaths_observed,daily_deaths_model_length_of_daily_deaths_observed)))
    wSSE_cases = np.dot(np.square(weights),np.square(np.subtract(daily_cases_observed,daily_cases_model_length_of_daily_deaths_observed)))
    wSSE = wSSE_deaths + wSSE_cases
    
    if np.isnan(wSSE):
        wSSE = 10e24
    
    # MSE = np.square(np.subtract(cum_deaths_observed,cum_deaths_model_length_of_cum_deaths_observed)).mean() 
    return wSSE

#initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q,hesitancy=hesitancy)   
ts = np.arange(-0, len(dates)-index_2020_10_13-0-1, dt) 
 
CDC_allocation = np.array([3,0,0,1,1,2,2,3,2,0,0,2,2,0,0,1,1],dtype=np.float64)
vacopt = CDC_allocation

#x=0.002
#q = 1-x*(mean_4agegp[-1]-mean_4agegp)
#x*=0.3/(1-np.dot(get_contact_matrix.N_4,q)/sum(get_contact_matrix.N_4))

if __name__ == '__main__':
    ## Genetic algorithm
    params = []
    sses_ga=[]
    for ite in range(nsim):
        ts = np.arange(-0, len(dates)-index_2020_10_13-0-1, dt) 
        varbound=np.array([[0,0.1],[0,0.01],[4,6],[1,4]])   
        algorithm_param = {'max_num_iteration':niter,\
                       'population_size': npop,\
                       'mutation_probability':0.1,\
                       'elit_ratio': 0.01,\
                       'crossover_probability': 0.5,\
                       'parents_portion': 0.3,\
                       'crossover_type':'uniform',\
                       'max_iteration_without_improv':None}
        model=ga(function=model_evaluation_param_est,dimension=4,variable_type='real',
            variable_boundaries=varbound,algorithm_parameters=algorithm_param)
        param_opt=model.run()
        param=model.output_dict['variable']
        sses_ga.append(model.output_dict['function'])
        params.append(param)

    # @jit(nopython=True, parallel=True)
    # def main(nsim):
    #     params = np.zeros((nsim,4),dtype=np.float64)
    #     sses_ga=np.zeros(nsim,dtype=np.float64)
    #     ts = np.arange(-0, len(dates)-index_2020_10_13-0-1, dt) 
    #     for ite in prange(nsim):
    #         varbound=np.array([[0,0.1],[0,0.01],[4,6],[1,4]])   
    #         algorithm_param = {'max_num_iteration':niter,\
    #                        'population_size': npop,\
    #                        'mutation_probability':0.1,\
    #                        'elit_ratio': 0.01,\
    #                        'crossover_probability': 0.5,\
    #                        'parents_portion': 0.3,\
    #                        'crossover_type':'uniform',\
    #                        'max_iteration_without_improv':None}
    #         model=ga(function=model_evaluation_param_est,dimension=4,variable_type='real',
    #             variable_boundaries=varbound,algorithm_parameters=algorithm_param)
    #         param_opt=model.run()
    #         param=model.output_dict['variable']
    #         sses_ga[ite] = model.output_dict['function']
    #         params[ite] = param  
    #     return  (params,sses_ga)
            
    # main(3)
        
#        beta = parm_to_beta(param[0:2])
#        midc=param[2]
#        exponent=param[3]**2
#        
#        number_of_daily_vaccinations = np.array(list(map(lambda x: get_vaccination_rates_new.vaccinations_on_day_t(x,hesitancy,2),range(500))),dtype=np.float64)
#
#        #mu_A=param[4] #used to be mu_C
#        ts = np.arange(-0, 365, dt)
#        initial_values=get_initial_values(Nsize,mu_A,mu_C,mu_Q,q,hesitancy=hesitancy)
#        Y = RK4(SYS_ODE_VAX_RK4,initial_values,ts,vacopt,beta,exponent,midc,mu_A,mu_C,mu_Q,q,sigma,delta,f_A,f_V,number_of_daily_vaccinations)
#        sol_by_compartment = np.zeros((ts.shape[0],Ncomp),dtype=np.float64)
#        for i in range(Ncomp):
#            sol_by_compartment[:,i]=np.sum(Y[:,i::Ncomp],axis=1) 
#        total_C=sol_by_compartment[:,15]+sol_by_compartment[:,16]
#        r = 1/(1+(midc/np.log10(total_C+1))**exponent)  
#
#        fig2, ((ax1), (ax2),(ax3),(ax4)) = plt.subplots(4, 1, figsize=(6,11)) 
#        
#        fig2.suptitle('\n'.join(list(map(str,map(lambda x: np.round(x,4),param))))+'\nSSE: '+str(sses_ga[-1]))
#        
#        ax1.plot(ts,sol_by_compartment[:,17],label='D')
#        ax1.plot(np.arange(-0, -0+len(cumdeaths[index_2020_10_13:])) ,cumdeaths[index_2020_10_13:],label='real D')
#        
#        ax2.plot(ts,sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19],label='C')
#        ax2.plot(np.arange(-0, -0+len(cumcases[index_2020_10_13:])) ,cumcases[index_2020_10_13:],label='real C')
#        
#        # f,ax = plt.subplots()
#        ax3.plot(np.arange(Ngp),beta,label='beta') 
#        
#        ax4.plot(total_C,r,label='contact reduction') 
#        
#        ax1.legend(loc='best') 
#        ax2.legend(loc='best')
#        ax3.legend(loc='best') 
#        ax4.legend(loc='best')  
#        plt.savefig('v%s_fitb_%s.pdf' % (version,str(ite)),bbox_inches = "tight") 



    args = [filename,nsim,niter,npop,SLURM_ID,vacopt,T,dt,hesitancy,option_vaccine_projection,mu_E,mu_P,mu_C,mu_Q,mu_A,q,sigma,delta,f_A,f_V,ve,last_day_for_fit]
    args = np.array(args,dtype='object')
    #to get args_names, run this ','.join(["'"+el+"'" for el in '''filename,nsim,niter,npop,SLURM_ID,vacopt,T,dt,hesitancy,additional_doses_per_day,mu_E,mu_P,mu_C,mu_Q,mu_A,q,sigma,delta,f_A,f_V,ve,last_day_for_fit'''.split(',')])
    args_names = ['filename','nsim','niter','npop','SLURM_ID','vacopt','T','dt','hesitancy','option_vaccine_projection','mu_E','mu_P','mu_C','mu_Q','mu_A','q','sigma','delta','f_A','f_V','ve','last_day_for_fit']

    f = open(output_folder+'optimal_param_v%s_nsim%i_niter%i_npop%i_ID%i.txt' % (version,nsim,niter,npop,SLURM_ID),'w')
    for ii in range(len(args_names)):#enumerate(zip(args,args_names)):
        if type(args[ii]) in [int,float,np.float64]:
            f.write(args_names[ii]+'\t'+str(args[ii])+'\n')    
        else:
            f.write(args_names[ii]+'\t'+'\t'.join(list(map(str,[el if type(el)!=float else round(el,6) for el in args[ii]])))+'\n')

    lines = ['\t'.join(['beta0','beta1','midC','sqrt(exponent)','SSE'])]
    for sse,param in zip(sses_ga,params):
        lines.append('\t'.join(list(map(str,param))) + '\t' + str(sse))
    f.write('\n'.join(lines))
    f.close()