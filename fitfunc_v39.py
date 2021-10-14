#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 13 13:57:09 2021

@author: ckadelka
"""

import numpy as np
import time
import sys
from numba import jit, prange
import parameter_est_39 as source2
import pandas as pd
import get_yll_and_mean_age
import get_vaccination_rates_new

version = '39'
output_folder = 'results/version%s/' % version

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 73#random.randint(0,100)

if len(sys.argv)>2:
    nsim = int(sys.argv[2])
else:
    nsim = 50#17500

if len(sys.argv)>3:
    option = int(sys.argv[3])
else:
    option = 1
    

dummy = SLURM_ID//100
data = pd.read_csv('v%s_GA_output_mod.csv' % version)
param = np.array(np.array(data)[dummy if dummy<7 else 0,6:6+4],dtype=np.float64)
beta = source2.parm_to_beta(param[0:2])
midc=param[2]
exponent=param[3]**2

hesitancy = 0.3
source2.f_A = 0.75 #CDC best estimate
source2.f_V=0.5 #big unknown, we'll vary it from 0 to 1
q17 = 0.85

if dummy<3:
    q17 = np.array([0.85,0.7,1])[dummy]
elif dummy<5:
    source2.f_A = np.array([0.25,1])[dummy-3]
elif dummy<7:
    source2.f_V = np.array([0,1])[dummy-5]
elif dummy<10:
    new_hesitancy = np.array([0.2,0.1,0])[dummy-7]
    
q = source2.q_based_on_q17(q17)

#source2.get_initial_values.recompile()
#source2.SYS_ODE_VAX_RK4.recompile()

@jit(nopython=True)#(nopython=True)#parallel=True)
def fitfunc_short(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta):
    vacopt = np.asarray(vacopt,dtype=np.float64)
    Ncomp = 20
    ts = np.arange(-0, 365+17, source2.dt)
    Y = source2.RK4(source2.SYS_ODE_VAX_RK4,initial_values,ts,vacopt,beta,exponent,midc,source2.mu_A,source2.mu_C,source2.mu_Q,q,sigma,delta,f_A,f_V,source2.number_of_daily_vaccinations)

    dead_per_group = Y[-1,17::Ncomp]
    cases_per_group = dead_per_group + Y[-1,15::Ncomp] + Y[-1,16::Ncomp] + Y[-1,18::Ncomp] + Y[-1,19::Ncomp]
    infections_per_group = source2.Nsize - Y[-1,0::Ncomp] - Y[-1,1::Ncomp] - Y[-1,2::Ncomp]

    return dead_per_group[0],np.sum(dead_per_group[1:9]),np.sum(dead_per_group[9:13]),np.sum(dead_per_group[13:]),cases_per_group[0],np.sum(cases_per_group[1:9]),np.sum(cases_per_group[9:13]),np.sum(cases_per_group[13:]),infections_per_group[0],np.sum(infections_per_group[1:9]),np.sum(infections_per_group[9:13]),np.sum(infections_per_group[13:])

def fitfunc_with_jump(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta,Tmax=382,new_hesitancy=0.1,time_index_jump=get_vaccination_rates_new.len_doses,full_output=False):
    vacopt = np.asarray(vacopt,dtype=np.float64)
    Ncomp = 20
    ts1 = np.arange(-0, time_index_jump, source2.dt)
    Y1 = source2.RK4(source2.SYS_ODE_VAX_RK4,initial_values,ts1,vacopt,beta,exponent,midc,source2.mu_A,source2.mu_C,source2.mu_Q,q,sigma,delta,f_A,f_V,source2.number_of_daily_vaccinations)

    conditions_at_jump = Y1[-1,:]
    proportion_newly_willing = 1-new_hesitancy/source2.hesitancy
    for i in [0,3,6,9,12]:
        number_of_people_to_move = conditions_at_jump[i+1::Ncomp]*proportion_newly_willing
        conditions_at_jump[i::Ncomp] += number_of_people_to_move
        conditions_at_jump[i+1::Ncomp] -= number_of_people_to_move

    ts2 = np.arange(time_index_jump, Tmax, source2.dt)
    Y2 = source2.RK4(source2.SYS_ODE_VAX_RK4,conditions_at_jump,ts2,vacopt,beta,exponent,midc,source2.mu_A,source2.mu_C,source2.mu_Q,q,sigma,delta,f_A,f_V,source2.number_of_daily_vaccinations)

    Y = np.r_[Y1,Y2]

    dead_per_group = Y[-1,17::Ncomp]
    cases_per_group = dead_per_group + Y[-1,15::Ncomp] + Y[-1,16::Ncomp] + Y[-1,18::Ncomp] + Y[-1,19::Ncomp]
    infections_per_group = source2.Nsize - Y[-1,0::Ncomp] - Y[-1,1::Ncomp] - Y[-1,2::Ncomp]
    
    if full_output:
        return Y
    else:
        return dead_per_group[0],np.sum(dead_per_group[1:9]),np.sum(dead_per_group[9:13]),np.sum(dead_per_group[13:]),cases_per_group[0],np.sum(cases_per_group[1:9]),np.sum(cases_per_group[9:13]),np.sum(cases_per_group[13:]),infections_per_group[0],np.sum(infections_per_group[1:9]),np.sum(infections_per_group[9:13]),np.sum(infections_per_group[13:])


def get_i1_to_i17(ID):
    def get_two_indices(dummy):
        counter = -1
        for index_smaller in range(4):
            for index_larger in range(index_smaller,4):
                counter+=1
                if counter==dummy:
                    break
            if counter==dummy:
                break    
        return (index_larger,index_smaller)
    
    indices = [0]*17
    indices[0] = int(ID/10**8)
    for i in range(8):
        ID = ID % (10**(8-i))
        dummy = int(ID / 10**(7-i))
        (indices[2*i+1],indices[2*i+2]) = get_two_indices(dummy)

    return indices

#def get_allocation_to_ID(allocation):
#    allocation = np.array(allocation,dtype=int)
#    desired_number = np.dot(5**np.arange(17-1,-1,-1),allocation)
#    ID_max = int(17.5*1e6)-1
#    ID_min = 0
#    ID = int((ID_max-ID_min)/2)
#    allo = get_i1_to_i17_which_satisfy_all_constraints(ID)
#    current_number = np.dot(5**np.arange(17-1,-1,-1),allo)
#    for ii in range(100):
#        if current_number < desired_number:
#            ID_max = ID
#        else:
#            ID_min = ID
#        ID = int((ID_max-ID_min)/2)
#        allo = get_i1_to_i17_which_satisfy_all_constraints(ID)
#        current_number = np.dot(5**np.arange(17-1,-1,-1),allo)
#        print(ID,ID_min,ID_max,allo)
#        
#        
#def get_ID_for_specific_allocation(allocation):
#    def x_greater_y(x,y):
#        for a,b in zip(x,y):
#            if a>b:
#                return 1
#            elif a<b:
#                return 0
#        return 0.5
#    allocation = np.array(allocation,dtype=int)
#    ID_max = int(17.5*1e6)-1
#    ID_min = 0
#    ID = int((ID_max-ID_min)/2)
#    allo = get_i1_to_i17_which_satisfy_all_constraints(ID)
#    for ii in range(100):
#        comparison = x_greater_y(allocation,allo)
#        print(ID,ID_min,ID_max,comparison,allo)
#        if comparison==1:
#            ID_min = ID
#            ID = int(np.round((ID_max+ID_min+2)/2))
#        elif comparison==0:
#            ID_max = ID
#            ID = int(np.round((ID_max+ID_min-2)/2))
#        else:
#            break
#        allo = get_i1_to_i17_which_satisfy_all_constraints(ID)
    
def satisfies_additional_constraint(indices):
    if indices[3] > indices[5]:
        return False
    elif indices[5] > indices[7]:
        return False
    elif indices[4] > indices[6]:
        return False
    elif indices[6] > indices[8]:
        return False
    elif indices[9] > indices[11]:
        return False  
    elif indices[10] > indices[12]:
        return False  
    elif indices[13] > indices[15]:
        return False  
    elif indices[14] > indices[16]:
        return False  
    elif len(set(indices))<4:
        return False
    else:
        return True

def get_i1_to_i4(ID):
    def get_two_indices(dummy):
        counter = -1
        for index_smaller in range(4):
            for index_larger in range(index_smaller,4):
                counter+=1
                if counter==dummy:
                    break
            if counter==dummy:
                break    
        return (index_larger,index_smaller)
    
    indices = [0]*4
    for i in range(2):
        ID = ID % (10**(2-i))
        dummy = int(ID / 10**(1-i))
        (indices[2*i],indices[2*i+1]) = get_two_indices(dummy)
    return indices

def get_i1_to_i8(ID):
    def get_two_indices(dummy):
        counter = -1
        for index_smaller in range(4):
            for index_larger in range(index_smaller,4):
                counter+=1
                if counter==dummy:
                    break
            if counter==dummy:
                break    
        return (index_larger,index_smaller)
    
    indices = [0]*8
    for i in range(4):
        ID = ID % (10**(4-i))
        dummy = int(ID / 10**(3-i))
        (indices[2*i],indices[2*i+1]) = get_two_indices(dummy)
    return indices

def satisfies_additional_constraint_4(indices):
    if indices[0] > indices[2]:
        return False
    elif indices[1] > indices[3]:
        return False
    else:
        return True
    
def satisfies_additional_constraint_8(indices):
    if indices[2] > indices[4]:
        return False
    elif indices[4] > indices[6]:
        return False
    elif indices[3] > indices[5]:
        return False
    elif indices[5] > indices[7]:
        return False
    else:
        return True

indices4 = [get_i1_to_i4(i) for i in range(100) if satisfies_additional_constraint_4(get_i1_to_i4(i))]
indices8 = [get_i1_to_i8(i) for i in range(10000) if satisfies_additional_constraint_8(get_i1_to_i8(i))]
def get_i1_to_i17_which_satisfy_all_constraints(ID): #ID= int in [0,4*1750*50*50)
    indices = [int(ID / (1750*50*50))]
    ID = ID % (1750*50*50)
    indices.extend(indices8[int(ID / (50*50))])
    ID = ID % (50*50)
    indices.extend(indices4[int(ID / (50))])
    ID = ID % (50)
    indices.extend(indices4[ID])    
    return indices

def get_i1_to_i17_which_satisfy_all_constraints_numba(ID): #ID= int in [0,4*1750*50*50)
    indices = [int(ID / (1750*50*50))]
    ID = ID % (1750*50*50)
    indices.extend(indices8[int(ID / (50*50))])
    ID = ID % (50*50)
    indices.extend(indices4[int(ID / (50))])
    ID = ID % (50)
    indices.extend(indices4[ID])    
    return np.array(indices,dtype=np.float64) 

def get_ID_for_feasible_allocation(allocation):
    index1 = allocation[0]
    index2 = list(np.dot(indices8,5**np.arange(8-1,-1,-1))).index(np.dot(allocation[1:9],5**np.arange(8-1,-1,-1)))
    index3 = list(np.dot(indices4,5**np.arange(4-1,-1,-1))).index(np.dot(allocation[9:13],5**np.arange(4-1,-1,-1)))
    index4 = list(np.dot(indices4,5**np.arange(4-1,-1,-1))).index(np.dot(allocation[13:17],5**np.arange(4-1,-1,-1)))
    return int((1750*50*50)*index1 + 50*50*index2 + 50*index3 + index4)

TIME = time.time()
CDC_allocation = np.array([3,0,0,1,1,2,2,3,2,0,0,2,2,0,0,1,1],dtype=np.float64)


if __name__ == '__main__':
    TIME = time.time()

    dummy = SLURM_ID//100
    data = pd.read_csv('v%s_GA_output_mod.csv' % version)
    param = np.array(np.array(data)[dummy,6:6+4],dtype=np.float64)
    beta = source2.parm_to_beta(param[0:2])
    midc=param[2]
    exponent=param[3]**2
    
    hesitancy = 0.3
    source2.f_A = 0.75 #CDC best estimate
    source2.f_V=0.5 #big unknown, we'll vary it from 0 to 1
    q17 = 0.85
    
    if dummy<3:
        q17 = np.array([0.85,0.7,1])[dummy]
    elif dummy<5:
        source2.f_A = np.array([0.25,1])[dummy-3]
    elif dummy<7:
        source2.f_V = np.array([0,1])[dummy-5]
    q = source2.q_based_on_q17(q17)    

    source2.get_initial_values.recompile()
    fitfunc_short.recompile()
    source2.SYS_ODE_VAX_RK4.recompile()
    source2.RK4.recompile()

    vacopts = []
    vacopts_ids = []
    res_from_runs = []
    initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)    
    for ID in range(SLURM_ID*nsim,(SLURM_ID+1)*nsim):
        vacopt = np.array(get_i1_to_i17_which_satisfy_all_constraints(ID))
        res_from_run = fitfunc_short(np.array(vacopt),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
        res_from_runs.append(res_from_run)
        vacopts.append(vacopt)
        vacopts_ids.append(ID)
    vacopts = np.array(vacopts)    
    vacopts_ids = np.array(vacopts_ids)
    res_from_runs = np.array(res_from_runs)

#    vacopts = np.zeros((nsim,17),dtype=np.float64)
#    vacopts_ids = np.arange(nsim*SLURM_ID,nsim*(SLURM_ID+1),dtype=np.int16)
#    for i in prange(nsim):
#        ID = int(SLURM_ID*nsim + i)
#        vacopts[i] = get_i1_to_i17_which_satisfy_all_constraints_numba(ID)
#    initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)    
#
#    @jit(nopython=True,parallel=True,fastmath=True)        
#    def main():
#        res_from_runs = np.zeros((nsim,12),dtype=np.float64)
#        for i in prange(nsim):
#            res_from_runs[i] = fitfunc_short(vacopts[i],initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
#        return res_from_runs
#    res_from_runs = main()
#        
    f = open(output_folder+'output_nsim%i_SLURM_ID%i_hes%i_fA%i_fV%i_q17%i.txt' % (nsim,SLURM_ID,int(source2.hesitancy*100),int(source2.f_A*100),int(source2.f_V*100),int(q17*100)) ,'w')
    f.write('filename\t'+filename+'\n')
    f.write('SLURM_ID\t'+str(SLURM_ID)+'\n')
    f.write('nsim\t'+str(nsim)+'\n')
    f.write('time in seconds\t'+str(int(time.time()-TIME))+'\n')
    f.write('allocation ID\t'+'\t'.join(list(map(str,vacopts_ids)))+'\n')
    #for i in range(17):
    #    f.write('phase of group '+str(i+1)+'\t'+'\t'.join(list(map(str,vacopts[:,i])))+'\n')
    counter=0
    for name in ['deaths','cases','infections']:
        for j in range(4):
            f.write(name+'_in_age_group_'+str(j+1)+'\t'+'\t'.join(list(map(lambda x: str(float(x)),res_from_runs[:,counter])))+'\n')
            counter+=1
    f.close()
    print(time.time()-TIME)


#import matplotlib.pyplot as plt
#Tmax=365+18+100
#hesitancies = np.arange(0,0.25,0.01)
#deaths = list(map(lambda x: sum(fitfunc_with_jump(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,x)[:4]),
#                  hesitancies))
#plt.plot(deaths)
        
#import matplotlib.pyplot as plt
#for f_A_index in range(3):
#    for f_V_index in range(3):            
#        TIME = time.time()
#        row_index = f_A_index*3+f_V_index
#        
#        source2.f_A = np.array([0.25,0.75,1])[f_A_index]
#        source2.f_V = np.array([0,0.5,1])[f_V_index]
#
#        param = np.array(np.array(data)[row_index,5:5+5],dtype=np.float64)
#        beta = source2.parm_to_beta(param[:2])
#        q = source2.parm_to_q(param[2])
#        midc = param[3]
#        exponent = param[4]**2    
#
#        vacopts = []
#        vacopts_ids = []
#        res_from_runs = []
#        total_cases = []
#        total_infections = []
#        ylls = []
#        max_infections = []
#        
#        initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)    
#
#        Tmax=365+18
#        Ncomp=20
#        hesitancies = np.arange(0.1,0.25+1e-7,0.005)
#        new_daily_infections_at_Tmax = []
#        new_daily_deaths_at_Tmax = []
#        total_deaths = []
#        total_cases = []
#        for i,hes in enumerate(hesitancies):
#            Y = fitfunc_with_jump(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,full_output=True)
#            new_daily_deaths_at_Tmax.append(np.sum(Y[-1,17::Ncomp])-np.sum(Y[-3,17::Ncomp]))
#            new_daily_infections_at_Tmax.append(np.sum(Y[-3,0::Ncomp])+np.sum(Y[-3,1::Ncomp])+np.sum(Y[-3,2::Ncomp])-(np.sum(Y[-1,0::Ncomp])+np.sum(Y[-1,1::Ncomp])+np.sum(Y[-1,2::Ncomp])))
#            total_deaths.append(np.sum(Y[-1,17::Ncomp]))
#            total_cases.append(np.sum(Y[-1,15::Ncomp])+np.sum(Y[-1,16::Ncomp])+np.sum(Y[-1,17::Ncomp])+np.sum(Y[-1,18::Ncomp])+np.sum(Y[-1,19::Ncomp]))            
#        
#        f,ax=plt.subplots(figsize=(3,3))
#        ax.plot(hesitancies,new_daily_deaths_at_Tmax,color='blue',label='deaths')
#        ax.set_xlabel('hesitancy')
#        ax.set_ylabel('daily new deaths (end of 2021)')
#        ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
#        ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
#        ax.set_ylim([0,ax.get_ylim()[1]])
#        ax2=ax.twinx()
#        ax2.plot([0.1,0.1],[0,0],color='blue',label='deaths')
#        ax2.plot(hesitancies,new_daily_infections_at_Tmax,color='orange',label='infections')
#        ax2.set_ylabel('daily new infections (end of 2021)')
#        ax2.legend(loc='best',frameon=False)
#        plt.savefig('daily_new_deaths_infections_at_end_%i%i.pdf' % (f_A_index,f_V_index),bbox_inches = "tight")
#
#        f,ax=plt.subplots(figsize=(3,3))
#        ax.plot(hesitancies,total_deaths,color='blue',label='deaths')
#        ax.set_xlabel('hesitancy')
#        ax.set_ylabel('total deaths (end of 2021)')
#        ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
#        ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
#        ax2=ax.twinx()
#        ax2.plot(hesitancies,total_cases,color='orange',label='cases')
#        [y1,y2] = ax2.get_ylim()
#        ax2.plot([0.1,0.1],[0,0],color='blue',label='deaths')
#        ax2.set_ylim([y1,y2])
#        ax2.set_ylabel('total cases (end of 2021)')
#        ax2.legend(loc='best',frameon=False)
#        plt.savefig('total_deaths_cases_at_end_%i%i.pdf' % (f_A_index,f_V_index),bbox_inches = "tight")
#


