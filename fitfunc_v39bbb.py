#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 13 13:57:09 2021

@author: ckadelka
"""

import numpy as np
import time
import sys
from numba import jit#, prange
import parameter_est_39 as source2
import pandas as pd
#import get_yll_and_mean_age
import get_vaccination_rates_new

version = '39'
output_folder = 'results/version%s/' % version

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 1#random.randint(0,100)
#SLURM_ID += 1200

if len(sys.argv)>2:
    nsim = int(sys.argv[2])
else:
    nsim = 50#17500

if len(sys.argv)>3:
    option = int(sys.argv[3])
else:
    option = 1
    
#if option==2:
#    ids_redo = [21, 191, 192, 295, 433, 489, 490, 492, 493, 571, 628, 629, 630, 649, 700, 861]
#    SLURM_ID = ids_redo[SLURM_ID]

#SLURM_ID += 1000

data = pd.read_csv('v%s_GA_output_mod.csv' % version)

dummy = SLURM_ID//100
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
    hesitancy = np.array([0.2,0.1,0])[dummy-7]
elif dummy==10:
    source2.delta = np.zeros(17,dtype=np.float64)
    source2.sigma = 0.9*np.ones(17,dtype=np.float64)
elif dummy==11:
    source2.sigma = np.zeros(17,dtype=np.float64)
    source2.delta = 0.9*np.ones(17,dtype=np.float64)
elif dummy==12:
    source2.sigma = 0.51949385*np.ones(17,dtype=np.float64)
    source2.delta = 0.79188612*np.ones(17,dtype=np.float64)
elif dummy==13:
    source2.sigma = 0.79188612*np.ones(17,dtype=np.float64)
    source2.delta = 0.51949385*np.ones(17,dtype=np.float64)





q = source2.q_based_on_q17(q17)
source2.hesitancy = hesitancy

# PLOT=False
# if PLOT:
#     f,ax = plt.subplots()
#     colors = cm.tab10
#     lss = ['-','--',':']
#     for dummy in range(3):
#         param = np.array(np.array(data)[dummy if dummy<7 else 0,6:6+4],dtype=np.float64)
#         beta = source2.parm_to_beta(param[0:2])
#         midc=param[2]
#         exponent=param[3]**2
        
#         hesitancy = 0.3
#         source2.f_A = 0.75 #CDC best estimate
#         source2.f_V=0.5 #big unknown, we'll vary it from 0 to 1
#         q17 = 0.85
#         q17 = np.array([0.85,0.7,1])[dummy]
#         q = source2.q_based_on_q17(q17)
        
#         for i,vec in enumerate([beta,q]):
#             ax.plot(vec,color=colors(i),ls=lss[dummy])

# f,ax = plt.subplots(figsize=(4,3))
# width=0.28
# lss = ['x:','o:','D:']
# for i in range(3):
#     q17i = np.array([0.7,0.85,1])[i]
#     qi = source2.q_based_on_q17(q17i)
#     #ax.bar(np.arange(4)-1.5*width+i*width,[qi[0],qi[1],qi[9],qi[13]],width=width)
#     ax.plot([qi[0],qi[1],qi[9],qi[13]],lss[i],label=r'$q_{75+} = $'+str(q17i))
#     ax.text(3,q17i-0.02,str(int(q17i*100))+'%',va='top',ha='center')
# ax.set_xticks(range(4))
# ax.set_xticklabels(['0-15','16-64','65-74','75+'])
# ax.set_xlabel('age')
# ax.set_ylabel('probability of symptomatic infection')
# ax.set_yticklabels([str(int(el*100))+'%' for el in ax.get_yticks()])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)    
# plt.savefig('qs.pdf',bbox_inches = "tight")
    
    




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

@jit(nopython=True)#(nopython=True)#parallel=True)
def fitfunc(vacopt,initial_values,beta,q,midc,exponent,f_A,f_V,sigma,delta):
    vacopt = np.asarray(vacopt,dtype=np.float64)
    ts = np.arange(-0, 365+17, source2.dt)
    Y = source2.RK4(source2.SYS_ODE_VAX_RK4,initial_values,ts,vacopt,beta,exponent,midc,source2.mu_A,source2.mu_C,source2.mu_Q,q,sigma,delta,f_A,f_V,source2.number_of_daily_vaccinations)
    return Y

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
        hesitancy = np.array([0.2,0.1,0])[dummy-7]
    elif dummy==10:
        source2.delta = np.zeros(17,dtype=np.float64)
        source2.sigma = 0.9*np.ones(17,dtype=np.float64)
    elif dummy==11:
        source2.sigma = np.zeros(17,dtype=np.float64)
        source2.delta = 0.9*np.ones(17,dtype=np.float64)
    elif dummy==12:
        source2.sigma = 0.51949385*np.ones(17,dtype=np.float64)
        source2.delta = 0.79188612*np.ones(17,dtype=np.float64)
    elif dummy==13:
        source2.sigma = 0.79188612*np.ones(17,dtype=np.float64)
        source2.delta = 0.51949385*np.ones(17,dtype=np.float64)



    q = source2.q_based_on_q17(q17)
    source2.hesitancy = hesitancy
    
    source2.get_initial_values.recompile()
    fitfunc_short.recompile()
    source2.SYS_ODE_VAX_RK4.recompile()
    source2.RK4.recompile()

    vacopts = []
    vacopts_ids = []
    res_from_runs = []
    initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)    
    for ID in range(SLURM_ID*nsim,(SLURM_ID+1)*nsim):
        ID = ID%17500000
        vacopt = np.array(get_i1_to_i17_which_satisfy_all_constraints(ID))
        #res_from_run = fitfunc_with_jump(np.array(vacopt),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,new_hesitancy=new_hesitancy)
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
    f = open(output_folder+'output_nsim%i_SLURM_ID%i_hes%i_fA%i_fV%i_q17%i.txt' % (nsim,SLURM_ID,int(hesitancy*100),int(source2.f_A*100),int(source2.f_V*100),int(q17*100)) ,'w')
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
#
#        import matplotlib.pyplot as plt
#
#
#        Tmax=365+17
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
#        #plt.savefig('daily_new_deaths_infections_at_end_%i%i.pdf' % (f_A_index,f_V_index),bbox_inches = "tight")
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
#        #plt.savefig('total_deaths_cases_at_end_%i%i.pdf' % (f_A_index,f_V_index),bbox_inches = "tight")
#
#
#
#
#
#        Tmax=365+18
#        Ncomp=20
#        q17s = np.arange(0.7,1,0.005)
#        new_daily_infections_at_Tmax = []
#        new_daily_deaths_at_Tmax = []
#        total_deaths = []
#        total_cases = []
#        CDC_allocation_mod = CDC_allocation.copy()
#        CDC_allocation_mod[0]=2
#        for i,q17 in enumerate(q17s):
#            q = source2.q_based_on_q17(q17)
#
#            Y_normal = fitfunc_with_jump(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,full_output=True)
#            Y_mod = fitfunc_with_jump(np.array(CDC_allocation_mod),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,full_output=True)
#            
#            Y = Y_normal - Y_mod
#            
#            new_daily_deaths_at_Tmax.append(np.sum(Y[-1,17::Ncomp])-np.sum(Y[-3,17::Ncomp]))
#            new_daily_infections_at_Tmax.append(np.sum(Y[-3,0::Ncomp])+np.sum(Y[-3,1::Ncomp])+np.sum(Y[-3,2::Ncomp])-(np.sum(Y[-1,0::Ncomp])+np.sum(Y[-1,1::Ncomp])+np.sum(Y[-1,2::Ncomp])))
#            total_deaths.append(np.sum(Y[-1,17::Ncomp]))
#            total_cases.append(np.sum(Y[-1,15::Ncomp])+np.sum(Y[-1,16::Ncomp])+np.sum(Y[-1,17::Ncomp])+np.sum(Y[-1,18::Ncomp])+np.sum(Y[-1,19::Ncomp]))            
#
#        f,ax=plt.subplots(figsize=(3,3))
#        ax.plot(q17s,total_deaths,color='blue',label='deaths')
#        ax.set_xlabel('q17')
#        ax.set_ylabel('total deaths (end of 2021)')
#        ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
#        ax2=ax.twinx()
#        ax2.plot(q17s,total_cases,color='orange',label='cases')
#        [y1,y2] = ax2.get_ylim()
#        ax2.plot([0.7,0.7],[0,0],color='blue',label='deaths')
#        ax2.set_ylim([y1,y2])
#        ax2.set_ylabel('total cases (end of 2021)')
#        ax2.legend(loc='best',frameon=False)
#        
#        
#        
#        
#        #changed hesitance from start
#        import matplotlib.pyplot as plt
#
#        
#        Tmax=365+17
#        Ncomp=20
#        hesitancies = np.arange(0,0.3+1e-7,0.01)
#        new_daily_infections_at_Tmax = []
#        new_daily_deaths_at_Tmax = []
#        total_deaths = []
#        total_cases = []
#        for i,hes in enumerate(hesitancies):
#            Y = fitfunc_with_jump(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,time_index_jump=1,full_output=True)
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
#        #plt.savefig('daily_new_deaths_infections_at_end_%i%i.pdf' % (f_A_index,f_V_index),bbox_inches = "tight")
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
#        #plt.savefig('total_deaths_cases_at_end_%i%i.pdf' % (f_A_index,f_V_index),bbox_inches = "tight")
#        
#        
#        
import matplotlib.pyplot as plt
from matplotlib import cm

optimal_deaths_allocation = np.array([4, 1, 1, 2, 2, 3, 3, 4, 3, 2, 1, 3, 3, 3, 1, 3, 2],dtype=np.float64)
optimal_deaths_allocation2 = np.array([4, 1, 1, 2, 2, 3, 3, 4, 3, 2, 2, 3, 3, 3, 1, 3, 2],dtype=np.float64)
optimal_deaths_allocation3 = np.array([4, 1, 1, 2, 2, 3, 3, 4, 3, 2, 2,4, 3, 3, 1, 3, 2],dtype=np.float64)

optimal_cases_allocation = np.array([4, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 4, 4],dtype=np.float64)
optimal_cases_allocation2 = np.array([4, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 4, 3, 4, 4],dtype=np.float64)
optimal_cases_allocation3 = np.array([4, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 4, 4, 3, 3, 4, 4],dtype=np.float64)

optimal_yll_allocation = np.array([4, 1, 1, 2, 2, 3, 2, 4, 3, 3, 2, 4, 3, 3, 2, 4, 3],dtype=np.float64)
optimal_yll_allocation2 = np.array([4, 1, 1, 2, 2, 3, 3, 4, 3, 3, 2, 4, 3, 3, 2, 4, 3],dtype=np.float64)

Tmax=365+17
Ncomp=20
hesitancies = np.arange(0,0.3+1e-7,0.01)
total_deaths = []
total_deaths_best = []
total_cases_best = []
total_cases = []
total_yll = []
total_yll_best = []
average_years_of_life_left = np.array([71.45519038, 41.66010998, 16.82498872,  7.77149779])
source2.hesitancy = 0.1
#for i,hes in enumerate(hesitancies):
#    Y = fitfunc_with_jump(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,time_index_jump=1,full_output=False)
#    total_deaths.append(sum(Y[:4])/1e3)
#    total_cases.append(sum(Y[4:8])/1e6)            
#    total_yll.append(np.dot(Y[:4],average_years_of_life_left)/1e6)            
#    Y = fitfunc_with_jump(np.array(optimal_deaths_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,time_index_jump=1,full_output=False)
#    total_deaths_best.append(sum(Y[:4])/1e3)
#    Y = fitfunc_with_jump(np.array(optimal_cases_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,time_index_jump=1,full_output=False)
#    total_cases_best.append(sum(Y[4:8])/1e6)            
#    Y = fitfunc_with_jump(np.array(optimal_yll_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta,Tmax,hes,time_index_jump=1,full_output=False)
#    total_yll_best.append(np.dot(Y[:4],average_years_of_life_left)/1e6)            

for i,hes in enumerate(hesitancies):
    initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=hes,START_DATE_INDEX=source2.index_2020_10_13)    
    Y = fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_deaths.append(sum(Y[:4])/1e3)
    total_cases.append(sum(Y[4:8])/1e6)            
    total_yll.append(np.dot(Y[:4],average_years_of_life_left)/1e6)            
    Y = fitfunc_short(np.array(optimal_deaths_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y2 = fitfunc_short(np.array(optimal_deaths_allocation2)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_deaths_best.append(min(sum(Y[:4]),sum(Y2[:4]))/1e3)
    Y = fitfunc_short(np.array(optimal_cases_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y2 = fitfunc_short(np.array(optimal_cases_allocation2)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y3 = fitfunc_short(np.array(optimal_cases_allocation3)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_cases_best.append(min([sum(Y[4:8]),sum(Y2[4:8]),sum(Y3[4:8])])/1e6)            
    Y = fitfunc_short(np.array(optimal_yll_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y2 = fitfunc_short(np.array(optimal_yll_allocation2)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_yll_best.append(min(np.dot(Y[:4],average_years_of_life_left),np.dot(Y2[:4],average_years_of_life_left))/1e6)            
    
    
f,ax=plt.subplots(figsize=(3,3))
ax.plot(hesitancies,total_deaths,color=cm.tab10(1),label='deaths')
ax.plot(hesitancies,total_deaths_best,color=cm.tab10(0),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total deaths [thousands]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
ax2=ax.twinx()
ax2.plot(hesitancies,[(el1/el2-1)*100 for el1,el2 in zip(total_deaths,total_deaths_best)],'k--',label='deaths')
ax2.set_xlabel('hesitancy')
ax2.set_ylabel('% difference in deaths')
ax2.set_xticklabels(['%i%%' % int(el*100) for el in ax2.get_xticks()])
ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
ax2.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
plt.savefig('v%s_deaths_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")          

f,ax=plt.subplots(figsize=(3,3))
ax.plot(hesitancies,total_cases,color=cm.tab10(1),label='cases')
ax.plot(hesitancies,total_cases_best,color=cm.tab10(0),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total cases [millions]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
ax2=ax.twinx()
ax2.plot(hesitancies,[(el1/el2-1)*100 for el1,el2 in zip(total_cases,total_cases_best)],'k--',label='deaths')
ax2.set_xlabel('hesitancy')
ax2.set_ylabel('% difference in cases')
ax2.set_xticklabels(['%i%%' % int(el*100) for el in ax2.get_xticks()])
ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
ax2.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
plt.savefig('v%s_cases_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")
   
f,ax=plt.subplots(figsize=(3,3))
ax.plot(hesitancies,total_yll,color=cm.tab10(1),label='YLL')
ax.plot(hesitancies,total_yll_best,color=cm.tab10(0),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('years of life lost [millions]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
ax2=ax.twinx()
ax2.plot(hesitancies,[(el1/el2-1)*100 for el1,el2 in zip(total_yll,total_yll_best)],'k--',label='deaths')
ax2.set_xlabel('hesitancy')
ax2.set_ylabel('% difference in YLL')
ax2.set_xticklabels(['%i%%' % int(el*100) for el in ax2.get_xticks()])
ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
ax2.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
plt.savefig('v%s_yll_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")

#                                
#        



hesitancies = np.arange(0,0.3+1e-7,0.005)

total_deaths = []
total_deaths_best = []
total_cases_best = []
total_cases = []
total_yll = []
total_yll_best = []
argmins_deaths = []
argmins_cases = []
argmins_yll = []
for i,hes in enumerate(hesitancies):
    initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=hes,START_DATE_INDEX=source2.index_2020_10_13)    
    Y = fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_deaths.append(sum(Y[:4])/1e3)
    total_cases.append(sum(Y[4:8])/1e6)            
    total_yll.append(np.dot(Y[:4],average_years_of_life_left)/1e6)            
    Y = fitfunc_short(np.array(optimal_deaths_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y2 = fitfunc_short(np.array(optimal_deaths_allocation2)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y3 = fitfunc_short(np.array(optimal_deaths_allocation3)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_deaths_best.append(min([sum(Y[:4]),sum(Y2[:4]),sum(Y3[:4])])/1e3)
    argmins_deaths.append(np.argmin([sum(Y[:4]),sum(Y2[:4]),sum(Y3[:4])]))
    Y = fitfunc_short(np.array(optimal_cases_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y2 = fitfunc_short(np.array(optimal_cases_allocation2)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y3 = fitfunc_short(np.array(optimal_cases_allocation3)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_cases_best.append(min([sum(Y[4:8]),sum(Y2[4:8]),sum(Y3[4:8])])/1e6)            
    argmins_cases.append(np.argmin([sum(Y[4:8]),sum(Y2[4:8]),sum(Y3[4:8])]))            
    Y = fitfunc_short(np.array(optimal_yll_allocation)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    Y2 = fitfunc_short(np.array(optimal_yll_allocation2)-1,initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    total_yll_best.append(min(np.dot(Y[:4],average_years_of_life_left),np.dot(Y2[:4],average_years_of_life_left))/1e6)            
    argmins_yll.append(np.argmin([np.dot(Y[:4],average_years_of_life_left),np.dot(Y2[:4],average_years_of_life_left)]))  
argmins_deaths = np.array(argmins_deaths)
argmins_cases = np.array(argmins_cases)
argmins_yll = np.array(argmins_yll)

colorstarter = [1,19,9]

SHOW_DIFFERENCES=True
ABSOLUTE_DIFFERENCES = False
zipp = zip([total_deaths_best,total_yll_best,total_cases_best],
           [total_deaths,total_yll,total_cases],
           ['total deaths\n[thousands]','years of life lost\n[millions]','total cases\n[millions]'],
           ['D','Y','C'],
           [argmins_deaths,argmins_yll,argmins_cases])
for jj,(total_best,total,ylabel,letter,argmins) in enumerate(zipp):
    f,ax=plt.subplots(figsize=(3,1.5))
    ax.plot(hesitancies,total_best,color='k')
    ax.plot(hesitancies,total,color=cm.tab10(3))
    ax.set_xlabel('hesitancy')
    ax.set_ylabel(ylabel)
    [y1,y2]=ax.get_ylim()
    for ii in range(max(argmins)+1):
        xs = hesitancies[np.where(argmins==ii)[0][0]:np.where(argmins==ii)[0][-1]+1]
        try:
            print([letter,min(xs),max(xs),ii])
        except:
            pass
        ax.fill_between(xs,[0]*len(xs),1e8*xs,color=cm.tab20b(colorstarter[jj]+ii))
        ax.text(np.mean(xs),y1+0.9*(y2-y1),letter+str(ii+1),ha='center',va='center')
    ax.set_ylim([y1,y2])
    ax.set_xlim([min(hesitancies),max(hesitancies)])
    ax.set_xticks([0,0.1,0.2,0.3])
    ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if SHOW_DIFFERENCES:
        ax2=ax.twinx()
        ax2.plot(hesitancies,[el1-el2 if ABSOLUTE_DIFFERENCES else (el1/el2-1)*100 for el1,el2 in zip(total,total_best)],'k:')
        ax2.set_ylabel('difference' if ABSOLUTE_DIFFERENCES else '% difference')
        if not ABSOLUTE_DIFFERENCES:
            ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
        ax2.spines['top'].set_visible(False)
        
    plt.savefig('v%s_%s_optimal_vs_CDC_for_different_hesitancy%s.pdf' % (version,letter,('_absolute_differences' if ABSOLUTE_DIFFERENCES else '_percent_differences') if SHOW_DIFFERENCES else ''),bbox_inches = "tight")          







    
f,ax=plt.subplots(figsize=(3,1.8))
ax.plot(hesitancies,total_deaths_best,color='k',label='deaths')
ax.plot(hesitancies,total_deaths,color=cm.tab10(3),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total deaths\n[thousands]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
[y1,y2]=ax.get_ylim()
for ii in range(max(argmins_deaths)+1):
    xs = hesitancies[np.where(argmins_deaths==ii)[0][0]:np.where(argmins_deaths==ii)[0][-1]]
    ax.fill_between(xs,[0]*len(xs),1e8*xs,color=cm.tab20b(colorstarter[0]+ii))
    ax.text(np.mean(xs),y1+0.93*(y2-y1),'D'+str(ii+1),ha='center',va='center')
ax.set_ylim([y1,y2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('v%s_deaths_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")          



f,ax=plt.subplots(figsize=(3,1.8))
ax.plot(hesitancies,total_deaths_best,color='k',label='deaths')
ax.plot(hesitancies,total_deaths,color=cm.tab10(3),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total deaths\n[thousands]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
[y1,y2]=ax.get_ylim()
for ii in range(max(argmins_deaths)+1):
    xs = hesitancies[np.where(argmins_deaths==ii)[0][0]:np.where(argmins_deaths==ii)[0][-1]]
    ax.fill_between(xs,[0]*len(xs),1e8*xs,color=cm.tab20b(colorstarter[0]+ii))
    ax.text(np.mean(xs),y1+0.93*(y2-y1),'D'+str(ii+1),ha='center',va='center')
ax.set_ylim([y1,y2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('v%s_deaths_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")          

colorstarter = [1,9,17]
f,ax=plt.subplots(figsize=(3,1.8))
ax.plot(hesitancies,total_deaths_best,color='k',label='deaths')
ax.plot(hesitancies,total_deaths,color=cm.tab10(3),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total deaths\n[thousands]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
[y1,y2]=ax.get_ylim()
for ii in range(max(argmins_deaths)+1):
    xs = hesitancies[np.where(argmins_deaths==ii)[0][0]:np.where(argmins_deaths==ii)[0][-1]]
    ax.fill_between(xs,[0]*len(xs),1e8*xs,color=cm.tab20b(colorstarter[0]+ii))
    ax.text(np.mean(xs),y1+0.93*(y2-y1),'D'+str(ii+1),ha='center',va='center')
ax.set_ylim([y1,y2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('v%s_yll_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")          

colorstarter = [1,9,17]
f,ax=plt.subplots(figsize=(3,1.8))
ax.plot(hesitancies,total_deaths_best,color='k',label='deaths')
ax.plot(hesitancies,total_deaths,color=cm.tab10(3),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total deaths\n[thousands]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
[y1,y2]=ax.get_ylim()
for ii in range(max(argmins_deaths)+1):
    xs = hesitancies[np.where(argmins_deaths==ii)[0][0]:np.where(argmins_deaths==ii)[0][-1]]
    ax.fill_between(xs,[0]*len(xs),1e8*xs,color=cm.tab20b(colorstarter[0]+ii))
    ax.text(np.mean(xs),y1+0.93*(y2-y1),'D'+str(ii+1),ha='center',va='center')
ax.set_ylim([y1,y2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('v%s_cases_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")          

    
    
    
    
    
    
    
    
    
    
    
f,ax=plt.subplots(figsize=(3,3))
ax.plot(hesitancies,total_deaths,color=cm.tab10(1),label='deaths')
ax.plot(hesitancies,total_deaths_best,color=cm.tab10(0),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total deaths [thousands]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
ax2=ax.twinx()
ax2.plot(hesitancies,[(el1/el2-1)*100 for el1,el2 in zip(total_deaths,total_deaths_best)],'k--',label='deaths')
ax2.set_xlabel('hesitancy')
ax2.set_ylabel('% difference in deaths')
ax2.set_xticklabels(['%i%%' % int(el*100) for el in ax2.get_xticks()])
ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
ax2.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
plt.savefig('v%s_deaths_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")          

f,ax=plt.subplots(figsize=(3,3))
ax.plot(hesitancies,total_cases,color=cm.tab10(1),label='cases')
ax.plot(hesitancies,total_cases_best,color=cm.tab10(0),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('total cases [millions]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
ax2=ax.twinx()
ax2.plot(hesitancies,[(el1/el2-1)*100 for el1,el2 in zip(total_cases,total_cases_best)],'k--',label='deaths')
ax2.set_xlabel('hesitancy')
ax2.set_ylabel('% difference in cases')
ax2.set_xticklabels(['%i%%' % int(el*100) for el in ax2.get_xticks()])
ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
ax2.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
plt.savefig('v%s_cases_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")
   
f,ax=plt.subplots(figsize=(3,3))
ax.plot(hesitancies,total_yll,color=cm.tab10(1),label='YLL')
ax.plot(hesitancies,total_yll_best,color=cm.tab10(0),label='best')
ax.set_xlabel('hesitancy')
ax.set_ylabel('years of life lost [millions]')
ax.set_xticklabels(['%i%%' % int(el*100) for el in ax.get_xticks()])
ax.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
ax2=ax.twinx()
ax2.plot(hesitancies,[(el1/el2-1)*100 for el1,el2 in zip(total_yll,total_yll_best)],'k--',label='deaths')
ax2.set_xlabel('hesitancy')
ax2.set_ylabel('% difference in YLL')
ax2.set_xticklabels(['%i%%' % int(el*100) for el in ax2.get_xticks()])
ax2.set_yticklabels(['%s%%' % str(round(el,3)) for el in ax2.get_yticks()])
ax2.set_title(r'$f_A = %s, f_V = %s$' % (str(source2.f_A),str(source2.f_V)))
plt.savefig('v%s_yll_optimal_vs_CDC_for_different_hesitancy.pdf' % version,bbox_inches = "tight")


dt=0.5
ts = np.arange(-0, 365+17, dt)  

dummy = 0
param = np.array(np.array(data)[dummy if dummy<7 else 0,6:6+4],dtype=np.float64)
beta = source2.parm_to_beta(param[0:2])
midc=param[2]
exponent=param[3]**2

hesitancy = 0.3
source2.f_A = 0.75 #CDC best estimate
source2.f_V=0.5 #big unknown, we'll vary it from 0 to 1
q17 = 0.85

q = source2.q_based_on_q17(q17)
source2.hesitancy = hesitancy

source2.get_initial_values.recompile()
fitfunc.recompile()
source2.SYS_ODE_VAX_RK4.recompile()
source2.RK4.recompile()
initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)    
Y = fitfunc(np.array(vacopt),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)




Ncomp=20

# initial_2020_12_14=Y[62*2,::] 
sol_by_compartment = np.zeros((ts.shape[0],Ncomp),dtype=np.float64) 

for i in range(Ncomp):
    sol_by_compartment[:,i]=np.sum(Y[:,i::Ncomp],axis=1) 
    
ending = '.pdf'

f,ax = plt.subplots()
# ax.semilogy(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1] + sol_by_compartment[:,2],label='S')
ax.plot(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1],label='S')
ax.plot(ts,sol_by_compartment[:,2],label='V')
ax.plot(ts,sol_by_compartment[:,3] + sol_by_compartment[:,4] + sol_by_compartment[:,5],label='E')
ax.plot(ts,sol_by_compartment[:,6] + sol_by_compartment[:,7] + sol_by_compartment[:,8],label='A')
ax.plot(ts,sol_by_compartment[:,9] + sol_by_compartment[:,10] + sol_by_compartment[:,11],label='RA')
ax.plot(ts,sol_by_compartment[:,12] + sol_by_compartment[:,13] + sol_by_compartment[:,14],label='P')
ax.plot(ts,sol_by_compartment[:,15] + sol_by_compartment[:,16],label='C')
ax.plot(ts,sol_by_compartment[:,19],label='Q')
ax.plot(ts,sol_by_compartment[:,17],label='D')
ax.plot(ts,sol_by_compartment[:,18],label='R')
ax.legend(loc='best')
plt.savefig('v%s_dynamics_CDC_scenario%i_linear%s' % (version,1,ending),bbox_inches = "tight")

f,ax = plt.subplots()
# ax.semilogy(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1] + sol_by_compartment[:,2],label='S')
ax.semilogy(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1],label='S')
ax.plot(ts,sol_by_compartment[:,2],label='V')
ax.plot(ts,sol_by_compartment[:,3] + sol_by_compartment[:,4] + sol_by_compartment[:,5],label='E')
ax.plot(ts,sol_by_compartment[:,6] + sol_by_compartment[:,7] + sol_by_compartment[:,8],label='A')
ax.plot(ts,sol_by_compartment[:,9] + sol_by_compartment[:,10] + sol_by_compartment[:,11],label='RA')
ax.plot(ts,sol_by_compartment[:,12] + sol_by_compartment[:,13] + sol_by_compartment[:,14],label='P')
ax.plot(ts,sol_by_compartment[:,15] + sol_by_compartment[:,16],label='C')
ax.plot(ts,sol_by_compartment[:,19],label='Q')
ax.plot(ts,sol_by_compartment[:,17],label='D')
ax.plot(ts,sol_by_compartment[:,18],label='R')
ax.legend(loc='best')
plt.savefig('v%s_dynamics_CDC_scenario%i_log%s' % (version,1,ending),bbox_inches = "tight")



import datetime
f,ax = plt.subplots(figsize=(4,3))
# ax.semilogy(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1] + sol_by_compartment[:,2],label='S')
ax.semilogy(ts, sol_by_compartment[:,0]+sol_by_compartment[:,1],label='unvaccinated\nsusceptible')
ax.plot(ts,sol_by_compartment[:,2],color=cm.tab10(4),label='vaccinated\nsusceptible')
ax.plot(ts,sol_by_compartment[:,3] + sol_by_compartment[:,4] + sol_by_compartment[:,5]+sol_by_compartment[:,6] + sol_by_compartment[:,7] + sol_by_compartment[:,8]+sol_by_compartment[:,12] + sol_by_compartment[:,13] + sol_by_compartment[:,14]+sol_by_compartment[:,15] + sol_by_compartment[:,16]+sol_by_compartment[:,19],color=cm.tab10(3),label='infected')
ax.plot(ts,sol_by_compartment[:,9] + sol_by_compartment[:,10] + sol_by_compartment[:,11]+sol_by_compartment[:,18],color=cm.tab10(2),label='recovered')
ax.plot(ts,sol_by_compartment[:,17],color='gray',label='dead')
ax.set_ylim([1,4*1e8])
ax.set_ylabel('US population')
last_day = len(source2.cumcases[source2.index_2020_10_13+1:])
xticks = [0,last_day,365+17]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(xticks)
ax.set_xticklabels([el+'\n'+year for el,year in zip(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)),['2020','2020','2021','2021'])])
#ax.legend(loc='best',frameon=False)
ax.legend(bbox_to_anchor=(1,0, 0.5, 1), loc="right", mode="expand",frameon=False)
plt.savefig('v%s_dynamics_CDC_scenario%i_log%s' % (version,1,ending),bbox_inches = "tight")



f,ax = plt.subplots()
ax.plot(ts,sol_by_compartment[:,17],label='D')
ax.plot(np.arange(-0, -0+len(source2.cumdeaths[source2.index_2020_10_13:])) ,source2.cumdeaths[source2.index_2020_10_13:],label='real D')
ax.legend(loc='best',title='cum deaths')
xticks = [0,last_day,365+17]
ax.set_xticks(xticks)
ax.set_xticklabels(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)))

f,ax = plt.subplots()
last_day = len(source2.cumcases[source2.index_2020_10_13:])-0
ax.plot(ts,sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19],label='C')
ax.plot(np.arange(-0, last_day) ,source2.cumcases[source2.index_2020_10_13:],label='real D')
ax.legend(loc='best',title='cum cases')
xticks = [0,last_day,365+17]
ax.set_xticks(xticks)
ax.set_xticklabels(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)))


f,ax = plt.subplots(figsize=(4,3))
last_day = len(source2.cumcases[source2.index_2020_10_13:])-0
ax.plot(ts,sol_by_compartment[:,17],color=cm.tab10(0),label='model')
ax.plot(np.arange(-0, last_day) ,source2.cumdeaths[source2.index_2020_10_13:],ls='--',lw=2,color=cm.tab10(1),label='US data')
ax.set_xticks(xticks)
ax.set_xticklabels([el+'\n'+year for el,year in zip(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)),['2020','2020','2021','2021'])])
ax.legend(loc=5,title='cumulative deaths',frameon=False)
ax.set_ylabel('cumulative deaths [thousands]')
ax.set_yticklabels([int(el//1000) for el in ax.get_yticks()])
ax.set_ylim([ax.get_ylim()[0],600000])
ax.spines['top'].set_visible(False)
[y1,y2] = ax.get_ylim()
ax.plot([0,0],[y1,y2],'k:')
ax.plot([last_day,last_day],[y1,y2],'k:')
ax.set_ylim([y1,y2])
ax2 = ax.twinx()
ax2.plot(ts,sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19],color=cm.tab10(2),label='model')
ax2.plot(np.arange(-0, last_day) ,source2.cumcases[source2.index_2020_10_13:],ls='--',lw=2,color=cm.tab10(4),label='US data')
ax2.set_ylabel('cumulative cases [millions]')
ax2.set_yticklabels([int(el//1000000) for el in ax2.get_yticks()]) 
ax2.legend(loc=4,title='cumulative cases',frameon=False)
ax2.set_ylim([5*1e6,32.2*1e6])
ax2.spines['top'].set_visible(False)
plt.savefig('v%s_fit%s' % (version,ending),bbox_inches = "tight")

        


