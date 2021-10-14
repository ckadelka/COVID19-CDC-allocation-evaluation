#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:19:45 2020

@author: ckadelka
"""



#built-in modules
import os

#added modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import parameter_est_39 as source2

import pandas as pd

plt.rcParams.update({'font.size': 10})
matplotlib.rcParams['text.usetex'] = False

version='39'
infix = 'version'+version

values_last_day_for_fit=['2021-04-28']

folder = 'results/version%s/' % version

params=[[] for h in range(9)]
sses=[[] for h in range(9)]

slurm_ids = []

for fname in os.listdir(folder):
    if fname.endswith('.txt') and version+'_nsim' in fname:
        SLURM_ID=int(fname.split('_')[-1][2:].split('.')[0])
        if SLURM_ID<90:#delete at end
            continue
        slurm_ids.append(SLURM_ID)
        #print(fname)
        f = open(folder+fname,'r')
        textsplit = f.read().splitlines()
        f.close()
        
        dummy = SLURM_ID % 9
        
        for ii,line in enumerate(textsplit):
            if ii==21:
                index0 = values_last_day_for_fit.index(''.join(line.split('\t')[1:]))
            if ii<23:
                continue
            line_split = line.split('\t')
            params[dummy].append(list(map(float,line_split[:4])))
            sses[dummy].append(float(line_split[4]))

CDC_allocation = np.array([3,0,0,1,1,2,2,3,2,0,0,2,2,0,0,1,1],dtype=np.float64)
vacopt = CDC_allocation
ts = np.arange(-0, 365, source2.dt)
Ncomp = source2.Ncomp
Ngp = source2.Ngp
cumdeaths = source2.cumdeaths
cumcases = source2.cumcases
index_2020_10_13 = source2.index_2020_10_13

param_opts = []
sse_opts = []
rows = []
for dummy in range(4):
    try:
        argmin = np.argmin(sses[dummy])
    except ValueError: #empty sequence
        continue
    
    f_A = 0.75 #CDC best estimate
    f_V=0.5 #big unknown, we'll vary it from 0 to 1
    q17 = 0.85
    
    if dummy==0:
        f_A = 0.25
        f_V = 0
    if dummy==1:
        f_A = 1
        f_V = 0    
    if dummy==2:
        f_A = 0.25
        f_V = 1
    if dummy==3:
        f_A = 1
        f_V = 1  
    
    param_opt = params[dummy][argmin]
    param_opts.append(param_opt)
    sse_opts.append(sses[dummy][argmin])
    print(source2.last_day_for_fit,len(params[dummy]),np.min(sses[dummy]),np.mean(sses[dummy]))
    rows.append([source2.last_day_for_fit,source2.hesitancy,f_A,f_V,q17]+param_opt+[sse_opts[-1]])
A = pd.DataFrame(rows,columns=['last day for fit','hesitancy','f_A','f_V','q17','beta0','beta1','midC','exponent','SSE']).to_csv('v%s_bGA_output.csv' % version) 

def format_param(param_opt,min_sse,fixed):
    text=''
    var = [r'$b_0$',r'$b_1$','c','k']
    output = list(map(str,map(lambda x: np.round(x,4),param_opt)))
    for i in range(len(output)):
        text+=var[i]+' = '+output[i]
        if i%4==3:
            text+='\n'
        elif i<len(output)-1:
            text+=', '  
    text+='\nmin SSE [*1e9] = '+str(np.round(min_sse/1e9,2))+'\n'
    
    var = [r'$f_A$',r'$f_V$',r'$\sigma$',r'$\delta$']
    output = list(map(str,map(lambda x: np.round(x,2),fixed)))
    for i in range(len(output)):
        text+=var[i]+' = '+output[i]
        if i<len(output)-1:
            text+=', '      
    return text
    

rows = []
counter = 0
PLOT=False
h=0
for dummy in range(4):
    try:
        argmin = np.argmin(sses[dummy])
    except ValueError: #empty sequence
        continue
    param_opt = params[dummy][argmin]            
    beta = source2.parm_to_beta(param_opt[0:2])
    midc=param_opt[2]
    exponent=param_opt[3]**2
    
    hesitancy = 0.3
    source2.f_A = 0.75 #CDC best estimate
    source2.f_V=0.5 #big unknown, we'll vary it from 0 to 1
    q17 = 0.85
    
    if dummy==0:
        f_A = 0.25
        f_V = 0
    if dummy==1:
        f_A = 1
        f_V = 0    
    if dummy==2:
        f_A = 0.25
        f_V = 1
    if dummy==3:
        f_A = 1
        f_V = 1  
    q = source2.q_based_on_q17(q17)
    
    initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=hesitancy,START_DATE_INDEX=source2.index_2020_10_13)    
    Y = source2.RK4(source2.SYS_ODE_VAX_RK4,initial_values,ts,vacopt,beta,exponent,midc,source2.mu_A,source2.mu_C,source2.mu_Q,q,source2.sigma,source2.delta,source2.f_A,source2.f_V,source2.number_of_daily_vaccinations)

    sol_by_compartment = np.zeros((ts.shape[0],Ncomp),dtype=np.float64)
    for ii in range(source2.Ncomp):
        sol_by_compartment[:,ii]=np.sum(Y[:,ii::Ncomp],axis=1) 
        
    rows.append([source2.last_day_for_fit,source2.hesitancy,f_A,f_V,q17]+param_opt+[sse_opts[counter]]+[sol_by_compartment[-1,17]/1e3,(sol_by_compartment[-1,15]+sol_by_compartment[-1,16]+sol_by_compartment[-1,17]+sol_by_compartment[-1,18]+sol_by_compartment[-1,19])/1e6])    
    
    total_C=sol_by_compartment[:,15]+sol_by_compartment[:,16]
    r = 1/(1+(midc/np.log10(total_C+1))**exponent)    
    
    if PLOT:
        fig2, ((ax1), (ax2),(ax3),(ax4)) = plt.subplots(4, 1, figsize=(6,11)) 
        #fig2.suptitle('\n'.join(list(map(str,map(lambda x: np.round(x,4),param_opt))))+'\nSSE: '+str(np.min(sses[i][j][k])))
        fig2.suptitle(format_param(param_opt,np.min(sses[dummy]),[source2.f_A,source2.f_V,source2.sigma[0],source2.delta[0]]))
        ax1.plot(ts,sol_by_compartment[:,17],label='D')
        ax1.text(150,300000,str(int(sol_by_compartment[-1,17])),ha='center',va='center')
        ax1.plot(np.arange(-0, -0+len(cumdeaths[index_2020_10_13:])) ,cumdeaths[index_2020_10_13:],label='real D')
        ax2.plot(ts,sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19],label='C')
        ax2.plot(np.arange(-0, -0+len(cumcases[index_2020_10_13:])) ,cumcases[index_2020_10_13:],label='real C')
        ax3.plot(np.arange(Ngp),beta,label='beta') 
        ax4.plot(total_C,r,label='contact reduction') 
        ax1.legend(loc='best') 
        ax2.legend(loc='best')   
        ax3.legend(loc='best') 
        ax4.legend(loc='best')  
        plt.savefig('v%s_optimal_fit_upto%s_fa%s_fv%s_sigma%s.pdf' % (version,source2.last_day_for_fit,source2.f_A,source2.f_V,source2.sigma[0]))
    counter += 1
A = pd.DataFrame(rows,columns=['last day for fit','hesitancy','f_A','f_V','q17','beta0','beta1','midC','sqrt(exponent)','SSE','total deaths [thousands]','total cases [millions]']).to_csv('v%s_bGA_output_with_deaths_and_cases.csv' % version) 
