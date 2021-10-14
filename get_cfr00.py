#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:06:01 2021

@author: rafiul
"""

import pandas as pd
import numpy as np

#https://data.census.gov/cedsci/table?q=age&tid=ACSST1Y2019.S0101&hidePreview=false
def mod_us_census_data(dummy):
    lines = dummy.replace('  ','').split('\n')
    which =[]
    n=[]
    for line in lines:
        which.append(line.split('\t')[0])
        n.append(int(line.split('\t')[1]))
    return which,n
        

dummy_a='''        Under 5 years	19404835
        5 to 9 years	19690437
        10 to 14 years	21423479
        15 to 19 years	21353524
        20 to 24 years	21468680
        25 to 29 years	23233299
        30 to 34 years	22345176
        35 to 39 years	21728259
        40 to 44 years	20186586
        45 to 49 years	20398226
        50 to 54 years	20464881
        55 to 59 years	21484060
        60 to 64 years	20984053
        65 to 69 years	17427013
        70 to 74 years	14148548
        75 to 79 years	9759764
        80 to 84 years	6380474
        85 years and over	6358229'''
        
        
dummy_b='''5 to 14 years	41113916
        15 to 17 years	12449034
        Under 18 years	72967785
        18 to 24 years	30373170
        15 to 44 years	130315524
        16 years and over	263534161
        18 years and over	255271738
        21 years and over	241886206
        60 years and over	75058081
        62 years and over	66395660
        65 years and over	54074028
        75 years and over	22498467'''

which_a,n_a = mod_us_census_data(dummy_a)
which_b,n_b = mod_us_census_data(dummy_b)

population_by_age_gp_model = [sum(n_a)-n_b[which_b.index('16 years and over')],
                              n_b[which_b.index('16 years and over')]-n_b[which_b.index('65 years and over')],
                              n_b[which_b.index('65 years and over')]-n_b[which_b.index('75 years and over')],
                              n_b[which_b.index('75 years and over')]]

A=pd.read_csv('cases_and_deaths_by_age_group_CDC.csv') #https://covid.cdc.gov/covid-data-tracker/#demographics
deaths= np.array(A['Count of deaths'])
cases=np.array(A['Count of cases'])

cases_age_adjusted = np.array([cases[0]+11/13*cases[1],2/13*cases[1]+sum(cases[2:6]),cases[6],cases[7]+cases[8]],dtype=np.float64)
deaths_age_adjusted = np.array([deaths[0]+11/13*deaths[1],2/13*deaths[1]+sum(deaths[2:6]),deaths[6],deaths[7]+deaths[8]],dtype=np.float64)

death_prop = deaths_age_adjusted/np.sum(deaths_age_adjusted)
cases_prop = cases_age_adjusted/np.sum(cases_age_adjusted)

age_cfr = deaths_age_adjusted/cases_age_adjusted

#https://s3.amazonaws.com/media2.fairhealth.org/whitepaper/asset/Risk%20Factors%20for%20COVID-19%20Mortality%20among%20Privately%20Insured%20Patients%20-%20A%20Claims%20Data%20Analysis%20-%20A%20FAIR%20Health%20White%20Paper.pdf
factor_cfr_comorbid = (467773*0.59*0.8329)/(467773*0.5171) / ((467773*0.59*0.1671)/(467773*0.4829))
factor_cfr_comorbid = (0.59*0.8329)/(0.5171) / ((0.59*0.1671)/(0.4829))
CFR_age = [0.01,0.03,0.08,0.21,0.55,1.23,5.19]
CFR_age_no_comorbid = [0,0.02,0.06,0.14,0.4,0.97,2.74]

#https://wwwnc.cdc.gov/EID/article/26/8/20-0679-T1
p_comorbid = [19.8,26.8,38.1,55.1,68,79.5,80.7]
comor_prop_by_by_18agegp = [.198,.198,.198,.198,.198,.198, .268,.268,.381,.381,.551,.551,.680, .680,.795,.795,.807,.807] #BRF
prod_population_comor = np.zeros(18)

comor_prop_4agegp = np.zeros(4)
dummy = n_a
for i in range(18):
    prod_population_comor[i] = dummy[i]*comor_prop_by_by_18agegp[i]
comor_prop_4agegp[0]= (np.sum(prod_population_comor[0:3])+ prod_population_comor[3]*1/5)/(population_by_age_gp_model[0]+n_a[3]*1/5)
comor_prop_4agegp[1]= (prod_population_comor[3]*4/5 + np.sum(prod_population_comor[4:13]))/(n_a[3]*4/5+sum(n_a[4:13]))
comor_prop_4agegp[2]= np.sum(prod_population_comor[13:15])/population_by_age_gp_model[2]
comor_prop_4agegp[3]= np.sum(prod_population_comor[15:])/population_by_age_gp_model[3]    


#ACIP and other source for congested elderly living
Nsize_gps_ires_comor = np.array([population_by_age_gp_model[0],
                                 21*1e6,30*1e6,20*1e6,population_by_age_gp_model[1]-(21+30+20)*1e6,
                                 1.041*1e6,population_by_age_gp_model[2]-1.041*1e6,
                                 1.959*1e6,population_by_age_gp_model[3]-1.959*1e6])
Nsize = np.zeros(17,dtype=np.float64)     
gp9_in_agegp = np.array([0,1,1,1,1,2,2,3,3])
Nsize[0] = Nsize_gps_ires_comor[0]
for i in range(1,9):
    Nsize[2*i-1] = Nsize_gps_ires_comor[i] * (1-comor_prop_4agegp[gp9_in_agegp[i]]) 
    Nsize[2*i] = Nsize_gps_ires_comor[i] * (comor_prop_4agegp[gp9_in_agegp[i]])


CFR_nocomorbid_4agegp = [age_cfr[i]/(comor_prop_4agegp[i]*factor_cfr_comorbid+(1-comor_prop_4agegp[i])) for i in range(4)]
CFR = [age_cfr[0]]
for i in range(1,9):
    CFR.append(CFR_nocomorbid_4agegp[gp9_in_agegp[i]])
    CFR.append(CFR_nocomorbid_4agegp[gp9_in_agegp[i]]*factor_cfr_comorbid)
CFR = np.array(CFR,dtype=np.float64)
