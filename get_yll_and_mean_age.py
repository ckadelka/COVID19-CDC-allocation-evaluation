#https://www.ssa.gov/oact/STATS/table4c6.html#ss

import pandas as pd
import numpy as np

A = pd.read_csv('SSA_death_rates.csv')
death_rate_male = np.array(A['Male death rate'])
death_rate_female = np.array(A['Female death rate'])
life_expectancy_male =np.array(A['Male life expectancy'])
life_expectancy_female =np.array(A['Female life expectancy'])

prop_survival_male = np.ones(120) 
prop_survival_female = np.ones(120)
for i in range(1,120):
    prop_survival_male[i] = prop_survival_male[i-1]*(1-death_rate_male[i-1])
    prop_survival_female[i] = prop_survival_female[i-1]*(1-death_rate_female[i-1])
prop_survival = 0.5*(prop_survival_male+prop_survival_female)
yll = 0.5*(life_expectancy_male+life_expectancy_female)

age = np.arange(120)
agebin = [0,16,65,75,-1]    

mean_4agegp = np.zeros(4,dtype=np.float64)
yll_4agegp = np.zeros(4,dtype=np.float64)
for i in range(4):
    mean_4agegp[i]= np.dot(prop_survival[agebin[i]:agebin[i+1]], age[agebin[i]:agebin[i+1]])/np.sum(prop_survival[agebin[i]:agebin[i+1]])
    yll_4agegp[i] = np.dot(prop_survival[agebin[i]:agebin[i+1]], yll[agebin[i]:agebin[i+1]])/np.sum(prop_survival[agebin[i]:agebin[i+1]])
