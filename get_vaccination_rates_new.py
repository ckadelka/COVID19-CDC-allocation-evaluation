#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 00:15:22 2021

@author: ckadelka
"""

import pandas as pd
import numpy as np
#import datetime

Ntotal = 328239523
A = pd.read_csv('trends_in_number_of_covid19_vaccinations_in_the_us_new.csv')

which = np.bitwise_and(A.Program=='US',A['Date Type']=='Report')
data = np.array(A)[which,:]
#dates_str = data[:,list(A.columns).index('Date')]
#dates = list(map(lambda x: datetime.datetime.strptime(x, '%m/%d/%y'),dates_str))
doses = np.array(data[:,list(A.columns).index('7-Day Avg Total Doses Administered Daily Change')],dtype=np.float64)

index_max_doses = np.argmax(doses)
additional_doses_per_day = doses[index_max_doses]/index_max_doses/2
len_doses = doses.shape[0]

daily_decline_after_max = (additional_doses_per_day*index_max_doses - doses[-1]/2)/(len_doses-index_max_doses)
daily_decline_to_reach_full_vaccination = (doses[-1]/2) / ((Ntotal-np.sum(doses)/2) / (doses[-1]/2) * 2)

#def vaccinations_on_day_t(t,option=1):
#    if t<0:
#        return 0
#    elif t<index_max_doses:
#        return additional_doses_per_day*t
#    elif t<len_doses:
#        return additional_doses_per_day*index_max_doses - daily_decline_after_max*(t-index_max_doses)
#    else:
#        if option==1:
#            return additional_doses_per_day*index_max_doses - daily_decline_after_max*(len_doses-index_max_doses)
#        elif option==2:
#            return 
        
def vaccinations_on_day_t(t,hesitancy=0.25,option=1):
    daily_decline_to_reach_full_vaccination = (doses[-1]/2) / ((Ntotal*(1-hesitancy)-np.sum(doses)/2) / (doses[-1]/2) * 2)
    if t<0:
        return 0
    elif t<len_doses:
        return doses[t]/2
    else:
        if option==1:
            if np.sum(doses)/2 + doses[-1]/2*(t-len_doses) < Ntotal*(1-hesitancy):
                return doses[-1]/2
            else:
                return 0
        elif option==2:
            return max(0,doses[-1]/2-daily_decline_to_reach_full_vaccination*(t-len_doses))

#daily_vaccinations_option1 = np.array(list(map(lambda x: vaccinations_on_day_t(x,hesitancy,1),range(len_doses+380))))
#daily_vaccinations_option2 = np.array(list(map(lambda x: vaccinations_on_day_t(x,hesitancy,2),range(len_doses+380))))

#f,ax=plt.subplots()
#ax.plot(doses)
#ax.plot(np.arange(len(doses)),additional_doses_per_day*np.arange(len(doses)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lss = ['-','--',':']
    for option in [1,2]:
        f,ax = plt.subplots(figsize=(4,4))
        for ii,hesitancy in enumerate([0.25,0.1,0]):
            ax.plot(1e-6*np.array(list(map(lambda x: vaccinations_on_day_t(x,hesitancy,option),range(len_doses+380)))),ls=lss[ii],label=str(int(hesitancy*100))+'%')
        ax.plot(1e-6*np.array(list(map(lambda x: vaccinations_on_day_t(x,hesitancy,option),range(len_doses)))),color='k')
        ax.legend(loc='best',title='hesitancy',frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0,len_doses-1,365])
        ax.set_xticklabels(['12/14/20','05/04/21','12/14/21'])
        ax.set_ylabel('newly fully vaccinated individuals [millions]')
        plt.savefig('projection_vaccination_option%i.pdf' % option,bbox_inches = "tight")

