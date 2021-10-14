#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:22:33 2021

@author: ckadelka
"""

import os
import pandas as pd
import numpy as np
import scipy.signal as signal

#go to https://minhaskamal.github.io/DownGit/#/home
#to download this: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data
folder='csse_covid_19_daily_reports_us/'

dates = []
cases = []
deaths = []
recovered = []
data_death = []
data_cases = []
data_recovered = []
for fname in os.listdir(folder):
    if fname.endswith('.csv'):
        mm,dd,yyyy = fname.split('.csv')[0].split('-')
        dates.append('%s-%s-%s' % (yyyy,mm,dd))
        A=pd.read_csv(folder+fname)
        A=A[A.Province_State!='Recovered']
        cases.append(np.sum(A.Confirmed))
        #deaths.append(np.sum(A.Deaths[:-1]))
        deaths.append(np.sum(A.Deaths))
        recovered.append(np.sum(A.Recovered))
        #print(len(A.Province_State))
        data_death.append(A.Deaths)
        data_cases.append(A.Confirmed)
        data_recovered.append(A.Recovered)
data_recovered = np.array(data_recovered)
data_cases = np.array(data_cases)
data_death = np.array(data_death)

if __name__ == '__main__':
    for data, name in zip([data_recovered,data_cases,data_death],['recovered','cases','death']):
        #B = pd.DataFrame(np.c_[dates,data],columns=['dates']+list(A.Province_State))   
        B = pd.DataFrame(data,columns=list(A.Province_State),index=dates)  
        #B['dates']=dates
        #B = B.sort_values(['dates'])
        B = B.sort_index()
        B.to_excel('JHU_%s_by_state.xlsx' % name)  
     
dates = np.array(dates)
cases = np.array(cases)
deaths = np.array(deaths)
recovered = np.array(recovered)
A = pd.DataFrame(np.c_[dates,list(map(int,cases)),list(map(int,deaths)),list(map(int,recovered))],columns='dates,cases,deaths,recovered'.split(','))
A = A.sort_values(['dates'])
if __name__ == '__main__':
    A.to_excel('JHU_cases_deaths.xlsx')

dates = np.array(np.array(A.dates))
cumcases = np.array(list(map(int,np.array(A.cases))))
cumdeaths = np.array(list(map(int,np.array(A.deaths))))
cumrecovered = np.array(list(map(int,np.array(A.recovered))))

index_2020_10_13 = list(dates).index('2020-10-13') 

window_size=7
poly_max_order=1

daily_deaths = cumdeaths[1:]-cumdeaths[:-1]
filtered_daily_deaths = signal.savgol_filter(daily_deaths,window_size,poly_max_order)
daily_cases = cumcases[1:]-cumcases[:-1]
filtered_daily_cases = signal.savgol_filter(daily_cases,window_size,poly_max_order)