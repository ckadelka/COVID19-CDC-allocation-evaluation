#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:19:45 2020

@author: ckadelka
"""



#built-in modules
import sys
import random
import os

#added modules
import numpy as np
import networkx as nx
import itertools
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib
import scipy.signal as signal
import fitfunc_v39 as source
import parameter_est_39 as source2
from matplotlib import cm

#see  https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
from scipy import spatial
from functools import reduce

def filter_(pts, pt):
    """
    Get all points in pts that are not Pareto dominated by the point pt
    """
    weakly_worse   = (pts >= pt).all(axis=-1)
    strictly_worse = (pts > pt).any(axis=-1)
    return pts[~(weakly_worse & strictly_worse)]


def get_pareto_undominated_by(pts1, pts2=None):
    """
    Return all points in pts1 that are not Pareto dominated
    by any points in pts2
    """
    if pts2 is None:
        pts2 = pts1
    return reduce(filter_, pts2, pts1)


def get_pareto_frontier(pts):
    """
    Iteratively filter points based on the convex hull heuristic
    """
    pareto_groups = []

    # loop while there are points remaining
    while pts.shape[0]:
        # brute force if there are few points:
        if pts.shape[0] < 10:
            pareto_groups.append(get_pareto_undominated_by(pts))
            break

        # compute vertices of the convex hull
        try:
            hull_vertices = spatial.ConvexHull(pts).vertices
        except:
            return np.vstack(pareto_groups)
        # get corresponding points
        hull_pts = pts[hull_vertices]

        # get points in pts that are not convex hull vertices
        nonhull_mask = np.ones(pts.shape[0], dtype=bool)
        nonhull_mask[hull_vertices] = False
        pts = pts[nonhull_mask]

        # get points in the convex hull that are on the Pareto frontier
        pareto   = get_pareto_undominated_by(hull_pts)
        pareto_groups.append(pareto)

        # filter remaining points to keep those not dominated by
        # Pareto points of the convex hull
        pts = get_pareto_undominated_by(pts, pareto)

    return np.vstack(pareto_groups)

# --------------------------------------------------------------------------------
# previous solutions
# --------------------------------------------------------------------------------

def is_pareto_efficient_dumb(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs>=c, axis=1))
    return is_efficient


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def is_pareto_efficient_2d(costs1,costs2,RETURN_INDICES=False):
    """
    :two costs: two (n_points) arrays
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    assume no ties in costs
    """
    indices_sorted_by_costs1 = np.argsort(costs1)
    costs1=costs1[indices_sorted_by_costs1]
    costs2=costs2[indices_sorted_by_costs1]
    min_costs1 = costs1[0]
    min_costs2 = costs2[0]
    pareto = [[min_costs1,min_costs2]]
    indices = [indices_sorted_by_costs1[0]]
    for i, c in enumerate(costs2[1:]):
        if c < min_costs2:
            min_costs2 = c
            pareto.append([costs1[i+1],c])
            indices.append(indices_sorted_by_costs1[i+1])
    if RETURN_INDICES:
        return pareto,indices
    else:
        return pareto
    
def is_pareto_efficient_3d(costs1,costs2,costs3,RETURN_INDICES=False):
    """
    :three costs: two (n_points) arrays
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    assume no ties in costs
    """
    indices_sorted_by_costs1 = np.argsort(costs1)
    costs1=costs1[indices_sorted_by_costs1]
    costs2=costs2[indices_sorted_by_costs1]
    costs3=costs3[indices_sorted_by_costs1]
    min_costs1 = costs1[0]
    min_costs2 = costs2[0]
    min_costs3 = costs3[0]
    pareto = [[min_costs1,min_costs2,min_costs3]]
    indices = [indices_sorted_by_costs1[0]]
    for i, (c2,c3) in enumerate(zip(costs2[1:],costs3[1:])):
        NEW_OPTIMAL=False
        if c2 < min_costs2:
            min_costs2 = c2
            NEW_OPTIMAL=True
        if c3 < min_costs3:
            min_costs3 = c3
            NEW_OPTIMAL=True
        if NEW_OPTIMAL:
            pareto.append([costs1[i+1],c2,c3])
            indices.append(indices_sorted_by_costs1[i+1])
    if RETURN_INDICES:
        return pareto,indices
    else:
        return pareto
       
def sort_convex_hull_points(points):
    argmin = np.argmin(points[:,0])
    ordered = np.array([points[argmin]])
    lower_points = points[points[:,1]<points[argmin,1],:]
    ordered = np.r_[ordered, lower_points[np.argsort(lower_points[:,0]),:] ]
    higher_points = points[points[:,1]>=points[argmin,1],:]
    ordered = np.r_[ordered, higher_points[np.argsort(higher_points[:,0]),:][::-1,:] ]
    return ordered


def dominates(row, rowCandidate):
    return all(r >= rc for r, rc in zip(row, rowCandidate))


def cull(pts, dominates):
    dominated = []
    cleared = []
    remaining = pts
    while remaining:
        candidate = remaining[0]
        new_remaining = []
        for other in remaining[1:]:
            [new_remaining, dominated][dominates(candidate, other)].append(other)
        if not any(dominates(other, candidate) for other in new_remaining):
            cleared.append(candidate)
        else:
            dominated.append(candidate)
        remaining = new_remaining
    return cleared, dominated

def keep_efficient(pts):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
    pts = pts[pts.sum(1).argsort()[::-1]]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        # process each point in turn
        n = pts.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i+1:n] = (pts[i+1:] < pts[i]).any(1) 
        # keep points undominated so far
        pts = pts[undominated[:n]]
    return pts


plt.rcParams.update({'font.size': 10})
matplotlib.rcParams['text.usetex'] = False

allocations=[]





version='39'
infix = 'version'+version

folder = 'results/%s/' % infix



for dummy in range(0,1):
    nsim = 175000
    nr_scenarios = 1
    
    
    infix = 'version'+version
    
    
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
        
    id_scenario = 'hes%i_fA%i_fV%i_q17%i' % (int(hesitancy*100),int(source2.f_A*100),int(source2.f_V*100),int(q17*100))
    infix+='_'+id_scenario
    
    param = np.array(np.array(source.data)[dummy if dummy<7 else 0,6:6+4],dtype=np.float64)
    beta = source2.parm_to_beta(param[:2])
    midc = param[2]
    exponent = param[3]**2
    
    CDC_allocation = np.array([3,0,0,1,1,2,2,3,2,0,0,2,2,0,0,1,1])
    
    deathss=[]
    casess=[]
    infectionss=[]
    entropiess=[]
    yllss=[]
    allocation_idss = []
    
    for scen in range(1,nr_scenarios+1):
        deaths=[[] for _ in range(4)]
        cases=[[] for _ in range(4)]
        infections=[[] for _ in range(4)]
        max_infections=[]
        ylls = []
        allocation_ids = []
        time_in_seconds = []
        scens = []
        counter = 0
        slurm_ids = []
    
        for fname in os.listdir(folder):
            if fname.endswith('%s.txt' % id_scenario) and 'nsim%i' % nsim in fname:
                SLURM_ID=int(fname.split('_')[3][2:])
                if SLURM_ID<dummy*100 or SLURM_ID >= (dummy+1)*100:#delete at end
                    continue
                slurm_ids.append(SLURM_ID)
                #print(fname)
                f = open(folder+fname,'r')
                textsplit = f.read().splitlines()
                f.close()
                counter_d=0
                counter_c=0
                counter_i=0
                if counter==0:
                    filename = fname.split('_seed')[0]
                    dict_fixed_parameters = {}
                for line in textsplit:
                    line_split = line.split('\t')
                    if line_split[0] == 'time in seconds':
                        time_in_seconds.append(int(line_split[1]))
                    if 'deaths_in_age_group' in line_split[0]:
                        deaths[counter_d].extend(list(map(int,map(float,line_split[1:])))) 
                        counter_d+=1
                    if 'cases_in_age_group' in line_split[0]:
                        #cases[counter_c].extend(list(map(int,line_split[1:]))) 
                        cases[counter_c].extend(list(map(int,map(float,line_split[1:])))) 
                        counter_c+=1
                    if 'infections_in_age_group' in line_split[0]:
                        #infections[counter_i].extend(list(map(int,line_split[1:]))) 
                        infections[counter_i].extend(list(map(int,map(float,line_split[1:])))) 
                        counter_i+=1
                    elif line_split[0] == 'allocation ID':
                        allocation_ids.extend(list(map(int,line_split[1:]))) 
                    elif len(line_split)==2:
                        if counter==0:
                            dict_fixed_parameters.update({line_split[0]:line_split[1]})
                counter+=1
        allocation_ids = np.array(allocation_ids)
        indices = np.argsort(allocation_ids)
        deaths = np.array(deaths)
        cases = np.array(cases)
        infections = np.array(infections)
    
        deaths = deaths[:,indices]
        cases = cases[:,indices]
        infections = infections[:,indices]
        allocation_ids = allocation_ids[indices]
    
        #get total deaths
        total_deaths = np.sum(deaths,0)
        
        #get yll
        average_years_of_life_left = np.array([71.45519038, 41.66010998, 16.82498872,  7.77149779])
        ylls = np.dot(average_years_of_life_left,deaths)
        
        #get total cases
        total_cases = np.sum(cases,0)
        
        #get total infections
        total_infections = np.sum(infections,0)
        
        #equitable
        total_deaths_prop = np.multiply(deaths,1/total_deaths)
        entropies = np.sum(np.multiply(total_deaths_prop,np.log(total_deaths_prop)),0)
    
        deathss.append(total_deaths)
        casess.append(total_cases)
        infectionss.append(total_infections)
        entropiess.append(entropies*1000)
        yllss.append(ylls)
        allocation_idss.append(allocation_ids)
    
    infix += '_sigma%i' % (source2.sigma[0]*10000)
    
    def get_outcomes(fitfunc_short_output,average_years_of_life_left = np.array([71.45519038, 41.66010998, 16.82498872,  7.77149779])):
        deaths = fitfunc_short_output[:4]
        total_deaths = sum(deaths)
        total_cases = sum(fitfunc_short_output[4:8])
        total_infections = sum(fitfunc_short_output[8:12])
        ylls = np.dot(average_years_of_life_left,deaths)
        total_deaths_prop = np.multiply(deaths,1/total_deaths)
        entropies = np.sum(np.multiply(total_deaths_prop,np.log(total_deaths_prop)))
        return (total_deaths,total_cases,total_infections,ylls,entropies)
        #return (total_deaths,total_cases,total_infections,ylls,entropies,total_deaths_prop[-1])
    
    
    optimal_allocationss = []
    optimal_idss = []
    optimal_values = []
    mean_values = []
    CDC_outcome = np.zeros((nr_scenarios,5))
    equal_outcome = np.zeros((nr_scenarios,5))
    deaths_optimal = np.zeros((nr_scenarios,5))
    cases_optimal = np.zeros((nr_scenarios,5))
    infections_optimal = np.zeros((nr_scenarios,5))
    ylls_optimal = np.zeros((nr_scenarios,5))
    entropies_optimal = np.zeros((nr_scenarios,5))
    deaths_per_age_optimal = np.zeros((nr_scenarios,5,4))
    
    CDC_ID = source.get_ID_for_feasible_allocation(CDC_allocation)
    equal_ID = 0
    for j in range(nr_scenarios):
        optimal_allocationss.append([])
        optimal_idss.append([])
        optimal_values.append([])
        mean_values.append([])
        for i,vec in enumerate([deathss[j],casess[j],infectionss[j],yllss[j],entropiess[j]]):
            OPTIMAL_ID = np.argmin(vec)
            optimal_allocation = source.get_i1_to_i17_which_satisfy_all_constraints(allocation_idss[j][OPTIMAL_ID])
            optimal_idss[j].append(allocation_ids[OPTIMAL_ID])
            optimal_allocationss[j].append(optimal_allocation)
            optimal_values[j].append(vec[OPTIMAL_ID])
            mean_values[j].append(np.mean(vec))
            CDC_outcome[j,i] = vec[CDC_ID]
            equal_outcome[j,i] = vec[equal_ID]
            deaths_optimal[j,i] = deathss[j][OPTIMAL_ID]
            cases_optimal[j,i] = casess[j][OPTIMAL_ID]
            infections_optimal[j,i] = infectionss[j][OPTIMAL_ID]
            ylls_optimal[j,i] = yllss[j][OPTIMAL_ID]
            entropies_optimal[j,i] = entropiess[j][OPTIMAL_ID]
            deaths_per_age_optimal[j,i] = deaths[:,OPTIMAL_ID]
    deaths_per_age_CDC = deaths[:,CDC_ID]
    deaths_per_age_equal = deaths[:,equal_ID]
    optimal_idss = np.array(optimal_idss)
    optimal_allocationss = np.array(optimal_allocationss)
    optimal_values = np.array(optimal_values)
    mean_values = np.array(mean_values)
    
    types= ['deaths','cases','infections','yll','entropy']
    B=np.zeros((0,21))
    for j in range(len(types)):
    
        A = pd.DataFrame(np.c_[mean_values[:,j],optimal_values[:,j],optimal_idss[:,j],1+optimal_allocationss[:,j,:]],index=['scenario '+str(i+1) for i in range(nr_scenarios)],columns=['mean '+types[j],'min '+types[j],'allocation_ID']+['Group '+str(i+1) for i in range(17)])
        A.to_excel('summary_'+types[j]+'.xlsx')
        
        B = np.r_[B,np.c_[mean_values[:,j],optimal_values[:,j],optimal_idss[:,j],CDC_outcome[:,j],1+optimal_allocationss[:,j,:]]]
    B = pd.DataFrame(B,index=[types[j]+', scenario '+str(i+1) for j in range(len(types)) for i in range(nr_scenarios) ],columns=['mean ','min ','allocation_ID','CDC outcome']+['Group '+str(i+1) for i in range(17)])
    B.to_excel('%s_summary_all.xlsx' % infix)
    
    
    #how good is each optimal allocation in other objectives
    types= ['Deaths','Cases','Infections','YLL','Equity']
    types_legend = ['fewest deaths','fewest cases','fewest infections','lowest YLL','highest equity']
    types_legend = ['fewest deaths','fewest cases','fewest infections','lowest YLL','highest equity']
    for i in range(nr_scenarios):
        (deaths_CDC,cases_CDC,infections_CDC,ylls_CDC,entropies_CDC) = CDC_outcome[i]
        (deaths_equal,cases_equal,infections_equal,ylls_equal,entropies_equal) = equal_outcome[i]
        deaths = np.append(deaths_CDC,deaths_optimal[i,np.array([0,3,1,2])])
        YLLs = np.append(ylls_CDC,ylls_optimal[i,np.array([0,3,1,2])])
        cases = np.append(cases_CDC,cases_optimal[i,np.array([0,3,1,2])])
        infections = np.append(infections_CDC,infections_optimal[i,np.array([0,3,1,2])])
    
    deaths_perc = (deaths/np.min(deaths)-1)*100
    YLLs_perc = (YLLs/np.min(YLLs)-1)*100
    cases_perc = (cases/np.min(cases)-1)*100
    infections_perc = (infections/np.min(infections)-1)*100
    
    C = pd.DataFrame(np.c_[np.append(deaths/1e3,deaths_perc),np.append(YLLs/1e6,YLLs_perc),np.append(cases/1e6,cases_perc),np.append(infections/1e6,infections_perc)])
    C.to_excel('%s_summary_all_relative_comparison.xlsx' % infix)
    
    if allocations==[]:
        allocations = np.array(list(map(source.get_i1_to_i17_which_satisfy_all_constraints,allocation_idss[0])))        
    
    A,indices = is_pareto_efficient_3d(total_deaths,total_cases,ylls,True)
    A = np.array(A)
    AA = pd.DataFrame(np.c_[A[:,0]/1e3,A[:,1]/1e6,A[:,2]/1e6,[int(el[0]<deaths_CDC)+int(el[1]<cases_CDC)+int(el[2]<ylls_CDC) for el in A],allocations[np.array(indices)]],columns = ['deaths','cases','YLL','pareto dominant over CDC']+list(map(str,range(1,18))))
    AA.to_excel('%s_pareto_frontier_3d.xlsx' % infix)
    
    ending = '.pdf'
    
    types= ['Deaths','Cases','Infections','YLL','Equity']
    types_legend = ['fewest deaths','fewest cases','fewest infections','lowest YLL','highest equity']
    types_legend = ['fewest deaths','fewest cases','fewest infections','lowest YLL','highest equity']
    for i in range(nr_scenarios):        
        death_counts=deathss[i]
        total_cases =casess[i]
    
        dummy = is_pareto_efficient_2d(death_counts,total_cases,True)
        pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
        colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9)]
        colors = [cm.tab20b(1),cm.tab10(4),cm.tab10(2),cm.tab10(9)]
        colors = [cm.tab10(3),cm.tab20b(2),cm.tab20b(9),cm.tab20b(17),cm.tab10(1)]
        markers = ['x','o','P','<','D','>']
        f,ax = plt.subplots(figsize=(3,3))
        ax.plot(pareto[:,0],pareto[:,1],'k-',label='Pareto frontier')
        counter=0
        ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],marker=markers[counter],label='CDC',zorder=3)
        for j in range(len(types_legend)):
            if j==2 or j==4:
                continue
            counter+=1
            ax.plot([deaths_optimal[i,j]],[cases_optimal[i,j]],'X',color=colors[counter],marker=markers[counter],label=types_legend[j])  
        #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
        ax.legend(loc=0,frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('deaths [thousands]')
        ax.set_xticklabels([int(el//1000) for el in ax.get_xticks()])
        ax.set_ylabel('cases [millions]')
        ax.set_yticklabels([round(el/1000000,1) for el in ax.get_yticks()])    
        plt.savefig('%s_overall_pareto_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    
        pts = np.c_[death_counts,total_cases]
        indices = spatial.ConvexHull(pts).vertices
        ordered = sort_convex_hull_points(pts[indices])
        argmin = np.argmin(ordered,0)[1]
        f,ax = plt.subplots(figsize=(5.7,3))
        ax.plot(ordered[argmin:,0],ordered[argmin:,1],'k--')
        ax.plot(pareto[:,0],pareto[:,1],'k-')
        colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(3)]
        counter=0
        ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],label='CDC',zorder=3)
        for j in range(len(types_legend)):
            if j==2:
                continue
            counter+=1
            ax.plot([deaths_optimal[0,j]],[cases_optimal[0,j]],'X',color=colors[counter],label=types_legend[j])  
        ax.plot([deaths_equal],[cases_equal],'X',color=cm.Set1(8),label='no phases')
        #ax.legend(loc='best',frameon=False)
        ax.set_xlabel('deaths [thousands]')
        ax.set_xticklabels([int(el//1000) for el in ax.get_xticks()])
        ax.set_ylabel('cases [millions]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticklabels([int(el//1000000) for el in ax.get_yticks()]) 
        ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3,frameon=False)
        plt.savefig('%s_overall_convex_hull_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")





        dummy = is_pareto_efficient_2d(death_counts,total_cases,True)
        pareto1,_ = np.array(dummy[0]),np.array(dummy[1])
        
        dummy = is_pareto_efficient_2d(death_counts,-total_cases,True)
        pareto2,_ = np.array(dummy[0]),np.array(dummy[1])
        pareto2[:,1]*=-1

        dummy = is_pareto_efficient_2d(-death_counts,-total_cases,True)
        pareto3,_ = np.array(dummy[0]),np.array(dummy[1])
        pareto3*=-1        

        dummy = is_pareto_efficient_2d(-death_counts,total_cases,True)
        pareto4,_ = np.array(dummy[0]),np.array(dummy[1])
        pareto4[:,0]*=-1
        
        f,ax = plt.subplots()
        for ii,pareto in enumerate([pareto1,pareto2,pareto3,pareto4]):
            ax.plot(pareto[:,0],pareto[:,1],'k:' if ii>0 else 'k-')
        colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(3)]
        colors = [cm.tab10(3),cm.tab20b(2),cm.tab20b(9),cm.tab20b(17),cm.tab10(1)]
        markers = ['x','o','P','<','D','>']
        counter=0
        ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],marker=markers[counter],label='CDC',zorder=3)
        for j in range(len(types_legend)):
            if j==2:
                continue
            counter+=1
            ax.plot([deaths_optimal[0,j]],[cases_optimal[0,j]],'X',color=colors[counter],marker=markers[counter],label=types_legend[j])  
        ax.plot([deaths_equal],[cases_equal],'X',color=cm.tab10(7),marker=markers[-1],label='no phases')
        #ax.legend(loc='best',frameon=False)
        ax.set_xlabel('deaths [thousands]')
        ax.set_xticklabels([int(el//1000) for el in ax.get_xticks()])
        ax.set_ylabel('cases [millions]')
        ax.set_yticks(np.array([36,40,44,48,52])*1e6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticklabels([int(el//1000000) for el in ax.get_yticks()]) 
        ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3,frameon=False)
        plt.savefig('%s_overall_convex_hull_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")

    
    
    
    
    
    #equitable?
    gp_in_agegp = np.array([0,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3])
    for i in range(nr_scenarios):
    #    for j in range(len(types)):
    #        result = source.fitfunc_short(np.array(optimal_allocationss[i][j]),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #        deaths_per_age = np.array(result[:4])
    #        print(i,j,deaths,deaths_per_age/sum(deaths_per_age))
    #        f,ax = plt.subplots(figsize=(4,2))
    #        ax.bar(range(4),deaths_per_age/sum(deaths_per_age),color='#2F2F2F')
    #        ax.set_xticks(range(4))
    #        ax.set_xticklabels(['0-15','16-64','65-74','75+'])
    #        ax.set_yticks([0,0.5])
    #        ax.set_yticklabels(['0%','50%'])
    #        for k in range(4):
    #            ax.text(k,0.06,str(round((100*deaths_per_age/sum(deaths_per_age))[k],2))+'%',va='center',ha='center',color='white' if k>0 else '#2f2f2F')
    #        plt.savefig(('%s_deaths_distribution_per_age_using_optimal_%s_allocation_scenario%i' % (infix,types[j],i+1))+ending,bbox_inches = "tight")
    #
    #
    #        
    #    result_CDC = source.fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #    deaths_per_age = np.array(result_CDC[:4])
    #    print(i,'CDC',deaths,deaths_per_age/sum(deaths_per_age))
    #    f,ax = plt.subplots(figsize=(4,1.2))
    #    ax.bar(range(4),deaths_per_age/sum(deaths_per_age),color='#2F2F2F')
    #    ax.set_xticks(range(4))
    #    ax.set_xticklabels(['0-15','16-64','65-74','75+'])
    #    ax.set_yticks([0,0.5])
    #    ax.set_yticklabels(['0%','50%'])
    #    for k in range(4):
    #        ax.text(k,0.06,str(round((100*deaths_per_age/sum(deaths_per_age))[k],2))+'%',va='center',ha='center',color='white' if k>0 else '#2f2f2F')
    #    plt.savefig(('%s_deaths_distribution_per_age_using_CDC_allocation_scenario%i' % (infix,i+1))+ending,bbox_inches = "tight")
    #
    #
    #
    #    result_equal = source.fitfunc_short(np.ones(17,dtype=np.float64),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #    deaths_per_age = np.array(result_equal[:4])
    #    print(i,'equal',deaths,deaths_per_age/sum(deaths_per_age))
    #    f,ax = plt.subplots(figsize=(4,2))
    #    ax.bar(range(4),deaths_per_age/sum(deaths_per_age),color='#2F2F2F')
    #    ax.set_xticks(range(4))
    #    ax.set_xticklabels(['0-15','16-64','65-74','75+'])
    #    ax.set_yticks([0,0.5])
    #    ax.set_yticklabels(['0%','50%'])
    #    for k in range(4):
    #        ax.text(k,0.06,str(round((100*deaths_per_age/sum(deaths_per_age))[k],2))+'%',va='center',ha='center',color='white' if k>0 else '#2f2f2F')
    #    plt.savefig(('%s_deaths_distribution_per_age_using_equal_allocation_scenario%i' % (infix,i+1))+ending,bbox_inches = "tight")
    # 
    #    f,ax = plt.subplots(figsize=(6,2))
    #    counter=0
    #    i=0
    #    result_CDC = source.fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #    deaths_per_age = np.array(result_CDC[:4])
    #    ax.bar(np.arange(4)*4+2+(counter-2)*0.8,deaths_per_age/sum(deaths_per_age),color=cm.Set1(counter),label='CDC')
    #    counter+=1
    #    for j in range(len(types)):
    #        if j==2 or j==4:
    #            continue
    #        result = source.fitfunc_short(np.array(optimal_allocationss[i][j]),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #        deaths_per_age = np.array(result[:4])
    #        ax.bar(np.arange(4)*4+2+(counter-2)*0.8,deaths_per_age/sum(deaths_per_age),color=cm.Set1(counter),label=types[j])
    #        counter+=1
    #    ax.legend(loc='best',frameon=False)
        
        colors = ['k']*10
        types_legend = ['CDC','fewest deaths','fewest cases','lowest YLL']
        f,ax = plt.subplots(nrows=2,ncols=2,sharey=True,sharex=True,figsize=(7.4,3))
        counter=0
        ax[0,0].bar(range(4),deaths_per_age_CDC/sum(deaths_per_age_CDC),color='#2F2F2F',label='CDC')
        ax[counter//2,counter%2].text(1.5,0.5,'CDC',color=colors[counter],va='center',ha='center')
        for k in range(4):
            ax[counter//2,counter%2].text(k,0.06,str(round((100*deaths_per_age_CDC/sum(deaths_per_age_CDC))[k],2))+'%',va='center',ha='center',color='white' if k>0 else '#2f2f2F')
        for j in range(len(types)):
            if j==2 or j==4:
                continue
            counter+=1
            ax[counter//2,counter%2].bar(range(4),deaths_per_age_optimal[i,j]/sum(deaths_per_age_optimal[i,j]),color='#2F2F2F',label=types_legend[j])
            ax[counter//2,counter%2].text(1.5,0.5,types_legend[counter],color=colors[counter],va='center',ha='center')
            for k in range(4):
                ax[counter//2,counter%2].text(k,0.06,str(round((100*deaths_per_age_optimal[i,j]/sum(deaths_per_age_optimal[i,j]))[k],2))+'%',va='center',ha='center',color='white' if k>0 else '#2f2f2F')
        for counter in range(4):
            ax[counter//2,counter%2].set_xticks(range(4))
            ax[counter//2,counter%2].set_xticklabels(['0-15','16-64','65-74','75+'])
            ax[counter//2,counter%2].set_yticks([0,0.5])
            ax[counter//2,counter%2].set_yticklabels(['0%','50%'])    
            ax[counter//2,counter%2].spines['top'].set_visible(False)
            ax[counter//2,counter%2].spines['right'].set_visible(False)
        ax[0,0].text(-1.3,-0.04,'Proportion of deaths per age group',va='center',ha='center',clip_on=False,rotation=90)
        f.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(('%s_deaths_distribution_per_age_all_scenario%i' % (infix,i+1))+ending,bbox_inches = "tight")
    
    D = pd.DataFrame(np.c_[deaths_per_age_optimal[0].T,deaths_per_age_CDC,deaths_per_age_equal],columns=types+['CDC','equal'])
    D.to_excel('%s_deaths_per_age.xlsx' % infix)
    
    
    
    
    #death_counts_per_scenario = np.array([[deathss[i,optimal_idss[j,0]] for j in range(nr_scenarios)] for i in range(nr_scenarios)])
    #death_counts_per_scenario_relative = np.multiply(death_counts_per_scenario.T,1/np.diag(death_counts_per_scenario)).T
    
    
    
    
    
    ## Spearman correlations
    #types= ['Deaths','Cases','Infections','YLL','Equity']
    types= ['Deaths','YLL','Cases','Infections']
    ending = '.pdf'
    spearman = np.ones((nr_scenarios,len(types),len(types)))
    n_types=len(types)
    for j in range(nr_scenarios):
        #vecs = [deathss[j],casess[j],total_infectionss[j],yllss[j]]
        #vecs = [deathss[j],casess[j],yllss[j],entropiess[j]]
        vecs = [deathss[j],yllss[j],casess[j],infectionss[j]]
        for ii in range(len(vecs)):
            for jj in range(ii+1,len(vecs)):
                spearman[j,ii,jj] = stats.spearmanr(vecs[ii],vecs[jj])[0]
                spearman[j,jj,ii] = spearman[j,ii,jj]
    
        f,ax = plt.subplots(figsize=(2.7,2.7))
        #cax = ax.imshow(spearman[j],cmap=matplotlib.cm.RdBu,vmin=-1,vmax=1,extent=(-0.5, spearman.shape[2]-0.5, -0.5, spearman.shape[1]-0.5))
        cax = ax.imshow(spearman[j],cmap=matplotlib.cm.Blues,vmin=0,vmax=1,extent=(-0.5, spearman.shape[2]-0.5, -0.5, spearman.shape[1]-0.5))
        for ii in range(len(vecs)):
            for jj in range(len(vecs)): 
                ax.text(ii,3-jj,str(round(spearman[j,ii,jj],3)) if spearman[j,ii,jj]<1 else '1',va='center',ha='center',color='k' if spearman[j,ii,jj]<0.65 else 'white')
        #cbar = ax.figure.colorbar(im, ax=ax)
        #cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")
        ax.set_xticks(np.arange(n_types))
        ax.set_yticks(np.arange(n_types))
        ax.set_xticklabels(types,rotation=90)
        ax.set_yticklabels(types[::-1])
        ax.set_ylim([-.5,(n_types)-.5])
        ax.set_xlim([-.5,(n_types)-.5])
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        caxax = divider.append_axes("right", size="4%", pad=0.1)
        cbar=f.colorbar(cax,cax=caxax)
        cbar.ax.set_ylabel('Spearman correlation (Scenario %i)' % (j+1), rotation=-90, va="bottom")
        cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")
        plt.savefig(('%s_spearman_correlations_scenario%i' % (infix,j+1))+ending,bbox_inches = "tight")
    
    #types= ['deaths','YLL','cases','infections','equity']
    ##types= ['Deaths','YLL','Cases','Infections']
    #ending = '.pdf'
    #spearman = np.ones((nr_scenarios,len(types),len(types)))
    #n_types=len(types)
    #for j in range(nr_scenarios):
    #    #vecs = [deathss[j],casess[j],total_infectionss[j],yllss[j]]
    #    #vecs = [deathss[j],casess[j],yllss[j],entropiess[j]]
    #    #vecs = [deathss[j],yllss[j],casess[j],infectionss[j]]
    #    vecs = [deathss[j],yllss[j],casess[j],infectionss[j],-entropiess[j]]
    #    for ii in range(len(vecs)):
    #        for jj in range(ii+1,len(vecs)):
    #            spearman[j,ii,jj] = stats.spearmanr(vecs[ii],vecs[jj])[0]
    #            spearman[j,jj,ii] = spearman[j,ii,jj]
    #
    #    f,ax = plt.subplots(figsize=(3.4,3.4))
    #    cax = ax.imshow(spearman[j],cmap=matplotlib.cm.RdBu,vmin=-1,vmax=1,extent=(-0.5, spearman.shape[2]-0.5, -0.5, spearman.shape[1]-0.5))
    #    for ii in range(len(vecs)):
    #        for jj in range(len(vecs)): 
    #            ax.text(ii,len(types)-1-jj,str(round(spearman[j,ii,jj],3)) if spearman[j,ii,jj]<1 else '1',va='center',ha='center',color='k' if spearman[j,ii,jj]<0.7 else 'white')
    #    #cbar = ax.figure.colorbar(im, ax=ax)
    #    #cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")
    #    ax.set_xticks(np.arange(n_types))
    #    ax.set_yticks(np.arange(n_types))
    #    ax.set_xticklabels(types,rotation=90)
    #    ax.set_yticklabels(types[::-1])
    #    ax.set_ylim([-.5,(n_types)-.5])
    #    ax.set_xlim([-.5,(n_types)-.5])
    #    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #    divider = make_axes_locatable(ax)
    #    caxax = divider.append_axes("right", size="4%", pad=0.1)
    #    cbar=f.colorbar(cax,cax=caxax)
    #    cbar.ax.set_ylabel('Spearman correlation (Scenario %i)' % (j+1), rotation=-90, va="bottom")
    #    cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")
    #    plt.savefig(('%s_spearman_correlations_scenario%i_5att' % (infix,j+1))+ending,bbox_inches = "tight")
    
    
    
    
    
    
    
    
    
    
    
    
    #pts = np.c_[death_counts,total_cases]
    #pareto_pts = get_pareto_frontier(pts)
    #which = is_pareto_efficient(pareto_pts)
    #pareto_pts = pareto_pts[which==True]
    #indices = sorted(range(pareto_pts.shape[0]),key=lambda x: pareto_pts[x,0])
    #plt.plot(pareto_pts[indices,0],pareto_pts[indices,1])
    
    
    
    
    
    
    ##takes 34 seconds
    #allocations = np.array(list(map(source.get_i1_to_i17_which_satisfy_all_constraints,allocation_idss[0])))
    #
    #A,indices = is_pareto_efficient_3d(total_deaths,total_cases,ylls,True)
    #A = np.array(A)
    #AA = pd.DataFrame(np.c_[A[:,0]/1e3,A[:,1]/1e6,A[:,2]/1e6,[int(el[0]<deaths_CDC)+int(el[1]<cases_CDC)+int(el[2]<ylls_CDC) for el in A],allocations[indices]],columns = ['deaths','cases','YLL','pareto dominant over CDC']+list(map(str,range(1,18))))
    #AA.to_excel('%s_pareto_frontier_3d.xlsx' % infix)
    #
    #
    #types= ['Deaths','Cases','Infections','YLL','Equity']
    #types_legend = ['fewest deaths','fewest cases','fewest infections','lowest YLL','highest equity']
    #types_legend = ['fewest deaths','fewest cases','fewest infections','lowest YLL','highest equity']
    #for i in range(nr_scenarios):
    #    deaths_optimal,cases_optimal,infections_optimal,ylls_optimal,entropies_optimal = np.zeros(len(types)),np.zeros(len(types)),np.zeros(len(types)),np.zeros(len(types)),np.zeros(len(types))
    #    for j in range(len(types)):
    #        result = source.fitfunc_short(np.array(optimal_allocationss[i][j]),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #
    #        (deaths_optimal[j],cases_optimal[j],infections_optimal[j],ylls_optimal[j],entropies_optimal[j]) = get_outcomes(result)
    #    
    #    result_CDC = source.fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #    (deaths_CDC,cases_CDC,infections_CDC,ylls_CDC,entropies_CDC) = get_outcomes(result_CDC)
    #    result_equal = source.fitfunc_short(np.ones(17,dtype=np.float64),initial_values,beta,q,midc,exponent,source2.f_A,source2.f_V,source2.sigma,source2.delta)
    #    (deaths_equal,cases_equal,infections_equal,ylls_equal,entropies_equal) = get_outcomes(result_equal)
    #
    #
    #    death_counts=deathss[i]
    #    total_cases =casess[i]
    #
    #    dummy = is_pareto_efficient_2d(death_counts,total_cases,True)
    #    pareto,indices_pareto = np.array(dummy[0]),np.array(dummy[1])
    #    colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9)]
    #    f,ax = plt.subplots(figsize=(3,3))
    #    ax.plot(pareto[:,0],pareto[:,1],'k-',label='Pareto frontier')
    #    counter=0
    #    ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],label='CDC',zorder=3)
    #    for j in range(len(types_legend)):
    #        if j==2 or j==4:
    #            continue
    #        counter+=1
    #        ax.plot([deaths_optimal[j]],[cases_optimal[j]],'X',color=colors[counter],label=types_legend[j])  
    #    #ax.plot([deaths_equal],[cases_equal],'bo',label='equal allocation')
    #    ax.legend(loc=0,frameon=False)
    #    ax.spines['top'].set_visible(False)
    #    ax.spines['right'].set_visible(False)
    #    ax.set_xlabel('deaths [thousands]')
    #    ax.set_xticklabels([int(el//1000) for el in ax.get_xticks()])
    #    ax.set_ylabel('cases [millions]')
    #    ax.set_yticklabels([round(el/1000000,1) for el in ax.get_yticks()])    
    #    plt.savefig('%s_overall_pareto_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    #        
    #    allocations_pareto = allocations[indices_pareto]    
    #    df = pd.DataFrame(np.c_[pareto,allocations_pareto+1])
    #    df.to_csv('pareto_frontier.csv')
    #    
    #    
    #    pts = np.c_[death_counts,total_cases]
    #    indices = spatial.ConvexHull(pts).vertices
    #    ordered = sort_convex_hull_points(pts[indices])
    #    argmin = np.argmin(ordered,0)[1]
    #    f,ax = plt.subplots(figsize=(5.7,3))
    #    ax.plot(ordered[argmin:,0],ordered[argmin:,1],'k--')
    #    ax.plot(pareto[:,0],pareto[:,1],'k-')
    #    colors = [cm.tab10(1),cm.tab10(4),cm.tab10(2),cm.tab10(9),cm.tab10(3)]
    #    counter=0
    #    ax.plot([deaths_CDC],[cases_CDC],'X',color=colors[counter],label='CDC',zorder=3)
    #    for j in range(len(types_legend)):
    #        if j==2:
    #            continue
    #        counter+=1
    #        ax.plot([deaths_optimal[j]],[cases_optimal[j]],'X',color=colors[counter],label=types_legend[j])  
    #    ax.plot([deaths_equal],[cases_equal],'X',color=cm.Set1(8),label='no phases')
    #    #ax.legend(loc='best',frameon=False)
    #    ax.set_xlabel('deaths [thousands]')
    #    ax.set_xticklabels([int(el//1000) for el in ax.get_xticks()])
    #    ax.set_ylabel('cases [millions]')
    #    ax.spines['top'].set_visible(False)
    #    ax.spines['right'].set_visible(False)
    #    ax.set_yticklabels([int(el//1000000) for el in ax.get_yticks()]) 
    #    ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3,frameon=False)
    #    plt.savefig('%s_overall_convex_hull_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    #
    #
    #    which = np.random.randint(0,allocation_ids.shape[0],5000)
    #    f,ax = plt.subplots()
    #    ax.plot(ordered[:,0],ordered[:,1],'k-',label='convex hull')
    #    ax.scatter(death_counts[which],total_cases[which],c=ylls[which],cmap=cm.viridis)
    #    ax.set_xlabel('total death count')
    #    ax.set_ylabel('total cases')
    #    plt.savefig('%s_random5000points_colored_by_yll_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    #
    #    which = np.random.randint(0,allocation_ids.shape[0],50000)
    #    f,ax = plt.subplots()
    #    ax.plot(ordered[:,0],ordered[:,1],'k-',label='convex hull')
    #    ax.scatter(death_counts[which],total_cases[which],c=entropies[which],cmap=cm.viridis)
    #    ax.set_xlabel('total death count')
    #    ax.set_ylabel('total cases')
    #    plt.savefig('%s_random5000points_colored_by_entropies_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    #
    #
    #
    #    pts = np.c_[death_counts,entropies]
    #    indices = spatial.ConvexHull(pts).vertices
    #    ordered = sort_convex_hull_points(pts[indices])
    #    f,ax = plt.subplots()
    #    ax.plot(ordered[:,0],ordered[:,1],'k-',label='convex hull')
    #    ax.plot([deaths_CDC],[entropies_CDC],'o',color=cm.Set1(0),label='CDC')
    #    counter=0
    #    for j in range(len(types)):
    #        if j==2:
    #            continue
    #        counter+=1
    #        ax.plot([deaths_optimal[j]],[entropies_optimal[j]],'o',color=cm.Set1(counter),label=types_legend[j])    
    #    ax.legend(loc='best')
    #    ax.set_xlabel('total death count')
    #    ax.set_ylabel('entropies')
    #    ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)
    #    plt.savefig('%s_overall_convex_hull_death_entropy_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    #
    #
    #    pts = np.c_[total_cases,entropies]
    #    indices = spatial.ConvexHull(pts).vertices
    #    ordered = sort_convex_hull_points(pts[indices])
    #    f,ax = plt.subplots()
    #    ax.plot(ordered[:,0],ordered[:,1],'k-',label='convex hull')
    #    ax.plot([cases_CDC],[entropies_CDC],'o',color=cm.Set1(0),label='CDC')
    #    counter=0
    #    for j in range(len(types)):
    #        if j==2:
    #            continue
    #        counter+=1
    #        ax.plot([cases_optimal[j]],[entropies_optimal[j]],'o',color=cm.Set1(counter),label=types_legend[j])    
    #    ax.legend(loc='best')
    #    ax.set_xlabel('total cases')
    #    ax.set_ylabel('entropies')
    #    ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)
    #    plt.savefig('%s_overall_convex_hull_cases_entropy_scenario%i%s' % (infix,i+1,ending),bbox_inches = "tight")
    #
    #
    #
    
    
    
    
    









#for which_group in [0,1,7,14]:
#    for j,hesitancy in enumerate([0.05]):
#        death_counts=deathss[j]
#        total_cases =casess[j]
#    
#    #    which = np.random.randint(0,allocation_ids.shape[0],1000)
#    #    colors = ['r','g','c','k']
#    #    f,ax = plt.subplots()
#    #    for i in range(4):
#    #        which_x = allocations[which,which_group]==i
#    #        ax.plot(death_counts[which][which_x],total_cases[which][which_x],'o',color=colors[i],alpha=0.5,label='phase '+str(i+1))
#    #    ax.legend(loc='best',title='Group %i in' % (which_group+1))
#    #    ax.set_xlabel('total death count')
#    #    ax.set_ylabel('total cases')
#    #    plt.savefig('%s_random1000points_stratified_by_group%i_scenario%i%s' % (infix,j+1,which_group+1,ending),bbox_inches = "tight")
#    #    
#        #show pareto frontier for specific subsets, e.g. kids in phase 1,2,3,4
#        colors = ['r','g','c','k']
#        pts = np.c_[death_counts,total_cases]
#        f,ax = plt.subplots()
#        pareto_ptss = []
#        for i in range(4):
#            which_pts = allocations[:,which_group]==i
#            #pareto_pts = get_pareto_frontier(which_pts)
#            #which = is_pareto_efficient(pareto_pts)
#            #pareto_pts = pareto_pts[which==True]
#            #indices = sorted(range(pareto_pts.shape[0]),key=lambda x: pareto_pts[x,0])
#            pareto_pts = np.array(is_pareto_efficient_2d(pts[which_pts,0],pts[which_pts,1]))
#            ax.plot(pareto_pts[:,0],pareto_pts[:,1],color=colors[i],label='phase '+str(i+1))
#            pareto_ptss.append(pareto_pts)
#        #ax.set_title('Pareto frontier for fixed phase assignments for group')
#        ax.legend(loc='best',title='Group %i in' % (which_group+1))
#        ax.set_xlabel('total death count')
#        ax.set_ylabel('total cases')
#        plt.savefig('%s_paretofrontier_stratified_by_group%i_scenario%i%s' % (infix,j+1,which_group+1,ending),bbox_inches = "tight")

for j in range(nr_scenarios):
    pareto_ptss = [[] for i in range(17)]
    death_counts=deathss[j]
    total_cases =casess[j]    
    pts = np.c_[death_counts,total_cases]
    for which_group in range(17):
        for i in range(4):
            which_pts = allocations[:,which_group]==i        
            pareto_pts = np.array(is_pareto_efficient_2d(pts[which_pts,0],pts[which_pts,1]))
            pareto_ptss[which_group].append(pareto_pts)

    colors = [cm.tab10(1),'g','c','k']
#    for which_group in range(17):
#        f,ax = plt.subplots(figsize=(4,3))
#        for i in range(4):
#            ax.plot(pareto_ptss[which_group][i][:,0],pareto_ptss[which_group][i][:,1],color=colors[i],label='phase '+str(i+1))
#        #ax.set_title('Pareto frontier for fixed phase assignments for group')
#        ax.legend(loc='best',title='Group %i in' % (which_group+1))
#        ax.set_xlabel('total death count')
#        ax.set_ylabel('total cases')
#        ax.set_xlabel('deaths [thousands]')
#        ax.set_xticklabels([int(el//1000) for el in ax.get_xticks()])
#        ax.set_ylabel('cases [millions]')
#        ax.spines['top'].set_visible(False)
#        ax.spines['right'].set_visible(False)
#        ax.set_yticklabels([round(el/1000000,1) for el in ax.get_yticks()]) 
#        plt.savefig('%s_paretofrontier_stratified_by_group%i_scenario%i%s' % (infix,which_group+1,j+1,ending),bbox_inches = "tight")
#        
    maxvalue = max([np.max(list(map(lambda x: np.max(x,0),pareto_ptss[i]))) for i in range(17)])
    f,ax = plt.subplots(nrows=5,ncols=4,sharey=True,sharex=True,figsize=(10,12))
    for k,which_group in enumerate(range(17)):
        if k==0:
            k-=3
        for i in range(4):
            ax[(k+3)//4,(k+3)%4].plot(pareto_ptss[which_group][i][:,0],pareto_ptss[which_group][i][:,1],color=colors[i],label='phase '+str(i+1))
        #ax.set_title('Pareto frontier for fixed phase assignments for group')
        #ax.legend(loc='best',title='Group %i in' % (which_group+1))
        if (k+3)%4==0 or k==-3:
            ax[(k+3)//4,(k+3)%4].set_ylabel('cases [millions]')
        if k>12:
            ax[(k+3)//4,(k+3)%4].set_xlabel('deaths [thousands]')
        ax[(k+3)//4,(k+3)%4].set_xticklabels([int(el//1000) for el in ax[(k+3)//4,(k+3)%4].get_xticks()])
        ax[(k+3)//4,(k+3)%4].spines['top'].set_visible(False)
        ax[(k+3)//4,(k+3)%4].spines['right'].set_visible(False)
        ax[(k+3)//4,(k+3)%4].set_yticklabels([int(el//1000000) for el in ax[(k+3)//4,(k+3)%4].get_yticks()])
        ax[(k+3)//4,(k+3)%4].text(ax[(k+3)//4,(k+3)%4].get_xlim()[0] + 0.6*(ax[(k+3)//4,(k+3)%4].get_xlim()[1]-ax[(k+3)//4,(k+3)%4].get_xlim()[0]),maxvalue,'Group %i' % (which_group+1),va='top',ha='center')
        if k==-3:
            ax[(k+3)//4,(k+3)%4].legend(bbox_to_anchor=(2.9,0.5), loc="center",frameon=False,title='Group X in')
            ax[(k+3)//4,(k+3)%4].set_yticklabels([round(el/1000000,1) for el in ax[(k+3)//4,(k+3)%4].get_yticks()])                
    for k in range(1,4):
        ax[0,k].axis('off')
        
    plt.savefig('%s_paretofrontier_stratified_by_all_scenario%i%s' % (infix,j+1,ending),bbox_inches = "tight")
           
    
    colors = [cm.tab10(1),'g','c','k']    
    f,ax = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(3,6.4))
    for k,(which_group,name) in enumerate(zip([0,1,7],['children','healthcare workers\nwithout comorbidities','16-64 year olds\nwithout comorbidities'])):
        for i in range(4):
            ax[k].plot(pareto_ptss[which_group][i][:,0],pareto_ptss[which_group][i][:,1],color=colors[i],label=str(i+1))
        #ax.set_title('Pareto frontier for fixed phase assignments for group')
        if k==1:
            ax[k].legend(bbox_to_anchor=(1,0.5), loc="center",title='Phase',frameon=False)
        ax[k].set_ylabel('cases [millions]')
        if k==2:
            ax[k].set_xlabel('deaths [thousands]')
#        ax[k].text(750000,43.4*1e6,'Group %i' % (which_group+1),va='center',ha='center')
    ylims = np.max(np.array([ax[k].get_ylim() for k in range(3)]),0)
    for k,(which_group,name) in enumerate(zip([0,1,7],['children','healthcare workers\nwithout comorbidities','16-64 year olds\nwithout comorbidities'])):
        ax[k].set_ylim(ylims+np.array([0,1.5e6]))
        ax[k].text(np.mean(ax[k].get_xlim()),ax[k].get_ylim()[1],name,va='top',ha='center')
        ax[k].set_xticklabels([int(el//1000) for el in ax[k].get_xticks()])
        ax[k].spines['top'].set_visible(False)
        ax[k].spines['right'].set_visible(False)
        ax[k].set_yticklabels([int(el//1000000) for el in ax[k].get_yticks()])        #ax[k].set_ylim([39*1e6,43.2*1e6])
        #ax[k].set_xlim([690000,825000])
        #if k==-3:
        #    ax[(k+3)//4,(k+3)%4].legend(bbox_to_anchor=(2.9,0.5), loc="center",frameon=False,title='Group X in')
        #    ax[(k+3)//4,(k+3)%4].set_yticklabels([round(el/1000000,1) for el in ax[(k+3)//4,(k+3)%4].get_yticks()])                
        
    plt.savefig('%s_paretofrontier_stratified_by_selected_scenario%i%s' % (infix,j+1,ending),bbox_inches = "tight")
            













colors = ['r','g','c','k']
pts = np.c_[death_counts,total_cases]
f,ax = plt.subplots()
pareto_ptss = []
for i in range(4):
    which_pts = pts[allocations[:,0]==i,:]
    indices = spatial.ConvexHull(which_pts).vertices
    ax.plot(which_pts[indices,0],which_pts[indices,1],'o',color=colors[i])


#...


OPTIMAL_ID_DEATHS = allocation_ids[np.argmin(death_counts)]
optimal_allocation_deaths = source.get_i1_to_i17_which_satisfy_all_constraints(OPTIMAL_ID_DEATHS)

OPTIMAL_ID_CASES = allocation_ids[np.argmin(total_cases)]
optimal_allocation_cases = source.get_i1_to_i17_which_satisfy_all_constraints(OPTIMAL_ID_CASES)




n_intervals = 10
ve = 0.9
fvs = np.linspace(0,1,n_intervals+1)
deltas = np.linspace(1-np.sqrt(1-ve),ve,n_intervals//2+1)
deltas = np.append(1-(1-ve)/(1-deltas[1:]), deltas )
deltas_p = np.sort(deltas)
sigmas_p = deltas_p[len(deltas_p)-1::-1]

f,ax=plt.subplots(figsize=(2.5,2.5))
ves = [0.5,0.75,ve]
for ii,ve in enumerate(ves):
    deltas = np.arange(0.01,min(0.99,ve),0.0001)
    sigmas = 1-(1-ve)/(1-deltas)
    ax.plot(deltas,sigmas,label=str(int(ve*100))+'%',color=cm.tab10(ii*2))
    if ii==2:
        ax.plot(deltas_p,sigmas_p,'o',color=cm.tab10(ii*2))
    x = 1-np.sqrt(1-ve)
    ax.text(x+0.06,x+0.06,str(int(ve*100))+'%',va='center',ha='center')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xlabel(r'$\sigma = \frac{Prob(S^V \to\ E^V)}{Prob(S^W \to\ E^W)}$')
ax.set_ylabel(r'$\delta = \frac{Prob(E^V \to\ P^V)}{Prob(E^W \to\ P^W)}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2,title='vaccine effectiveness')
plt.savefig('ve_vs_sigma_and_delta.pdf',bbox_inches = "tight")

f,ax=plt.subplots(figsize=(2.5,2.5))
ves = [0.5,0.75,ve]
for ii,ve in enumerate(ves):
    deltas = np.arange(0.01,min(0.99,ve),0.0001)
    sigmas = 1-(1-ve)/(1-deltas)
    ax.plot(deltas,sigmas,label=str(int(ve*100))+'%',color='k')

    x = 1-np.sqrt(1-ve)
    ax.text(x+0.06,x+0.06,str(int(ve*100))+'%',va='center',ha='center')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xticklabels(['0%','50%','100%'])
ax.set_yticklabels(['0%','50%','100%'])
ax.set_xlabel('vaccine-induced reduction of\nsusceptibility to infection '+r'$(\sigma)$')
ax.set_ylabel('vaccine-induced reduction of\nsymptoms (when infected) '+r'$(\delta)$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2,title='vaccine effectiveness')
plt.savefig('ve_vs_sigma_and_delta_v2.pdf',bbox_inches = "tight")




dt=0.5
ts = np.arange(-62, 365, dt)  
#ts = np.linspace(0, 365, int(T/dt)+1)  
#_,_,_,_,_,Y = fitfunc(vacopt,beta)
fv=1.
f_A=0.75

n_intervals = 10
ve = 0.9
fvs = np.linspace(0,1,n_intervals+1)
deltas = np.linspace(1-np.sqrt(1-ve),ve,n_intervals//2+1)
deltas = np.append(1-(1-ve)/(1-deltas[1:]), deltas )
deltas = np.sort(deltas)
sigmas = deltas[len(deltas)-1::-1]

deaths = np.zeros((n_intervals+1,n_intervals+1))
cases = np.zeros((n_intervals+1,n_intervals+1))

q = source2.q_based_on_q17(q17)
# source2.hesitancy = hesitancy

# source2.get_initial_values.recompile()
# fitfunc_short.recompile()
# source2.SYS_ODE_VAX_RK4.recompile()
# source2.RK4.recompile()

# vacopts = []
# vacopts_ids = []
# res_from_runs = []
initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)



for ii,(sigma,delta) in enumerate(zip(sigmas,deltas)):
    for jj,f_V in enumerate(fvs):
        #result_CDC = source.fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,f_A,f_V,sigma*np.ones(17,dtype=np.float64),delta*np.ones(17,dtype=np.float64))
        result_CDC = source.fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,f_A,f_V,sigma*np.ones(17,dtype=np.float64),delta*np.ones(17,dtype=np.float64))
        deaths[ii,jj] = np.sum(result_CDC[:4])
        cases[ii,jj] = np.sum(result_CDC[4:8])
        
f,ax=plt.subplots(figsize=(4,2.5))
im=ax.imshow(deaths,cmap=cm.inferno_r,aspect='auto')
ax.set_yticks(range(n_intervals+1))
ax.set_yticklabels(list(map(str,map(lambda x: round(x,2),sigmas))))
ax.set_ylabel(r'$\sigma$')
ax.set_xticks(range(n_intervals+1))
ax.set_xticklabels(list(map(str,map(lambda x: round(x,1),fvs))),rotation=90)
ax.set_xlabel(r'$f_V$')
ax2 = ax.twinx()
ax2.set_yticks(range(n_intervals+1))
ax2.set_yticklabels(list(map(str,map(lambda x: round(x,2),deltas))))
ax2.set_ylabel(r'$\delta$')
ax2.set_ylim(ax.get_ylim())
ax.autoscale(False)
ax2.autoscale(False)
f.subplots_adjust(left=0.1,right=0.6,bottom=0.15,top=0.9)
cbar_ax = f.add_axes([0.78, 0.15, 0.03, 0.75])
cbar = f.colorbar(im, cax=cbar_ax)
#cbar = f.colorbar(im)
#ticks = np.log2([0.01,0.1,1,5,10])
cbar.set_ticks(cbar.get_ticks())
cbar.set_ticklabels([str(int(el//1000)) for el in cbar.get_ticks()])
cbar.set_label('deaths [thousands]')
plt.savefig('%s_sigma_delta_fv_deaths_base_scenario%i_ve%s%s' % (infix,1,ve,ending),bbox_inches = "tight")

f,ax=plt.subplots(figsize=(4,2.5))
im=ax.imshow(cases,cmap=cm.inferno_r,aspect='auto')
ax.set_yticks(range(n_intervals+1))
ax.set_yticklabels(list(map(str,map(lambda x: round(x,2),sigmas))))
ax.set_ylabel(r'$\sigma$')
ax.set_xticks(range(n_intervals+1))
ax.set_xticklabels(list(map(str,map(lambda x: round(x,1),fvs))),rotation=90)
ax.set_xlabel(r'$f_V$')
ax2 = ax.twinx()
ax2.set_yticks(range(n_intervals+1))
ax2.set_yticklabels(list(map(str,map(lambda x: round(x,2),deltas))))
ax2.set_ylabel(r'$\delta$')
ax2.set_ylim(ax.get_ylim())
ax.autoscale(False)
ax2.autoscale(False)
f.subplots_adjust(left=0.1,right=0.6,bottom=0.15,top=0.9)
cbar_ax = f.add_axes([0.78, 0.15, 0.03, 0.75])
cbar = f.colorbar(im, cax=cbar_ax)
#cbar = f.colorbar(im)
#ticks = np.log2([0.01,0.1,1,5,10])
cbar.set_ticks(cbar.get_ticks())
cbar.set_ticklabels([str(int(el//1000000)) for el in cbar.get_ticks()])
cbar.set_label('cases [millions]')
plt.savefig('%s_sigma_delta_fv_cases_base_scenario%i_ve%s%s' % (infix,1,ve,ending),bbox_inches = "tight")





n_intervals = 20
ve = 0.9
fvs = np.linspace(0,1,n_intervals+1)
deltas = np.linspace(1-np.sqrt(1-ve),ve,n_intervals//2+1)
deltas = np.append(1-(1-ve)/(1-deltas[1:]), deltas )
deltas = np.sort(deltas)
sigmas = deltas[len(deltas)-1::-1]

sigmas = sigmas[::-1]
deltas = deltas[::-1]

deaths = np.zeros((n_intervals+1,3,3))
cases = np.zeros((n_intervals+1,3,3))

fvs = np.array([0,0.5,1])
fas = np.array([0.25,0.75,1])
initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)

for jj,f_V in enumerate(fvs):
    for kk,f_A in enumerate(fas):
        if jj!=1 and kk!=1:
            continue
        else:
            if jj==1 and kk==1:
                dummy=0
            elif jj==0 and kk==1:
                dummy=5
            elif jj==2 and kk==1:
                dummy=6
            elif jj==1 and kk==0:
                dummy=3
            elif jj==1 and kk==2:
                dummy=4
        param = np.array(np.array(source.data)[dummy if dummy<7 else 0,6:6+4],dtype=np.float64)
        beta = source2.parm_to_beta(param[:2])
        midc = param[2]
        exponent = param[3]**2

        initial_values=source2.get_initial_values(source2.Nsize,source2.mu_A,source2.mu_C,source2.mu_Q,q,hesitancy=source2.hesitancy,START_DATE_INDEX=source2.index_2020_10_13)


        for ii,(sigma,delta) in enumerate(zip(sigmas,deltas)):

            result_CDC = source.fitfunc_short(np.array(CDC_allocation),initial_values,beta,q,midc,exponent,f_A,f_V,sigma*np.ones(17,dtype=np.float64),delta*np.ones(17,dtype=np.float64))
            deaths[ii,jj,kk] = np.sum(result_CDC[:4])
            cases[ii,jj,kk] = np.sum(result_CDC[4:8])

lss = ['--','-',':']
f,ax = plt.subplots(figsize=(3,2))
for jj,f_V in enumerate(fvs):
    for kk,f_A in enumerate(fas):
        if jj!=1 and kk!=1:
            continue
        if kk!=1:
            continue
        ax.plot(deaths[:,jj,kk]/1e3,'k-',label='%i%%' % int(100*f_V),ls=lss[jj])
ax.legend(loc='upper center',frameon=False,title='relative contagiousness of vaccinated',ncol=3,bbox_to_anchor=[0.4,1.33])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(range(0,n_intervals+1,4))
ax.set_xticklabels([str(int(round(x*100,0)))+'%' for x in sigmas[::4]])
for i,el in enumerate([str(int(round(x*100,0)))+'%' for x in deltas[::4]]):
    ax.text(4*i,530,el,va='center',ha='center',clip_on=False)
ax.set_ylim([512,ax.get_ylim()[1]])
ax.set_xlim([-2,ax.get_xlim()[1]])
ax.set_ylabel('deaths [thousands]')
ax.set_xlabel(r'$\sigma$')
ax.text(n_intervals//2,580,r'$\delta$',va='center',ha='center',clip_on=False)
plt.savefig('%s_sigma_delta_fv_deaths_ve%s%s' % (infix,ve,ending),bbox_inches = "tight")

lss = ['--','-',':']
f,ax = plt.subplots(figsize=(3.5,2.5))
for jj,f_V in enumerate(fvs):
    for kk,f_A in enumerate(fas):
        if jj!=1 and kk!=1:
            continue
        if jj!=1:
            continue
        ax.plot(deaths[:,jj,kk]/1e3,'-',label=r'$f_A=$'+'%i%%' % int(100*f_A),ls=lss[kk])
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(range(0,n_intervals+1,4))
ax.set_xticklabels([str(int(round(x*100,0)))+'%' for x in sigmas[::4]])
for i,el in enumerate([str(int(round(x*100,0)))+'%' for x in deltas[::4]]):
    ax.text(4*i,550,el,va='center',ha='center',clip_on=False)
ax.set_ylim([535,ax.get_ylim()[1]])
ax.set_xlim([-2,ax.get_xlim()[1]])
ax.set_ylabel('deaths [thousands]')
ax.set_xlabel(r'$\sigma$')
ax.text(n_intervals//2,580,r'$\delta$',va='center',ha='center',clip_on=False)


















VE=0.9
f,ax = plt.subplots()
for alpha in [0.01,1,100,10000]:
    x=np.random.beta(alpha,alpha*(1-VE)/VE,10000)
    ax.hist(x,range=(0,1),bins=97,label=str(alpha))
    #print(np.mean(x),np.std(x))
ax.legend(title='alpha',loc='best')
ax.set_title(r'$(\alpha,\alpha \frac{1-VE}{VE})$-Beta distribution')
plt.savefig('beta.pdf')













sigmas = np.linspace(0,ve,n_intervals+1)
deltas = 1-(1-ve)/(1-sigmas)

deaths = np.zeros((n_intervals+1,n_intervals+1))
for ii,(sigma,delta) in enumerate(zip(sigmas,deltas)):
    for jj,fv in enumerate(fvs):
        Y=source.fitfunc(np.array(CDC_allocation),source.beta,hesitancy=source.hesitancy,sigma=sigma*np.ones(17,dtype=np.float64),delta=delta*np.ones(17,dtype=np.float64),fv=fv,fa=fa)[-1]
        deaths[ii,jj] = np.sum(Y[-1,17::20])
   
f,ax=plt.subplots(figsize=(4,2.5))
im=ax.imshow(deaths,cmap=cm.inferno_r,aspect='auto')
ax.set_yticks(range(n_intervals+1))
ax.set_yticklabels(list(map(str,map(lambda x: round(x,3),sigmas))))
ax.set_ylabel(r'$\sigma$')
ax.set_xticks(range(n_intervals+1))
ax.set_xticklabels(list(map(str,map(lambda x: round(x,1),fvs))),rotation=90)
ax.set_xlabel(r'$f_V$')
ax2 = ax.twinx()

ax2.set_yticks(range(n_intervals+1))
ax2.set_yticklabels(list(map(str,map(lambda x: round(x,3),deltas))))
ax2.set_ylabel(r'$\delta$')
ax2.set_ylim(ax.get_ylim())
ax.autoscale(False)
ax2.autoscale(False)
f.subplots_adjust(left=0.1,right=0.6,bottom=0.15,top=0.9)
cbar_ax = f.add_axes([0.78, 0.15, 0.03, 0.75])
cbar = f.colorbar(im, cax=cbar_ax)
#cbar = f.colorbar(im)
#ticks = np.log2([0.01,0.1,1,5,10])
cbar.set_ticks(cbar.get_ticks())
cbar.set_ticklabels([str(int(el//1000)) for el in cbar.get_ticks()])
cbar.set_label('deaths [thousands]')
plt.savefig('%s_sigma_delta_fv_deaths_scenario%i%s' % (infix,1,ending),bbox_inches = "tight")



deltas = np.linspace(0,ve,n_intervals+1)
sigmas = 1-(1-ve)/(1-deltas)

deaths = np.zeros((n_intervals+1,n_intervals+1))
for ii,(sigma,delta) in enumerate(zip(sigmas,deltas)):
    for jj,fv in enumerate(fvs):
        Y=source.fitfunc(np.array(CDC_allocation),source.beta,hesitancy=source.hesitancy,sigma=sigma*np.ones(17,dtype=np.float64),delta=delta*np.ones(17,dtype=np.float64),fv=fv,fa=fa)[-1]
        deaths[ii,jj] = np.sum(Y[-1,17::20])
   

f,ax=plt.subplots(figsize=(4,2.5))
im=ax.imshow(deaths,cmap=cm.inferno_r,aspect='auto')
ax.set_yticks(range(n_intervals+1))
ax.set_yticklabels(list(map(str,map(lambda x: round(x,3),deltas))))
ax.set_ylabel(r'$\delta$')
ax.set_xticks(range(n_intervals+1))
ax.set_xticklabels(list(map(str,map(lambda x: round(x,1),fvs))),rotation=90)
ax.set_xlabel(r'$f_V$')
ax2 = ax.twinx()

ax2.set_yticks(range(n_intervals+1))
ax2.set_yticklabels(list(map(str,map(lambda x: round(x,3),sigmas))))
ax2.set_ylabel(r'$\sigma$')
ax2.set_ylim(ax.get_ylim())
ax.autoscale(False)
ax2.autoscale(False)
f.subplots_adjust(left=0.1,right=0.6,bottom=0.15,top=0.9)
cbar_ax = f.add_axes([0.78, 0.15, 0.03, 0.75])
cbar = f.colorbar(im, cax=cbar_ax)
#cbar = f.colorbar(im)
#ticks = np.log2([0.01,0.1,1,5,10])
cbar.set_ticks(cbar.get_ticks())
cbar.set_ticklabels([str(int(el//1000)) for el in cbar.get_ticks()])
cbar.set_label('deaths [thousands]')
plt.savefig('%s_sigma_delta_fv_flipped_scenario%i%s' % (infix,1,ending),bbox_inches = "tight")

















scen=1
(source.hesitancy,source.sigma,source.delta) = source.get_globals(scen+1)
source.SYS_ODE_VAX_RK4.recompile()
source.mat_vecmul.recompile()
source.RK4.recompile()
source.fitfunc.recompile() 

dt=0.5
ts = np.arange(-62, 365, dt)  
#ts = np.linspace(0, 365, int(T/dt)+1)  
#_,_,_,_,_,Y = fitfunc(vacopt,beta)
Y=source.fitfunc(np.array(CDC_allocation),source.beta,hesitancy=source.hesitancy,sigma=source.sigma,delta=source.delta)[-1]



Ncomp=20

# initial_2020_12_14=Y[62*2,::] 
sol_by_compartment = np.zeros((ts.shape[0],Ncomp),dtype=np.float64) 

for i in range(Ncomp):
    sol_by_compartment[:,i]=np.sum(Y[:,i::Ncomp],axis=1) 
    

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
plt.savefig('%s_dynamics_CDC_scenario%i_linear%s' % (infix,scen,ending),bbox_inches = "tight")

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
plt.savefig('%s_dynamics_CDC_scenario%i_linear%s' % (infix,scen,ending),bbox_inches = "tight")




f,ax = plt.subplots(figsize=(4,3))
# ax.semilogy(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1] + sol_by_compartment[:,2],label='S')
ax.semilogy(ts, sol_by_compartment[:,0]+sol_by_compartment[:,1],label='unvaccinated\nsusceptible')
ax.plot(ts,sol_by_compartment[:,2],color=cm.tab10(4),label='vaccinated\nsusceptible')
ax.plot(ts,sol_by_compartment[:,3] + sol_by_compartment[:,4] + sol_by_compartment[:,5]+sol_by_compartment[:,6] + sol_by_compartment[:,7] + sol_by_compartment[:,8]+sol_by_compartment[:,12] + sol_by_compartment[:,13] + sol_by_compartment[:,14]+sol_by_compartment[:,15] + sol_by_compartment[:,16]+sol_by_compartment[:,19],color=cm.tab10(3),label='infected')
ax.plot(ts,sol_by_compartment[:,9] + sol_by_compartment[:,10] + sol_by_compartment[:,11]+sol_by_compartment[:,18],color=cm.tab10(2),label='recovered')
ax.plot(ts,sol_by_compartment[:,17],color='gray',label='dead')
ax.set_ylim([1,4*1e8])
ax.set_ylabel('US population')
last_day = len(source.cumcases[source.index_2020_10_13:])-62
xticks = [-62,0,last_day,365]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(xticks)
ax.set_xticklabels([el+'\n'+year for el,year in zip(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)),['2020','2020','2021','2021'])])
#ax.legend(loc='best',frameon=False)
ax.legend(bbox_to_anchor=(1,0, 0.5, 1), loc="right", mode="expand",frameon=False)
plt.savefig('%s_dynamics_CDC_scenario%i_log%s' % (infix,scen,ending),bbox_inches = "tight")



f,ax = plt.subplots()
ax.plot(ts,sol_by_compartment[:,17],label='D')
ax.plot(np.arange(-62, -62+len(source.cumdeaths[source.index_2020_10_13:])) ,source.cumdeaths[source.index_2020_10_13:],label='real D')
ax.legend(loc='best',title='cum deaths')
xticks = [-62,0,last_day,365]
ax.set_xticks(xticks)
ax.set_xticklabels(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)))

f,ax = plt.subplots()
last_day = len(source.cumcases[source.index_2020_10_13:])-62
ax.plot(ts,sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19],label='C')
ax.plot(np.arange(-62, last_day) ,source.cumcases[source.index_2020_10_13:],label='real D')
ax.legend(loc='best',title='cum cases')
xticks = [-62,0,last_day,365]
ax.set_xticks(xticks)
ax.set_xticklabels(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)))


f,ax = plt.subplots(figsize=(4,3))
last_day = len(source.cumcases[source.index_2020_10_13:])-62
ax.plot(ts,sol_by_compartment[:,17],color=cm.tab10(0),label='model')
ax.plot(np.arange(-62, last_day) ,source.cumdeaths[source.index_2020_10_13:],ls='--',lw=2,color=cm.tab10(1),label='US data')
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
ax2.plot(np.arange(-62, last_day) ,source.cumcases[source.index_2020_10_13:],ls='--',lw=2,color=cm.tab10(4),label='US data')
ax2.set_ylabel('cumulative cases [millions]')
ax2.set_yticklabels([int(el//1000000) for el in ax2.get_yticks()]) 
ax2.legend(loc=4,title='cumulative cases',frameon=False)
ax2.set_ylim([5*1e6,32.2*1e6])
ax2.spines['top'].set_visible(False)
plt.savefig('%s_fit%s' % (infix,ending),bbox_inches = "tight")

real_d = source.cumdeaths[source.index_2020_10_13:]
model_d = sol_by_compartment[:,17][::2][:len(source.cumdeaths[source.index_2020_10_13:])]

real_c = source.cumcases[source.index_2020_10_13:]
model_c = (sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19])[::2][:len(source.cumdeaths[source.index_2020_10_13:])]

pd.DataFrame(np.c_[real_d,model_d,real_d-model_d,real_c,model_c,real_c-model_c]).to_excel('%s_fit%s' % (infix,'.xlsx'))





#daily deaths/cases
import datetime
f,ax = plt.subplots(figsize=(4,3))
last_day = len(source2.cumcases[source2.index_2020_10_13:])-62
cum_deaths_real = source2.cumdeaths[source2.index_2020_10_13:]
cum_deaths_model = sol_by_compartment[:,17][::2][:len(cum_deaths_real)]
daily_deaths_real = source2.filtered_daily_deaths[source2.index_2020_10_13:]#cum_deaths_real[1:]-cum_deaths_real[:-1]
daily_deaths_model = cum_deaths_model[1:]-cum_deaths_model[:-1]
xticks = [-62+1,0,last_day]
ax.plot(np.arange(-61, last_day) ,daily_deaths_model,color=cm.tab10(0),label='model')
ax.plot(np.arange(-61, last_day) ,daily_deaths_real,ls='--',lw=2,color=cm.tab10(1),label='US data [7-day average]')
ax.set_xticks(xticks)
ax.set_xticklabels([el+'\n'+year for el,year in zip(list(map(lambda x: str(datetime.date(2020,12,14)+datetime.timedelta(days=x))[5:],xticks)),['2020','2020','2021','2021'])])
ax.legend(['',''],bbox_to_anchor=[0.2,1.15],title='            deaths',frameon=False, loc="center", mode="expand", ncol=1)
ax.set_ylabel('daily deaths')
#ax.set_yticklabels([int(el//1000) for el in ax.get_yticks()])
#ax.set_ylim([ax.get_ylim()[0],600000])
ax.spines['top'].set_visible(False)
[y1,y2] = ax.get_ylim()
ax.plot([0,0],[0,y2],'k:')
#ax.plot([last_day,last_day],[y1,y2],'k:')
ax.set_ylim([0,y2])

cum_cases_real = source2.cumcases[source2.index_2020_10_13:]
cum_cases_model = (sol_by_compartment[:,15]+sol_by_compartment[:,16]+sol_by_compartment[:,17]+sol_by_compartment[:,18]+sol_by_compartment[:,19])[::2][:len(cum_deaths_real)]
daily_cases_real = source2.filtered_daily_cases[source2.index_2020_10_13:]#cum_deaths_real[1:]-cum_deaths_real[:-1]
daily_cases_model = cum_cases_model[1:]-cum_cases_model[:-1]

ax2 = ax.twinx()
ax2.plot(np.arange(-61, last_day) ,daily_cases_model,color=cm.tab10(2),label='model')
ax2.plot(np.arange(-61, last_day) ,daily_cases_real,ls='--',lw=2,color=cm.tab10(4),label='US data [7-day average]')
ax2.set_ylabel('daily cases')
#ax2.set_yticklabels([int(el//1000000) for el in ax2.get_yticks()]) 
ax2.legend(bbox_to_anchor=[0.4,1.15],title='            cases',frameon=False, loc="center", mode="expand", ncol=1)
#ax2.set_ylim([5*1e6,32.2*1e6])
ax2.spines['top'].set_visible(False)
ax2.set_ylim([0,ax2.get_ylim()[1]])
plt.savefig('%s_fit_daily%s' % (infix,ending),bbox_inches = "tight")








# ax.plot(cumdeaths[index_2020_10_13:],label='real D')

# # ax.plot(np.arange(-62, -62+len(cumdeaths[index_2020_10_13:])) ,cumdeaths[index_2020_10_13:],label='real D')


f,ax = plt.subplots()
# ax.semilogy(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1] + sol_by_compartment[:,2],label='S')
# ax.plot(ts,sol_by_compartment[:,0] + sol_by_compartment[:,1] + sol_by_compartment[:,2],label='S')
# ax.plot(ts,sol_by_compartment[:,3] + sol_by_compartment[:,4] + sol_by_compartment[:,5],label='E')
# ax.plot(ts,sol_by_compartment[:,6] + sol_by_compartment[:,7] + sol_by_compartment[:,8],label='A')
# ax.plot(ts,sol_by_compartment[:,9] + sol_by_compartment[:,10] + sol_by_compartment[:,11],label='RA')
# ax.plot(ts,sol_by_compartment[:,12] + sol_by_compartment[:,13] + sol_by_compartment[:,14],label='P')
ax.plot(ts,sol_by_compartment[:,15] + sol_by_compartment[:,16],label='C')
ax.plot(ts,sol_by_compartment[:,17],label='D')
# ax.plot(ts,sol_by_compartment[:,18],label='R')
ax.legend(loc='best')


total_C = np.arange(0.001,10,0.05)
f,ax = plt.subplots()
for posneq_log10_dC_over_C in np.arange(-0.25,0.2501,0.125):
    ax.plot(total_C,1/(1+(5/total_C-posneq_log10_dC_over_C)**8),label=str(posneq_log10_dC_over_C))
ax.legend(loc='best',title='log10((C+dC)/C)')

total_C = np.arange(0,10,0.05)
f,ax = plt.subplots()
for t in [0,30,60,90,120,150]:
    ax.plot(total_C,1/(1+((5+t/180)/total_C)**8),label=str(t))
ax.legend(loc='best',title='t')


total_C = np.arange(0,8,0.05)
f,ax = plt.subplots(figsize=(5,3))
lss = ['--',':']
colors = [cm.tab10(0),cm.tab10(1)]
for i,mid in enumerate([4,6]):
    for j,exp in enumerate([2,16]):
        ax.plot(total_C,1/(1+(mid/total_C)**exp),ls=lss[i],color=colors[j],label='      %i        %i' % (mid,exp))      
mid = 5.632
exp = 15.62
ax.plot(total_C,1/(1+(mid/total_C)**exp),'k',lw=3,label='fit:  5.63   15.62')
ax.legend(loc='best',title='          c        k',frameon=False)
ax.set_xticklabels(list(map(lambda x: r'$10^{%i}$' % int(x),ax.get_xticks())))
ax.set_xlabel('active cases')
ax.set_ylabel('contact reduction')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('%s_r%s' % (infix,ending),bbox_inches = "tight")

#
#total_C = 10**np.arange(0,8,0.05)
#f,ax = plt.subplots()
#lss = ['--',':']
#colors = [cm.tab10(0),cm.tab10(1)]
#for i,mid in enumerate([4,6]):
#    for j,exp in enumerate([2,16]):
#        #ax.semilogx(total_C,1/(1+(10**mid/total_C)**exp),ls=lss[i],color=colors[j],label='      %i        %i' % (mid,exp))
#        ax.semilogx(total_C,1/(1+(mid/np.log10(total_C))**exp),ls=lss[i],color=colors[j],label='      %i        %i' % (mid,exp))
#        
#mid = 5.800705739
#exp = 3.429015227
#ax.semilogx(total_C,1/(1+(mid/np.log10(total_C))**exp),'k',lw=3,label='fit:  5.80   3.42')
#ax.legend(loc='best',title='          c        k',frameon=False)
##ax.set_xticklabels(list(map(lambda x: r'$10^{%i}$' % int(x),ax.get_xticks())))
#ax.set_xlabel('active cases')
#ax.set_ylabel('contact reduction')
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#plt.savefig('%s_r%s' % (infix,ending),bbox_inches = "tight")







f,ax = plt.subplots(figsize=(5,4))
im=ax.imshow(np.log2(source2.contact_matrix),cmap=cm.Greens,vmin=np.log2(0.01),vmax=np.max(np.log2(source2.contact_matrix)))
ax.set_yticks(range(17))
ax.set_yticklabels(list(map(str,range(1,17+1))))
ax.set_ylabel('Sub-population of Contactor')
ax.set_xticks(range(17))
ax.set_xticklabels(list(map(str,range(1,17+1))))
ax.set_xlabel('Sub-population of Contactee')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
#f.subplots_adjust(left=0.1,right=0.7,bottom=0.2,top=0.9)
#cbar_ax = f.add_axes([0.72, 0.212, 0.03, 0.6775])
#cbar = f.colorbar(im, cax=cbar_ax)
##cbar = f.colorbar(im)
#ticks = np.log2([0.01,0.1,1,5,10])
#cbar.set_ticks(ticks)
#cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
#cbar.set_label('average daily contacts')

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
caxax = divider.append_axes("right", size="4%", pad=0.1)
cbar=f.colorbar(im,cax=caxax)
ticks = np.log2([0.01,0.1,1,5,10])
cbar.set_ticks(ticks)
cbar.set_ticklabels([r'$\leq$0.01']+list(map(str,[0.1,1,5,10])))
cbar.set_label('average daily contacts')
 
plt.savefig('contact_matrix.pdf',bbox_inches = "tight")



f,ax = plt.subplots(figsize=(5,4))
ax.fill_between([0,14,21,35],[0,0,0,0],[0,80,80,90],label='Pfizer',alpha=0.3)
day=(35-(14*40+7*80+14*85)/(90*35)*35)
ax.fill_between([0,day,day+1e-10,35],[0,0,0,0],[0,0,90,90],label='Pfizer adjusted',alpha=0.3)
ax.set_xlabel('day post first shot')
ax.set_ylabel('% protection')
ax.set_yticks([0,80,90])
ax.set_xticks([0,day,14,21,35])

f,ax = plt.subplots(figsize=(5,4))
ax.fill_between([0,14,28,42],[0,0,0,0],[0,80,80,90],label='Moderna',alpha=0.3)
day=(42-(14*40+14*80+14*85)/(90*42)*42)
ax.fill_between([0,day,day+1e-10,42],[0,0,0,0],[0,0,90,90],label='Moderna adjusted',alpha=0.3)
ax.set_xlabel('day post first shot')
ax.set_ylabel('% protection')
ax.set_yticks([0,80,90])


f,ax = plt.subplots(figsize=(6,2.5))
ax.plot([0,14,21,35],[0,80,80,90],label='Pfizer')
day=(35-(14*40+7*80+14*85)/(90*35)*35)
ax.plot([0,day,day+1e-10,35],[0,0,90,90],label='model')
ax.set_xlabel('days post first shot')
ax.set_ylabel('% protection')
ax.set_yticks([0,80,90])
ax.legend(loc='best',frameon=False)
ax.set_xticks([0,day,14,21,35])
plt.savefig('Pfizer.pdf')

f,ax = plt.subplots(figsize=(6,2.5))
ax.plot([0,14,28,42],[0,80,80,90],label='Moderna')
day=(42-(14*40+14*80+14*85)/(90*42)*42)
ax.plot([0,day,day+1e-10,42],[0,0,90,90],label='model')
ax.set_xlabel('days post first shot')
ax.set_ylabel('% protection')
ax.set_yticks([0,80,90])
ax.legend(loc='best',frameon=False)
ax.set_xticks([0,day,14,28,42])
plt.savefig('Moderna.pdf')



A = pd.read_csv('variants.csv')
beta_multiplier = []
relative_transmissibility = np.array(A['transmissibility'])/100
dates_variants = [str(el)[:10] for el in A.columns[2:]]
dates_variants_index = [list(source2.dates).index(d)-7 for d in dates_variants]
A_data = np.array(A.iloc[:,2:])

def function(x,x0,k):
    return 1+0.5/(1+np.exp(-k*(x-x0)))

from scipy.optimize import curve_fit
params, covs = curve_fit(function, range(6),np.dot(relative_transmissibility,A_data))
xs = np.arange(-dates_variants_index[0]/14,30,1/14)
#
#f,ax = plt.subplots()
#ax.plot(dates_variants_index,np.dot(relative_transmissibility,A_data),'x')
#x0 = np.arange(3,5,0.1)
#ax.plot(dates_variants_index[0]+14*xs,function(xs,params[0],params[1]))

overall_transmissibility = function(xs,params[0],params[1])
overall_transmissibility = overall_transmissibility[source2.index_2020_10_13:]

A.name[4] = 'other'

cmap = cm.Paired
colors = [cmap(5),cmap(4),cmap(9),cmap(8),cmap(0)]
f,ax = plt.subplots(figsize=(5,3))
for i in range(5):
    ax.bar(range(len(A_data[i])),A_data[i],bottom = np.sum(A_data[:i,:],0),color = colors[i],label=A.name[i])#+' (%i%%)' % int(relative_transmissibility[i]*1e4) )
ax.legend(bbox_to_anchor=(0,0, 1.3, 1), loc="right",  ncol=1, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('% of viruses')
ax.set_xticks(range(6))
ax.set_xticklabels(['1/30/21','2/13/21','2/27/21','3/13/21','3/27/21','4/10/21'])
ax.set_xlabel('Collection date, two weeks ending')
plt.savefig('variants1.pdf',bbox_inches = "tight")

xs = np.arange(-dates_variants_index[0]/14,30,1/14)
f,ax = plt.subplots(figsize=(3,3))
x0 = np.arange(3,5,0.1)
ax.plot(dates_variants_index[0]+14*xs,100*function(xs,params[0],params[1]))
ax.plot(dates_variants_index,100*np.dot(relative_transmissibility,A_data),'o')
ax.set_xlim([source2.index_2020_10_13,source2.index_2020_10_13+365])
ax.set_ylabel('relative infectivity of\ncirculating virus strains')
ax.set_xticks([source2.index_2020_10_13,dates_variants_index[-1],source2.index_2020_10_13+365])
ax.set_xticklabels(['12/14/20','4/3/21','12/14/21'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticklabels(['%i%%' % el for el in ax.get_yticks()])
ax.set_xlabel('Date')
plt.savefig('variants2.pdf',bbox_inches = "tight")









