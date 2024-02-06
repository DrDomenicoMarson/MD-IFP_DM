import glob, os
import sys
import subprocess
import numpy as np

import pandas as pd
from pandas import ExcelFile 

from matplotlib import *
from matplotlib import cm
import matplotlib.ticker
import  pylab as plt
import seaborn
import seaborn as sns
import matplotlib.gridspec as GS

#import ipywidgets as widgets

from scipy import stats



from Scripts.IFP_generation import *


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cluster import KMeans



def plot_graph_New(df_ext,out_base_name = "",ligand = "",draw_round = False,water = False):
    """
    Graph-based  representation of ligand dissociation trajectories
    
    Parameters:
    df_ext - IFP database
    out_base_name - file name to save an image
    ligand - generate dissociation pathways for a selected ligand only (note, that clusters properties still include data for all ligands)
    draw_round - type of representation (plain/round)
    water - visualize number of water molecules in the ligand solvation shell for each clutser
    
    Returns:
    cluster label sorted by increase of the average ligand RMSD in each cluster
    """
    from matplotlib.patches import ArrowStyle
    from matplotlib.patches import Ellipse
    from scipy.spatial import distance
    df_ext_ligand = df_ext
    if len(ligand)> 0:
        try:
            df_ext_ligand = df_ext[df_ext.ligand.isin(ligand)]
            print("Edges will be shown for one ligand:",ligand)
        except:
            print("ligand "+ligand+" was not found in the database. Whole database will be analyzed")
   
    label_rmsd = []    # rmsd
    label_com = []    # COM
    label_size = []
    label_water = []


    labels_list,nodes = np.unique(df_ext.label.values,return_counts= True)
    edges = np.zeros((labels_list.shape[0],labels_list.shape[0]),dtype = float)
    coms = np.zeros((labels_list.shape[0],labels_list.shape[0]),dtype = float)

    # loop ove all cluaters and compute their  population (label_size)
    for i,l in enumerate(labels_list):
        t = df_ext[df_ext.label == l]
  #      t_lig = df_ext[df_ext.label == l]
        label_rmsd.append(t.RMSDl.mean())
        label_com.append(np.array((t.COM_x.mean(),t.COM_y.mean(),t.COM_z.mean())))
        print("STD: ",l,t.COM_x.std(),t.COM_y.std(),t.COM_z.std(),t.RMSDl.std())
        label_size.append(100*t.shape[0]/df_ext.shape[0])
        if water:
            label_water.append(int(t.WAT.mean()))
        # compute distances between clusters COM
        for j in range(0,i):
            coms[i,j] = distance.euclidean(label_com[i],label_com[j])
            
    # loop to compute edges dencity
    for l,(df_label,df_time) in enumerate(zip(df_ext_ligand.label.values,df_ext_ligand.time.values)):
        if df_time != 0: 
            if(df_ext_ligand.label.values[l-1] != df_label):
                edges[df_ext_ligand.label.values[l-1],df_label] += labels_list.shape[0]/df_ext_ligand.label.values.shape[0] 
         
 #   print(np.max(edges), edges[edges > 0.5*np.max(edges)])
    indx_first_com = np.argwhere((np.asarray(label_rmsd) == min(label_rmsd)))[0][0]
    dist_com = []
    for i,l in enumerate(labels_list): dist_com.append(np.round(distance.euclidean(label_com[i],label_com[indx_first_com]),2))
    dist_com = (10*np.asarray(dist_com)/np.max(dist_com)).astype(int)
    
    print(indx_first_com, min(label_rmsd),dist_com)
    fig = plt.figure(figsize = (6,2),facecolor='w',dpi=150) 
    gs = GS.GridSpec(1,2, width_ratios=[ 1,1],wspace=0.08) 
    ax = plt.subplot(gs[0]) 
    plt.title("Transition density")
    plt.imshow(edges,cmap='Blues')
    ax = plt.subplot(gs[1])
    plt.title("Flow")
    flow = edges-edges.T
    plt.imshow(flow,cmap='Reds')
    plt.plot()


    starting_labels = df_ext[df_ext.time == 0].label.values # list of clusters of all first frames in all trajectories
    starting_list, starting_count = np.unique(starting_labels, return_counts=True) # list of clusters that appear as first frame

    #------------ older cluster position-------------
    print("RMSD: ",np.sort(label_rmsd))
    label2scale = label_rmsd # what value will be on x
    label2order = labels_list #nodes #np.roll(labels_list,1) #nodes # what value will be on y

    index_x_t = np.argsort(label2scale)
    # x and y positions of each cluster in the plot
    label_x = np.zeros((len(labels_list)),dtype = float)  
    label_y = np.zeros((len(labels_list)),dtype = float)  
    color_com = np.zeros((len(labels_list)),dtype = float)  

    
    # order label_x: 
    max_x = max(label2scale)
    step_x = 1.0*max_x/max(label2scale)
    # firt clasters that contain first frames of trajectories
    for i,s in enumerate(starting_list):
        label_x[s] = label2scale[s]*step_x
        label_y[s] = label2order[s]
        color_com[s] = dist_com[s]
    # then the rest 
    j = 0
    for l in index_x_t:
        if (labels_list[l] not in starting_list):
            while (label_x[j] != 0):
                j += 1
                if(j == len(labels_list)): break
            if(j == len(labels_list)):break
            label_x[labels_list[l]] = label2scale[l]*step_x 
            label_y[labels_list[l]] = label2order[l]
            color_com[labels_list[l]] = dist_com[l]
             
    # set logarythmic scale
    label_x = np.log10(label_x)
    x_tick_lable = []
    x_tick_pos = []
    for k in range(0,2):
        for ii,i in enumerate(range(pow(10,k),pow(10,k+1),pow(10,k))):  
            x_tick_lable.append(str(i))
            x_tick_pos.append(np.log10(i))
            if(i > 25): break
  
    if draw_round:
        alpha = 0.9*2*3.14*label_x/np.max(label_x)
        alpha_regular = 0.9*2*3.14*np.asarray(x_tick_pos)/max(x_tick_pos)
        label_y = np.sin(alpha)
        label_x = np.cos(alpha)
        fig = plt.figure(figsize=(8, 8))
        gs = GS.GridSpec(1, 1) #, width_ratios=[1, 1]) 
        ax = plt.subplot(gs[0])
        plt.scatter(x=np.cos(alpha_regular),y=np.sin(alpha_regular), c='k',s=10)
        for l,p in zip(x_tick_lable,x_tick_pos):
            ax.annotate(str(l)+"A", (1.2*np.cos(0.9*2*3.14*p/max(x_tick_pos)),np.sin(0.9*2*3.14*p/max(x_tick_pos))),fontsize=14,color="gray")
        plt.xlim=(-1.3,1.3)
        plt.ylim=(-1.3,1.3)
    else:
        fig = plt.figure(figsize=(10, 6))
        gs = GS.GridSpec(1, 1) #, width_ratios=[1, 1]) 
        ax = plt.subplot(gs[0])
        ax.set_ylabel('Cluster', fontsize=18)
        ax.set_xlabel('<RMSD> /Angstrom', fontsize=18) 
        plt.xticks(x_tick_pos,x_tick_lable, fontsize=18)
        ax.tick_params(labelsize=18)
        ax.set_ylim(-1,len(label_y)+1)
 #       plt.grid()

    for l in range(0,label_x.shape[0]):
        for n in range(l+1,label_x.shape[0]):
            # total number of transitions in both directions
            if (label_rmsd[l] > label_rmsd[n]):
                a = n
                b = l
            else:
                a = l
                b = n
            xy=(label_x[b],label_y[b])
            xytext=(label_x[a],label_y[a])   
            if (edges[l,n] > 0) :
                if  (np.abs((label_rmsd[l] - label_rmsd[n])) > 0.5* min(label_rmsd)) or (draw_round == False):
                    ax.annotate("", xy=xy, xycoords='data',
                        xytext=xytext, textcoords='data',
                        size=edges[l,n]*500,
                        arrowprops=dict(arrowstyle="Fancy,head_length=0.2, head_width=0.4, tail_width=0.2", 
                                fc="orange", ec="none", alpha=0.2 ,
                                connectionstyle="arc3,rad=-0.5"),
                        )
                if  (np.abs((label_rmsd[l] - label_rmsd[n])) > 0.5* min(label_rmsd)) or (draw_round == False):
                    ax.annotate("", xy=xytext, xycoords='data',
                        xytext=xy, textcoords='data',
                        size=edges[l,n]*500,
                        arrowprops=dict(arrowstyle="Fancy,head_length=0.2, head_width=0.4, tail_width=0.2", 
                                fc="orange", ec="none", alpha=0.2 ,
                                connectionstyle="arc3,rad=-0.5"),
                        )
            #  the flow
            flow = edges[l,n] - edges[n,l]  # flow l ----> n
            if (flow > 0) :
                a = l
                b = n
            else:
                a = n
                b = l                
            xy=(label_x[b],label_y[b])
            xytext=(label_x[a],label_y[a])   
            ax.annotate("", xy=xy, xycoords='data',
                        xytext=xytext, textcoords='data',
                        size=np.abs(flow)*5000,
                        arrowprops=dict(arrowstyle="Simple,head_length=0.2, head_width=0.4, tail_width=0.2", 
                                fc="0.6", ec="none", alpha=0.8 ,
                                connectionstyle="arc3,rad=-0.5"),
                        )

    for i,txt in enumerate(labels_list):
            ax.annotate(txt, (label_x[txt],label_y[txt]+0.05*pow(i,0.5)),fontsize=18)
    if water:         
        ax.scatter(label_x,label_y,facecolors='none',c=color_com,edgecolors="lightskyblue",s=500*np.asarray(label_size),cmap='Oranges',\
               linewidths=np.asarray(label_water))
        print("WATERS:",np.asarray(label_water))
    else:
        ax.scatter(label_x,label_y,facecolors='none',c=color_com,edgecolors="k",s=500*np.asarray(label_size),cmap='Oranges')
    if out_base_name != "": plt.savefig(out_base_name+'.or.pdf',dpi=300)  
    else:    plt.show()

    return(np.argsort(label_rmsd))

