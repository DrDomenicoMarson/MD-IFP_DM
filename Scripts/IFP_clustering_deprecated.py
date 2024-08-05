#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from loguru import logger
from IFP_clustering import unify_resi

########################################################################
#  deprecated, will be removed in the next version
########################################################################
def get_from_prop(list_x, df, list_l=None, threshold=0.1):
    """
    This function extracts a su-set of the pkl file for the user-defined list of IFPs and generated its properies
    Parameters:
    list_x - list of IFPs to be analyzed
    df,list_l= []
    threshold = 0.1
    
    Returns:
    ar - array of mean values
    ar_SD - array of standard deviations
    x - list of IFPs with mean below a pre-defined threshold
    
    """
    list_l = [] if list_l is None else list_l
    if len(list_l) == 0:
        list_l = np.unique(df.ligand.tolist())
    ar = []
    ar_SD = []
    for ligand in np.unique(df.ligand.tolist()):
        df_ligand = df[df.ligand==ligand]
        if ligand in list_l:
            ar_repl = []
            for Repl in np.unique(df_ligand.Repl.tolist()):
                df_ligand_Repl = df_ligand[df_ligand.Repl==Repl]
                repl_mean = df_ligand_Repl[list_x].mean().values
                ar_repl.append(repl_mean)
            ar.append(np.mean(np.asarray(ar_repl),axis=0))
            ar_SD.append(np.std(np.asarray(ar_repl),axis=0))
    ar = np.asarray(ar)
    ar_SD = np.asarray(ar_SD)
    ind = np.where(np.mean(ar,axis=0)<threshold)
    ar = np.delete(ar,ind,1)
    ar_SD = np.delete(ar_SD,ind,1)
    x = np.delete(list_x,ind)
    return ar, ar_SD, x


########################################################################
#    deprecated, will be removed in the next version
########################################################################
def ar_complete_ligand(ligand, df_tot, resi_list_sorted, properties=None):
    """
    combines an numpy array of selected part of the complete IFP dataset
    selection can be done by ligand, residue, and IFP properties
    Parameters:
    ligand - ligand name
    df_tot - ifp database 
    resi_list_sorted - list of residues 
    properties - ifp properties
    Returns:
    mean value and STD for each property
    
    """
    properties =["RE", "AR", "HD", "HA", "HY", "WB"] if properties is None else properties
    df_ligand = df_tot[df_tot.ligand==ligand]
    ar_complete = np.zeros((len(properties), len(resi_list_sorted)), dtype=float)
    ar_SD_complete = np.zeros((len(properties), len(resi_list_sorted)), dtype=float)
    
    for k,pr in enumerate(properties):
        list_x = get_resn_list(df_ligand.columns.tolist(),pr)
        ar_repl = []
        for Repl in np.unique(df_ligand.Repl.tolist()):
            df_ligand_Repl = df_ligand[df_ligand.Repl == Repl]
            repl_mean = df_ligand_Repl[list_x].mean().values
            ar_repl.append(repl_mean)
        ar= np.mean(np.asarray(ar_repl),axis=0)
        ar_SD = (np.std(np.asarray(ar_repl),axis=0))
        for i,xx in enumerate(list_x):
            ind = np.argwhere(xx[6:] == resi_list_sorted)
            ar_complete[k][ind] = ar[i]
            ar_SD_complete[k][ind] = ar_SD[i]

    return(ar_complete,ar_SD_complete)


########################################################################
# deprecated, will be removed in the next version
########################################################################
def Print_IFP_averaged(df_tot,resi_list_sorted,ligandsi,resi_name_list_sorted,properties=["AR","HD","HA","HY","WB","IP","IN"],threshold = 0.01):
    """
    generate a list of residues, combine all properties for each residue, sort them by the residue number
    
    Parameters:
        
    Returns:
    IFP plot
    
    """
    index_no_zero_IFP = np.asarray([])
    threshold = 0.01


    for i, pr in enumerate(properties):
        list_x = get_resn_list(df_tot.columns.tolist(),pr)
        ar_complete,ar_SD_complete=unify_resi(list_x,df_tot,resi_list_sorted,ligandsi)
        ind = np.argwhere(ar_complete.mean(axis=0) > threshold).flatten()
        index_no_zero_IFP = np.concatenate((index_no_zero_IFP,ind))

    index_no_zero_IFP = np.sort(np.unique(index_no_zero_IFP.astype(int)))
    part_resi =np.asarray(resi_name_list_sorted)[index_no_zero_IFP]
    logger.info(f"{np.asarray(resi_name_list_sorted)[index_no_zero_IFP]}")
    logger.info(f"{len(index_no_zero_IFP), len(resi_list_sorted)}")
    ind_part_resi = []
    for pr in part_resi:
        t = np.argwhere(resi_name_list_sorted == pr)
        ind_part_resi.append(t[0][0])

    # Plot average IFP map 
    color_ifp = ["k","magenta","skyblue","orange","darkgreen","red","blue","red"]


    resi_list_eq = resi_list_sorted
    ligands_group = np.asarray(ligandsi)
    ligands_name = np.asarray(ligandsi)  #np.unique(df_tot.ligand.tolist())

    #ligands_group = exp[exp.type == 'D'].ligand.tolist()
    #ligands_name = exp[exp.type == 'D'].name.tolist()
    logger.info(f"{ligands_group}")
    logger.info(f"{ligands_name}")

    fig = plt.figure(figsize = (16, 2*len(ligands_group)),facecolor='w')
    fig.subplots_adjust(hspace=0.05, wspace=0.25)
    for i,pr in enumerate(properties):
        list_x = get_resn_list(df_tot.columns.tolist(),pr)
        ar,ar_SD,x = get_from_prop(list_x, df_tot,threshold=0.1)
        ar_complete,ar_SD_complete=unify_resi(list_x,df_tot,resi_list_eq,ligands_group,threshold=-6)
        ax = plt.subplot(6,1,1)
        ax.set_xticks(np.asarray(range(0,len(ind_part_resi))))
        ax.set_yticks(2*np.arange(0,len(ligands_group)))
        ax.set_xticklabels(resi_name_list_sorted[ind_part_resi],rotation=90,fontsize=12)
        ax.set_yticklabels(ligands_name,fontsize=12)
        for l,ar_l in enumerate(ar_complete[:,ind_part_resi]):
            for r in range(0,len(ind_part_resi)):
                ax.scatter(r-0.4+i*0.1,2*l-0.4+i*0.1,color=color_ifp[i],marker='s',alpha=0.8,s=120*ar_l[r])
        ax.scatter(-2,0,color=color_ifp[i],alpha=0.9,s=120,label = pr,marker='s') 
#    plt.title(pr,fontsize=16)
        ax.grid(which="both")
        plt.xlim((-0.6,len(ind_part_resi)+2))
    plt.legend(fontsize=10,loc='upper right', bbox_to_anchor=(1.05, 1.))
    plt.show()
    return(ind_part_resi)
