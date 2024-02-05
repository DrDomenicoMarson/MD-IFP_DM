#!/usr/bin/env python
# coding: utf-8
#
#  Package for analysis of RAMD dissociation trajectories using Interaction Fingerprints 
#
#############################
### v 1.0
#
#    Copyright (c) 2020
#    Released under the EUPL Licence, v1.2 or any higher version
#    
### Author: Daria Kokh
#    Daria.Kokh@h-its.org
#    Heidelberg Institute of Theoretical Studies (HITS, www.h-its.org)
#    Schloss-Wolfsbrunnenweg 35
#    69118 Heidelberg, Germany
################################# 




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





########################################################################
# Sorting of residue by number
########################################################################
def rank_IFP_resi(df,ifp_type=['AR','HY','HA','HD','HL','IP','IN',"IO"]):
    """    
    script extracts and ranks by the residue number IFP list  from the IFP table 
    
    Parameters:
    df - pkl table
    ifp_type - list of IFP types to be considered
    
    Results:
    columns_IFP - list of IFP
    columns_R
    """
    columns_IFP = []  # standard IFP
    number = []
    for c in df.columns.tolist():
        if c[0:2] in ifp_type: 
            columns_IFP.append(c)
            if c[3:].isdigit():    number.append(int(c[3:]))
            elif c[4:].isdigit():    number.append(int(c[4:]))
            elif c[5:].isdigit():    number.append(int(c[5:]))
            else: number.append(int(c[6:]))
    columns_IFP = np.asarray(columns_IFP)[np.argsort(np.asarray(number))]

    columns_RE = []  # standard IFP
    number = []
    for c in df.columns.tolist():
        if c[0:2] == "RE": 
            columns_RE.append(c)
            if c[3:].isdigit():    number.append(int(c[3:]))
            elif c[4:].isdigit():    number.append(int(c[4:]))
            elif c[5:].isdigit():    number.append(int(c[5:]))
            else: number.append(int(c[6:]))

    columns_RE = np.asarray(columns_RE)[np.argsort(np.asarray(number))]
    return(columns_IFP,columns_RE)

########################################################################
# 
########################################################################

def remove_dissociated_parts(df_tot,max_rmsd=15,max_dcom=3,max_drmsd=3):
    """    
    this function checks if there is a jump in the ligand position in  two neighbour frames, which may appear due to incomplete wrapping system back to the box (i.e. because of the PBC),  
    a part of the trajectory starting from the detected jump will be removed from the dataset
    
    Parameters:
    df_tot - pkl table
    columns_IFP - list of IFP types to be considered
    max_rmsd - frames with RMSD below this threshold wil be analyzed
    max_dcom - maximum distance between center of mass (COM) of the ligand from the first snapshot that will be considered as an indication of jump
    max_drmsd - maximum distance between center of mass (RMSD) of the ligand from the first snapshot that will be considered as an indication of jump
    Results:
    df_new - new dataset
    """
    # remove trajectory part after dissociation
    df_new= pd.DataFrame()
    for l in np.unique(df_tot.ligand.values):
        df_tot_lig = df_tot[df_tot.ligand == l]
        for j in np.unique(df_tot_lig.Repl.values):
            df_tot_Repl = df_tot_lig[df_tot_lig.Repl == j]
            for i in np.unique(df_tot_Repl.Traj.values.astype(int)):
                df_tot_Repl_Traj = df_tot_Repl[df_tot_Repl.Traj == str(i)]
                #s = np.sum(df_tot_Repl_Traj.values.astype(int),axis=1)
                s = df_tot_Repl_Traj.sum(axis =1).values
                rmsd = df_tot_Repl_Traj.RMSDl.values
                comx = df_tot_Repl_Traj.COM_x.values
                comy = df_tot_Repl_Traj.COM_y.values
                comz = df_tot_Repl_Traj.COM_z.values
                skip = -1
                for r in range(1,rmsd.shape[0]):
                    dcom = np.linalg.norm(np.asarray([float(comx[r-1]),float(comy[r-1]),float(comz[r-1])])-np.asarray([float(comx[r]),float(comy[r]),float(comz[r])]))
                    drmsd = (rmsd[r]-rmsd[r-1]) 
                    if  rmsd[r] < max_rmsd and (dcom > max_dcom or drmsd > max_drmsd):
                        plt.plot(df_tot_Repl_Traj.time.values,df_tot_Repl_Traj.RMSDl.values)
                        skip = r
                        continue
                if (np.argwhere(s == 0 ).flatten().shape[0] > 0) :
                    mm = np.argwhere(s == 0 ).flatten()[0]
                else: mm = -1
                if  mm > 0 or skip > 0:
                    if mm > 0 and skip > 0:   mmr = min(mm,r)
                    elif mm > 0: mmr =mm
                    else: mmr = skip
                    df_new = df_new.append(df_tot_Repl_Traj[df_tot_Repl_Traj.time.astype(int) < mmr])
                    plt.plot(df_tot_Repl_Traj.time,df_tot_Repl_Traj.RMSDl)
                else:
                    df_new = df_new.append(df_tot_Repl_Traj)
    plt.xlabel("frame",fontsize=14)
    plt.ylabel("ligand RMSD",fontsize=14)
    plt.title("Trajectories with a ligand jump")
    plt.savefig("jump_traj.pdf")
    return(df_new)


########################################################################
# reading IFP databases
# Additionally column with ligand name is added
# and COM column is splitted to COM_x, COM_y, COM_z
########################################################################
def standard_IFP(unpickled_dfi,ligandsi):
    """
    Script for reading IFP databases; Additionally column with ligand name is added and COM column is splitted to COM_x, COM_y, COM_z
    
    Parameters:
    dictionary of files with IFP databases {name1:file_path1[,name2:filepath2],...}
    
    Returns:
    combined IFP database
    """

    # add ligand names and make a joint list of columns
    intersect = []
    for (df,lig) in zip(unpickled_dfi,ligandsi):
        df["ligand"] = np.repeat(lig,df.shape[0]) 
        diff = np.setdiff1d(np.asarray(df.columns.tolist()),intersect)
        if(len(intersect) == 0):  intersect = diff
        else: intersect = np.append(intersect,diff)
    
    # add empty columns for those that are present in the joint list but absent in the database
    unpickled_df = pd.DataFrame(columns=intersect) 
    for (df,lig) in zip(unpickled_dfi,ligandsi):
        for ifp in intersect:
            if ifp not in df.columns.tolist():
                df[ifp] = np.repeat(np.int8(0),df.shape[0])    
        unpickled_df = pd.concat([unpickled_df, df], axis=0, sort=False)

    # converge COM string to  x y z components
    if "COM"  in unpickled_df.columns.tolist():
        COM_x = []
        COM_y = []
        COM_z = []
        for l in unpickled_df.COM:
            COM_x.append(l[0])
            COM_y.append(l[1])
            COM_z.append(l[2])
        unpickled_df["COM_x"] = COM_x
        unpickled_df["COM_y"] = COM_y
        unpickled_df["COM_z"] = COM_z
    
    return(unpickled_df)

########################################################################
# separate IFP by type 
#
########################################################################
def separate_IFP(complete_list_IFP):
    """
    
    Parameters:
    complete_list_IFP - list of IFPs
    
    Returns:
    resi_list_sorted -
    resi_name_list_sorted- 
    ifp_list - 
    """
    resi_list = []
    ifp_list = []
    resi_name_list = []
    for i,name in enumerate(complete_list_IFP):
        if(name[2]== "_"):
            resi = name[6:]
            if resi not in resi_list: 
                resi_list.append(resi)
                resi_name_list.append(name[3:])
                ifp_list.append([0,0,0,0,0])
            ind = np.argwhere(np.asarray(resi_list) == resi)[0][0]
            if(name[0:2] == "AR"):  ifp_list[ind][0]=1
            if(name[0:2] == "HY"):  ifp_list[ind][1]=1
            if(name[0:2] == "HD" or name[0:2] == "HA"):  ifp_list[ind][2] = 1
            if(name[0:2] == "WB"):  ifp_list[ind][3] = 1
            if(name[0:2] == "RE"):  ifp_list[ind][4] = 1
        else:
            print(name, "- skip no-IFP property")
    ind_sorted = np.argsort(np.asarray(resi_list).astype(int))
    resi_list_sorted = np.sort(np.asarray(resi_list).astype(int)).astype(str)
    resi_list_sorted = np.asarray(resi_list)[ind_sorted]  
    resi_name_list_sorted = np.asarray(resi_name_list)[ind_sorted]
    return(resi_list_sorted,resi_name_list_sorted,ifp_list)

########################################################################
#  deprecated, will be removed in the next version
########################################################################
def get_from_prop(list_x, df,list_l= [],threshold = 0.1):
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
    if len(list_l) == 0:
        list_l = np.unique(df.ligand.tolist())
    ar = []
    ar_SD = []
    for ligand in np.unique(df.ligand.tolist()):
        df_ligand = df[df.ligand == ligand]
        if ligand in list_l:
            ar_repl = []
            for Repl in np.unique(df_ligand.Repl.tolist()):
                df_ligand_Repl = df_ligand[df_ligand.Repl == Repl]
                repl_mean = df_ligand_Repl[list_x].mean().values
                ar_repl.append(repl_mean)
            ar.append(np.mean(np.asarray(ar_repl),axis=0))
            ar_SD.append(np.std(np.asarray(ar_repl),axis=0))
    ar= np.asarray(ar)
    ar_SD= np.asarray(ar_SD)
    ind = np.where(np.mean(ar,axis=0)<threshold)
    ar=np.delete(ar,ind,1)
    ar_SD=np.delete(ar_SD,ind,1)
    x = np.delete(list_x,ind)
    return(ar,ar_SD,x)

########################################################################
########################################################################
def unify_resi(list_resi, df,resi_list_sorted,list_l= [], threshold=3):
    """
    Parameters:
    
    list_resi - a complete list of IFP contacts to be considered
    resi_list_sorted - sorted residue numbers to be included in the IFP matrix
    list_l - list of ligands to be considered
    
    Returns:
    ar_complete
    ar_SD_complete
    """
    if len(list_l) == 0:
        list_l = np.unique(df.ligand.tolist())
    ar = []
    ar_SD = []
    for ligand in list_l:
        df_ligand = df[df.ligand == str(ligand)]
        comx = df_ligand[df_ligand.time == 0].COM_x.mean(axis=0)
        comy = df_ligand[df_ligand.time == 0].COM_y.mean(axis=0)
        comz = df_ligand[df_ligand.time == 0].COM_z.mean(axis=0)
#        print(">>>>>>>>>>>>>>>>>>>>>>",comx,comy,comz)
        t = (df_ligand.COM_x-comx)*(df_ligand.COM_x-comx)+\
        (df_ligand.COM_y-comy)*(df_ligand.COM_y-comy)+(df_ligand.COM_z-comz)*(df_ligand.COM_z-comz)  
        if(threshold > 0):
            df_ligand_diss = df_ligand[t > threshold*threshold]
        else:
            df_ligand_diss = df_ligand[t < threshold*threshold]
        ar_repl = []
        ar.append(np.asarray(df_ligand_diss[list_resi].mean().values))
        ar_SD.append(np.asarray(df_ligand_diss[list_resi].std().values))
    ar= np.asarray(ar)
    ar_SD= np.asarray(ar_SD)
    x = list_resi
        
    ar_complete = np.zeros((len(list_l),len(resi_list_sorted)),dtype = float)
    ar_SD_complete = np.zeros((len(list_l),len(resi_list_sorted)),dtype = float)
    for k,ligand in enumerate(list_l):
        for i,xx in enumerate(x):
            try:
                ind = np.argwhere(xx[6:] == resi_list_sorted)
                ar_complete[k][ind] = ar[k][i]
                ar_SD_complete[k][ind] = ar_SD[k][i]
            except:
                pass  # this is in the case if we  left out part of residues in resi_list_sorted
    return(ar_complete,ar_SD_complete)

########################################################################
#    deprecated, will be removed in the next version
########################################################################
def ar_complete_ligand(ligand,df_tot,resi_list_sorted,properties=["RE","AR","HD","HA","HY","WB"]):
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
    df_ligand = df_tot[df_tot.ligand == ligand]
    ar_complete = np.zeros((len(properties),len(resi_list_sorted)),dtype = float)
    ar_SD_complete = np.zeros((len(properties),len(resi_list_sorted)),dtype = float)
    
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
########################################################################
def read_databases(d,name_template,name_len = 8):
    """    
    Parameters:
    d - directory with multiple datasets 
    name_template - name of IFP pkl files (may contain *)
    name_len - number of the first letters in the name of  pkl files to be used as a ligand name
    Results:
    df_tot - concatenated dataset
    ligandsi - a set of ligand names
    
    """
    unpickled_dfi = []
    ligandsi = []
  #  list_IFP = {}    
    new_list_col = []
    for i,lig_pkl in enumerate(glob.glob(d+name_template)):
        name= lig_pkl[len(d):len(d)+name_len]
        print(lig_pkl,name)
        trj_length = []
        if os.path.exists(lig_pkl[:-3]+"dat"):
            print("Trajectory length will be taken from "+lig_pkl[:-3]+"dat")
            with open(lig_pkl[:-3]+"dat", 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.find("Complete TRJ") >= 0:  trj_length.append(int(line[line.find("xtc")+4:]))
    #    list_IFP.update( {name:lig_pkl})
        df_lig= pd.read_pickle(lig_pkl)
        if len(trj_length) == df_lig.shape[0]:
            df_lig.loc['length'] = trj_length
        list_col = df_lig.columns.tolist()
        new_list_col = [] # we need this array only once
        #--- to re-number residues
        """
        for l in list_col:
            if(l[2] == "_"):  
                new_list_col.append(l)
            else: new_list_col.append(l)
        df_lig.columns = new_list_col
        """
        unpickled_dfi.append(df_lig)
        ligandsi.append(name)
#    if len(new_list_col) <= 0:
#        print("There is no files in :",d+name_template)
    df_tot = standard_IFP(unpickled_dfi,ligandsi)

    return(df_tot,ligandsi,new_list_col)


########################################################################
# deprecated, will be removed in the next version
########################################################################
def clean_ramd(df_tot,threshold = 0.9,check_z = False):
    """
    Parameters:
    check_z - check if z coordinate is changed and drop trajectories where it did not
    
    Returns:
    """
    df_tot_new = pd.DataFrame(columns=df_tot.columns.tolist())
    if "ligand" in np.unique(df_tot.columns.values):
        for ligand in np.unique(df_tot.ligand):
            df_tot_ligand= df_tot[df_tot.ligand == ligand]
            for Repl in np.unique(df_tot_ligand.Repl):
                df_tot_ligand_Repl = df_tot_ligand[df_tot_ligand.Repl == Repl]
                list_out = 0
                for Traj in np.unique(df_tot_ligand_Repl.Traj):
                    df_tot_ligand_Repl_Traj = df_tot_ligand_Repl[df_tot_ligand_Repl.Traj == Traj]
                    if check_z:  
                        if((df_tot_ligand_Repl_Traj.COM_z.tolist()[0])*threshold > df_tot_ligand_Repl_Traj.COM_z.tolist()[-1]):
                            print("Dropped",ligand, Repl, Traj)
                            list_out += 1
                    else:
                        df_tot_new = pd.concat([df_tot_new,df_tot_ligand_Repl_Traj])
                if(list_out > 5): print(ligand,"be discarded:   ",Repl)
    else:
             for Repl in np.unique(df_tot.Repl):
                df_tot_Repl = df_tot[df_tot.Repl == Repl]
                list_out = 0
                for Traj in np.unique(df_tot_Repl.Traj):
                    df_tot_Repl_Traj = df_tot_Repl[df_tot_Repl.Traj == Traj]
                    if check_z:
                        if((df_tot_Repl_Traj.COM_z.tolist()[0])*threshold > df_tot_Repl_Traj.COM_z.tolist()[-1]):
                            list_out += 1
                            print("Dropped",ligand, Repl, Traj)
                        else:
                            df_tot_new = pd.concat([df_tot_new,df_tot_Repl_Traj])
                if(list_out > 5): print(ligand,"be discarded:   ",Repl)
       
    print(df_tot.shape, df_tot_new.shape)
    return(df_tot_new)



##########################################################################
######################################################################
def GRID_PRINT(file_name,pdrv,gr_orgn,gr_dim,grid_stp):
    """
    function that saved dx grid 
    Parameters:
    file_name - name of the grid file
    pdrv - grid
    gr_orgn - grid origin
    gr_dim - grid dimension
    grid_stp - grid step
    Returns:
    
    """
    header = "#  density  \n"
    header += "object 1 class gridpositions counts %3i" %gr_dim[0]+" %3i" %gr_dim[1]+" %3i" %gr_dim[2]+"\n"
    header += "origin %5.3f" %gr_orgn[0]+" %5.3f" %gr_orgn[1]+" %5.3f" %gr_orgn[2]+"\n"
    header += "delta %5.2f" %(grid_stp)+" 0 0 \n"
    header += "delta 0 %5.2f" %(grid_stp)+" 0 \n"
    header += "delta 0 0 %5.2f" %(grid_stp)+" \n"
    header += "object 2 class gridconnections counts %5i" %(gr_dim[0])+" %5i" %(gr_dim[1])+" %5i" %(gr_dim[2])+"\n"
    header += "object 3 class array type double rank 0 items %7i" %(gr_dim[0]*gr_dim[1]*gr_dim[2])+" data follows \n"

    check_range = int((3.0/grid_stp) + 1)

    output = []
    count = 0
    for i in pdrv.reshape(-1):
        output.append("%12.3e" %(i))
        count += 1
        if count%3 == 0:
            output.append("\n")
            
    with open(file_name,"w") as f:
        f.write(header)
        f.write("".join(output))
    return


##################################
################################
def Map_3D_grid(df_tot_to_save,filename):
    """
    Mapping ligand motion trajectory from the IFP file on the 3D grid and saving the grid in dx format
    
    Parameters:
    df_tot_to_save - dataset containing COM as columns COM_x, COM_y, and COM_z
    filename - the name of the output grid
    
    Returns:
    
    """
    COM_x = []
    COM_y = []
    COM_z = []
    for x in df_tot_to_save.COM.values:
        COM_x.append(x[0])
        COM_y.append(x[1])
        COM_z.append(x[2])
    COM_x = np.asarray(COM_x)
    COM_y = np.asarray(COM_y)
    COM_z = np.asarray(COM_z)
    grid_mm_x = [COM_x.min(),COM_x.max()]
    grid_mm_y = [COM_y.min(),COM_y.max()]
    grid_mm_z = [COM_z.min(),COM_z.max()]
    grid_step = 1
    grid_dim= [int((grid_mm_x[1]-grid_mm_x[0])/grid_step+1),int((grid_mm_y[1]-grid_mm_y[0])/grid_step+1),int((grid_mm_z[1]-grid_mm_z[0])/grid_step+1)]
    grid = np.zeros((grid_dim),dtype=float)
    for (x,y,z) in zip(COM_x,COM_y,COM_z):
        ix= int((x-COM_x.min())/grid_step)
        iy= int((y-COM_y.min())/grid_step)
        iz= int((z-COM_z.min())/grid_step)
        grid[ix,iy,iz] += 1
    grid_origin = [grid_mm_x[0],grid_mm_y[0],grid_mm_z[0]]    
    GRID_PRINT(filename,grid,grid_origin,grid_dim,grid_step)
    return

########################################
#  Plot COM RMSD in the clusters
#
#######################################
def Plot_COM(df_ext, out_name=""):
    """
    plotting average COM (x, y, and z separately) and the number of water molecules in the ligand solvation shell in each clusters
    Parameters:
    df_ext - IFP database with COM columns
    
    Returns:
    
    """
    labels_list = np.unique(df_ext["label"].values)
    if "WAT" in df_ext.columns.values:
        list_properties = ["COM_x","COM_y","COM_z","RGyr","WAT"]
    else:
        list_properties = ["COM_x","COM_y","COM_z","RGyr"]
    pos = []
    pos_SD = []
    for j in range(0,len(list_properties)):
        pos.append([])
        pos_SD.append([])

    for l in labels_list:
        dd = df_ext[df_ext["label"] == l]
        for j,c in enumerate(list_properties):
            pos[j].append(dd[c].mean())
            pos_SD[j].append(dd[c].std())

    fig = plt.figure(figsize=(16,5))
    gs = GS.GridSpec(1, len(list_properties),wspace=0.2) #,height_ratios=[1,1],width_ratios=[2,2,1,1])
    color = ["blue","green","red","k","orange","cyan"]
    label = list_properties

    for i,(pos,SD) in enumerate(zip(pos,pos_SD)):
        ax1 = plt.subplot(gs[i])
        plt.scatter(x = labels_list,y = pos, color =color[i],label = label[i])
        plt.errorbar(x = labels_list,y = pos,yerr= pos_SD[i], color = "gray" , fmt='o--', markersize=1)  
        ax1.set_title(label[i], fontsize=18)
        ax1.set_xlabel('cluster', fontsize=16)   
        if i == 0:     ax1.set_ylabel('COM [arb. u.]', fontsize=18)  
        else:  ax1.set_ylabel('', fontsize=18)  
        ax1.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.show()
    if out_name:
        plt.savefig(out_name)
    return




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


    for i,pr in enumerate(properties):
        list_x = get_resn_list(df_tot.columns.tolist(),pr)
        ar_complete,ar_SD_complete=unify_resi(list_x,df_tot,resi_list_sorted,ligandsi)
        ind = np.argwhere(ar_complete.mean(axis=0) > threshold).flatten()
        index_no_zero_IFP = np.concatenate((index_no_zero_IFP,ind))

    index_no_zero_IFP = np.sort(np.unique(index_no_zero_IFP.astype(int)))
    part_resi =np.asarray(resi_name_list_sorted)[index_no_zero_IFP]
    print(np.asarray(resi_name_list_sorted)[index_no_zero_IFP])
    print(len(index_no_zero_IFP),len(resi_list_sorted))
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
    print(ligands_group)
    print(ligands_name)

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


###################################
#
##################################
def last_frames_by_contact(df_tot,columns_IFP,contacts):
    """
    functin that build an numpy array of the IFP properties extracting from the TFP dataset only the several last frame with a pre-defined number of the protein-ligand contacts 
    Parameters:
    contacts - number of contacts 
    df_tot - complete dataset
    columns_IFP - columns to be analyzed
    Returns:
    ar - numpy array containg IFP of the selected frames
    r_t_f - list of selected replica-trajectory-frame  
    df - IFP database from selected frames
    np.asarray(com_tot) - just COM valsues
    np.asarray(diss) - just trajectory length (column "length" from the original data set )
    """
    r_t_f = []
    com_tot = []
    diss = []
    for r in df_tot.Repl.unique():  
        df_repl = df_tot[df_tot.Repl == r]
        for t in df_repl.Traj.unique():
            df_traj = df_repl[df_repl.Traj == t]
            sum_IFP = df_traj[columns_IFP].sum(1).values
            last_frame = np.max(np.argwhere(sum_IFP > contacts))
            r_t_f.append((r,t,last_frame))
    df = pd.DataFrame(columns = df_tot.columns)     
    for (r,t,f) in r_t_f:  
        df_repl = df_tot[df_tot.Repl == r]
        df_traj = df_repl[df_repl.Traj == t]
        df = df.append(df_traj[df_traj.time == f], ignore_index = True)
        com_tot.append(df_traj[df_traj.time == f][["COM_x","COM_y","COM_z"]].values)
        diss.append(df_traj[df_traj.time == f]["length"].values)
    ar = df[columns_IFP].values
    return(ar,r_t_f,df,np.asarray(com_tot),np.asarray(diss))


###################################
#
##################################



def bootstrapp(t, rounds=50000):
    """
    function for getting approximate residence time for a sub-set of trajectories (for example from a selected channel)
    Parameters:
    t - set of trajectory length to be used for bootsrapping
    Returns:
    relative residence time
    """
    max_shuffle = rounds
    alpha = 0.8
    sub_set = int(alpha*len(t))
    tau_bootstr = []
    for i in range(1,max_shuffle):
        # generate a sub-set
        np.random.shuffle(t)
        t_b = t[:sub_set]
        # find residence time from a sub-stet
        t_b_sorted_50 =(np.sort(t_b)[int(len(t_b)/2.0-0.5)]+np.sort(t_b)[int(len(t_b)/2)])/2.0
        tau_bootstr.append(t_b_sorted_50)
    return(tau_bootstr)
