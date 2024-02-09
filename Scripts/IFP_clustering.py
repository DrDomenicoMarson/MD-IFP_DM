#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#import IFP_generation as gen
from loguru import logger

def remove_dissociated_parts(df, max_rmsd=15, max_dcom=4.5, max_drmsd=5, out_name=None) -> pd.DataFrame:
    """    
    this function checks if there is a jump in the ligand position in  two neighbour frames,
    which may appear due to incomplete wrapping system back to the box (i.e. because of the PBC).  
    The trajectory frames starting from the detected jump will be removed from the dataset
    
    arguments:
    df - dataframe to work on
    max_rmsd - frames with RMSD below this threshold wil be analyzed
    max_dcom - maximum distance between COM of the ligand from the previous snapshot that will be considered as an indication of jump
    max_drmsd - maximum distance between RMSD of the ligand from the previous snapshot that will be considered as an indication of jump
    
    returns:
    df_new - new dataset
    """
    # remove trajectory part after dissociation
    fig, (ax_keep, ax_disc) = plt.subplots(2, 1)
    df_new = pd.DataFrame()
    for l in np.unique(df.ligand.values):
        df_lig = df[df.ligand==l]
        for repl in np.unique(df_lig.Repl.values):
            df_repl = df_lig[df_lig.Repl==repl]
            for traj in np.unique(df_repl.Traj.values):
                df_traj = df_repl[df_repl.Traj==traj]
                rmsd = df_traj.RMSDl.values
                com = df_traj.COM.values
                skip = -1
                for r in range(1, rmsd.shape[0]):
                    dcom = np.linalg.norm(com[r]-com[r-1])
                    drmsd = rmsd[r]-rmsd[r-1]
                    if rmsd[r] < max_rmsd and (dcom > max_dcom or drmsd > max_drmsd):
                        skip = r
                        continue

                # not quite understood, but usually not visited...
                mm = -1
                # s = df_traj.sum(axis=1).values
                # if np.argwhere(s==0).flatten().shape[0] > 0:
                #     mm = np.argwhere(s==0).flatten()[0]

                if  mm > 0 or skip > 0:
                    if mm > 0 and skip > 0:
                        mmr = min(mm, r)
                    elif mm > 0:
                        mmr = mm
                    else:
                        mmr = skip

                    df_disc = df_traj[df_traj.time.astype(int) > mmr]
                    df_traj = df_traj[df_traj.time.astype(int) <= mmr]
                    ax_disc.plot(df_disc.time, df_disc.RMSDl, linewidth=0.2)
                # df_new = df_new.append(df_traj)
                df_new = pd.concat([df_new, df_traj])

                ax_keep.plot(df_traj.time, df_traj.RMSDl, linewidth=0.2)

    for ax in [ax_keep, ax_disc]:
        ax.set_xlabel("frame")
        ax.set_ylabel("ligand RMSD")
    ax_keep.set_title("Kept trajs")
    ax_disc.set_title("Discarded trajs")
    fig.tight_layout()
    if out_name is not None:
        fig.savefig(out_name)
    else:
        fig.show()
    plt.close(fig)
    return df_new


########################################################################
# reading IFP databases
# Additionally column with ligand name is added
# and COM column is splitted to COM_x, COM_y, COM_z
########################################################################
def standard_IFP(unpickled_dfs, ligands):
    """
    reads IFP databases
        column with ligand name is added
        COM column is splitted to COM_x, COM_y, COM_z

    returns:
        combined IFP database
    """

    # add ligand names and make a joint list of columns
    columns = None
    for df, lig in zip(unpickled_dfs, ligands):
        df['ligand'] = pd.Series([lig for _ in range(df.shape[0])])

        #df["ligand"] = np.repeat(lig, df.shape[0])
        if columns is None:
            columns = np.array(df.columns.tolist())
        else:
            diff = np.setdiff1d(np.asarray(df.columns.tolist()), columns)
            columns = np.append(columns, diff)

    # add empty columns for those that are present in the joint list but absent in the database
    unpickled_df = pd.DataFrame(columns=columns)
    for df, lig in zip(unpickled_dfs, ligands):
        for ifp in columns:
            if ifp not in df.columns.tolist():
                df[ifp] = np.repeat(np.int8(0), df.shape[0])
        unpickled_df = pd.concat([unpickled_df, df], axis=0, sort=False)

    if "COM" in unpickled_df.columns.tolist():
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

    return unpickled_df

########################################################################
# separate IFP by type
########################################################################

def separate_IFP(columns):
    logger.info("separating IFP columns based on type, and sorting")
    res_list = []
    ifp_by_type = []
    res_name_list = []
    for col in columns:
        if col[2] == "_":
            res = int(col[6:])
            if res not in res_list:
                res_list.append(res)
                res_name_list.append(col[3:])
                ifp_by_type.append([0, 0, 0, 0, 0])
            ind = np.argwhere(np.asarray(res_list)==res)[0][0]
            if col[0:2] == "AR":
                ifp_by_type[ind][0] = 1
            if col[0:2] == "HY":
                ifp_by_type[ind][1] = 1
            if col[0:2] in ["HD", "HA"]:
                ifp_by_type[ind][2] = 1
            if col[0:2] == "WB":
                ifp_by_type[ind][3] = 1
            if col[0:2] == "RE":
                ifp_by_type[ind][4] = 1
        else:
            logger.debug(f"Column '{col}' skipped, no-IFP property")
    ind_sorted = np.argsort(res_list)
    res_idx_sorted = np.array(res_list)[ind_sorted]  # NOTE: was .astype(str) in legacy
    res_name_sorted = np.array(res_name_list)[ind_sorted]
    ifp_by_type_sorted = np.array(ifp_by_type)[ind_sorted]  # NOTE: was not sorted in legacy
    return res_idx_sorted, res_name_sorted, ifp_by_type_sorted

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
def read_databases(lst_of_pkl_files, lst_of_ligand_names):
    """
    returns:
    df_tot - the concatenated datasets
    ligands - the list of ligands names
    
    """
    dfs, ligands = [], []
    for pkl_df, name in zip(lst_of_pkl_files, lst_of_ligand_names):
        dfs.append(pd.read_pickle(pkl_df))
        ligands.append(name)

    return standard_IFP(dfs, ligands)


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

def plot_cluster_info(df, out_name=""):
    """
    plotting average COM (x, y, and z separately)
    and the number of water molecules in the ligand solvation shell
    in each clusters    
    """
    labels_list = np.unique(df["label"].values)
    list_properties = ["COM_x", "COM_y", "COM_z", "RGyr", "RMSDl"]
    if "WAT" in df.columns.values:
        list_properties += ["WAT"]

    pos_means = []
    pos_stds = []
    for j in range(0, len(list_properties)):
        pos_means.append([])
        pos_stds.append([])

    for l in labels_list:
        dd = df[df["label"]==l]
        for j, c in enumerate(list_properties):
            pos_means[j].append(dd[c].mean())
            pos_stds[j].append(dd[c].std())

    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    colors = ["blue", "green", "red", "k", "orange", "cyan"]

    for pos_mean, pos_std, color, label, ax in zip(
        pos_means, pos_stds, colors, list_properties, axes.flatten()):
        ax.scatter(x=labels_list, y=pos_mean, color=color, label=label)
        ax.errorbar(x=labels_list,y=pos_mean, yerr=pos_std, color="gray", fmt='--', markersize=1, linewidth=0.7)
        ax.set_title(label)
        ax.set_xlabel('cluster #')
        #ax.set_ylabel(label)
        ax.grid(color='gray', linestyle='-', linewidth=0.1, alpha=0.1)

    fig.tight_layout()
    if out_name:
        fig.savefig(out_name)
    else:
        fig.show()
    plt.close(fig)




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
