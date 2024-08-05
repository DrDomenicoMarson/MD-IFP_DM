#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    fig, axes = plt.subplots(2, 1)
    assert isinstance(axes, np.ndarray)
    ax_keep, ax_disc = axes.flat

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
                r = 1
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


def unify_resi(list_resi, df, resi_list_sorted, list_l=None, threshold=3):
    """
    Parameters:
    
    list_resi - a complete list of IFP contacts to be considered
    resi_list_sorted - sorted residue numbers to be included in the IFP matrix
    list_l - list of ligands to be considered
    
    Returns:
    ar_complete
    ar_SD_complete
    """

    list_l = [] if list_l is None else list_l

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
        if threshold>0:
            df_ligand_diss = df_ligand[t>threshold*threshold]
        else:
            df_ligand_diss = df_ligand[t<threshold*threshold]
        ar.append(np.asarray(df_ligand_diss[list_resi].mean().values))
        ar_SD.append(np.asarray(df_ligand_diss[list_resi].std().values))
    ar = np.asarray(ar)
    ar_SD = np.asarray(ar_SD)
    x = list_resi

    ar_complete = np.zeros((len(list_l), len(resi_list_sorted)), dtype=float)
    ar_SD_complete = np.zeros((len(list_l), len(resi_list_sorted)), dtype=float)
    for k,ligand in enumerate(list_l):
        for i, xx in enumerate(x):
            try:
                ind = np.argwhere(xx[6:]==resi_list_sorted)
                ar_complete[k][ind] = ar[k][i]
                ar_SD_complete[k][ind] = ar_SD[k][i]
            except IndexError:
                pass  # this is in the case if we left out part of residues in resi_list_sorted
    return ar_complete, ar_SD_complete


def read_databases(lst_of_pkl_files, lst_of_ligand_names):
    """
    returns:
    df_tot - the concatenated datasets
    ligands - the list of ligands names
    
    """
    dfs = []
    columns = []
    for pkl_df, ligand_name in zip(lst_of_pkl_files, lst_of_ligand_names):
        df = pd.read_pickle(pkl_df)
        df['ligand'] = ligand_name
        dfs.append(df)

        if not columns:
            columns = np.array(df.columns.tolist())
        else:
            diff = np.setdiff1d(np.asarray(df.columns.tolist()), columns)
            columns = np.append(columns, diff)


    # add empty columns for those that are present in the joint list but absent in the database
    new_df = pd.DataFrame(columns=columns)
    for df in dfs:
        for ifp in columns:
            if ifp not in df.columns.tolist():
                df[ifp] = 0
        new_df = pd.concat([new_df, df], axis=0, sort=False)

    if "COM" in new_df.columns.tolist():
        COM_x = []
        COM_y = []
        COM_z = []
        for l in new_df.COM:
            COM_x.append(l[0])
            COM_y.append(l[1])
            COM_z.append(l[2])
        new_df["COM_x"] = COM_x
        new_df["COM_y"] = COM_y
        new_df["COM_z"] = COM_z

    return new_df


def GRID_PRINT(file_name, pdrv, gr_orgn, gr_dim, grid_stp):
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

    header = f"#  density  \n" \
        f"object 1 class gridpositions counts {gr_dim[0]:3i} {gr_dim[1]:3i} {gr_dim[2]:3i}\n" \
        f"origin {gr_orgn[0]:5.3f} {gr_orgn[1]:5.3f} {gr_orgn[2]:5.3f}\n" \
        f"delta {grid_stp:5.2f} 0 0 \n" \
        f"delta 0 {grid_stp:5.2f} 0 \n" \
        f"delta 0 0 {grid_stp:5.2f} \n" \
        f"object 2 class gridconnections counts {gr_dim[0]:5i} {gr_dim[1]:5i} {gr_dim[2]:5i}\n" \
        f"object 3 class array type double rank 0 items {gr_dim[0]*gr_dim[1]*gr_dim[2]:7i} data follows\n" \

    # check_range = int((3.0/grid_stp) + 1)
    output = []
    count = 0
    for i in pdrv.reshape(-1):
        output.append(f"{i:12.3e}" %(i))
        count += 1
        if count % 3 == 0:
            output.append("\n")

    with open(file_name, "w") as f:
        f.write(header)
        f.write("".join(output))


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
    grid_mm_x = [COM_x.min(), COM_x.max()]
    grid_mm_y = [COM_y.min(), COM_y.max()]
    grid_mm_z = [COM_z.min(), COM_z.max()]
    grid_step = 1
    grid_dim= [
        int((grid_mm_x[1]-grid_mm_x[0])/grid_step+1),
        int((grid_mm_y[1]-grid_mm_y[0])/grid_step+1),
        int((grid_mm_z[1]-grid_mm_z[0])/grid_step+1)]
    grid = np.zeros((grid_dim),dtype=float)
    for (x,y,z) in zip(COM_x,COM_y,COM_z):
        ix= int((x-COM_x.min())/grid_step)
        iy= int((y-COM_y.min())/grid_step)
        iz= int((z-COM_z.min())/grid_step)
        grid[ix, iy, iz] += 1
    grid_origin = [grid_mm_x[0],grid_mm_y[0],grid_mm_z[0]]
    GRID_PRINT(filename, grid, grid_origin, grid_dim, grid_step)


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
    assert isinstance(axes, np.ndarray)
    colors = ["blue", "green", "red", "k", "orange", "cyan"]

    for pos_mean, pos_std, color, label, ax in zip(pos_means, pos_stds, colors, list_properties, axes.flat):
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
        #df = df.append(df_traj[df_traj.time==f], ignore_index=True)
        df = pd.concat([df, df_traj[df_traj.time==f]], ignore_index=True)
        com_tot.append(df_traj[df_traj.time == f][["COM_x", "COM_y", "COM_z"]].values)
        diss.append(df_traj[df_traj.time == f]["length"].values)
    ar = df[columns_IFP].values
    return ar, r_t_f, df, np.asarray(com_tot), np.asarray(diss)


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
    for _ in range(max_shuffle):
        # generate a sub-set
        np.random.shuffle(t)
        t_b = t[:sub_set]
        # find residence time from a sub-stet
        t_b_sorted_50 = (np.sort(t_b)[int(len(t_b)/2-0.5)] + np.sort(t_b)[int(len(t_b)/2)]) / 2.0
        tau_bootstr.append(t_b_sorted_50)
    return tau_bootstr
