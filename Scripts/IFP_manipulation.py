#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from loguru import logger


####################################################################################################
# combine three tables
#    - with protein-ligand hydrogen bonds
#    - with protein-ligand water bridges
#    - with other protein-ligand interaction properties (IFP)
# in one table
####################################################################################################

def table_combine(df_HB, df_WB, df_prop, ligand_name, residues_name=None, start=0, stop=None, step=1):
    """
    Parameters:
    df_HB - H-bond table
    df_prop - IFP table 
    ligand_name      - ligand nale
    residues_name a list of properties that (column names) that can be used to generate tables with the same column list
    
    Return:
    updated table
    """
    residues_name = [] if residues_name is None else residues_name
    if stop :
        if len(range(start,stop,step)) != np.asarray(df_prop.shape[0]):
            stop = (df_prop.shape[0] -start)*step
    else:
        stop = df_prop.shape[0]

    #---------------- extract hydrogen bonds between ligand and protein and add to IFP table----------
    columns_resname = []
    df_prop["time"] = range(start, stop, step)
    #------- list of the residues making donor-HB  with the  ligand, but not water
    df_noWatD = df_HB[~df_HB.donor_resnm.isin([ligand_name, "WAT"])]  # protein donor
    df_noWatD = df_noWatD[df_noWatD.acceptor_resnm == ligand_name]   # protein donor and ligand acceprot

    #------- list of the residues making acceptor-HB  with the  ligand , but not water
    df_noWatA = df_HB[~df_HB.acceptor_resnm.isin([ligand_name, "WAT"])]  # protein acceptor
    df_noWatA = df_noWatA[df_noWatA.donor_resnm == ligand_name]  # protein acceptor and ligand donor

    t_list = []
    for t in df_HB.time.unique().tolist():
        raw = int(t)
        if not df_noWatD.empty:
            df_noWatD_t = df_noWatD[(df_noWatD.time == t)]
            if not df_noWatD_t.empty:
                for d in df_noWatD_t.donor_resid.tolist():
                    r = "HD_" + df_noWatD_t[df_noWatD_t.donor_resid==d].donor_resnm.tolist()[0] + str(d)
                    if r not in columns_resname:
                        columns_resname.append(r)
                    t_list.append((raw, r))
        if not df_noWatA.empty:
            df_noWatA_t = df_noWatA[(df_noWatA.time == t)]
            if not df_noWatA_t.empty:
                for d in df_noWatA_t.acceptor_resid.tolist():
                    r = "HA_"+df_noWatA_t[df_noWatA_t.acceptor_resid == d].acceptor_resnm.tolist()[0]+str(d)
                    if r not in columns_resname:
                        columns_resname.append(r)
                    t_list.append((raw,r))
    properties =  np.zeros((len(df_prop.index.values.tolist()), len(columns_resname)), dtype=np.int8)
    for j, c in enumerate(np.sort(np.asarray(columns_resname))):
        for cc in t_list:
            if c==cc[1]:
                properties[cc[0], j] = 1
        df_prop[c] = properties[:, j]

    #---------------- extract water bridges between ligand and protein and add to IFP table----------
    if df_WB is not None:
        # we have to check naming since it was changed in later version
        #------ new version 16-12-2019
        t_list = []
        column_resi = []
        # get a list of INH-WAT (sele1 - sele2)
        df_WB_INH = df_WB[(df_WB.sele1_resnm.isin([ligand_name]) & df_WB.sele2_resnm.isin(["WAT","HOH","SOL","TIP3"]))]
        # get a list of WAT-Prot (sele1 - sele2)
        df_WB_Prot = df_WB[(~(df_WB.sele2_resnm.isin([ligand_name, "WAT","HOH","SOL","TIP3"])) & (df_WB.sele1_resnm.isin(["WAT","HOH","SOL","TIP3"])))]
        for t in df_WB.time.unique().tolist():
            raw = int(t)
            df_WB_Prot_t = df_WB_Prot[df_WB_Prot.time == t]
            df_WB_INH_t = df_WB_INH[df_WB_INH.time == t]
            if ((not df_WB_Prot_t.empty) and (not df_WB_INH_t.empty)):
                for r in np.unique(df_WB_Prot_t.sele2_resid.values):
                    r1 = "WB_"+df_WB_Prot_t[df_WB_Prot_t.sele2_resid == r].sele2_resnm.values[0]+str(r)
                    t_list.append((raw,r1))
                    if r1 not in column_resi: 
                        column_resi.append(r1)
        properties =  np.zeros((len(df_prop.index.values.tolist()),len(column_resi)),dtype=np.int8)
        for j,c in enumerate(column_resi):
            for cc in t_list:
                if c == cc[1]:
                    properties[cc[0], j] = 1
            df_prop[c] = properties[:, j]

    # add more columns for IFP provided as input but not found in the current
    for rr in residues_name:
        if rr not in df_prop.columns.tolist():
            df_prop[rr] = 0

    # cleaning the table; fist order residues by number
    df_prop_order_new = []
    for df_prop_order in df_prop.columns.tolist():
        if df_prop_order.find("_") > 0:
            df_prop_order_new.append(int(df_prop_order[df_prop_order.find("_")+4:]))
        else: df_prop_order_new.append(0)
    properties = np.asarray(df_prop.columns.tolist())[np.argsort(df_prop_order_new)]
    # then use order of the properties HY - HD - HA - IP - IN
    for i_df in range(1,len(properties)):
        if (properties[i_df].find("_")> 0) and (properties[i_df-1].find("_")>0):
            if properties[i_df][properties[i_df].find("_"):] == properties[i_df-1][properties[i_df-1].find("_"):]:
                properties[i_df-1:i_df+1] = np.sort(properties[i_df-1:i_df+1])
    df_prop = df_prop[properties]
    #--- change column position puttinhg time at the beginning and Water at the end---
    df_prop = df_prop[np.concatenate((["time"], df_prop.columns[df_prop.columns != "time"].tolist()))]
    if "WAT" in df_prop.columns.tolist():
        df_prop = df_prop[np.concatenate((df_prop.columns[df_prop.columns != "WAT"].tolist(), ["WAT"]))]

    return df_prop

############################################################################################################
# reads IFP database; additionally,     column with ligand name is added
#                                       COM column is splitted to COM_x, COM_y, COM_z
############################################################################################################
def read_IFP(list_IFP):
    """
    Parameters:
    dictionary of files with ITP databases {name1:file_path1[,name2:filepath2],...}
    
    Returns:
    combined IFP database
    """
    unpickled_dfi = []
    ligandsi = []
    for lig in list_IFP:
        unpickled_dfi.append(pd.read_pickle(list_IFP[lig]))
        ligandsi.append(lig)

    # add ligand names and make a joint list of columns
    intersect = []
    for df, lig in zip(unpickled_dfi, ligandsi):
        df["ligand"] = np.repeat(lig,df.shape[0])
        diff = np.setdiff1d(np.asarray(df.columns.tolist()),intersect)
        if len(intersect) == 0:
            intersect = diff
        else:
            intersect = np.append(intersect,diff)

    # add empty columns for those that are present in the joint list but absent in the database
    unpickled_df = pd.DataFrame(columns=intersect)
    for df, lig in zip(unpickled_dfi, ligandsi):
        for ifp in intersect:
            if ifp not in df.columns.tolist():
                df[ifp] = np.repeat(np.int8(0), df.shape[0])
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

    return unpickled_df


def Plot_IFP(df, ifp_list, out_name=""):
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)
    color = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
    columns_IFP = []  # standard IFP
    columns_RE = []  # just contacts
    for c in df.columns.tolist():
        if c[0:2]=="RE":
            columns_RE.append(c)
        elif c[0:2] in ifp_list:
            columns_IFP.append(c)

    if df[columns_IFP].values.shape[0] < 2:
        return



    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [4, 2, 1]})
    assert isinstance(axes, np.ndarray)
    ax_ifp, ax_re, ax_wat = axes.flat
    ax_ifp.set_title('IFP')
    ax_re.set_title('RE')
    ax_wat.set_title('Water shell')

    sns.heatmap(np.float32(df[columns_IFP].values),
                cmap="YlGnBu",
                xticklabels=columns_IFP if len(columns_IFP) < 25 else 'auto',
                ax=ax_ifp)

    if df[columns_RE].shape[1] > 0:
        sns.heatmap(np.float32(df[columns_RE].values), cmap="YlGnBu", ax=ax_re)

    if "WAT" in df.columns.tolist():
        ax_wat.set_ylim(0, max(df["WAT"].tolist()))
        if "Repl" in df.columns.tolist():
            for i, r in enumerate(np.unique(df.Repl.tolist())):
                ax_wat.plot(df[df.Repl==r]["WAT"],
                            marker='o', markersize=1, linewidth=0, color=color[i], label=r)
            if np.unique(df.Repl.tolist()).shape[0] < 10:
                ax_wat.legend()
        else:
            ax_wat.plot(df["WAT"], marker='o', linewidth=0, markersize=1)
        ax_wat.set_xlabel('frame #')
        ax_wat.set_ylabel('# of water molecules')

    if out_name == "":
        fig.show()
    else:
        fig.savefig(out_name)
    plt.close(fig)
    return


def extract_and_rank_by_resnum(df: pd.DataFrame, ifp_type) -> tuple[np.ndarray, np.ndarray]:
    """    
    old name was rank_IFP_resi
    extracts and ranks by the residue number IFP list from the IFP table 
    
    arguments:
      df - IFP df
      ifp_type - list of IFP types to be considered
    
    return:
      columns_IFP - list of IFP based on ift_type
      columns_RE  - list of unspecific IFP
    """

    def add_to_numbers(number):
        if c[3:].isdigit():
            number.append(int(c[3:]))
        elif c[4:].isdigit():
            number.append(int(c[4:]))
        elif c[5:].isdigit():
            number.append(int(c[5:]))
        else:
            number.append(int(c[6:]))

    columns_IFP, columns_RE = [], []
    number_IFP, number_RE = [], []
    for c in df.columns.tolist():
        if c[0:2] == "RE":
            columns_RE.append(c)
            add_to_numbers(number_RE)
        elif c[0:2] in ifp_type:
            columns_IFP.append(c)
            add_to_numbers(number_IFP)

    columns_IFP = np.asarray(columns_IFP)[np.argsort(np.asarray(number_IFP))]
    columns_RE = np.asarray(columns_RE)[np.argsort(np.asarray(number_RE))]
    return columns_IFP, columns_RE


def plot_IF_trajectory(df_tot, ifp_type, head_tail=-1, out_base_name=""):

    if head_tail < 0:
        head_tail = int(df_tot.shape[0]/3)

    columns_IFP, columns_RE = extract_and_rank_by_resnum(df_tot, ifp_type)

    for columns, name in zip([columns_IFP, columns_RE], ["IFP", "RE"]):
        if len(columns) == 0:
            logger.info(f"contacts of type '{name}' not found")
            continue
        df = df_tot[columns]
        columns = columns[df.mean().values > 0.01]
        df = df_tot[np.append(columns, "time")]
        #columns_HB = np.array([c for c in columns if (c[:2] == "HD" or c[:2] == "HA")])
        #n_hb = len(columns_HB[(df[columns_HB].mean()> 0.75).values])

        fig, ax = plt.subplots(1, 1)
        ax.bar(
            range(0, len(columns)), df[df.time < head_tail][columns].mean(),
            alpha=0.6, label=f"mean over first {head_tail} frames")
        ax.bar(
            range(0, len(columns)), df[df.time > df.shape[0]-head_tail][columns].mean(),
            alpha=0.6, label=f"mean over last {head_tail} frames")
        ax.bar(
            range(0, len(columns)), df[columns].mean(),
            color="None", label="mean over all frames", edgecolor ='k', hatch="/")

        ax.set_xticks(range(0, len(columns)))
        ax.set_xticklabels(columns)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=4)
        ax.legend(loc='upper left', fontsize=8)

        if out_base_name != "":
            fig.savefig(f"{out_base_name}.{name}.pdf")
        else:
            fig.show()
        plt.close(fig)
