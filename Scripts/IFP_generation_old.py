#!/usr/bin/env python

from dataclasses import dataclass
import datetime
import numpy as np
import pandas as pd
from loguru import logger

import MDAnalysis.analysis.hbonds as hba
from MDAnalysis.lib.distances import calc_angles


r_cat = 5 # cation-aromatic
r_ari = 5.5  # pi-pi
r_hyd = 4.0 # hydrophobic
r_sar = 4.5 # S-aromatic
r_sal = 4.5 # salt bridge
r_hal = 3.5 # halogen interactions
r_wat = 3.5 # water shell
r_dis = 5.0 # all protein-ligand contacts
r_lip = 5.0 # specific residues (in particular, lipids)
r_ion = 3.4  # salt bridges with ions

at_aromatic = "((resname PHE TRP TYR HIS HIE HID HE2) and (name CZ* CD* CE* CG* CH* NE* ND*))"
at_positive =  "((resname ARG LYS ) and (name NH* NZ)) or ((resname HI2 ) and (name HD HE))"
at_negative = " ((resname ASP GLU) and (name OE* OD*))"
at_sulfur = "(protein and (name S*))"
at_hydrophob = " (protein  and (name C*  S*) and (not  (name CG and resname ASN ASP))   and (not  (name CD and resname GLU GLN ARG))  and (not  (name CZ and resname TYR ARG))  and (not  (name CE and resname LYS)) and (not  (name CB and resname SER THR))   and (not backbone))"
at_pos_ions = "(resname  MN ZN Mn Zn Ca CA NA Na)"
at_water = "(resname WAT HOH SOL TIP3)"
at_halogens = "( type I CL BR Br Cl)"
at_noH = "( not name H* )"

angle_CHal_O = 150  # currently is not used

resi_aromatic = ["HIS", "HIE", "HID", "HI2", "TYR", "TRP", "PHE"]

@dataclass
class IFP_prop:
    """
    CLASS of the Interaction Fingerprint properties for particular type of the protein-ligand interection
    
    name - type of interaction
    atoms -ligand atom that belong to this type
    sel_a - string describing a selection of protein residues/atoms 
    sel_b - string describing a selection of ligand atoms 
    dist - distance to be used for analysis
    contacts - list of contact 
    """
    def __init__(self, name, atoms, sel_a, sel_b, dist):
        self.name = name
        self.atoms = atoms
        self.sel_a = sel_a
        self.sel_b = sel_b
        self.dist = dist
        self.contacts = []

def IFP_list(property_list, sel_ligands, RE=True, Lipids=None):
    """
    Parameters:
    property_list - ligand atom properties as generated by Rdkit
    sel_ligands - ligand residue name
    
    Returns:
    a list of properties that can be used to extract IFP from a snapshot
    """

    IFP_prop_list = []

    def add_to_IFP_prop_list(prop_key, name, r, sel_a):
        line = ""
        for l in tuple(set(property_list[prop_key])):
            line += l + " "
        IFP_prop_list.append(IFP_prop(
            name=name, atoms=line, dist=r, sel_a=sel_a,
            sel_b="((resname " + sel_ligands + ") and (name " + line + ") )"))

    if "Hydrophobe" in property_list.keys():
        add_to_IFP_prop_list(prop_key="Hydrophobe", name="HY", r=r_hyd, sel_a=at_hydrophob)

    if "PosIonizable" in property_list.keys():
        add_to_IFP_prop_list(prop_key="PosIonizable", name="IP", r=r_sal, sel_a=at_negative)
        add_to_IFP_prop_list(prop_key="PosIonizable", name="AR", r=r_cat, sel_a=at_aromatic)

    if "NegIonizable" in property_list.keys():
        add_to_IFP_prop_list(prop_key="NegIonizable", name="IN", r=r_sal, sel_a=at_positive)
        add_to_IFP_prop_list(prop_key="NegIonizable", name="IO", r=r_ion, sel_a=at_pos_ions)

    if "Aromatic" in property_list.keys():
        add_to_IFP_prop_list(prop_key="Aromatic", name="AR", r=r_ari, sel_a=at_aromatic)
        add_to_IFP_prop_list(prop_key="Aromatic", name="AR", r=r_cat, sel_a=at_positive)

    sel_a = at_aromatic + " or " + at_negative + " or " + " (backbone and name O) " + " or " + at_sulfur
    sel_b = "((resname " + sel_ligands + " ) and " + at_halogens + " )"
    IFP_prop_list.append(IFP_prop(name="HL", atoms="HL", sel_a=sel_a, sel_b=sel_b, dist=r_hal))

    sel_a = " (resname WAT HOH SOL TIP3 and name O*) "
    sel_b = "(resname " + sel_ligands + " ) and " + at_noH
    IFP_prop_list.append(IFP_prop(name="WA", atoms="HE", sel_a=sel_a, sel_b=sel_b, dist=r_wat))

    if RE:
        sel_a = "(not resname WAT HOH SOL TIP3) and (not name H*)"
        sel_b = "(resname " + sel_ligands + " ) and " + at_noH
        IFP_prop_list.append(IFP_prop(name="RE", atoms="HE", sel_a=sel_a, sel_b=sel_b, dist=r_dis))

    if Lipids is not None:
        line = ""
        for l in Lipids:
            line += l + " "
        sel_a = "((resname " + line + " ) and " + at_noH + ") "
        sel_b = "(resname " + sel_ligands + ") and " + at_noH
        IFP_prop_list.append(IFP_prop(name="LL", atoms=line, sel_a=sel_a, sel_b=sel_b, dist=r_lip))

    return IFP_prop_list


def make_IFT_table(IFP_prop_list, snaps, columns_extended=None):
    """
    Most of interections are taken from the list given in 
    https://www.cambridgemedchemconsulting.com/resources/molecular_interactions.html
    
    Parameters:
    IFP_prop_list list of IFP objects 
    
    Returns:
    column names and matrix with IFP    values
    """
    columns_extended = [] if columns_extended is None else columns_extended

    # first make a list of all types of contacts observed (columns)
    if len(columns_extended) == 0:
        columns = []
        for IFP_type  in IFP_prop_list:  # loop over frames
            for s in IFP_type.contacts:  # loop over different contacts
                for c in s[1]:           # loop over  particular contacts in a particular frame                
                    if IFP_type.name == "WA":
                        IFP_element = "WAT"
                    elif IFP_type.name == "LL":
                        IFP_element = "LIP"
                    else:# combine contact type with residues type and name
                        IFP_element = c[0]
                    columns.append(IFP_element)
        columns = np.unique(np.asarray(columns).flatten())
    else:
        columns = columns_extended


    times = np.linspace(0,snaps,snaps)
    IFP_matrix = np.zeros((len(times),len(columns)),dtype = np.int8)

    for IFP_type  in IFP_prop_list: # loop over different contacts
        for s in IFP_type.contacts: # loop over frames
            for c in s[1]:          # loop over  particular contacts in a particular frame (frame - s[0])
                if IFP_type.name == "WA":
                    IFP_element = "WAT"
                if IFP_type.name == "LL":
                    IFP_element = "LIP"
                else:  # combine contact type with residues type and name
                    # IFP_element = IFP_type.name+"_"+c[0]+str(c[1]) 
                    IFP_element = c[0]
                #try:
                col = np.argwhere(columns == IFP_element).flatten()[0]
                #except:
                #    logger.error(f"{IFP_element} was not found in {columns}, {len(columns_extended)}")
                #try:
                if IFP_type.name == "WA":
                    IFP_matrix[s[0], col] += 1
                else:
                    IFP_matrix[s[0], col] = 1
                #except:
                #    logger.warning(f"IFP was not found: {IFP_element}, {col}, {s[0]}")
    return columns, IFP_matrix

def IFP(u_mem, sel_ligands, property_list, WB_analysis=True, RE=True, Lipids=None, WB_debug=False):
    """
    Parameters:
    u - trajectory - universe object
    ligand name -  ligand residue name
    property_list - python dictionary of ligand atom properties (created by ligand_analysis)
    
    Reterns:
    """

    Lipids = [] if Lipids is None else Lipids

    #---------------------------------------------------------------
    #- find hydrogen bonds between ptotein and ligand

    logger.debug("Computing HBONDS with old MDA, based on CHARMM27 naming")
    hba.HydrogenBondAnalysis.DEFAULT_DONORS['OtherFF'] = hba.HydrogenBondAnalysis.DEFAULT_DONORS['CHARMM27']
    hba.HydrogenBondAnalysis.DEFAULT_ACCEPTORS['OtherFF'] = hba.HydrogenBondAnalysis.DEFAULT_ACCEPTORS['CHARMM27']
    hba.WaterBridgeAnalysis.DEFAULT_DONORS['OtherFF'] = hba.WaterBridgeAnalysis.DEFAULT_DONORS['CHARMM27']
    hba.WaterBridgeAnalysis.DEFAULT_ACCEPTORS['OtherFF'] = hba.WaterBridgeAnalysis.DEFAULT_ACCEPTORS['CHARMM27']
    if "Donor" in set(property_list):
        donor_line = tuple(set(property_list["Donor"]))
        logger.debug(f"Adding {donor_line} to DEFAULT_DONORS")
        hba.HydrogenBondAnalysis.DEFAULT_DONORS['OtherFF'] += donor_line
        hba.WaterBridgeAnalysis.DEFAULT_DONORS['OtherFF'] += donor_line
    if "Acceptor" in set(property_list):
        acceptor_line = tuple(set(property_list["Acceptor"]))
        logger.debug(f"Adding {acceptor_line} to DEFAULT_ACCEPTORS")
        hba.HydrogenBondAnalysis.DEFAULT_ACCEPTORS['OtherFF'] += acceptor_line
        hba.WaterBridgeAnalysis.DEFAULT_ACCEPTORS['OtherFF'] += acceptor_line

    logger.debug("default donors: ", hba.HydrogenBondAnalysis.DEFAULT_DONORS['OtherFF'])
    logger.debug("default accepors: ", hba.HydrogenBondAnalysis.DEFAULT_ACCEPTORS['OtherFF'])
    h = hba.HydrogenBondAnalysis(u_mem,
                                 selection1 = f'resname {sel_ligands}',
                                 selection2 = f'not resname WAT HOH SOL {sel_ligands}',
                                 distance=3.3, angle=100, forcefield='OtherFF')

    logger.info(f"Start HB analysis at {datetime.datetime.now().time()}")
    h.run()
    logger.info(f"         ... done at {datetime.datetime.now().time()}")

    logger.debug("Generating table from HB results")
    h.generate_table()
    df_HB = pd.DataFrame.from_records(h.table)
    logger.debug(df_HB)

    if WB_analysis :
        logger.info(f"Start WB analysis at {datetime.datetime.now().time()}")
        df_WB = Water_bridges(u_mem, sel_ligands,WB_debug)
        logger.debug(df_WB)
    else:
        df_WB = None

    logger.info(f"Start collecting IFPs: {datetime.datetime.now().time()}")
    IFP_prop_list = IFP_list(property_list, sel_ligands, RE, Lipids)
    u_list_all = []
    for IFP_type  in IFP_prop_list:
        line = IFP_type.sel_a + " and around " + str(IFP_type.dist) + " " + IFP_type.sel_b
        u_list_all.append(u_mem.select_atoms(line, updating=True))
    start = 0
    IFPs_unique_list = []
    for i, _ts in enumerate(u_mem.trajectory):
        _ = u_mem.trajectory[i]
        for u_list, IFP_type in zip(u_list_all, IFP_prop_list):
            found = []
            if IFP_type.name == "WA":
                for u in u_list:
                    found.append(["WAT", u.name])
            elif IFP_type.name == "LL":
                for u in u_list:
                    found.append(["LIP", u.name])
            elif IFP_type.name == "AR":
                u_ar = []
                u_ar_n = []
                for u in u_list:
                    u_ar.append(u.resid)
                    u_ar_n.append(u.resname)
                if len(u_ar) > 0:
                    ar_resid, ar_n = np.unique(u_ar, return_counts=True)
                    for u in u_list:
                        # check if this aromatic residue has more than 4 contacts with an aromatic fragment of a ligand
                        # print("AROMATIC  ",ar_resid,ar_n,u_ar,u_ar_n)
                        if u.resid in ar_resid[ar_n > 4]:
                            # check also residue name to deal the case of residues with the same id
                            if np.unique(np.asarray(u_ar_n)[np.where(u_ar==u.resid)[0]]).shape[0] == 1:
                                found.append([IFP_type.name+"_"+u.resname+str(u.resid),u.name])
                        # here we will check if cation (LYS or ARG) really contact an aromatic ring of the ligand
                        elif u.resname in ["LYS","ARG"]:
                            cation = u.resname
                            if "Aromatic" in property_list.keys():
                                line_ar = ""
                                for l in np.asarray(property_list["Aromatic"]): line_ar = line_ar + l +" "
                                line1 = "(resname "+sel_ligands+" and ( not type H O) and name "+line_ar+") and around "+str(r_cat)+" (resid "+str(u.resid[u.resname == cation][0]) + " and type N )" 
                                u1_list = u_mem.select_atoms(line1,updating=True)
                                if len(u1_list) > 4:
                                    found.append([IFP_type.name+"_"+u.resname+str(u.resid),u.name])
                        # now we check if aromatc residue is perpendicular to the aromatic fragment of the ligand (face-to-edge)
                        ### TOBE checked if this works!!!!!================================================
                        elif u.resname in resi_aromatic and u.resid in ar_resid[ar_n <= 4]:
                            if "Aromatic" in property_list.keys():
                                line_ar = ""
                                for l in np.asarray(property_list["Aromatic"]):
                                    line_ar = line_ar + l +" "
                                line1 = "(resname "+sel_ligands+" and ( not type H O) and name "+line_ar+") and around "+str(r_ari)+" (resid "+str(u.resid) + " and (name NE* ND* CE* CD* CZ* CH* CG*))"
                                u1_list = u_mem.select_atoms(line1, updating=True)
                                if len(u1_list) > 4:
                                    found.append([IFP_type.name+"_"+u.resname+str(u.resid),u.name])
                        ### TOBE checked if this works!!!!!================================================
            elif IFP_type.name == "HL":
                for u in u_list:
                    # here we will check if the angle  C-HAL.... O is about 180grad m this we look what atoms within r_hal+1 from O - should be only Hal
                    if u.type == "O" or u.type == "S":
                        line1 ="(resname "+sel_ligands+" ) and around "+str(r_hal+1.0)+" (protein and resid "+str(u.resid)+" and name O* S* )"
                        u1_list = u_mem.select_atoms(line1,updating=True)
#                        print(u.resid,u.name,":::",len(u1_list),u1_list)
                        if len(u1_list) < 2:
                            found.append([IFP_type.name+"_"+u.resname+str(u.resid),u.name])
 #                       else: print("HL-O contact found but will not be counted because of the too small angle: ",len(u1_list),u.resname+str(u.resid),u.name)
                        # TO BE DONE instead of previous criterion
                        # """
                        # else:
                        #     u1_list = (u_mem.select_atoms(" (resid "+str(u.resid)+" and type O )",updating=True)) # resi
                        #     u2_list = (u_mem.select_atoms("(resname "+sel_ligands+" and type Cl CL Br BR I) and around "+str(r_hal+1.0)+" (resid "+str(u.resid)+" and type O)",updating=True)) # ligand hal
                        #     u3_list = (u_mem.select_atoms("(resname "+sel_ligands+" and type C) and around "+str(r_hal+1.0)+" (resid "+str(u.resid)+" and type O)",updating=True)) # ligand carbon
                        #     B_center = B.centroid(u1_list)
                        #     BA = A.centroid(u2_list) - B_center
                        #     BC = C.centroid(u3_list) - B_center
                        #     alpha = np.arccos(np.dot(BA, BC)/(norm(BA)*norm(BC)))
                        #     if alpha > angle_CHal_O:
                        #         found.append([IFP_type.name+"_"+u.resname+str(u.resid),u.name])
                        # """

                # now we check if halogen atom is perpendicular to the aromatic residue
                ### TOBE checked if this works!!!!!
                # ====HAL=====================================
                u_ar = [u.resid for u in u_list if u.resname in resi_aromatic]
                if len(u_ar) > 0:
                    ar_resid, ar_n = np.unique(u_ar, return_counts=True)
                    for u in u_list:
                        if u.resid in ar_resid[ar_n > 4]:
                            found.append([IFP_type.name + "_" + u.resname + str(u.resid), u.name])
                ### TOBE checked if this works!!!!!====HAL========================================
            else:
                #print("HY/IP: ",IFP_type.name,len(u_list))
                for u in u_list:
                        found.append([IFP_type.name+"_"+u.resname+str(u.resid),u.name]) 
#                        if IFP_type.name == "HL": print("HAL:",u.resname,u.resid,u.name)

            if found:
                IFP_type.contacts.append((i,found))
                if start == 0:
                    IFPs_unique_list = np.unique(np.asarray(found)[:,0])
                    start += 1
                else:
                    IFPs_unique_list = np.unique(np.append(IFPs_unique_list,np.asarray(found)[:,0]))
#                print(IFPs_unique_list)

    logger.info(f"Start building IFP table at {datetime.datetime.now().time()}")
    if len(IFP_prop_list) > 0:
        columns,IFP_matrix = make_IFT_table(IFP_prop_list,len(u_mem.trajectory), columns_extended=IFPs_unique_list)
        df_prop = pd.DataFrame(data=IFP_matrix, index=None, columns=columns)
        logger.success(f"IFP database is ready at {datetime.datetime.now().time()}")
        return df_prop, df_HB, df_WB
    logger.critical("Something is wrong - IFP property list is empty")
    raise ValueError()


def Water_bridges(u_mem, sel_ligands, WB_debug=False):
    """
    A very simple procedure for detection of possible protein-ligand water bridges 
    Parameters:
    u_mem - trajectory
    residues_name a list of properties that (column names) that can be used to generate tables with the same column list
    sel_ligands - ligand residue name 
    Returns:
    df_WB - pkl table with all components of water bridges
    """
    col_exchange = {"sele1_index": "sele2_index", "sele2_index": "sele1_index" ,"sele1_resnm": "sele2_resnm", "sele1_resid": "sele2_resid", "sele1_atom": "sele2_atom","sele2_resnm": "sele1_resnm", "sele2_resid": "sele1_resid","sele2_atom": "sele1_atom"}
    col_transfer = {"donor_index": "sele1_index", "acceptor_index": "sele2_index","donor_resnm": "sele1_resnm", "donor_resid": "sele1_resid","donor_atom": "sele1_atom", "acceptor_resnm": "sele2_resnm","acceptor_resid": "sele2_resid","acceptor_atom": "sele2_atom"}

    angle_th =100
    dist_th =3.3

    #-------------------------------------------------------------
    def clean_dataset(wb_check, sel_ligands):
        """
        function that checks if water molecule has contacts to both protein and ligand and removes water that are not
        Parameters:
        dataset of potential water bridges
        Returns:
        clean dataset
        """
#        print(wb_check)
        water_list_d = wb_check[wb_check["donor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])].donor_resid.values
        water_list_a = wb_check[wb_check["acceptor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])].acceptor_resid.values
        l = np.concatenate((water_list_d, water_list_a))
        res_w, c_w = np.unique(l,return_counts=True)
        fexcl = res_w[c_w<2]
        wb_check_cleaned = wb_check
        # check if water appeares just once - remove it then:
        if fexcl.shape[0] > 0:
            if not wb_check_cleaned[wb_check_cleaned.donor_resid.isin(fexcl)].empty:
                wb_check_cleaned = wb_check_cleaned[~wb_check_cleaned.donor_resid.isin(fexcl)]
            if not wb_check_cleaned[wb_check_cleaned.acceptor_resid.isin(fexcl)].empty:
                wb_check_cleaned = wb_check_cleaned[~wb_check_cleaned.acceptor_resid.isin(fexcl)]
        # check if water connects to different residues:
        for wat in res_w:
            nowat_list_d = wb_check_cleaned[wb_check_cleaned["donor_resid"]== wat]
            nowat_list_a = wb_check_cleaned[wb_check_cleaned["acceptor_resid"] == wat]
            res, c = np.unique(np.concatenate((nowat_list_d.acceptor_resid.values, nowat_list_a.donor_resid.values)),return_counts=True)
            both_prot = True
            #check if ligand has contacts with water
            for r in res:
                if nowat_list_d[nowat_list_d.acceptor_resid == r].acceptor_resnm.values.shape[0]> 0:
                    if sel_ligands in nowat_list_d[nowat_list_d.acceptor_resid == r].acceptor_resnm.values:  both_prot = False
                if nowat_list_a[nowat_list_a.donor_resid == r].donor_resnm.values.shape[0]> 0:
                    if sel_ligands in nowat_list_a[nowat_list_a.donor_resid == r].donor_resnm.values: both_prot = False
            if res.shape[0] < 2  or both_prot:
                if not wb_check_cleaned[wb_check_cleaned["donor_resid"]== wat].empty:
                    wb_check_cleaned = wb_check_cleaned[~(wb_check_cleaned["donor_resid"]== wat)]
                if not wb_check_cleaned[wb_check_cleaned["acceptor_resid"] == wat].empty:
                    wb_check_cleaned = wb_check_cleaned[~(wb_check_cleaned["acceptor_resid"] == wat)]
#        print(wb_check_cleaned)
        return wb_check_cleaned
        #-------------------------------------------------

    df_WB = pd.DataFrame()

    # 1. we will find a list of ligand- water h-bonds in the all trajectory
    hba.HydrogenBondAnalysis.DEFAULT_DONORS['OtherFF'] = hba.WaterBridgeAnalysis.DEFAULT_DONORS['OtherFF']+tuple(set("O"))
    h = hba.HydrogenBondAnalysis(u_mem, selection1 ='resname '+sel_ligands,selection2='  resname WAT HOH SOL TIP3', distance=dist_th, angle=angle_th, forcefield='OtherFF') #,update_selection1= False)
    h.run()
    h.generate_table()
    wb1_tot = pd.DataFrame.from_records(h.table)
    if WB_debug:
        logger.debug(f"1, Lig-Wat, -----------------\n{wb1_tot}")
    # 2. make a list of water molecules 
    lista = []
    if not wb1_tot[wb1_tot.donor_resnm == sel_ligands].empty:
        lista = wb1_tot[wb1_tot.donor_resnm == sel_ligands].acceptor_resid.values
    listd = []
    if not wb1_tot[wb1_tot.acceptor_resnm == sel_ligands].empty:
        listd = wb1_tot[wb1_tot.acceptor_resnm == sel_ligands].donor_resid.values

    if len(lista)+len(listd) > 0:    
        list_wb = "resid "
        for l in np.unique(lista):
            list_wb = list_wb + str(l)+" " 
        for l in np.unique(listd):
            list_wb = list_wb + str(l)+" " 
    #3. make a table of water- protein contacts (only selected water molecules are considered)
        h = hba.HydrogenBondAnalysis(u_mem, selection1 =list_wb,selection2=' not resname WAT HOH SOL TIP3'+sel_ligands, distance=dist_th , angle=angle_th, forcefield='OtherFF')
        h.run()
        h.generate_table()
        wb2_tot = pd.DataFrame.from_records(h.table)
        if WB_debug:
            logger.debug(f"2, Prot-Wat -----------------\n{wb2_tot}")
            
        if wb2_tot.shape[0] > 0: 
            # loop over frames
            for time, ts in enumerate(u_mem.trajectory):
                u_mem.trajectory[time] 
                wb2 = wb2_tot[wb2_tot.time.astype(int) == time]
                wb1 = wb1_tot[wb1_tot.time.astype(int) == time]
                if wb2.shape[0] > 0:
                    # exclude the cases where the same hydrogen is a donor for both protein and ligand
                    wb2 = wb2[~(wb2.donor_index.isin(wb1[wb1["donor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])].donor_index))]
                    # exclude the cases where the same oxigen is an acceptor for both protein and ligand
                    wb2 = wb2[~(wb2.acceptor_index.isin(wb1[wb1["acceptor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])].acceptor_index))]
                    wb12 = wb1.append(wb2)
# 5. check additionally angles
# 5(a) make a list water molecules  that have H-bonds with a ligand
#                    list_ld = wb12[wb12.acceptor_resnm ==  sel_ligands].donor_resid.values
#                    list_la = wb12[wb12.donor_resnm == sel_ligands].acceptor_resid.values
#                    list_w_l = np.concatenate((list_la, list_ld))
#                    if len(list_w_l) == 0: continue 
#                    wb12 = wb12[(wb12.donor_resid.isin(list_w_l))].append(wb12[(wb12.acceptor_resid.isin(list_w_l))])
                    wb12 = clean_dataset(wb12,sel_ligands)
                    if WB_debug:
                        logger.debug(f"{time} -----------------\n{wb12}")
                    if wb12.empty:
                        continue
                    wat_donor = wb12[wb12["donor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])]
                    wat_acceptor = wb12[wb12["acceptor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])]                   
                    lig_donor = wb12[wb12["donor_resnm"].isin([sel_ligands])]
                    prot_donor = wb12[~(wb12["donor_resnm"].isin(["WAT", "HOH", "SOL","TIP3", sel_ligands]))]
                    
# 5(a) ---- check the angle for prot/lig(H)...water(O)...water(H) and remove lines in the table  where it is less then threshold
                    if not prot_donor.empty or not lig_donor.empty:
                        check_list =[]
                        # loop over water donor atoms
                        for wat_hid,wat_hat in zip(wat_donor.donor_resid.values,wat_donor.donor_atom.values):
                        #    if wat_hid not in list_w_l: continue
                            wat_oid = wat_acceptor[wat_acceptor["acceptor_resid"] == wat_hid].acceptor_resid.values 
                            # loop over water acceptor atoms
                            for wid in wat_oid:
                                # loop over non-water donor bound to water acceptors
                                wid_nonwat = wat_acceptor[wat_acceptor["acceptor_resid"] == wid]
                                for (nowat_hid,nowat_hat,nowat_hin) in zip(wid_nonwat.donor_resid,wid_nonwat.donor_atom,wid_nonwat.donor_index):
                                        if (wat_hid,wat_hat,nowat_hid,nowat_hat,nowat_hin) not in check_list:
                                            check_list.append((wat_hid,wat_hat,nowat_hid,nowat_hat,nowat_hin))
 #                       print("Check list:  ", check_list)
                        for wat_hid, wat_hat, nowat_hid, nowat_hat, nowat_hin in check_list:
                            try:
                                angles = np.rad2deg(
                                    calc_angles(
                                        u_mem.select_atoms("resid "+str(wat_hid)+" and name "+wat_hat,updating=True).positions,
                                        u_mem.select_atoms("resid "+str(wat_hid)+" and name O*",updating=True).positions,
                                        u_mem.select_atoms("resid "+str(nowat_hid)+" and name "+nowat_hat,updating=True).positions,
                                    )
                                    )
#                                print("Check angle: ",np.round(angles[0],1)," resid "+str(wat_hid)+" "+wat_hat," resid "+str(wid)+" O","resid "+str(nowat_hid)+" "+nowat_hat)
                            except:
                                angles = 0     
#                               print("Warning: problem with WB angles (maybe some residue numbers are duplicated): "+" resid "+str(wat_hid)+" "+wat_hat," resid "+str(wid)+" O","resid "+str(nowat_hid)+" "+nowat_hat)
                            if angles < angle_th:   
                                wr =(wat_hid,nowat_hin)
                                if WB_debug:
                                    logger.debug(f"REMOVE incorrect H-bonds from the WB list: {wr} , {nowat_hid} , {nowat_hat}")
                                wb12 = wb12[~((wb12.acceptor_resid == wr[0]) & (wb12.donor_index == wr[1]))]
#                               print("INTERMEDIATE -----------------\n",wb12[( (wb12.acceptor_resid == wr[0]) & (wb12.donor_index == wr[1]))])
        
# 5(b) make a list water molecules  that have H-bonds with a ligand, but first revise table
#                    print(time," -----------------\n",wb12)
#                    tt = clean_dataset(wb12)
#                    list_ld = wb12[wb12.acceptor_resnm ==  sel_ligands].donor_resid.values
#                    list_la = wb12[wb12.donor_resnm == sel_ligands].acceptor_resid.values
#                   list_w_l = np.concatenate((list_la, list_ld))
#                    wb12 = wb12[(wb12.donor_resid.isin(list_w_l))].append(wb12[(wb12.acceptor_resid.isin(list_w_l))])
#                    if len(list_w_l) == 0: continue        
                    wb12 = clean_dataset(wb12,sel_ligands)
                    if wb12.empty:
                        continue
#                    print(time," -----------------\n",wb12)
                    
                    wat_donor = wb12[wb12["donor_resnm"].isin(["WAT", "HOH", "SOL","TIP3"])]                    
                    lig_acceptor = wb12[wb12["acceptor_resnm"].isin([sel_ligands])]
                    prot_acceptor = wb12[~(wb12["acceptor_resnm"].isin(["WAT", "HOH", "SOL","TIP3", sel_ligands]))]
                    
# 5(b) ---- check the angle for prot/lig(O)...water(H)...water(O) 
                    if not prot_acceptor.empty or not lig_acceptor.empty:
                        check_list =[]
                        # loop over water acceptor atoms
                        for wat_hid,wat_hat in zip(wat_donor.donor_resid.values,wat_donor.donor_atom.values):
                            oid = wat_donor[wat_donor["donor_resid"] == wat_hid].acceptor_resid.values 
                            oin = wat_donor[wat_donor["donor_resid"] == wat_hid].acceptor_index.values 
                            oat = wat_donor[wat_donor["donor_resid"] == wat_hid].acceptor_atom.values
                            for (oid_nonwat, oat_nonwat, oin_nonwat) in zip(oid,oat,oin):
                                if( wat_hid,wat_hat,oid_nonwat, oat_nonwat, oin_nonwat) not in check_list :
                                    check_list.append((wat_hid,wat_hat,oid_nonwat, oat_nonwat, oin_nonwat))
#                        print("Check list: ",check_list)
                        for (wat_hid,wat_hat,oid_nonwat, oat_nonwat, oin_nonwat) in check_list:
                            try:
                                angles = np.rad2deg(
                                    calc_angles(
                                        u_mem.select_atoms("resid "+str(oid_nonwat)+" and name "+oat_nonwat,updating=True).positions,
                                        u_mem.select_atoms("resid "+str(wat_hid)+" and name "+wat_hat,updating=True).positions,
                                        u_mem.select_atoms("resid "+str(wat_hid)+" and name O*",updating=True).positions,
                                    )
                                    )
#                                print("Check angle: ",np.round(angles[0],1)," resid "+str(wat_hid)+" "+wat_hat," resid "+str(wid)+" O","resid "+str(oid_nonwat)+" "+oat_nonwat)
                            except:
                                angles = 0     
#                                print("Warning: problem with WB angles (maybe some residue numbers are duplicated): "+" resid "+str(wat_hid)+" "+wat_hat," resid "+str(wid)+" O","resid "+str(oid_nonwat)+" "+oat_nonwat)
                            if angles < angle_th:   
                                wr =(wat_hid,wat_hat,oin_nonwat)
                                if WB_debug:
                                    logger.debug(f"REMOVE incorrect H-bonds from the WB list: {wr} , {oid_nonwat} , {oat_nonwat}")
                                wb12 = wb12[~((wb12.donor_resid == wr[0]) & (wb12.donor_atom == wr[1]) & (wb12.acceptor_index == wr[2]))]
                            
                                                       
#6. change structure - rename donor/acceptor to sel_1 / sel_2 and move ligand to sel_1                            
                    wb12 = wb12.rename(columns=col_transfer)
                    # bring table to the following structure: sel1 - lig or water ; sel2- protein or water
                    # ---- sel1: lig - sel2: water
                    WB_t = wb12[wb12.sele1_resnm.isin([sel_ligands])] 
                    # ---- sel1: wat - sel2: any
                    wb12_t = wb12[wb12.sele1_resnm.isin(["WAT", "HOH", "SOL","TIP3"])] 
                    # ---- sel1: wat - sel2: prot
                    WB_t = WB_t.append(wb12_t[~(wb12_t.sele2_resnm.isin([sel_ligands]))])  
                    # ---- sel1:water - sel2: lgand  - replace sel1 and sel2
                    WB_t = WB_t.append(wb12[wb12.sele2_resnm.isin([sel_ligands])].rename(columns=col_exchange))
                    # ---- sel1:prot - sel2: wat - replace sel1 and sel2
                    WB_t = WB_t.append(wb12[~wb12.sele1_resnm.isin(["WAT", "HOH", "SOL","TIP3",sel_ligands])].rename(columns=col_exchange)) 
                    df_WB = df_WB.append(WB_t)
 #   print(df_WB)
    return df_WB