#!/usr/bin/env python

from dataclasses import dataclass
import datetime
import numpy as np
import pandas as pd
from loguru import logger

import MDAnalysis as mda
import MDAnalysis.analysis.hydrogenbonds.hbond_analysis as hba
import MDAnalysis.analysis.hydrogenbonds.wbridge_analysis as wba

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

def IFP(u_mem, sel_ligands, property_list, WB_analysis=True, RE=True, Lipids=None):
    """
    Parameters:
    u - trajectory - universe object
    ligand name -  ligand residue name
    property_list - python dictionary of ligand atom properties (created by ligand_analysis)
    
    Reterns:
    """

    Lipids = [] if Lipids is None else Lipids

    logger.debug("Computing HBONDS with new MDA (>2.0), with parameters adjusted to match the old selection as much as possible")
    h = hba.HydrogenBondAnalysis(u_mem,
                                between = [f'resname {sel_ligands}', f'not resname WAT HOH SOL {sel_ligands}'],
                                d_h_cutoff=1.35, d_a_cutoff=3.35, d_h_a_angle_cutoff=100)
    h.hydrogens_sel = h.guess_hydrogens(max_mass=3.025, min_charge=0.2) # to account for H-mass repartition
    h.donors_sel = h.guess_donors(max_charge=-0.3)
    h.acceptors_sel = h.guess_acceptors()

    logger.debug("hydrogens_selection:\n", h.hydrogens_sel)
    logger.debug("donors_selection:\n", h.donors_sel)
    logger.debug("acceptors_selection:\n", h.acceptors_sel)

    # upa = u_mem.select_atoms("resname UPA")
    # for atom in upa.atoms:
    #     print(atom.name, atom.charge)
    # DEFAULT_DONORS = ('NH2', 'NZ', 'OG', 'NH1', 'N', 'NE', 'OG1', 'NE1', 'OH', 'NE2', 'ND1', 'SG', 'ND2', 'OH2', 'OW')
    # DEFAULT_ACCEPTORS = ('OD1', 'OD2', 'OG', 'OC2', 'SD', 'OG1', 'OH', 'NE2', 'OH2', 'OC1', 'ND1', 'SG', 'OE2', 'OE1', 'O', 'OW')
    # if "Donor" in set(property_list):
    #     donor_line = tuple(set(property_list["Donor"]))
    #     logger.debug(f"Adding {donor_line} to DEFAULT_DONORS")
    #     DEFAULT_DONORS += donor_line
    # if "Acceptor" in set(property_list):
    #     acceptor_line = tuple(set(property_list["Acceptor"]))
    #     logger.debug(f"Adding {acceptor_line} to DEFAULT_ACCEPTORS")
    #     DEFAULT_ACCEPTORS += acceptor_line
    # donors_sel = "name "
    # for name in DEFAULT_DONORS[:-1]:
    #     donors_sel += name + " or name "
    # donors_sel += DEFAULT_DONORS[-1]
    # acceptor_sel = "name "
    # for name in DEFAULT_ACCEPTORS[:-1]:
    #     acceptor_sel += name + " or name "
    # acceptor_sel += DEFAULT_ACCEPTORS[-1]
    # h = hb.HydrogenBondAnalysis(u_mem,
    #                             between = [f'resname {sel_ligands}', f'not resname WAT HOH SOL {sel_ligands}'],
    #                             d_h_cutoff=1.5, d_a_cutoff=3.5, d_h_a_angle_cutoff=100,
    #                             donors_sel=donors_sel, acceptors_sel=acceptor_sel)
    # h.hydrogens_sel = h.guess_hydrogens(max_mass=3.025, min_charge=0.2) # to account for H-mass repartition

    logger.info(f"Start HB analysis at {datetime.datetime.now().time()}")
    h.run()
    logger.info(f"         ... done at {datetime.datetime.now().time()}")

    atoms: mda.AtomGroup = u_mem.atoms
    hb_table = []
    for hbresults in h.results.hbonds:
        time = float(hbresults[0])
        hydrogen_index = int(hbresults[2])
        acceptor_index = int(hbresults[3])
        distance = float(hbresults[4])
        angle = float(hbresults[5])
        hydrogen = atoms[hydrogen_index]
        acceptor = atoms[acceptor_index]
        hb_table.append([
            time, hydrogen_index, acceptor_index,
            hydrogen.resname, hydrogen.resid, hydrogen.name,
            acceptor.resname, acceptor.resid, acceptor.name,
            distance, angle
        ])

    df_HB = pd.DataFrame(hb_table, columns=[
        'time', 'donor_index', 'acceptor_index',
        'donor_resnm', 'donor_resid', 'donor_atom',
        'acceptor_resnm', 'acceptor_resid', 'acceptor_atom',
        'distance', 'angle'])
    logger.debug("df from HB results")
    logger.debug(df_HB)

    if WB_analysis :
        logger.info(f"Start WB analysis at {datetime.datetime.now().time()}")
        wba.WaterBridgeAnalysis.DEFAULT_DONORS['OtherFF'] = wba.WaterBridgeAnalysis.DEFAULT_DONORS['CHARMM27']
        wba.WaterBridgeAnalysis.DEFAULT_ACCEPTORS['OtherFF'] = wba.WaterBridgeAnalysis.DEFAULT_ACCEPTORS['CHARMM27']
        #NOTE: this is needed to "match" the original results, but it's strange...
        wba.WaterBridgeAnalysis.DEFAULT_DONORS['OtherFF'] += tuple(set("O"))
        if "Donor" in set(property_list):
            donor_line = tuple(set(property_list["Donor"]))
            logger.debug(f"Adding {donor_line} to wba DEFAULT_DONORS")
            wba.WaterBridgeAnalysis.DEFAULT_DONORS['OtherFF'] += donor_line
        if "Acceptor" in set(property_list):
            acceptor_line = tuple(set(property_list["Acceptor"]))
            logger.debug(f"Adding {acceptor_line} to wba DEFAULT_ACCEPTORS")
            wba.WaterBridgeAnalysis.DEFAULT_ACCEPTORS['OtherFF'] += acceptor_line
        wb = wba.WaterBridgeAnalysis(u_mem,
                                    selection1=f'resname {sel_ligands}',
                                    selection2=f'not resname WAT HOH SOL {sel_ligands}',
                                    water_selection="resname WAT HOH SOL",
                                    forcefield="OtherFF", distance=3.3, angle=100)
        wb.run()
        wb_table = wb.generate_table()
        df_WB = pd.DataFrame.from_records(wb_table)

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
                    if u.type in ["O", "S"]:
                        line1 ="(resname " + sel_ligands + " ) and around " + str(r_hal+1.0) + " (protein and resid " + str(u.resid) + " and name O* S* )"
                        u1_list = u_mem.select_atoms(line1,updating=True)
                        if len(u1_list) < 2:
                            found.append([IFP_type.name + "_" + u.resname + str(u.resid), u.name])
                u_ar = [u.resid for u in u_list if u.resname in resi_aromatic]
                if len(u_ar) > 0:
                    ar_resid, ar_n = np.unique(u_ar, return_counts=True)
                    for u in u_list:
                        if u.resid in ar_resid[ar_n > 4]:
                            found.append([IFP_type.name + "_" + u.resname + str(u.resid), u.name])
                 ### TOBE checked if this works!!!!!====HAL========================================
            else:
                for u in u_list:
                    found.append([IFP_type.name + "_" + u.resname + str(u.resid), u.name]) 

            if found:
                IFP_type.contacts.append((i,found))
                if start == 0:
                    IFPs_unique_list = np.unique(np.asarray(found)[:,0])
                    start += 1
                else:
                    IFPs_unique_list = np.unique(np.append(IFPs_unique_list,np.asarray(found)[:,0]))

    logger.info(f"Start building IFP table at {datetime.datetime.now().time()}")
    if len(IFP_prop_list) > 0:
        columns,IFP_matrix = make_IFT_table(IFP_prop_list,len(u_mem.trajectory), columns_extended=IFPs_unique_list)
        df_prop = pd.DataFrame(data=IFP_matrix, index=None, columns=columns)
        logger.success(f"IFP database is ready at {datetime.datetime.now().time()}")
        return df_prop, df_HB, df_WB
    logger.critical("Something is wrong - IFP property list is empty")
    raise ValueError()
