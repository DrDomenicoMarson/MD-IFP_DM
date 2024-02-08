#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

from loguru import logger


import glob, os
import sys
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem import EState
#from rdkit.Chem import MolDb
#from rdkit.Chem import QED
from rdkit.Chem import rdchem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import Draw

import MDAnalysis as mda
#from MDAnalysis.lib.formats.libdcd import DCDFile
from MDAnalysis.analysis import contacts,align,rms
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

#from sklearn import linear_model
#from sklearn import preprocessing

from IFP_generation import IFP, table_combine #, Plot_IFP


class Trj_Properties:
    def __init__(self):
        self.start = 0
        self.stop = 0
        self.step = 1
        self.length = 0
        self.df_properties = None  
        self.rmsd_prot = None  
        self.rmsd_lig = None
        self.rgyr_prot = None
        self.rgyr_lig = None
        self.com_lig = None
        self.rmsd_auxi = None



class trajectories:
    """
   
    1 Functions:
        __init__    
    2 Variables: (see description in the function __init__())
        ligand - object of the class ligand
        namd -  object of the class namd
        ramd -  object of the class ramd
        PRJ_DIR
        pdb
        top
        ramd_traj_tmpl  
        namd_traj_tmpl 
        ramd_tmpl 
        namd_tmpl 
        timestep         
        tau_exp 
        tau_exp_SD
        type

        
    3 Sub-Classes
        3.1 ligand
                Functions:
                    _int__
                Variable:
                    ligand_names
                    property_list
                    mol
                    ligand_2D
                    radius
            
        3.2 namd  
                Functions:
                    __init__
                Variable:
                    repl_traj = []
                    repl_names = [] 
                    length = []
                    step = 1
                    start = 0
                # the next set of parameters are computed by the analysis_all_namd() function
                    df_properties = []
                    rmsd_prot = []
                    rmsd_lig = []
                    rgyr_prot = []
                    rgyr_lig = []
                    contact_collection = []  # this is a container where we will accomulate all protein-ligand contacts

        3.3 ramd  
                Functions:
                    __init__
                    IFP_unify(self,subset = [])
                    IFP_save(self,subset=[],file="IFP.pkl")
                Variables:
                    repl_traj = []
                    repl_names = [] 
                    length = []
                    replicas_distr = [] # distribution for each replica that is generated in the bootstrapping procedure
                    replicas = []  # tauRAMD for each replica
                    replicas_SD = []
                    contact_collection - a complete list of contact residues  

    """
    
       
    #========================================================    
    def __init__(self,PRJ_DIR = "./",namd_tmpl= "NAMD*", ramd_tmpl= "RAMD*", pdb = "ref.pdb", 
                ligand_pdb = None,ligand_mol2 = None,\
                ramd_traj_tmpl = "*dcd",namd_traj_tmpl = "*dcd",timestep=1):
        """
        Class constructor:
        
        INPUT PARAMETERS:
        PRJ_DIR - directory to story data
        namd_dirs - TEMPLATE used for search of directories with NAMD simulations (should contain also complete path)
        ramd_dirs - TEMPLATE used for search of directories with RAMD simulations (should contain also complete path)
        timestep - timestep used to save MD frames in ps
                
        """
        #---------  check/make directory to work in -----------
#                try:  
#                    os.mkdir(PRJ_DIR)
#                except OSError:  
#                    print ("Creation of the directory %s failed" % PRJ_DIR)
#                else:  
#                    print ("Data generated will be located at %s " % PRJ_DIR)

        if not os.path.isdir(PRJ_DIR):
            PRJ_DIR = os.getcwd()
        if PRJ_DIR[-1] != "/":
            PRJ_DIR = PRJ_DIR+"/"
        self.PRJ_DIR = PRJ_DIR

        #------------ check ref structure --------------
        if not os.path.isfile(PRJ_DIR+pdb):
            logger.error(f"file {PRJ_DIR+pdb} was not found ")
            return

        self.pdb = pdb
        self.ramd_traj_tmpl ="/"+ramd_traj_tmpl
        self.namd_traj_tmpl = "/"+namd_traj_tmpl
        self.ramd_tmpl ="/"+ramd_tmpl
        self.namd_tmpl = "/"+namd_tmpl
        self.timestep = timestep
        self.sub_system = "protein or (resname MG MN CA Mg Mn) or (resname WAT HOH SOL TIP3)"  # G G3 G5 U5 C C3  can be added for RNA

        try:
            self.ligand = Ligand(PRJ_DIR, ligand_pdb, ligand_mol2)
            u = mda.Universe(self.PRJ_DIR+self.pdb)
            lig_atoms = u.select_atoms(f'resname {self.ligand.ligands_names[0]}')
            logger.info(f"Ligand {self.ligand.ligands_names[0]} found in the trajectory with {len(lig_atoms)} atoms")
            for atom in lig_atoms:
                logger.debug(atom)

        except mda.exceptions.SelectionError as exc:
            logger.exception(f"Selection error with this ligand name: {self.ligand.ligands_names[0]}")
            raise exc

        #self.ligand = self.createLigand(PRJ_DIR,ligand_pdb,ligand_mol2)
        self.namd = self.createNamd() 
        self.ramd = self.createRamd(PRJ_DIR,pdb,timestep) 
        self.tau_exp = None
        self.tau_exp_SD = None
        self.type = None

        logger.info(f"Equilibration trajectories will be searched using the template: {PRJ_DIR+self.namd_tmpl+self.namd_traj_tmpl}")
        for file_n in np.sort(glob.glob(PRJ_DIR+self.namd_tmpl+self.namd_traj_tmpl)): 
            if ((file_n.find("vel") < 0)  and (file_n.find("heat") < 0)) and os.path.isfile(file_n):
                self.namd.repl_traj.append(file_n)
                self.namd.names.append(os.path.basename(os.path.dirname(file_n)))
        logger.info(f"Equilibration trajectories found: {len(self.namd.names)}")

        logger.info(f"Dissociation trajectories will be searched using the template: {PRJ_DIR+self.ramd_tmpl+self.ramd_traj_tmpl}")
        for dir_ramd in np.sort(glob.glob(PRJ_DIR+self.ramd_tmpl)):
            ramd_list = []
            for file_n in np.sort(glob.glob(dir_ramd+self.ramd_traj_tmpl)):
                if (file_n.find("vel") < 0) and os.path.isfile(file_n):
                    ramd_list.append(file_n)
            if len(ramd_list)> 0: 
                self.ramd.repl_traj.append(ramd_list)
                self.ramd.names.append(os.path.basename(dir_ramd))
                logger.info(f"{len(ramd_list)} RAMD traj found in {dir_ramd}")
        return

    #========================================================
    class Namd:
        """
                Functions:
                    __init__(self)
                    compare_all_namd()
                    IFP_unify(self,subset = [])
                    IFP_save(self,subset=[],file="IFP.pkl")
                Variable:
                    repl = []
                    names = [] 
                    length = []
                    step = 1
                    start = 0
                    contact_collection = []
                # the next set of parameters are computed by the analysis_all_namd() function
                    traj = []
                    contact_collection = []  # this is a container where we will accomulate all protein-ligand contacts
        """
        def __init__(self, timestep=1):
            self.repl_traj = []  # a list of full pathes to a equilibation trajectories 
            self.names = [] 
            self.length = []
            self.step = 1
            self.start = 0
            self.timestep = timestep
            # the next set of parameters will be filled by the analysis_all_namd() function
            self.traj = []   # an array of Trj_Properties objects for all trajectories in a replica
            self.contact_collection = []  # this is a container where we will accomulate all protein-ligand contacts

        ##############################################################
        #
        #  function that analyze all NAMD trajectories for a particular compound 
        #
        ###############################################################
        def compare_all_namd(self):
            """
            Parameters:
            uses results of the analysis_all_namd function
            Results:
            """
            IFP_list = []
            for j,nmd in enumerate(self.names):
                for c in self.traj.df_properties[j].columns[1:].tolist():
                    if c  in IFP_list:
                        pass
                    elif(c != "WAT"):
                        IFP_list.append(c)

            all_namd_prop  = pd.DataFrame(np.zeros((len(self.names),len(IFP_list))),index = self.names, columns = IFP_list) 

            for j,nmd in enumerate(tr.names):
  #              print(nmd,self.names[j],self.replicas[j])
                a = self.traj.df_properties[j][self.traj.df_properties[j].columns[1:]].mean(axis=0)
                for key in a.index:
                    all_namd_prop.loc[nmd][key] = a.loc[key]
            fig = plt.figure(figsize=(6, 4))
            sns.heatmap(all_namd_prop, cmap="Greys", linewidths=.5)
            plt.show()
            return(all_namd_prop)


        ###################################################################
        #
        # FUNCTION to unify comlumns of a set of IFP databases for generated from several trajectories
        #
        ###################################################################
        def IFP_unify(self,subset = []):
            """
            Parameters:
            optional - a list of IFP to be used to unify tables of IFP for all compounds
            Results:
            """
            if len(subset) == 0: r_subset = self.traj
            else: r_subset = np.take(self.traj,subset)
            IFP_list = []

            for j,tr_c in enumerate(r_subset):
                try:
                    for c in tr_c.df_properties.columns.tolist():
                        if c  in IFP_list:    pass
                        else: IFP_list.append(c)
                except:
                    pass
            for j,tr_c in enumerate(r_subset):
                try:
                    to_add = np.setdiff1d(IFP_list, np.asarray(tr_c.df_properties.columns.tolist()))
                    for k in to_add:
                        tr_c.df_properties[k] = np.zeros(tr_c.df_properties.shape[0],dtype=np.int8)
                    tr_c.df_properties = tr_c.df_properties[np.concatenate((["time"],tr_c.df_properties.columns[tr_c.df_properties.columns != "time"].tolist()))]
                    if "WAT" in tr_c.df_properties.columns.tolist():
                        tr_c.df_properties = tr_c.df_properties[np.concatenate((tr_c.df_properties.columns[tr_c.df_properties.columns != "WAT"].tolist(),["WAT"]))]
                except:
                    pass
            self.contact_collection = IFP_list
            return(IFP_list)

        ###################################################################
        #
        # Save IFP for a selected replicas of RAMD simulations
        #
        ###################################################################
        def IFP_save(self, file, subset=[]):
            """
            Parameters:
            optional - a list of IFP to be used to unify tables of IFP for all compounds
            Results:
            df1 - IFP database in pkl format
            """
            df1 = None
            if len(subset) == 0:
                r_subset = self.traj
                n_subset = self.names
                if len(n_subset) == 0:
                    logger.critical("IFP were not generated, please check if input data")
            else:
                r_subset = np.take(self.traj,subset)
                n_subset = np.take(self.names,subset)
                if n_subset.shape[0] == 0:
                    logger.critical("IFP were not generated, please check if input data")
            
            IFP_list = self.IFP_unify(subset)
            n_auxi = len(r_subset[0].rmsd_auxi)
            for i,(tr_c,n_replica) in enumerate(zip(r_subset,n_subset)):
                    logger.info(f"# Replica: {i}, {n_replica}")
                    tt = tr_c.df_properties
                    tt["Repl"] = np.repeat(n_replica,tr_c.df_properties.shape[0]) 
                    tt["Traj"] = np.repeat(str(i),tt.shape[0]) 
                    tt["RMSDl"] = tr_c.rmsd_lig
                    tt["RMSDp"] = tr_c.rmsd_prot
                    tt["RGyr"] = tr_c.rgyr_lig
                    tt["length"] = tr_c.length
                    tt["COM"] = tr_c.com_lig
                    
                    for k in range(0,n_auxi):
                         tt["Auxi_"+str(k)] = tr_c.rmsd_auxi[k]
                    df1 = pd.concat([df1, tt])
            if  len(r_subset) > 0:
                df1.to_pickle(file)
            else:
                logger.info("No IFP for equilibration simulations were generated")
            sys.stdout.flush()
            return(df1)
        
            
   #-----------------------------------------       
    class Ramd:
        """
            Functions:
                __init__(self)
                IFP_unify(self,subset = [])
                IFP_save(self,subset=[],file="IFP.pkl")
                bootstrapp(self,t)
                scan_ramd(self)
            Variables:
                repl_traj = []
                repl_names = [] 
                length = []
                replicas_distr = [] # distribution for each replica that is generated in the bootstrapping procedure
                replicas_distr_raw = [] # length of trajectories in one replica
                replicas = []  # tauRAMD for each replica
                replicas_SD = []
                contact_collection - a complete list of contact residues  
        """
        def __init__(self,PRJ_DIR,pdb,timestep):
            #---   # array of replica parameters- each is array of trajectories
            self.repl_traj = [] #  directories with RAMD simulations for all replicas 
            self.names = [] # name of  replicas 
            self.length = []   # in ns 
            # array of replica parameters
            self.replicas_distr = [] # distribution of dissociation times  for each replica that is generated in the bootstrapping procedure
            self.replicas_distr_raw = [] # dissociation times in one replica
            self.replicas = []  # tauRAMD for each replica
            self.replicas_SD = []
            self.tau = None  # computed res.time
            self.tau_SD = None
            self.PRJ_DIR = PRJ_DIR
            self.pdb = pdb
            self.timestep = timestep
            # the next set of parameters will be filled by the analysis_all_namd() function
            self.traj = []   # an array of Trj_Properties objects for wach trajectory in a replica
            self.contact_collection = []  # this is a container where we will accomulate all protein-ligand contacts

        ###################################################################
        #
        # FUNCTION to unify comlumns of a set of  IFP databases for generated from several trajectories  
        #
        ###################################################################
        def IFP_unify(self, subset=None):
            """
            Parameters:
            optional - a list of IFP to be used to unify tables of IFP for all compounds
            Results:
            """
            if subset is None:
                r_subset = self.traj
            else:
                r_subset = np.take(self.traj,subset)
            IFP_list = []

            for tr_replica in r_subset:
                for tr_c in tr_replica:
                    try:  # this is in the case when for some trajectory IFPs were not generated for some reasons
                        for c in tr_c.df_properties.columns.tolist():
                            if c  in IFP_list:
                                pass
                            else:
                                IFP_list.append(c)
                    except:
                        pass
            for tr_replica in r_subset:
                for tr_c in tr_replica:
                    try:
                        to_add = np.setdiff1d(IFP_list, np.asarray(tr_c.df_properties.columns.tolist()))
                        for k in to_add:
                            tr_c.df_properties[k] = np.zeros(tr_c.df_properties.shape[0],dtype=np.int8)
                        tr_c.df_properties = tr_c.df_properties[np.concatenate((["time"],tr_c.df_properties.columns[tr_c.df_properties.columns != "time"].tolist()))]
                        if "WAT" in tr_c.df_properties.columns.tolist():
                            tr_c.df_properties = tr_c.df_properties[np.concatenate((tr_c.df_properties.columns[tr_c.df_properties.columns != "WAT"].tolist(),["WAT"]))]
                    except:
                        pass
            self.contact_collection = IFP_list
            return IFP_list

        ###################################################################
        #
        # Save IFP for a selected replicas of RAMD simulations
        #
        ###################################################################
        def IFP_save(self, file, subset=None):
            """
            Parameters:
            optional - a list of IFP to be used to unify tables of IFP for all compounds
            Results:
            """
            df1 = None
            if subset is None: 
                r_subset = self.traj
                n_subset = self.names
            else:
                r_subset = np.take(self.traj,subset)
                n_subset = np.take(self.names,subset)
            IFP_list = self.IFP_unify(subset)
            logger.info(f"Will be saved: {self.names}")

            for tr_replica,tr_name in zip(r_subset,n_subset):
                for i,tr_c in enumerate(tr_replica):
                    try:
                        tt = tr_c.df_properties
                        tt["Repl"] = np.repeat(tr_name,tr_c.df_properties.shape[0]) 
                        tt["Traj"] = np.repeat(str(i),tt.shape[0]) 
                        tt["RMSDl"] = tr_c.rmsd_lig
                        tt["RMSDp"] = tr_c.rmsd_prot
                        tt["RGyr"] = tr_c.rgyr_lig
                        tt["length"] = tr_c.length
                        tt["COM"] = tr_c.com_lig
                        for k in range(0, len(tr_c.rmsd_auxi)):
                            rmsd_auxi = []
                            logger.info(len(rmsd_auxi), len(tr_c.rmsd_lig), len(tr_c.rmsd_lig))
                            tt["Auxi_"+str(k)] = tr_c.rmsd_auxi[k]
                        df1 = pd.concat([df1, tt])
                    except:
                        logger.critical(f"failed to save properties for the replica {tr_name} since the trajectiry was not analyzed")
                        pass
            df1.to_pickle(file)
            return df1

        ##############################################################
        #
        #    bootstrapping procedure for estimation of the residence time based on the set of RAMD dissociation times
        #
        ###############################################################
        def bootstrapp(self,t):
            """
            Parameters:
            t - list of RAMD dissociation times
            max_shuffle - number of bootstrapping iterations
            in each iteration alpha = 80% of the list is used
            Results:
            """
            max_shuffle = 500
            alpha = 0.8
            sub_set = int(alpha*len(t))        
            tau_bootstr = []
            if sub_set > 6:
                for i in range(1,max_shuffle):
                    np.random.shuffle(t)
                    t_b = t[:sub_set]
                    # select time when 50% of ligand dissocate
                    t_b_sorted_50 = (np.sort(t_b)[int(len(t_b)/2.0-0.5)]+np.sort(t_b)[int(len(t_b)/2)])/2.0
                    tau_bootstr.append(t_b_sorted_50)
            return tau_bootstr 


        ##############################################################
        #
        #   function that analyze all RAMD trajectories for a particular compound and compute its residence time  
        #
        ###############################################################
        def scan_ramd(self):
            """
            Parameters: re
            Results:
            """
            #--- read exp data-----------

            u = mda.Universe(self.PRJ_DIR+self.pdb)
            sd_max = 0

            for i,(rmd,repl) in enumerate(zip(self.names,self.repl_traj)): # loop over all replicas
                ramd_l = []
                traj_properties = []
                for j,t in enumerate(repl):  # loop over trajectories in each replica
                    try:
                        u.load_new(t)
                    except:
                        logger.error(f"Error while Reading a trajectory {t}")
                        pass
                    if len(u.trajectory) > 2:
                        ramd_l.append((self.timestep/1000.0)*len(u.trajectory)) # should be 2.0/1000.0 ?
                    traj_properties.append(Trj_Properties())
                self.length.append(ramd_l)
                if len(ramd_l) > 7:
                    #-------------- bootstrapping procedure
                    distr = self.bootstrapp(ramd_l)
                    self.replicas_distr.append(distr)
                    self.replicas.append(np.mean(distr))
                    self.replicas_SD.append(np.std(distr))
                    self.replicas_distr_raw.append(ramd_l)
                    sd_max = max(sd_max, np.nanmax(np.std(distr)))
                    logger.info(f"{rmd} tau = {np.mean(distr):.3f} +- {np.std(distr):.3f}")
                    logger.debug(f"List of tau from trajs:\n{ramd_l}")
                else:
                    logger.warning(f"RAMD trajectory set for {rmd} is too small ({len(ramd_l)} traj), tau will not be computed for this replica")
                    self.replicas.append(None)
                    self.replicas_SD.append(None)
                    self.replicas_distr.append([])
                    self.replicas_distr_raw.append([])

                self.traj.append(traj_properties)
            #-- compute tauRAMD residence time as an average over all replicas (skip empty replicas)
            if len(self.replicas) == 0:
                logger.warning(f"RAMD trajectories were not found in {self.names}")
            else:
                # we will estimate final tau RAMD from all replicas as an average (omitting Nans, ie. incolplete simulations)
                non_empty  = np.asarray(self.replicas)[np.isnan(np.asarray(self.replicas).astype(float)) == False]
                if len(non_empty)>0:
                    self.tau =  np.nanmean(non_empty)
                    self.tau_SD = max(np.nanstd(non_empty),sd_max)
            return

        ########################################
        #
        #     PLOT RAMD analysis for a particular ligand
        #
        ########################################

        def Plot_RAMD(self, tau_lims=(0,0)):
            """
            Parameters:
            lims -  x-limits
            Returns:
            plot
            """
            lims = tau_lims
            color = ['r','b','forestgreen','orange','lime','m','teal','c','yellow','goldenrod','olive','tomato','salmon','seagreen']
            meanpointprops = dict(linestyle='--', linewidth=2.5, color='firebrick')
            fig = plt.figure(figsize=(18, 2))
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1],wspace=0.1)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            time_max = 0
            for r in self.replicas_distr:
                if len(r)>0: time_max = max(time_max,np.max(r))
            x = np.linspace(start=lims[0],stop=lims[1],num=100)
            for i,(d, b) in enumerate(zip(self.replicas_distr,self.length)):
                if (self.replicas_distr[i] and len(d) > 2):
                    sns.distplot(d, kde=True, hist = True, norm_hist = True, bins = 3, color =color[i],label=self.names[i], ax = ax1);   
                    ymin, ymax = ax1.get_ylim()

            replicas_distr = [x for x in self.replicas_distr if x]  # here we eliminate all empty lists

            if tau_lims == (0,0):
                lims =(0,1.2*np.max(np.asarray(replicas_distr).flatten()))

            ax1.set_xlim(lims)
            try:
                ax2.set_ylim(lims)
                ax2.boxplot(self.replicas_distr,labels=self.names,showmeans=True, meanline=True,meanprops=meanpointprops) 
                ax2.plot([0,len(self.names)+1], [self.tau,self.tau], 'k-', lw=3,alpha=0.3)
                ax2.plot([0,len(self.names)+1], [self.tau-self.tau_SD,self.tau-self.tau_SD], 'k--', lw=3,alpha=0.3)
                ax2.plot([0,len(self.names)+1], [self.tau+self.tau_SD,self.tau+self.tau_SD], 'k--', lw=3,alpha=0.3)
            except:
                logger.error("Error in the boxplot: there is probably no data for one of the replicas")
                pass
            if self.tau and self.tau_SD:
                logger.info(f"Average over replicas: tauRAMD =  {np.round(self.tau,3)} +- {np.round(self.tau_SD,3)}")
                gauss = np.exp(-np.power((x - self.tau)/self.tau_SD, 2.)/2.)
                ax1.plot(x,ymax*gauss/max(gauss), 'k-', label='total', linewidth=3,alpha=0.3)
            ax1.legend(ncol=2 if i > 4 else 1)
            ax2.set_ylabel('tau /ns', fontsize=12)
            ax1.set_xlabel('tau /ns', fontsize=12)
            ax2.set_xlabel('replica', fontsize=12)
            try:
                ax2.set_ylim(0, 1.2*np.max(np.asarray(replicas_distr).flatten()))
            except:
                pass
            plt.show()
            return

       
    ##############################################################
    #
    #   Functions generationg objects of the sub-classes
    #
    ###############################################################

    def createLigand(self,PRJ_DIR,ligand_pdb,ligand_mol2):
        return Ligand(PRJ_DIR,ligand_pdb,ligand_mol2)
#       return (trajectories.Ligand(PRJ_DIR,ligand_pdb,ligand_mol2))

    def createNamd(self):
        return trajectories.Namd()

    def createRamd(self,PRJ_DIR,pdb,timestep):
        return trajectories.Ramd(PRJ_DIR,pdb,timestep)

    # ##############################################################
    # #
    # #  function that analyze a membrane density along z axis in a trajectory 
    # #
    # ###############################################################

    # def mambrane_traj(self,traj,start_analysis,step_analysis):
    #     """
    #     Parameters:
    #     step_analysis - step to be used to scan over the trajectory
    #     start_analysis - starting snapshot for analysis; if
    #     start_analysis < 0 - count from the end of the trajectory
    #     if(0 < start_analysis < 1) -  a fraction of the trajectory = start_analysis will be skipped
    #     traj - location and name of the trajectory to be analyzed   
        
    #     Results:
    #     u_length - total number of fraimes in the trajectory
    #     """
    #     sel_ligands = self.ligand.ligands_names[0]
    #     ref = mda.Universe(self.PRJ_DIR+self.pdb)
    #     Rgr0 = ref.select_atoms("protein").radius_of_gyration() 
    #     all_atoms = ref.select_atoms("not type H")
    #     zmin = 100

                                       

    #     return

    ##############################################################
    #
    #  function that analyze a trajectory
    #  RMSD of protein and ligand, Radius of Gyration, and IFP table are computed for every nth frame
    #  input parameters - stride and the first snapshot for analysis 
    #
    ###############################################################

    def analysis_traj(self,traj,start_analysis,step_analysis,WB_analysis,RE,Lipids,auxi_selection=None, reference = "ref"):
        """
        Parameters:
        step_analysis - step to be used to scan over the trajectory
        start_analysis - starting snapshot for analysis; if start_analysis < 0, count from the end of the trajectory
        if(0 < start_analysis < 1) -  a fraction of the trajectory = start_analysis will be skipped
        WB_analysis - True if water briges has to be traced, default- False
        RE
        Lipids
        auxi_selection
        traj - location and name of the trajectory to be analyzed  
        reference - 
        
        Results:
        rmsd_prot,rmsd_lig - RMSD of the protein and ligand
        auxi_rmsd 
        rgyr_prot, rgyr_lig - radius of Gyration of the protein and ligand
        df_prop_complete - Pandas table with a complete set of IFP
        u_length - total number of fraimes in the trajectory
        """
        auxi_selection = [] if auxi_selection is None else auxi_selection
        sel_ligands = self.ligand.ligands_names[0]
  #      sel_l = "resname "+sel_ligands
        rmsd_prot = []
        rmsd_lig = []
        rgyr_prot = []
        rgyr_lig = []
        com_lig = []
        df_prop = None
        df_HB = None 
        df_WB = None
        u = mda.Universe(self.PRJ_DIR+self.pdb)
        u.load_new(traj)
        u_length = len(u.trajectory)
        u_size = os.path.getsize(traj)/(1024.*1024.)
        logger.info(f"   total number of frames = {u_length}; file size {u_size:.1f} M")

        if reference=="ref":
            ref = mda.Universe(self.PRJ_DIR+self.pdb)
#            Rgr0 = ref.select_atoms("protein").radius_of_gyration()
        else:
            u.trajectory[0]
            ref = u
        Rgr0 = ref.select_atoms("protein").radius_of_gyration()

        stop = len(u.trajectory)
        if start_analysis < 0:
            if -start_analysis >= len(u.trajectory):
                start = 0  # or better 1, as was originally?
            else:
                start = stop + start_analysis
        #elif start_analysis < 1:  # doesn't make sense, as it can only be 0 here, and so start will  be 0...
        #    start = len(u.trajectory)*start_analysis
        else:
            start = start_analysis
        if start > stop:
            start = stop -1
        step = max(1,step_analysis)
        if len(Lipids) > 0:
            lipid_line = ""
            for l in Lipids:
                lipid_line = lipid_line+" "+l
            selection = f"(resname {sel_ligands}) or {self.sub_system} or (resname {lipid_line})"
        else:
            selection = f"{self.sub_system} or (resname  {sel_ligands})"

        selection_rmsd = ["protein and (not type H)","resname "+sel_ligands+" and (not type H)"]
        auxi_rmsd = []
        for auxi in auxi_selection:
            if len(auxi) > 2:  # this is a check if a single string or a list of strings is given
                selection = selection + " or "+auxi
                auxi_rmsd.append([])
                selection_rmsd.append(auxi)
            else:
                selection = selection + " or "+auxi_selection
                auxi_rmsd.append([])
                selection_rmsd.append(auxi)
                break


        system_reduced = u.select_atoms(selection)
        logger.info(f"selected sub-system: {selection}")
        # try:
        u_mem = mda.Merge(system_reduced).load_new(AnalysisFromFunction(lambda ag: ag.positions.copy(), system_reduced).run(start=start, stop=stop, step=step).results, format=MemoryReader)
        #past_the_rest = False
        # except:
        #     logger.warning("Failed to read this trajectory")
        #     past_the_rest = True
        #     pass
        protein = u_mem.select_atoms("protein")
        ligand = u_mem.select_atoms(f"resname {sel_ligands}")
        #if not past_the_rest:
        logger.info(f"frames to be analyzed: {len(u_mem.trajectory)}")
        for frame, traj_frame in enumerate(u_mem.trajectory):
            #assert frame == traj_frame.frame
            _ = u_mem.trajectory[frame]
            u_mem.dimensions = u.dimensions
            _ = pbc(u_mem, Rgr0)
            rgyr_prot.append(protein.radius_of_gyration())
            rgyr_lig.append(ligand.radius_of_gyration())
            rmsd = superimpose_traj(ref, u_mem, selection_rmsd)
            if rmsd[0] > 10.0:
                logger.info(f"for the frame {traj_frame.frame} protein RMSD is very large: {rmsd[0]:.3f}")
            if rmsd[1] > 10.0:
                logger.info(f"for the frame {traj_frame.frame} ligand RMSD is very large: {rmsd[1]:.3f}")
            com_lig.append(ligand.center_of_mass())
            rmsd_prot.append(rmsd[0])
            rmsd_lig.append(rmsd[1])
            for j, selected_auxi_rmsd in enumerate(auxi_rmsd):
                selected_auxi_rmsd.append(rmsd[j+2])
            if frame % 1000 == 0:
                logger.info(f"RMSD protein: {rmsd[0]:.3f}")
                logger.info(f"RMSD {self.ligand.ligands_names[0]}:  {rmsd[1]:.3f}")
        df_prop, df_HB, df_WB = IFP(u_mem, sel_ligands, self.ligand.property_list, WB_analysis, RE, Lipids)

        return u_length, start, rmsd_prot, rmsd_lig, auxi_rmsd, rgyr_prot, rgyr_lig, com_lig, df_prop, df_HB, df_WB



    ##############################################################
    #
    #  function that analyze all NAMD trajectories for a particular compound
    #  RMSD of protein and ligand, Radius of Gyration, and IFP table are computed for every nth frame
    #  input parameters - stride and the first snapshot for analysis 
    #
    ###############################################################
    def analysis_all_namd(
            self,
            WB_analysis = True,
            step_analysis=1,
            start_analysis=0,
            RE=True,
            Lipids=None,
            auxi_selection=None):
        """
        Parameters:
        step_analysis - step to be used to scan over the trajectory
        start_analysis - starting snapshot for analysis; if start_analysis < 0 - cound from the end of the trajectory
        WB_analysis - True if water briges has to be traced, default- False
        Results:
        """
        Lipids = [] if Lipids is None else Lipids
        auxi_selection = [] if auxi_selection is None else auxi_selection
#        ligands_name,property_list = ligand_analysis(dir_ligand+"/"+self.ligand.ligand_pdb)
        sel_ligands = self.ligand.ligands_names[0]
        sel_l = "resname "+self.ligand.ligands_names[0]
#        mol,ligand_2D = read_ligands(dir_ligand+"/"+ligand_pdb)

        ref = mda.Universe(self.PRJ_DIR+self.pdb)
        Rgr0 = ref.select_atoms("protein").radius_of_gyration()
        for j, (nmd, repl) in enumerate(zip(self.namd.names, self.namd.repl_traj)):
            logger.info(f"Replica: {repl}")
            step = max(step_analysis, 1)
            length, _start, rmsd_prot,rmsd_lig, rmsd_auxi,rgyr_prot, rgyr_lig, com_lig, df_prop, df_HB, df_WB = self.analysis_traj(repl,start_analysis,step,WB_analysis, RE,Lipids,auxi_selection,reference ="ref")
            df_prop_complete = table_combine(df_HB, df_WB, df_prop, sel_ligands, self.namd.contact_collection)
            self.namd.length.append((self.timestep/1000)*length)
            Plot_traj(rmsd_prot, rmsd_lig, rmsd_auxi, rgyr_prot, rgyr_lig, nmd, out_name=None)
            for contact_name in df_prop.columns.tolist():
                if contact_name not in self.namd.contact_collection:
                    self.namd.contact_collection.append(contact_name)
#            if(j > 0): print("....",len(self.namd.traj),j-1,self.namd.traj[j-1].rmsd_auxi[0][:3])
            self.namd.traj.append(Trj_Properties())
#            if(j > 0): print("....",len(self.namd.traj),j-1,self.namd.traj[j-1].rmsd_auxi[0][:3])
            self.namd.traj[j].step = step
            self.namd.traj[j].start = start_analysis
            self.namd.traj[j].stop = length
            self.namd.traj[j].length = length
            self.namd.traj[j].df_properties=df_prop_complete
            self.namd.traj[j].rmsd_prot = rmsd_prot
            self.namd.traj[j].rmsd_lig = rmsd_lig
            self.namd.traj[j].rgyr_prot = rgyr_prot
            self.namd.traj[j].rgyr_lig = rgyr_lig
            self.namd.traj[j].com_lig = com_lig
            self.namd.traj[j].rmsd_auxi = rmsd_auxi
 #           print(j,rmsd_auxi[0][:3],self.namd.traj[j].rmsd_auxi[0][:3])
 #           if(j > 0): print("....",j-1,self.namd.traj[j-1].rmsd_auxi[0][:3])
            #Plot_IFP(df_prop_complete,out_name="") # "namd-"+str(j)+".png")
        return
    
    ##############################################################
    #
    #  function that analyze all NAMD trajectories for a particular compound 
    #
    ###############################################################
    def analysis_all_ramd(self,WB_analysis = True,step_analysis=1,start_analysis=0,repl_list= [],RE = True,Lipids = [],auxi_selection = []):
        """
        Parameters:
        step_analysis - step to be used to scan over the trajectory
        start_analysis - starting snapshot for analysis; id start_analysis < 0 - cound from the end of the trajectory
        WB_analysis - run analysis of water bridges between protein and ligand; quite time-consuming
        repl_list - a list of replica numbers (indexes from a complete replica list) to be analyzed 
        Results:
        """

        sel_ligands = self.ligand.ligands_names[0]
        sel_l = "resname "+self.ligand.ligands_names[0]

        ref = mda.Universe(self.PRJ_DIR+self.pdb)
        Rgr0 = ref.select_atoms("protein").radius_of_gyration()  

        if len(repl_list) > 0 : repl_scan =  repl_list
        else:    repl_scan = range(0,len(self.ramd.repl_traj))
        for j1 in repl_scan:
            rmd = self.ramd.names[j1]
            repl = self.ramd.repl_traj[j1]
            logger.info(f"Replica {j1}: {rmd}")
            repl_df_properties = []
            repl_rmsd_prot = []
            repl_rmsd_lig = []
            repl_Rgr_prot = []
            repl_Rgr_lig = []
            if len(self.ramd.traj) < 1:
                logger.error("RAMD trajectories must be loaded first using the function ramd.scan_ramd() (trajectory class function)")
                sys.exit()
 
            for j2, repli in enumerate(repl):
                logger.info(f"traj {j2}, file {repli}")
                step = max(step_analysis, 1)
                #try:
                length,start,rmsd_prot,rmsd_lig, rmsd_auxi,rgyr_prot, rgyr_lig,com_lig,df_prop,df_HB,df_WB = self.analysis_traj(repli,start_analysis,step,WB_analysis,RE,Lipids,auxi_selection)
                df_prop_complete = table_combine(df_HB,df_WB,df_prop,sel_ligands,self.ramd.contact_collection)
                self.namd.length.append((self.timestep/1000)*length)
                Plot_traj(rmsd_prot, rmsd_lig, rmsd_auxi, rgyr_prot, rgyr_lig, rmd, out_name=None)

                for contact_name in df_prop.columns.tolist():
                    if contact_name not in self.ramd.contact_collection:
                        self.namd.contact_collection.append(contact_name)
                self.ramd.traj[j1][j2].step = step
                self.ramd.traj[j1][j2].start = start
                self.ramd.traj[j1][j2].stop = length
                self.ramd.traj[j1][j2].length = length
            
                self.ramd.traj[j1][j2].df_properties = df_prop_complete
                self.ramd.traj[j1][j2].rmsd_prot = rmsd_prot
                self.ramd.traj[j1][j2].rmsd_lig = rmsd_lig
                self.ramd.traj[j1][j2].rgyr_prot = rgyr_prot
                self.ramd.traj[j1][j2].rgyr_lig = rgyr_lig
                self.ramd.traj[j1][j2].com_lig = com_lig
                self.ramd.traj[j1][j2].rmsd_auxi = rmsd_auxi
                # Plot_IFP(df_prop_complete,out_name="") #"ramd-"+str(j2)+".png")
                #except:
                #    logger.error("IFP either were not generated or could not be stored in the traj object!")
                #    raise exc

        return


class  Ligand:
    """
    Functions:
        _int__(self,PRJ_DIR,ligand_pdb,ligand_mol2="moe.mol2")
    Variable:
        ligands_names
        property_list
        mol
        ligand_2D
        radius
    """
    def __init__(self, PRJ_DIR, ligand_pdb, ligand_mol2="moe.mol2"):
        self.ligands_names = []
        self.property_list = {}
        self.mol = None
        self.ligand_2D = None
        resnames = []
        list_labelsF = []
        DO_PDB = False
        self.properties_list = []
        if ligand_pdb or ligand_mol2:                   
            if os.path.isfile(PRJ_DIR+"/"+ligand_pdb) or  os.path.isfile(PRJ_DIR+"/"+ligand_mol2):
                try:
                    mol, list_labels, resnames = self.ligand_Mol2(PRJ_DIR+"/"+ligand_mol2)
                except:
                    logger.info("Mol2 is absent, PDB file will be used for ligand structure analysis")
                    DO_PDB = True                        
            elif os.path.isfile(PRJ_DIR+ligand_pdb):
                    DO_PDB = True
            else:
                logger.error(f"nether ligand PDB nor Mol2 were found in {PRJ_DIR} expected file names: {ligand_pdb} {ligand_mol2}")
                return
            if DO_PDB:
                logger.warning("mol2 file was not found or contains errors")
                logger.warning("pdb file will be used instead\n Aromatic atoms can not be recognized")
                mol,list_labels, resnames = self.ligand_PDB(PRJ_DIR+"/"+ligand_pdb)
            if mol is None:
                if DO_PDB:
                    logger.warning("RDKit cannot read PDB structure, atom naming will be corrected")
                    rename_H(PRJ_DIR+"/"+ligand_pdb,PRJ_DIR+"/Corrected_Ligand.pdb")
                    mol,list_labels, resnames = self.ligand_PDB(PRJ_DIR+"/Corrected_Ligand.pdb")
                    properties_list,ligand_2D = self.ligand_properties(mol, list_labels)
                else:
                    properties_list = []
                    logger.error(f"RDKit cannot read MOL2 structure {mol}")
            else:
                properties_list,ligand_2D = self.ligand_properties(mol,list_labels)
            # add fluorine as hydrophobic atoms (absent in RDkit)
            if len(properties_list) > 0:
                if DO_PDB:
                    list_labelsF  = self.ligand_PDB_F(PRJ_DIR+"/"+ligand_pdb)
                    list_labelsPO3 = []
                    list_labelsP = []
                else:
                    list_labelsF, list_labelsPO3, list_labelsP  = self.ligand_Mol2_F_PO3(PRJ_DIR+"/"+ligand_mol2)

                if len(list_labelsF) > 0:
                    if  'Hydrophobe' in properties_list:  
                        new_properties_list_H = properties_list['Hydrophobe']
                        for at in list_labelsF:  new_properties_list_H.append(at)
                    else:  properties_list.update({'Hydrophobe': list_labelsF})
                    properties_list['Hydrophobe'] = new_properties_list_H
                    logger.info(f"Fluorine atoms are found (will be considered as Hydrophobe): {list_labelsF}")

                if len(list_labelsP) > 0:
                    if  'NegIonizable' in properties_list: 
                        new_properties_list_H = properties_list['NegIonizable']
                        for at in list_labelsP: new_properties_list_H.append(at)
                        properties_list['NegIonizable'] = new_properties_list_H
                    else:  properties_list.update({'NegIonizable': list_labelsP})
                    logger.info(f"PO3 group is found (P atoms will be considered as NegIonizable): {list_labelsP}")

                if len(list_labelsPO3) > 0:
                    if  'Acceptor' in properties_list: 
                        new_properties_list_H = properties_list['Acceptor']
                        for at in list_labelsPO3: new_properties_list_H.append(at)
                        properties_list['Acceptor'] = new_properties_list_H
                    else:  properties_list.update({'Acceptor': list_labelsPO3})
                    logger.info(f"PO3 group is found (O atoms will be considered as Acceptors): {list_labelsPO3}")


                logger.info("...............Ligand properties:................")
                self.properties_list = properties_list
                for k in properties_list:
                    logger.info(f"{k} {properties_list[k]}")
            else:
                logger.error("RDKit cannot generate ligand property list")

            try:
                self.ligand_2D=ligand_2D
                self.mol = mol
                self.property_list=properties_list
            except:
                if DO_PDB:
                    logger.error("RDKit cannot read file - some errors found in the ligand structure")
                    sys.exit()
                else:
                    try:
                        mol,list_labels,resnames = self.ligand_PDB(PRJ_DIR+"/"+ligand_pdb)
                    except:
                        logger.error("RDKit cannot read file- some errors found in the ligand structure")
                        sys.exit()
            self.ligands_names = np.unique(resnames)
            logger.info(f"The following residue names will be used to identify ligand in the PDB file: {self.ligands_names}")
        else:
            logger.warning("ligand PDB and Mol2 are not defined")
        return
        ########################################
        #
        #     get ligand chemical properties from PDB and Mol2 files
        #
        ########################################

    def ligand_Mol2(self, ligand_mol2):
        """
        Parameters:
        ligand_mol2 - ligand structure file  in the Mol2 format (not all mol2 format work, but generated by MOE does)
        Results:
        mol - RDKit molecular object
        list_labels - list of atom names
        resnames - list of residue names (for all atoms)
        """
        with open(ligand_mol2, "r") as ff:
            lines = ff.readlines()
        list_labels = []
        resnames = []
        start = False
        for line in lines:
            items = line.split()
            if line.find("<TRIPOS>ATOM") >= 0:
                start = True
            elif line.find("<TRIPOS>BOND") >= 0:
                start = False
            else:
                if start:
                    list_labels.append(items[1])
                    resnames.append(items[7])
        mol = Chem.rdmolfiles.MolFromMol2File(ligand_mol2, removeHs=False)
        if len(list_labels) == 0:
            raise ValueError(f"No atoms found in the mol2 file {ligand_mol2}")
        logger.info(f"Atoms found in the MOL2 file:\n{list_labels}")
        return mol, list_labels, resnames

        ########################################
        #
        #     get ligand chemical properties from PDB file only
        #
        ########################################
    def ligand_PDB(self,ligand_pdb):
        """
        Parameters:
        ligand_pdb - ligand structure file  in the PDB format
        Results:
        mol - RDKit molecular object
        list_labels - list of atom names
        resnames - list of residue names (for all atoms)
        """
        ff=open(ligand_pdb,"r")
        lines = ff.readlines()
        ff.close()
        list_labels = []
        resnames = []

        for line in lines:
            if line.split()[0] == 'ATOM':
                list_labels.append(line.split()[2])
                resnames.append(line.split()[3])
        
        mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False)
        return mol, list_labels, resnames
        ########################################
        #
        #     get ligand chemical properties for Fluorine from PDB file only
        #
        ########################################
    def ligand_PDB_F(self,ligand_pdb):
        """
        Parameters:
        ligand_pdb - ligand structure file  in the PDB format
        Results:
        list_labels - list of  names for F atoms found
        """
        ff=open(ligand_pdb,"r")
        lines = ff.readlines()
        ff.close()
        list_labels = []

        for line in lines:
            if len(line.split()) > 5:
                if (line.split()[0] == 'ATOM' or line.split()[0] == 'HETATM'):
                    if (line.split()[2][0] == "F"):
                        list_labels.append(line.split()[2])
        return list_labels
        ########################################
        #
        #     get ligand chemical properties for Fluorine from PDB file only
        #
        ########################################
    def ligand_Mol2_F_PO3(self, ligand_mol2):
        """
        Parameters:
        ligand_mol2 - ligand structure file  in the MOL2 format
        Results:
        list_labels_P - list of  names for oxygen atoms bound to P
        list_labels - list of names for F atoms found
        """
        ff=open(ligand_mol2,"r")
        lines = ff.readlines()
        ff.close()
        list_labels_O = []
        list_labels_P = []
        list_labels_F = []
        list_atoms = []
        list_P = []
        resnames = []
        start = 0
        for line in lines:
            key = line.split()
            if line.find("<TRIPOS>ATOM") >= 0: start = 1
            elif line.find("<TRIPOS>BOND") >= 0: start = 2
            elif line.find("<TRIPOS>SUBSTRUCTURE") >= 0: break
            else:
                if start == 1: 
                    list_atoms.append(key[1]) 
                    if(key[1][0]) == "P":
                        list_P.append(key[0])
                        list_labels_P.append(key[1])
                    if(key[1][0]) == "F":
                        list_labels_F.append(key[1])
                if start == 2: 
                    for P in list_P:
                        if int(key[1]) == int(P):
                            if list_atoms[int(key[2])-1][0] == 'O':
                                if list_atoms[int(key[2])-1] not in list_labels_O:
                                    list_labels_O.append(list_atoms[int(key[2])-1])
                        if int(key[2]) == int(P):
                            if list_atoms[int(key[1])-1][0] == 'O':
                                if list_atoms[int(key[1])-1] not in list_labels_O:
                                    list_labels_O.append(list_atoms[int(key[1])-1])
        
        return list_labels_F, list_labels_O, list_labels_P


        ########################################
        #
        #     get ligand chemical properties from PDB file only
        #
        ########################################
    def ligand_properties(self,mol,list_labels):
        """
        Parameters:
        mol - RDKit molecular object
        list_labels - list of atom names
        Results:
        properties_list - a dictionary containing types of chemical properties and corresponding atoms 
        """
        ligand_2D = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        feats = factory.GetFeaturesForMol(mol)
        properties_list = {}
        for f in feats:
            prop = f.GetFamily()  #  get property name
            at_indx  = list(f.GetAtomIds())  # get atom index
            if prop not in properties_list.keys():
                properties_list[prop]=[]
            if(len(at_indx) > 0):
                for l in at_indx: properties_list[prop].append(list_labels[l])
            else: properties_list[prop].append(list_labels[at_indx[0]])
        return(properties_list,ligand_2D)
        
        ########################################
        #
        #     correct PDB file (name and position of hydrogen atoms) to make it readable by RDKit
        #
        ########################################
    def rename_H(self,ligand_H_name,ligand_H_name_new = ""):
        """
        rename hydrogen atoms in a ligand file generated by openbabel ( openbabel generats all hydrogen as H)
        remove connectivity lines
        adjust position of hydrogen atom names so that H always occupies 14th position (requiered by Rdkit)
        Parameters:
        ligand pdb file
        Returns:
         name of the ligand 
         rewrite ligand structure
        """
        ff=open(ligand_H_name,"r")
        lines = ff.readlines()
        ff.close()
        lig = []
        hi = 1
        for line in lines:
            key = line.split()
            if len(line) > 20:
                if (line[12:16].strip()[0] == "H") or (line[12:13] != " "):
                    s = list(line)
                    if line[12:16].strip() == "H": new_name = "H"+str(hi)
                    else: new_name =  line[12:16].strip()
                    s[12:17] = str(" %-4s" %(new_name))
                    hi += 1
                    line = "".join(s)
                if line.split()[0] == "ATOM" or line.split()[0] == "HETATM":   # we will skip connectivity lines
                    lig.append(line.replace("HETATM","ATOM  "))
                    resname = line[16:20].strip()

        if ligand_H_name_new == "": ligand_H_name_new = ligand_H_name

        if len(lig) > 0:
            with open(ligand_H_name_new, "w") as ff:
                for p in lig:
                    ff.write(p)
        return
        ##########################################################################################################

#######################################################################
#
#     FUNCTION FOR SUPERIMPOSISION of THE TRAJECTRY FRAMES TO A REFERENCE STRUCTURE
#
#######################################################################

def superimpose_traj(ref, u, sel_list=None):
    """
    Parameters:
    u - trajectory - universe object
    ref - reference structure -universe object
    sel_list - a list of atom groups to compute RMSD
    for example - "protein" or "resname IXO"
    Results:
    """
    ur = ref
    sel_list = [] if sel_list is None else sel_list
    ref_CA = ur.select_atoms("name CA")
    ref0 = ref_CA.positions - ref_CA.center_of_mass() 

    u_CA = u.select_atoms("name CA")
    u.atoms.translate(-u_CA.center_of_mass())
    u0 =  u_CA.positions - u_CA.center_of_mass()
    R, _rmsd = align.rotation_matrix(u0, ref0)  # compute rotation matrix
    u.atoms.rotate(R)
    u.atoms.translate(ref_CA.center_of_mass()) # translate back to the old center of mass position

    rmsd_list = []
    for s in sel_list:
        rmsd_list.append(rms.rmsd(u.select_atoms(s).positions, ur.select_atoms(s).positions))

    return rmsd_list

#######################################################################
#
#     FUNCTION FOR PUTTING SYSTEM BACK INTO A PB BOX
#
#######################################################################

def pbc(u,Rgr0):
    """
    Parameters:
    as a check if the transformation is correct we compare radius of gyration with the reference one
    u - trajectory - universe object
    ref - reference structure -universe object
    Results:
    """
    u_CA = u.select_atoms("name CA")
    sel_p = "protein"

    # getting all system elements back to the box; it is important to repeat this twice in the case when protein is splitted into two parts
    u.atoms.translate(-u_CA.center_of_mass()+0.5*u.dimensions[0:3])
    u.atoms.pack_into_box(box=u.dimensions) 
    u.atoms.translate(-u_CA.center_of_mass()+0.5*u.dimensions[0:3])
    u.atoms.pack_into_box(box=u.dimensions) 
    Rgr = u.select_atoms(sel_p).radius_of_gyration()      
    if Rgr > Rgr0*1.1:
#        print("Radius of gyration is too large: %s  of that in the first frame; Try to pack system back into a box once more " %(Rgr/Rgr0)) 
        u.atoms.translate(-u_CA.center_of_mass()+0.5*u.dimensions[0:3])
        u.atoms.pack_into_box(box=u.dimensions) 
        Rgr = u.select_atoms(sel_p).radius_of_gyration()  
#        print("Radius of gyration is now: %s  of the first frame" %(Rgr/Rgr0)) 
    if Rgr > Rgr0*1.1:
        logger.info(f"failed to pack the system back into a box radius of gyration is too large: {Rgr/Rgr0:.3f} of that in the first frame")
    return Rgr


#######################################################################
#
#     FUNCTION FOR READING LIGAND structure in PDB format USING RDKit
#
#######################################################################

def read_ligands(ligand_pdb):
    """
    Parameters:
    ligand pdb file
    Returns:
    Rkit molecular object for the ligand and ligand image
    """
    tmp_name = ligand_pdb[:-4]+"-tmp.pdb"
    rename_H(ligand_pdb,tmp_name)   # correct position of names of hydrogen atoms
    mol = Chem.MolFromPDBFile(tmp_name, removeHs=False)
    t1 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return mol, t1
###############################################################################
#
#   FUNCTION that renames hydrogen atoms in a ligand file generated by openbabel (openbabel generats all hydrogen as H)
#   remove connectivity lines
#   adjust position of hydrogen atom names so that H always occupies 14th position (requiered by Rdkit)
#
###############################################################################
def rename_H(ligand_H_name,ligand_H_name_new = ""):
    """
    Parameters:
    ligand pdb file
    Returns:
     name of the ligand 
     rewrite ligand structure
    """
    ff=open(ligand_H_name,"r")
    lines = ff.readlines()
    ff.close()
    lig = []
    hi = 1
    for line in lines:
        key = line.split()
        if len(line) > 20:
            if ((line[12:16].strip()[0] == "H") or (line[12:13] != " ")):
                s = list(line)
                if line[12:16].strip() == "H":
                    new_name = "H"+str(hi)
                else:
                    new_name =  line[12:16].strip()
                s[12:17] = str(" %-4s" %(new_name))
                hi += 1
                line = "".join(s)
            if line.split()[0] == "ATOM" or line.split()[0] == "HETATM":   # we will skip connectivity lines
                lig.append(line.replace("HETATM", "ATOM  "))
                resname = line[16:20].strip()

    if ligand_H_name_new == "":
        ligand_H_name_new = ligand_H_name
    if len(lig) > 0:
        ff=open(ligand_H_name_new,"w")
        for p in lig:  ff.write(p)
        ff.close()
    return resname

#######################################################################
#
#      FUNCTION FOR READING LIGAND sctucture in mol2 format USING RDKit
#
#######################################################################

def read_ligands_mol2(ligand_mol2):
    """
    Parameters:
    ligand mol2 file - IMPORTANT: mol2 file created by antechamber does not work! created by MOE works
    Returns:
    Rkit molecular object for the ligand and ligand image
    """
    mol = Chem.rdmolfiles.MolFromMol2File(ligand_mol2,removeHs=False)
    try:
        sm = Chem.MolToSmiles(mol)
        t1 =Chem.MolFromSmiles(sm)
        return mol, t1
    except:
        logger.error(f"ERROR in mol2 file {ligand_mol2}")
        return mol, None

#######################################################################
#
#      FUNCTION FOR READING atom labels from mol2 file
#
#######################################################################
def read_ligands_mol2_AtomLabels(ligand_mol2):
    """
    Parameters:
    Returns:
    """
    radius = 0
    list_labels = []
    ff=open(ligand_mol2,"r")
    lines = ff.readlines()
    ff.close()
    start = False
    list_resname = []
    list_pos = []
    for line in lines:
        key = line.split()
        if line.find("<TRIPOS>ATOM") >= 0:
            start = True
        elif line.find("<TRIPOS>BOND") >= 0:
            start = False
        else:
            if start:
                list_labels.append(key[1])
                list_resname.append(key[7])
                list_pos.append([float(key[2]), float(key[3]), float(key[4])])
    center = np.mean(np.asarray(list_pos), axis=1)[0]
    radius=max(np.sum(np.abs(np.asarray(list_pos)-center)**2,axis=1)**0.5)
    return list_labels, list_resname, radius
#######################################################################
#
#      FUNCTION FOR READING atom labels from pdb file
#
#######################################################################
def read_ligands_pdb_AtomLabels(ligand_pdb):
    """
    Parameters:
    Returns:
    """
    radius = 0
    list_labels = []
    list_resname = []
    list_pos = []
    ff=open(ligand_pdb,"r")
    lines = ff.readlines()
    ff.close()
    for line in lines:
        key = line.split()
        if key[0] == 'ATOM': 
            list_labels.append(key[2]) 
            list_resname.append(key[3])
            try:
                list_pos.append([float(key[5]),float(key[6]),float(key[7])])
            except:
                logger.warning(f"Format error in {ligand_pdb}. Ligand pdb file should not contain chain infromation")
    center = np.mean(np.asarray(list_pos),axis=1)[0]
    radius=max(np.sum(np.abs(np.asarray(list_pos)-center)**2,axis=1)**0.5)
    return (list_labels,list_resname,radius)


########################################
#
#     get ligand chemical properties
#
########################################
def  ligand_properties(ligand_pdb, ligand_mol2):
    """
    ligand_pdb - ligand structure file  in the PDB format
    ligand_mol2 - ligand structure file  in the Mol2 format (not all mol2 format work, but generated by MOE does)
    """
    with open(ligand_pdb,"r") as ff:
        list_labels = [
            l.split()[2] for l in ff.readlines() if line.split()[0] in ['ATOM', "HETATM"]
            ]

    if not os.path.exists(ligand_mol2):
        logger.warning("MOL2 does not exist; ligand properties will be derived from the PDB file i.e. aromatic properties will be missed")

    mol = Chem.rdmolfiles.MolFromMol2File(
        ligand_mol2 if os.path.exists(ligand_mol2) else ligand_pdb,
        removeHs=False)

    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)

    properties_list = {}
    for f in feats:
        prop = f.GetFamily()  #  get property name
        at_indx  = list(f.GetAtomIds())  # get atom index
        if prop not in properties_list.keys():
            properties_list[prop] = []
        if len(at_indx) > 0:
            for l in at_indx:
                properties_list[prop].append(list_labels[l])
        else:
            properties_list[prop].append(list_labels[at_indx[0]])
    return properties_list, mol

########################################
#
#     PLOT Trajectory analysis (RMSD, radius of gyration) for protein and ligand
#
########################################

def Plot_traj(rmsd_prot, rmsd_lig, auxi_rmsd, rgyr_prot, rgyr_lig, name, out_name=None):
    """
    Parameters:
    Returns:
    """
    color = ['forestgreen','lime','m','c','teal','orange','yellow','goldenrod','olive','tomato','salmon','seagreen']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 3))
    ax1.plot(rmsd_prot, label="protein")
    ax1.plot(rmsd_lig, label="ligand", color="red")
    max_value = max(np.max(np.asarray(rmsd_prot)), np.max(np.asarray(rmsd_lig)))
    for i, auxi in enumerate(auxi_rmsd):
        ax1.plot(auxi, label = str(i) ,color = color[i])
        max_value = max(max_value, np.max(np.asarray(auxi)))
    ax1.set_ylim(0, min(20, 1.2*max_value))
    ax1.legend(framealpha=0.0, edgecolor ='None')
    ax1.set_title('RMSD', fontsize=14)
    ax1.set_xlabel('frame #', fontsize=12)
    ax1.set_ylabel('RMSD /Angstrom', fontsize=12)

    if len(rgyr_prot) > 0:
        ax2.plot(0.1*np.asarray(rgyr_prot), label="protein(*0.1)")
    if len(rgyr_lig) > 0:
        ax2.plot(rgyr_lig, label="ligand", color = "red")
    if len(rgyr_prot) > 0   and len(rgyr_lig) > 0:
        ax2.set_ylim(0,1.5*max(np.max(np.asarray(rgyr_lig)),np.max(0.1*np.asarray(rgyr_prot) )))
    ax2.legend(framealpha = 0.0,edgecolor ='None')
    ax2.set_title('Radius of gyration', fontsize=14)
    ax2.set_xlabel('frame #', fontsize=12)
    ax2.set_ylabel('RGYR /Angstrom', fontsize=12)
    if out_name is not None:
        fig.tight_layout()
        fig.savefig(out_name)
    plt.close(fig)

    return

########################################
#
#     PLOT tauRAMD evaluation for a set of compounds with available exp.data
#
########################################
def PLOT_tauRAMD_dataset(tr,tr_name = None,types_list = [""],xlims=[0,4]):
    """
    Parameters:
    tr_FAK_- a set of trajectory objects (for each ligand)
    tr_FAK_name - name of the ligand to be indicated in the plot
    types_list - a list of ligand types to be shown in different colors
    Returns:
    """
    fig = plt.figure(figsize=(16, 8))
    plt.xlim(xlims)
    color = ['b','r','k','m','c','olive','tomato','firebrick','salmon','seagreen','salmon','peru']
    #color =  cm.rainbow(np.logspace(0.1, 1, len(tr)))


    x_tick_lable = []
    x_tick_pos = []
    for k in range(0,6):
        for ii,i in enumerate(range(pow(10,k),pow(10,k+1),pow(10,k))):  
            if(ii == 0): x_tick_lable.append(str(i/10.))
            else: x_tick_lable.append("")
            x_tick_pos.append(np.log10(i/10.))
    y_tick_lable = []
    y_tick_pos = []
    for k in range(0,3):
        for ii,i in enumerate(range(pow(10,k),pow(10,k+1),pow(10,k))):            
            if(ii == 0): y_tick_lable.append(str(i/10.))
            else: y_tick_lable.append("")
            y_tick_pos.append(np.log10(i/10.))
    Xt = []
    yt = []
    Xt_err = []
    yt_err = []
    txt = []
    for i,type_comp in enumerate(types_list): 
        X = []
        y = []
        X_err = []
        y_err = []
        for j,t in enumerate(tr):
            if t.tau_exp and t.ramd.tau:
                if ((t.type == type_comp) or ( t.type == "")):
                    X.append(np.log10(t.ramd.tau))
                    y.append(t.tau_exp)
                    Xt.append(np.log10(t.ramd.tau))
                    yt.append(t.tau_exp)
                    y_err.append(t.tau_exp_SD)
                    X_err.append(1/(t.ramd.tau*np.log(10))*t.ramd.tau_SD)
                    yt_err.append(t.tau_exp_SD)
                    Xt_err.append(1/(t.ramd.tau*np.log(10))*t.ramd.tau_SD)
                    if (len(tr) == len(tr_name)):
                        txt.append(tr_name[j])
#        print(type_comp,len(y),len(yt))
#        ax = fig.add_subplot(111)
        if(len(y) > 0):
            plt.errorbar(x=y,y=X,xerr=y_err,yerr= X_err, color = "gray" , fmt='o', markersize=1 )
            plt.scatter(x=y,y=X, color = color[i] , s=50 )
        else:
            logger.info(f"type {t.type} was not found")
            
#        if tr_name:
#            for j, t in enumerate(txt):    ax.annotate(t, (y[j], X[j]))
        if(len(y) > 8):
            slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=y,y=X)
            fitt = np.asarray(y)*slope+intercept
            ind = np.argwhere(np.abs(fitt-X) < 0.5).flatten()
            slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.asarray(y)[ind],y=np.asarray(X)[ind])
            fitt = np.asarray(y)*slope+intercept
            ind = np.argwhere(np.abs(fitt-X) >= 0.5).flatten()
#        if (len(ind) > 0):
#            if tr_name: print("Outliers: ",np.asarray(txt)[ind])
            plt.plot(y,fitt,color = color[i],linewidth=0.5,linestyle='dotted')

    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=yt,y=Xt)
    logger.info(f"Complete set: R2 = {r_value}")
    fitt = np.asarray(yt)*slope+intercept
    ind = np.argwhere(np.abs(fitt-Xt) < 0.5).flatten()
    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=np.asarray(yt)[ind],y=np.asarray(Xt)[ind])
    ind = np.argwhere(np.abs(fitt-Xt) >= 0.5).flatten()
    if (len(ind) > 0):
        if tr_name:
            logger.info(f"Outliers: {np.asarray(txt)[ind]}")
        plt.scatter(x=np.asarray(yt)[ind], y=np.asarray(Xt)[ind], color = 'orange', alpha=0.5, s=200)
    plt.plot(yt,fitt,color = 'k',linewidth=2)
    plt.grid(True)
    plt.xticks(x_tick_pos,x_tick_lable, fontsize=16)
    plt.yticks(y_tick_pos,y_tick_lable, fontsize=16)
    plt.show()
    logger.info(f"Without Outliers: R2 = {r_value}")
    return
