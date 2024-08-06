# import warnings
# warnings.filterwarnings("ignore")
import glob
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from loguru import logger
from rdkit import Chem
from rdkit import RDConfig

import MDAnalysis as mda

from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

from IFP_manipulation import table_combine

if mda.__version__ == "1.1.1":
    from IFP_generation_old import IFP
    IS_OLD_MDA = True
else:
    from IFP_generation_new import IFP
    IS_OLD_MDA = False

@dataclass
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
    def __init__(self, PRJ_DIR, pdb=None, top=None, timestep=1):
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
        self.prj_dir = PRJ_DIR
        self.pdb = pdb
        self.top = top
        self.timestep = timestep
        self.traj = []   # an array of Trj_Properties objects for wach trajectory in a replica
        self.contact_collection = []  # this is a container where we will accomulate all protein-ligand contacts

    def IFP_unify(self, subset=None):
        """
        unify comlumns of a set of IFP databases for generated from several trajectories
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
                for c in tr_c.df_properties.columns.tolist():
                    if c  in IFP_list:
                        pass
                    else:
                        IFP_list.append(c)
        for tr_replica in r_subset:
            for tr_c in tr_replica:
                to_add = np.setdiff1d(IFP_list, np.asarray(tr_c.df_properties.columns.tolist()))
                for k in to_add:
                    tr_c.df_properties[k] = np.zeros(tr_c.df_properties.shape[0],dtype=np.int8)
                tr_c.df_properties = tr_c.df_properties[np.concatenate((["time"],tr_c.df_properties.columns[tr_c.df_properties.columns != "time"].tolist()))]
                if "WAT" in tr_c.df_properties.columns.tolist():
                    tr_c.df_properties = tr_c.df_properties[np.concatenate((tr_c.df_properties.columns[tr_c.df_properties.columns != "WAT"].tolist(),["WAT"]))]
        self.contact_collection = IFP_list
        return IFP_list

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
        _IFP_list = self.IFP_unify(subset)
        for tr_replica, tr_name in zip(r_subset, n_subset):
            for i, tr_c in enumerate(tr_replica):
                tt = tr_c.df_properties
                tt["Repl"] = tr_name
                tt["Traj"] = str(i)
                tt["RMSDl"] = tr_c.rmsd_lig
                tt["RMSDp"] = tr_c.rmsd_prot
                tt["RGyr"] = tr_c.rgyr_lig
                tt["length"] = tr_c.length
                tt["COM"] = tr_c.com_lig
                for k, rmsd_auxi in enumerate(tr_c.rmsd_auxi):
                    #rmsd_auxi_ls = []
                    #logger.info(len(rmsd_auxi_ls), len(tr_c.rmsd_lig), len(tr_c.rmsd_lig))
                    # below, replacement for above nonsense with a lesser nonsense...
                    logger.info(f"rmsd_auxi={rmsd_auxi}, len(tr_c.rmsd_lig)={len(tr_c.rmsd_lig)}")
                    tt["Auxi_"+str(k)] = rmsd_auxi
                df1 = pd.concat([df1, tt])
                #except:
                #    logger.critical(f"failed to save properties for the replica {tr_name} since the trajectiry was not analyzed")
        if df1 is None:
            raise ValueError("Failed to create database dataframe")
        df1.to_pickle(file)
        logger.info(f"Database saved to pickle file {file}")
        return df1

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
            for _ in range(max_shuffle):
                np.random.shuffle(t)
                t_b = t[:sub_set]
                # select time when 50% of ligand dissocate
                t_b_sorted_50 = (np.sort(t_b)[int(len(t_b)/2.0-0.5)] + np.sort(t_b)[int(len(t_b)/2)]) / 2.0
                tau_bootstr.append(t_b_sorted_50)
        return tau_bootstr

    def scan_ramd(self):
        """
        Parameters: re
        Results:
        """
        #--- read exp data-----------
        if self.top is not None and self.pdb is not None:
            u = mda.Universe(self.prj_dir + self.top, self.prj_dir + self.pdb)
        elif self.pdb is not None:
            u = mda.Universe(self.prj_dir + self.pdb)
        else:
            raise ValueError("No pdb (and/or topology) file provided")
        sd_max = 0
        for rmd, repl in zip(self.names,self.repl_traj):
            ramd_l = []
            traj_properties = []
            for t in repl:  # loop over trajectories in each replica
                try:
                    u.load_new(t)
                except FileNotFoundError:
                    logger.error(f"Error while Reading a trajectory {t}")
                if len(u.trajectory) > 2:
                    ramd_l.append((self.timestep/1000.0)*len(u.trajectory)) # should be 2.0/1000.0 ?
                traj_properties.append(Trj_Properties())
            self.length.append(ramd_l)
            if len(ramd_l)>7:
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
            # we will estimate final tau RAMD from all replicas as an average (omitting Nans, ie. incomplete simulations)
            #non_empty  = np.asarray(self.replicas)[np.isnan(np.asarray(self.replicas).astype(float)) == False]
            non_empty  = np.asarray(self.replicas)[not np.isnan(np.asarray(self.replicas).astype(float))]
            if len(non_empty)>0:
                self.tau =  np.nanmean(non_empty)
                self.tau_SD = max(np.nanstd(non_empty),sd_max)

    def Plot_RAMD(self, tau_lims=(0,0)):
        """
        Parameters:
        lims -  x-limits
        Returns:
        plot
        """
        lims = tau_lims
        color = ['r', 'b', 'forestgreen', 'orange', 'lime', 'm', 'teal', 'c', 'yellow', 'goldenrod', 'olive', 'tomato', 'salmon', 'seagreen']
        meanpointprops = {"linestyle":'--', "linewidth":2.5, "color":'firebrick'}
        fig = plt.figure(figsize=(18, 2))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        time_max = 0
        for r in self.replicas_distr:
            if len(r)>0: time_max = max(time_max,np.max(r))
        x = np.linspace(start=lims[0],stop=lims[1],num=100)
        for i, (d, _b) in enumerate(zip(self.replicas_distr, self.length)):
            if self.replicas_distr[i] and len(d) > 2:
                sns.histplot(d, kde=True, common_norm=True, bins=3, color=color[i], label=self.names[i], ax=ax1)
                # sns.distplot(d, kde=True, hist = True, norm_hist = True, bins = 3, color =color[i],label=self.names[i], ax = ax1)
        _ymin, ymax = ax1.get_ylim()
        replicas_distr = [x for x in self.replicas_distr if x]  # here we eliminate all empty lists
        if tau_lims == (0,0):
            lims =(0,1.2*np.max(np.asarray(replicas_distr).flatten()))
        ax1.set_xlim(lims)
        try:
            ax2.set_ylim(lims)
            ax2.boxplot(self.replicas_distr, tick_labels=self.names, showmeans=True, meanline=True, meanprops=meanpointprops)
            ax2.plot([0, len(self.names)+1], [self.tau, self.tau], 'k-', lw=3, alpha=0.3)
            ax2.plot([0, len(self.names)+1], [self.tau-self.tau_SD, self.tau-self.tau_SD], 'k--', lw=3, alpha=0.3)
            ax2.plot([0, len(self.names)+1], [self.tau+self.tau_SD, self.tau+self.tau_SD], 'k--', lw=3, alpha=0.3)
        except IndexError:
            logger.error("Error in the boxplot: there is probably no data for one of the replicas")
        if self.tau and self.tau_SD:
            logger.info(f"Average over replicas: tauRAMD =  {np.round(self.tau,3)} +- {np.round(self.tau_SD,3)}")
            gauss = np.exp(-np.power((x - self.tau)/self.tau_SD, 2.)/2.)
            ax1.plot(x, ymax*gauss/max(gauss), 'k-', label='total', linewidth=3, alpha=0.3)
        ax1.legend(ncol=2 if i > 4 else 1)
        ax2.set_ylabel('tau [ns]', fontsize=12)
        ax1.set_xlabel('tau [ns]', fontsize=12)
        ax2.set_xlabel('replica', fontsize=12)
        try:
            ax2.set_ylim(0, 1.2*np.max(np.asarray(replicas_distr).flatten()))
        except ValueError:
            pass
        plt.show()
        plt.close(fig)

class trajectories:
    """
        ligand - object of the class ligand
        ramd -  object of the class ramd
        PRJ_DIR
        pdb
        top
        ramd_traj_tmpl  
        ramd_tmpl 
        timestep         
        tau_exp 
        tau_exp_SD
        type
        
    Sub-Classes
        3.1 ligand
                Functions:
                    _int__
                Variable:
                    ligand_names
                    property_list
                    mol
                    ligand_2D
                    radius
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

    def __init__(self,
                 PRJ_DIR,
                 ramd_tmpl="RAMD*",
                 pdb=None,
                 top=None,
                 ligand_pdb=None,
                 ligand_mol2=None,
                 ramd_traj_tmpl="*dcd",
                 timestep=1):
        """
        PRJ_DIR - directory to story data
        ramd_dirs - TEMPLATE used for search of directories with RAMD simulations (should contain also complete path)
        timestep - timestep used to save MD frames in ps
        """

        if pdb is None:
            raise ValueError("Need to provide at least a pdb reference file")

        if not os.path.isdir(PRJ_DIR):
            PRJ_DIR = os.getcwd()
        if PRJ_DIR[-1] != "/":
            PRJ_DIR = PRJ_DIR + "/"
        self.prj_dir = str(PRJ_DIR)


        if pdb is not None and not os.path.isfile(self.prj_dir + pdb):
            msg = f"file {self.prj_dir + pdb} was not found"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if top is not None and not os.path.isfile(self.prj_dir + top):
            msg = f"file {self.prj_dir + top} was not found"
            logger.error(msg)
            raise FileNotFoundError(msg)

        self.pdb = pdb
        self.top = top
        self.ramd_traj_tmpl ="/"+ramd_traj_tmpl
        self.ramd_tmpl ="/"+ramd_tmpl
        self.timestep = timestep
        self.sub_system = "protein or (resname MG MN CA Mg Mn) or (resname WAT HOH SOL TIP3)"  # G G3 G5 U5 C C3  can be added for RNA

        try:
            self.ligand = Ligand(PRJ_DIR, ligand_pdb, ligand_mol2)
            if self.top is not None and self.pdb is not None:
                u = mda.Universe(self.prj_dir + self.top, self.prj_dir + self.pdb)
            elif self.pdb is not None:
                u = mda.Universe(self.prj_dir + self.pdb)
            else:
                raise ValueError("No pdb (and/or topology) file provided")
            lig_atoms = u.select_atoms(f'resname {self.ligand.ligands_names[0]}')
            logger.info(f"Ligand {self.ligand.ligands_names[0]} found in the trajectory with {len(lig_atoms)} atoms")
            for atom in lig_atoms:
                logger.debug(atom)
        except mda.exceptions.SelectionError as exc:
            logger.exception(f"Selection error with this ligand name: {self.ligand.ligands_names[0]}")
            raise exc

        self.ramd = Ramd(PRJ_DIR, pdb, top, timestep)
        self.tau_exp = None
        self.tau_exp_SD = None
        self.type = None

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


    def analysis_traj(self, traj, start_analysis, step_analysis, WB_analysis, RE, Lipids, auxi_selection=None, reference="ref"):
        """
        Parameters:
        step_analysis - step to be used to scan over the trajectory
        start_analysis - starting snapshot for analysis; if start_analysis < 0, count from the end of the trajectory
        if (0 < start_analysis < 1) -  a fraction of the trajectory = start_analysis will be skipped
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
        rmsd_prot = []
        rmsd_lig = []
        rgyr_prot = []
        rgyr_lig = []
        com_lig = []
        df_prop = None
        df_HB = None
        df_WB = None

        if self.top is not None and self.pdb is not None:
            u = mda.Universe(self.prj_dir + self.top, self.prj_dir + self.pdb)
        elif self.pdb is not None:
            u = mda.Universe(self.prj_dir + self.pdb)
        else:
            raise ValueError("No pdb (and/or topology) file provided")
        u.load_new(traj)
        u_length = len(u.trajectory)
        u_size = os.path.getsize(traj)/(1024.*1024.)
        logger.info(f"   total number of frames = {u_length}; file size {u_size:.1f} M")

        if reference=="ref":
            if self.top is not None and self.pdb is not None:
                ref = mda.Universe(self.prj_dir + self.top, self.prj_dir + self.pdb)
            elif self.pdb is not None:
                ref = mda.Universe(self.prj_dir + self.pdb)
            else:
                raise ValueError("No pdb (and/or topology) file provided")
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
            start = stop - 1
        step = max(1, step_analysis)
        if len(Lipids) > 0:
            lipid_line = ""
            for l in Lipids:
                lipid_line = lipid_line+" "+l
            selection = f"(resname {sel_ligands}) or {self.sub_system} or (resname {lipid_line})"
        else:
            selection = f"{self.sub_system} or (resname  {sel_ligands})"

        selection_rmsd = ["protein and (not type H)", "resname " + sel_ligands + " and (not type H)"]
        auxi_rmsd = []
        for auxi in auxi_selection:
            # NOTE: this was found like that, it's a bit strange, the else path shoud not work,
            # but in my use-cases I never reached this code-path, so...
            logger.warning("GOING THROUGH A NOT-TESTED CODE-PATH WITH POSSIBLE ERRORS, WATCH OUT!")
            if len(auxi) > 2:  # this is a check if a single string or a list of strings is given
                selection = selection + " or " + auxi
                auxi_rmsd.append([])
                selection_rmsd.append(auxi)
            else:
                selection = selection + " or " + auxi_selection
                auxi_rmsd.append([])
                selection_rmsd.append(auxi)
                break

        system_reduced = u.select_atoms(selection)
        logger.info(f"selected sub-system: {selection}")

        copy_pos = AnalysisFromFunction(lambda ag: ag.positions.copy(), system_reduced)
        copy_pos.run(start=start, stop=stop, step=step)
        if IS_OLD_MDA:
            copy_pos_results = copy_pos.results
        else:
            copy_pos_results = copy_pos.results.timeseries
        u_mem = mda.Merge(system_reduced).load_new(copy_pos_results, format=MemoryReader)

        protein = u_mem.select_atoms("protein")
        ligand = u_mem.select_atoms(f"resname {sel_ligands}")
        #if not past_the_rest:
        logger.info(f"frames to be analyzed: {len(u_mem.trajectory)}")
        for frame, traj_frame in enumerate(u_mem.trajectory):
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

    def analysis_all_ramd(self,
                          WB_analysis=True,
                          step_analysis=1,
                          start_analysis=0,
                          repl_list= None,
                          RE = True,
                          Lipids = None,
                          auxi_selection = None):
        """
        Parameters:
        step_analysis - step to be used to scan over the trajectory
        start_analysis - starting snapshot for analysis; id start_analysis < 0 - cound from the end of the trajectory
        WB_analysis - run analysis of water bridges between protein and ligand; quite time-consuming
        repl_list - a list of replica numbers (indexes from a complete replica list) to be analyzed 
        Results:
        """
        repl_list = [] if repl_list is None else repl_list
        Lipids = [] if Lipids is None else Lipids
        auxi_selection = [] if auxi_selection is None else auxi_selection

        sel_ligands = self.ligand.ligands_names[0]

        if self.top is not None and self.pdb is not None:
            ref = mda.Universe(self.prj_dir + self.top, self.prj_dir + self.pdb)
        elif self.pdb is not None:
            ref = mda.Universe(self.prj_dir + self.pdb)
        else:
            raise ValueError("Need a pdb or top+pdb file(s) in input for reference")

        _Rgr0 = ref.select_atoms("protein").radius_of_gyration()

        if len(repl_list) > 0 : repl_scan =  repl_list
        else:    repl_scan = range(0,len(self.ramd.repl_traj))
        for j1 in repl_scan:
            rmd = self.ramd.names[j1]
            repl = self.ramd.repl_traj[j1]
            logger.info(f"Replica {j1}: {rmd}")
            if len(self.ramd.traj) < 1:
                logger.error("RAMD trajectories must be loaded first using the function ramd.scan_ramd() (trajectory class function)")
                sys.exit()

            for j2, repli in enumerate(repl):
                logger.info(f"traj {j2}, file {repli}")
                step = max(step_analysis, 1)
                length, start, rmsd_prot, rmsd_lig, rmsd_auxi, rgyr_prot, rgyr_lig, com_lig, df_prop, df_HB, df_WB = self.analysis_traj(
                    repli, start_analysis, step, WB_analysis, RE, Lipids, auxi_selection)

                df_prop_complete = table_combine(df_HB,df_WB,df_prop,sel_ligands,self.ramd.contact_collection)
                Plot_traj(rmsd_prot, rmsd_lig, rmsd_auxi, rgyr_prot, rgyr_lig)
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


class  Ligand:
    def __init__(self, PRJ_DIR, ligand_pdb:str|None=None, ligand_mol2:str|None=None):
        self.ligands_names = []
        self.property_list = {}
        self.mol = None
        resnames = []
        list_labelsF = []
        USE_PDB = ""
        if ligand_mol2 is not None:
            if os.path.isfile(PRJ_DIR + "/" + ligand_mol2):
                try:
                    self.mol, list_labels, resnames = self.ligand_Mol2(PRJ_DIR + "/" + ligand_mol2)
                except ValueError:
                    logger.info("Mol2 is absent or unreadable, PDB file will be used for ligand structure analysis")
                    logger.warning("Aromatic atoms can not be recognized using PDB file")

            if self.mol is None and os.path.isfile(PRJ_DIR + "/" + ligand_pdb):
                self.mol, list_labels, resnames = self.ligand_PDB(PRJ_DIR + "/" + ligand_pdb)
                if self.mol is None:
                    logger.warning("RDKit cannot read PDB structure, trying to correct atom naming")
                    rename_H(PRJ_DIR + "/" + ligand_pdb, PRJ_DIR + "/Corrected_naming_ligand.pdb")
                    self.mol, list_labels, resnames = self.ligand_PDB(PRJ_DIR + "/Corrected_naming_ligand.pdb")
                    if self.mol is None:
                        raise ValueError("RDKit cannot read PDB structure, even after correcting atom naming")
                    USE_PDB = "Corrected_naming_ligand.pdb"
                else:
                    USE_PDB = ligand_pdb
            else:
                logger.error(f"nether ligand PDB nor Mol2 were found in {PRJ_DIR} expected file names: {ligand_pdb} {ligand_mol2}")
                raise FileNotFoundError()

            self.properties_list, self.ligand_2D = self.ligand_properties(self.mol, list_labels)

            # add fluorine as hydrophobic atoms (absent in RDkit)
            if len(self.properties_list) > 0:
                if USE_PDB:
                    list_labelsF  = self.ligand_PDB_F(PRJ_DIR + "/" + USE_PDB)
                    list_labelsPO3 = []
                    list_labelsP = []
                else:
                    list_labelsF, list_labelsPO3, list_labelsP  = self.ligand_Mol2_F_PO3(PRJ_DIR + "/" + ligand_mol2)

                if len(list_labelsF) > 0:
                    if  'Hydrophobe' in self.properties_list:
                        new_properties_list_H = self.properties_list['Hydrophobe']
                        for at in list_labelsF:
                            new_properties_list_H.append(at)
                        self.properties_list['Hydrophobe'] = new_properties_list_H
                    else:
                        self.properties_list.update({'Hydrophobe': list_labelsF})
                    logger.info(f"Fluorine atoms are found (will be considered as Hydrophobe): {list_labelsF}")

                if len(list_labelsP) > 0:
                    if  'NegIonizable' in self.properties_list:
                        new_properties_list_H = self.properties_list['NegIonizable']
                        for at in list_labelsP:
                            new_properties_list_H.append(at)
                        self.properties_list['NegIonizable'] = new_properties_list_H
                    else:
                        self.properties_list.update({'NegIonizable': list_labelsP})
                    logger.info(f"PO3 group is found (P atoms will be considered as NegIonizable): {list_labelsP}")

                if len(list_labelsPO3) > 0:
                    if  'Acceptor' in self.properties_list:
                        new_properties_list_H = self.properties_list['Acceptor']
                        for at in list_labelsPO3:
                            new_properties_list_H.append(at)
                        self.properties_list['Acceptor'] = new_properties_list_H
                    else:
                        self.properties_list.update({'Acceptor': list_labelsPO3})
                    logger.info(f"PO3 group is found (O atoms will be considered as Acceptors): {list_labelsPO3}")


                logger.info("...............Ligand properties:................")
                for k, val in self.properties_list.items():
                    logger.info(f"{k} {val}")
            else:
                logger.error("RDKit cannot generate ligand property list")
                raise ValueError

            self.ligands_names = np.unique(resnames)
            logger.info(f"The following residue names will be used to identify ligand in the PDB file: {self.ligands_names}")
        else:
            logger.warning("ligand PDB and Mol2 are not defined")


    def ligand_Mol2(self, ligand_mol2):
        """
        Parameters:
        ligand_mol2 - ligand structure file  in the Mol2 format (not all mol2 format work, but generated by MOE does)
        Returns:
        mol - RDKit molecular object
        list_labels - list of atom names
        resnames - list of residue names (for all atoms)
        """
        list_labels = []
        resnames = []
        start = False
        with open(ligand_mol2, "r") as ff:
            for line in ff:
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

    def ligand_PDB(self, ligand_pdb):
        """
        Parameters:
        ligand_pdb - ligand structure file  in the PDB format
        Results:
        mol - RDKit molecular object
        list_labels - list of atom names
        resnames - list of residue names (for all atoms)
        """
        list_labels = []
        resnames = []
        with open(ligand_pdb, "r") as ff:
            for line in ff:
                if line.split()[0] == 'ATOM':
                    list_labels.append(line.split()[2])
                    resnames.append(line.split()[3])
        mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False)
        return mol, list_labels, resnames

    def ligand_PDB_F(self, ligand_pdb):
        """
        Parameters:
        ligand_pdb - ligand structure file  in the PDB format
        Results:
        list_labels - list of  names for F atoms found
        """
        list_labels = []
        with open(ligand_pdb, "r") as ff:
            for line in ff:
                if len(line.split()) > 5:
                    if (line.split()[0] == 'ATOM' or line.split()[0] == 'HETATM'):
                        if line.split()[2][0] == "F":
                            list_labels.append(line.split()[2])
        return list_labels

    def ligand_Mol2_F_PO3(self, ligand_mol2):
        """
        Parameters:
        ligand_mol2 - ligand structure file  in the MOL2 format
        Results:
        list_labels_P - list of  names for oxygen atoms bound to P
        list_labels - list of names for F atoms found
        """
        list_labels_O = []
        list_labels_P = []
        list_labels_F = []
        list_atoms = []
        list_P = []
        start = 0
        with open(ligand_mol2, "r") as ff:
            for line in ff:
                k = line.split()
                if line.find("<TRIPOS>ATOM") >= 0:
                    start = 1
                elif line.find("<TRIPOS>BOND") >= 0:
                    start = 2
                elif line.find("<TRIPOS>SUBSTRUCTURE") >= 0:
                    break
                else:
                    if start == 1:
                        list_atoms.append(k[1])
                        if k[1][0] == "P":
                            list_P.append(k[0])
                            list_labels_P.append(k[1])
                        if k[1][0] == "F":
                            list_labels_F.append(k[1])
                    if start == 2:
                        for P in list_P:
                            if int(k[1]) == int(P):
                                if list_atoms[int(k[2])-1][0] == 'O':
                                    if list_atoms[int(k[2])-1] not in list_labels_O:
                                        list_labels_O.append(list_atoms[int(k[2])-1])
                            if int(k[2]) == int(P):
                                if list_atoms[int(k[1])-1][0] == 'O':
                                    if list_atoms[int(k[1])-1] not in list_labels_O:
                                        list_labels_O.append(list_atoms[int(k[1])-1])
        return list_labels_F, list_labels_O, list_labels_P

    def ligand_properties(self, mol, list_labels):
        """
        Parameters:
        mol - RDKit molecular object
        list_labels - list of atom names
        Results:
        properties_list - a dictionary containing types of chemical properties and corresponding atoms 
        """
        ligand_2D = Chem.rdmolfiles.MolFromSmiles(Chem.rdmolfiles.MolToSmiles(mol))
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = Chem.ChemicalFeatures.BuildFeatureFactory(fdefName)
        feats = factory.GetFeaturesForMol(mol)
        properties_list = {}
        for f in feats:
            prop = f.GetFamily()  #  get property name
            at_indx  = list(f.GetAtomIds())  # get atom index
            if prop not in properties_list:
                properties_list[prop] = []
            if len(at_indx) > 0:
                for l in at_indx:
                    properties_list[prop].append(list_labels[l])
            else:
                properties_list[prop].append(list_labels[at_indx[0]])
        return properties_list, ligand_2D

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

def pbc(u, Rgr0):
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
        u.atoms.translate(-u_CA.center_of_mass()+0.5*u.dimensions[0:3])
        u.atoms.pack_into_box(box=u.dimensions)
        Rgr = u.select_atoms(sel_p).radius_of_gyration()
    if Rgr>Rgr0*1.1:
        logger.info(f"failed to pack the system back into a box radius of gyration is too large: {Rgr/Rgr0:.3f} of that in the first frame")
    return Rgr

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

def rename_H(ligand_H_name, ligand_H_name_new=None) -> str:
    """
    Function that:
        - rename H atoms in a ligand pdb file generated by openbabel (openbabel generats all hydrogen as H)
        - remove connectivity lines
        - adjust position of hydrogen atom names so that H always occupies 14th position (requiered by Rdkit)
    """
    ligand_H_name_new = ligand_H_name if ligand_H_name_new is None else ligand_H_name_new
    resname = None
    lig = []
    hi = 1
    with open(ligand_H_name, "r") as ff:
        for line in ff:
            if len(line)>20:
                if ((line[12:16].strip()[0] == "H") or (line[12:13] != " ")):
                    s = list(line)
                    if line[12:16].strip() == "H":
                        new_name = "H" + str(hi)
                    else:
                        new_name =  line[12:16].strip()
                    s[12:17] = f"{new_name:<4}"
                    hi += 1
                    line = "".join(s)
                if line.split()[0] == "ATOM" or line.split()[0] == "HETATM":
                    lig.append(line.replace("HETATM", "ATOM  "))
                    resname = line[16:20].strip()

    if len(lig) > 0:
        with open(ligand_H_name_new, "w") as ff:
            for p in lig:
                ff.write(p)
    if resname is None:
        raise ValueError(f"Failed renaming H in file {ligand_H_name}")
    return resname

def read_ligands_mol2(ligand_mol2):
    """
    Parameters:
    - ligand mol2 file; NB: mol2 file created by antechamber does not work! created by MOE works
    Returns:
    - Rkit molecular object for the ligand and ligand image
    """
    mol = Chem.rdmolfiles.MolFromMol2File(ligand_mol2, removeHs=False)
    try:
        sm = Chem.MolToSmiles(mol)
        t1 =Chem.MolFromSmiles(sm)
        return mol, t1
    except ValueError:
        logger.error(f"ERROR in mol2 file {ligand_mol2}")
        return mol, None

def read_ligands_mol2_AtomLabels(ligand_mol2):

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
        if line.find("<TRIPOS>ATOM")>=0:
            start = True
        elif line.find("<TRIPOS>BOND")>=0:
            start = False
        else:
            if start:
                list_labels.append(key[1])
                list_resname.append(key[7])
                list_pos.append([float(key[2]), float(key[3]), float(key[4])])
    center = np.mean(np.asarray(list_pos), axis=1)[0]
    radius=max(np.sum(np.abs(np.asarray(list_pos)-center)**2,axis=1)**0.5)
    return list_labels, list_resname, radius

def read_ligands_pdb_AtomLabels(ligand_pdb):
    radius = 0
    list_labels = []
    list_resname = []
    list_pos = []
    with open(ligand_pdb, "r") as ff:
        for line in ff:
            key = line.split()
            if key[0]=='ATOM':
                list_labels.append(key[2])
                list_resname.append(key[3])
                try:
                    list_pos.append([float(key[5]),float(key[6]),float(key[7])])
                except (IndexError, ValueError):
                    logger.warning(f"Format error in {ligand_pdb}. Ligand pdb file should not contain chain infromation")
    center = np.mean(np.asarray(list_pos),axis=1)[0]
    radius=max(np.sum(np.abs(np.asarray(list_pos)-center)**2,axis=1)**0.5)
    return list_labels, list_resname, radius

# def  ligand_properties(ligand_pdb, ligand_mol2):
#     """
#     ligand_pdb - ligand structure file  in the PDB format
#     ligand_mol2 - ligand structure file  in the Mol2 format (not all mol2 format work, but generated by MOE does)
#     """
#     with open(ligand_pdb,"r") as ff:
#         list_labels = [l.split()[2] for l in ff.readlines() if l.split()[0] in ['ATOM', "HETATM"]]

#     if not os.path.exists(ligand_mol2):
#         logger.warning("MOL2 does not exist; ligand properties will be derived from the PDB file i.e. aromatic properties will be missed")

#     mol = Chem.rdmolfiles.MolFromMol2File(
#         ligand_mol2 if os.path.exists(ligand_mol2) else ligand_pdb,
#         removeHs=False)

#     fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
#     factory = Chem.ChemicalFeatures.BuildFeatureFactory(fdefName)
#     feats = factory.GetFeaturesForMol(mol)

#     properties_list = {}
#     for f in feats:
#         prop = f.GetFamily()  #  get property name
#         at_indx  = list(f.GetAtomIds())  # get atom index
#         if prop not in properties_list:
#             properties_list[prop] = []
#         if len(at_indx) > 0:
#             for l in at_indx:
#                 properties_list[prop].append(list_labels[l])
#         else:
#             properties_list[prop].append(list_labels[at_indx[0]])
#     return properties_list, mol

def Plot_traj(rmsd_prot, rmsd_lig, auxi_rmsd, rgyr_prot, rgyr_lig, out_name=None):
    """
    Parameters:
    Returns:
    """
    color = ['forestgreen','lime','m','c','teal','orange','yellow','goldenrod','olive','tomato','salmon','seagreen']
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))
    assert isinstance(axes, np.ndarray)
    ax1, ax2 = axes
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

def PLOT_tauRAMD_dataset(tr, tr_name: list[str], types_list: list[str]|None = None, xlims: list[float]|None = None):
    """
    Parameters:
    tr - a set of trajectory objects (for each ligand)
    tr_name - name of the ligand to be indicated in the plot
    types_list - a list of ligand types to be shown in different colors
    Returns:
    """
    xlims = [0, 4] if xlims is None else xlims
    types_list = [""] if types_list is None else types_list

    fig = plt.figure(figsize=(16, 8))
    plt.xlim(xlims)
    color = ['b','r','k','m','c','olive','tomato','firebrick','salmon','seagreen','salmon','peru']
    #color =  cm.rainbow(np.logspace(0.1, 1, len(tr)))


    x_tick_lable = []
    x_tick_pos = []
    for k in range(0, 6):
        for ii,i in enumerate(range(pow(10,k),pow(10,k+1),pow(10,k))):
            if ii==0:
                x_tick_lable.append(str(i/10.))
            else:
                x_tick_lable.append("")
            x_tick_pos.append(np.log10(i/10.))
    y_tick_lable = []
    y_tick_pos = []
    for k in range(0, 3):
        for ii,i in enumerate(range(pow(10,k),pow(10,k+1),pow(10,k))):
            if ii==0:
                y_tick_lable.append(str(i/10.))
            else:
                y_tick_lable.append("")
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
                    if len(tr)==len(tr_name):
                        txt.append(tr_name[j])
        if len(y)>0:
            plt.errorbar(x=y,y=X,xerr=y_err,yerr= X_err, color = "gray" , fmt='o', markersize=1 )
            plt.scatter(x=y,y=X, color = color[i] , s=50 )
        else:
            logger.info(f"type {t.type} was not found")

        if len(y)>8:
            slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=y,y=X)
            fitt = np.asarray(y)*slope+intercept
            ind = np.argwhere(np.abs(fitt-X) < 0.5).flatten()
            slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=np.asarray(y)[ind],y=np.asarray(X)[ind])
            fitt = np.asarray(y)*slope+intercept
            ind = np.argwhere(np.abs(fitt-X) >= 0.5).flatten()
            plt.plot(y,fitt,color = color[i],linewidth=0.5,linestyle='dotted')

    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=yt,y=Xt)
    logger.info(f"Complete set: R2 = {r_value}")
    fitt = np.asarray(yt)*slope+intercept
    ind = np.argwhere(np.abs(fitt-Xt) < 0.5).flatten()
    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x=np.asarray(yt)[ind],y=np.asarray(Xt)[ind])
    ind = np.argwhere(np.abs(fitt-Xt) >= 0.5).flatten()
    if len(ind)>0:
        if tr_name:
            logger.info(f"Outliers: {np.asarray(txt)[ind]}")
        plt.scatter(x=np.asarray(yt)[ind], y=np.asarray(Xt)[ind], color = 'orange', alpha=0.5, s=200)
    plt.plot(yt,fitt,color = 'k',linewidth=2)
    plt.grid(True)
    plt.xticks(x_tick_pos,x_tick_lable, fontsize=16)
    plt.yticks(y_tick_pos,y_tick_lable, fontsize=16)
    plt.show()
    plt.close(fig)
    logger.info(f"Without Outliers: R2 = {r_value}")
