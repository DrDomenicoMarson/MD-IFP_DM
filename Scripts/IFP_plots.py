import numpy as np
#from matplotlib import *
#from matplotlib.patches import ArrowStyle
#from matplotlib.patches import Ellipse
from scipy.spatial import distance

import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
#sns.set_context("paper", font_scale=0.6)

def plot_graph(
        df_ext, out_base_name="",
        ligand: list = None,
        draw_round = False,
        water = False):
    """
    Create a graph-based representation of ligand dissociation, derived from IFPs
    Each cluster is shown by a node with the size indicating the cluster population.

    Nodes are
     - positioned on an increasing logarithmic scale of the average ligand COM position
       (or RMSD from the starting snapshot??)
     - the node color denotes change of the ligand RMSD in the cluster from the starting structure.
     - optionally (arg. water) surrounded by blue layer,
        with thickness proportional to the ligand solvation shell in that cluster

    Light-orange/pink arrows 
     - represent the transitions Ci-->Cj and Cj-->Ci between two nodes
     - has the width proportional to the number of transitions

    Gray arrows
     - indicate the total flow between two nodes
        i.e., transitions (Ci-->Cj)-(Cj-->Ci)

    Create also transition density and Flow plots,
    that illustrate the number of (Ci-->Cj) transitions and net flow

    
    Parameters
    ----------
    df_ext: pd.DataFrame
        dataframe with cluster information in label column
    out_base_name: str
        base file name to save the images
    ligand: list[str] | None
        generate dissociation pathways for a selected ligands only
        (note that clusters properties still include data for all ligands)
    draw_round: bool, optional [False]
        type of representation (plain/round)
    water: bool, optional [False]
        visualize number of water molecules in the ligand solvation shell for each clutser
    
    Returns
    ----------
    the idx that would sort the cluster labels by increase of the average ligand RMSD in each cluster
    """

    print("Ploting the graph...")

    df_ext_ligand = df_ext
    if ligand is not None and len(ligand) > 0:
        df_ext_ligand = df_ext[df_ext.ligand.isin(ligand)]
        print("Edges will be shown for one ligand:", ligand)

    label_rmsd = []
    label_com = []
    label_size = []
    label_water = []

    #labels_list, nodes = np.unique(df_ext.label.values,return_counts= True)
    labels_list = np.unique(df_ext.label.values)
    edges = np.zeros((labels_list.shape[0], labels_list.shape[0]), )
    coms = np.zeros((labels_list.shape[0], labels_list.shape[0]), )

    # loop ove all cluaters and compute their  population (label_size)
    for i, l in enumerate(labels_list):
        current_set = df_ext[df_ext.label == l]
        label_rmsd.append(current_set.RMSDl.mean())
        label_com.append(current_set.COM.mean(axis=0))
        print(f"cluster {l}:")
        print(f"   STD of COM: {current_set.COM_x.std():.3f} {current_set.COM_y.std():.3f} {current_set.COM_z.std():.3f}")
        print(f"  STD of RMSD: {current_set.RMSDl.std():.3f}")
        print(f"        Water: {label_water}")

        label_size.append(100*current_set.shape[0]/df_ext.shape[0])
        if water:
            label_water.append(int(current_set.WAT.mean()))
        # compute distances between clusters COM
        for j in range(0, i):
            coms[i, j] = distance.euclidean(label_com[i], label_com[j])

    # loop to compute edges dencity
    for l, (df_label, df_time) in enumerate(zip(df_ext_ligand.label.values, df_ext_ligand.time.values)):
        if df_time != 0:
            if df_ext_ligand.label.values[l-1] != df_label:
                edges[df_ext_ligand.label.values[l-1], df_label] += labels_list.shape[0]/df_ext_ligand.label.values.shape[0] 

    # print(np.max(edges), edges[edges > 0.5*np.max(edges)])

    idx_com_with_min_rmsd = np.argwhere((np.asarray(label_rmsd) == min(label_rmsd)))[0][0]

    def get_relative_distance_from_min_rmsd():
        dist_from_min_rmsd = np.array(
            [distance.euclidean(com, label_com[idx_com_with_min_rmsd]) for com in label_com])
        return ((100*dist_from_min_rmsd)/np.max(dist_from_min_rmsd)).astype(np.int64)

    relative_dist_from_min_rmsd = get_relative_distance_from_min_rmsd()

    print(f"Index of cluster with min lRMSD: {idx_com_with_min_rmsd} (lRMSD {min(label_rmsd):.4f})")
    print(f"    Relative distance (%) from the cluster with min(lRMSD): {relative_dist_from_min_rmsd}")
    print(f"    sorted lRMSDs: {np.sort(label_rmsd)}")

    def plot_transition_density_and_flow():
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.set_xlabel("cluster #")
            ax.set_ylabel("cluster #")
        axes[0].set_title("Transition density")
        axes[0].imshow(edges, cmap='Blues')
        axes[1].set_title("Flow")
        axes[1].imshow(edges-edges.T, cmap='Reds')
        fig.savefig(f"{out_base_name}.transition_density_and_flow.pdf")
        plt.close(fig)
    plot_transition_density_and_flow()


    starting_labels = df_ext[df_ext.time == 0].label.values  # list of clusters of all first frames in all trajectories
    starting_list = np.unique(starting_labels) # list of clusters that appear as first frame

    #------------ older cluster position-------------
    # label2scale = label_rmsd # what value will be on x
    # label2order = labels_list # what value will be on y

    index_x_t = np.argsort(label_rmsd)  #label2scale
    # x and y positions of each cluster in the plot
    label_x = np.zeros((len(labels_list)))
    label_y = np.zeros((len(labels_list)))
    color_com = np.zeros((len(labels_list)))


    max_x = max(label_rmsd)
    step_x = 1.0*max_x/max(label_rmsd) # WTF??? WHY???
    # firt clusters that contain first frames of trajectories
    for i, s in enumerate(starting_list):
        label_x[s] = label_rmsd[s]*step_x #label2scale
        label_y[s] = labels_list[s] ##label2order
        color_com[s] = relative_dist_from_min_rmsd[s]
    j = 0
    for l in index_x_t:
        if labels_list[l] not in starting_list:
            while label_x[j]!=0:
                j += 1
                if j==len(labels_list):
                    break
            if j==len(labels_list):
                break
            label_x[labels_list[l]] = label_rmsd[l]*step_x  #label2scale
            label_y[labels_list[l]] = labels_list[l] #label2order
            color_com[labels_list[l]] = relative_dist_from_min_rmsd[l]

    # set logarythmic scale
    label_x = np.log10(label_x)
    x_tick_label = []
    x_tick_pos = []
    for k in range(0, 2):
        for i in range(pow(10, k), pow(10, k+1), pow(10, k)):
            x_tick_label.append(str(i))
            x_tick_pos.append(np.log10(i))
            if i > 25: # <-- why? whould go to 90, stopping at (1...9, 10, 30)
                break

    fig_flow, ax_flow = plt.subplots(1, 1)
    fig_number, ax_number = plt.subplots(1, 1)
    axes = [ax_flow, ax_number]
    figs = [fig_flow, fig_number]

    if draw_round:
        alpha = 0.9*2*3.14*label_x/np.max(label_x)
        alpha_regular = 0.9*2*3.14*np.asarray(x_tick_pos)/max(x_tick_pos)
        label_y = np.sin(alpha)
        label_x = np.cos(alpha)
        for ax in axes:
            ax.scatter(x=np.cos(alpha_regular), y=np.sin(alpha_regular), c='k', s=10)
            for l, p in zip(x_tick_label, x_tick_pos):
                ax.annotate(
                    str(l)+"A",
                    (1.2*np.cos(0.9*2*3.14*p/max(x_tick_pos)),
                    np.sin(0.9*2*3.14*p/max(x_tick_pos))),
                    fontsize=14, color="gray")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
    else:
        for ax in axes:
            ax.set_ylabel('Cluster #')
            ax.set_xlabel(r'RMSD [ $\mathrm{\AA}$ ]')
            # for matplotlib < 3.5, needed two calls,
            #   this is for newer matplotlib: 
            #   ax.set_xticks(x_tick_pos, labels=x_tick_label)
            ax.set_xticks(x_tick_pos)
            ax.set_xticklabels(x_tick_label)
            ax.set_ylim(-1, len(label_y)+1)

    for l in range(0, label_x.shape[0]):
        for n in range(l+1, label_x.shape[0]):
            # total number of transitions in both directions

            a = n if label_rmsd[l] > label_rmsd[n] else l
            b = l if label_rmsd[l] > label_rmsd[n] else n

            xy = (label_x[b], label_y[b])
            xytext = (label_x[a], label_y[a])
            if edges[l, n] > 0:
                if  draw_round is False or np.abs(label_rmsd[l] - label_rmsd[n]) > min(label_rmsd)/2:
                    ax_number.annotate(
                        "",
                        xy = xy,
                        xycoords = 'data',
                        xytext = xytext,
                        textcoords = 'data',
                        size = edges[l, n]*500,
                        arrowprops = dict(
                            arrowstyle = "Fancy, head_length=0.2, head_width=0.4, tail_width=0.2",
                            fc = "pink",
                            ec = "none",
                            alpha = 0.5,
                            connectionstyle = "arc3, rad=-0.5"
                            )
                        )
                    ax_number.annotate(
                        "",
                        xy = xytext,
                        xycoords = 'data',
                        xytext = xy,
                        textcoords = 'data',
                        size = edges[l, n]*500,
                        arrowprops = dict(
                            arrowstyle = "Fancy, head_length=0.2, head_width=0.4, tail_width=0.2",
                            fc = "purple",
                            ec = "none",
                            alpha = 0.5,
                            connectionstyle = "arc3, rad=-0.5"
                            )
                        )
            flow = edges[l, n] - edges[n, l]  # flow l --> n
            a = l if flow > 0 else n
            b = n if flow > 0 else l
            xy = (label_x[b], label_y[b])
            xytext = (label_x[a], label_y[a])
            ax_flow.annotate(
                "",
                xy = xy,
                xycoords = 'data',
                xytext = xytext,
                textcoords = 'data',
                size = np.abs(flow)*1000,
                arrowprops = dict(
                    arrowstyle = "Simple, head_length=0.2, head_width=0.4, tail_width=0.2",
                    fc = "0.8",
                    ec = "none",
                    alpha=0.8,
                    connectionstyle = "arc3, rad=-0.5"
                    )
                )

    for ax in axes:
        for i, txt in enumerate(labels_list):
            ax.annotate(txt, (label_x[txt], label_y[txt]+0.05*pow(i,0.5)))
        if water:
            ax.scatter(
                label_x,
                label_y,
                facecolors='none',
                c=color_com,
                edgecolors="lightskyblue",
                s=500*np.asarray(label_size),
                cmap='Oranges',
                linewidths=np.asarray(label_water))
            print("WATERS:", np.asarray(label_water))
        else:
            ax.scatter(
                label_x,
                label_y,
                facecolors='none',
                c=color_com,
                edgecolors="k",
                s=500*np.asarray(label_size),
                cmap='Oranges')

    for fig, name in zip(figs, ["flow", "number"]):
        fig.tight_layout()
        if out_base_name != "":
            fig.savefig(f"{out_base_name}.{name}.pdf")
        else:
            fig.show()
        plt.close(fig)

    return np.argsort(label_rmsd)
