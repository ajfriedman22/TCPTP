#!/ usr / bin / env python

#Load MDTraj trajectory and process to output one full trajectory, one with no solvent, one with all protein residues, and one with protein bb atoms only
#Input: File_traj = GROMACS trajectory in XTC format, File_gro = GROMACS GRO file, a7_res = residues which make up the a7 helix
#Output: 
#traj_bb = MDTraj trajectory with only protein bb atoms
#traj_prot = MDTraj trajectory with all protein atoms and residues
#traj_ns = MDTraj trajectory with solvent molecules removed
#traj_a7 = MDTraj trajectory with only the atoms in the a7 helix
#miss_first = Returns True if the first residue of PTP1B is missing and all indices need to be subtracted by 1
def mdtraj_load(File_traj, File_gro, a7_res, offset):
    #Import required packages
    import mdtraj as md

    #Load trajectories
    traj = md.load(File_traj, top=File_gro)
    top = traj.topology

    #Process Trajectory
    traj_bb = traj.atom_slice(top.select('backbone')) #Backbond atoms of PTP1B only
    traj_prot = traj.atom_slice(top.select('protein')) #Select only atoms in the protein
    traj_ns = traj.remove_solvent() #Remove solvent from the trajectory leaving only protein (and ligand if applicable)
    
    #Set residues for a7 helix and adjust to base 0
    first = int(a7_res[0]) - offset
    last = int(a7_res[1]) - offset

    traj_a7 = traj.atom_slice(top.select(str(first) + ' <= resid and resid <= ' + str(last))) #Select only atoms in the a7 helix

    return traj_bb, traj_prot, traj_ns, traj_a7

#Function to set arrays for all PTP1B protein sections
#Input:
#miss_first = Returns True if the first residue of PTP1B is missing and all indices need to be subtracted by 1
#lig = Name of ligand if no ligand is present input 'none'
#a7_present = If the a7 helix is present in the protein
#Output:
#group sections for each of the structures of interest
def set_sect(lig, a7_present, offset):
    import numpy as np
    #MDtraj numbers residues starting with zero. Add an offset to ensure that the same residues are compared between trajectories
    group_WPD = np.linspace(179-offset, 187-offset, num = 9) #residues in the WPD loop
    group_3 = np.linspace(188-offset, 202-offset, num = 15) #residues in the a3 helix
    group_4 = np.linspace(221-offset, 238-offset, num = 18) #residues in the a4 helix
    group_5 = np.linspace(245-offset, 254-offset, num = 10) #residues in the a5 helix
    group_6 = np.linspace(262-offset, 277-offset, num = 16) #residues in the a6 helix
    group_bend = np.linspace(278-offset, 283-offset, num = 6) #residues in the bend b/w the a6 and a7 helices
    group_7 = np.linspace(284-offset, 294-offset, num = 11) #residues in the a7 helix
    group_L11 = np.linspace(149-offset, 152-offset, num = 4) #residues in the L11 loop
    
    #Set the index for the ligand if present in trajectory
    return group_WPD, group_3, group_4, group_5, group_6, group_bend, group_7, group_L11

#Function computes the % of time there are simultaneous contacts between the ligand and residues in pairs A with those in pairs A, B, and C 
#Input:
#pairs_A, pairs_B, pairs_C  = The residue interaction pairs for the three setions of interest
#time_uncorr = indices for uncorrelated frames in the trajectory
#dist_A, dist_B, dist_C = distance between the residue pairs above
#num_1 = start index for simul_contacts
#simul_contacts = base array to augment with % each simultaneous interaction is formed
#a7_present = If the a7 helix is present in the protein

def compute_simul_comtacts(pairs_A, pairs_B, pairs_C, time_uncorr, dist_A, dist_B, dist_C, num_1, simul_contacts, a7_present):
    for i in range(pairs_A):
        count = 0
        count2 = 0
        num_2 = 0
        for j in range(pairs_A):
            for t in range(time_uncorr):
                if dist_A[t][i] <= 0.5 and dist_A[t][j] <= 0.5:
                    count += 1
            simul_contacts[num_1][num_2] = count/time_uncorr

            count = 0
            count2 = 0
            num_2 += 1
        for k in range(pairs_B):
            for t in range(time_uncorr):
                if dist_A[t][i] <= 0.5 and dist_B[t][k] <= 0.5:
                    count += 1
            simul_contacts[num_1][num_2] = count/time_uncorr

            count = 0
            count2 = 0
            num_2 += 1
        if a7_present == True:
            for l in range(pairs_C):
                for t in range(time_uncorr):
                    if dist_A[t][i] <= 0.5 and dist_C[t][l] <= 0.5:
                        count += 1
                simul_contacts[num_1][num_2] = count/time_uncorr

                count = 0
                count2 = 0
                num_2 += 1
        num_1 += 1
    return simul_contacts, num_1

#Function to determine if there are any contacts formed within the specified residue section
#Input:
#dist = matrix of distances between two points
#t = frame of trajectory
#low, high = define the section of intest from residue low to residue high
#Output: 1 if contact is present and 0 if not
def sect_contact(dist, t, low, high):
    dist_sect = dist[t][low:high]
    if min(dist_sect) <= 0.5:
        return 1
    else:
        return 0

#Determine the helicitiy and alpha helicity
#Input: dssp = matrix of DSSP assignments with frames in rows and residues in columns
#Output:
#alpha_per = % that each residue is in the alpha helical confrmation
#struct_per = % that each residue is in the any helical confrmation
#alpha_per_mean = mean % time each residue is in the alpha helical confrmation
#struct_per_mean = mean % time each residue is in any helical confrmation
#alpha_per_sem = standard error on the mean % time each residue is in the alpha helical confrmation
#struct_per_sem = standard error on the mean % time each residue is in any helical confrmation
#alpha_per_time = % alpha helicity of all residues in the a7 at each time point
def per_helx(dssp, time):
    import numpy as np
    from scipy import stats

    char_num = np.arange(2,22,2)
    
    num = 0
    time_tot = int(len(dssp))
    
    per_ind_tot = round(time_tot/25)
    alpha_per = np.zeros([len(char_num), per_ind_tot])
    struct_per = np.zeros([len(char_num), per_ind_tot])

    #Determine helicity for all residues at each time point
    alpha_per_time = np.zeros(time_tot)
    
    #% helicity for each reaidue
    alpha_char = np.zeros(len(char_num))
    struct_char = np.zeros(len(char_num))

    for i in dssp:
        alpha = np.zeros(len(char_num))
        struct = np.zeros(len(char_num))

        char = i
        c = 0
        #Determine DSSP Values
        for n in char_num:
            if char[n]=='H':
                alpha[c] += 1
                alpha_char[c] += 1
            if char[n] == 'H' or char[0] == 'G' or char[0] == 'I':
                struct[c] += 1
                struct_char[c] += 1
            #Every 20 time steps take a running percentage
            if num % 25 == 0 and num != 0 and num < per_ind_tot*25:
                t = int(num/25)
                alpha_per[c][t] = 100 * alpha_char[c] / 25
                struct_per[c][t] = 100 * struct_char[c] / 25
                alpha_char[c] = 0
                struct_char[c] = 0
            c+=1
        alpha_per_time[num] = 100* sum(alpha)/(len(char_num))
        #Iterate time step
        num += 1
    
    #Determine overall percent for each residue
    alpha_per_mean = np.zeros(len(char_num))
    alpha_per_sem = np.zeros(len(char_num))
    struct_per_mean = np.zeros(len(char_num))
    struct_per_sem = np.zeros(len(char_num))

    for i in range(len(char_num)):
        alpha_per_mean[i] = np.mean(alpha_per[i][:])
        struct_per_mean[i] = np.mean(struct_per[i][:])
        alpha_per_sem[i] = stats.sem(alpha_per[i][:])
        struct_per_sem[i] = stats.sem(struct_per[i][:])

    if time == False:
        return alpha_per, struct_per, alpha_per_mean, struct_per_mean, alpha_per_sem, struct_per_sem
    else:
        return alpha_per_time

#Compute the running average for a given data set
#Input: x = dataset, w = interval by which to compute the running average
#Output: Vector for the running average
def moving_average(x, w):
    import numpy as np
    return np.convolve(x, np.ones(w), 'valid') / w

#Output: Uncorrelated RMSD values and array of uncorrelated frames
#Input: traj = MDTraj trajectory to compute the RMSD, ref = MDTraj reference structure
#Output:
#rmsd_uncorr = RMSD for each frame at each uncorrelated frame in the trajectory
#t = uncorreated time frames in trajectory
def compute_rmsd(traj, ref, t_full):
    import mdtraj as md
    import sys
    #Import custom modules
    sys.path.insert(1,'/ocean/projects/cts160011p/afriedma/code/PTP1B/util')
    import uncorr

    rmsd = md.rmsd(traj, ref, parallel=True, precentered=False)
    if t_full == False:
        t = uncorr.ind(rmsd)
    else:
        t = t_full
    rmsd_uncorr = uncorr.sort(rmsd, t)
    return rmsd_uncorr, t

#Function to compute the RMSD for a given section of the full protein sequence
#Input:
#ref, ref_top = MDTraj structure and topology for reference
#traj, top = MDTreaj trajectory and toplogy over which to compute RMSD
#sect = The two indices defining the section of interst to compute RMSD
#ref_type = If the reference is from centroid of self or Apo open or Apo closed
#name = Name of section of interest for file name
#ref_name = Name corresponding to the reference type above
#miss_first = Whether the first residue is missing and indices need to be subtracted by 1
#Output none but save file for uncorrelated samples
def compute_save_rmsd_sect(ref, ref_top, traj, top, sect, ref_type, name, ref_name, offset, t_full):
    import mdtraj as md
    import numpy as np

    #Seperate section for the reference and trajectory
    if sect[1] == 0:
        ref_sect = ref.atom_slice(ref_top.select(str(sect[0] - offset) + ' == resid')) #Limit trajectory to the section of choice
        traj_sect = traj.atom_slice(top.select(str(sect[0] - offset) + ' == resid')) #Limit trajectory to the section of choice
    else:
        ref_sect = ref.atom_slice(ref_top.select(str(sect[0] - offset) + ' <= resid and resid <= ' + str(sect[1] - offset))) #Limit trajectory to the section of choice
        traj_sect = traj.atom_slice(top.select(str(sect[0] - offset) + ' <= resid and resid <= ' + str(sect[1] - offset))) #Limit trajectory to the section of choice
    
    #Compute RMSD for section of interest
    rmsd_sect_uncorr, t_sect = compute_rmsd(traj_sect, ref_sect, t_full)
    
    #Save RMSD to file
    np.savetxt('rmsd_' + name + '_ref_' + str(ref_name) + '.txt', rmsd_sect_uncorr)

def pair_dist(traj_ns, hel_inter, t_full, offset, directory):
    import mdtraj as md
    import numpy as np
    import sys

    #Import custom modules
    sys.path.insert(1, directory + '/util/')
    import uncorr
   
    #Adjust for MDTraj Numbering and missing reisudes
    pairs, x = np.shape(hel_inter)
    hel_inter_name = np.zeros((pairs, x), dtype = int)
    for i in range(pairs):
        hel_inter_name[i][0] = round(hel_inter[i][0])
        hel_inter_name[i][1] = round(hel_inter[i][1])
        hel_inter[i][0] = hel_inter[i][0] - offset
        hel_inter[i][1] = hel_inter[i][1] - offset

    #Compute distance between all pairs
    [dist, pairs] = md.compute_contacts(traj_ns, contacts=hel_inter, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)
    
    num_pairs = len(pairs)

    for i in range(num_pairs):
        #Remove uncorrelated samples
        dist_i = dist[:,i]
        if t_full != False:
            dist_i_uncorr = uncorr.sort(dist_i, t_full)
        else:
            print('Uncorrelated frames not removed')
            dist_i_uncorr = dist_i
        
        #Save to file
        np.savetxt(str(hel_inter_name[i][0]) + '_' + str(hel_inter_name[i][1]) + '_inter.txt', np.asarray(dist_i_uncorr))

def hel_inter(traj_ns, group_A, group_B, t_dist, time_uncorr, pt1_ind, pt2_ind, pt3_ind, file_mean, hel1, hel2, directory):
    #Import modules
    import mdtraj as md
    import numpy as np
    import sys
    from itertools import product
    
    #Import custom modules
    sys.path.insert(1, directory + '/util/')
    import uncorr

    #Set pairs to compute distances
    pair = list(product(group_A, group_B))
    
    #Compute distances betwen helix pairs
    [dist_all, pairs] = md.compute_contacts(traj_ns, contacts=pair, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    
    #Compute number of time points and number of diatance pairs for following loops
    time, num_pairs = np.shape(dist_all)
    
    #Set new arrays with uncorrelated samples
    dist_uncorr = np.zeros((time_uncorr, num_pairs))
    for i in range(num_pairs):
        dist = dist_all[:,i]
        dist_uncorr[:,i] = uncorr.sort(dist, t_dist)
    
    #Loop through all time points
    inter_all = []
    inter_indv = np.zeros((num_pairs, time))
    if pt1_ind != 0 and pt2_ind != 0:
        inter_pt1, inter_pt2 = [],[]
    if pt3_ind != 0:
        inter_pt3 = []

    for i in range(time_uncorr):
        #Set all interaction counters to zero for each new time point
        check_inter = 0
        check_pt1 = 0
        check_pt2 = 0
        check_pt3 = 0

        #Loop through all residue pairs in the a3 and a6 helices
        for j in range(num_pairs): #Determine # of contacts b/w the helices
            if dist_uncorr[i][j] <= 0.5: #Count the contact if the residue distance is less than 0.5 nm
                check_inter += 1
                if pt1_ind != 0 and pt2_ind != 0:
                    if j in pt1_ind: #If this residue pair index is in part 1 of the helices
                        check_pt1 += 1
                    if j in pt2_ind: #If this residue pair index is in part 2 of the a6 and a7 helices
                        check_pt2 += 1
                if pt3_ind != 0:
                    if j in pt3_ind: #If this residue pair index is in part 3 of the a6 and a7 helices
                        check_pt3 += 1
                inter_indv[j][i] = 1 #Record presence of contact for given residue pair(j) and time(i)
        inter_all.append(check_inter) #Save the total number of residue contacts b/w the a3 and a6 helices to array
        if pt1_ind != 0 and pt2_ind != 0:
            inter_pt1.append(check_pt1)
            inter_pt2.append(check_pt2)
        if pt3_ind != 0:
            inter_pt3.append(check_pt3)

    np.savetxt(hel1 + '_' + hel2 + '_inter.txt', np.asarray(inter_all))
    if pt1_ind != 0 and pt2_ind != 0:
        np.savetxt(hel1 + '_' + hel2 + 'pt1_tot_inter.txt', np.asarray(inter_pt1))
        np.savetxt(hel1 + '_' + hel2 + 'pt2_tot_inter.txt', np.asarray(inter_pt2))
    if pt3_ind != 0:
        np.savetxt(hel1 + '_' + hel2 + 'pt3_tot_inter.txt', np.asarray(inter_pt3))
    
    #Save all individual interactions to file
    file_inters = open(hel1 + '_' + hel2 + '_inter_all.txt', 'w')
    for i in range(num_pairs):
        file_inters.write(str(100*sum(inter_indv[i,:])/time_uncorr) + '\n')
    
    #Save mean # of interactions over all time points
    file_mean.write(hel1 + '-' + hel2 + ' inters mean: ' + str(sum(inter_all)/time_uncorr) + '\n')


