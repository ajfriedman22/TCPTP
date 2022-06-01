#!/ usr / bin / env python
import math
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.decomposition import PCA
from itertools import combinations
import argparse
from itertools import product
from statistics import stdev
import sys
from mpl_toolkits import mplot3d

#Declare arguments
parser = argparse.ArgumentParser(description = 'Determination of DSSP, H-bonds, Ligand Contacts, protein and ligand RMSD, Helical interactions and PCA for GROMACS Trajectory of PTP1B')
parser.add_argument('-t', required=True, help='File name for input trajectory')
parser.add_argument('-g', required=True, help= 'File name for input topology (gro format)')
parser.add_argument('-e', required=False, default=True, type=bool, help= 'Is the input an equilibrated trajectory?')
parser.add_argument('-f', required=True, help='Base for all output file names')
parser.add_argument('-r', required=False, help='Should the reference structure for RMSD and RMSF be Apo Open or Closed or other?')
parser.add_argument('-rn', required=False, help='Reference name for RMSD')
parser.add_argument('-l', required=False, default='none', help='If ligand analysis should be performed, which ligand is present?')
parser.add_argument('-lref', required=False, default='none', help='Ligand reference for RMSD')
parser.add_argument('-a', required=False, default=False, type=bool, help='Should DSSP be calculated?')
parser.add_argument('-b', required=False, default=False, type=bool, help='Should Hbond Analysis be preformed?')
parser.add_argument('-i', required=False, default=False, type=bool, help='Should helical interactions be measured?')
parser.add_argument('-p', required=False, default=False, type=bool, help='Should PCA be preformed?')
parser.add_argument('-m', required=False, default=0, type=int, help='How many residues are missing from the N-terminus?')
parser.add_argument('-rms', required=False, default=False, type=bool, help='Should RMSF and RMSD analysis be computed?')
parser.add_argument('-d', required=True, type=str, help='Directory Path for PTP1B repository')

#Import Arguments
args = parser.parse_args()
File_traj = args.t + '.xtc'
File_gro = args.g + '.gro'
File_base = args.f
ref_type = args.r
dssp_check = args.a
hbond_check = args.b
equil_check = args.e
check_hel = args.i
pca_ck = args.p
rms_chk = args.rms
lig = args.l
lig_ref_pdb = args.lref
directory = args.d
miss_num = args.m
if lig != 'none':
    lig_check = True
else:
    lig_check = False

#Import custom modules
sys.path.insert(1, directory + '/util/')
import mdfunc
import uncorr
import plot

#Load trajectories
traj_bb, traj_prot, traj_ns, traj_a7 = mdfunc.mdtraj_load(File_traj, File_gro, [283, 294], miss_num)

#Determine if the full a7 helix is present (residues 287 to 295)
if traj_prot.n_residues > (295 - miss_num):
    a7_present = True
else:
    a7_present = False

#Load reference trajectory if RMSD analysis is requested
if rms_chk == True:
    #Establish name of reference structure
    if ref_type == 'Apo_open': #Reference for RMSD is centroid of Apo open trajectory
        ref = directory + 'analysis_scripts/RMSD_ref/Apo_open_bb_cluster.pdb'
        ref_name = 'open'
    elif ref_type == 'Apo_closed': #Reference for RMSD is centroid of Apo Closed Trajectory
        ref = directory + 'analysis_scripts/RMSD_ref/Apo_closed_bb_cluster.pdb'
        ref_name = 'closed'
    else:
        ref = ref_type
        ref_name = args.rn
    
    #Load reference PDB
    ref_pdb = md.load_pdb(ref)
    top_ref = ref_pdb.topology
    
    #Limit reference PDB to Protein bb atoms
    ref_bb = ref_pdb.atom_slice(top_ref.select('backbone'))

#If reference for ligand RMSD is provided load the file
if lig_ref_pdb != 'none':
    lig_rmsd_check = True
    lig_ref = md.load_pdb(lig_ref_pdb) #Load reference PDB file
    lig_ref_ns = lig_ref.atom_slice(lig_ref.topology.select('protein or resname ' + lig)) #Reduce reference PDB to protein and ligand only
else:
    lig_rmsd_check = False
print('Topology Loaded')

#If requested complete RMSD of protein bb atoms
if rms_chk == True:
    #Determine reference based on input
    if (ref_type == 'Apo_open' and a7_present == True and miss_first == False) or ref_name == 'self':
        #Calculate RMSF from reference structure
        rmsf_data = md.rmsf(traj_bb, ref_bb, parallel=True, precentered=False)

        #Calculate RMSD for full protein relative to reference structure
        rmsd_full_uncorr, t_full = mdfunc.compute_rmsd(traj_bb, ref_bb)
        
        #Only save RMSD and RMSF values to file for equilibrated input trajectories
        if equil_check == True:
            np.savetxt('rmsd_full_ref_' + str(ref_name) + '.txt', rmsd_full_uncorr) #save to text file
            np.savetxt('rmsf_ref_' + str(ref_name) + '.txt', rmsf_data) #save to text file
            np.savetxt('uncorrelated_frames.txt', t_full) #Save indices for uncorrelated frames to file

        #Delete unneeded arrays to save memory
        del rmsf_data; del rmsd_full_uncorr
    
    #Only complete section RMSD for equilibrated input trajectories
    if equil_check == True:
        #Set Topology for the backbone atoms only of the trajectory and reference structure
        top_bb = traj_bb.topology
        top_ref_bb = ref_bb.topology
    
        #Set sections of interest
        if a7_present == True: #Only compute these distances if the a7 helix is presenty
            sect_names = ['WPD', 'P', 'SBL', 'a3', 'a4', 'a5', 'a6', 'a7', 'L11', 'Q'] #Section names
            sect_res = np.array([[177, 185], [214, 223], [114, 121], [186, 200], [219, 237], [239, 252], [261, 278], [283, 294], [151, 154], [256, 260]])#Section start and end points

        else: #Omit sections which include the a7 helix
            sect_names = ['WPD', 'P', 'SBL', 'a3', 'a4', 'a5', 'a6', 'L11', 'Q'] #Section names
            sect_res = np.array([[177, 185], [214, 223], [114, 121], [186, 200], [219, 237], [239, 252], [261, 278], [151, 154], [256, 260]])#Section start and end points

        #Compute RMSD for all sections of interest and save to text file
        for i in range(len(sect_names)):
            mdfunc.compute_save_rmsd_sect(ref_bb, top_ref_bb, traj_bb, top_bb, sect_res[1,:], ref_type, sect_names[i], ref_name, miss_num)
    
    print('RMSD and RMSF Analysis Completed')

#Skip RMSD and RMSF analysis if input option not selected
else:
    print('RMSD and RMSF Analysis Skipped')

#Only do DSSP if input option is selected
if dssp_check == True:
    #Compute Phi and Psi angles for all residues in the a7 helix in all frames
    phi_ind, phi_angle = md.compute_phi(traj_a7, periodic = True, opt = True)
    psi_ind, psi_angle = md.compute_psi(traj_a7, periodic = True, opt = True)
    time, angles = np.shape(phi_angle) #determine the number of frames and the number of angles computed
    
    #Compute Secondary Structure of all Residues in the a7 helix using MDTraj function
    dssp_list = md.compute_dssp(traj_a7, simplified=False) #Compute DSSP for all residues in the a7 helix for all trajectory frames
    file_dssp = open('DSSP_'+ File_base + '.txt','w') #Create output file for DSSP and write over if file is present
   
    if rms_chk == False:
        print('WARNING: RMSD not calculated. Uncorrelated samples not removed!\n')
        phi_uncorr = phi_angle
        psi_uncorr = psi_angle
        
        dssp_uncorr = dssp_list
    else:
        #Limit to uncorrelated data based on the uncorrelated samples for bb RMSD of the full protein
        phi_uncorr = np.zeros((len(t_full), angles)) #Declare empty vectors for data
        psi_uncorr = np.zeros((len(t_full), angles))
        for i in range(angles): #Loop through all angles
            phi_uncorr[:,i] = uncorr.sort(phi_angle[:,i], t_full)
            psi_uncorr[:,i] = uncorr.sort(psi_angle[:,i], t_full)

        #Limit to Uncorrelated data
        frame_max,residue = dssp_list.shape #Determine the number of frames and residues for which DSSP analysis was completed
        dssp_uncorr = np.full((len(t_full) - 1, residue), None) #Declare an empty vector to input uncorrelated DSSP values for each residue
        for i in range(residue): #Loop through each residue seperately
            dssp_res = dssp_list[:,i] #Seperate all time values for a single residue
            dssp_res_mod = [] 
            for j in dssp_res:
                if j == ' ': #In DSSP a space designates a loop or irregular element
                    dssp_res_mod.append('L') #Subsitute an L for this designation to prevent issues with reading character values
                else: #If not a space keep the same character designation
                    dssp_res_mod.append(j)
            dssp_uncorr[:,i] = uncorr.char(dssp_res_mod, t_full)
    
    #Output DSSP to file
    frame_uncorr, residue = dssp_uncorr.shape
    for i in range(frame_uncorr): #Each row is a single frame
        for j in range(residue):#Each column is a residue in the a7 helix
            file_dssp.write(dssp_uncorr[i,j] + ' ')
        file_dssp.write('\n') #New line between each time frame
    file_dssp.close() #close file
    
    #Save psi and phi angles to files and overwrite if file is present
    np.savetxt('phi_' + File_base + '.txt', phi_uncorr)
    np.savetxt('psi_' + File_base + '.txt', psi_uncorr)
    
    #Delete arrays to save memory
    del phi_angle; del psi_angle; del phi_uncorr; del psi_uncorr; del dssp_list; del dssp_uncorr; del dssp_res_mod

    #Print to screen to notify that DSSP analysis is finished
    print('DSSP File Written')
    
#Skip DSSP Analysis if not requested
else:
    print('DSSP Skipped')

#H-bond determination
if hbond_check == True:
    #Determine list of H-bonds present in the trajectory for over 60% of the frames
    hbonds = md.baker_hubbard(traj_ns, freq=0.6, exclude_water=True, periodic=False)
    label = lambda hbond : '%s -- %s' % (traj_ns.topology.atom(hbond[0]), traj_ns.topology.atom(hbond[2])) #Extract labels for h-bonds
    np.savetxt('Hbonds_atom_' + File_base + '.txt', hbonds) #Save all atom indicies of h-bonds to file for further analysis
    
    #Write all h-bonds present for >60% of trajectory to file
    file_object = open('Hbonds_'+ File_base +'.txt', 'w') 
    for hbond in hbonds:
        file_object.write(label(hbond)) #Maintain same order as atom indicies
        file_object.write('\n')
    file_object.close() #close file

    #Determine the exact percentage of time that each h-bond present for >60% of the trajectory is formed
    per = [] #Declare empty array for percentage of time h-bond is formed
    da_distances = md.compute_distances(traj_ns, hbonds[:,[1,2]], periodic=False) #Compute distance between h-bond donor and acceptor
    da_angles = md.compute_angles(traj_ns, hbonds[:,:], periodic=False) #Compute angle between h-bond donor and acceptor
    [num_t, num_h] = np.shape(da_distances) #save values for number of frames(num_t) and number of bonds(num_b) to caculate
    for j in range(num_h): #Loop through all h-bonds
        count = 0 #Initialize count
        for i in range(num_t): #Loop through all frames
            if da_distances[i,j] <= 0.25 and da_angles[i,j] >= 2.094: #If distance between donor and acceptor is less than 2.5A and the angle is greater than 120 degrees or ~ 2.094 radians
                count +=1
        per.append(100*count/num_t) #Percentage of time each h-bond is present in trajectory
    np.savetxt('Hbonds_per_' + File_base + '.txt',per)

    #Delete unneededarrays to save memory
    del hbonds; del label; del per; del da_distances; del da_angles

    #If ligand analysis is requested determine h-bonds between ligand and PTP1B residues
    if lig_check == True:
        file_lig = open('Hbonds_lig_' +  File_base + '.txt', 'w') #Create file for list of h-bonds determined to be present more than 10% of the trajectory
        if lig == 'both':            
            ligand = traj_ns.topology.select('resname AD') #Select the ligand by name (based on topology) from the trajectory
            ligand2 = traj_ns.topology.select('resname BBR') #Select the ligand by name (based on topology) from the trajectory
        else:
            ligand = traj_ns.topology.select('resname ' + str(lig)) #Select the ligand by name (based on topology) from the trajectory
        protein = traj_ns.topology.select('protein') #Select protein from the trajectory
        hbonds = md.baker_hubbard(traj_ns, freq = 0.1, exclude_water=True, periodic=True) #Find all h-bonds present >10% of the trajectory
        label = lambda hbond : '%s -- %s' % (traj.topology.atom(hbond[0]), traj.topology.atom(hbond[2])) #Seperate the names of the h-bonds based on topology nomenclature
        if lig == 'both':
            file_lig.write('AD:\n')
        for hbond in hbonds: #search all h-bonds
            if (hbond[0] in ligand) and (hbond[2] in protein) or (hbond[2] in ligand) and (hbond[0] in protein): #seperate only those h-bonds which form between the ligand and the protein
                file_lig.write(str(label(hbond)) + '\n') #Write h-bond names to file
                file_lig.write(str(hbond[0]) + ' ' + str(hbond[1]) + ' ' + str(hbond[2]) + '\n') #Write atoms involved in h-bond to file
        if lig == 'both':
            file_lig.write('BBR:\n')
            for hbond in hbonds: #search all h-bonds
                if (hbond[0] in ligand2) and (hbond[2] in protein) or (hbond[2] in ligand2) and (hbond[0] in protein): #seperate only h-bonds which form between the ligand and the protein
                    file_lig.write(str(label(hbond)) + '\n') #Write h-bond names to file
                    file_lig.write(str(hbond[0]) + ' ' + str(hbond[1]) + ' ' + str(hbond[2]) + '\n') #Write atoms involved in h-bond to file
        file_lig.close() #close file
        #Delete arrays to save memory
        del protein; del ligand; del hbonds; del label
        if lig == 'both':
            del ligand2
    #Write to screen when code section is completed
    print('Hbond Analysis Written')

#Skip Hbond analysis if desired
else:
    print('Hbond Analysis Skipped')

###################################################   REVISE FROM HERE    ##################################################################################################################
#Residue pairs needed for either Ligand Interaction of Protein Interaction Analysis
if check_hel == True or lig_check == True:
    #Set Residue Pairs
    if lig != 'none':
        if lig == 'both':
            group_WPD, group_3, group_4, group_5, group_6, group_bend, group_7, group_L11, pair_other, group_l, group_l2 = mdfunc.set_sect(miss_first, lig, a7_present)
        else:
            group_WPD, group_3, group_4, group_5, group_6, group_bend, group_7, group_L11, pair_other, group_l, = mdfunc.set_sect(miss_first, lig, a7_present)
    else:
        group_WPD, group_3, group_4, group_5, group_6, group_bend, group_7, group_L11, pair_other = mdfunc.set_sect(miss_first, lig, a7_present)

#Compute Interactions between the a3, a6, and a7 helices
if check_hel == True:
    #Set pairs to compute distances
    pair_a3_a6 = list(product(group_3, group_6))
    pair_a3_WPD = list(product(group_3, group_WPD))
    pair_a3_bend = list(product(group_3, group_bend))
    if traj_ns.n_residues > 297: #Only include these pairs if the a7 is present in the trajectory
        pair_a7_a3 = list(product(group_3, group_7))
        pair_a7_a6 = list(product(group_6, group_7))
        pair_a7_L11 = list(product(group_L11, group_7))

    #Set indicies for sections of helices as described from their standard orientation for visualization
    a3_a7_pt1_ind = [5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 54, 55, 56, 57, 58, 59, 60, 67, 68, 69, 70, 71, 72, 73, 79, 80, 81, 82, 
            83, 84, 85, 91, 92, 93, 94, 95, 96, 97] #Upper region of both helices
    a3_a7_pt2_ind = [98, 99, 100, 101, 102, 110, 111, 112, 113, 114, 122, 123, 124, 125, 126, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 158, 159, 160, 161, 162, 170, 171, 172, 173, 
            174] #Lower region of both helices
    a6_a7_pt1_ind = [5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 54, 55, 56, 57, 58, 59, 60, 67, 68, 69, 70, 71, 72, 73, 79, 80, 81, 82, 
            83, 84, 85, 91, 92, 93, 94, 95, 96, 97, 103, 104, 105, 106, 107, 108, 109, 115, 116, 117, 118, 119, 120, 121] #Upper region of both helices
    a6_a7_pt2_ind = [122, 123, 124, 125, 126, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 158, 159, 160, 161, 162, 170, 171, 172, 173, 174, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 206, 
            207, 208, 209, 210] #Lower region of both Helices
    a6_a7_pt3_ind = [127, 128, 129, 130, 131, 132, 133, 139, 140, 141, 142, 143, 144, 145, 151, 152, 153, 154, 155, 156, 157, 163, 164, 165, 166, 167, 168, 169, 175, 176, 177, 178, 179, 180, 181, 187, 
            188, 189, 190, 191, 192, 193, 199, 200, 201, 202, 203, 204, 205, 211, 212, 213, 214, 215, 216, 217] #Upper region of a7 with lower region of the a6 helix

    #Compute distances for all pairs for closest atoms to compute contacts
    [dist_a3_a6_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a3_a6, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_a3_bend_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a3_bend, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_a3_WPD_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a3_WPD, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    if a7_present == True: #Only compute these distances if the a7 helix is present
        [dist_a7_a3_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7_a3, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist_a7_a6_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7_a6, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist_a7_L11_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7_L11, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    #Compute distances for all pairs for alpha carbonds to compute distances between residues
    [dist_ca_a3_WPD_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a3_WPD, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_ca_a3_bend_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a3_bend, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_ca_other_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_other, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)
    if a7_present == True: #Only compute these distances if the a7 helix is present
        [dist_ca_a7_a3_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7_a3, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist_ca_a7_a6_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7_a6, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist_ca_a7_L11_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7_L11, scheme='ca', ignore_nonprotein = False, periodic=True, soft_min = False)

    #Compute number of time points and number of diatance pairs for following loops
    time, num_pairs_a3_a6 = np.shape(dist_a3_a6_all)
    time, num_pairs_a3_bend = np.shape(dist_a3_bend_all)
    time, num_pairs_a3_WPD = np.shape(dist_a3_WPD_all)
    time, num_pairs_ca_other = np.shape(dist_ca_other_all)
    if a7_present == True:
        time, num_pairs_a7_a3 = np.shape(dist_a7_a3_all)
        time, num_pairs_a7_a6 = np.shape(dist_a7_a6_all)
        time, num_pairs_a7_L11 = np.shape(dist_a7_L11_all)
    
    if rms_chk == False:
        print('WARNING: RMSD not calculated. Uncorrelated Samples not removed!')
        t_dist = np.linspace(0, time, num = time)
        time_uncorr = time
    else:
        #Limit to uncorrelated samples
        t_dist = t_full #Determine indices of uncorrelated samples from RMSD of full trajectory
        time_uncorr = len(t_dist) - 1

    #Set open arrays for all samples
    dist_a3_a6 = np.zeros((time_uncorr, num_pairs_a3_a6))
    dist_a3_bend = np.zeros((time_uncorr, num_pairs_a3_bend))
    dist_ca_a3_bend = np.zeros((time_uncorr, num_pairs_a3_bend))
    dist_a3_WPD = np.zeros((time_uncorr, num_pairs_a3_WPD))
    dist_ca_a3_WPD = np.zeros((time_uncorr, num_pairs_a3_WPD))
    dist_ca_other = np.zeros((time_uncorr, num_pairs_ca_other))
    if a7_present == True:
        dist_a7_a3 = np.zeros((time_uncorr, num_pairs_a7_a3))
        dist_ca_a7_a3 = np.zeros((time_uncorr, num_pairs_a7_a3))
        dist_a7_a6 = np.zeros((time_uncorr, num_pairs_a7_a6))
        dist_ca_a7_a6 = np.zeros((time_uncorr, num_pairs_a7_a6))
        dist_a7_L11 = np.zeros((time_uncorr, num_pairs_a7_L11))
        dist_ca_a7_L11 = np.zeros((time_uncorr, num_pairs_a7_L11))
    
    #Set new arrays with uncorrelated samples
    for i in range(num_pairs_a3_a6):
        dist = dist_a3_a6_all[:,i]
        dist_a3_a6[:,i] = uncorr.sort(dist, t_dist)
    for i in range(num_pairs_a3_bend):
        dist = dist_a3_bend_all[:,i]
        dist_a3_bend[:,i] = uncorr.sort(dist, t_dist)
        dist = dist_ca_a3_bend_all[:,i]
        dist_ca_a3_bend[:,i] = uncorr.sort(dist, t_dist)
    for i in range(num_pairs_a3_WPD):
        dist = dist_a3_WPD_all[:,i]
        dist_a3_WPD[:,i] = uncorr.sort(dist, t_dist)
        dist = dist_ca_a3_WPD_all[:,i]
        dist_ca_a3_WPD[:,i] = uncorr.sort(dist, t_dist)
    for i in range(num_pairs_ca_other):
        dist = dist_ca_other_all[:,i]
        dist_ca_other[:,i] = uncorr.sort(dist, t_dist)
    if a7_present == True: #Only open these files if the a7 helix is present
        for i in range(num_pairs_a7_a3):
            dist = dist_a7_a3_all[:,i]
            dist_a7_a3[:,i] = uncorr.sort(dist, t_dist)
            dist = dist_ca_a7_a3_all[:,i]
            dist_ca_a7_a3[:,i] = uncorr.sort(dist, t_dist)
        for i in range(num_pairs_a7_a6):
            dist = dist_a7_a6_all[:,i]
            dist_a7_a6[:,i] = uncorr.sort(dist, t_dist)
            dist = dist_ca_a7_a6_all[:,i]
            dist_ca_a7_a6[:,i] = uncorr.sort(dist, t_dist)
        for i in range(num_pairs_a7_L11):
            dist = dist_a7_L11_all[:,i]
            dist_a7_L11[:,i] = uncorr.sort(dist, t_dist)
            dist = dist_ca_a7_L11_all[:,i]
            dist_ca_a7_L11[:,i] = uncorr.sort(dist, t_dist)

    #Open files for the number of interactions with each protein region at each point in time
    file_200_282 = open('200_282_dist.txt', 'w') #a3 to a6
    file_179_191 = open('179_191_dist.txt', 'w') #WPD to a3
    file_185_191 = open('185_191_dist.txt', 'w') #WPD to a3
    file_151_191 = open('151_191_dist.txt', 'w') #L11 to a3
    file_264_185 = open('264_185_dist.txt', 'w') #WPD to a6
    file_178_150 = open('178_150_dist.txt', 'w') #WPD to L11
    file_81_199 = open('81_199_dist.txt', 'w')
    file_117_182 = open('117_182_dist.txt', 'w') #WPD to SBL
    file_117_217 = open('117_217_dist.txt', 'w') #P to SBL
    file_192_225 = open('192_225_dist.txt', 'w') #a3 top to a5
    file_182_220 = open('182_220_dist.txt', 'w') #WPD tp P-loop
    file_187_269 = open('187_269_dist.txt', 'w') #a3 top to a6 top
    file_178_190 = open('178_190_dist.txt', 'w') #Start of WPD loop to side of a3
    file_152_177 = open('152_177_dist.txt', 'w') #Start of WPD loop to L11 loop
    file_mean = open('inter_mean.txt', 'w') #File for the mean number of interactions between any two given residues
    if a7_present == True: #Only open these files if the a7 helix is present
        file_200_287 = open('200_287_dist.txt', 'w') #a3 to a7 (bottom)
        file_276_292 = open('276_292_dist.txt', 'w') #a6 to a7 (middle)
        file_189_295 = open('189_295_dist.txt', 'w') #a3 to a7 (top)
        file_280_287 = open('280_287_dist.txt', 'w') #a6 to a7 (bottom)
        file_152_297 = open('152_297_dist.txt', 'w') #L11 to a7

    #Declare arrays to keep track of all contacts
    a3_a6_all, a3_a7_all, a6_a7_all, a7_L11_all = [],[],[],[] #Arrays for the total contacts made b/w helices
    a3_a7_pt1, a3_a7_pt2, a6_a7_pt1, a6_a7_pt2, a6_a7_pt3 = [],[],[],[],[] #Arrays for the total contacts made b/w sections of helices
    if a7_present == True: #Only create these arrays if the a7 helix is present
        inter_a3_a7 = np.zeros([num_pairs_a7_a3, time_uncorr]) #Array for the running total of interactions over time between each residue pair for tha a3 and a7 helices
        inter_a6_a7 = np.zeros([num_pairs_a7_a6, time_uncorr]) #Array for the running total of interactions over time between each residue pair for tha a3 and a7 helices
    
    #Loop through all time points
    for i in range(time_uncorr):
        #Set all interaction counters to zero for each new time point
        check_a3_a6 = 0
        check_a7_a3 = 0
        check_a7_a3_pt1 = 0
        check_a7_a3_pt2 = 0
        check_a7_a6 = 0
        check_a7_a6_pt1 = 0
        check_a7_a6_pt2 = 0
        check_a7_a6_pt3 = 0
        check_a7_L11 = 0

        #At each time point save the distance between each residue pair of interest
        file_179_191.write(str(dist_ca_a3_WPD[i][47]) + '\n')
        file_185_191.write(str(dist_ca_a3_WPD[i][53]) + '\n')
        file_200_282.write(str(dist_ca_a3_bend[i][70]) + '\n')
        file_151_191.write(str(dist_ca_other[i][0]) + '\n')
        file_264_185.write(str(dist_ca_other[i][1]) + '\n')
        file_178_150.write(str(dist_ca_other[i][2]) + '\n')
        file_81_199.write(str(dist_ca_other[i][3]) + '\n')
        file_117_182.write(str(dist_ca_other[i][4]) + '\n')
        file_117_217.write(str(dist_ca_other[i][5]) + '\n')
        file_192_225.write(str(dist_ca_other[i][6]) + '\n')
        file_182_220.write(str(dist_ca_other[i][7]) + '\n')
        file_187_269.write(str(dist_ca_other[i][8]) + '\n')
        file_178_190.write(str(dist_ca_other[i][9]) + '\n')
        file_152_177.write(str(dist_ca_other[i][10]) + '\n')

        if a7_present == True: #Only save these distances if the a7 helix is present
            file_200_287.write(str(dist_ca_a7_a3[i][168]) + '\n')
            file_189_295.write(str(dist_ca_a7_a3[i][44]) + '\n')
            file_276_292.write(str(dist_ca_a7_a6[i][149]) + '\n')
            file_280_287.write(str(dist_ca_a7_a6[i][192]) + '\n')
            file_152_297.write(str(dist_ca_a7_L11[i][34]) + '\n')
        
        #Loop through all residue pairs in the a3 and a6 helices
        for j in range(num_pairs_a3_a6): #Determine # of contacts b/w the a3 and a6 helices
            if dist_a3_a6[i][j] <= 0.5: #Count the contact if the residue distance is less than 0.5 nm
                check_a3_a6 += 1
        a3_a6_all.append(check_a3_a6) #Save the total number of residue contacts b/w the a3 and a6 helices to array
        if a7_present == True: #Only calculate these interactions if the a7 helix is present
            for k in range(num_pairs_a7_a3): #Determine # of contacts b/w the a3 and a7 helices
                if dist_a7_a3[i][k] <= 0.5:
                    check_a7_a3 += 1
                    inter_a3_a7[k][i] = 1 #Record presence of contact for given residue pair(k) and time(i)
                    if k in a3_a7_pt1_ind: #If this residue pair index is in part 1 of the a3 and a7 helices
                        check_a7_a3_pt1 += 1
                    if k in a3_a7_pt2_ind: #If this residue pair index is in part 2 of the a3 and a7 helices
                        check_a7_a3_pt2 += 1
            #Save to array/file for each time point
            a3_a7_all.append(check_a7_a3)
            a3_a7_pt1.append(check_a7_a3_pt1)
            a3_a7_pt2.append(check_a7_a3_pt2)
            for l in range(num_pairs_a7_a6): #Determine # of contacts b/w the a6 and a7 helices
                if dist_a7_a6[i][l] <= 0.5:
                    check_a7_a6 += 1
                    inter_a6_a7[l][i] = 1 #Record presence of contact for given residue pair(l) and time(i)
                    if l in a6_a7_pt1_ind: #If this residue pair index is in part 1 of the a6 and a7 helices
                        check_a7_a6_pt1 += 1
                    if l in a6_a7_pt2_ind: #If this residue pair index is in part 2 of the a6 and a7 helices
                        check_a7_a6_pt2 += 1
                    if l in a6_a7_pt3_ind: #If this residue pair index is in part 3 of the a6 and a7 helices
                        check_a7_a6_pt3 += 1
            #Save to array/file for each time point
            a6_a7_all.append(check_a7_a6)
            a6_a7_pt1.append(check_a7_a6_pt1)
            a6_a7_pt2.append(check_a7_a6_pt2)
            a6_a7_pt3.append(check_a7_a6_pt3)
            for m in range(num_pairs_a7_L11): #Determine # of contacts b/w the L11 loop and a7 helix
                if dist_a7_L11[i][m] <= 0.5:
                    check_a7_L11 += 1
            a7_L11_all.append(check_a7_L11)
    
    np.savetxt('a3_a6_inter.txt', np.asarray(a3_a6_all))
    if a7_present == True:
        #Save interactions for 2 part helix interactions
        np.savetxt('a3_a7_pt1_tot_inter.txt', np.asarray(a3_a7_pt1))
        np.savetxt('a6_a7_pt1_tot_inter.txt', np.asarray(a6_a7_pt1))
        np.savetxt('a3_a7_pt2_tot_inter.txt', np.asarray(a3_a7_pt2))
        np.savetxt('a6_a7_pt2_tot_inter.txt', np.asarray(a6_a7_pt2))
        np.savetxt('a6_a7_pt3_tot_inter.txt', np.asarray(a6_a7_pt3))
        np.savetxt('a7_a3_inter.txt', np.asarray(a3_a7_all))
        np.savetxt('a7_a6_inter.txt', np.asarray(a6_a7_all))
        np.savetxt('a7_L11_inter.txt', np.asarray(a7_L11_all))

        #Save all individual interactions to file
        file_a3_a7_inters = open('a3_a7_inter_all.txt', 'w')
        for i in range(num_pairs_a7_a3):
            file_a3_a7_inters.write(str(100*sum(inter_a3_a7[i,:])/time_uncorr) + '\n')
        file_a6_a7_inters = open('a6_a7_inter_all.txt', 'w')
        for j in range(num_pairs_a7_a6):
            file_a6_a7_inters.write(str(100*sum(inter_a6_a7[j,:])/time_uncorr) + '\n')
        #Save mean # of interactions over all time points
        file_mean.write('a6-a7 inters mean: ' + str(sum(a6_a7_all)/time_uncorr) + '\n')
        file_mean.write('a3-a7 inters mean: ' + str(sum(a3_a7_all)/time_uncorr))
    file_mean.write('a3-a6 inters mean: ' + str(sum(a3_a6_all)/time_uncorr) + '\n')
    
    #Close all files
    file_200_282.close()
    file_179_191.close()
    file_185_191.close()
    file_151_191.close()
    file_264_185.close()
    file_178_150.close()
    file_mean.close()
    if a7_present == True: #Only close files that were opened
        file_200_287.close()
        file_276_292.close()
        file_189_295.close()
        file_280_287.close()
        file_152_297.close()
    
    #Delete unneeded arrays
    del pair_a3_a6; del pair_a3_WPD; del pair_a3_bend; del dist_a3_a6_all; del dist_a3_bend_all; del dist_ca_a3_WPD_all; del dist_ca_a3_bend_all; del dist_ca_other_all
    del dist_a3_a6; del dist_a3_bend; del dist_ca_a3_WPD; del dist_ca_a3_bend; del dist_ca_other; del dist; del a3_a6_all
    if a7_present == True:
        del a3_a7_pt1_ind; del a3_a7_pt2_ind; del a6_a7_pt1_ind; del a6_a7_pt2_ind; del a6_a7_pt3_ind; del pair_a7_a3; del pair_a7_a6; del pair_a7_L11
        del dist_a7_a3_all; del dist_a7_a6_all; del dist_a7_L11_all; del dist_ca_a7_a3_all; del dist_ca_a7_a6_all; del dist_ca_a7_L11_all
        del dist_a7_a3; del dist_a7_a6; del dist_a7_L11; del dist_ca_a7_a3; del dist_ca_a7_a6; del dist_ca_a7_L11
        del a3_a7_all; del a6_a7_all; del a7_L11_all; del a3_a7_pt1; del a3_a7_pt2; del a6_a7_pt1; del a6_a7_pt2; del a6_a7_pt3; del inter_a3_a7; del inter_a6_a7

    print('Helix Interaction Analysis Completed')
else:
    print('Helix Interaction Analysis Skipped')

#Ligand RMSD analysis
if lig_rmsd_check == True:
    #Calculate distance between Ligand and res 193
    if miss_first == True:
        res = 191
        res2 = 275
    else:
        res = 192
        res2 = 274
    if lig == 'both':
        res_pair = np.zeros((4,2))
    else:
        res_pair = np.zeros((4,2))
    res_pair[0][0] = traj_ns.topology.select('resid ' + str(res) + ' and name CA')
    res_pair[1][0] = traj_ns.topology.select('resid ' + str(res2) + ' and name CA')
    if lig == 'both':
        res_pair[2][0] = traj_ns.topology.select('resid ' + str(res) + ' and name CA')
        res_pair[3][0] = traj_ns.topology.select('resid ' + str(res2) + ' and name CA')
    if lig == 'AD':
        res_pair[0][1] = traj_ns.topology.select('resname AD and name C5')
        res_pair[1][1] = res_pair[0][1]
    elif lig == 'BBR':
        res_pair[0][1] = traj_ns.topology.select('resname BBR and name S1')
        res_pair[1][1] = res_pair[0][1]
    else:
        res_pair[0][1] = traj_ns.topology.select('resname AD and name C5')
        res_pair[1][1] = res_pair[0][1]
        res_pair[2][1] = traj_ns.topology.select('resname BBR and name S1')
        res_pair[3][1] = res_pair[2][1]

    lig_dist = md.compute_distances(traj_ns, res_pair, periodic = True)
    
    if rms_chk == False:
        print('WARNING: RMSD not calculated. Uncorrelated samples not removed!')
        dist_lig_193 = lig_dist[:,0]
        dist_lig_283 = lig_dist[:,1]
        if lig == 'both':
            dist_lig2_193 = lig_dist[:,2]
            dist_lig2_283 = lig_dist[:,3]
    else:
        #Remove uncorrelated samples
        dist_lig_193 = uncorr.sort(lig_dist[:,0], t_full)
        dist_lig_283 = uncorr.sort(lig_dist[:,1], t_full)
        if lig == 'both':
            dist_lig2_193 = uncorr.sort(lig_dist[:,2], t_full)
            dist_lig2_283 = uncorr.sort(lig_dist[:,3], t_full)

    #Save to text file
    np.savetxt('dist_lig_193.txt', dist_lig_193)
    np.savetxt('dist_lig_276.txt', dist_lig_283)
    if lig == 'both':
        np.savetxt('dist_lig2_193.txt', dist_lig2_193)
        np.savetxt('dist_lig2_276.txt', dist_lig2_283)

    if lig == 'AD' or lig == 'BBR':
        #Align trajectory and referance to residues 1 to 200
        lig_traj_top = traj_ns.topology
        lig_ref_top = lig_ref.topology
        traj_cmplx = traj_ns.atom_slice(traj_ns.topology.select('protein or resname AD or resname BBR'))

        res_align = lig_traj_top.select('resid 185 to 199 or resid 264 to 280')
        
        #traj_ns_align = traj_ns.superpose(traj_a7, lig_ref, atom_indices = res_align, ref_atom_indices = res_align)
        traj_ns_align = traj_cmplx.superpose(lig_ref_ns, atom_indices = res_align)

        #seperate ligand carbon atoms
        lig_only_ref = lig_ref.atom_slice(lig_ref_top.select('(resname ' + lig + ')')) #reference
        lig_only_traj = traj_ns_align.atom_slice(lig_traj_top.select('(resname ' + lig + ')')) #trajectory

        lig_only_ref_top = lig_only_ref.topology
        lig_only_traj_top = lig_only_traj.topology
        
        #Compute COM of ligand
        com = md.compute_center_of_mass(lig_only_traj)
        com_ref = md.compute_center_of_mass(lig_only_ref)
        
        #Compute displacment
        time, dim = np.shape(com)
        displacment = np.zeros(time)
        for i in range(time):
            displacment[i] = (com[i][0] - com_ref[0][0])**2 + (com[i][1] - com_ref[0][1])**2 + (com[i][2] - com_ref[0][2])**2
        if rms_chk == False:
            print('WARNING: RMSD not calculated. Uncorrelated samples not removed!')
        else:
            displacment = uncorr.sort(displacment, t_full)
        rmsd = np.array([math.sqrt(np.mean(displacment))])
        if File_base == 'AD_BBR' and lig == 'BBR':
             np.savetxt('lig2_rmsd.txt', rmsd)#Save to file
        else:
            np.savetxt('lig_rmsd.txt', rmsd)#Save to file

    print('Ligand RMSD Analysis Complete')

#Ligand Interaction analysis
if lig_check == True:
    #Compute Ligand location
    pair_a3 = list(product(group_l, group_3))
    pair_a4 = list(product(group_l, group_4))
    pair_a5 = list(product(group_l, group_5))
    pair_a6 = list(product(group_l, group_6))
    pair_bend = list(product(group_l, group_bend))
    if lig == 'both':
        pair2_a3 = list(product(group_l2, group_3))
        pair2_a4 = list(product(group_l2, group_4))
        pair2_a5 = list(product(group_l2, group_5))
        pair2_a6 = list(product(group_l2, group_6))
        pair2_bend = list(product(group_l2, group_bend))

    if a7_present == True:
        pair_a7 = list(product(group_l, group_7))
        if lig == 'both':
            pair2_a7 = list(product(group_l2, group_7))

        #set up array for total number of contacts with each residue
        tot_pairs = len(pair_a3) + len(pair_a6) + len(pair_bend) + len(pair_a7)
        if lig == 'both':
            tot_pairs2 = len(pair2_a3) + len(pair2_a6) + len(pair2_bend) + len(pair2_a7)

    else:
        #set up array for total number of contacts with each residue
        tot_pairs = len(pair_a3) + len(pair_a6) + len(pair_bend)
        if lig == 'both':
            tot_pairs2 = len(pair2_a3) + len(pair2_a6) + len(pair2_bend)

    #Array for the total number of contacts during the trajectory for each residue in the helices with the ligand
    lig_tot_cont = np.zeros(tot_pairs)
    if lig == 'both':
        lig2_tot_cont = np.zeros(tot_pairs2)

    #compute distances
    [dist_a3_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a3, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_a4_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a4, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_a5_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a5, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_a6_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a6, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    [dist_bend_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_bend, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    if a7_present == True:
        [dist_a7_all, pairs] = md.compute_contacts(traj_ns, contacts=pair_a7, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
    if lig == 'both':
        [dist2_a3_all, pairs] = md.compute_contacts(traj_ns, contacts=pair2_a3, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist2_a4_all, pairs] = md.compute_contacts(traj_ns, contacts=pair2_a4, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist2_a5_all, pairs] = md.compute_contacts(traj_ns, contacts=pair2_a5, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist2_a6_all, pairs] = md.compute_contacts(traj_ns, contacts=pair2_a6, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        [dist2_bend_all, pairs] = md.compute_contacts(traj_ns, contacts=pair2_bend, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)
        if a7_present == True:
            [dist2_a7_all, pairs] = md.compute_contacts(traj_ns, contacts=pair2_a7, scheme='closest', ignore_nonprotein = False, periodic=True, soft_min = False)

    #Compute time points in which the distance to any residue is less than
    time, num_pairs_a3 = np.shape(dist_a3_all)
    time, num_pairs_a4 = np.shape(dist_a4_all)
    time, num_pairs_a5 = np.shape(dist_a5_all)
    time, num_pairs_a6 = np.shape(dist_a6_all)
    time, num_pairs_bend = np.shape(dist_bend_all)
    if a7_present == True:
        time, num_pairs_a7 = np.shape(dist_a7_all)

    if rms_chk == False:
        print('WARNING: RMSD not calculated. Uncorrelated Samples not removed!')
        t_dist = np.linspace(0, time, num = time)
        time_uncorr = time
    else:
        #Limit to uncorrelated samples
        t_dist = t_full #Determine indices of uncorrelated samples from RMSD of full trajectory
        time_uncorr = len(t_dist)

    #Set open arrays for all samples
    dist_a3 = np.zeros((time_uncorr, num_pairs_a3))
    dist_bend = np.zeros((time_uncorr, num_pairs_bend))
    dist_a4 = np.zeros((time_uncorr, num_pairs_a4))
    dist_a5 = np.zeros((time_uncorr, num_pairs_a5))
    dist_a6 = np.zeros((time_uncorr, num_pairs_a6))
    if a7_present == True:
        dist_a7 = np.zeros((time_uncorr, num_pairs_a7))
    if lig == 'both':
        dist2_a3 = np.zeros((time_uncorr, num_pairs_a3))
        dist2_bend = np.zeros((time_uncorr, num_pairs_bend))
        dist2_a4 = np.zeros((time_uncorr, num_pairs_a4))
        dist2_a5 = np.zeros((time_uncorr, num_pairs_a5))
        dist2_a6 = np.zeros((time_uncorr, num_pairs_a6))
        if a7_present == True:
            dist2_a7 = np.zeros((time_uncorr, num_pairs_a7))

    #Set new arrays with uncorrelated samples
    for i in range(num_pairs_a3):
        dist = dist_a3_all[:,i]
        dist_a3[:,i] = uncorr.sort(dist, t_dist)
        if lig == 'both':
            dist2 = dist2_a3_all[:,i]
            dist2_a3[:,i] = uncorr.sort(dist2, t_dist)

    for i in range(num_pairs_a4):
        dist = dist_a4_all[:,i]
        dist_a4[:,i] = uncorr.sort(dist, t_dist)
        if lig == 'both':
            dist2 = dist2_a4_all[:,i]
            dist2_a4[:,i] = uncorr.sort(dist2, t_dist)

    for i in range(num_pairs_a5):
        dist = dist_a5_all[:,i]
        dist_a5[:,i] = uncorr.sort(dist, t_dist)
        if lig == 'both':
            dist2 = dist2_a5_all[:,i]
            dist2_a5[:,i] = uncorr.sort(dist2, t_dist)

    for i in range(num_pairs_a6):
        dist = dist_a6_all[:,i]
        dist_a6[:,i] = uncorr.sort(dist, t_dist)
        if lig == 'both':
            dist2 = dist2_a6_all[:,i]
            dist2_a6[:,i] = uncorr.sort(dist2, t_dist)

    if traj_ns.n_residues > 297:
        for i in range(num_pairs_a7):
            dist = dist_a7_all[:,i]
            dist_a7[:,i] = uncorr.sort(dist, t_dist)
            if lig == 'both':
                dist2 = dist2_a7_all[:,i]
                dist2_a7[:,i] = uncorr.sort(dist2, t_dist)

    #Set array for the crystal structure binding location and two alternatives
    if lig == 'AD' or lig == 'both':
        contact_loc1, contact_loc2, contact_loc3, contact_loc4, contact_unb = [],[],[],[],[]
    if lig == 'BBR' or lig == 'both':
       crys_bound, unbound = [],[] 

    #Open files for the number of interactions with each protein region at each point in time
    file_a3 = open('a3_inter.txt', 'w')
    file_a4 = open('a4_inter.txt', 'w')
    file_a5 = open('a5_inter.txt', 'w')
    file_a6 = open('a6_inter.txt', 'w')
    if traj_ns.n_residues > 298:
        file_a7 = open('a7_inter.txt', 'w')
    if lig == 'both':
        file2_a3 = open('a3_lig2_inter.txt', 'w')
        file2_a4 = open('a4_lig2_inter.txt', 'w')
        file2_a5 = open('a5_lig2_inter.txt', 'w')
        file2_a6 = open('a6_lig2_inter.txt', 'w')
        if traj_ns.n_residues > 298:
            file2_a7 = open('a7_lig2_inter.txt', 'w')
    if lig == 'BBR' or lig == 'both':
        #Determine if BBR is extended or not
        top_ns = traj_ns.topology
        ligand = traj_ns.atom_slice(top_ns.select('resname BBR')) #Select the ligand by name (based on topology) from the trajectory
        BBR_rg = md.compute_rg(ligand)

    #Loop through all frames
    for i in range(time_uncorr):
        #At each time point reset counters for helix interactions
        check_a3 = 0
        check_a3_top = 0
        check_a4 = 0
        check_a5 = 0
        check_a6 = 0
        check_a6_mid = 0
        check_bend = 0
        check_a7 = 0
        check_a7_top = 0
        if lig == 'both':
            check2_a3 = 0
            check2_a4 = 0
            check2_a5 = 0
            check2_a6 = 0
            check2_bend = 0
            check2_a7 = 0

        #Index for residues interactions in a3, a6, and a7 + the bend with ligand
        bond = 0
        for j in range(num_pairs_a3): #Determine # of contacts with a3 helix
            if dist_a3[i][j] <= 0.5:
                check_a3 += 1
                if j < 7:
                    check_a3_top += 1
                lig_tot_cont[bond] += 1
            if lig == 'both' and dist2_a3[i][j] <= 0.5:
                check2_a3 += 1
                lig2_tot_cont[bond] += 1
            bond += 1
        for k in range(num_pairs_a4): #Determine # of contacts with a4 helix
            if dist_a4[i][k] <= 0.5:
                check_a4 += 1
            if lig == 'both' and dist2_a4[i][k] <= 0.5:
                check2_a4 += 1
        for l in range(num_pairs_a5):  #Determine # of contacts with a5 helix
            if dist_a5[i][l] <= 0.5:
                check_a5 += 1
            if lig == 'both' and dist2_a5[i][l] <= 0.5:
                check2_a5 += 1
        for m in range(num_pairs_a6):  #Determine # of contacts with a6 helix
            if dist_a6[i][m] <= 0.5:
                check_a6 += 1
                if m < 8:
                    check_a6_mid += 1
                lig_tot_cont[bond] += 1
            if lig == 'both' and dist2_a6[i][m] <= 0.5:
                check2_a6 += 1
                lig2_tot_cont[bond] += 1
            bond += 1
        for o in range(num_pairs_bend): #Determine the # of contacts between a6 and a7 helices
            if dist_bend[i][o] <= 0.5:
                check_bend += 1
                lig_tot_cont[bond] += 1
            if lig == 'both' and dist2_bend[i][o] <= 0.5:
                check2_bend += 1
                lig2_tot_cont[bond] += 1
            bond += 1
    
        #Output number of interactions to files for a3-a6 at each time point
        file_a3.write(str(check_a3) + '\n')
        file_a4.write(str(check_a4) + '\n')
        file_a5.write(str(check_a5) + '\n')
        file_a6.write(str(check_a6) + '\n')
        if lig == 'both':
            file2_a3.write(str(check2_a3) + '\n')
            file2_a4.write(str(check2_a4) + '\n')
            file2_a5.write(str(check2_a5) + '\n')
            file2_a6.write(str(check2_a6) + '\n')
        
        if traj_prot.n_residues > 297:
            for n in range(num_pairs_a7):#Determine # of contacts with the a7 helix pt1
                if dist_a7[i][n] <= 0.5:
                    check_a7 += 1
                    if n > 8:
                        check_a7_top += 1
                    lig_tot_cont[bond] += 1
                bond += 1
            #Output number of interactions to file for a7
            file_a7.write(str(check_a7) + '\n')

        if lig == 'AD' or lig == 'both': 
            if traj_prot.n_residues > 297:
               #Determine ligand binding Location
                #Sum total interactions for each binding location
                total_contact_loc1 = check_a3 + check_a7 #Crystal structure binding location
                total_contact_loc2 = check_a6 + check_bend + check_a7 #Alt loc 1
                total_contact_loc3 = check_a4 + check_a5 + check_a6 #alt loc 2
                total_contact_loc4 = check_a3 + check_a4 + check_a6 #Alt loc 3
           
                #Binding location 1
                if check_a3 >= 1 and check_a7 >= 1 and total_contact_loc1 >= 4 and check_a6_mid == 0 and check_a4 == 0 and check_a5 == 0:
                    contact_loc1.append(1)
                    loc1 = True
                else:
                    contact_loc1.append(0)
                    loc1 = False

                #Binding Location 2
                if check_a6 >= 1 and check_a7 >= 1 and check_bend >= 1 and total_contact_loc2 >= 3 and check_a4 == 0 and check_a5 == 0 and check_a3 < 2 and loc1 == False:
                    contact_loc2.append(1)
                else:
                    contact_loc2.append(0)

            else:
                #Sum total interactions without the a7 helix
                total_contact_loc1 = check_a3 + check_a6
                total_contact_loc2 = check_a6 + check_bend
                total_contact_loc3 = check_a4 + check_a5 + check_a6 
                total_contact_loc4 = check_a3 + check_a4 + check_a6

                #Binding Location 1 w/o a7
                if check_a3 >= 1 and total_contact_loc1 >= 5 and check_a6 == 0 and check_a5 == 0 and check_a4 == 0:
                    contact_loc1.append(1)
                    loc1 = True
                else:
                    contact_loc1.append(0)
                    loc1 = False

                #Binding location 2 w/o a7
                if check_bend >= 1 and check_a6 >= 1 and total_contact_loc2 >= 5 and check_a4 == 0 and check_a5 == 0 and loc1 == False:
                    contact_loc2.append(1)
                else:
                    contact_loc2.append(0)

            #Binding Location 3
            if check_a4 >= 1 and check_a6 >= 1 and check_a3 >= 1 and total_contact_loc3 >= 5 and check_a7 == 0:
                contact_loc3.append(1)
            else:
                contact_loc3.append(0)

            #Binding Location 4
            if check_a4 >= 1 and check_a6 >= 1 and total_contact_loc4 >= 5 and check_a3 == 0 and check_a7 == 0:
                contact_loc4.append(1)
            else:
                contact_loc4.append(0)

            if check_a3 == 0 and check_a6 == 0 and check_a7 == 0 and check_a4 == 0:
                contact_unb.append(1)
            else:
                contact_unb.append(0)
 
    #Print % contact with each bond
    output_bond_cont = open('all_iter_frac.txt', 'w')
    output_bond_cont2 = open('all_iter_frac2.txt', 'w')

    #Set array for names of residues interacting with ligand
    contacts = np.append(group_3, group_6)
    contacts = np.append(contacts, group_bend)
    contacts = np.append(contacts, group_7)
    for i in range(tot_pairs):
        output_bond_cont.write(str(contacts[i]) + ' ' + str(lig_tot_cont[i]/time_uncorr) + '\n') #Write name of residue and fraction of time in contact fot full trajectory to file
    output_bond_cont.close() #Close File

    if lig == 'both':
        for i in range(tot_pairs):
            output_bond_cont2.write(str(contacts[i]) + ' ' + str(lig2_tot_cont[i]/time_uncorr) + '\n') #Write name of residue and fraction of time in contact fot full trajectory to file
        output_bond_cont2.close() #Close File

    if lig == 'AD' or lig == 'both':
        #Determine the fraction the ligand is in each location
        lig_frac = np.array([sum(contact_loc1)/len(contact_loc1), sum(contact_loc2)/len(contact_loc2), sum(contact_loc3)/len(contact_loc3), sum(contact_loc4)/len(contact_loc4), 
            sum(contact_unb)/len(contact_unb)])
        lig_other = 1 - sum(lig_frac)
        lig_frac = np.append(lig_frac, lig_other)

        #Print the percent time in each location
        Loc_frac = open('AD_bind_frac' + File_base + '.txt', 'w')
        plot.write_lig_bind(lig_frac, Loc_frac, 'AD')

        #Determine transitions between loc1 and loc2
        if sum(contact_loc1) != 0:
            trans_1_2, trans_2_1 = [],[] #Empty array for indices of transitions
            for i in range(1, len(contact_loc1)):
                if contact_loc1[i-1] == 1 and contact_loc2[i] == 1: #transition from loc1 to loc2
                    trans_1_2.append(i)
                if contact_loc2[i-1] == 1 and contact_loc1[i] == 1: #transition from loc2 to loc1
                    trans_2_1.append(i)
    
            #Determine time between transitions
            num_trans = min([len(trans_1_2), len(trans_2_1)])
            trans_time = np.zeros(num_trans)
            for i in range(num_trans):
                trans_time[i] = abs(trans_1_2[i] - trans_2_1[i])

            #Save transition times to file
            np.savetxt('AD_trans_time.txt', trans_time)

            #Delete unnedded arrays
            del trans_1_2; del trans_2_1; del trans_time
    
    #Compute Simultaneous Ligand Contacts
    if a7_present == True:
        pairs = num_pairs_a3 + num_pairs_a6 + num_pairs_a7

    else:
        pairs = num_pairs_a3 + num_pairs_a6
        num_pairs_a7 = 0
        dist_a7 = []

    simul_contacts = np.zeros([pairs, pairs])

    simul_contacts, num1 = mdfunc.compute_simul_comtacts(num_pairs_a3, num_pairs_a6, num_pairs_a7, time_uncorr, dist_a3, dist_a6, dist_a7, 0, simul_contacts, a7_present)
    simul_contacts, num1 = mdfunc.compute_simul_comtacts(num_pairs_a6, num_pairs_a3, num_pairs_a7, time_uncorr, dist_a6, dist_a3, dist_a7, num1, simul_contacts, a7_present)
    simul_contacts, num1 = mdfunc.compute_simul_comtacts(num_pairs_a7, num_pairs_a6, num_pairs_a3, time_uncorr, dist_a7, dist_a6, dist_a3, num1, simul_contacts, a7_present)
    
    #Save array to test file
    np.savetxt('simul_lig_contact_' + File_base + '.txt', simul_contacts)

    if lig == 'both':
        simul_contacts2 = np.zeros([pairs, pairs])
        simul_contacts2, num1 = mdfunc.compute_simul_comtacts(num_pairs_a3, num_pairs_a6, num_pairs_a7, time_uncorr, dist2_a3, dist2_a6, dist2_a7, 0, simul_contacts2, a7_present)
        simul_contacts2, num1 = mdfunc.compute_simul_comtacts(num_pairs_a6, num_pairs_a3, num_pairs_a7, time_uncorr, dist2_a6, dist2_a3, dist2_a7, num1, simul_contacts2, a7_present)
        simul_contacts2, num1 = mdfunc.compute_simul_comtacts(num_pairs_a7, num_pairs_a6, num_pairs_a3, time_uncorr, dist2_a7, dist2_a6, dist2_a3, num1, simul_contacts2, a7_present)
        
        #Save array to text file
        np.savetxt('simul_lig2_contact_' + File_base + '.txt', simul_contacts2)

    #Simultaneour Ligand Contacts for Sections
    lig_cont_sect = np.zeros([9,9])
    if lig == 'both':
        lig2_cont_sect = np.zeros([9,9])
    
    #Create an empty vector for ligand contacts
    lig_contacts = np.zeros(9)
    if lig == 'both':
        lig2_contacts = np.zeros(9)

    #Loop through all time points
    for t in range(time_uncorr):
        #Residue 186 to 192
        lig_contacts[0] = mdfunc.sect_contact(dist_a3, t, 0, 6)
        #Residue 193 to 196
        lig_contacts[1] = mdfunc.sect_contact(dist_a3, t, 7, 10)
        #Residue 197 to 200
        lig_contacts[2] = mdfunc.sect_contact(dist_a3, t, 11, 14)
        #Residue 264 to 270
        lig_contacts[3] = mdfunc.sect_contact(dist_a6, t, 0, 6)
        #Residue 193 to 196
        lig_contacts[4] = mdfunc.sect_contact(dist_a6, t, 7, 11)
        #Residue 197 to 200
        lig_contacts[5] = mdfunc.sect_contact(dist_a6, t, 12, 16)
        
        if a7_present == True:
            #Residue 287 to 290
            lig_contacts[6] = mdfunc.sect_contact(dist_a7, t, 0, 3)
            #Residue 291 to 294
            lig_contacts[7] = mdfunc.sect_contact(dist_a7, t, 4, 7)
            #Residue 295 to 298
            lig_contacts[8] = mdfunc.sect_contact(dist_a7, t, 8, 11)
        
        #Make matrix for simultaneous contacts
        for i in range(len(lig_contacts)):
            for j in range(len(lig_contacts)):
                if lig_contacts[i] == 1 and lig_contacts[j] == 1:
                    lig_cont_sect[i][j] += 1
                    lig_cont_sect[j][i] += 1

        if lig == 'both':
            #Residue 186 to 192
            lig_contacts[0] = mdfunc.sect_contact(dist_a3, t, 0, 6)
            #Residue 193 to 196
            lig_contacts[1] = mdfunc.sect_contact(dist_a3, t, 7, 10)
            #Residue 197 to 200
            lig_contacts[2] = mdfunc.sect_contact(dist_a3, t, 11, 14)
            #Residue 264 to 270
            lig_contacts[3] = mdfunc.sect_contact(dist_a6, t, 0, 6)
            #Residue 193 to 196
            lig_contacts[4] = mdfunc.sect_contact(dist_a6, t, 7, 11)
            #Residue 197 to 200
            lig_contacts[5] = mdfunc.sect_contact(dist_a6, t, 12, 16)
        
            if a7_present == True:
                #Residue 287 to 290
                lig_contacts[6] = mdfunc.sect_contact(dist_a7, t, 0, 3)
                #Residue 291 to 294
                lig_contacts[7] = mdfunc.sect_contact(dist_a7, t, 4, 7)
                #Residue 295 to 298
                lig_contacts[8] = mdfunc.sect_contact(dist_a7, t, 8, 11)
        
            #Make matrix for simultaneous contacts
            for i in range(len(lig_contacts)):
                for j in range(len(lig_contacts)):
                    if lig_contacts[i] == 1 and lig_contacts[j] == 1:
                        lig_cont_sect[i][j] += 1
                        lig_cont_sect[j][i] += 1

    #Calculate the percentage that each interaction was present over the trajectory and save to text file
    lig_cont_sect_per = 100 * (lig_cont_sect/time_uncorr)
    np.savetxt('simul_lig_contact_sect' + File_base + '.txt', lig_cont_sect_per)

    if lig == 'both':
        lig2_cont_sect_per = 100 * (lig2_cont_sect/time_uncorr)
        np.savetxt('simul_lig2_contact_sect' + File_base + '.txt', lig2_cont_sect_per)
    
    #Delete unneeded arrays
    del pair_a3; del pair_a4; del pair_a5; del pair_a6; del pair_bend; del lig_tot_cont; 
    del dist_a3_all; del dist_a4_all; del dist_a5_all; del dist_a6_all; del dist_bend_all
    del dist_a3; del dist_a4; del dist_a5; del dist_a6; del dist_bend; del dist
    if lig == 'AD' or lig == 'both':
        del contact_loc1; del contact_loc2; del contact_loc3; del contact_loc4; del contacts
    del simul_contacts; del lig_contacts; del lig_cont_sect

    if a7_present == True:
        del pair_a7; del dist_a7_all; del dist_a7; 
    
    if lig == 'both':
        del dist2_a3_all; del dist2_a4_all; del dist2_a5_all; del dist2_a6_all; del dist2_bend_all
        del dist2_a3; del dist2_a4; del dist2_a5; del dist2_a6; del dist2_bend; del dist2
        del simul_contacts2; del lig2_contacts

        if a7_present == True:
            del dist2_a7_all; del dist2_a7; 

    print('Ligand Interaction Analysis Complete')
else:
    print('Ligand Interaction Analysis Skipped')

if pca_ck == True:
    #Principle Component Analysis
    pca = PCA(n_components=10) #initialize
    traj_bb.superpose(traj_bb, 0) #align trajectory
    reduced_cartesian = pca.fit_transform(traj_bb.xyz.reshape(traj_bb.n_frames, traj_bb.n_atoms * 3))
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals =2)
    accum_per_var = [ i for i in [ np . sum ( per_var [: j ]) for j in range (1 ,11)]]

    #Plot PCA
    fig = plt.figure()
    plt.scatter(reduced_cartesian[:, 0], reduced_cartesian[:,1], marker='x', c=traj.time)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Cartesian coordinate PCA for '+File_base)
    cbar = plt.colorbar()
    cbar.set_label('Time [ps]')
    fig.savefig('PCA_'+File_base+'.png')
    plt.close(fig)

    fig.ax = plt.figure(figsize=(10,5.5))

    #Scree Plot
    labels = [ 'PC' + str(x) for x in range(1 , len(per_var) +1) ]
    ax1 = plt.subplot (1 ,2 ,1)
    ax2 = ax1.twinx ()
    ax1.bar( x = range(1 , len(per_var) +1) , height = per_var , tick_label = labels , alpha = 0.85)
    ax2.plot(range(1 , len(per_var) +1), accum_per_var, color = 'r' , marker = 'o')
    ax2.grid( True )
    xlocs , xlabs = plt.xticks()
    ax1.set_ylabel ( 'Percentage of explained variance (%)', size = '12')
    ax2.set_ylabel ( 'Accumulated explained variance (%)', size = '12')
    ax1.set_xlabel ( 'Principal Components' , size = '12')
    ax1.set_ylim ([0 , max(per_var)*1.1 ] )
    ax2.set_ylim ([0 , max(accum_per_var)*1.1 ] )
    plt.title ( 'Scree Plot' , size = '14')
    plt.grid ( True )
    plt.savefig('Scree_'+File_base+'.png')
    print('PCA Completed')

if pca_ck == False:
    print('PCA Skipped')
