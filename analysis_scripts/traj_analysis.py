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
import os.path

def lig_dist_uncorr(num_pairs, dist_all, lig, dist2_all, t_dist):
    time_uncorr = len(t_dist)
    dist_uncorr = np.zeros((time_uncorr, num_pairs))
    if lig == 'both':
        dist2_uncorr = np.zeros((time_uncorr, num_pairs))
    for i in range(num_pairs):
        dist = dist_all[:,i]
        dist_uncorr[:,i] = uncorr.sort(dist, t_dist)
        if lig == 'both':
            dist2 = dist2_all[:,i]
            dist2_uncorr[:,i] = uncorr.sort(dist2, t_dist)
    if lig == 'both':
        return dist_uncorr, dist2_uncorr
    else:
        return dist_uncorr

def lig_hel_inter(num_pairs, dist, dist2, n, lig_tot_cont, lig, lig2_tot_cont, bond):
    check = 0
    check2 = 0
    for j in range(num_pairs): #Determine # of contacts with a3 helix
            if dist[i][j] <= 0.5:
                check += 1
                if j < n:
                    check_top += 1
                lig_tot_cont[bond] += 1
            if lig == 'both' and dist2[i][j] <= 0.5:
                check2 += 1
                lig2_tot_cont[bond] += 1
            bond += 1
    if lig == 'both' and n > 0:
        return check, check2, check_top, lig_tot_cont, lig2_tot_cont, bond
    elif lig == 'both' and n == 0:
        return check, check2, lig_tot_cont, lig2_tot_cont, bond
    elif lig != 'both' and n > 0:
        return check, check_top, lig_tot_cont, bond
    elif lig != 'both' and n == 0:
        return check, lig_tot_cont, bond

#Declare arguments
parser = argparse.ArgumentParser(description = 'Determination of DSSP, H-bonds, Ligand Contacts, protein and ligand RMSD, Helical interactions and PCA for GROMACS Trajectory of TCPTP')
parser.add_argument('-t', required=True, help='File name for input trajectory')
parser.add_argument('-g', required=True, help= 'File name for input topology (gro format)')
parser.add_argument('-e', required=False, default=0, type=int, help= 'If input trajectory is equilibrated input equilibration time otherwise input 0(ns)')
parser.add_argument('-f', required=True, help='Base for all output file names')
parser.add_argument('-r', required=False, help='Should the reference structure for RMSD and RMSF be Apo Open or Closed or other?')
parser.add_argument('-rn', required=False, help='Reference name for RMSD')
parser.add_argument('-l', required=False, default='none', help='If ligand analysis should be performed, which ligand is present?')
parser.add_argument('-lref', required=False, default='none', help='Ligand reference for RMSD')
parser.add_argument('-a', required=False, default=False, type=bool, help='Should DSSP be calculated?')
parser.add_argument('-b', required=False, default=False, type=bool, help='Should full Hbond Analysis be preformed?')
parser.add_argument('-bn', required=False, default=False, type=bool, help='Should Hbond Network Analysis be preformed?')
parser.add_argument('-i', required=False, default=False, type=bool, help='Should helical interactions be measured?')
parser.add_argument('-hd', required=False, default=False, type=bool, help='Should individual helical distances be measured?')
parser.add_argument('-input', required=False, default=False, type=str, help='Provide input file for residue pairs. Otherwise use default.')
parser.add_argument('-p', required=False, default=False, type=bool, help='Should PCA be preformed?')
parser.add_argument('-rms', required=False, default=False, type=bool, help='Should RMSF and RMSD analysis be computed?')
parser.add_argument('-w', required=False, default=False, type=bool, help='Should WPD loop analysis be completed?')
parser.add_argument('-d', required=True, type=str, help='Directory Path for PTP1B repository')
parser.add_argument('-n', required=False, default = 300, type=int, help='Total length of trajectory(ns)')
parser.add_argument('-m', required=False, default = 0, type=int, help='Number of Missing Residues')

#Import Arguments
args = parser.parse_args()
File_traj = args.t
if File_traj.split('.')[-1] != 'xtc': #Add file extension if not in input
    File_traj = File_traj + '.xtc'
File_gro = args.g
if File_gro.split('.')[-1] != 'gro': #Add default file extension if not in input
    File_gro = File_gro + '.gro'
File_base = args.f
ref_type = args.r
dssp_check = args.a
hbond_check = args.b
hbond_net_check = args.bn
eq_time = args.e
check_hel = args.i
check_hel_indv = args.hd
input_inter = args.input
pca_ck = args.p
wpd_ck = args.w
rms_chk = args.rms
lig = args.l
lig_ref_pdb = args.lref
directory = args.d
traj_time = args.n
offset = args.m + 1
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
traj_bb, traj_prot, traj_ns, traj_a7, miss_first = mdfunc.mdtraj_load(File_traj, File_gro, [284, 294]) 

#Determine if the full a7 helix is present (residues 287 to 295)
if traj_prot.n_residues > 294:
    a7_present = True
else:
    a7_present = False

#Print to output if dealing with an equilibrated trajectory or not
if eq_time == 0:
    name_add = '_full'
    print('Unequilibrated Trajectory!')
else:
    name_add = ''
    print('Processing equilibrated trajectory')

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
        if ref_type.split('.')[-1] != 'pdb': #Add default file extension if not in input
            ref = ref_type + '.pdb'
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
    if traj_bb.n_residues == ref_bb.n_residues:
        #Calculate RMSF from reference structure
        rmsf_data = md.rmsf(traj_bb, ref_bb, parallel=True, precentered=False)

        #Calculate RMSD for full protein relative to reference structure
        rmsd_full_uncorr, t_full = mdfunc.compute_rmsd(traj_bb, ref_bb, False)
        
        #Determine time in ns for each uncorrelated frame
        frame_per_ns = traj_bb.n_frames/traj_time
        uncorr_time = np.zeros(len(t_full))
        for i in range(len(t_full)):
            uncorr_time[i] = t_full[i]/frame_per_ns + eq_time

        #Only save RMSD and RMSF values to file for equilibrated input trajectories
        if eq_time != 0 and ref_name == 'self':
            np.savetxt('rmsd_full_ref_' + str(ref_name) + '.txt', rmsd_full_uncorr) #save to text file
            np.savetxt('rmsf_ref_' + str(ref_name) + '.txt', rmsf_data) #save to text file
        np.savetxt('uncorrelated_frames' + name_add + '.txt', t_full) #Save indices for uncorrelated frames to file
        np.savetxt('uncorrelated_time' + name_add + '.txt', uncorr_time) #Save time for uncorrelated frames to file

        #Delete unneeded arrays to save memory
        del rmsf_data; del rmsd_full_uncorr
    else:
        t_full = False
    #Only complete section RMSD for equilibrated input trajectories
    if eq_time != 0:
        #Set Topology for the backbone atoms only of the trajectory and reference structure
        top_bb = traj_bb.topology
        top_ref_bb = ref_bb.topology
    
        #Set sections of interest
        if a7_present == True: #Only compute these distances if the a7 helix is presenty
            sect_names = ['WPD', 'WPD_a3', 'P', 'CYS', 'WPD_181', 'SBL', 'a3', 'a3_top', 'a4', 'a5', 'a6', 'a6_bot', 'a7', 'L11', 'Q'] #Section names
            sect_res = np.array([[179, 187], [187, 190], [214, 221], [216, 0], [182, 0], [115, 120], [188, 202], [188, 193], [221, 238], [245, 254], [262, 277],
                [273, 277], [286, 294], [149, 152], [258, 262]])#Section start and end points

        else: #Omit sections which include the a7 helix
            sect_names = ['WPD', 'WPD_a3', 'P', 'CYS', 'WPD_181', 'SBL', 'a3', 'a3_top', 'a4', 'a5', 'a6', 'a6_bot', 'L11', 'Q'] #Section names
            sect_res = np.array([[179, 187], [187, 190], [214, 221], [216, 0], [182, 0], [115, 120], [188, 202], [188, 193], [221, 238], [245, 254], [262, 277],
                [273, 277], [149, 152], [258, 262]])#Section start and end points

        #Compute RMSD for all sections of interest and save to text file
        for i in range(len(sect_names)):
            mdfunc.compute_save_rmsd_sect(ref_bb, top_ref_bb, traj_bb, top_bb, sect_res[i,:], ref_type, sect_names[i], ref_name, offset, t_full)
    
    print('RMSD and RMSF Analysis Completed')

#Skip RMSD and RMSF analysis if input option not selected
else:
    print('RMSD and RMSF Analysis Skipped')
    t_full = False

#WPD Loop Analysis
if wpd_ck == True:
    #Determine distance between catalytic residues 181 and 215
    res1 = 182 - offset
    res2 = 216 - offset 
    
    residues = np.zeros((1,2))
    
    top = traj_ns.topology
   
    residues[0][0] = top.select('resid ' + str(res1) + ' and name CA') #Select protein from the trajectory select residue 181
    residues[0][1] = top.select('resid ' + str(res2) + ' and name CA') #Select protein from the trajectory select residue 215
    
    WPD_dist = md.compute_distances(traj_ns, residues, periodic=False) #Compute distance between h-bond donor and acceptor
    
    #Remove uncorrelated samples
    if rms_chk == False:
        if os.path.exists('uncorrelated_frames.txt'):
            lines = open('uncorrelated_frames.txt', 'r').readlines()
            t_full = []
            for i in lines:
                t_full.append(float(i.strip()))
            WPD_uncorr_dist = uncorr.sort(WPD_dist, t_full)
        else:
            print('WARNING: Uncorrelated samples not removed!')
            WPD_uncorr_dist = WPD_dist
    else:
        WPD_uncorr_dist = uncorr.sort(WPD_dist, t_full)
    
    #Save to file
    np.savetxt('WPD_dist' + name_add + '.txt', WPD_uncorr_dist)
    
    print('WPD Loop Analysis Completed')
else:
    print('WPD Loop Analysis Skipped')

#Only do DSSP if input option is selected
if dssp_check == True:
    #Compute Phi and Psi angles for all residues in the a7 helix in all frames
    phi_ind, phi_angle = md.compute_phi(traj_a7, periodic = True, opt = True)
    psi_ind, psi_angle = md.compute_psi(traj_a7, periodic = True, opt = True)
    time, angles = np.shape(phi_angle) #determine the number of frames and the number of angles computed
    
    #Compute Secondary Structure of all Residues in the a7 helix using MDTraj function
    dssp_list = md.compute_dssp(traj_a7, simplified=False) #Compute DSSP for all residues in the a7 helix for all trajectory frames
    file_dssp = open('DSSP_'+ File_base + name_add + '.txt','w') #Create output file for DSSP and write over if file is present

    if rms_chk == False:
        if os.path.exists('uncorrelated_frames.txt'):
            lines = open('uncorrelated_frames.txt', 'r').readlines()
            t_full = []
            for i in lines:
                t_full.append(float(i.strip()))
        else:
            print('WARNING: Uncorrelated samples not removed!\n')
            t_full = np.linspace(0, len(phi_angle), num=len(phi_angle))
    #limit to uncorrelated data based on the uncorrelated samples for bb rmsd of the full protein
    phi_uncorr = np.zeros((len(t_full), angles)) #declare empty vectors for data
    psi_uncorr = np.zeros((len(t_full), angles))
    for i in range(angles): #loop through all angles
        phi_uncorr[:,i] = uncorr.sort(phi_angle[:,i], t_full)
        psi_uncorr[:,i] = uncorr.sort(psi_angle[:,i], t_full)

    #limit to uncorrelated data
    frame_max,residue = dssp_list.shape #determine the number of frames and residues for which dssp analysis was completed
    dssp_uncorr = np.full((len(t_full) - 1, residue), none) #declare an empty vector to input uncorrelated dssp values for each residue
    for i in range(residue): #loop through each residue seperately
        dssp_res = dssp_list[:,i] #seperate all time values for a single residue
        dssp_res_mod = [] 
        for j in dssp_res:
            if j == ' ': #in dssp a space designates a loop or irregular element
                dssp_res_mod.append('l') #subsitute an l for this designation to prevent issues with reading character values
            else: #if not a space keep the same character designation
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
    np.savetxt('phi_' + File_base + name_add + '.txt', phi_uncorr)
    np.savetxt('psi_' + File_base + name_add + '.txt', psi_uncorr)

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
    np.savetxt('Hbonds_atom_' + File_base + name_add + '.txt', hbonds) #Save all atom indicies of h-bonds to file for further analysis

    #Write all h-bonds present for >60% of trajectory to file
    file_object = open('Hbonds_'+ File_base + name_add + '.txt', 'w') 

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
    np.savetxt('Hbonds_per_' + File_base + name_add + '.txt', per)

    #Delete unneededarrays to save memory
    del hbonds; del label; del per; del da_distances; del da_angles

    #If ligand analysis is requested determine h-bonds between ligand and PTP1B residues
    if lig_check == True:
        file_lig = open('Hbonds_lig_' +  File_base + name_add + '.txt', 'w') #Create file for list of h-bonds determined to be present more than 10% of the trajectory
        if lig == 'both':            
            ligand = traj_ns.topology.select('resname AD') #Select the ligand by name (based on topology) from the trajectory
            ligand2 = traj_ns.topology.select('resname BBR') #Select the ligand by name (based on topology) from the trajectory
        else:
            ligand = traj_ns.topology.select('resname ' + str(lig)) #Select the ligand by name (based on topology) from the trajectory
        protein = traj_ns.topology.select('protein') #Select protein from the trajectory
        hbonds = md.baker_hubbard(traj_ns, freq = 0.1, exclude_water=True, periodic=True) #Find all h-bonds present >10% of the trajectory
        label = lambda hbond : '%s -- %s' % (traj_ns.topology.atom(hbond[0]), traj_ns.topology.atom(hbond[2])) #Seperate the names of the h-bonds based on topology nomenclature
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

if hbond_net_check == True:
    #Look specifically for hbonds important to the allosteric network
    hbond_allo_name = open(directory + '/analysis_scripts/Hbond_allo.txt', 'r').readlines()
    top = traj_ns.topology

    hbond_allo_atom = np.zeros((len(hbond_allo_name), 3))
    for i in range(len(hbond_allo_name)):
        bond = hbond_allo_name[i].strip().split()

        res1 = int(bond[1]) - offset
        res2 = int(bond[4]) - offset
        
        if traj_prot.n_residues > res1 and traj_prot.n_residues > res2:
            hbond_allo_atom[i,0] = top.select('resid ' + str(res1) + ' and name ' + str(bond[2]))
            hbond_allo_atom[i,2] = top.select('resid ' + str(res2) + ' and name ' + str(bond[5]))
            hbond_allo_atom[i,1] = hbond_allo_atom[i,0]+1

    per = [] #Declare empty array for percentage of time h-bond is formed
    da_distances = md.compute_distances(traj_ns, hbond_allo_atom[:,[1,2]], periodic=False) #Compute distance between h-bond donor and acceptor
    da_angles = md.compute_angles(traj_ns, hbond_allo_atom[:,:], periodic=False) #Compute angle between h-bond donor and acceptor
    [num_t, num_h] = np.shape(da_distances) #save values for number of frames(num_t) and number of bonds(num_b) to caculate
    for j in range(num_h): #Loop through all h-bonds
        count = 0 #Initialize count
        for i in range(num_t): #Loop through all frames
            if da_distances[i,j] <= 0.25 and da_angles[i,j] >= 2.094: #If distance between donor and acceptor is less than 2.5A and the angle is greater than 120 degrees or ~ 2.094 radians
                count +=1
        per.append(100*count/num_t) #Percentage of time each h-bond is present in trajectory
    
    #Save hbond percentage to file
    output = open('Hbond_allo_per' + name_add + '.txt', 'w')
    for i in range(len(hbond_allo_name)):
        output.write(str(hbond_allo_name[i]) + ': ' + str(per[i]) + '\n')

    #Delete unneededarrays to save memory
    del hbond_allo_name; del hbond_allo_atom; del per; del da_distances; del da_angles

    #Write to screen when code section is completed
    print('Hbond Network Analysis Written')

#Skip Hbond analysis if desired
else:
    print('Hbond Network Analysis Skipped')

#Residue pairs needed for either Ligand Interaction of Protein Interaction Analysis
if check_hel == True or lig_check == True:
    #Set Residue Pairs
    group_WPD, group_3, group_4, group_5, group_6, group_bend, group_7, group_L11 = mdfunc.set_sect(lig, a7_present, offset)

#Compute Interactions between the a3, a6, and a7 helices
if check_hel == True:
    if rms_chk == False:
        if os.path.exists('uncorrelated_frames.txt'):
            lines = open('uncorrelated_frames.txt', 'r').readlines()
            t_dist = []
            for i in lines:
                t_dist.append(float(i.strip()))
        else:
            print('WARNING: Uncorrelated Samples not removed!')
            t_dist = np.linspace(0, traj_ns.n_frames, num = traj_ns.n_frames)
    else:
        #Limit to uncorrelated samples
        t_dist = t_full #Determine indices of uncorrelated samples from RMSD of full trajectory
    time_uncorr = len(t_dist)

    file_mean = open('inter_mean.txt', 'w') #File for the mean number of interactions between any two given residues
    
    #Compute Distances
    mdfunc.hel_inter(traj_ns, group_3, group_6, t_dist, time_uncorr, file_mean, 'a3', 'a6', directory)
    mdfunc.hel_inter(traj_ns, group_3, group_WPD, t_dist, time_uncorr, file_mean, 'a3', 'WPD', directory)
    mdfunc.hel_inter(traj_ns, group_3, group_bend, t_dist, time_uncorr, file_mean, 'a3', 'bend', directory)
    if traj_ns.n_residues > 297: #Only include these pairs if the a7 is present in the trajectory
        mdfunc.hel_inter(traj_ns, group_3, group_7, t_dist, time_uncorr, file_mean, 'a7', 'a3', directory)
        mdfunc.hel_inter(traj_ns, group_6, group_7, t_dist, time_uncorr, file_mean, 'a7', 'a6', directory)
        mdfunc.hel_inter(traj_ns, group_L11, group_7, t_dist, time_uncorr, file_mean, 'a7', 'L11', directory)
   
    print('Helix Interaction Analysis Completed')
else:
    print('Helix Interaction Analysis Skipped')

if check_hel_indv == True:
    #Compute percent individual interactions are formed
    #Determine interaction pairs to evaluate
    if input_inter == False:
        if a7_present == True: #Only include these pairs if the a7 is present in the trajectory
            hel_inter = np.array([[200, 282], [179, 191], [185, 191], [151, 191], [264, 185], [178, 150], [81, 199], [117, 182], [117, 217], [192, 225], [182, 220], [187, 269], [178, 190], [152, 177], [200, 287], [276, 292], [189, 295], [280, 287], [152, 297]])
        else:
            hel_inter = np.array([[200, 282], [179, 191], [185, 191], [151, 191], [264, 185], [178, 150], [81, 199], [117, 182], [117, 217], [192, 225], [182, 220], [187, 269], [178, 190], [152, 177], [200, 287], [280, 287]])

    else:
        #Read input file if provided
        hel_str = open(input_inter, 'r').readlines()
        hel_inter = np.zeros((len(hel_str), 2))
        for i in range(len(hel_str)):
            res1, res2 = hel_str[i].split()
            hel_inter[i][:] = [int(res1), int(res2)]
 
    if rms_chk == False:
        if os.path.exists('uncorrelated_frames.txt'):
            lines = open('uncorrelated_frames.txt', 'r').readlines()
            t_full = []
            for i in lines:
                t_full.append(float(i.strip()))
        else:
            print('WARNING: Uncorrelated Samples not removed!')
            t_full = np.linspace(0, traj_ns.n_frames, num = traj_ns.n_frames)
   
    #Send to function to compute distances and save to file
    mdfunc.pair_dist(traj_ns, hel_inter, t_full, offset, directory)

    print('Individual Helix Interaction Analysis Completed')
else:
    print('Individual Helix Interaction Analysis Skipped')

#Ligand RMSD analysis
if lig_rmsd_check == True:
    #Calculate distance between Ligand and res 193
    res = 194 - offset
    res2 = 276 - offset

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
        if os.path.exists('uncorrelated_frames.txt'):
            lines = open('uncorrelated_frames.txt', 'r').readlines()
            t_full = []
            for i in lines:
                t_full.append(float(i.strip()))
            #Remove uncorrelated samples
            dist_lig_193 = uncorr.sort(lig_dist[:,0], t_full)
            dist_lig_283 = uncorr.sort(lig_dist[:,1], t_full)
            if lig == 'both':
                dist_lig2_193 = uncorr.sort(lig_dist[:,2], t_full)
                dist_lig2_283 = uncorr.sort(lig_dist[:,3], t_full)
        else:
            print('WARNING: Uncorrelated samples not removed!')
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
    np.savetxt('dist_lig_194.txt', dist_lig_193)
    np.savetxt('dist_lig_276.txt', dist_lig_283)
    if lig == 'both':
        np.savetxt('dist_lig2_194.txt', dist_lig2_193)
        np.savetxt('dist_lig2_276.txt', dist_lig2_283)

    if lig == 'AD' or lig == 'BBR':
        #Align trajectory and referance to residues 1 to 200
        lig_traj_top = traj_ns.topology
        lig_ref_top = lig_ref.topology
        traj_cmplx = traj_ns.atom_slice(traj_ns.topology.select('protein or resname AD or resname BBR'))

        res_align = lig_traj_top.select('resid ' + str(188 - offset) + ' to ' + str(202 - offset) + ' or resid ' + str(262 - offset) + ' to ' + str(277 - offset))
        
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
            if os.path.exists('uncorrelated_frames.txt'):
                lines = open('uncorrelated_frames.txt', 'r').readlines()
                t_full = []
                for i in lines:
                    t_full.append(float(i.strip()))
                displacment = uncorr.sort(displacment, t_full)
            else:
                print('WARNING: Uncorrelated samples not removed!')
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
    #Define ligand group
    if lig == 'both':
        group_l = traj_prot.n_residues + 1
        group_l2 = traj_prot.n_residues + 2
    else:
         group_l = traj_prot.n_residues + 1
       
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
        if os.path.exists('uncorrelated_frames.txt'):
            lines = open('uncorrelated_frames.txt', 'r').readlines()
            t_dist = []
            for i in lines:
                t_dist.append(float(i.strip()))
        else:
            print('WARNING: Uncorrelated Samples not removed!')
            t_dist = np.linspace(0, time, num = time)
    else:
        #Limit to uncorrelated samples
        t_dist = t_full #Determine indices of uncorrelated samples from RMSD of full trajectory
    time_uncorr = len(t_dist)

    #Set new arrays with uncorrelated samples
    if lig == 'both':
        dist_a3 = lig_dist_uncorr(num_pairs_a3, dist_a3_all, lig, dist2_a3_all, t_dist)
        dist_a4 = lig_dist_uncorr(num_pairs_a4, dist_a4_all, lig, dist2_a4_all, t_dist)
        dist_a5 = lig_dist_uncorr(num_pairs_a5, dist_a5_all, lig, dist2_a5_all, t_dist)
        dist_a6 = lig_dist_uncorr(num_pairs_a6, dist_a6_all, lig, dist2_a6_all, t_dist)
        dist_bend = lig_dist_uncorr(num_pairs_bend, dist_bend_all, lig, dist2_bend_all, t_dist)
        if a7_present == True:
            dist_a7 = lig_dist_uncorr(num_pairs_a7, dist_a7_all, lig, dist2_a7_all, t_dist)

    else:
        dist_a3, dist2_a3 = lig_dist_uncorr(num_pairs_a3, dist_a3_all, lig, 0, t_dist)
        dist_a4, dist2_a4 = lig_dist_uncorr(num_pairs_a4, dist_a4_all, lig, 0, t_dist)
        dist_a5, dist2_a5 = lig_dist_uncorr(num_pairs_a5, dist_a5_all, lig, 0, t_dist)
        dist_a6, dist2_a6 = lig_dist_uncorr(num_pairs_a6, dist_a6_all, lig, 0, t_dist)
        dist_bend, dist2_bend = lig_dist_uncorr(num_pairs_bend, dist_bend_all, lig, 0, t_dist)
        if a7_present == True:
            dist_a7, dist2_a7 = lig_dist_uncorr(num_pairs_a7, dist_a7_all, lig, 0, t_dist)

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
        #Index for residues interactions in a3, a6, and a7 + the bend with ligand
        bond = 0
        if lig == 'both':
            check_a3, check2_a3, check_a3_top, lig_tot_cont, lig2_tot_cont, bond = lig_hel_inter(num_pairs_a3, dist_a3, dist2_a3, 7, lig_tot_cont, lig, lig2_tot_cont, bond)
            check_a4, check2_a4, lig_tot_cont, lig2_tot_cont, bond = lig_hel_inter(num_pairs_a4, dist_a4, dist2_a4, 0, lig_tot_cont, lig, lig2_tot_cont, bond)
            check_a5, check2_a5, lig_tot_cont, lig2_tot_cont, bond = lig_hel_inter(num_pairs_a5, dist_a5, dist2_a5, 0, lig_tot_cont, lig, lig2_tot_cont, bond)
            check_a6, check2_a6, check_a6_mid, lig_tot_cont, lig2_tot_cont, bond = lig_hel_inter(num_pairs_a6, dist_a6, dist2_a6, 8, lig_tot_cont, lig, lig2_tot_cont, bond)
            check_bend, check2_bend, lig_tot_cont, lig2_tot_cont, bond = lig_hel_inter(num_pairs_bend, dist_bend, dist2_bend, 0, lig_tot_cont, lig, lig2_tot_cont, bond)

        else:
            check_a3, check_a3_top, lig_tot_cont, bond = lig_hel_inter(num_pairs_a3, dist_a3, 0, 7, lig_tot_cont, lig, 0, bond)
            check_a4, lig_tot_cont, bond = lig_hel_inter(num_pairs_a4, dist_a4, 0, 0, lig_tot_cont, lig, 0, bond)
            check_a5, lig_tot_cont, bond = lig_hel_inter(num_pairs_a5, dist_a5, 0, 0, lig_tot_cont, lig, 0, bond)
            check_a6, check_a6_mid, lig_tot_cont, bond = lig_hel_inter(num_pairs_a6, dist_a6, 0, 8, lig_tot_cont, lig, 0, bond)
            check_bend, lig_tot_cont, bond = lig_hel_inter(num_pairs_bend, dist_bend, 0, 0, lig_tot_cont, lig, 0, bond)


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
        
        if a7_present == True:
            if lig == 'both':
                check_a7, check2_a7, check_a7_top, lig_tot_cont, lig2_tot_cont, bond = lig_hel_inter(num_pairs_a7, dist_a7, dist2_a7, 8, lig_tot_cont, lig, lig2_tot_cont, bond)
            else:
                check_a7, check_a7_top, lig_tot_cont, bond = lig_hel_inter(num_pairs_a7, dist_a7, 0, 8, lig_tot_cont, lig, 0, bond)

            #Output number of interactions to file for a7
            file_a7.write(str(check_a7) + '\n')
            if lig == 'both':
                file2_a7.write(str(check2_a7) + '\n')

        if lig == 'AD' or lig == 'both': 
            if a7_present == True:
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
    plt.scatter(reduced_cartesian[:, 0], reduced_cartesian[:,1], marker='x', c=traj_ns.time)
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
