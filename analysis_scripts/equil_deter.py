#!/ usr / bin / env python
#Import packages
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

#Declare arguments
parser = argparse.ArgumentParser(description = 'Determination of Equilibration Time from RMSD')
parser.add_argument('-n', required=True, help='File name for BB RMSD (xvg)')
parser.add_argument('-d', required=False, default='.', type=str, help='File Directory for RMSD')
parser.add_argument('-t', required=True, type=int, help='Total time for trajectory (ns)')
parser.add_argument('-df', required=False, default='$PROJECT/code/PTP1B', type=str, help='Directory Path for Functions')

#Import Arguments
args = parser.parse_args()
file_name = args.n
file_dir = args.d
dir_util = args.df
t_max = args.t

#Import custom modules
sys.path.insert(1,dir_util + '/util')
import plot

#Load data
t, rmsd = plot.col2_load_data(file_dir, file_name, False)

#Average rmsd every 1 ns
int_per_ns = int(len(rmsd)/(t_max))
rmsd_max = np.zeros(int_per_ns*5)
rmsd_min = np.zeros(int_per_ns*5)
n=0

for i in range(len(rmsd_max)):
    k = int(n + (int_per_ns/5))
#    print(str(n) + ' ' + str(k) + ' ' + str(len(rmsd[n:k])))
    rmsd_max[i] = max(rmsd[n:k])
    rmsd_min[i] = min(rmsd[n:k])
    n = k

#Determine Equilibration time
output = open('equilibration_time.txt', 'w')
time = np.linspace(0, t_max, num = len(rmsd_max))
count = 0
for i in range(1, len(rmsd_max)):
    diff = abs(rmsd_max[i] - rmsd_min[i-1])
    if diff < 2:
        count += 1
    else:
        count = 0
    if count > 50:
        eq_time = 5 * (round(time[i]/5) + (time[i] % 5 > 0)) #round up to nearest 5 ns
        output.write('Eq Time: ' + str(eq_time) + ' ns\n')#Write to file
        start_i = i
        break

#Examine trajectory for convergence
diff = []
for i in range(int(start_i*5), len(rmsd)):
    diff.append(rmsd[i] - rmsd[i-1])

#Check when the difference is less than 2A
diff = np.array(diff)
check_converge = ((diff <= 2).sum() == diff.size).astype(np.int)
if check_converge == True:
    output.write('Convergence Reached')
else:
    output.write('Convergence NOT Reached')
   
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('RMSD Convergence', fontsize = 20)
ax1.set_ylabel('Difference in RMSD', fontsize = 18)
ax1.plot(diff)
fig.savefig('rmsd_convergence.png')
plt.close(fig)

