#Load data from a 2-column file with a prefix
#Input:
#file_dir = Directory input file is stored in
#file_name = Name of input file
#covert = Whether a conversion from nm to A is necessary
#Output: x, y = vectors for each of the two columns in the input file
def col2_load_data(file_dir, file_name, convert):
    x,y = [],[]
    #Add file extension
    if file_name.split('.')[-1] != 'xvg': #Add file extension if not in input
        file_name = file_name + '.xvg'

    #Load data
    with open(file_dir + '/' + file_name) as f:
        for _ in range(18):
            next(f)
        for line in f:
            cols = line.split()
            x.append(float(cols[0]))
            if convert == True:
                y.append(float(cols[1])*10)
            else:
                y.append(float(cols[1]))

    return x,y

#Make a box plot comparing Apo open, apo closed, AD, and BBR bound states
#Input:
#d_Apo_open, d_Apo_close, d_AD, d_BBR = data for each of the four states to plot
#inter1, inter2 = Interactions to plot data for file name
#p, p1 = p-values between Apo open and AD and Apo open and BBR states
#ylim = Y-axis limit
#Output: none just save graph as png
def plot_apo_lig_box(d_Apo_open, d_Apo_close, d_AD, d_BBR, inter1, inter2, p, p1, ylim):
    d_Apo_open_df = pd.DataFrame({'Apo Open': d_Apo_open})
    d_Apo_close_df = pd.DataFrame({'Apo Closed': d_Apo_close})
    d_AD_df = pd.DataFrame({'AD': d_AD})
    d_BBR_df = pd.DataFrame({'BBR': d_BBR})
    mean = np.array([np.mean(d_Apo_open), np.mean(d_Apo_close), np.mean(d_AD), np.mean(d_BBR)])

    df = pd.concat([d_Apo_open_df, d_Apo_close_df, d_AD_df, d_BBR_df])

    ax = sns.stripplot(data = df, dodge=True, alpha=0.05, zorder=1, palette='bright')
    ax = sns.pointplot(data = df, join=False, scale=0.75, palette='dark')
    error_bar(0, 2, mean[0], mean[2], p, 1, 'b')
    error_bar(0, 3, mean[0], mean[3], p1, 1, 'b')
    ax.set_ylim(0,ylim)
    if inter2 == 'L11':
        plt.title(r'Helical Interactions b/w $\alpha$-' + inter1 + ' and ' + inter2)
        plt.savefig('Hel_inter_a' + inter1 + '_' + inter2 + '_box.png')
    else:
        plt.title(r'Helical Interactions b/w $\alpha$-' + inter1 + r' and $\alpha$-' + inter2)
        plt.savefig('Hel_inter_a' + inter1 + '_a' + inter2 + '_box.png')

#Generalized box plot for comparison of two arrays
#Input:
#d_1, d_2 = array for data at each state 
#name1, name2 = name of each state
#p = p-value between the two states
#ylim = Upper-limit for the y-axis (-1 means automated)
#x_name, y_name, Title_name = Labels for the x-axis, y-axis, and title for graph. Leave a space in plane of the name for no axis label.
#File_name = Name to save file
#x_dim, y_dim = Dimensions for the file (-1 means automated)
#Output: none just save graph as png
def plot_gen_box(d_1, d_2, name1, name2, p, ylim, x_name, y_name, Title_name, File_name, x_dim, y_dim):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    d_1_df = pd.DataFrame({name1: d_1})
    d_2_df = pd.DataFrame({name2: d_2})
    mean = np.array([np.mean(d_1), np.mean(d_2)])
    
    if x_dim != -1 and y_dim != -1:
        fig = plt.figure(figsize=(x_dim,y_dim))
    else:
        fig = plt.figure()

    df = pd.concat([d_1_df, d_2_df])

    ax = sns.stripplot(data = df, dodge=True, alpha=0.5, zorder=1, palette = 'bright')
    ax = sns.pointplot(data = df, join=False, scale=1.0, palette = 'dark')
    
    error_bar(0, 1, mean[0], mean[1], p, 0.5, 'k')
    if ylim != -1:
        ax.set_ylim(0, ylim)
    if x_name != '':
        plt.xlabel(x_name, fontsize = 14)
    if y_name != '':
        plt.ylabel(y_name, fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.title(Title_name, fontsize = 18)
    plt.savefig(File_name + '.png')
    plt.close()

#Write ligand binding conformations to file
#Input:
#lig_frac = Array with fraction of trajectory in each binding conformation and unbound
#Loc_frac = File to output to
#lig = Name of ligand
#Output: None but write to file
def write_lig_bind(lig_frac, Loc_frac, lig):
    if lig == 'AD':
        Loc_frac.write('Binding location 1(crystal): ' + str(100 * lig_frac[0]) + '\n')
        Loc_frac.write('Binding location 2(alt 1): ' + str(100 * lig_frac[1]) + '\n')
        Loc_frac.write('Binding location 3(alt 2): ' + str(100 * lig_frac[2]) + '\n')
        Loc_frac.write('Binding location 4(alt 3): ' + str(100 * lig_frac[3]) + '\n')
        Loc_frac.write('Unbound: ' + str(100 * lig_frac[4]) + '\n')
        Loc_frac.write('Other Bound: ' + str(100 * lig_frac[5]))
    if lig == 'BBR':
        Loc_frac.write('Binding location 1(crystal): ' + str(100 * lig_frac[0]) + '\n')
        Loc_frac.write('Unbound: ' + str(100 * lig_frac[1]) + '\n')
        Loc_frac.write('Other Bound: ' + str(100 * lig_frac[2]))
    Loc_frac.close() #Close file

#Write error bar for plot
#Input:
#(x1, y1) and (x2, y2) = coordinates for the two points to connect the error bar
#p = p-value to establish the presence and label for the error bar
#h = height of the error bar
#col = color for the error bar
#Output: None but add error bar to plot
def error_bar(x1, x2, y1, y2, p, h, col):
    import matplotlib.pyplot as plt
    y = max([y1, y2])
    if p < 0.05 and p > 0.01:
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*0.5, y+h, "*" , ha='center', va='bottom', color=col)
    if p < 0.01 and p > 0.001:
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*0.5, y+h, "**" , ha='center', va='bottom', color=col)
    if p < 0.001:
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*0.5, y+h, "***" , ha='center', va='bottom', color=col)

