# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:25:35 2019

For internal use only. Do not share without permission.

@author: laramos, rricci
"""
import sys
import numpy as np
import matplotlib.pylab as pylab
#from matplotlib import pyplot, transforms
#from matplotlib.widgets import Button
#from basic_units import cm, inch
#import pywfdb
import scipy.signal as sps
import matplotlib.gridspec as gridspec
import os

type = 'negative'

# dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Clean_data_bonus\rhythm'+'\\'+type
# dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset\rhythm'+'\\'+type
dir = r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\excluded_examples'

os.chdir(dir)

keys = []
# include = ['009', '050', '054', '071', '131', '137', '177', '216', '221', '233', '263', '267', '295', '329', '342', '348', '388', '389', '401', '408', '410', '428', '446', '478', '527', '528', '545', '646', '663', '002', '059', '122', '141', '155', '157', '163', '170', '181', '211', '227', '234', '236', '243', '244', '249', '250', '253', '256', '259', '262', '282', '296', '300', '307', '331', '332', '340', '351', '365', '377', '387', '434', '442', '443', '450', '456', '466', '467', '474', '475', '526', '556', '588', '608', '609', '623', '626', '649', '654', '661', '668']
for file in os.listdir(dir):
    # if file[5:8] in include: #use this for the saving the plots of only the newly introduced ecg that serve as replacements of noisy ones
    # keys.append(file[5:8])
    keys.append(file[0:19])

# dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset\rhythm\pre_plot_np'+'\\'+type
dir = r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\mapje'
os.chdir(dir)

data = np.load('ecgs2.npy',allow_pickle=True)  
# print(data)
# print(keys)
# sys.exit()

# for i in range(len(data)): # go over patients
#     for j in range(12): # go over patient leads
#         for k in range(len(data[i][j])): # go over values in 1 lead
#             data[i][j][k] = float(data[i][j][k])/1000
# np.save('ecgs2', data)
# sys.exit()

# data = np.load(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\pre_plot_np\Negative\neg_ecgs',allow_pickle=True)  
names = ['I','V1','II','V2','III','V3','aVR','V4','aVL','V5','aVF','V6']  
# keys =['2884957','5727882','4819191','2768979']
for i,k in enumerate(keys): 
    s = data[i]
    
    # s=s[:]/1000.0

    # l3 = s[1,:] - s[0,:]
    # avr = (s[0,:]+s[1,:])/2
    # avl = (s[0,:]-s[1,:])/2
    # avf = (s[1,:]-s[0,:])/2
    
    s_f = np.zeros((12, len(s[0])))
                                            #    0   1   2   3  4  5 6  7   8  9 10 11
    s_f[0,:] = s[0] #1                     #  III aVR aVL aVF I II V1 V2 V3 V4 V5 V6  is the correct sequence
    s_f[1,:] = s[5] #v1
    s_f[2,:] = s[1] #2
    s_f[3,:] = s[6] #v2
    
    s_f[4,:] = s[2] #l3
    s_f[5,:] = s[7] #v3
    s_f[6,:] = s[3] #aVR
    s_f[7,:] = s[8] #v4
    
    s_f[8,:] = s[4] #avl
    s_f[9,:] = s[9] #v5
    s_f[10,:] = s[5] #aVF
    s_f[11,:] = s[11] #v6
    
    
    start_time = 1                              # beginning position in seconds
    time = 10                                     # data length in seconds  10
    frequency = 250
    
    sample_length = int(time * frequency) # data length in sample units
    end_time = 6                # end position in seconds  6
    
    signals_num = 6                   # number of signals in record  6
    
    
    gs1 = gridspec.GridSpec(6, 2)
    gs1.update(wspace=0.0, hspace=0.05) # set the spacing between axes.
    
    
    ylims = (-1.5, 1.5)
    t = np.arange(start_time, end_time, 1/frequency)
    vl = np.arange(start_time, end_time, 0.2)
    hl = np.arange(ylims[0], ylims[1], 0.5)
    
    f = pylab.figure(figsize=(25, 19))
    
    f.subplots_adjust(hspace=0.00001)
    
    axes = []
    ax = pylab.subplot(gs1[0])
    
    
    
    #pylab.title('ID: %d                                                                         PLN: [ ]YES    [ ]NO'%(new_ids[i]),fontsize=20)
    
    #annotation_lines = [ann.time/frequency for ann in annotations]
    for i in range(0,s_f.shape[0]):
        ax = pylab.subplot(gs1[i])
        axes.append(ax)
        if i%2==1:
            ax.yaxis.set_visible(False)
            ax.text(1, -0.25, names[i], fontsize=12)
        else:
            ax.text(0.999,-0.25, names[i], fontsize=12)
        
        pylab.ylabel('voltage (%s)' % "mm")
        
        # draw pink grid
        ax.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
        ax.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
        res = sps.resample(s_f[i,:], 2500)
        # read data for specified signal
        # equal to record.read(i, ...
        y = res[0:1250]
        
        # draw signal
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(90)
        #ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
        ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label='q')
        ax.text(start_time + 1.01*time , 1.2, '1', color='k' )
        
        pylab.ylim(-1.5, 1.5)
        pylab.xlim(start_time, end_time)  
    
    pylab.ylim(-1.5, 1.5)
    pylab.xlim(start_time, end_time)  
    
    pylab.xlabel('time (s)')
    #xticklabels = [a.get_xticklabels() for a in axes[:-1]]
    #pylab.setp(xticklabels, visible=False)

    # os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset\rhythm\plots'+'\\'+type)
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\plots')
    
    pylab.savefig(k+'ecg_plot.jpg',orientation='landscape')
    
    # pylab.show()


#
#
#ax = pylab.subplot(gs1[1])
#ax.axis('off')
#axes.append(ax)
#
#pylab.ylabel('voltage (%s)' % "mm")
#
## draw pink grid
#ax.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
#ax.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
#res = sps.resample(s[0,:], 2500)
## read data for specified signal
## equal to record.read(i, ...
#y = res[0:1250]
#
## draw signal
##base = pyplot.gca().transData
##rot = transforms.Affine2D().rotate_deg(90)
##ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
#ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label='q',axis='off')
#ax.text(start_time + 1.01*time , 1.2, '1', color='k' )
#
#pylab.ylim(-1.5, 1.5)
#pylab.xlim(start_time, end_time) 
#
#
#
#ax = pylab.subplot(gs1[2])
#axes.append(ax)
#
#pylab.ylabel('voltage (%s)' % "mm")
#
## draw pink grid
#ax.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
#ax.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
#res = sps.resample(s[0,:], 2500)
## read data for specified signal
## equal to record.read(i, ...
#y = res[0:1250]
#
## draw signal
##base = pyplot.gca().transData
##rot = transforms.Affine2D().rotate_deg(90)
##ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
#ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label='q')
#ax.text(start_time + 1.01*time , 1.2, '1', color='k' )
#
#pylab.ylim(-1.5, 1.5)
#pylab.xlim(start_time, end_time) 
#
#
#
#ax = pylab.subplot(gs1[3])
#ax.axis('off')
#axes.append(ax)
#
#pylab.ylabel('voltage (%s)' % "mm")
#
## draw pink grid
#ax.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
#ax.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
#res = sps.resample(s[0,:], 2500)
## read data for specified signal
## equal to record.read(i, ...
#y = res[0:1250]
#
## draw signal
##base = pyplot.gca().transData
##rot = transforms.Affine2D().rotate_deg(90)
##ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
#ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label='q')
#ax.text(start_time + 1.01*time , 1.2, '1', color='k' )
#
#pylab.ylim(-1.5, 1.5)
#pylab.xlim(start_time, end_time)  
#
#pylab.xlabel('time (s)')
#xticklabels = [a.get_xticklabels() for a in axes[:-1]]
#pylab.setp(xticklabels, visible=False)
#
#pylab.savefig('ecg_plot.png')
#
#pylab.show()
#
#
#for i, name in enumerate(names):
#    
#    ax = pylab.subplot(signals_num, 1, i+1)
#    axes.append(ax)
#
#    pylab.ylabel('voltage (%s)' % "mm")
#
#    # draw pink grid
#    ax.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
#    ax.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
#    res = sps.resample(s[i,:], 2500)
#    # read data for specified signal
#    # equal to record.read(i, ...
#    y = res[0:1250]
#
#    # draw signal
#    #base = pyplot.gca().transData
#    #rot = transforms.Affine2D().rotate_deg(90)
#    #ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
#    ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name)
#    ax.text(start_time + 1.01*time , 1.2, name, color='k' )
#    
#    pylab.ylim(-1.5, 1.5)
#    pylab.xlim(start_time, end_time)  
#    
##    ax = pylab.subplot(signals_num, 2, i+1)
##    axes.append(ax)
##
##    pylab.ylabel('voltage (%s)' % "mm")
##
##    # draw pink grid
##    ax.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
##    ax.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
##    res = sps.resample(s[i,:], 2500)
##    # read data for specified signal
##    # equal to record.read(i, ...
##    y = res[0:1250]
##
##    # draw signal
##    #base = pyplot.gca().transData
##    #rot = transforms.Affine2D().rotate_deg(90)
##    #ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
##    ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name)
##    ax.text(start_time + 1.01*time , 1.2, name, color='k' )
##    
##    pylab.ylim(-1.5, 1.5)
##    pylab.xlim(start_time, end_time)  
##    
#
#    #mark_annotations(ax)  
#
##    ax2 = pylab.subplot(signals_num, 2, i+1)
##    axes.append(ax)
##
##    pylab.ylabel('voltage (%s)' % "mm")
##
##    # draw pink grid
##    ax2.vlines(vl, ylims[0], ylims[1], colors ='r', alpha=0.3)
##    ax2.hlines(hl, start_time, end_time, colors ='r', alpha=0.3)
##
##    res = sps.resample(s[i,:], 2500)
##    # read data for specified signal
##    # equal to record.read(i, ...
##    y = res[0:1250]
##
##    # draw signal
##    #base = pyplot.gca().transData
##    #rot = transforms.Affine2D().rotate_deg(90)
##    #ax.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name,transform= rot + base)
##    ax2.plot(t, y, linewidth=1, color='k', alpha=1.0, label=name)
##    ax2.text(start_time + 1.01*time , 1.2, name, color='k' )
##    #ax2.axis('off')
##    #mark_annotations(ax)
##
##    pylab.ylim(-1.5, 1.5)
##    pylab.xlim(start_time, end_time)   
#
#pylab.xlabel('time (s)')
#xticklabels = [a.get_xticklabels() for a in axes[:-1]]
#pylab.setp(xticklabels, visible=False)
#
#pylab.savefig('ecg_plot.png')
#
#pylab.show()