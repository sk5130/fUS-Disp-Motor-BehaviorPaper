## -------------------------- Modules ------------------------------ ##
import numpy as np
import matplotlib
import os
import h5py
import math
import numpy as np
from natsort import natsorted
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw
import pandas as pd
import seaborn as sn
import glob
import scipy.stats as stats
import statistics
import matplotlib.patches as mpatches
from numba import jit
from scipy import signal
from scipy.io import savemat, loadmat
from scipy.stats import gamma
## ------------------------ Types of Results ------------------------ ##
TimeCourseCBV = False
averageCBV = True # mean CBV responses
CorrelationMap = True # Pearson's correlation map
CaptureFrame = False
## ---------------------- Dataset selection --------------------- ##
dataset = 'Figure5' # 'Figure3' or 'Figure5'

if dataset == 'Figure3':
    path = 'C:\\Users\\UEIL10\\Downloads\\fUS-Disp-Motor-BehaviorPaper-Publication\\Data\\Figure3\\CBV\\'
    AcqInfo = 'C:\\Users\\UEIL10\\Downloads\\fUS-Disp-Motor-BehaviorPaper-Publication\\Data\\Figure3\\AcqInfo.mat'
    left_xlim = 5.5 # mm
    right_xlim = 3 # mm
    corr_threshold = 0.2
    corr_lag = -1 # s, 3s, and 5s
    Corrlim = 0.35
    CaptureFrame = True # visualize individual CBV frames
elif dataset == 'Figure5':
    path = 'C:\\Users\\UEIL10\\Downloads\\fUS-Disp-Motor-BehaviorPaper-Publication\\Data\\Figure5\\CBV\\'
    AcqInfo = 'C:\\Users\\UEIL10\\Downloads\\fUS-Disp-Motor-BehaviorPaper-Publication\\Data\\Figure5\\AcqInfo.mat'
    left_xlim = 3.25 # mm, LThFUS
    right_xlim = 5.25 # mm, LThFUS
    corr_threshold = 0.3
    corr_lag = 0
    Corrlim = 0.7

# CBV
wn = 4
CBVlim = 15
vmax = 0.4
medfilt_size = 5
# functional framerate about 1 s
f_framerate=1;

cbv_window = int(30*f_framerate); #duration + post_stim
bsl_window = int(10*f_framerate); #pre_stim
bsl_fig_window = int(5*f_framerate); # time frame we wanna show in the figure

## -------------------------- Functions ----------------------------- ##
def moving_average(x, w):
    return np.convolve(x,np.ones(w),'valid') / w

@jit(nopython = True)
def calc_r(s,A,v,w):
    r = 0

    r1 = r; r2 = r; r3 = r
    for i in range(len(A)):
        r1 += (s[v,w,i]-np.mean(s[v,w,:]))*(A[i]-np.mean(A))
        r2 += (s[v,w,i]-np.mean(s[v,w,:]))**2
        r3 += (A[i]-np.mean(A))**2
    r = r1/(np.sqrt(r2)*np.sqrt(r3))
    return r

def hrf(times):
    peak_values = gamma.pdf(times, 3)
    undershoot_values = gamma.pdf(times, 3)
    values = peak_values - 0.9 * undershoot_values
    return values /np.max(values) * 0.6

## -------------------------- Colormaps ----------------------------- ##
cmap = plt.cm.RdBu
newRdBu = cmap(np.arange(cmap.N))
newRdBu[:,-1] = np.abs(np.linspace(-1, 1, cmap.N))**2
newRdBu = matplotlib.colors.ListedColormap(newRdBu);

cmap = plt.cm.gray
newgray = cmap(np.arange(cmap.N))
newgray[:,-1] = np.abs(np.linspace(-1, 1, cmap.N))**2
newgray = matplotlib.colors.ListedColormap(newgray);

## -------------------------- Processing ---------------------------- ##                
for u, v in h5py.File(AcqInfo, mode='r').items():
    exec("%s = v" % u)
imgsize = [int(CUDArecon['imZsize'][0][0]), int(CUDArecon['imXsize'][0][0])]
baseline = int(P['stim']['baseline'][0][0]*f_framerate)
cooldown = int(P['stim']['cooldown'][0][0]*f_framerate)
duration = int(P['stim']['duration'][0][0]*f_framerate)
total_stim= int(P['numstims'][0][0])
print("num stims:", total_stim, ", baseline:", baseline, ", stim:", duration, ", cooldown:", cooldown);

frames = os.listdir(path)
nframes = len(frames)-(wn-1)

stim_frames = np.zeros(len(frames));

# setting regressor for correlation analyses
cooldown= cooldown+5;
duration= duration-5;
for i in range(0,total_stim):
    stim_frames[baseline+(i)*(duration+cooldown)+corr_lag+1:baseline+(i)*(duration+cooldown)+duration+corr_lag]=0;
    stim_frames[baseline+(i)*(duration+cooldown)+corr_lag+1:baseline+(i)*(duration+cooldown)+duration+corr_lag]=1;

baseline = int(P['stim']['baseline'][0][0]*f_framerate)
cooldown = int(P['stim']['cooldown'][0][0]*f_framerate)
duration = int(P['stim']['duration'][0][0]*f_framerate)

stim_frames_mw = moving_average(stim_frames,wn)
stim_frames_mw[stim_frames_mw>0]=1

RF = np.zeros((imgsize[0],imgsize[1],nframes));
non_norm_RF = np.zeros((imgsize[0],imgsize[1],nframes));
idx = 0;
s = np.zeros((imgsize[0],imgsize[1],len(frames)));

frames = os.listdir(path)

imgsize = [int(CUDArecon['imZsize'][0][0]), int(CUDArecon['imXsize'][0][0])]
im_extent = [CUDArecon['imXrange'][0][0],CUDArecon['imXrange'][-1][0],CUDArecon['imZrange'][-1][0],CUDArecon['imZrange'][0][0]]
im_extent = [x*Trans['wl'][0][0]*1e3 for x in im_extent]

lower_ylim = im_extent[2];
upper_ylim = im_extent[3];

text_x = -left_xlim+2.5;
text_y = upper_ylim+0.3;
brain_figratio = abs((left_xlim+right_xlim)/abs(upper_ylim-lower_ylim));
brain_figsize = [13,round(13/(brain_figratio+0.4),1)]
rcvdata = np.zeros((imgsize[0],imgsize[1],len(frames)))
files = natsorted(frames)
for i,f in enumerate(files):
    filepath = os.path.join(path,f)
    for u, v in h5py.File(filepath, mode = 'r').items():
        rcvdata[:,:,i] =np.transpose(np.array(v))
        print(i, 'th frame imported')
rcvdata = rcvdata/np.max(rcvdata);

for z in range(imgsize[0]):
    for x in range(imgsize[1]):
        RF[z,x,:] = moving_average(rcvdata[z,x,:],wn)

## -------------------------------------------------------------------- ##
dCBV = np.zeros((imgsize[0],imgsize[1],nframes))
baseline_rf = np.zeros((imgsize[0],imgsize[1]))
baseline_rf[:,:] = np.mean(RF[:,:,:baseline-wn+1],axis=2)

for frame in range(nframes):
    dCBV[:,:,frame] = signal.medfilt2d((RF[:,:,frame]-baseline_rf[:,:])/baseline_rf[:,:]*100,medfilt_size)

scale1 = {'vmin':0, 'vmax': vmax, 'cmap':newgray, 'aspect':'auto'}
scale2 = {'vmin':0, 'vmax': vmax, 'cmap':newgray, 'extent':im_extent, 'aspect':'auto'}
scale2_hot = {'vmin':0, 'vmax': vmax, 'cmap':'afmhot', 'extent':im_extent, 'aspect':'auto'}
scale_contour = {'vmin':corr_threshold, 'vmax': Corrlim, 'cmap':'hot', 'extent':im_extent, 'aspect':'auto'}
scale3 = {'vmin':-CBVlim, 'vmax': CBVlim, 'alpha':1,'cmap':newRdBu,'extent':im_extent, 'aspect':'auto'}

if (TimeCourseCBV):
    all_masks = []
    num_regions = 2
    for region in range(num_regions):
        fig, ax1 = plt.subplots(1,1,figsize=(10,5))
        im1 = ax1.imshow((np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2))),**scale1)
        ROI = plt.ginput(n = -1, timeout = 0, show_clicks = True)
        plt.close()
        region = [(np.round(x[0]),np.round(x[1])) for x in ROI]
        img = Image.new('L', (imgsize[1],imgsize[0]), 0)
        ImageDraw.Draw(img).polygon(region, outline = 1, fill = 1)
        mask = np.array(img)
        all_masks.append(mask)

    any_x=[x/framerate for x in range(0,nframes)]
    fig, ax1 = plt.subplots(1,1,figsize=(10,5))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1,0],axis=2)),**scale2)
    im1 = ax1.imshow(all_masks[0] + all_masks[1],alpha = 0.5,**scale2)
    ax1.set_title('ROI selection')
    plt.show();
    dCBV_trace = []
    dCBV_trace_err = []
    dCBV_mean = np.zeros((nframes,1))
    dCBV_stddev = np.zeros((nframes,1))
    a = np.zeros((nframes,total_acq))
    for mask in all_masks:
        for i in range(total_acq):
            for frame in range(nframes):
                a[frame,i] = np.mean(dCBV[:,:,frame,i]*mask)
        dCBV_mean = np.mean(a,axis=-1);
        dCBV_stddev = np.std(a,axis=-1)
        dCBV_trace.append(dCBV_mean)
        dCBV_trace_err.append(dCBV_stddev)

    fig,[ax1, ax2]=plt.subplots(2,1,figsize=(14,6),sharex=True)
    im = ax1.plot(any_x, dCBV_trace[0]*100);
    ax1.fill_between(any_x,np.subtract(dCBV_trace[0]*100,dCBV_trace_err[0]*100).squeeze(),np.add(dCBV_trace[0]*100,dCBV_trace_err[0]*100).squeeze(), alpha=0.1)
    ymin, ymax = ax1.get_ylim();
    for i in range(0,total_stim):
        stim_window = mpatches.Rectangle(((baseline+i*(duration+cooldown)-wn+1)/framerate,ymin),duration/framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.2);
        ax1.add_patch(stim_window);
    ax1.set_title("CBV - ROI in Left Hemisphere (Imagewise)")

    im = ax2.plot(any_x, dCBV_trace[1]*100);
    ax2.fill_between(any_x,np.subtract(dCBV_trace[1]*100,dCBV_trace_err[1]*100).squeeze(),np.add(dCBV_trace[1]*100,dCBV_trace_err[1]*100).squeeze(), alpha=0.1)
    ymin, ymax = ax2.get_ylim();
    for i in range(0,total_stim):
        stim_window = mpatches.Rectangle(((baseline+i*(duration+cooldown)-wn+1)/framerate,ymin),duration/framerate,ymax-ymin, fill = True, linewidth = None, color = "red", alpha=0.2);
        ax2.add_patch(stim_window);
    ax2.set_title("CBV - ROI in Right Hemisphere (Imagewise)")
    plt.suptitle("$\Delta$CBV/CBV [%]")
    plt.xlabel("second [s]")
    plt.show();

if (averageCBV):
    dCBV_cropped_bsl=np.zeros((imgsize[0],imgsize[1],bsl_window+cbv_window+1,total_stim))
    bsl_rf = np.zeros((imgsize[0],imgsize[1],total_stim))
    for j in range(total_stim):
        bsl_rf[:,:,j] = np.mean(RF[:,:,baseline-bsl_window+j*(cooldown+duration)-wn+1:baseline+j*(cooldown+duration)-wn+1],axis=2)
    for i in range(total_stim):
        for frame in range(bsl_window+cbv_window+1):
            dCBV_cropped_bsl[:,:,frame,i] = signal.medfilt2d((RF[:,:,baseline-bsl_window+i*(cooldown+duration)-wn+1+frame]-bsl_rf[:,:,i])/bsl_rf[:,:,i]*100,medfilt_size)  

if (averageCBV):
    def init():
        data1 = np.mean(RF[:,:,:baseline-wn+1], axis=2) / np.max(np.mean(RF[:,:,:baseline-wn+1], axis=2))
        im1.set_data(data1)

        data2 = np.mean(dCBV_cropped_bsl[:,:,0,:], axis=-1)
        im2.set_data(data2)
        return im1, im2

    def animate(i):
        data1 = np.mean(RF[:,:,:baseline-wn+1], axis=2) / np.max(np.mean(RF[:,:,:baseline-wn+1], axis=2))
        im1.set_data(data1)

        data2 = np.mean(dCBV_cropped_bsl[:,:,i,:], axis=-1)
        im2.set_data(data2)

        if i >= bsl_window and i < bsl_window + duration:
            t.set_text('FUS ON - time = {:.2f} s'.format((i-bsl_window+1)/f_framerate))
            t.set_color('yellow')
        else:
            t.set_text('FUS OFF - time = {:.2f} s'.format((i-bsl_window+1)/f_framerate))
            t.set_color('white')
        return im1, im2, t

    scale3 = {'vmin':-CBVlim, 'vmax': CBVlim, 'alpha':1,'cmap':newRdBu,'extent':im_extent, 'aspect':'auto'}

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-2),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_ylim((lower_ylim,upper_ylim))
    t = ax1.text(text_x,text_y,'FUS OFF - time = {:.2f} s'.format(-bsl_window),color='white',fontweight='bold',fontsize=24)
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ani = animation.FuncAnimation(fig,animate,interval = 200,frames=range(0,bsl_window+cbv_window+1-5),init_func = init, repeat=True)
    plt.show()

if (CorrelationMap):
    a = np.zeros((imgsize[0], imgsize[1], nframes))
    r = np.zeros((imgsize[0], imgsize[1]));
    for frame in range(nframes):
        a[:,:,frame] = RF[:,:,frame]
    for z in range(0,imgsize[0]):
        for x in range(0,imgsize[1]):
            r[z,x] = calc_r(a[:,:,:],stim_frames_mw[:],z,x);
    r0=np.copy(r);
    r0[r0<corr_threshold] = 0;
    r[:,:]=signal.medfilt2d(r[:,:],medfilt_size);
    r0[:,:]=signal.medfilt2d(r0[:,:],medfilt_size);
    
    fig, ax9 = plt.subplots(1, 1, figsize=(brain_figsize[0],brain_figsize[1]));
    ax9.set_xlim((-left_xlim,right_xlim))
    ax9.set_ylim((lower_ylim,upper_ylim))
    ax9.set_xlabel('mm',fontsize=20);
    ax9.set_ylabel('mm',fontsize=20);
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    alpha_arr=(r0>0).astype(float);
    ax9.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im8 = ax9.imshow(r0[:,:], cmap = 'hot', vmin=corr_threshold, vmax=Corrlim, alpha=alpha_arr, extent = im_extent, aspect = 'auto');
    cbar = fig.colorbar(im8, ax = ax9, fraction = 0.05, pad = 0.01)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel(' ', rotation = 270, labelpad = 10, fontsize=16)
    plt.show();


if(CaptureFrame):
    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=10)
    ax1.set_ylabel('mm',fontsize=10)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,9,:],axis=-1)) # 0 s snapshot
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,12,:],axis=-1)) # 3 s snapshot
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,13,:],axis=-1)) # 4 s snapshot
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,16,],axis=-1)) # 7 s snapshot
    plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(brain_figsize[0],brain_figsize[1]))
    im1 = ax1.imshow(np.mean(RF[:,:,:baseline-wn+1],axis=2)/np.max(np.mean(RF[:,:,:baseline-wn+1],axis=2)),**scale2)
    im2 = ax1.imshow(np.mean(dCBV_cropped_bsl[:,:,0,:],axis=-1),**scale3)
    ax1.set_xlim((-left_xlim,right_xlim))
    ax1.set_xlabel('mm',fontsize=20)
    ax1.set_ylabel('mm',fontsize=20)
    cbar = fig.colorbar(im2, ax = ax1, fraction = 0.05, pad = 0.01)
    cbar.ax.set_ylabel('$\Delta$CBV/CBV [%]', rotation = 270, labelpad = 10, fontsize=20)
    im2.set_data(np.mean(dCBV_cropped_bsl[:,:,32,:],axis=-1)) # 23 s snapshot
    plt.show()
