# file for extracting GSR, HR,  and Respiration features
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.fft import fft
#from numpy.fft import fft
import scipy.signal as signal
import heartpy as hp
from scipy.stats import skew, kurtosis
from scipy.signal import argrelextrema

def extract_resp_feats(Resp,fs=128):
    # Respiration Features:
    # power in frequency bands 0.25 to 2.75 (windows of 0.25)
    # Mean peak value
    # mean magnitude of derivative
    # standard deviation
    # range

    # normalization ??? 

    # Respiration Data Preproc pipleline before feature 
    errors = []
    Resp -= Resp.mean()
    b_lp,a_lp = signal.butter(5,.1,btype='high',fs=128,analog=False)
    b_hp,a_hp = signal.butter(5,3,btype='low',fs=128,analog=False)
    Resp = signal.filtfilt(b_lp,a_lp,Resp)
    Resp = signal.filtfilt(b_hp,a_hp,Resp)
    
    # mean peak value
    peak_avg = np.zeros([Resp.shape[0],])
    for i in range(Resp.shape[0]):
        peaks = signal.find_peaks(Resp[i,:],height=300,width=75)
        if len(peaks[1]['peak_heights']) == 0:
            errors.append(i)
            peak_avg[i] = -1
        else:
            peak_avg[i] = peaks[1]['peak_heights'].mean()

    # Range
    range_feat = Resp.max(1) - Resp.min(1)

    # standard deviation
    std_feat = Resp.std(1)

    # mean magintude of derivative
    mean_der = np.abs(np.diff(Resp,axis=1)*fs).mean(1)

    # extracting power bands
    R_fft = fft(Resp)
    w = np.arange(0,Resp.shape[1]/2)*fs/Resp.shape[1]
    win_size = np.floor(0.25 * Resp.shape[1]/fs)//2
    fcs = np.linspace(0.25,2.5,10)
    ends = fcs+.25/2
    starts = fcs-.25/2
    means = []
    for i in range(10):
        start_idx = np.where((w>starts[i]))[0].min()
        end_idx = np.where((w<ends[i]))[0].max()
        means.append(np.mean(np.abs(R_fft[:,start_idx:end_idx]),axis=1))

    power_feats = np.vstack(means).T
    
    names = ['RMP1', 'RMP2','RMP3', 'RMP4','RMP5', 'RMP6','RMP7', 'RMP8','RMP9', 'RMP10', 'RRange','RMeanDer','Rstd','RMeanPeak' ]
    feats = np.hstack([power_feats,range_feat.reshape((40,1)),mean_der.reshape((40,1)),std_feat.reshape((40,1)),peak_avg.reshape((40,1))])
    return feats, names,errors

def extract_T_feats(T,fs=128):
    T_mean = T.mean(1)
    T_std = T.std(1)
    T_der = (np.diff(T,axis=1)*fs).mean(1)
    feats = np.vstack([T_mean,T_std,T_der]).T
    names = ['Tmean','Tstd','T_der']
    return feats,names

def extract_HR_feats(HR,fs=128):
    HR_feats = []
    errors = []
    for i in range(40):
        try:
            wd, m = hp.process(HR[i,:], sample_rate = fs)
            wd = hp.analysis.calc_rr(wd['peaklist'],sample_rate = 128,working_data = wd)
            HRV = np.sqrt(wd['RR_diff'].mean())
            feats = [HRV]
            names=['HRV']
            for key in m.keys():
                feats.append(m[key])
                names.append(key)
        except:
            feats = np.ones((14,))*-1
            errors.append(i)

        HR_feats.append(np.hstack(feats))
    HR_feats = np.vstack(HR_feats)
    return HR_feats, names, errors

def stat_feats(data,fs=128):
    mean = np.mean(data)
    std = np.std(data)
    Skew = skew(data)
    kurt = kurtosis(data)
    mean_fst_absdiff = np.mean(abs(np.diff(data)))
    mean_snd_absdiff = np.mean(abs(np.diff(np.diff(data))))
    mean_fst_diff = np.mean(np.diff(data))
    mean_snd_diff = np.mean(np.diff(np.diff(data)))
    mean_neg_diff = np.mean(np.diff(data)[np.where(np.diff(data)<0)])
    proportion_neg_diff = len(np.where(np.diff(data)<0)[0])/(len(np.diff(data)))
    number_local_min = len(argrelextrema(data,np.less)[0])
    number_local_max = len(argrelextrema(data,np.greater)[0])
    f1 = [mean,std,Skew,kurt]
    f2 = [mean_fst_absdiff,mean_snd_absdiff,mean_fst_diff,mean_snd_diff,mean_neg_diff,proportion_neg_diff]
    f3 = [number_local_min,number_local_max]
    f = f1+f2+f3
    names = ['Gmean','Gstd','Gskew','Gkurtosis','Gme.1absdf','Gme.2absdf','Gme.1df','Gme.2df','Gme.negdf','Gro.negdf','Gnum.argmi','Gnum.argma']
    return names,f

def extract_GSR_feats(gsr,fs=128):
    GSR_feats = []
    for i in range(40):
        names, feats = stat_feats(gsr[i,:])
        GSR_feats.append(feats)
    GSR_feats = np.vstack(GSR_feats)
    return GSR_feats, names

feat_names = ['hEOG','vEOG','zEMG','tEMG','GSR','Resp','Pleth','T']
feats = []
all_errors = dict()
for i in range(1,33):
    with open('D:\\BMED_6517_emotional_state_classifier\\Data\\DEAP\\data_preprocessed_python\\s{:02d}.dat'.format(i),'rb') as file:
        #subj_labels = pickle.load(file,encoding='latin1')
        full = pickle.load(file,encoding='latin1')
        labels = full['labels']
        data_non_eeg = full['data'][:,32:,:]
        data = dict()
        for j in range(len(feat_names)):
            data[feat_names[j]] = data_non_eeg[:,j,:]
    errors = []
    start = 3*128 # first three seconds are before trial starts. Will add this back in if needed for normalization
    R_feats,R_names,R_errors = extract_resp_feats(data['Resp'][:,start:])
    T_feats,T_names = extract_T_feats(data['T'][:,start:])
    HR_feats, HR_names,H_errors = extract_HR_feats(data['Pleth'][:,start:])
    errors = R_errors + H_errors
    GSR_feats, G_names = extract_GSR_feats(data['GSR'][:,start:])
    sub = np.ones((40,1))*i
    names = ['subject'] + R_names + T_names  + G_names
    feats.append(np.hstack([sub,R_feats,T_feats,GSR_feats]))
    if len(errors)>0:
        all_errors[str(i)] = errors

feats = np.vstack(feats)
H_feats = feats[:,18:31]
np.where(H_feats[:,0] == -1)
print(names)