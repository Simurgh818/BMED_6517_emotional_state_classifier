import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.fft import fft
import scipy.signal as signal
from scipy.stats import pearsonr
import torch

def extract_eeg_bands(data, fs=128):
    # filtering
    b_alpha,a_alpha = signal.butter(3,(1,7),'bandpass',fs=fs)
    b_beta,a_beta = signal.butter(3,(8,13),'bandpass',fs=fs)
    b_theta,a_theta = signal.butter(3,(14,30),'bandpass',fs=fs)
    b_gamma,a_gamma = signal.butter(3,(30,45),'bandpass',fs=fs)

    alpha = signal.filtfilt(b_alpha,a_alpha,data,axis=2)
    beta = signal.filtfilt(b_beta,a_beta,data,axis=2)
    theta = signal.filtfilt(b_theta,a_theta,data,axis=2)
    gamma = signal.filtfilt(b_gamma,a_gamma,data,axis=2)

    return [alpha,beta,theta,gamma]

def window_data(data,labels,window=8,step=4,fs=128):
    split_data = []
    split_labels = []
    window_size = window*fs

    k = 0
    for band in data:
        short_trial = []
        short_labels = []
        start = 0
        end = window*fs
        step_idx = step*fs
        while end < band.shape[2]:
            short_trial.append(band[:,:,start:end])
            start+=step_idx
            end+=step_idx
            if k == 0:
                short_labels.append(labels)
        split_data.append(np.vstack(short_trial))
        if k == 0:
            split_labels.append(np.vstack(short_labels))
            k+=1
    split_data = np.stack(split_data,1)
    split_labels = np.vstack(split_labels)
    return split_data,split_labels

def PCC(data):
    corr_mat = np.zeros((data.shape[0],data.shape[1],data.shape[1]))
    for b in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                r, p = pearsonr(data[b,i,:], data[b,j,:])
                corr_mat[b,i,j] = r
    return corr_mat

def extract_PCC(data):
    feats = []
    for trial in data:
        feats.append(PCC(trial))
    return np.stack(feats)

full_labels = []
feats = []
for i in range(1,32): # iterate over subjects
    with open('D:\\BMED_6517_emotional_state_classifier\\Data\\DEAP\\data_preprocessed_python\\s{:02d}.dat'.format(i),'rb') as file:
        #subj_labels = pickle.load(file,encoding='latin1')
        full = pickle.load(file,encoding='latin1')
        labels = full['labels'][:,:2]
        data = full['data'][:,:32,:]
        data = data[:,:,3*128:]
        print(data.shape)

    split_data = extract_eeg_bands(data)
    windowed_split_data, split_labels = window_data(split_data,labels)
    feats.append(extract_PCC(windowed_split_data))
    full_labels.append(split_labels)

feats = np.vstack(feats)
full_labels = np.vstack(full_labels)


print(feats.shape)
print(full_labels.shape)
np.save('feats_win8_4.npy',feats)
np.save('labels_win8_4.npy',full_labels)