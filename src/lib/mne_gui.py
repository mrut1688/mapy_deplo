import pandas as pd 
import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt
import os 
from tkinter import *
from tkinter.filedialog import askdirectory,askopenfilename
import matplotlib as mpl
import mne

def ret_filename():
    path_=askdirectory()
    filen=askopenfilename()
    
    return filen,path_

def ica_plots(filen):
    global ica
    global df1
    global sfreq
    global raw1
    plt.style.use('dark_background')
    df1=pd.read_csv(filen,header=None,on_bad_lines='skip')
    df1=df1.drop([0,1,2],axis=1)
    df1.columns=['T7','C3','C4','T8']
    sfreq=len(df1['T7'])/120
    info=mne.create_info(ch_names=['T7','C3','C4','T8'],sfreq=sfreq)
    try:
        raw1=mne.io.RawArray(df1.values.T,info)
    except ValueError :
        df1 = pd.to_numeric(df1.stack(), errors='coerce').unstack().fillna(0)
        raw1=mne.io.RawArray(df1.values.T,info)
    raw1.set_channel_types({'T7':'eeg','C3':'eeg','C4':'eeg','T8':'eeg'})
    raw1.set_montage('standard_1020')
    raw1.filter(4,30,picks=['T7','C3','C4','T8'])
    ica = mne.preprocessing.ICA(n_components=4,max_iter=1000,random_state=99)
    raw1.filter(l_freq=1.0,h_freq=None)
    ica.fit(raw1)
    
    ica.plot_components()
    plt.ion()
    
    ica_plot = ica.plot_components()
    
    plt.style.use('ggplot')
    output_file = 'D:/matlab/ymaps_code/data_visualise/plots_mne/output_eeg.fif.gz'
    raw1.save(output_file, overwrite=True)
    ica.plot_overlay(raw1)
    plt.ion()
    plt.show()

def muscle_state_exdf(sfreq):
    
    
    time10=int(sfreq*10)
    time20=int(sfreq*20)
    time30=time20+time10
    time50=time30+time20
    time110=time50+time20
    time130=time110+time20
    time150=time130+time20
    time200=time150+time10

    musclestate = [0] * time200

    segments =[
    (0, time10, 0),        # rest state of 10s
    (time10, time30, 1),   # right hand slow motion for 20s
    (time30, time50, 2),   # right hand fast motion for 20s
    (time50, time110, 0),  # rest state of 20s
    (time110, time130, 3), # left hand slow motion for 20s
    (time130, time150, 4), # left hand fast motion for 20s
    (time150, time200, 5)  # left hand fast motion for 20s
    ]

    for start, end, state in segments:
       musclestate[start:end] = [state] * (end - start)
    musc_df = pd.DataFrame(musclestate, columns=['Task_code'])
    lookup_table = {0: 'rest', 1: 'rs', 2: 'rf', 3: 'ls', 4: 'lf', 5: 'random'}
    
    musc_df['Task_name']=musc_df['Task_code'].map(lookup_table)
    
    return musc_df



def epoch_plots(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())

    threshold = 0.1  
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    filen_=filen.split('/').pop()
    raw_clean.save(f'D:/matlab/ymaps_code/data/cleaned_data{filen_}.fif', overwrite=True)
    # target = pd.read_csv("D:/DownloadsM/muscle_state_wn.csv")
    # muscle_df = target.drop(['Unnamed: 2', 'Unnamed: 3'], axis = 1)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    

    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    
    
    
    evoked_0 = epochs['Rest'].average()
    # plt.subplot(2,1,1)
    plt.style.use('ggplot')
    evoked_0.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    evoked_0.plot_topomap(times=[0.1, 0.2, 0.3,0.4])
    evoked_1= epochs['right_hand_slow'].average()
    # plot(2,1,1)
    plt.style.use('ggplot')
    evoked_1.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')
    evoked_1.plot_topomap(times=[0.1, 0.2, 0.3,0.4])
    
    
    
    # plt.subplot(2,1,1)
    evoked_2 = epochs['right_hand_fast'].average()
    plt.style.use('ggplot')
    evoked_2.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')
    evoked_2.plot_topomap(times=[0.1, 0.2, 0.3, 0.4])
    
    
    # plt.subplot(2,1,1)
    evoked_3 = epochs['left_hand_slow'].average()
    plt.style.use('ggplot')
    evoked_3.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')
    evoked_3.plot_topomap(times=[0.1, 0.2, 0.3, 0.4])
    
    
    
    # plt.subplot(2,1,1)
    evoked_4 = epochs['left_hand_fast'].average()
    plt.style.use('ggplot')
    evoked_4.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')
    evoked_4.plot_topomap(times=[0.1, 0.2, 0.3, 0.4])
    
    
    
    # plt.subplot(2,1,1)
    evoked_5 = epochs['random_movement'].average()
    plt.style.use('ggplot')
    evoked_5.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')
    evoked_5.plot_topomap(times=[0.1, 0.2, 0.3, 0.4])
    plt.show()
  
def epochp1(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['Rest'].average()
    # plt.subplot(2,1,1)
    plt.style.use('ggplot')
    evoked_0.plot(gfp=True)
    # # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    # evoked_0.plot_topomap(times=np.arange(-0.4,0.4,0.1))
    evoked_0.plot_joint(times=np.arange(-0.4,0.4,0.1))
    
def epochp2(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['right_hand_slow'].average()
    # plt.subplot(2,1,1)
    # plt.style.use('ggplot')
    evoked_0.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    evoked_0.plot_joint(times=np.arange(-0.4,0.4,0.1))   
    
    
def epochp3(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['right_hand_fast'].average()
    # plt.subplot(2,1,1)
    plt.style.use('ggplot')
    evoked_0.plot(gfp=True)
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    evoked_0.plot_joint(times=np.arange(-0.4,0.4,0.1))

def epochp4(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['left_hand_slow'].average()
    # plt.subplot(2,1,1)
    plt.style.use('ggplot')
    evoked_0.plot(gfp=True)
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    # evoked_0.plot_topomap(times=[0.1, 0.2, 0.3,0.4])
    evoked_0.plot_joint(times=np.arange(-0.4,0.4,0.1))
    
def epochp5(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['left_hand_fast'].average()
    # plt.subplot(2,1,1)
    # plt.style.use('ggplot')
    evoked_0.plot(gfp=True)
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    # evoked_0.plot_topomap(times=[0.1, 0.2, 0.3,0.4])
    evoked_0.plot_joint(times=np.arange(-0.4,0.4,0.1))
    
def epochp6(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['random_movement'].average()
    # plt.subplot(2,1,1)
    plt.style.use('ggplot')
    evoked_0.plot(gfp=True)
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    # evoked_0.plot_topomap(times=[0.1, 0.2, 0.3,0.4])
    evoked_0.plot_joint(times=np.arange(-0.4,0.4,0.1))
    

def epochp1_trial(filen):
    global ica
    global df1
    global sfreq
    global raw1
    explained_var = np.abs(ica.get_components())
    threshold = 0.1 
    exclude_components = np.where(explained_var < threshold)[0]
    ica.exclude = exclude_components
    raw_clean = ica.apply(raw1, exclude=ica.exclude)
    target=muscle_state_exdf(sfreq)
    muscle_df = target
    events = [
    [0, 0, 0],  
    [int(sfreq*10), 0, 1],
    [int(sfreq*30), 1, 2],
    [int(sfreq*50), 2, 0],
    [int(sfreq*70), 0, 3],
    [int(sfreq*90), 3, 4],
    [int(sfreq*110), 4, 5]
    ]
    event_id = {
    'Rest': 0,
    'right_hand_slow': 1,
    'right_hand_fast': 2,
    'left_hand_slow': 3,
    'left_hand_fast': 4,
    'random_movement': 5
    } 
    epochs = mne.Epochs(raw1, events, event_id=event_id, tmin=-0.5, tmax=0.5)
    epochs.apply_baseline((None, 0))
    epochs.load_data()
    epochs.filter(l_freq=0.5, h_freq=40)
    evoked=epochs.average()
    evoked_0 = epochs['Rest'].average()
    # plt.subplot(2,1,1)
    # plt.style.use('ggplot')
    evoked_0.plot()
    # plt.subplot(2,1,2)
    plt.style.use('dark_background')      
    # evoked_0.plot_topomap(times=[-0.4,-0.3,-0.2,-0.1,0,0.1, 0.2, 0.3,0.4])
    fig, anim = evoked_0.animate_topomap(times=np.arange(-0.4,0.4,0.05), frame_rate=2, blit=False)
    plt.show()
    anim.save('D:/matlab/animation.')
    
    
if  __name__=="__main__" :
    
    filen=askopenfilename()
    ica_plots(filen)
    epochp1_trial(filen)