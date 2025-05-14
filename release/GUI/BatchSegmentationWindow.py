# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:13:53 2021

@author: ariasvts
"""

import os,sys
from os import path
#-
import numpy as np
import pandas as pd
#-
sys.path.append('./Utils')
from Utils import read_data
from Utils.preprocessing import Voice_Activity_Detection as VAD
#-
sys.path.append('./Measurements')
#-
from PyQt5 import QtWidgets, uic, QtGui
#-
from GUI import DialogsWindow
from GUI import _BatchSegmentationWindowSettings as segSettings
import logging
logging.basicConfig(filename='Batch_Log.log', level=logging.INFO)

class BatchSegWin(QtWidgets.QMainWindow,segSettings.Settings):
    """
    Automatic segmentation of a batch of files window
    """
     # Call the inherited classes __init__ method
    def __init__(self):
        super(BatchSegWin, self).__init__()
        uic.loadUi('./GUI/BatchSegmentationWindow.ui', self) # Load the .ui file
        
        #Variable used to save the temporary path from were signals are being loaded
        self.OpenFilePath = ''
        self.SaveFilePath = './'
        self.FileFormat = ''#wav, adicht or csv (generated with GAT)
        self.ListFiles = [] #List with the name of the files to load
        
        #Dictionary to store all of the voice segments
        self.voiceSegs = {}
        #VAD energy threshold
        self.VAD_Ethr = None
        #Envelope, saturation points, etc...
        self.onsetData = {}
                
        #----------------------------------------------------------------------
        #-----------------    Load file    --------------------
        #----------------------------------------------------------------------
        self.LoadFile = self.findChild(QtWidgets.QAction,'actionWAV_files')
        self.LoadFile.triggered.connect(lambda:self.getfolder('.wav'))
        #-
        self.LoadFile = self.findChild(QtWidgets.QAction,'actionADICHT_files_2')
        self.LoadFile.triggered.connect(lambda:self.getfolder('.adicht'))
        #-
        self.LoadFile = self.findChild(QtWidgets.QAction,'actionGAW_GAT_files')
        self.LoadFile.triggered.connect(lambda:self.getfolder('.csv','GAW_csv'))
        #-
        #-----------------    GAT Trajectory    --------------------
        self.LoadFile = self.findChild(QtWidgets.QAction,'actionTrajectory_GAT_files')
        self.LoadFile.triggered.connect(lambda:self.getfolder('.csv','TRAJ_csv'))
        #Group box
        self.grpGATTraj = self.findChild(QtWidgets.QGroupBox,'groupBox_GAT_Trajectory')
        self.grpGATTraj.setVisible(False)#Make invisible until they are required
        #Slider and label
        self.sldPos = self.findChild(QtWidgets.QSlider,'sldGATTrajectoryPos')
        seg = 50
        self.sldPos.setValue(seg)
        #Label
        self.labPos = self.findChild(QtWidgets.QLabel,'labelGATTrajPos')
        self.labPos.setText(str(seg)+' %')
        self.sldPos.valueChanged.connect(lambda:self.labPos.setText(str(self.sldPos.value())+' %'))
        #Sampling frequency
        self.dropFS = self.findChild(QtWidgets.QComboBox,'box_SamplingFreq')
        self.dropFS.addItems(['1000','2000','4000','6000','8000'])
        self.dropFS.setCurrentText('4000')
        
        
        #Dropdown list for channels: acoustic, egg, airflow
        self.dropSignals = self.findChild(QtWidgets.QComboBox,"dropSignals")

        
        self.grpSettFilterOpt = self.findChild(QtWidgets.QGroupBox,'Settings_box')
        
        #----------------------------------------------------------------------
        #-----------------    Filter option    --------------------
        #----------------------------------------------------------------------
        self.grpSettFilterOpt = self.findChild(QtWidgets.QGroupBox,'Settings_Filter_Bandopt')
        self.grpSettFilterOpt.setVisible(False)#Make the options for VAD (energy) detection invisible until they are required
        self.grpSettFilterHiLoOpt = self.findChild(QtWidgets.QGroupBox,'Settings_Filter_HiLoopt')
        self.grpSettFilterHiLoOpt.setVisible(False)#Make the options for VAD (energy) detection invisible until they are required
        self.dropSettFilterType = self.findChild(QtWidgets.QComboBox,'Settings_Filter_Type')
        self.dropSettFilterType.addItems(['None','bandpass','highpass','lowpass'])
        self.dropSettFilterType.currentTextChanged.connect(lambda:segSettings.Settings.do_filter(self))
        self.linefcut = self.findChild(QtWidgets.QLineEdit,'Settings_Filter_Cutoff')
        self.linefcut.setValidator(QtGui.QIntValidator(10,8000))
        self.linef1 = self.findChild(QtWidgets.QLineEdit,'Settings_Filter_Freq1')
        self.linef1.setValidator(QtGui.QIntValidator(10,8000))
        self.linef2 = self.findChild(QtWidgets.QLineEdit,'Settings_Filter_Freq2')
        self.linef2.setValidator(QtGui.QIntValidator(10,8000))
        self.linef2.setText('75')
        self.linef1.setText('300')
        self.linefcut.setText('300')

        #----------------------------------------------------------------------
        #-----------------    VAD Settings    --------------------
        #----------------------------------------------------------------------
        self.VADMeth = VAD.VAD_methods()
        self.dropSettVADMethod = self.findChild(QtWidgets.QComboBox,"Settings_VAD_method")
        self.dropSettVADMethod.addItems(['None','Energy'])
        self.dropSettVADMethod.currentTextChanged.connect(lambda:segSettings.Settings.do_VAD(self))
        #-
        self.grpSettVADEnergOpt = self.findChild(QtWidgets.QGroupBox,'Settings_VAD_EnergyOpt')
        self.grpSettVADEnergOpt.setVisible(False)#Make the options for VAD (energy) detection invisible until they are required
        win = 15
        self.sldSettVADEnerWin = self.findChild(QtWidgets.QSlider,'Settings_VAD_EneWinSize')
        self.sldSettVADEnerWin.setValue(win)
        self.labSettVADEnerWinVal = self.findChild(QtWidgets.QLabel,'Settings_VAD_EneWinSizeVal')
        self.labSettVADEnerWinVal.setText(str(win)+' ms')
        #-
        step = 10
        self.sldSettVADEnerStep = self.findChild(QtWidgets.QSlider,'Settings_VAD_EneStepSize')
        self.sldSettVADEnerStep.setValue(step)
        self.labSettVADEnerStepVal = self.findChild(QtWidgets.QLabel,'Settings_VAD_EneStepSizeVal')
        self.labSettVADEnerStepVal.setText(str(step)+' ms')
        #-
        self.sldSettVADEnerStep.valueChanged.connect(lambda:self.labSettVADEnerStepVal.setText(str(self.sldSettVADEnerStep.value())+' ms'))
        self.sldSettVADEnerStep.setMaximum(self.sldSettVADEnerWin.value())#The step should not be larger than the window size
        self.sldSettVADEnerWin.valueChanged.connect(lambda:(self.labSettVADEnerWinVal.setText(str(self.sldSettVADEnerWin.value())+' ms'),
                                                            self.sldSettVADEnerStep.setMaximum(self.sldSettVADEnerWin.value())))
        
        #----------------------------------------------------------------------
        #-----------------    Voice onset options    --------------------
        #----------------------------------------------------------------------
        self.dropSettVOMethod = self.findChild(QtWidgets.QComboBox,'Settings_VO_Method')
        self.dropSettVOMethod.addItems(segSettings.Settings.get_VoiceOnset_Methods(self))
        self.grpSettVOMethod = self.findChild(QtWidgets.QGroupBox,'Settings_VoiceOnset_VRT')
        self.grpSettVOMethod.setVisible(False)
        self.dropSettVOMethod.currentTextChanged.connect(lambda:segSettings.Settings.do_VoiceOnset(self))
        #-
        self.sldSettVOSegment = self.findChild(QtWidgets.QSlider,'Settings_VO_Segment')
        seg = 200
        self.sldSettVOSegment.setValue(seg)
        self.labSettVOSegment = self.findChild(QtWidgets.QLabel,'Settings_VO_SegmentLabel')
        self.labSettVOSegment.setText(str(seg)+' ms')
        self.sldSettVOSegment.valueChanged.connect(lambda:self.labSettVOSegment.setText(str(self.sldSettVOSegment.value())+' ms'))
        self.sldSettVOOnsetTHR = self.findChild(QtWidgets.QSlider,'Settings_VO_OnsetTHR')
        self.labSettVOOnsetTHR = self.findChild(QtWidgets.QLabel,'Settings_VO_OnLabel')
        self.sldSettVOSatTHR = self.findChild(QtWidgets.QSlider,'Settings_VO_SatTHR')
        self.labSettVOSatTHR = self.findChild(QtWidgets.QLabel,'Settings_VO_SatLabel')
        Othr = 10
        Sthr = 90
        self.sldSettVOOnsetTHR.setValue(Othr)
        self.sldSettVOSatTHR.setValue(Sthr)
        self.labSettVOOnsetTHR.setText(str(Othr)+'%')
        self.labSettVOSatTHR.setText(str(Sthr)+'%')
        self.sldSettVOOnsetTHR.valueChanged.connect(lambda:self.labSettVOOnsetTHR.setText(str(self.sldSettVOOnsetTHR.value())+'%'))
        self.sldSettVOSatTHR.valueChanged.connect(lambda:self.labSettVOSatTHR.setText(str(self.sldSettVOSatTHR.value())+'%'))
        
        #-Envelope method
        self.dropSettVOEnvelope = self.findChild(QtWidgets.QComboBox,'Settings_VO_Envelope')
        self.dropSettVOEnvelope.addItems(segSettings.Settings.get_Envelope_Methods(self))
        self.dropSettVOEnvelope.currentTextChanged.connect(lambda:segSettings.Settings.do_VoiceOnset(self))
        self.labSettEnvWarning = self.findChild(QtWidgets.QLabel,'label_envelopewarn')
        self.labSettEnvWarning.setText('')
        #-
        #Smoothing factor
        self.sldSettEnvSmooth = self.findChild(QtWidgets.QSlider,'Settings_VO_SmoothFactor')
        val = 3
        self.sldSettEnvSmooth.setValue(val)
        self.labSettEnvSmooth = self.findChild(QtWidgets.QLabel,'Settings_VO_SmoothLabel')
        self.labSettEnvSmooth.setText(str(val))
        self.sldSettEnvSmooth.valueChanged.connect(lambda:self.labSettEnvSmooth.setText(str(self.sldSettEnvSmooth.value())))
        
        
        #----------------------------------------------------------------------
        #-----------------    Button apply    --------------------
        #----------------------------------------------------------------------
        self.btnApplySettings = self.findChild(QtWidgets.QPushButton,"Settings_btnApply")
        self.btnApplySettings.clicked.connect(self.apply_settings)
        #-
        #Progress bar
        self.barProgressBatch = self.findChild(QtWidgets.QProgressBar,"progressBar")
        self.barProgressBatch.setVisible(False)
        
        #-----------------    Button SavePath    --------------------
        self.btnSavePath = self.findChild(QtWidgets.QPushButton,"btn_SaveFolder")
        self.btnSavePath.clicked.connect(self.SetSavePath)
        self.lineSavePath = self.findChild(QtWidgets.QLineEdit,'line_SaveFolder')
        
        #----------------------------------------------------------------------
        
    def getfolder(self,fileformat,opt=''):
        """
        This function is used to get the Path of the files to be analyzed.   
        
        fileformat: wav, adicht or csv (generated with GAT)
        opt: extra file options. For example, the .csv can be GAW or Trajectory, both generated with GAT
        """
        #--------------------Prepare data path-----------------------------
        #First check if the selected path exist
        if path.exists(self.OpenFilePath):
            filepath = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:',self.OpenFilePath, QtWidgets.QFileDialog.ShowDirsOnly)
        else:
            filepath = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', './', QtWidgets.QFileDialog.ShowDirsOnly)
    
        #Get filename+directory
        self.dropSignals.clear()
        self.OpenFilePath = filepath
        self.FileFormat = fileformat
        self.FileOpt = opt
        if self.OpenFilePath!='':
            #-
            temp_files = os.listdir(self.OpenFilePath)
            
            #Create list with the files in the supported format
            self.ListFiles = []
            for ifile in temp_files:
                if ifile.upper().find(self.FileFormat.upper())!=-1:
                    self.ListFiles.append(ifile)
            #-    
            #Show a message to the user with the number of files found in the folder
            if len(self.ListFiles)!=0:
                DialogsWindow.InfoDiagWin(str(len(self.ListFiles))+' '+self.FileFormat+' files were found in '+self.OpenFilePath)
                
                #Allow user to select signal to be analyzed'
                
                #Available channels
                if self.FileFormat == '.adicht':
                    self.dropSignals.addItems(['Acoustic mic','EGG','Airflow-Input A'])
                #-
                elif self.FileFormat == '.wav':
                    self.dropSignals.addItems(['channel_1','channel_2'])
                #-
                elif self.FileFormat == '.csv':
                        if opt=='GAW_csv':
                            self.dropSignals.addItems(['Total GAW','Left GAW','Right GAW'])
                        elif opt=='TRAJ_csv': 
                            self.dropSignals.addItems(['Left Trajectory','Right Trajectory'])
                
                if opt=='TRAJ_csv':
                    segSettings.Settings.box_properties(self,self.grpGATTraj)
                    # self.grpGATTraj.setVisible(True)#Make invisible until they are required
                    
                else:
                    self.grpGATTraj.setVisible(False)#Make invisible until they are required
            #---
            #
            #------                    
            else:
                DialogsWindow.ErrorDiagWin('0 '+self.FileFormat+' files were found in '+self.OpenFilePath)
    
    #----------------------------------------------------------------------
    
    def get_standard_files(self):
        #Open dialog window to select files
        #Load the data, if possible
        if self.filename.upper().find('.adicht'.upper())!=-1:
            #Read the data in the adicht file
            self.data,_ = read_data.read_labchart(self.OpenFilePath+'/'+self.filename) 
                               
            #---------------end Read adicht file------------------------------------------------------------
            
        #---------------Read wav file-------------------------------------------------------------------
        elif self.filename.upper().find('.wav'.upper())!=-1:
            #Read wav file
            self.data,_ = read_data.read_wav(self.OpenFilePath+'/'+self.filename)
            
    #---------------------------------------------------------------------
    
    def get_GAT_files(self,opt):
        #Load GAT the data, if possible
        if self.filename.upper().find('.csv'.upper())!=-1:
            if opt=='GAW_csv':
                self.data,_ = read_data.read_GAT_GAW(self.OpenFilePath+'/'+self.filename)  
                #-                                      
            #-
            elif opt=='TRAJ_csv': 
                self.data,_ = read_data.read_GAT_Trajectory(self.OpenFilePath+'/'+self.filename,
                                                                                sel_traj = 50,
                                                                                fs = 4000)        
    #----------------------------------------------------------------------
    
    def apply_settings(self):
        """
        This is what is going to happen after clicking on the "Apply" button.
        Returns
        -------
        None.

        """        
        #-
        
        #Force the user to select a save path
        if self.SaveFilePath=='./':
            DialogsWindow.ErrorDiagWin('Please select the folder where you want to save the results')
        else:
            #Check if there is there is a folder to open
            if (self.OpenFilePath!='')or(len(self.ListFiles)!=0):
                savefilename = 'BatchSegmentation_VOAT.xlsx'#For the excel file
                self.recname = 'record_1'#Because adicht file can have multiple signal in one file. Lets take the first one for now
                self.channel = self.dropSignals.currentText()#The channel selected by the user
                self.filename = ''
                self.barProgressBatch.setVisible(True)
                idxbar = 0
                for ifile in self.ListFiles:
                    self.filename = ifile
                            
                    #Dictionary to store all of the voice segments
                    self.voiceSegs = {}
                    #Envelope, saturation points, etc...
                    self.onsetData = {}
                    
                    try:
                        if (self.FileFormat.upper().find('.wav'.upper())!=-1)or(self.FileFormat.upper().find('.adicht'.upper())!=-1):
                            self.get_standard_files()
                        else:
                            self.get_GAT_files(self.FileOpt)
                        
                        #Perform the selected Filter method.
                        segSettings.Settings.do_filter(self,apply=True)
                        
                        #Perform the selected VAD method.
                        segSettings.Settings.do_VAD(self,apply=True)
                        
                        #Voice onset detection
                        segSettings.Settings.do_VoiceOnset(self,apply=True)
                        
                        #Save Excel file
                        saveflag = self.SaveResults(savefilename)
                        
                        #Progression bar
                        idxbar+=1
                        self.barProgressBatch.setValue((idxbar/len(self.ListFiles))*100)
                        
                    except:
                        saveflag=False
                    
                if saveflag==True:
                    DialogsWindow.InfoDiagWin('Results saved as "'+self.SaveFilePath+'/'+savefilename+'"')
                elif saveflag==False:
                    DialogsWindow.ErrorDiagWin('The results were not saved because the excel file BatchSegmentation_VOAT.xlsx is open by another application')      
            else:
                DialogsWindow.ErrorDiagWin('You need to select a folder containing the files to be processed.')
            
                
    #----------------------------------------------------------------------
                
    def SaveResults(self,savefilename = 'BatchSegmentation_VOAT.xlsx'):  
        """
        Save a file with the time stamps of the recordings, including absolute start and end times from the voice activity detection results.

        Returns
        -------
        save: Boolean. Whether it was possible to save the file or not
        """
        
        Nsegs = len(self.voiceSegs)
        # print(f"Number of segments to process: {Nsegs}")  # Debugging line
        
        if Nsegs!=0:
            if self.SaveFilePath!='':
                Table = pd.DataFrame()
                for SegName in self.voiceSegs:
                    try:
                        # Debugging: print the SegName to be processed
                        # print(f"Processing segment: {SegName}")
                        
                        #JH - print the content of self.voiceSegs and self.onsetData:
                        # print(f"voiceSegs: {self.voiceSegs}")
                        # print(f"onsetData: {self.onsetData}")
                        
                        #onset_key = 'Onset_' + SegName
                        #if onset_key not in self.onsetData['Data']:
                        #    print(f"Onset data key '{onset_key}' not found in onsetData")
                        #    raise KeyError(f"Onset data key '{onset_key}' not found")
                        
                        #onsetP = int(self.onsetData['Data'][onset_key][0][0] * 1000)
                        #saturP = int(self.onsetData['Data'][onset_key][0][1] * 1000)
                        onsetP = int(self.onsetData['Data']['Onset_'+SegName][0][0]*1000)
                        saturP = int(self.onsetData['Data']['Onset_'+SegName][0][1]*1000)
                        
                        #Features
                        X = self.onsetData['Data']['Features_'+SegName]
                        feats = np.asarray(list(X.values()))
                        feat_name = list(X.keys())

                        #JH - Segment start and end times
                        SegStart = int(self.voiceSegs[SegName][0][0]*1000)
                        SegEnd = int(self.voiceSegs[SegName][0][1]*1000)
                        
                        # Debugging: print onsetP and saturP values
                        # print(f"onsetP: {onsetP}, saturP: {saturP}")
                        
                        # Debugging: print SegStart and SegEnd values
                        # print(f"SegStart: {SegStart}, SegEnd: {SegEnd}")
                        
                        voiceonset = saturP-onsetP
                        SegDur = SegEnd-SegStart
                        
                        # Debugging: print values used for creating df
                        # print(f"Creating df with values: {[self.filename, SegName, self.onsetData['Settings'], onsetP, saturP, voiceonset, SegStart, SegEnd, SegDur]}")
                        
                        df = np.hstack([self.filename,
                                        SegName,
                                        self.onsetData['Settings'],
                                        onsetP,
                                        saturP,
                                        voiceonset,
                                        SegStart,
                                        SegEnd,
                                        SegDur,
                                        feats])
                        
                    except:
#                    except Exception as e:
                        # Detailed exception logging
                        # print(f"Exception occurred: {e}")
                        # print(f"Filename: {self.filename}, Segment: {SegName}")
                        
                        df = np.hstack([self.filename,
                                        SegName,
                                        self.onsetData['Settings'],
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0])  
                    #-
                    Table = pd.concat([Table,pd.DataFrame(df).T],ignore_index=True)#Rename columns
                    
                cols = {0:'ID',
                        1:'Label',
                        2:'Settings',
                        3:'Start [ms]',
                        4:'End [ms]',
                        5:'Voice onset [ms]',
                        6:'Segment start [ms]',
                        7:'Segment end [ms]',
                        8:'Segment duration [ms]'}
                
                inip = len(cols)
                for i in range(inip,len(feat_name)+inip):
                    cols[i] = feat_name[i-inip]
                
                Table = Table.rename(columns=cols)

                try:
                    #Append dataframe if there is already one create at the location
                    if os.path.exists(self.SaveFilePath+'/'+savefilename):
                        Table_Zero = pd.read_excel(self.SaveFilePath+'/'+savefilename)
                        Table = pd.concat([Table_Zero,Table], ignore_index=True)
                        
                    for cname in ['Start [ms]','End [ms]','Voice onset [ms]','Segment start [ms]','Segment end [ms]', 'Segment duration [ms]']:
                        Table[cname] = pd.to_numeric(Table[cname])
                        
                    Table.to_excel(self.SaveFilePath+'/'+savefilename,index=False)
                    save = True
                except:
#                except Exception as e:
                    # print(f"Exception occurred while saving the file: {e}")
                    save = False
        else:
            save = None
        return save
        #-----------------------------------------------------------------------------------
    def SetSavePath(self):
        #-----------------Prepare data path--------------------------------------
        #First check if the selected path exist
        fname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a folder to save the results',self.SaveFilePath)
            
        self.SaveFilePath = fname
        self.lineSavePath.setText(self.SaveFilePath)