# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:13:53 2021

@author: ariasvts
"""

import os,sys,multiprocessing
from os import path
#-
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
#-
sys.path.append('./Utils')
from Utils import read_data, playsound
#-
sys.path.append('./Measurements')
#-
from PyQt5 import QtWidgets, uic, QtGui, QtCore
#-
from GUI import DialogsWindow,mplwidget,mplwidget_Bottom
from GUI import _SegmentationWindowSettings as segSettings
import logging

class SegWin(QtWidgets.QMainWindow,segSettings.Settings):
    """
    Segmentation window
    """
     # Call the inherited classes __init__ method
    def __init__(self):
        super(SegWin, self).__init__()
        uic.loadUi('./GUI/SegmentationWindow.ui', self) # Load the .ui file
        
        #Variable used to save the temporary path from were signals are being loaded
        self.OpenFilePath = './'
        self.SaveFilePath = './'
        #Dictionary to store all of the voice segments
        self.voiceSegs = {}
        #VAD energy threshold
        self.VAD_Ethr = None
        #Envelope, saturation points, etc...
        self.onsetData = {}
        #-
        ssheet = """QPushButton{background-color: transparent} 
                    QPushButton:hover{background-color: rgb(200,225, 255, 60%); border: 1px solid rgb(160,180, 200, 70%)}"""
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #---------------------Menu bar------------------------------
        #Load file
        self.LoadFile = self.findChild(QtWidgets.QAction,'actionLoad_original_file')
        self.LoadFile.triggered.connect(self.get_standard_files)
        #--------------
        #Load GAT files
        #GAW
        self.LoadGAT_GAW_File = self.findChild(QtWidgets.QAction,'actionGlottal_Area_Waveform')
        self.LoadGAT_GAW_File.triggered.connect(lambda:self.get_GAT_files('GAW_csv'))
        #Trajectories (PVG)
        self.LoadGAT_TRAJ_File = self.findChild(QtWidgets.QAction,'actionTrajectory')
        self.LoadGAT_TRAJ_File.triggered.connect(lambda:self.get_GAT_files('TRAJ_csv'))
        #Save results
        self.mSaveSeg = self.findChild(QtWidgets.QAction,'actionSave_segmentation_file')
        self.mSaveSeg.triggered.connect(self.SaveResults)
        self.btnSaveResults = self.findChild(QtWidgets.QPushButton,"btnSaveResults")
        self.btnSaveResults.clicked.connect(self.SaveResults)
        #-
        #---------------------End Menu bar--------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #-------------Dropdown list for adicht files--------------------------
        self.dropRecords = self.findChild(QtWidgets.QComboBox,"dropRecords")
        self.dropRecords.currentTextChanged.connect(self.RecordChanged)
        #Dropdown list for channels: acoustic, egg, airflow
        self.dropSignals = self.findChild(QtWidgets.QComboBox,"dropSignals")
        self.dropSignals.currentTextChanged.connect(lambda:self.plot_signal())
        # self.dropSignals.currentTextChanged.connect(self.showSegment)
        #--------End List of records in .adicht file--------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #---------------------Settings----------------------------------------
        self.grpSettings = self.findChild(QtWidgets.QGroupBox,"Settings_box")
        self.grpSettings.setEnabled(False)#Enable only when there is something to compute
        self.btnApplySettings = self.findChild(QtWidgets.QPushButton,"Settings_btnApply")
        self.btnApplySettings.clicked.connect(self.apply_settings)
        
        #Filter option
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
        
        #VAD Settings
        self.dropSettVADMethod = self.findChild(QtWidgets.QComboBox,"Settings_VAD_method")
        self.dropSettVADMethod.addItems(segSettings.Settings.get_VAD_Methods(self))
        self.dropSettVADMethod.currentTextChanged.connect(lambda:segSettings.Settings.do_VAD(self))
        
        #Manual segmentation
        self.grpSettVADManualOpt = self.findChild(QtWidgets.QGroupBox,'Settings_VAD_ManualOpt')
        self.grpSettVADManualOpt.setVisible(False)#Make the options for VAD (energy) detection invisible until they are required
        
        #-Energy VAD
        self.grpSettVADEnergOpt = self.findChild(QtWidgets.QGroupBox,'Settings_VAD_EnergyOpt')
        self.grpSettVADEnergOpt.setVisible(False)#Make the options for VAD (energy) detection invisible until they are required
        win = 15
        self.sldSettVADEnerWin = self.findChild(QtWidgets.QSlider,'Settings_VAD_EneWinSize')
        self.sldSettVADEnerWin.setValue(win)
        self.labSettVADEnerWinVal = self.findChild(QtWidgets.QLabel,'Settings_VAD_EneWinSizeVal')
        self.labSettVADEnerWinVal.setText(str(win)+' ms')
        step = 10
        self.sldSettVADEnerStep = self.findChild(QtWidgets.QSlider,'Settings_VAD_EneStepSize')
        self.sldSettVADEnerStep.setValue(step)
        self.sldSettVADEnerStep.setMaximum(self.sldSettVADEnerWin.value())#The step should not be larger than the window size
        self.labSettVADEnerStepVal = self.findChild(QtWidgets.QLabel,'Settings_VAD_EneStepSizeVal')
        self.labSettVADEnerStepVal.setText(str(step)+' ms')
        self.sldSettVADEnerStep.valueChanged.connect(lambda:self.labSettVADEnerStepVal.setText(str(self.sldSettVADEnerStep.value())+' ms'))
        self.sldSettVADEnerWin.valueChanged.connect(lambda:(self.labSettVADEnerWinVal.setText(str(self.sldSettVADEnerWin.value())+' ms'),
                                                            self.sldSettVADEnerStep.setMaximum(self.sldSettVADEnerWin.value())))
        
        #-Voice onset options
        self.dropSettVOMethod = self.findChild(QtWidgets.QComboBox,'Settings_VO_Method')
        self.dropSettVOMethod.addItems(segSettings.Settings.get_VoiceOnset_Methods(self))
        self.grpSettVOMethod = self.findChild(QtWidgets.QGroupBox,'Settings_VoiceOnset_VRT')
        self.grpSettVOMethod.setVisible(False)
        self.dropSettVOMethod.currentTextChanged.connect(lambda:segSettings.Settings.do_VoiceOnset(self))
        #-
        self.sldSettVOSegment = self.findChild(QtWidgets.QSlider,'Settings_VO_Segment')
        seg = 400
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
        
        #---------------------End of settings---------------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #-----------Segmentation buttons--------------------------------------------------
        #Button used for voice activity detection (VAD)
        self.btnEnergyThresh  = self.findChild(QtWidgets.QPushButton,"btnEnergyThresh")
        
        # ssheet_VAD = """QPushButton{background-color: transparent; border: 1px solid rgb(180,180, 180, 100%)} 
                    # QPushButton:hover{background-color: rgb(200,225, 255, 60%); border: 1px solid rgb(80,180, 255, 100%)}"""
        # self.btnEnergyThresh.setStyleSheet(ssheet_VAD)
        self.btnEnergyThresh.setCheckable(True)
        # self.btnEnergyThresh.setEnabled(False)#Enable only when there is something to process
        #-
        #Button used to add a new voice segment manually
        self.btnAddSeg  = self.findChild(QtWidgets.QPushButton,"btnAddSeg")
        # self.btnAddSeg.setStyleSheet(ssheet)
        self.btnAddSeg.clicked.connect(self.AddVoiceSegment)
        self.btnAddSeg.setEnabled(False)#Enable only when there is something to 
        #Button used to delete the selected voice segment
        self.btnDeleteSeg  = self.findChild(QtWidgets.QPushButton,"btnDeleteSeg")
        # self.btnDeleteSeg.setStyleSheet(ssheet)
        self.btnDeleteSeg.clicked.connect(self.DeleteVoiceSegment)
        self.btnDeleteSeg.setEnabled(False)#Enable only when there is something to 
        #Button used to cut parts of the selected signal
        self.btnCutSignal  = self.findChild(QtWidgets.QPushButton,"btnCutSeg")
        self.btnCutSignal.setStyleSheet(ssheet)
        self.btnCutSignal.clicked.connect(self.CutSignal)
        self.btnCutSignal.setEnabled(False)#Enable only when there is something to 
        #-
        #List view with the segmented parts
        self.listSegms = self.findChild(QtWidgets.QListWidget,'listViewSegments')
        self.listSegms.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listSegms.itemSelectionChanged.connect(self.showSegment)
        
        #-
        
        #Edition of voice segments
        #Group box containing the editing options for the voice segments
        self.grpTimeStamps = self.findChild(QtWidgets.QGroupBox,'groupTimeStamps')
        # self.grpTimeStamps.setVisible(False)
        self.grpTimeStamps.setEnabled(False)
        #Name of the segment
        self.lineEditName = self.findChild(QtWidgets.QLineEdit,'lineEditName')
        self.lineEditName.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("\w+")))
        #Start point
        self.lineStart = self.findChild(QtWidgets.QLineEdit,'lineStart')
        # self.lineStart.editingFinished.connect(self.EditSegment)
        self.lineStart.setValidator(QtGui.QDoubleValidator(bottom=0.0000,top=9999.0000,decimals=10,notation=QtGui.QDoubleValidator.StandardNotation))
        #End point
        self.lineEnd = self.findChild(QtWidgets.QLineEdit,'lineEnd')
        # self.lineEnd.editingFinished.connect(self.EditSegment)
        self.lineEnd.setValidator(QtGui.QDoubleValidator(bottom=0.0000,top=9999.0000,decimals=10,notation=QtGui.QDoubleValidator.StandardNotation))
        #Confirmation button
        self.btnConfirmSeg = self.findChild(QtWidgets.QPushButton,"btnConfirmSeg")
        self.btnConfirmSeg.clicked.connect(self.EditSegment)
        # self.btnConfirmSeg.setEnabled(False)
        # self.btnSaveResults.setEnabled(False)#Enable only when there is something to 
        #-----------End Segmentation--------------------------------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------        
        #-----------------------Sound player------------------------------------------
        self.btnPlaySound = self.findChild(QtWidgets.QPushButton,"btnPlaySound")
        self.btnPlaySound.setStyleSheet(ssheet)
        self.btnStopSound = self.findChild(QtWidgets.QPushButton,"btnStopSound")
        self.btnStopSound.setStyleSheet(ssheet)
        self.btnPlaySound.clicked.connect(self.playAudio)
        self.btnStopSound.clicked.connect(self.stopAudio)
        self.btnPlaySound.setEnabled(False)
        self.btnStopSound.setEnabled(False)
        #--------------------End Sound player-------------------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------        
        #-----------Plotting------------------------------------------------------
        # self.btnZoomPlot = self.findChild(QtWidgets.QPushButton,"btnZoomPlot")
        # self.btnZoomPlot.setStyleSheet(ssheet)
        self.btnZoomInPlot = self.findChild(QtWidgets.QPushButton,"btnZoomInPlot")
        self.btnZoomInPlot.setStyleSheet(ssheet)
        self.btnZoomOutPlot = self.findChild(QtWidgets.QPushButton,"btnZoomOutPlot")
        self.btnZoomOutPlot.setStyleSheet(ssheet)
        
        self.buttons_canvas = {'eThresh': self.btnEnergyThresh,
                               'SelItem':self.listSegms,
                               'ZoomIn':self.btnZoomInPlot,
                               'ZoomOut':self.btnZoomOutPlot,
                               'AddSeg':self.btnAddSeg,
                               'DelSeg':self.btnDeleteSeg,
                               'CutSeg':self.btnCutSignal,
                               'PlaySound':self.btnPlaySound,
                               'StopSound':self.btnStopSound}
        self.pltSignal =  mplwidget.MplWidget(self,'pltSignal',self.buttons_canvas)
        self.pltSignal_ZoomIn =  mplwidget_Bottom.MplWidget(self,'pltSignal_ZoomIn')
        #-----------end Plotting-----------------------------------------------
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------
                
    #=========================================================================
    
    def get_standard_files(self):
        #Open dialog window to select files
        FileFormats = "*.adicht *.wav"
        self.getfile(FileFormats=FileFormats)
        #In case the user press "close" instead of "cancel" while attemtinpg to load a file
        if len(self.filepath)>0:
            #Load the data, if possible
            if self.filename.upper().find('.adicht'.upper())!=-1:
                #Read the data in the adicht file
                self.data,self.ListChannels = read_data.read_labchart(self.OpenFilePath+'/'+self.filename)                                              
                #If there is no data in the labchart, then show an error message
                if len(self.ListChannels)==0:                
                    DialogsWindow.ErrorDiagWin('There is no Acoustic mic, EGG, and/or Airflow-Input A  data in the .adicht file')
            #---------------end Read adicht file------------------------------------------------------------
            
            #---------------Read wav file-------------------------------------------------------------------
            elif self.filename.upper().find('.wav'.upper())!=-1:
                #Read wav file
                self.data,self.ListChannels = read_data.read_wav(self.OpenFilePath+'/'+self.filename)
                        
            #Clear lists and plot the selected signal        
            if self.data!=0:
                self.plot_preamble()
            else:
                DialogsWindow.ErrorDiagWin('It was not possible to load the specified wav file')
            
    #---------------------------------------------------------------------
    
    def get_GAT_files(self,opt):
    
        #Open dialog window to select files
        FileFormats = "*.csv"
        self.getfile(FileFormats=FileFormats)
        
        #In case the user press "close" instead of "cancel" while attemtinpg to load a file
        if len(self.filepath)>0:
            #Load GAT the data, if possible
            if self.filename.upper().find('.csv'.upper())!=-1:
                if opt=='GAW_csv':
                    #Read the data in the adicht file
                    self.data,self.ListChannels = read_data.read_GAT_GAW(self.OpenFilePath+'/'+self.filename)                                              
              
                elif opt=='TRAJ_csv': 
                    w = DialogsWindow.GAT_TrajWin()
                    if w.exec_() == QtWidgets.QDialog.Accepted:                    
                        #Read the data in the adicht file
                        self.data,self.ListChannels = read_data.read_GAT_Trajectory(self.OpenFilePath+'/'+self.filename,
                                                                                    sel_traj = w.options['Trajectory_position'],
                                                                                    fs = w.options['Sampling_frequency'])        
                    else: 
                        self.data = 1
                #-
                #Clear lists and plot the selected signal
                if self.data==1:
                    None
                elif self.data!=0:
                    self.plot_preamble()
                else:
                    DialogsWindow.ErrorDiagWin('Is not possible to open the selected file. Please read the documentation and make sure that the format is correct')
            
    #=========================================================================
    
    def getfile(self,FileFormats):
        """
        This function is used to get the Path of the files to be analyzed.

        """
        #--------------------Prepare data path--------------------------------------------------------------------
        #First check if there is a temporal path saved. Otherwise, just open form where VOAT is being called.
        if path.exists(self.OpenFilePath):
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',self.OpenFilePath,"Files ("+FileFormats+")")
        else:
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file','./',"Files ("+self.FileFormats+")")
        #-
        #Get filename+directory
        self.filepath = fname[0]
        #Is there any file to read?
        self.filename = ''
        if len(self.filepath)>0:
            #-
            #Get the path of the files only looking at the first element
            temp_path = self.filepath.split('/')
            #Set the temporal path as the directory used to load the signals
            self.OpenFilePath = self.filepath.replace(temp_path[-1:][0],'')
            #-
            #Create list of names with the selected files
            onlyfile = self.filepath.split('/')
            self.filename = onlyfile[-1]
            #--------------------end Prepare data path---------------------------------------------------------
        
    #=========================================================================
        
    def plot_preamble(self):
        #Clear segmented signal and edit options
        self.ResetSegments()
        #-Clear drop down list of records and signals
        self.dropRecords.clear()
        self.dropSignals.clear()      
        #Display all records and signals in dropdown buttons
        self.dropRecords.addItems(list(self.data.keys()))
        self.dropSignals.addItems(self.ListChannels)
        #-
        #Enbale button for manual segmentation
        self.btnAddSeg.setEnabled(True)
        #Enable button to edit (cut) the signal
        self.btnCutSignal.setEnabled(True)
        #Enable settings box    
        self.grpSettings.setEnabled(True)#Enable only when there is something to compute
        # self.GroupBox_Activation(self.grpSettings,True,color='#86c1e5')
        #-
        #Plot data
        self.plot_signal()
                                  
    #=========================================================================
    
    def plot_signal(self):
        """
        Plot a signal selected by the user

        Returns
        -------
        None.

        """
        #-
        #Get name of the selected item from list
        self.recname = self.dropRecords.currentText()
        self.channel = self.dropSignals.currentText()
        #Make sure that something can be plotted
        if (self.recname!='')and(self.channel!=''):
            #Get signal and plot
            sig = self.data[self.recname][self.channel]['signal']
            fs = self.data[self.recname][self.channel]['fs']    
            self.dursig = len(sig)/fs 
            #-
            #Create temporal wav file for acoustic signal (it is assume that is the first channel. 
            #This is used to reproduce sound
            # if 'channel_1' in self.data[self.recname]:
            try:
                #Enable audio player buttons
                self.btnPlaySound.setEnabled(True)
                #Write audio
                write('./temp_VOAT_audio.wav',int(fs),(sig*2**15).astype(np.int16))
            except:
                #Enable audio player buttons
                self.btnPlaySound.setEnabled(False)
                self.btnStopSound.setEnabled(False)
            #-
            #Plot signal
            self.pltSignal.update_graph(sig,fs,self.filename)
        
    #=========================================================================
    
    def apply_settings(self):
        
        #deselect current item
        # self.listSegms.setCurrentItem(self.listSegms.currentItem(),QtCore.QItemSelectionModel.Deselect)
        
        #Perform the selected Filter method.
        segSettings.Settings.do_filter(self,apply=True)
        
        #Perform the selected VAD method.
        segSettings.Settings.do_VAD(self,apply=True)
        
        #Voice onset detection
        segSettings.Settings.do_VoiceOnset(self,apply=True)
        
        #Update segmentation list and plot the last one
        self.update_segments()
        self.listSegms.setCurrentItem(self.listSegms.currentItem())
        # self.listSegms.setCurrentItem(self.listSegms.item(len(self.voiceSegs)-1))
        
        #Reset filter option to None.
        self.dropSettFilterType.setCurrentText('None')
                        
    #=========================================================================
    
    def update_segments(self):
        """
        Update the list that shows the voice segments

        Returns
        -------
        None.

        """
        self.listSegms.clear()
        self.listSegms.addItems(self.voiceSegs)
        
        #If there are no voice segments, then disable
        #the delete button and hide the editing options
        if len(self.voiceSegs)==0:
            self.btnDeleteSeg.setEnabled(False)
            # self.grpTimeStamps.setEnabled(False)
            self.GroupBox_Activation(self.grpTimeStamps,False)
            # self.grpTimeStamps.setVisible(False)
            
    #=========================================================================
        
    def showSegment(self):
        """
        Highlights the selected voice segment as a colored rectangle in the time signal.

        Returns
        -------
        None.

        """
        try:
            #Clear marks (for example when selecting a segment) in the plot
            self.pltSignal.reset_canvas()
            # self.pltSignal.reset_markers()
            if len(self.listSegms.selectedItems())>0:
                #Activate/Show the group box for edition
                self.GroupBox_Activation(self.grpTimeStamps,True,color='#CCE5FF')#green #7ecf88
                #Enable delete button
                self.btnDeleteSeg.setEnabled(True)
                #-
                currSegm = self.listSegms.currentItem().text()
                self.lineEditName.setText(currSegm)
                #Put the start and end points in the edition boxes
                #start point
                starT = self.voiceSegs[currSegm][0,0]
                self.lineStart.setText(str(np.round(starT,3)))
                #end point
                endT = self.voiceSegs[currSegm][0,1]
                self.lineEnd.setText(str(np.round(endT,3)))
                #Show all selected segments
                segments = []
                for litem in self.listSegms.selectedItems():
                    segments.append(self.voiceSegs[litem.text()])
                #Highlight segment in the plot
                facecolor = ['b','r','g','gold','indigo','orange','cyan','hotpink','lime']
                self.pltSignal.make_error_boxes(np.vstack(segments),facecolor=facecolor)
                #-----------------------------------
                #Plot the zoom in version of the segment
                sig = self.data[self.recname][self.channel]['signal']
                fs = self.data[self.recname][self.channel]['fs'] 
                
                #Display onset for Vocal Rise Time
                if self.onsetData['Type'] == 'VRT'+self.EnvelopeMethod:
                    #Get data
                    env = self.onsetData['Data']['Envelope_'+currSegm]
                    onsetP = self.onsetData['Data']['Onset_'+currSegm][0][0]
                    saturP = self.onsetData['Data']['Onset_'+currSegm][0][1]
                    #-
                    label = self.dropSettVOMethod.currentText()+' - '+currSegm
                    self.pltSignal_ZoomIn.canvas_signal(sig[int(starT*fs):int(starT*fs)+len(env)],fs,label)
                    # Time vector
                    t = np.arange(0,len(env)/fs,1/fs)
                    t = t[0:len(env)]
                    #Plot the envelope
                    self.pltSignal_ZoomIn.canvas_plot(t,env,color='k',linestyle='--',lw=2)
                    #Highlight the onset segment
                    self.pltSignal_ZoomIn.make_error_boxes(np.vstack([onsetP,saturP]).reshape(1,-1),facecolor='blue',alpha=0.2)
                    #-
                    #Display the VRT and plot the onset and saturation points
                    label = self.dropSettVOMethod.currentText()+': '+str(int((saturP-onsetP)*1000))+' ms'
                    # pointxt = 'Onset point: '+str(int(onsetP*1000))+' ms'
                    pointxt = str(int(onsetP*1000))+' ms'
                    self.pltSignal_ZoomIn.canvas_plot(onsetP,self.sldSettVOOnsetTHR.value()/100,color='blue',marker='o',ms=10,pointxt=pointxt)
                    #-
                    # pointxt = 'Saturation point: '+str(int(saturP*1000))+' ms'
                    pointxt = str(int(saturP*1000))+' ms'
                    self.pltSignal_ZoomIn.canvas_plot(saturP,self.sldSettVOSatTHR.value()/100,label=label,color='blue',marker='o',ms=10,pointxt=pointxt)
                    #PLot maximum amplitude
                    self.pltSignal_ZoomIn.canvas_plot(t[env==np.max(env)],np.max(env),label='Max. envelope amplitude',color='b',marker='*',ms=12)                
            #-
            else:
                #disable the group box for edition
                self.GroupBox_Activation(self.grpTimeStamps,False,color='#E0E0E0')
                #Put blank spaces into the line edit
                self.lineEditName.setText('')
                self.lineStart.setText('')
                self.lineEnd.setText('')
                #disable delete button
                self.btnDeleteSeg.setEnabled(False)
                self.pltSignal.reset_canvas()
                self.pltSignal_ZoomIn.clear_canvas()
                
            #Unselect item
            # self.listSegms.setCurrentItem(self.listSegms.currentItem(),QtCore.QItemSelectionModel.Deselect)
        except:
            logging.info('Trying to show a deleted segment in showSegment (SegmentationWindow.py)')
        
    #=========================================================================
        
    def AddVoiceSegment(self):
        """
        Used to add new segments

        Returns
        -------
        None.

        """
        try:
            if np.abs(self.pltSignal.StartP-self.pltSignal.EndP)>0:
                #Put blank spaces into the line edit
                TempSegm ='NewSegment_'+str(len(self.voiceSegs)+1)
                self.lineEditName.setText(TempSegm)
                self.voiceSegs[TempSegm] = np.asarray([[self.pltSignal.StartP,self.pltSignal.EndP]]).reshape(1,-1)
                self.update_segments()
                self.listSegms.setCurrentItem(self.listSegms.item(len(self.voiceSegs)-1))
            else:
                DialogsWindow.ErrorDiagWin('You must select a segment from the figure')
        except:
            DialogsWindow.ErrorDiagWin('You must select a valid segment from the figure')
        
    #=========================================================================
        
    def DeleteVoiceSegment(self):
        """
        Delete the segmented voice segment

        Returns
        -------
        None.

        """
        try:
            #Delete all selected items
            for litem in self.listSegms.selectedItems():
                currSegm = litem.text()
                del self.voiceSegs[currSegm]
            self.update_segments()
        except:
            DialogsWindow.ErrorDiagWin('Please select the item to be deleted')
        #----------------
        self.plot_signal()
        #disable the group box for edition
        self.GroupBox_Activation(self.grpTimeStamps,False,color='#E0E0E0')
        #Put blank spaces into the line edit
        self.lineEditName.setText('')
        self.lineStart.setText('')
        self.lineEnd.setText('')
        #disable delete button
        self.btnDeleteSeg.setEnabled(False)
        
    #=========================================================================
    
    def CutSignal(self):
        """
        Used to cut a segment of the signal

        Returns
        -------
        None.

        """
        if (self.pltSignal.StartP is not None)and(self.pltSignal.EndP is not None):
            #-
            fs = self.data[self.recname][self.channel]['fs']
            #-
            for channel in list(self.data[self.recname].keys()):
                #Temporal signal                
                newsig = self.data[self.recname][channel]['signal'].copy()  
                #-
                if self.pltSignal.EndP > len(newsig):
                    self.pltSignal.EndP = len(newsig-1)
                if self.pltSignal.StartP<0:
                    self.pltSignal.StartP = 0
                #-
                index = np.arange(int(self.pltSignal.StartP*fs),int(self.pltSignal.EndP*fs))
                newsig = np.delete(newsig,index)
                #Copy new signal
                self.data[self.recname][channel]['signal'] = newsig.copy()
            #Reset segments. The time stamp will be invalid if the signal was edited
            self.ResetSegments()
            #Reset plot
            # self.pltSignal.StartP = None
            # self.pltSignal.EndP = None
            # self.pltSignal.pc = None#Reset artist
            self.plot_signal()
            
        # print(self.pltSignal.StartP,self.pltSignal.EndP)
        
    #=========================================================================
    
    def playAudio(self):
        """
        Control how to play sounds.
        """
        
        logging.info('playAudio(): Checking if there is sound already being reproduced') 
        try:
            #This is used to reset the audio in case the user press the play button
            #while something is being reproduced
            if self.player.is_alive():
                #Stop the current audio
                self.stopAudio()
        except:
            self.btnStopSound.setEnabled(False)
        #-
        #Start audio
        self.startAudio()
        
    #=========================================================================
    
    def startAudio(self):
        """
        Start reproducing sound    

        Returns
        -------
        None.

        """
        #Enable stop button
        self.btnStopSound.setEnabled(True)
        #Read speech signal
        sig = self.data[self.recname][self.channel]['signal']
        fs = self.data[self.recname][self.channel]['fs']  
        #Was there a segment selected?
        if (self.pltSignal.StartP is not None)and(self.pltSignal.EndP is not None):
            #-    
            inip = int(float(self.pltSignal.StartP)*1000)
            #If there is a line in the figure, start from Starp
            if self.pltSignal.StartP == self.pltSignal.EndP:
                endp = int(1000*len(sig)/fs)
            #If there is an actual segment selected, then reproduce just the segment
            else:
                endp = int(float(self.pltSignal.EndP)*1000)
        #If not, then reproduce everything
        else:
            inip = 0
            endp = int(1000*len(sig)/fs)
        #-    
        logging.info('startAudio(): Starting sound player')  
        #Start the sound player as a process. This will allow us to stop reproducing sounds (terminate the process) with a button
        self.player = multiprocessing.Process(target=playsound.playsound, args=("./temp_VOAT_audio.wav",str(inip),str(endp),))
        self.player.start()
            
    #=========================================================================
    
    def stopAudio(self):
        """
        Stop playing sound
        """
        self.player.terminate()
        
    #=========================================================================
    
    def EditSegment(self):
        """
        Confirm the time stamps of the edited voice segment

        Returns
        -------
        None.

        """    
        SegName = self.lineEditName.text()
        if SegName!='':
            try:
                currSegm = self.listSegms.currentItem().text()
            except:
                currSegm = SegName
            #-
            starT = self.lineStart.text()
            endT = self.lineEnd.text()
            #Make sure that there are not empyt fields
            if (starT!='')and(endT!=''):
                #In case the user input a comma instead of dot (German keyboard)
                starT = starT.replace(',','.')
                endT = endT.replace(',','.')
                #Convert to float
                starT = np.float(starT)
                endT = np.float(endT)
                #Check that the end time is bigger than the start
                if endT>starT:
                    #Is it an edited segment or is it a completely new segment?
                    if (currSegm in self.voiceSegs): 
                        self.voiceSegs[SegName] = self.voiceSegs.pop(currSegm)
                    else:
                        self.voiceSegs[SegName] = np.asarray([[0.00,0.00]]).reshape(1,-1)
                    #Change name            
                    self.voiceSegs[SegName][0,0] = starT
                    self.voiceSegs[SegName][0,1] = endT
                    #Recompute settings for the new segments
                    self.apply_settings()
                    #Plot new segment
                    self.update_segments()
                    self.listSegms.setCurrentItem(self.listSegms.item(len(self.voiceSegs)-1))
                    self.showSegment()
                        # logging.info('!'*100)
                        # logging.info(currSegm+' was deleted from the list of segments')
                        # logging.info('!'*100)
                else:
                    DialogsWindow.ErrorDiagWin('The "End" time must be higher than the "Start" time')
            else:
                #start point
                starT = self.voiceSegs[currSegm][0,0]
                self.lineStart.setText(str(np.round(starT,3)))
                #end point
                endT = self.voiceSegs[currSegm][0,1]
                self.lineEnd.setText(str(np.round(endT,3)))
        else:
            DialogsWindow.ErrorDiagWin('The selected segment needs a name')
                
    #=========================================================================
    
    def RecordChanged(self):
        """
        If there are several recordings in one file (like in labchart data) then
        reset the selected segments when another record is selected.
        
        Returns
        -------
        None.

        """
        self.ResetSegments()
        self.plot_signal()
    
    #=========================================================================
    
    def ResetSegments(self):
        """
        Reset all segments when a new file is selected

        """
        self.pltSignal.reset_canvas()
        self.voiceSegs = {}
        self.onsetData = {}
        self.lineEditName.setText('')
        self.lineStart.setText('')
        self.lineEnd.setText('')
        self.update_segments()
        
    #=========================================================================
    
    def GroupBox_Activation(self,obj,enable,color='#E0E0E0'):
        """
        Instructions to modify the Group box when is enable/disable
        Returns
        -------
        None.

        """
        obj.setStyleSheet("QGroupBox { background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 "+color+", stop: 1 #FFFFFF);}")
        obj.setEnabled(enable)
        
    #=========================================================================
    
    def SaveResults(self):
        """
        Save a file with the time stamps of the recordings

        Returns
        -------
        None.

        """
        Nsegs = self.listSegms.count()
        #is there any segment to save?
        if Nsegs!=0:
            #--------------------Prepare data path--------------------------------------------------------------------
            #First check if the selected path exist
            if path.exists(self.SaveFilePath):
                fname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a folder',self.SaveFilePath)
            else:
                fname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a folder','./')
            
            self.SaveFilePath = fname
            filename = 'Segmentation_VOAT.xlsx'
            if self.SaveFilePath!='':
                Table = pd.DataFrame()
                
                
                for i in range(Nsegs):
                    SegItem = self.listSegms.item(i)
                    #Get segments name
                    SegName = SegItem.text()
                    
                    #Try to save onset data
                    try:
                        onsetP = self.onsetData['Data']['Onset_'+SegName][0][0]
                        saturP = self.onsetData['Data']['Onset_'+SegName][0][1]
                        
                        voiceonset = int((saturP-onsetP)*1000)
                        
                        
                        #Features
                        X = self.onsetData['Data']['Features_'+SegName]
                        feats = np.asarray(list(X.values()))
                        feat_name = list(X.keys())
                        
                        #Create dataframe and append results
                        df = np.hstack([self.filename,
                                        SegName,
                                        self.onsetData['Settings'],
                                        voiceonset,
                                        feats])
                        Table = pd.concat([Table,pd.DataFrame(df).T],ignore_index=True)
                        #Rename columns
                        cols = {0:'ID',
                                1:'Label',
                                2:'Settings',
                                3:'Voice onset [ms]'}   
                        
                    #-
                    #If is not possible to save the onset data, then save the time stamps
                    #of the segmented recording
                    except:
                        inip = self.voiceSegs[SegName][0,0]
                        endp = self.voiceSegs[SegName][0,1]
                        df = np.hstack([self.filename,
                                        SegName,
                                        inip,
                                        endp])
                        Table = pd.concat([Table,pd.DataFrame(df).T],ignore_index=True)#Rename columns
                        cols = {0:'ID',
                                1:'Label',
                                2:'Settings',
                                3:'Start [seconds]',
                                4:'End [seconds]'}   
                        
                #-
                
                inip = len(cols)
                for i in range(inip,len(feat_name)+inip):
                    cols[i] = feat_name[i-inip]
                    
                Table = Table.rename(columns=cols)
                #Append dataframe if there is already one create at the location
                if os.path.exists(self.SaveFilePath+'/'+filename):
                    Table_Zero = pd.read_excel(self.SaveFilePath+'/'+filename)
                    Table = pd.concat([Table_Zero,Table], ignore_index=True)
                    
                cols = list(Table.columns)
                for cname in cols[3:]:
                    Table[cname] = pd.to_numeric(Table[cname])
                            
                #-
                logging.info('Saving segmentation results in '+self.SaveFilePath+'/'+filename)
                # Table.to_csv(self.SaveFilePath+'/'+filename+'.csv',index=False,sep=';')
                try:
                    Table.to_excel(self.SaveFilePath+'/'+filename,index=False)
                    DialogsWindow.InfoDiagWin('Results saved as "'+self.SaveFilePath+'/'+filename+'"')
                    
                except:
                    DialogsWindow.ErrorDiagWin('The file was not save. Make sure that is not already openned by another application')
                    
        else:
            DialogsWindow.ErrorDiagWin('There are no segments to save')