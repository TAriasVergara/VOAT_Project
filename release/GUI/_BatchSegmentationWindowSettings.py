# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:01:46 2022

@author: ariasvts
"""


import sys
sys.path.append('./Utils')
from Utils.preprocessing import Voice_Activity_Detection as VAD
from Utils.preprocessing import Signal_processing as sigpro
#-
sys.path.append('./Measurements')
from Measurements.Voice_onset import main_Onsets as vOnsets
#-
import scipy as sp
import numpy as np
#-
from GUI import DialogsWindow

class Settings:
    
    #============================================================
    
    def get_VAD_Methods(self):
        self.VADMeth = VAD.VAD_methods()
        return list(self.VADMeth.keys())
    
    #============================================================
    
    def get_VoiceOnset_Methods(self):
        """
        Get the name of the available voice onset methods
        """
        self.VOMeths = vOnsets.VoiceOnset_Methods()
        return list(self.VOMeths.keys())
    
    #============================================================
    
    def get_Envelope_Methods(self):
        """
        Get the name of the available envelope methods
        """
        self.EnvelMeth = vOnsets.Envelope_Methods()
        return list(self.EnvelMeth.keys())
        
    #============================================================
    
    def reset_visibility_VAD(self):
        """
        Make the individual options for each VAD method invisible again. This must
        be done everything a new method is selected in order to enbale only the options
        valid for the current method.

        Returns
        -------
        None.

        """
        self.grpSettVADEnergOpt.setVisible(False)#Make the options for VAD (energy) detection invisible until they are required
                    
    #============================================================
    
    def reset_visibility_Filter(self):
        """
        Make the individual options for the filter invisible again. This must
        be done everything a new method is selected in order to enbale only the options
        valid for the current method.

        Returns
        -------
        None.

        """
        self.grpSettFilterOpt.setVisible(False)
        self.grpSettFilterHiLoOpt.setVisible(False)
                    
    #============================================================
        
    def do_VAD(self,apply=False):
        """
        Voice activity detection method.
        
        It can be energy-based

        Returns
        -------
        None.

        """
        #Reset option visibility in the GUI
        self.reset_visibility_VAD()
        #Get the selected VAD method
        self.VADmethod = self.dropSettVADMethod.currentText()
        
        #Is it different than None?
        if self.VADmethod != 'None':
            #-
            #-
            #----------------------------------------------------------------
            #Voice activity detection using energy
            #----------------------------------------------------------------
            if self.VADmethod == 'Energy':
                #Display options when the method is selected from dropdown
                # self.box_properties(self.grpSettVADEnergOpt)
                #Only perform VAD when the button is pressed
                if apply:
                    #Perform VAD on the current signal
                    sig = self.data[self.recname][self.channel]['signal']
                    fs = self.data[self.recname][self.channel]['fs']
                    #Slider values for window size and step size
                    win = self.sldSettVADEnerWin.value()
                    step = self.sldSettVADEnerStep.value()
                    #-
                    times,self.VAD_Ethr = self.VADMeth['Energy'](sig,fs,win=win/1000,step=step/1000,VAD_thr=None)
                    if len(times)!=0:
                        for i in range(len(times)):
                            self.voiceSegs['segment_'+str(i+1)] = times[i].reshape(1,-1)
                        #Refresh plot with new values
                        # self.update_segments()
                    #-
                    # else:
                        # DialogsWindow.ErrorDiagWin('Segments not found')
                    
        # else:
            # #Only do something when the button is pressed
            # if apply:
            #     self.ResetSegments()
                
                
    #============================================================
    
    def do_filter(self,apply=False):
        #Reset visibility
        self.reset_visibility_Filter()
        if self.dropSettFilterType.currentText() != 'None':
            #-
            if self.dropSettFilterType.currentText() == 'bandpass':
                #-
                #Activate bandpass filter options
                #-
                self.box_properties(self.grpSettFilterOpt)
                if apply:
                    fLow = int(self.linef2.text())
                    fHigh =int(self.linef1.text())
                    self.data[self.recname][self.channel]['signal'] = sigpro.filtering(self.data[self.recname][self.channel]['signal'],
                                                                                       self.data[self.recname][self.channel]['fs'],
                                                                                       [fLow,fHigh],
                                                                                       ftype=self.dropSettFilterType.currentText())
            #-
            else:
                #-
                #Activate high/low pass filter options
                #-
                self.box_properties(self.grpSettFilterHiLoOpt)
                if apply:
                    cutoff = int(self.linefcut.text())
                    self.data[self.recname][self.channel]['signal'] = sigpro.filtering(self.data[self.recname][self.channel]['signal'],
                                                                                       self.data[self.recname][self.channel]['fs'],
                                                                                       cutoff,
                                                                                       ftype=self.dropSettFilterType.currentText())
           
    #============================================================
    
    def do_VoiceOnset(self,apply=False):
        """
        Returns
        -------
        None.

        """
        self.VoiceOnsetmeth = self.dropSettVOMethod.currentText()
        
        #Is it different than None?
        if self.VoiceOnsetmeth != 'None':
                #-
                self.box_properties(self.grpSettVOMethod)
                self.EnvelopeMethod = self.dropSettVOEnvelope.currentText()
                
                if  self.EnvelopeMethod == 'Peak Amplitude':
                    # self.dropSettFilterType.setCurrentText ('bandpass')
                    self.labSettEnvWarning.setText('Warning: This method requires bandpass filtering')
                else:                    
                    self.labSettEnvWarning.setText('')
                    # self.dropSettFilterType.setCurrentText('None')
                
                #-
                if apply:  
                    #Load signal
                    sig = self.data[self.recname][self.channel]['signal']
                    lsig = len(sig)
                    fs = self.data[self.recname][self.channel]['fs']#Sampling frequency
                    #-
                    #If there are no VAD segments, then used the complete signal
                    if (len(self.voiceSegs.values())==0)and(self.VADmethod == 'None'):
                        self.voiceSegs['signal'] = np.asarray([[0.00,float(lsig/fs)]]).reshape(1,-1)  
                    time_stamps = list(self.voiceSegs.values())
                    labels = list(self.voiceSegs.keys())
                    #----------------------------------------------------------------
                    #----------------------------------------------------------------
                    #                       Vocal Rise Time
                    #----------------------------------------------------------------
                    #----------------------------------------------------------------
                    if self.VoiceOnsetmeth == 'Vocal Rise Time':
                        self.onsetData['Type'] = 'VRT'+self.EnvelopeMethod
                        segment = self.sldSettVOSegment.value()/1000
                        #Onset and saturation points for VRT
                        onTHR = self.sldSettVOOnsetTHR.value()/100
                        satTHR = self.sldSettVOSatTHR.value()/100
                        kargs = {'Onset point':onTHR,
                                  'Saturation point':satTHR,
                                  'Segment':segment,
                                  'Method':self.EnvelopeMethod,
                                  'Smooth factor':self.sldSettEnvSmooth.value()
                                  }
                        
                        #-
                        self.onsetData['Data'] = self.VOMeths['Vocal Rise Time'](sig,fs,time_stamps,labels,method='VRT',kargs=kargs)
                        
                        #THIS IS FOR THE EXCEL FILE
                        self.onsetData['Settings'] = 'Onset type: '+self.VoiceOnsetmeth+' - Filter: '+self.dropSettFilterType.currentText()+' - Envelope: '+self.EnvelopeMethod+' - Segment analyzed[ms]: '+str(int(segment*1000))+' - Onset[%]: '+str(int(onTHR*100))+' - Saturation[%]: '+str(int(satTHR*100))
                    
        else:
              self.grpSettVOMethod.setVisible(False)
                # self.update_segments()
    #============================================================
    
    def box_properties(self,grp_obj,color = '#86c1e5'):
        grp_obj.setVisible(True)#Make the options for phoneme detection invisible until they are required
        grp_obj.setStyleSheet("QGroupBox { background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 "+color+", stop: 1 #FFFFFF);}")
    
    #============================================================