# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:03:32 2022

@author: ariasvts
"""

from PyQt5 import QtWidgets, uic


class VoiceOnsetsWin(QtWidgets.QDialog):
    """
    Voice onsets window
    """
     # Call the inherited classes __init__ method
    def __init__(self):
        super(VoiceOnsetsWin, self).__init__()
        uic.loadUi('./GUI/VoiceOnsetsWindow.ui', self) # Load the .ui file
        
        self.options = []
        #-
        self.textInfo = self.findChild(QtWidgets.QTextEdit,"textDescription")
        #-
        
        #Voice onsets
        self.checkVAT = self.findChild(QtWidgets.QCheckBox,"check_Onset_VAT")
        # self.checkVAT.stateChanged.connect(lambda:self.textInfo.setText('Requires Acoustic and EGG signals'))
        self.checkVAT.stateChanged.connect(self.setInfo)
        self.checkVRT = self.findChild(QtWidgets.QCheckBox,"check_Onset_VRT")
        self.checkVRT.stateChanged.connect(self.setInfo)
        self.checkVOC = self.findChild(QtWidgets.QCheckBox,"check_Onset_VOC")
        self.checkVOC.stateChanged.connect(self.setInfo)
        self.checkVOwT = self.findChild(QtWidgets.QCheckBox,"check_Onset_VOwT")
        self.checkVOwT.stateChanged.connect(self.setInfo)
        #-
        
        self.btnCompute  = self.findChild(QtWidgets.QPushButton,"btnComputeOnsets")
        self.btnCompute.clicked.connect(self.compute_onsets)
        
        #-
        self.btnCancel  = self.findChild(QtWidgets.QPushButton,"btnCancel")
        self.btnCancel.clicked.connect(lambda:self.close())
        #-
        
    #======================================================================
    
    def setInfo(self):
        if self.checkVAT.isChecked():
            self.textInfo.setText('Requires acoustic and EGG signals (Sustained phonation)')
        if self.checkVRT.isChecked():
            self.textInfo.setText('Requires an acoustic signal (Sustained phonation)')
        if self.checkVOC.isChecked():
            self.textInfo.setText('Requires airflow and EGG signals (Sustained phonation)')
        if self.checkVOwT.isChecked():
            self.textInfo.setText('Requires an acoustic signal (Sustained phonation)')
        if (self.checkVAT.isChecked()==False and self.checkVRT.isChecked()==False and self.checkVOC.isChecked()==False and self.checkVOwT.isChecked()==False):
            self.textInfo.clear()
            
    #======================================================================
    
    def compute_onsets(self):
        #-
        if self.checkVAT.isChecked():
            self.options.append('VAT')
        #-
        if self.checkVOC.isChecked():
            self.options.append('VOC')
        #-
        if self.checkVRT.isChecked():
            self.options.append('VRT')
        #-
        if self.checkVOwT.isChecked():
            self.options.append('VOwT')
        #-
        self.accept()
            
        
#======================================================================
#======================================================================
#======================================================================

class PhonemeWin(QtWidgets.QDialog):
    """
    Phoneme segmentation window
    """
     # Call the inherited classes __init__ method
    def __init__(self):
        super(PhonemeWin, self).__init__()
        uic.loadUi('./GUI/PhonemeWindow.ui', self) # Load the .ui file
        
        self.options = []
        #-
        self.textInfo = self.findChild(QtWidgets.QTextEdit,"textDescription")
        #-
        #Phoneme detection
        self.selPhoneLang = self.findChild(QtWidgets.QComboBox,"selPhoneLang")
        langs = ['English (US)', 'German (Germany)', 'Spanish (Mexico)','Multilanguge']
        self.selPhoneLang.addItems(langs)
        #-
        self.selPhoneType= self.findChild(QtWidgets.QComboBox,"selPhoneType")
        ptype = ['Manner', 'Place', 'Voicing']
        self.selPhoneType.addItems(ptype)
        #-
        self.btnCompute  = self.findChild(QtWidgets.QPushButton,"btnCompute")
        self.btnCompute.clicked.connect(self.compute_phoneme)
        #-
        self.btnCancel  = self.findChild(QtWidgets.QPushButton,"btnCancel")
        self.btnCancel.clicked.connect(lambda:self.close())
        #-
        
    #======================================================================
    
    # def setInfo(self):
    #     if self.checkVAT.isChecked():
    #         self.textInfo.setText('Requires acoustic and EGG signals (Sustained phonation)')
    #     if self.checkVRT.isChecked():
    #         self.textInfo.setText('Requires an acoustic signal (Sustained phonation)')
    #     if self.checkVOC.isChecked():
    #         self.textInfo.setText('Requires airflow and EGG signals (Sustained phonation)')
    #     if self.checkVOwT.isChecked():
    #         self.textInfo.setText('Requires an acoustic signal (Sustained phonation)')
    #     if (self.checkVAT.isChecked()==False and self.checkVRT.isChecked()==False and self.checkVOC.isChecked()==False and self.checkVOwT.isChecked()==False):
    #         self.textInfo.clear()
    #======================================================================
    
    def compute_phoneme(self):
        lang = self.selPhoneLang.currentText()
        ptype = self.selPhoneType.currentText()
        self.options.append([lang,ptype])
        self.accept()
        