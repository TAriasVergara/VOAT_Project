# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:11:43 2021

@author: ariasvts
"""

from PyQt5 import QtWidgets, uic


class ErrorDiagWin(QtWidgets.QDialog):
    """
    Error Dialog window
    """
     # Call the inherited classes __init__ method
    def __init__(self,msg):
        super(ErrorDiagWin, self).__init__()
        uic.loadUi('./GUI/ErrorDialog.ui', self) # Load the .ui file
        
        self.ErrorMsg = self.findChild(QtWidgets.QLabel,'ErrorMsg')
        self.ErrorMsg.setText(msg)
        
        self.btnOk = self.findChild(QtWidgets.QPushButton,'btnOK')
        self.btnOk.clicked.connect(lambda:self.close())
        
        self.show()
    #======================================================================
    
class InfoDiagWin(QtWidgets.QDialog):
    """
    Information Dialog window
    """
     # Call the inherited classes __init__ method
    def __init__(self,msg):
        super(InfoDiagWin, self).__init__()
        uic.loadUi('./GUI/InfoDialog.ui', self) # Load the .ui file
        
        self.Msg = self.findChild(QtWidgets.QLabel,'InfoMsg')
        self.Msg.setText(msg)
        
        self.btnOk = self.findChild(QtWidgets.QPushButton,'btnOK')
        self.btnOk.clicked.connect(lambda:self.close())
        
        self.show()
    #======================================================================

class GAT_TrajWin(QtWidgets.QDialog):
    
    def __init__(self):
        super(GAT_TrajWin, self).__init__()
        uic.loadUi('./GUI/GATTrajectoryWindow.ui', self) # Load the .ui file
        
        self.options = {}
        
        #-
        self.btnOk = self.findChild(QtWidgets.QPushButton,'btnGATTrajOK')
        self.btnOk.clicked.connect(self.get_data)
        #-
        self.btnCancel = self.findChild(QtWidgets.QPushButton,'btnGATTrajCancel')
        self.btnCancel.clicked.connect(lambda:self.close())
        
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
                
        
    def get_data(self):
        #-
        
        self.options['Trajectory_position'] = self.sldPos.value()/100
        self.options['Sampling_frequency'] = int(self.dropFS.currentText())
        self.accept()