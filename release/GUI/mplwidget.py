# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:40:56 2021

@author: ariasvts
"""
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle

import numpy as np

sys.path.append('./Utils')
from Utils.preprocessing import Signal_processing as sg

class MplWidget(QtWidgets.QWidget):
     # Call the inherited classes __init__ method
    def __init__(self,obj,objname,buttons_canvas):
        """
        Call the widget that will be used for plotting

        Parameters
        ----------
        obj : Window object that contains the widget defined for plotting
            DESCRIPTION.
        objname : Name of the widget used for plotting
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(MplWidget, self).__init__()
        
        self.StartP = None #Start point for editing signals
        self.EndP = None  #End point for editing signals
        self.pc = None #Collection to draw in the canvas
        self.key  = None #Used to crop the signal when shift is pressed
        self.HomeViewRange = None #Used to store the original view range in the plot
        self.setThresh = None #Used to select a threshold by clicking in the figure
        
        #Look for the widget as named in the UI
        self.mplw = obj.findChild(QtWidgets.QWidget,objname)
        #
        self.canvas = FigureCanvas(Figure())
        
        #On click event. Used to draw vertical lines on the plot for 
        #editing the signals
        self.cid = self.canvas.mpl_connect('button_press_event', self.set_start_point)
        self.cid2 = self.canvas.mpl_connect('button_release_event', self.set_end_point)
        self.cid3 = self.canvas.mpl_connect('motion_notify_event', self.moved_and_pressed)
        #The following two lines are necessary to detect the key event with QT
        #key press events in general are not processed unless you "activate the focus of qt onto your mpl canvas"
        self.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.canvas.setFocus()
        
        #Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        #Delete some buttons
        unwanted_buttons = ['Home','Back','Forward','Pan','Zoom']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        #Custom toolbar button
        nacts = 1
        #--
        self.buttons_canvas = buttons_canvas
        self.buttons_canvas['eThresh'].clicked.connect(self.Set_ethresh)
        #-
        self.buttons_canvas['ZoomIn'].clicked.connect(self.zoom_in)
        self.buttons_canvas['ZoomOut'].clicked.connect(self.zoom_out)
        #--
        self.toolbar.insertSeparator(self.toolbar.actions()[nacts])
        self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['ZoomOut'])
        self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['ZoomIn'])
        #--
        self.toolbar.insertSeparator(self.toolbar.actions()[nacts])
        # self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['eThresh'])
        self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['CutSeg'])
        # self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['DelSeg'])
        # self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['AddSeg'])
        #--
        self.toolbar.insertSeparator(self.toolbar.actions()[nacts])
        self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['StopSound'])
        self.toolbar.insertWidget(self.toolbar.actions()[nacts], buttons_canvas['PlaySound'])
        #--
        #Set layout
        self.layout = QtWidgets.QVBoxLayout()
        self.mplw.setLayout(self.layout)
        #Add widgets
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        #Allows to set another layout when updating the plots
        self.layout.deleteLater()
        
    #=========================================================================
    
    def reset_canvas(self):
        """
        Remove drawings in the plot

        Returns
        -------
        None.

        """
        try:
            self.StartP = None #Start point for editing signals
            self.EndP = None  #End point for editing signals
            if self.pc is not None:
                self.pc.remove()
                self.pc = None
            self.canvas.draw()
        except:
            pass    
    #=========================================================================
    
    def Set_ethresh(self):
        try:
            #Is there something plotted already?
            if self.HomeViewRange is not None:
                self.reset_canvas()
                self.buttons_canvas['eThresh'].setChecked(True)
                self.canvas.draw()
        except:
            pass
    #=========================================================================
        
    def zoom_in(self):
        try:
            if self.EndP==self.StartP:
                self.ax.set_xlim([self.StartP,self.EndP+0.1])
                # self.ax_spec.set_xlim([self.StartP,self.EndP+0.1])
            else:
                #Zoom in for time series
                self.ax.set_xlim([self.StartP,self.EndP])
                self.ax.set_ylim([self.HomeViewRange['Max Amp'], self.HomeViewRange['Min Amp']])
                #Zoom in for spectrum
                # self.ax_spec.set_xlim([self.StartP,self.EndP])
                
            self.canvas.draw()
        except:
            pass
    
        
    #=========================================================================
        
    def zoom_out(self):
        try:
            # self.ax_spec.set_xlim([0,self.HomeViewRange['Length']])#Spectrum
            self.ax.set_xlim([0,self.HomeViewRange['Length']])#Timeseries
            self.ax.set_ylim([self.HomeViewRange['Max Amp'], self.HomeViewRange['Min Amp']])
            self.canvas.draw()
        except:
            pass
        
    #=========================================================================
    
    def set_start_point(self,event):
        """
        Function used to draw lines in the figure

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        try:
            #Deselect any segment item (SegmentationWindow.py
            self.buttons_canvas['SelItem'].setCurrentItem(self.buttons_canvas['SelItem'].currentItem(),QtCore.QItemSelectionModel.Deselect)
            
            #Activate when the LEFT mouse button is pressed (value = 1)
            self.key = event.key
            
            self.reset_canvas()
            #Select energy threshold
            if (self.buttons_canvas['eThresh'].isChecked())and(event.button.value == 1):
                self.setThresh = event.ydata
                self.buttons_canvas['eThresh'].setChecked(False)
                
            
            elif (event.button.value == 1):
            # if (event.button.value == 1)and(self.key == "shift"):
                #Clear any object that has been drawn in the figure
                self.StartP = event.xdata
                        
            if (event.button.value == 3):
                self.reset_canvas()
        except:
            self.StartP = None
            
    #=========================================================================
    
    def set_end_point(self,event):
        """
        Function used to draw lines in the figure

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.button.value == 1:
        # if (event.button.value == 1)and(self.key == "shift"):
            self.EndP = event.xdata
            #Draw a rectangle
            if (self.StartP is not None)and(self.EndP is not None):
                #Make sure that StartP is always the lowest number
                if self.StartP>self.EndP:
                    temp = self.StartP.copy()
                    self.StartP = self.EndP.copy()
                    self.EndP = temp
                #Draw the rectangle
                times = np.hstack([self.StartP,self.EndP]).reshape(1,-1)
                #-
                if (self.EndP==self.StartP)and(self.buttons_canvas['eThresh'].isChecked()==False):
                    self.make_line()
                #-
                else:
                    self.make_error_boxes(times,height=3, facecolor='b',edgecolor='k', alpha=0.3)
                # self.key = None
            else:
                self.canvas.draw()
                # self.reset_canvas()
    
    #=========================================================================
    
    def moved_and_pressed(self,event):
        try:
            # if (event.button.value == 1)and(self.key == "shift"):
            if self.buttons_canvas['eThresh'].isChecked():
                line = [[(0,event.ydata),(self.HomeViewRange['Length'],event.ydata)]]
                self.make_line(line,linecolor='r',linewidths=1.5,linestyles='dashed')
                #-
                self.canvas.draw()
                #Remove previous drawing to avoid overlapping rectangles
                self.pc.remove()
                
            elif event.button.value == 1:
                times = np.hstack([self.StartP,event.xdata]).reshape(1,-1)
                self.make_error_boxes(times,height=3, facecolor='b', alpha=0.3)
                self.canvas.draw()
                #Remove previous drawing to avoid overlapping rectangles
                self.pc.remove()
        except:
               pass
            
    #=========================================================================
        
    def update_graph(self,sig,fs,title):
        """
        Plot signals

        Parameters
        ----------
        sig: Signal to be plotted
        fs: Sampling frequency of the signal
        title: Name of the plot
        -------
        None.

        """ 
        #Time vector
        t = np.arange(0,len(sig)/fs,1/fs)
        t = t[0:len(sig)]#Make sure that the length of the time array do not exceeds the length of the signal array 
        self.HomeViewRange = {'Length':t[-1:],
                              'Max Amp': np.min(sig)-0.05,
                              'Min Amp':np.max(sig)+0.05}
        #Clear canvas
        self.canvas.figure.clear()
        #-
        self.ax = self.canvas.figure.add_subplot(1,1,1)
        #-
        self.ax.plot(t, sig,'k',alpha=0.6)
        self.ax.set_xlim([0,t[-1:]])
        self.ax.set_ylim([np.min(sig)-0.05,np.max(sig)+0.05])
        self.ax.set_title(title)
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time [seconds]')
        #Plot spectrum
        # self.plot_spectrum(sig,fs)
        #Draw
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        
        
    #=========================================================================
    
    def plot_spectrum(self,sig,fs,win_time = 0.02,step_time = 0.01,n_padded=1024,cmap='jet'):
        #Get spectrum
        spec = sg.get_spectrum(sig,fs,win_time,step_time,n_padded)
        mxfreqz = fs/2#Maximum frequency of the spectrum
        #-
        self.ax_spec = self.canvas.figure.add_subplot(2,1,2)
        #-
        self.ax_spec.imshow(np.flipud(spec.T),
                    #interpolation='bilinear',
                    aspect='auto',
                    extent=[0,spec.shape[0]*step_time,0,mxfreqz],
                    cmap=cmap)
        self.ax_spec.set_title('Spectrogram')
        self.ax_spec.set_ylabel('Frequency [Hz]')
        self.ax_spec.set_xlabel('Time [seconds]')
    
    #=========================================================================
        
    def make_error_boxes(self,times,height=3, facecolor='g',edgecolor='None', alpha=0.4,zorder=1):
        """
        Draw rectangles to highlight segments of signal in a plot

        Parameters
        ----------
        ax : axes properties (matplotlib)
        times: Matrix MxN with the time stamps. 
        facecolor : Color of the rectangle. The default is 'r'.
        edgecolor: Color of the edge of the rectangle.
        alpha : Transparency (0 clear, 1 is dark). The default is 0.4.
        zorder: Whether the patch is in front or behind the plot

        Returns
        -------
        None.

        """
        #Set the position and shape of the rectangles
        xdata = times[:,0]#Anchor point 'x'
        ydata = np.ones(times.shape[0])*-(height/2)#Anchor point 'y'
        xerror = times[:,1]-times[:,0]#Width of the box
        yerror = np.ones(len(ydata))*height#Height of the box
        
        #For the "zoom in" option with selected items from list
        if len(self.buttons_canvas['SelItem'].selectedItems())>0:
            self.StartP = times[0,0]#Start point
            self.EndP = times[0,1]#End point
            
        
        
        # Loop over data points; create box from errors at each point
        errorboxes = [Rectangle((x, y), xe, ye)
                      for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]
        
        # Create patch collection with specified colour/alpha
        self.pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor,zorder=zorder)
    
        # Add collection to axes
        self.ax.add_collection(self.pc)
        self.canvas.draw()
    
    #=========================================================================      
    
    def make_line(self,lines=None,linecolor='b',linewidths=2,linestyles='solid'):
        """
        

        Parameters
        ----------
        lines : TYPE, optional
            DESCRIPTION. List with the lines to draw
        linecolor : TYPE, optional
            DESCRIPTION. The default is 'b'.
        linewidths : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        if lines==None:
            # Create patch collection with specified colour/alpha
            lines = [[(self.StartP,-1.5),(self.StartP,1.5)]]
            
        #Line properties
        self.pc = LineCollection(lines,color=linecolor,linewidths=linewidths,linestyles=linestyles)
    
        # Add collection to axes
        self.ax.add_collection(self.pc)
        self.canvas.draw()
    
    #=========================================================================       
    