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
    def __init__(self,obj,objname):
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
        
        #The following two lines are necessary to detect the key event with QT
        #key press events in general are not processed unless you "activate the focus of qt onto your mpl canvas"
        self.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.canvas.setFocus()
        
        #Set layout
        self.layout = QtWidgets.QVBoxLayout()
        self.mplw.setLayout(self.layout)
        #Add widgets
        # self.layout.addWidget(self.toolbar)
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
          
    #=========================================================================
        
    def canvas_signal(self,sig,fs,title):
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
        sig = sig-np.mean(sig)
        # sig = sig/np.max(np.abs(sig))
        #Normalize respect to the maximum positive value, so the envelope matches the amplitude values
        #when has larger ngative values.
        sig = sig/np.max(sig)
        #Time vector
        t = np.arange(0,len(sig)/fs,1/fs)
        t = t[0:len(sig)]#Make sure that the length of the time array do not exceeds the length of the signal array 
        
        #Clear canvas
        self.canvas.figure.clear()
        #-
        self.ax = self.canvas.figure.add_subplot(1,1,1)
        #-
        self.ax.plot(t, sig,'k',alpha=0.4)
        self.ax.set_xlim([0,t[-1:]])
        self.ax.set_ylim([np.min(sig)-0.05,1.1])
        self.ax.set_title(title)
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time [seconds]')
        #Plot spectrum
        # self.plot_spectrum(sig,fs)
        #Draw
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        
    #=========================================================================
    
    def canvas_plot(self,x,y,label='',color='r',linestyle='-',lw=2,marker=None,ms=None,pointxt = None ):
        """
        Plot other data such as envelope, saturation poin, etc.
        -------
        None.

        """ 
        self.ax.plot(x,y,label=label,color=color,linestyle=linestyle,lw=lw,marker=marker,ms=ms)
        #For the point plots of the onsets
        if pointxt!=None:
            if x>0:
                self.ax.annotate(pointxt,(x+0.003,y-0.2),fontsize=8.5,fontweight='bold',backgroundcolor='w')
            else:
                self.ax.annotate(pointxt,(0,y),color = 'r',fontsize=8.5,fontweight='bold',backgroundcolor='w')
            
        self.ax.legend()        
        self.canvas.draw()
        
    #=========================================================================
        
    def clear_canvas(self):
        self.canvas.figure.clear()
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
        #Line properties
        self.pc = LineCollection(lines,color=linecolor,linewidths=linewidths,linestyles=linestyles)
    
        # Add collection to axes
        self.ax.add_collection(self.pc)
        self.canvas.draw()
    
    #=========================================================================       
    