from TCtrl_Tune_ui import *
import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog
from WTG_Class import *

import scipy.io
import control as ctrl
import harold as hctrl
import time
import math
import matplotlib.pyplot as plt

import numpy as np

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        pg.setConfigOption('background','w')
        pg.setConfigOption('foreground','k')
        #self.plot_1_1 = CustomPlotWidget('plot_1_1',self.frame_plot_1_1)
        
        #self.splitter.setSizes([16777215,16777215])
        self.splitter.setStretchFactor(0,1)
        self.splitter.setStretchFactor(1,25)
        self.wid_Filter1.label_filter_n.setText('Filter #1')
        self.wid_Filter2.label_filter_n.setText('Filter #2')
        self.wid_Filter3.label_filter_n.setText('Filter #3')
        self.wid_Filter4.label_filter_n.setText('Filter #4')
        self.wid_Filter5.label_filter_n.setText('Filter #5')
        self.wid_Filter6.label_filter_n.setText('Filter #6')

        self.wid_Filter1.setValues(['None','Notch','Low pass','High pass'],1,1)
        self.wid_Filter2.setValues(['None','Notch','Low pass','High pass'],1,1)
        self.wid_Filter3.setValues(['None','Notch','Low pass','High pass'],1,1)
        self.wid_Filter4.setValues(['None','Notch','Low pass','High pass'],1,1)
        self.wid_Filter5.setValues(['None','Notch','Low pass','High pass'],1,1)
        self.wid_Filter6.setValues(['None','Notch','Low pass','High pass'],1,1)

        self.actionZoom.triggered.connect(self.ZoomTriggered)
        self.actionLoad_Model.triggered.connect(self.LoadModelTriggered)

    def ZoomTriggered(self,q):
        pass


    def LoadModelTriggered(self,q):

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self,"Select linearized model file:","U:/Loads/i115_gl2012_2B12_IC/Linearization/LinMod/","Matlab Files (*.mat)" , options=options)
        azimuth_index = 0 
        wtg = WTG(filename,azimuth_index)
        kvect = np.linspace(0,2e10,1000)

        start = time.time()
        rlist,klist = ctrl.root_locus(wtg.T_G_1,kvect,Plot=False)
        omega_vect = np.logspace(-2,2,1000)
        mag,phase,omega = ctrl.bode(wtg.T_G_1,omega_vect,Plot=False)
        magdb = 20*np.log10(mag)
        phasedeg = phase*180/np.pi
        omegahz = omega / 2 / np.pi
        self.frame_plot_1_2.plot_bode(phase,magdb,omegahz)
        self.frame_plot_2_1.plot_bode(phase,magdb,omegahz)
        self.frame_plot_2_2.plot_bode(phase,magdb,omegahz)
        

        for ii in range(0,rlist.shape[1]):
            locs = rlist[:,ii]
            x = locs.real
            y = locs.imag
            self.frame_plot_1_1.plot_rlocus.plot(x,y,pen=pg.mkPen('k', width=1, style=QtCore.Qt.SolidLine))





        plt.show()
        end = time.time()
        print(end-start)
        print('done')

class CustomPlotWidget(pg.PlotWidget):
    def __init__(self,title,parent):
        super().__init__(parent)
        self.setAcceptDrops(True)
        

        

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
