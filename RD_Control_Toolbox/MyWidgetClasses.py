from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

import pyqtgraph as pg
import time

class MyFrame(QtWidgets.QFrame):
    def resizeEvent(self,e):
        container_geometry = self.geometry()
        a=self.children()
        for eachchild in a:
            eachchild.setGeometry(QtCore.QRect(0, 0, container_geometry.width(), container_geometry.height()))
        
class MyBodeFrame(QtWidgets.QFrame):
    def __init__(self,parent):
        super().__init__(parent)
        pg.setConfigOption('background','w')
        pg.setConfigOption('foreground','k')


        self.plot_gain = CustomPlotWidget(self)
        self.plot_phase = CustomPlotWidget(self)

        self.plot_gain.sigRangeChanged.connect(self.onSigRangeChanged)
        self.plot_phase.sigRangeChanged.connect(self.onSigRangeChanged)
        self.plot_gain.setLabel('left','Magnitude [dB]')
        self.plot_gain.setLabel('bottom','Frequency [hz]')
        self.plot_gain.showGrid(x=True,y=True,alpha=0.3)
        
        self.plot_phase.setLabel('left','Phase [deg]')
        self.plot_phase.setLabel('bottom','Frequency [hz]')
        self.plot_phase.showGrid(x=True,y=True,alpha=0.3)

  
    def resizeEvent(self,e):
        container_geometry = self.geometry()
        self.plot_gain.setGeometry(QtCore.QRect(0, 0, container_geometry.width(), container_geometry.height()/2))
        self.plot_phase.setGeometry(QtCore.QRect(0, container_geometry.height()/2, container_geometry.width(), container_geometry.height()/2))

    def plot_bode(self,phase,gain,omega):
        self.plot_gain.plot(omega,gain,pen=pg.mkPen('k', width=1, style=QtCore.Qt.SolidLine))
        self.plot_gain.setXRange(min(omega),max(omega),padding=1)
        self.plot_gain.setLogMode(True,False)
        

        self.plot_phase.plot(omega,phase,pen=pg.mkPen('k', width=1, style=QtCore.Qt.SolidLine))
        self.plot_phase.setXRange(min(omega),max(omega),padding=1)
        self.plot_phase.setLogMode(True,False)

    def onSigRangeChanged(self,r):
        self.plot_gain.sigRangeChanged.disconnect(self.onSigRangeChanged)
        self.plot_phase.sigRangeChanged.disconnect(self.onSigRangeChanged)

        if self.plot_gain == r:
            self.plot_phase.setRange(xRange = r.getAxis('bottom').range)
        elif self.plot_phase == r:
            self.plot_gain.setRange(xRange = r.getAxis('bottom').range)
        
        self.plot_gain.sigRangeChanged.connect(self.onSigRangeChanged)
        self.plot_phase.sigRangeChanged.connect(self.onSigRangeChanged)


class MyRLocusFrame(QtWidgets.QFrame):
    def __init__(self,parent):
        super().__init__(parent)
        pg.setConfigOption('background','w')
        pg.setConfigOption('foreground','k')

        self.rlocus_init = 1
        self.plot_rlocus = CustomPlotWidget(self)

        self.plot_rlocus.setLabel('left','Imag Axis')
        self.plot_rlocus.setLabel('bottom','Real Axis')
        
        self.plot_rlocus.setXRange(-1,1,padding=0.01)
        self.plot_rlocus.setYRange(-1,1,padding=0.01)
        self.grid = []
        self.grid_labels = []
        self.drawgrid()
        self.plot_rlocus.sigRangeChanged.connect(self.onSigRangeChanged)
        self.rlocus_init = 0


    
    def resizeEvent(self,e):
        container_geometry = self.geometry()

        self.plot_rlocus.setGeometry(QtCore.QRect(0, 0, container_geometry.width(), container_geometry.height()))
        

    def drawgrid(self):
        n_radius = 18;
        step_angle = 180 / n_radius
        n_circles = 10;
        xaxis = self.plot_rlocus.getAxis('bottom').range
        yaxis = self.plot_rlocus.getAxis('left').range
        x0 = 0
        y0 = 0

        
        tini = time.time()
        ii = 0
        for angle in np.linspace(-90,90,n_radius):
            damping = np.cos(angle*np.pi/180)
            if angle < 0:
                y1 = yaxis[0]
                x1 = y1 * np.tan((angle+90) * np.pi/180)
                if x1 < xaxis[0]:
                    x1 = xaxis[0]
                    y1 = x1 / np.tan((angle+90) * np.pi/180)
                x = np.array([x0,x1])
                y = np.array([y0,y1])

            if angle > 0:
                x1 = xaxis[0]
                y1 = x1 / np.tan((angle+90) * np.pi/180)
                if y1 > yaxis[1]:
                    y1 = yaxis[1]
                    x1 = y1 * np.tan((angle+90) * np.pi/180)
                x = np.array([x0,x1])
                y = np.array([y0,y1])
        
            if self.rlocus_init == 1:
                self.grid.append(self.plot_rlocus.plot(x,y,pen=pg.mkPen(0.9, width=1, style=QtCore.Qt.DashLine)))
                if y1 < yaxis[1]*0.98:
                    self.grid_labels.append(pg.TextItem(text="{:.3f}".format(damping),anchor=(0,1),color=(200,0,0)))
                else:
                    self.grid_labels.append(pg.TextItem(text="{:.3f}".format(damping),anchor=(0,0),color=(200,0,0)))

                self.plot_rlocus.addItem(self.grid_labels[ii])
                self.grid_labels[ii].setPos(x1,y1)
                font=QtGui.QFont()
                font.setPixelSize(8)
                self.grid_labels[ii].setFont(font)


            else:
                self.grid[ii].setData(x,y)
                self.grid_labels[ii].setPos(x1,y1)
                if y1 < yaxis[1]*0.9:
                    self.grid_labels[ii].setAnchor([0,1])
                else:
                    self.grid_labels[ii].setAnchor([0,0])



            ii = ii+1

        

        for r in np.linspace(0.0,max(abs(xaxis[0]),abs(yaxis[1]),abs(yaxis[0])),n_circles):
            center = np.array([0,0])
            x,y = self.drawcircle(center,r,20)
            if self.rlocus_init == 1:
                self.grid.append(self.plot_rlocus.plot(x,y,pen=pg.mkPen(0.9, width=1, style=QtCore.Qt.DashLine,alpha=0.1)))
                self.grid_labels.append(pg.TextItem(text="{:.3f}".format(r),anchor=(0,1),color=(200,0,0)))
                self.plot_rlocus.addItem(self.grid_labels[ii])
                self.grid_labels[ii].setPos(x[0],y[0])
                font=QtGui.QFont()
                font.setPixelSize(8)
                self.grid_labels[ii].setFont(font)
            else:
                self.grid[ii].setData(x,y)
                self.grid_labels[ii].setPos(x[0],y[0])
            ii = ii+1
        tend = time.time()

    def drawcircle(self,center,r,n_dots):
        x=np.empty(n_dots,float)
        y=np.empty(n_dots,float)
        for ii,angle in enumerate(np.linspace(0,np.pi,n_dots)):
            x_ii = center[0] + r * np.cos(angle + np.pi/2)
            y_ii = center[1] + r * np.sin(angle + np.pi/2)

            x[ii] = x_ii
            y[ii] = y_ii
        return x,y

    def onSigRangeChanged(self,r):   
        
        #for each_grid_component in self.grid:
        #    each_grid_component.clear()
        #self.grid = []
        self.drawgrid()
            

    



class CustomPlotWidget(pg.PlotWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
        
