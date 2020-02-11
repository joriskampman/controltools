from FilterDefinition_ui import *
import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *


class FilterDefinition(QtWidgets.QWidget,Ui_FilterDefinition):

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.slider_freq.valueChanged.connect(self.updateValues)
        self.slider_damping.valueChanged.connect(self.updateValues)

    def setValues(self,listItems,frequency,damping):
        self.combo_filtertype.addItems(listItems)
        self.slider_damping.setValue(damping*100)
        self.slider_freq.setValue(frequency*100)

        val_freq = self.slider_freq.value()/100
        val_damp = self.slider_damping.value()/100

        self.label_frequency.setText(str(val_freq)+'Hz')
        self.label_damping.setText(str(val_damp))
    
    def updateValues(self):
        val_freq = self.slider_freq.value()/100
        val_damp = self.slider_damping.value()/100

        self.label_frequency.setText(str(val_freq)+'Hz')
        self.label_damping.setText(str(val_damp))
        
    