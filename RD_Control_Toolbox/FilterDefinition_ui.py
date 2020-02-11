# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FilterDefinition.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_FilterDefinition(object):
    def setupUi(self, FilterDefinition):
        FilterDefinition.setObjectName("FilterDefinition")
        FilterDefinition.resize(524, 183)
        self.horizontalLayout = QtWidgets.QHBoxLayout(FilterDefinition)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_filter_n = QtWidgets.QLabel(FilterDefinition)
        self.label_filter_n.setAlignment(QtCore.Qt.AlignCenter)
        self.label_filter_n.setObjectName("label_filter_n")
        self.verticalLayout.addWidget(self.label_filter_n)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(FilterDefinition)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.combo_filtertype = QtWidgets.QComboBox(FilterDefinition)
        self.combo_filtertype.setObjectName("combo_filtertype")
        self.verticalLayout_3.addWidget(self.combo_filtertype)
        self.label_2 = QtWidgets.QLabel(FilterDefinition)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(FilterDefinition)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.slider_freq = QtWidgets.QSlider(FilterDefinition)
        self.slider_freq.setMaximum(1000)
        self.slider_freq.setOrientation(QtCore.Qt.Horizontal)
        self.slider_freq.setObjectName("slider_freq")
        self.verticalLayout_4.addWidget(self.slider_freq)
        self.label_frequency = QtWidgets.QLabel(FilterDefinition)
        self.label_frequency.setText("")
        self.label_frequency.setAlignment(QtCore.Qt.AlignCenter)
        self.label_frequency.setObjectName("label_frequency")
        self.verticalLayout_4.addWidget(self.label_frequency)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(FilterDefinition)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.slider_damping = QtWidgets.QSlider(FilterDefinition)
        self.slider_damping.setMaximum(500)
        self.slider_damping.setSingleStep(1)
        self.slider_damping.setOrientation(QtCore.Qt.Horizontal)
        self.slider_damping.setObjectName("slider_damping")
        self.verticalLayout_2.addWidget(self.slider_damping)
        self.label_damping = QtWidgets.QLabel(FilterDefinition)
        self.label_damping.setText("")
        self.label_damping.setAlignment(QtCore.Qt.AlignCenter)
        self.label_damping.setObjectName("label_damping")
        self.verticalLayout_2.addWidget(self.label_damping)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(FilterDefinition)
        QtCore.QMetaObject.connectSlotsByName(FilterDefinition)

    def retranslateUi(self, FilterDefinition):
        _translate = QtCore.QCoreApplication.translate
        FilterDefinition.setWindowTitle(_translate("FilterDefinition", "Form"))
        self.label_filter_n.setText(_translate("FilterDefinition", "Filter #"))
        self.label.setText(_translate("FilterDefinition", "Filter Type"))
        self.label_3.setText(_translate("FilterDefinition", "Frequency (Hz)"))
        self.label_5.setText(_translate("FilterDefinition", "Damping (-)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FilterDefinition = QtWidgets.QWidget()
    ui = Ui_FilterDefinition()
    ui.setupUi(FilterDefinition)
    FilterDefinition.show()
    sys.exit(app.exec_())

