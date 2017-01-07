from PyQt4 import QtGui, QtCore

class CorrectLabel(QtGui.QComboBox):
    def __init__(self, parent=None):
        super(CorrectLabel, self).__init__()
        for i in range(10):
            self.addItem(str(i))
