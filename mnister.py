import os, shutil, sys, datetime
import PyQt4.QtGui
import PyQt4.QtCore
import PyQt4.uic
import PIL.Image, PIL.ImageOps
import numpy as np
import pickle

form, base = PyQt4.uic.loadUiType('./mainUI.ui')
form1, base1 = PyQt4.uic.loadUiType('./dialogUI.ui')
form2, base2 = PyQt4.uic.loadUiType('./modelselectUI.ui')

MODEL_PATH = './models/'
MODEL_LIST = []
CURRENT_MODEL = 'models.SimpleConvNet.model'
CURRENT_PARAM = './models/SimpleConvNet/params_trained_simpleconv.pkl'


class ModelSelectDialog(form2, base2):
    def __init__(self, parent=None):
        super(ModelSelectDialog, self).__init__(parent)
        self.setupUi(self)
        self.initModelList()
        self.establishConnections()

    def initModelList(self):
        global MODEL_LIST
        for x in os.listdir(MODEL_PATH):
            if os.path.isdir(MODEL_PATH + x):
                MODEL_LIST.append(x)
                self.ModelList.addItem(x)

    def updateParameterList(self):
        self.ParameterList.clear()
        model = self.ModelList.currentText()
        for x in os.listdir(MODEL_PATH + model):
            self.ParameterList.addItem(x)

    def setModel(self):
        global CURRENT_MODEL, CURRENT_PARAM
        print CURRENT_MODEL
        model = self.ModelList.currentText()
        param = self.ParameterList.currentText()
        CURRENT_MODEL = 'models.{0}.model'.format(model)
        CURRENT_PARAM = '{0}{1}/{2}'.format(MODEL_PATH, model, param)
        print CURRENT_MODEL
        print CURRENT_PARAM
        print 'setModel()'

    def establishConnections(self):
        self.ModelList.activated.connect(self.updateParameterList)
        self.ButtonBox.accepted.connect(self.setModel)
        self.ButtonBox.accepted.connect(main_form.initNetwork)


class SecretDialog(form1, base1):
    def __init__(self, parent=None):
        super(SecretDialog, self).__init__(parent)
        self.setupUi(self)


class Color(object):
    def __init__(self, R=0, G=0, B=0):
        self.R = R
        self.G = G
        self.B = B


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def set(self, x, y):
        self.x = X
        self.y = y


class Shape(object):
    def __init__(self, coordinate=Point(0, 0), width=0.0, color=Color(0, 0, 0), shape_number=0):
        self.coordinate = coordinate
        self.width = width
        self.color = color
        self.shape_number = shape_number


class Shapes(object):
    def __init__(self):
        self.__Shapes = []

    def countShape(self):
        return len(self.__Shapes)

    def updateShape(self, coordinate, width, color, shape_number):
        shape = Shape(coordinate, width, color, shape_number)
        self.__Shapes.append(shape)

    def getShape(self, index):
        return self.__Shapes[index]

    def removeShape(self, coordinate, threshold):
        i = 0
        while True:
            if i == len(self.__Shapes):
                break
            if abs(coordinate.x - self.__Shapes[i].coordinate.x) < threshold and abs(coordinate.y - self.__Shapes[i].coordinate.y) < threshold:
                self.__Shapes.pop(i)
                for n in xrange(len(self.__Shapes) - i):
                    self.__Shapes[n + i].shape_number += 1
                i -= 1
            i += 1


class Painter(PyQt4.QtGui.QWidget):
    def __init__(self, parent):
        super(Painter, self).__init__()
        self.parent_link = parent
        self.mouse_coordinate = Point(0, 0)
        self.last_coordinate = Point(0, 0)
        palette = PyQt4.QtGui.QPalette()
        palette.setColor(PyQt4.QtGui.QPalette.Background, PyQt4.QtCore.Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def mousePressEvent(self, event):
        if self.parent_link.is_brush:
            self.parent_link.is_painting = True
            self.parent_link.shape_num += 1
            self.last_coordinate = Point(0, 0)
        else:
            self.parent_link.is_eraseing = True

    def mouseReleaseEvent(self, event):
        if self.parent_link.is_painting:
            self.parent_link.is_painting = False
        if self.parent_link.is_eraseing:
            self.parent_link.is_eraseing = False

    def mouseMoveEvent(self, event):
        if self.parent_link.is_painting:
            self.mouse_coordinate = Point(event.x(), event.y())
            is_moving_to_x = self.last_coordinate.x != self.mouse_coordinate.x
            is_moving_to_y = self.last_coordinate.y != self.mouse_coordinate.y
            if is_moving_to_x and is_moving_to_y:
                self.last_coordinate = Point(event.x(), event.y())
                self.parent_link.drawing_shapes.updateShape(
                    self.last_coordinate,
                    self.parent_link.current_width,
                    self.parent_link.current_color,
                    self.parent_link.shape_num)
                self.repaint()
        if self.parent_link.is_eraseing:
            self.mouse_coordinate = Point(event.x(), event.y())
            self.parent_link.drawing_shapes.removeShape(self.mouse_coordinate, 10)
            self.repaint()

    def paintEvent(self, event):
        painter = PyQt4.QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

    def drawLines(self, event, painter):
        painter.setRenderHint(PyQt4.QtGui.QPainter.Antialiasing)

        for i in xrange(self.parent_link.drawing_shapes.countShape() - 1):
            T = self.parent_link.drawing_shapes.getShape(i)
            T1 = self.parent_link.drawing_shapes.getShape(i + 1)

            if T.shape_number == T1.shape_number:
                pen = PyQt4.QtGui.QPen(
                    PyQt4.QtGui.QColor(T.color.R, T.color.G, T.color.B),
                    T.width / 2,
                    PyQt4.QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(
                    T.coordinate.x,
                    T.coordinate.y,
                    T1.coordinate.x,
                    T1.coordinate.y)


class MainUI(base, form):
    def __init__(self):
        super(base, self).__init__()
        self.setupUi(self)
        self.setObjectName('MNISTer')
        self.paint_panel = Painter(self)
        self.paint_panel.close()
        self.BlackBoard.insertWidget(0, self.paint_panel)
        self.BlackBoard.setCurrentWidget(self.paint_panel)
        self.establishConnections()
        self.initClassLabels()
        self.is_brush = True
        self.is_painting = False
        self.is_eraseing = False
        self.is_mouseing = False
        self.drawing_shapes = Shapes()
        self.current_color = Color(0, 0, 0)
        self.shape_num = 0
        self.current_width = 20
        self.initImageDir()

    def initImageDir(self):
        if not os.path.exists('./images'):
            os.mkdir('./images')
        if not os.listdir('./images'):
            [os.mkdir('./images/label{}'.format(d)) for d in xrange(10)]

    def initNetwork(self):
        print 'initNetwork()'
        params = self.unpickle(CURRENT_PARAM)
        tmp = 'from {0} import Network'.format(CURRENT_MODEL)
        print tmp
        exec(tmp)
        self.network = Network(params=params)

    def initClassLabels(self):
        for i in xrange(10):
            self.ClassLabels.addItem(str(i))

    def unpickle(self, fname):
        with open(fname, 'rb') as f:
            y = pickle.load(f)
        return y

    def switchBrush(self):
        self.is_brush = not self.is_brush

    def judge(self):
        raw_pixmap = PyQt4.QtGui.QPixmap.grabWidget(self.paint_panel)
        raw_pixmap.save('temp.png')
        img = PIL.Image.open('./temp.png').convert('L')
        img = PIL.ImageOps.invert(img)
        img.thumbnail((28, 28))
        img.save('temp28x28.png')
        scaled_pixmap = PyQt4.QtGui.QPixmap('temp28x28.png')
        scaled_pixmap = scaled_pixmap.scaledToHeight(226)
        self.label.setPixmap(scaled_pixmap)
        scaled_img_array = np.asarray(PIL.Image.open('temp28x28.png').convert('L'))
        self.judged_class = self.network.judge(scaled_img_array.reshape(784))
        self.ClassificationResult.setText(str(self.judged_class))
        self.CorrectButton.setEnabled(True)
        self.SubmitButton.setEnabled(True)
        self.ClassLabels.setEnabled(True)
        self.ClassificationResult.setEnabled(True)
        self.JudgeButton.setEnabled(False)

    def changeThickness(self, num):
        self.current_width = num

    def clearSlate(self):
        self.drawing_shapes = Shapes()
        self.paint_panel.repaint()
        self.label.clear()
        self.CorrectButton.setEnabled(False)
        self.SubmitButton.setEnabled(False)
        self.ClassLabels.setEnabled(False)
        self.ClassificationResult.setEnabled(False)
        self.JudgeButton.setEnabled(True)

    def correct(self):
        self.correct_class = self.judged_class
        present_time = datetime.datetime.now()
        shutil.copy('temp28x28.png', './images/label{0}/img{1:%Y%m%d%H%M%S}.png'.format(self.correct_class, present_time))
        self.clearSlate()

    def submit(self):
        self.correct_class = self.ClassLabels.currentIndex()
        present_time = datetime.datetime.now()
        shutil.copy('temp28x28.png', './images/label{0}/img{1:%Y%m%d%H%M%S}.png'.format(self.correct_class, present_time))
        self.clearSlate()

    def callSecretDialog(self):
        self.secret_dialog = SecretDialog()
        self.secret_dialog.show()

    def callModelSelectDialog(self):
        self.model_select_dialog = ModelSelectDialog()
        self.model_select_dialog.show()

    def establishConnections(self):
        PyQt4.QtCore.QObject.connect(self.BrushButton, PyQt4.QtCore.SIGNAL('clicked()'), self.switchBrush)
        PyQt4.QtCore.QObject.connect(self.JudgeButton, PyQt4.QtCore.SIGNAL('clicked()'), self.judge)
        PyQt4.QtCore.QObject.connect(self.ClearButton, PyQt4.QtCore.SIGNAL('clicked()'), self.clearSlate)
        PyQt4.QtCore.QObject.connect(self.ThicknessSpinner, PyQt4.QtCore.SIGNAL('valueChanged(int)'), self.changeThickness)
        PyQt4.QtCore.QObject.connect(self.CorrectButton, PyQt4.QtCore.SIGNAL('clicked()'), self.correct)
        PyQt4.QtCore.QObject.connect(self.SubmitButton, PyQt4.QtCore.SIGNAL('clicked()'), self.submit)
        PyQt4.QtCore.QObject.connect(self.SecretAction, PyQt4.QtCore.SIGNAL('triggered()'), self.callSecretDialog)
        PyQt4.QtCore.QObject.connect(self.ModelSelectAction, PyQt4.QtCore.SIGNAL('triggered()'), self.callModelSelectDialog)


if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)

    main_form = MainUI()
    main_form.show()

    sys.exit(app.exec_())
