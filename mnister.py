import sys
import PyQt4.QtGui
import PyQt4.QtCore
import PyQt4.uic
import cv2

uifile = './mainUI.ui'
form, base = PyQt4.uic.loadUiType(uifile)


class Colour3(object):
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
    def __init__(self, coordinate=Point(0, 0), width=0.0, colour=Colour3(0, 0, 0), shape_number=0):
        self.coordinate = coordinate
        self.width = width
        self.colour = colour
        self.shape_number = shape_number


class Shapes(object):
    def __init__(self):
        self.__Shapes = []

    def countShape(self):
        return len(self.__Shapes)

    def updateShape(self, coordinate, width, colour, shape_number):
        shape = Shape(coordinate, width, colour, shape_number)
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
                    self.parent_link.current_colour,
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
                    PyQt4.QtGui.QColor(T.colour.R, T.colour.G, T.colour.B),
                    T.width / 2,
                    PyQt4.QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(
                    T.coordinate.x,
                    T.coordinate.y,
                    T1.coordinate.x,
                    T1.coordinate.y)


class CreateUI(base, form):
    is_brush = True
    drawing_shapes = Shapes()
    is_painting = False
    is_eraseing = False

    current_colour = Colour3(0,0,0)
    current_width = 10
    shape_num = 0
    is_mouseing = False
    paint_panel = 0

    def __init__(self):
        super(base, self).__init__()
        self.setupUi(self)
        self.setObjectName('MNISTer')
        self.paint_panel = Painter(self)
        self.paint_panel.close()
        self.BlackBoard.insertWidget(0, self.paint_panel)
        self.BlackBoard.setCurrentWidget(self.paint_panel)
        self.establishConnections()
        self.ClassLabels.addItem('0')
        self.ClassLabels.addItem('1')
        self.ClassLabels.addItem('2')
        self.ClassLabels.addItem('3')
        self.ClassLabels.addItem('4')
        self.ClassLabels.addItem('5')
        self.ClassLabels.addItem('6')
        self.ClassLabels.addItem('7')
        self.ClassLabels.addItem('8')
        self.ClassLabels.addItem('9')

    def switchBrush(self):
        self.is_brush = not self.is_brush

    def judge(self):
        pixmap = PyQt4.QtGui.QPixmap.grabWidget(self.paint_panel)
        pixmap.save('temp.jpg')
        img = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)
        img28x28 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        negaposi = cv2.bitwise_not(img28x28)
        cv2.imwrite('temp28x28.jpg', negaposi)

    def changeThickness(self, num):
        self.current_width = num

    def clearSlate(self):
        self.drawing_shapes = Shapes()
        self.paint_panel.repaint()

    def correct(self):
        pass

    def submit(self):
        pass

    def establishConnections(self):
        PyQt4.QtCore.QObject.connect(self.BrushButton, PyQt4.QtCore.SIGNAL('clicked()'), self.switchBrush)
        PyQt4.QtCore.QObject.connect(self.JudgeButton, PyQt4.QtCore.SIGNAL('clicked()'), self.judge)
        PyQt4.QtCore.QObject.connect(self.ClearButton, PyQt4.QtCore.SIGNAL('clicked()'), self.clearSlate)
        PyQt4.QtCore.QObject.connect(self.ThicknessSpinner, PyQt4.QtCore.SIGNAL('valueChanged(int)'), self.changeThickness)
        PyQt4.QtCore.QObject.connect(self.CorrectButton, PyQt4.QtCore.SIGNAL('clicked()'), self.correct)
        PyQt4.QtCore.QObject.connect(self.SubmitButton, PyQt4.QtCore.SIGNAL('clicked()'), self.submit)

if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)
    main_form = CreateUI()
    main_form.show()
    sys.exit(app.exec_())
