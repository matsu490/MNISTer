import sys
import PyQt4.QtGui
import PyQt4.QtCore
import PyQt4.uic
import cv2
import numpy as np
import pickle

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
        self.current_colour = Colour3(0,0,0)
        self.shape_num = 0
        self.current_width = 20
        params = self.unpickle('params.pkl')
        self.network = TwoLayerNet(params)

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
        img = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)
        img28x28 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        negaposi = cv2.bitwise_not(img28x28)
        cv2.imwrite('temp28x28.png', negaposi)
        scaled_pixmap = PyQt4.QtGui.QPixmap('temp28x28.png')
        scaled_pixmap = scaled_pixmap.scaledToHeight(200)
        self.label.setPixmap(scaled_pixmap)
        scaled_img_array = cv2.imread('temp28x28.png', cv2.IMREAD_GRAYSCALE)
        digit = self.network.judge(scaled_img_array.reshape(784))
        self.ClassificationResult.setText(str(digit))

    def changeThickness(self, num):
        self.current_width = num

    def clearSlate(self):
        self.drawing_shapes = Shapes()
        self.paint_panel.repaint()
        self.label.clear()

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


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (w * h)
        it.iternext()

    return grad


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


class TwoLayerNet(object):
    def __init__(self, params, input_size=784, hidden_size=100, output_size=10):
        self.params = {}
        self.params['W1'] = params['W1']
        self.params['b1'] = params['b1']
        self.params['W2'] = params['W2']
        self.params['b2'] = params['b2']

    def initParams(self, input_size=784, hidden_size=100, output_size=10, weight_init_std=0.01):
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def judge(self, x):
        return np.argmax(self.predict(x))

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)
    main_form = CreateUI()
    main_form.show()
    sys.exit(app.exec_())
