import shutil, sys, datetime
import PyQt4.QtGui
import PyQt4.QtCore
import PyQt4.uic
import cv2
import numpy as np
import pickle
from collections import OrderedDict

uifile = './mainUI.ui'
form, base = PyQt4.uic.loadUiType(uifile)
uifile = './dialogUI.ui'
form1, base1 = PyQt4.uic.loadUiType(uifile)


class SecretDialog(form1, base1):
    def __init__(self, parent=None):
        super(SecretDialog, self).__init__(parent)
        self.setupUi(self)


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
        self.current_colour = Colour3(0,0,0)
        self.shape_num = 0
        self.current_width = 20
        params = self.unpickle('params_trained_simpleconv.pkl')
        # self.network = TwoLayerNet(params)
        self.network = SimpleConvNet(params=params)

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

    def establishConnections(self):
        PyQt4.QtCore.QObject.connect(self.BrushButton, PyQt4.QtCore.SIGNAL('clicked()'), self.switchBrush)
        PyQt4.QtCore.QObject.connect(self.JudgeButton, PyQt4.QtCore.SIGNAL('clicked()'), self.judge)
        PyQt4.QtCore.QObject.connect(self.ClearButton, PyQt4.QtCore.SIGNAL('clicked()'), self.clearSlate)
        PyQt4.QtCore.QObject.connect(self.ThicknessSpinner, PyQt4.QtCore.SIGNAL('valueChanged(int)'), self.changeThickness)
        PyQt4.QtCore.QObject.connect(self.CorrectButton, PyQt4.QtCore.SIGNAL('clicked()'), self.correct)
        PyQt4.QtCore.QObject.connect(self.SubmitButton, PyQt4.QtCore.SIGNAL('clicked()'), self.submit)
        PyQt4.QtCore.QObject.connect(self.SecretAction, PyQt4.QtCore.SIGNAL('triggered()'), self.callSecretDialog)


def sigmoid(x):
    # measure to deal with overflow
    x = np.clip(x, -709, 100000)
    return 1. / (1. + np.exp(-x))


def softmax(x):
    c = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))


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


class Relu(object):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Affine(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_xhape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_xhape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_xhape)
        return dx


class SoftmaxWithLoss(object):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in xrange(filter_h):
        y_max = y + stride * out_h
        for x in xrange(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in xrange(filter_h):
        y_max = y + stride * out_h
        for x in xrange(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Pooling(object):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class Convolution(object):
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class SimpleConvNet(object):
    def __init__(self, params, input_dim=(1, 28, 28),
            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
            hidden_size=100,
            output_size=10,
            weight_init_std=0.01):

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = params['W1']
        self.params['W2'] = params['W2']
        self.params['W3'] = params['W3']
        self.params['b1'] = params['b1']
        self.params['b2'] = params['b2']
        self.params['b3'] = params['b3']

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(
                self.params['W1'],
                self.params['b1'],
                conv_param['stride'],
                conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def initParams(self):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def judge(self, x):
        return np.argmax(self.predict(x.reshape(1, 1, 28, 28)))

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in xrange(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {} 
        for idx in xrange(1, 4):
            grads['W{}'.format(idx)] = numerical_gradient(loss_w, self.params['W{}'.format(idx)])
            grads['b{}'.format(idx)] = numerical_gradient(loss_w, self.params['b{}'.format(idx)])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['W2'] = self.layers['Affine1'].dW
        grads['W3'] = self.layers['Affine2'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['b2'] = self.layers['Affine1'].db
        grads['b3'] = self.layers['Affine2'].db
        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W{}'.format(i + 1)]
            self.layers[key].b = self.params['b{}'.format(i + 1)]

if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)

    main_form = MainUI()
    main_form.show()

    sys.exit(app.exec_())
