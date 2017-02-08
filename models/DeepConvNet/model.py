# coding: utf-8
from ..module import *


class Network(object):
    ''' Network construction
        conv - relu - conv - relu - pool -
        conv - relu - conv - relu - pool -
        conv - relu - conv - relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    '''
    def __init__(self, params):
        self.params = params
        conv_param1 = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}
        conv_param2 = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}
        conv_param3 = {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}
        conv_param4 = {'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1}
        conv_param5 = {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}
        conv_param6 = {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}

        # layer1
        self.layers = []
        self.layers.append(Convolution(
            self.params['W1'], self.params['b1'],
            conv_param1['stride'], conv_param1['pad']))
        self.layers.append(Relu())

        # layer2
        self.layers.append(Convolution(
            self.params['W2'], self.params['b2'],
            conv_param2['stride'], conv_param2['pad']))
        self.layers.append(Relu())

        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        # layer3
        self.layers.append(Convolution(
            self.params['W3'], self.params['b3'],
            conv_param3['stride'], conv_param3['pad']))
        self.layers.append(Relu())

        # layer4
        self.layers.append(Convolution(
            self.params['W4'], self.params['b4'],
            conv_param4['stride'], conv_param4['pad']))
        self.layers.append(Relu())

        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        # layer5
        self.layers.append(Convolution(
            self.params['W5'], self.params['b5'],
            conv_param5['stride'], conv_param5['pad']))
        self.layers.append(Relu())

        # layer6
        self.layers.append(Convolution(
            self.params['W6'], self.params['b6'],
            conv_param6['stride'], conv_param6['pad']))
        self.layers.append(Relu())

        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        # layer7
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))

        # layer8
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        # output layer
        self.last_layer = SoftmaxWithLoss()

    def initParams(self, input_dim=(1, 28, 28),
            conv_param1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
            conv_param2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
            conv_param3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
            conv_param4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
            conv_param5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
            conv_param6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
            hidden_size=50,
            output_size=10):

        pass
        # TODO: パラメータを初期化する処理を書く
        '''
        conv_params = [conv_param1, conv_param2, conv_param3, conv_param4, conv_param5, conv_param6]
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(conv_params):
            self.params['W{}'.format(idx + 1)] = weight_init_scales[idx] * \
                np.random.randn(
                    conv_param['filter_num'],
                    pre_channel_num,
                    conv_param['filter_size'],
                    conv_param['filter_size'])
            self.params['b{}'.format(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, hidden_size)
        self.params['b8'] = np.zeros(output_size)
        '''

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def judge(self, x):
        return np.argmax(self.predict(x.reshape(1, 1, 28, 28)))

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in xrange(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
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
        layers = self.layers[:]
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W{}'.format(i + 1)] = self.layers[layer_idx].dw
            grads['b{}'.format(i + 1)] = self.layers[layer_idx].db
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
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W{}'.format(i + 1)]
            self.layers[layer_idx].b = self.params['b{}'.format(i + 1)]
