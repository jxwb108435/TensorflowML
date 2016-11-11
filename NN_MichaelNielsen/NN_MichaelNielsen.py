# coding=utf-8
# -*- coding:cp936 -*-
""" http://neuralnetworksanddeeplearning.com/index.html """

# 如何为神经网络量身设计一种学习算法呢?
# 假设:
# 网络权值(偏移)上的小改变, 将会为网络的输出结果带来相应的改变, 且这种改变也必须是轻微的, 满足这样的性质才能使学习变得可能
# 以此性质为基础, 我们就可以改变权值和偏移来使得网络的表现越来越接近我们预期

# 例如, 假设原始的网络会将一张写着「9」的手写数字图片错误分类为「8」
# 我们可以尝试找到一个正确的轻微改变权值和偏移的方法, 来使得我们网络的输出更接近于正确答案——将该图片分类为「9」
# 重复这个过程, 不断地修改权值和偏移并且产生越来越好的结果, 这样我们的网络就开始学习起来了

# 感知机的局限
# 但问题在于, 当我们的网络包含感知机时情况就与上述描述的不同了
# 事实上, 轻微改变网络中任何一个感知机的权值或偏移有时甚至会导致感知机的输出完全翻转——比如说从0变为1
# 这个翻转行为可能以某种非常复杂的方式彻底改变网络中其余部分的行为
# 所以即使现在「9」被正确分类了, 但网络在处理所有其他图片时的行为可能因一些难以控制的方式被彻底改变了
# 这导致我们逐步改变权值和偏移来使网络行为更加接近预期的学习方法变得很难实施
# 也许存在一些巧妙的方法来避免这个问题, 但对于这种由感知机构成的网络, 它的学习算法并不是显而易见的

# 由此我们引入 sigmoid 神经元来解决这个问题
# sigmoid 神经元与感知机有些相似, 但做了一些修改使得轻微改变其权值和偏移时只会引起小幅度的输出变化
# 这是使由 sigmoid 神经元构成的网络能够学习的关键因素

# 如果把σ函数换成阶梯函数, 那么 sigmoid 神经元就变成了一个感知机
# 这是因为此时它的输出只随着 w⋅x+b 的正负不同而仅在1或0这两个离散值上变化
# 当使用σ函数时我们就得到了一个平滑的感知机, σ函数的平滑属性才是其关键, 不用太在意它的具体代数形式
# σ函数的平滑属性意味着当我们在权值和偏移上做出值为 Δwj, Δb 的轻微改变时, 神经元的输出也将只是轻微地变化 Δoutput
# Δoutput ≈ sum(∂output/∂wj Δwj, j) + ∂output/∂b Δb

# 我们可以把手写数字识别问题拆分为两个子问题
# 首先, 找到一种方法能够把一张包含若干数字的图像分割为若干小图片, 其中每个小图像只包含一个数字
# 当图像被分割之后, 接下来的任务就是如何识别每个独立的手写数字

# 我们将把精力集中在实现程序去解决第二个问题, 即如何正确分类每个单独的手写数字
# 因为事实证明, 只要你解决了数字分类的问题, 分割问题相对来说不是那么困难
# 分割问题的解决方法有很多
# 一种方法是尝试不同的分割方式, 用数字分类器对每一个切分片段打分
# 如果数字分类器对每一个片段的置信度都比较高, 那么这个分割方式就能得到较高的分数
# 如果数字分类器在一或多个片段中出现问题, 那么这种分割方式就会得到较低的分数
# 这种方法的思想是, 如果分类器有问题, 那么很可能是由于图像分割出错导致的
# 这种思想以及它的变种能够比较好地解决分割问题
# 因此, 与其关心分割问题, 我们不如把精力集中在设计一个神经网络来解决更有趣, 更困难的问题, 即手写数字的识别

# 定义输入 x, 定义网络输出为 a, 定义分类为 y
# 代价函数:
# 为评价网络输出 a 距离目标结果 y 的程度, 需要一个算法找到合适的权重和偏置, 对所有的训练输入为 x 的输出 a 都近似于y
# C(w, b) = (1/2n) * sum((yi - ai)**2, i=1,2,...,n)
# 我们称C为二次代价函数, 有时我们也称它为平方误差或者MSE
# 显而易见, 输出 a 取决于 x, w, b

# 通过代价函数的形式我们可以得知C(w, b)是非负的, 因为加和的每一项都是非负的
# 另外, 当所有的训练输入 x 的输出 a 基本都等于 y 时, 代价函数C(w,b)的值会变小, 可能有C(w,b)≈0
# 因此如果我们的学习算法可以找到合适的权重和偏置使得C(w,b)≈0, 那么这就是一个好的学习算法
# 反过来来说, 如果C(w,b)很大, 那么就说明学习算法的效果很差——这意味着对于大量的输入, 我们的结果 a 与正确结果 y 相差很大
# 因此, 我们的训练算法的目标就是通过调整函数的权重和偏置来最小化代价函数C(w,b)
# 换句话说, 我们想寻找合适的权重和偏置让代价函数尽可能地小, 我们将使用一个叫做梯度下降法(gradient descent)的算法来达到这个目的

import numpy as np
from random import shuffle
import pickle
import gzip


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    _training_data, _validation_data, _test_data = pickle.load(f, encoding='bytes')
    f.close()
    return _training_data, _validation_data, _test_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarray containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    _training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    _validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    _test_data = zip(test_inputs, te_d[1])
    return _training_data, _validation_data, _test_data


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    # 当输入z是一个向量或者Numpy数组时, Numpy自动的应用元素级的sigmoid函数, 也就是向量化


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        """ The list ``sizes`` contains the number of neurons in the respective layers of the network.
        For example, if the list was [2, 3, 1] then it would be a three-layer network,
        with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs from later layers. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # 后面将会发现更好的初始化权重和偏差的方法, 但是现在将采用随机初始化
        # 假设第一层神经元是一个输入层, 并对这些神经元不设置任何偏差, 因为偏差仅在之后的层中使用
        # 偏差和权重以列表存储在Numpy矩阵中

    def feedforward(self, a):
        """ Return the output of the network if "a" is input. """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
            # 假定输入 a 是Numpy的n维数组(n,1), 而不是向量(n,) n是输入到网络的数目
            # 虽然使用向量(n,)看上去好像是更自然的选择
            # 但是使用n维数组(n,1)使它特别容易的修改代码来立刻前馈多层输入, 并且有的时候这很方便
        return a

    def update_mini_batch(self, mini_batch, eta):
        """ Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta" is the learning rate. """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, _test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in _test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y

    def SGD(self, _training_data, epochs, mini_batch_size, eta, _test_data=None):
        """ Train using mini-batch stochastic gradient descent.
        The "training_data" is a list of tuples "(x, y)" representing the inputs and the desired outputs.
        The other non-optional parameters are self-explanatory.
        If "test_data" is provided then the network will be evaluated against the test data after each epoch,
        and partial progress printed out.
        This is useful for tracking progress, but slows things down substantially. """
        if _test_data: n_test = len(_test_data)
        n = len(_training_data)
        for j in range(epochs):
            shuffle(_training_data)
            mini_batches = [_training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if _test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(_test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


training_data, validation_data, test_data = load_data_wrapper()  # type of zip
training_data = list(training_data)  # from zip to list
validation_data = list(validation_data)
test_data = list(test_data)
print('finished')

net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, _test_data=test_data)

'''
net = Network([2, 3, 4, 1])

net.sizes
[2, 3, 4, 1]

net.biases
[
array([[-0.23156881], [ 1.43512051], [ 0.1168981 ]]),
array([[-1.18827722], [-0.10952338], [ 1.1448427 ], [-0.55425634]]),
array([[ 0.49222209]])
]

net.weights
[
array([[ 0.2638044 , -1.60106835],
      [-1.75854801,  0.00618937],
      [ 0.74749548, -0.48369898]]),

 array([[ 1.82634775, -0.53237056, -0.44476561],
        [-0.54723161, -1.37861775, -0.10106673],
        [ 0.45659677,  2.44257481,  2.80789491],
        [-0.86749162, -0.67488724,  1.44580879]]),

 array([[-0.35329474, -1.56688543, -0.10181945, -1.26570313]])
]
'''
