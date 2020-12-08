"""All the layer functions go here.

"""
#Chenwei Wu 
#HW2
from __future__ import print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the fully connected layer.
        b (np.array): the biases of the fully connected layer.
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        name (str): the name of the layer.

    """

    def __init__(
        self, d_in, d_out, weights_init=None, bias_init=None, name="FullyConnected"
    ):
        shape = (d_out, d_in)
        self.W = weights_init.initialize(shape) \
            if weights_init else np.random.randn(*shape).astype(np.float32)
        self.b = bias_init.initialize((shape[0])) \
            if bias_init else np.random.randn(shape[0]).astype(np.float32)
        self.shape = shape
        self.name = name

    def __repr__(self):
        return "{}({}, {})".format(self.name, self.shape[0], self.shape[1])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        Y = np.dot(self.W, x) + self.b
        return Y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer.
            dv_y (np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input.
            dv_W (np.array): The derivative of the loss with respect to the
                weights.
            dv_b (np.array): The derivative of the loss with respect to the
                biases.

        """

        # TODO: write your implementation below
        #print("w",self.W.shape)
        #print("x",x.shape)
        #print("dv/dy",dv_y.shape)
        dv_x = np.empty(x.shape, dtype=np.float32)
        dv_W = np.empty(self.W.shape, dtype=np.float32)
        dv_b = np.empty(self.b.shape, dtype=np.float32)
        dv_x = np.dot(dv_y,self.W)
        dv_b = np.dot(dv_y,1)
        dv_W = np.dot(dv_y.reshape(dv_y.shape[0],1),x.reshape(1,x.shape[0]))
        # don't change the order of return values
        return dv_x, dv_W, dv_b


class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, filter_height, filter_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            in_channels).
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, in_channel, out_channel, kernel_size, stride, padding,
            weights_init=None, bias_init=None, name="Conv2D"):
        filter_size = (out_channel, in_channel, *kernel_size)

        self.W = weights_init.initialize(filter_size) \
            if weights_init else np.random.randn(*filter_size).astype(np.float32)
        self.b = bias_init.initialize((filter_size[0], 1)) \
            if bias_init else np.random.randn(out_channel, 1).astype(np.float32)

        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        y_shape = (
            self.W.shape[0],
            int((x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0]) + 1,
            int((x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1]) + 1,
        )
        y = np.empty(y_shape, dtype=np.float32)

        for k in range(y.shape[0]):
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    y[k, i, j] = np.sum(
                        x_padded[
                            :,
                            i * s[0] : i * s[0] + self.W.shape[2],
                            j * s[1] : j * s[1] + self.W.shape[3]
                        ] * self.W[k]
                    ) + self.b[k]
        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).
            dv_y (np.array): The derivative of the loss with respect to the
                output. A 3D array of shape (out_channels, out_heights,
                out_weights).

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input. It has the same shape as x.
            dv_W (np.array): The derivative of the loss with respect to the
                weights. It has the same shape as self.W
            dv_b (np.array): The derivative of the loss with respect to the
                biases. It has the same shape as self.b

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )


        dv_W = np.empty(self.W.shape, dtype=np.float32)
        dv_b = np.empty(self.b.shape, dtype=np.float32)
        dv_x = np.empty(x.shape, dtype=np.float32)
        dv_W.fill(0)
        dv_b.fill(0)
        dv_x.fill(0)
        #https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
        #inspired by 
        C_in, W_in, H_in = x.shape
        print("----C_in",C_in,"W_in",W_in,"H_in",H_in)
        
        C_out, W_out, H_out = dv_y.shape
        print("C_out",C_out,"W_out",W_in,"H_out",H_in)
        C_out, C_in, K_h, K_w = dv_W.shape
        
        print("C_out",C_out,"C_in",C_in,"K_h",K_h,"K_w",K_w)
        s_w, s_h = s
        print("s_w",s_w,"s_h",s_h)
        p_w, p_h = p        
        print("p_w",s_w,"p_h",s_h)
        
  
        
        for inn in range(C_in):            
            for outt in range(C_out):
                for m in range(K_h):
                    for n in range(K_w):
                        for i in range(W_out):
                            for j in range(H_out):
                                dv_W[outt, inn, m, n] += dv_y[outt, i, j] * x_padded[inn, i * s_w + m, j * s_h + n]

        for outt in range(C_out):
            dv_b[outt] += np.sum(dv_y[outt, :, :])
        
        
        for inn in range(C_in):            
            for outt in range(C_out):
                for v in range(W_in):
                    for u in range(H_in):
                        for m in range(K_h):
                            for n in range(K_w):
                                ip = (v + p_w - m) / s_w 
                                jp = (u + p_h - n) / s_h 
                                if (ip >= 0 and jp >= 0 and ip < W_out and jp < H_out ):
                                    dv_x[inn, v, u] += dv_y[outt, int(ip), int(jp)] * self.W[outt,inn, m, n]
        return dv_x, dv_W, dv_b


class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
            'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
            'Width doesn\'t work'

        y_shape = (
            x.shape[0],
            int((x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0]) + 1,
            int((x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1]) + 1,
        )
        y = np.empty(y_shape, dtype=np.float32)

        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                y[:, i, j] = np.max(x_padded[
                                    :,
                                    i * s[0]: i * s[0] + self.kernel_size[0],
                                    j * s[1]: j * s[1] + self.kernel_size[1]
                                    ].reshape(-1, self.kernel_size[0] * self.kernel_size[1]),
                                    axis=1
                                    )

        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
                respect to the input.

                Args:
                    x (np.array): the input of the layer. A 3D array of shape (
                        in_channels, in_heights, in_weights).
                    dv_y (np.array): The derivative of the loss with respect to the
                        output. A 3D array of shape (out_channels, out_heights,
                        out_weights).

                Returns:
                    dv_x (np.array): The derivative of the loss with respect to the
                        input. It has the same shape as x.
                """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        dv_x = np.empty(x.shape, dtype=np.float32)
        dv_x.fill(0)
        dv_xp = np.empty(x_padded.shape, dtype=np.float32)
        dv_xp.fill(0)
        #https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html
        #inspired by 
        C_in, W_in, H_in = x.shape
        
        C_out, W_out, H_out = dv_y.shape
        K_w, K_h = self.kernel_size
        s_w, s_h = s
        p_w, p_h = p

        def unpad(x, pad_width):
            slices = []
            for c in pad_width:
                e = None if c[1] == 0 else -c[1]
                slices.append(slice(c[0], e))
            return x[tuple(slices)]
        for t in range(C_in):
            for v in range(W_out):
                for u in range(H_out):
                            x_pool=x_padded[t,v*s_w:v*s_w+K_w,u*s_h:u*s_h+K_h]
                            mask=(x_pool==np.max(x_pool))
                            dv_xp[t, v*s_w:v*s_w+K_w,u*s_h:u*s_h+K_h] += dv_y[t, v,u] * mask
                            
        dv_x=unpad(dv_xp,((p[0], p[0]), (p[1], p[1])))

        return dv_x

