import numpy
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
 
def label2onehot(n_labels, labels):
    return numpy.array([numpy.eye(n_labels)[label] for label in labels]).reshape(-1,n_labels,1,1)

class Generator(chainer.Chain):
    def __init__(self, n_hidden, n_labels, bottom_width=3, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.n_labels = n_labels
        self.ch = ch
        self.bottom_width = bottom_width
 
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(in_size=n_hidden + n_labels, out_size=bottom_width*bottom_width*ch, initialW=w)
 
            self.dc1 = L.Deconvolution2D(in_channels=ch, out_channels=ch//2, ksize=2, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(in_channels=ch//2, out_channels=ch//4, ksize=2, stride=2, pad=1, initialW=w)
            self.dc3 = L.Deconvolution2D(in_channels=ch//4, out_channels=ch//8, ksize=2, stride=2, pad=1, initialW=w)
            self.dc4 = L.Deconvolution2D(in_channels=ch//8, out_channels=1, ksize=3, stride=3, pad=1, initialW=w)
 
            self.bn0 = L.BatchNormalization(size=self.bottom_width*self.bottom_width*self.ch)
            self.bn1 = L.BatchNormalization(size=ch)
            self.bn2 = L.BatchNormalization(size=ch//2)
            self.bn3 = L.BatchNormalization(size=ch//4)
            self.bn4 = L.BatchNormalization(size=ch//8)
 
    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(numpy.float32)
 
    def __call__(self, z, t):
        xp = cuda.get_array_module(z.data)
        t = xp.array(label2onehot(self.n_labels, t),dtype=xp.float32)
        h = self.l0(F.concat((z, t), axis=1))
        h = F.relu(self.bn0(h))
        h = F.reshape(h, (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(h))
        h = F.relu(self.bn2(self.dc1(h)))
        h = F.relu(self.bn3(self.dc2(h)))
        h = F.relu(self.bn4(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))
        # x = F.tanh(self.dc4(h))
        return x
 
class Discriminator(chainer.Chain):
    def __init__(self, n_labels, bottom_width=3, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.n_labels = n_labels
            self.c0 = L.Convolution2D(in_channels=1+n_labels, out_channels=64, ksize=3, stride=3, pad=1, initialW=w)
            self.c1 = L.Convolution2D(in_channels=ch//8, out_channels=128, ksize=2, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(in_channels=ch//4, out_channels=256, ksize=2, stride=2, pad=1, initialW=w)
            self.c3 = L.Convolution2D(in_channels=ch//2, out_channels=512, ksize=2, stride=2, pad=1, initialW=w)
            self.l4 = L.Linear(in_size=None, out_size=1, initialW=w)
 
            self.bn0 = L.BatchNormalization(size=ch//8, use_gamma=False)
            self.bn1 = L.BatchNormalization(size=ch//4, use_gamma=False)
            self.bn2 = L.BatchNormalization(size=ch//2, use_gamma=False)
            self.bn3 = L.BatchNormalization(size=ch//1, use_gamma=False)
 

    def __call__(self, x, t):
        xp = cuda.get_array_module(x.data)
        t = xp.array(label2onehot(self.n_labels, t),dtype=xp.float32)
        t = t*xp.ones((len(x), self.n_labels, 28, 28),dtype=xp.float32)
        h = F.concat((x, t), axis=1)
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        y = self.l4(h)
 
        return y