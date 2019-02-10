import numpy as np
import chainer

class dnn(chainer.Chain):
    def __init__(n_in=13,hid=256,n_out=2):
        super(dnn, self).__init__(
            self.l1=L.Linear(n_in, hid),
            self.l2=L.Linear(hid, hid),
            self.l3=L.Linear(hid, hid),
            self.l4=L.Linear(hid,n_out),
            self.bnorm1=L.BatchNormalization(c1),
            self.bnorm2=L.BatchNormalization(c2),
            self.bnorm3=L.BatchNormalization(c3)
        )

    def __call__(self, x):
        h = F.relu(self.bnorm1(self.l1(x)))
        h = F.relu(self.bnorm2(self.l2(h)))
        h = F.relu(self.bnorm3(self.l3(h)))
        y = self.l4(h)
        return y
        