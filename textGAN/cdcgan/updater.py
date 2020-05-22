import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
class ConditionalDCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")
        super(ConditionalDCGANUpdater, self).__init__(*args, **kwargs)
    
    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real))/batchsize
        L2 = F.sum(F.softplus(y_fake))/batchsize
        loss = L1 + L2
        chainer.report({"loss":loss},dis)
        return loss
    
    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()        
        batchsize = len(batch)
        images = [batch[i][0] for i in range(batchsize)]
        labels = [batch[i][1] for i in range(batchsize)]

        x_real = Variable(self.converter(images, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)
        gen, dis = self.gen, self.dis

        y_real = dis(x_real,labels)
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z,labels)
        y_fake = dis(x_fake,labels)
        # fake_labels = np.random.randint(0,10,batchsize)
        # x_fake = gen(z,fake_labels)
        # y_fake = dis(x_fake,fake_labels)
        

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
