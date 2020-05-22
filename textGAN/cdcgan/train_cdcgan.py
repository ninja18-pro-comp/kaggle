import argparse
import os
 
import chainer
from chainer import training
from chainer.training import extensions
 
from cdcgan import Discriminator
from cdcgan import Generator
from updater import ConditionalDCGANUpdater
from visualizer import out_generated_image
 
 
def main():
    parser = argparse.ArgumentParser(description='Chainer: DCGAN MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--n_labels', '-l', type=int, default=10,
                        help='Number of labels')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()
 
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')
 
    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden, n_labels=args.n_labels)
    dis = Discriminator(n_labels=args.n_labels)
 
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()
 
    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
 
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)
 
    # Load the MNIST dataset
    train, _ = chainer.datasets.get_mnist(ndim=3, scale=255.) # ndim=3 : (ch,width,height)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
 
    # Set up a trainer
    updater = ConditionalDCGANUpdater(models=(gen, dis), iterator=train_iter, optimizer={'gen':opt_gen, 'dis':opt_dis}, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
 
    epoch_interval = (1, 'epoch')
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    # trainer.extend(extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    # trainer.extend(extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'dis/loss',]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(('gen/loss', 'dis/loss')))
    trainer.extend(out_generated_image(gen, dis, 10, 10, args.seed, args.out), trigger=epoch_interval)
 
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
 
    # Run the training
    trainer.run()
 
 
if __name__ == '__main__':
    main()