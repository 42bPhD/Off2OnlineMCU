
import argparse
from utils import *
import random
import warnings
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from quantization.quant_qat_train import sgd_optimizer, WarmupCosineAnnealingLR, train, validate, load_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Whole Model Quant')
parser.add_argument(
                    "--data-dir",
                    default="E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96",
                    type=str,
                    help="Path to dataset (will be downloaded).",
                )
parser.add_argument(
        "--image-size", default=96, type=int, help="Input image size (square assumed)."
    )
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-C1')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of epochs for each run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=100, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='learning rate for finetuning', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', 
                    default="", 
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--base-weights', default=None, type=str,
                    help='weights of the base model.')
parser.add_argument('--tag', default='quantization', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('--fpfinetune', dest='fpfinetune', action='store_true',
                    help='full precision finetune')
parser.add_argument('--fixobserver', dest='fixobserver', action='store_true',
                    help='fix observer?')
parser.add_argument('--fixbn', dest='fixbn', action='store_true',
                    help='fix bn?')
parser.add_argument('--quantlayers', default='all', type=str, choices=['all', 'exclud_first_and_linear', 'exclud_first_and_last'],
                    help='the tag for identifying the log and model files. Just a string.')

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1
    log_file = 'quant_{}_exp.txt'.format(args.tag)

    #   1.  Build and load base model
    from repvgg import get_RepVGG_func_by_name
    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    base_model = repvgg_build_func(deploy=True)
    from tools.insert_bn import directly_insert_bn_without_init
    directly_insert_bn_without_init(base_model)
    if args.base_weights is not None:
        load_checkpoint(base_model, args.base_weights)

    #   2.
    if not args.fpfinetune:
        from quantization.repvgg_quantized import RepVGGWholeQuant
        qat_model = RepVGGWholeQuant(repvgg_model=base_model, quantlayers=args.quantlayers)
        print(qat_model)
        qat_model.prepare_quant()
    else:
        qat_model = base_model
        log_msg('===================== not QAT, just full-precision finetune ===========', log_file)

    #===================================================
    #   From now on, the code will be very similar to ordinary training
    # ===================================================
    for n, p in qat_model.named_parameters():
        print(n, p.size())
    for n, p in qat_model.named_buffers():
        print(n, p.size())
    log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)
    #   You will see it now has quantization-related parameters (zero-points and scales)

    qat_model = qat_model.cuda()
    from data.dataloader import get_dataloader
    # todo
    train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir,
                                                batch_size=args.batch_size,
                                                image_size=args.image_size,
                                                shuffle=True,
                                                num_workers=args.workers)


    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = sgd_optimizer(qat_model, args.lr, args.momentum, args.weight_decay)

    warmup_epochs = 1
    
    lr_scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, 
                                        T_cosine_max=args.epochs * (len(train_loader)+len(val_loader)),
                                        eta_min=0,
                                        warmup=warmup_epochs * (len(train_loader)+len(val_loader)))


    # optionally resume from a checkpoint
    
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        
        checkpoint = torch.load(args.resume)
        
        
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['max_accuracy']
        qat_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    if args.evaluate:
        validate(val_loader, qat_model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, qat_model, criterion, optimizer, epoch, args, lr_scheduler, is_main=True)

        if args.fixobserver and epoch > (3 * args.epochs // 8):
            # Freeze quantizer parameters
            qat_model.apply(torch.quantization.disable_observer)  #TODO testing. May not be useful
            log_msg('fix observer after epoch {}'.format(epoch), log_file)

        if args.fixbn and epoch > (2 * args.epochs // 8):    #TODO testing. May not be useful
        #     Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            log_msg('fix bn after epoch {}'.format(epoch), log_file)

        # evaluate on validation set
        acc1 = validate(val_loader, qat_model, criterion, args)
        msg = '{}, base{}, quant, epoch {}, QAT acc {}'.format(args.arch, args.base_weights, epoch, acc1)
        log_msg(msg, log_file)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': qat_model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }, is_best,
            filename = '{}_{}.pth'.format(args.arch, args.tag),
            best_filename='{}_{}_best.pth'.format(args.arch, args.tag))
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)