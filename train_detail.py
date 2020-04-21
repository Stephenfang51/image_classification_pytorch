import argparse
import os
import utils as util

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class train_detail():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser = argparse.ArgumentParser()

        self._parser.add_argument('--train_path', type=str, help='train dataset path')
        self._parser.add_argument('--model', type=str, help='se_resnext50_32x4d || resnet18 || efficientb0 || efficientb5 || resnest50 || resnest101')
        self._parser.add_argument('--checkpoints', type=str,  default='result_model', help='your checkpoints model name')
        self._parser.add_argument('--loss', type=str, default = 'CrossEntropy', help='FocalLoss || CrossEntropy || LabelSmoothSoftmaxCE')
        self._parser.add_argument('--freeze', type=str2bool, default = False, help='freeze backbone params')
        self._parser.add_argument('--resume', type=str2bool, default = False, help='if use checkpoints or not')
        self._parser.add_argument('--num_classes', type=int, default = '3', help='number of classes')
        self._parser.add_argument('--input_size', type=int, default = '224', help='size of input image')
        self._parser.add_argument('--batch_size', type=int, default = '256', help='batch to put in')
        self._parser.add_argument('--num_epochs', type=int, default = '600', help='total epochs to train')
        self._parser.add_argument('--init_lr', type=float, default = '0.0001', help='learning rate at first')
        self._parser.add_argument('--lr_scheduler', type=str, help ='steplr || warmup') #TODO 新增fixed lr
        self._parser.add_argument('--step_size', type=int, default = '20', help='step size betweeen next adjustion')
        self._parser.add_argument('--multiplier', type=int, default = '80', help='?')
        self._parser.add_argument('--total_epoch', type=int, default = '20', help='?')
        self._parser.add_argument('--alpha', type=float, default = '0.2', help='focal loss')
        self._parser.add_argument('--gamma', type=int, default = '5', help='focal loss')
        #--- data aug ----#
        self._parser.add_argument('--re', type=str2bool, default=False, help='use of RandomErasing')
        # self._parser.add_argument('--warm_up', type=str2bool, default = True, help='if warm up lr at first training')

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # self.set_zero_thread_for_Win()

        # set is train or set
        # self._opt.is_train = self.is_train

        # set and check load_epoch
        # self._set_and_check_load_epoch()

        # get and set gpus
        # self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        # self._save(args)

        return self._opt

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')



    # def _save(self, args):
    #     expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
    #     print(expr_dir)
    #     util.mkdirs(expr_dir)
    #     file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
    #     with open(file_name, 'wt') as opt_file:
    #         opt_file.write('------------ Options -------------\n')
    #         for k, v in sorted(args.items()):
    #             opt_file.write('%s: %s\n' % (str(k), str(v)))
    #         opt_file.write('-------------- End ----------------\n')
