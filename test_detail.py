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


class test_detail():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser = argparse.ArgumentParser()

        self._parser.add_argument('--csv', type=str, help='test dataset csv')
        self._parser.add_argument('--pre_model', type=str, help='set pre-model path to test')
        self._parser.add_argument('--test_path', type=str, help='set test dataset path to test')


        self._parser.add_argument('--train_path', type=str, help='train dataset path')
        self._parser.add_argument('--input_size', type=int, default = '224', help='size of input image')
        self._parser.add_argument('--batch_size', type=int, default = '256', help='batch to put in')


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
