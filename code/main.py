from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import re
import sys
import glob
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms


import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=123)
    parser.add_argument('--b_validation', type=bool)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--train_split', type=float)
    parser.add_argument('--validation_split', type=float)

    # Train params
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--snapshot-interval', type=int)
    parser.add_argument('--net_g', default='')
    parser.add_argument('--b_net_d', type=bool)
    parser.add_argument('--discriminator_lr', type=float)
    parser.add_argument('--generator_lr', type=float)
    parser.add_argument('--net_e', default='')
    parser.add_argument('--gamma1', type=float)
    parser.add_argument('--gamma2', type=float)
    parser.add_argument('--gamma3', type=float)
    parser.add_argument('--lambda', dest='lambda_', type=float)

    # GAN
    parser.add_argument('--df_dim', type=int)
    parser.add_argument('--gf_dim', type=int)
    parser.add_argument('--z_dim', type=int)
    parser.add_argument('--r_num', type=int)

    # Text arguments
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--captions_per_image', type=int)
    parser.add_argument('--words_num', type=int)

    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '{}/example_filenames.txt'.format(cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '{}/{}/{}.txt'.format(cfg.DATA_DIR, 'text', name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.b_validation is not None:
        cfg.B_VALIDATION = args.b_validation

    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.train_split is not None:
        cfg.TRAIN_SPLIT = args.train_split

    if args.validation_split is not None:
        cfg.VALIDATION_SPLIT = args.validation_split

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.output_dir != '':
        cfg.OUTPUT_DIR = args.output_dir

    # Train params
    if args.max_epoch is not None:
        cfg.TRAIN.MAX_EPOCH = args.max_epoch

    if args.snapshot_interval is not None:
        cfg.TRAIN.SNAPSHOT_INTERVAL = args.snapshot_interval

    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.net_g != '':
        cfg.TRAIN.NET_G = args.net_g

    # Evaluate param
    if args.b_net_d is not None:
        cfg.TRAIN.B_NET_G = args.b_net_d

    if args.predict:
        cfg.TRAIN.FLAG = False

    ####
    if args.discriminator_lr is not None:
        cfg.TRAIN.DISCRIMINATOR_LR = args.discriminator_lr

    if args.generator_lr is not None:
        cfg.TRAIN.GENERATOR_LR = args.generator_lr

    if args.net_e != '':
        cfg.TRAIN.NET_E = args.net_e

    if args.gamma1 is not None:
        cfg.TRAIN.SMOOTH.GAMMA1 = args.gamma1

    if args.gamma2 is not None:
        cfg.TRAIN.SMOOTH.GAMMA2 = args.gamma2

    if args.gamma3 is not None:
        cfg.TRAIN.SMOOTH.GAMMA3 = args.gamma3

    if args.lambda_ is not None:
        cfg.TRAIN.SMOOTH.LAMBDA = args.lambda_

    # GAN
    if args.df_dim is not None:
        cfg.GAN.DF_DIM = args.df_dim

    if args.gf_dim is not None:
        cfg.GAN.GF_DIM = args.gf_dim

    if args.z_dim is not None:
        cfg.GAN.Z_DIM = args.z_dim

    if args.r_num is not None:
        cfg.GAN.R_NUM = args.r_num

    # Text arguments
    if args.embedding_dim is not None:
        cfg.TEXT.EMBEDDING_DIM = args.embedding_dim

    if args.captions_per_image is not None:
        cfg.TEXT.CAPTIONS_PER_IMAGE = args.captions_per_image

    if args.words_num is not None:
        cfg.TEXT.WORDS_NUM = args.words_num

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = cfg.OUTPUT_DIR + '/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
