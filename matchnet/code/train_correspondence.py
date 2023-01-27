import argparse
import time

import torch
import numpy as np
# from torchvision import transforms
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.optim as optim
import os
import sys
import random
sys.path.append(os.getcwd())
from matchnet.code.ml.dataloader import get_corr_loader
from matchnet.code.ml.models.correspondence import CorrespondenceNet
from matchnet.code.ml import losses
from torch.utils.tensorboard import SummaryWriter   
import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import logging

SEED=666
tb_path = './tb_log'
if not os.path.exists(tb_path):
        os.makedirs(tb_path)
writer = SummaryWriter(tb_path)
logger= logging.getLogger(__file__)
def save_ckpt(savepath,epoch,model,optimizer,scheduler):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
        }
    if (epoch + 1) % 5 == 0:
        savedpath = savepath + 'corr_epoch' + str(epoch) + '.pth'
        torch.save(checkpoint, savedpath)
    savedpath = savepath + 'corr_last_epoch' + '.pth'
    torch.save(checkpoint, savedpath)

def set_seed(set_benchmark:bool):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)  
    torch.cuda.manual_seed(SEED)
    if set_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True

# python form2fit/code/train_correspondence.py 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Form2Fit suction Module")
    parser.add_argument("--batchsize", type=int, default=1, help="The batchsize of the dataset.")
    parser.add_argument("--sample_ratio", type=int, default=5, help="The ratio of negative to positive labels.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--augment","-a", action='store_true', help="(bool) Whether to apply data augmentation.")
    parser.add_argument("--background_subtract", type=tuple, default=None, help="apply mask.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--imgsize", type=list, default=[848,480], help="size of final image.")
    parser.add_argument("--root", type=str, default="", help="the path of dataset")
    parser.add_argument("--savepath", type=str, default="matchnet/code/ml/savedmodel/0127/", help="the path of saved models")
    parser.add_argument("--resume","-r",  action='store_true', help="whether to resume")
    parser.add_argument("--checkpoint","-c",  type=str, default="matchnet/code/ml/savedmodel/corr_final.pth", help="the path of resume models")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = opt.batchsize
    epochs = opt.epochs
    kit_name = "bear"
    savepath = opt.savepath
    
    testroot = "dataset/test"
    background_subtract = opt.background_subtract
    use_color = True
    num_channels = 4
    sample_ratio = opt.sample_ratio
    radius = 1
    
    set_seed(True)


    print("--------------start preparing data--------------")
    
    
    train_loader = get_corr_loader(kit_name, 
                                        dtype="train", 
                                        batch_size=batch_size, 
                                        use_color = use_color,
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=True,
                                        background_subtract=background_subtract)

    model = CorrespondenceNet(num_channels=num_channels, num_descriptor=64, num_rotations=20).to(device)
    criterion = losses.CorrespondenceLoss(sample_ratio=sample_ratio, device=device, margin=8, num_rotations=20, hard_negative=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    start_epoch = -1
    if opt.resume:
        state_dict = torch.load(opt.checkpoint, map_location=device)
        model.load_state_dict(state_dict["model"])
        start_epoch = state_dict["epoch"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
    # valid_loss = []
    train_epochs_loss = []
    # valid_epochs_loss = []
    one_epoch_step = len(train_loader)
    for epoch in tqdm(range(start_epoch +1, epochs)):
        model.train()
        train_epoch_loss = []

        for i, (imgs, labels, centers) in enumerate(train_loader):

            imgs = imgs.to(device)
            cuda_labels = []
            for j in range(len(labels)):
                cuda_labels.append(labels[j].to(device))

            out_s, out_t = model(imgs,centers[0][0], centers[0][1])
            optimizer.zero_grad()
            match_loss, no_match_loss = criterion(out_s, out_t, cuda_labels) #TODO:check output shape and split
            loss = (sample_ratio * match_loss) + no_match_loss
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())

            global_step = epoch * one_epoch_step + i
            writer.add_scalar("loss/step", loss.item(), global_step=global_step, walltime=None)
            writer.add_scalar("loss/match", match_loss.item(), global_step=global_step, walltime=None)
            writer.add_scalar("loss/no_match", no_match_loss.item(), global_step=global_step, walltime=None)
            writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], global_step=global_step, walltime=None)
            

            #print("epoch={}/{},{}/{}of train, loss={}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
            logger.warning("epoch = {}/{}, {}/{} of train, loss = {}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
        scheduler.step()
        writer.add_scalar("loss/epoch", np.mean(train_epoch_loss), global_step=epoch, walltime=None)

        save_ckpt(savepath,epoch, model,optimizer,scheduler)
    writer.close()