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
from matchnet.code.eval_form2fit import validation_correspondence,COORD_NAMES

import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import logging

EXP_NAME="bear_3_32"
SEED=666
tb_path = './tb_log_corres_' + EXP_NAME
if not os.path.exists(tb_path):
        os.makedirs(tb_path)
writer = SummaryWriter(tb_path)
logger= logging.getLogger(__file__)


def save_ckpt(savepath,net_type,epoch,model,optimizer,scheduler,metric_value):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
        }
    if (epoch + 1) % 50 == 0:
        savedpath = os.path.join(savepath,net_type+'_epoch' + str(epoch) + '.pth')
        torch.save(checkpoint, savedpath)
    if metric_value >= 0.6:
        savedpath = os.path.join(savepath,net_type+'_epoch' + str(epoch) + str(metric_value) + '.pth')
        torch.save(checkpoint, savedpath)
    savedpath = os.path.join(savepath,net_type +'_last_epoch.pth')
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
    parser.add_argument("--epochs", type=int, default=500, help="The number of training epochs.")
    parser.add_argument("--augment","-a", action='store_true', help="(bool) Whether to apply data augmentation.",default=True)
    parser.add_argument("--background_subtract", type=tuple, default=None, help="apply mask.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--imgsize", type=list, default=[848,480], help="size of final image.")
    parser.add_argument("--root", type=str, default="", help="the path of dataset")
    parser.add_argument("--savepath", type=str, default="matchnet/code/ml/savedmodel/"+EXP_NAME, help="the path of saved models")
    parser.add_argument("--resume","-r",  action='store_true', help="whether to resume",default=False)
    parser.add_argument("--checkpoint","-c",  type=str, default="matchnet/code/ml/savedmodel/mix0128_5/corrs_epoch99.pth", help="the path of resume models")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = opt.batchsize
    epochs = opt.epochs
    dataset_name = "../datasets_bear"
    savepath = opt.savepath
    background_subtract = opt.background_subtract
    use_color = True
    num_channels = 4
    sample_ratio = opt.sample_ratio
    
    set_seed(True)

    train_loader = get_corr_loader(dataset_name, 
                                        dtype="train", 
                                        batch_size=batch_size, 
                                        use_color = use_color,
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=True,
                                        num_workers=8,
                                        background_subtract=background_subtract)

    valid_loader = get_corr_loader(dataset_name, 
                                dtype="valid", 
                                batch_size=1, 
                                use_color = use_color,
                                num_channels=num_channels, 
                                sample_ratio=1, 
                                augment=False,
                                shuffle=False,
                                background_subtract=None,
                                num_workers = 8)

    model = CorrespondenceNet(num_channels=num_channels, num_descriptor=32, num_rotations=20).to(device)
    criterion = losses.CorrespondenceLoss(sample_ratio=sample_ratio, device=device, margin=8, num_rotations=20, hard_negative=True)
    # optimizer = torch.optim.Adam(model.parameters(),lr=5e-2) # 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=[0.9,0.999],weight_decay=3e-6) # 1e-3
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, last_epoch=-1)
    scheduler = StepLR(optimizer, step_size=150, gamma=0.1) # gamma=0.1
    start_epoch = -1
    if opt.resume:
        state_dict = torch.load(opt.checkpoint, map_location=device)
        model.load_state_dict(state_dict["model"])
        start_epoch = state_dict["epoch"]
        optimizer.load_state_dict(state_dict["optimizer"])
        # for i in range(start_epoch+1):
        #     scheduler.step()
        # state_dict["scheduler"]["step_size"] = 200
        # state_dict["scheduler"]["gamma"] = 0.5
        scheduler.load_state_dict(state_dict["scheduler"])
    # valid_loss = []
    # train_epochs_loss = []
    # valid_epochs_loss = []
    logger.warning("train from %s epoch, ckpt = %s",start_epoch, opt.checkpoint)
    one_epoch_step = len(train_loader)
    for epoch in tqdm(range(start_epoch +1, epochs)):
        logger.warning("training...")
        model.train()
        train_epoch_loss = []

        for i, (imgs, labels, kit_centers, obj_centers) in enumerate(train_loader):

            imgs = imgs.to(device)
            cuda_labels = []
            for j in range(len(labels)):
                cuda_labels.append(labels[j].to(device))

            out_s, out_t = model(imgs,kit_centers[0][0], kit_centers[0][1])
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
        
        # each epoch
        logger.warning("validating...")
        writer.add_scalar("loss/epoch", np.mean(train_epoch_loss), global_step=epoch, walltime=None)

        train_pred_dict = validation_correspondence(train_loader, model, device, 16, 10)
        writer.add_scalar("train_metric/rot_ap_uniform", train_pred_dict["ap"][COORD_NAMES[0]], global_step=epoch, walltime=None)
        writer.add_scalar("train_metric/rot_acc_uniform", train_pred_dict["acc"][COORD_NAMES[0]], global_step=epoch, walltime=None)
        writer.add_scalar("train_metric/rot_ap_ccircle", train_pred_dict["ap"][COORD_NAMES[1]], global_step=epoch, walltime=None)
        writer.add_scalar("train_metric/rot_acc_ccircle", train_pred_dict["acc"][COORD_NAMES[1]], global_step=epoch, walltime=None)


        valid_pred_dict = validation_correspondence(valid_loader, model, device, 16)
        writer.add_scalar("valid_metric/rot_ap_uniform", valid_pred_dict["ap"][COORD_NAMES[0]], global_step=epoch, walltime=None)
        writer.add_scalar("valid_metric/rot_acc_uniform", valid_pred_dict["acc"][COORD_NAMES[0]], global_step=epoch, walltime=None)
        writer.add_scalar("valid_metric/rot_ap_ccircle", valid_pred_dict["ap"][COORD_NAMES[1]], global_step=epoch, walltime=None)
        writer.add_scalar("valid_metric/rot_acc_ccircle", valid_pred_dict["acc"][COORD_NAMES[1]], global_step=epoch, walltime=None)

        save_ckpt(savepath,"corrs",epoch, model,optimizer,scheduler, train_pred_dict["acc"][COORD_NAMES[1]])
        # save_ckpt(savepath,"corrs",epoch, model,optimizer,None)
        scheduler.step()
    writer.close()
