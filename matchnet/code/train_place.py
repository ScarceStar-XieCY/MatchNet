import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from matchnet.code.ml.dataloader.placement import get_placement_loader
from matchnet.code.ml.models.placement import PlacementNet
from matchnet.code.ml import losses
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from matchnet.code.train_correspondence import SEED, save_ckpt,set_seed
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import logging

tb_path = './tb_log_place'
if not os.path.exists(tb_path):
        os.makedirs(tb_path)
writer = SummaryWriter(tb_path)
logger= logging.getLogger(__file__)


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
    parser.add_argument("--savepath", type=str, default="matchnet/code/ml/savedmodel/place/", help="the path of saved models")
    parser.add_argument("--resume","-r",  action='store_true', help="whether to resume")
    parser.add_argument("--checkpoint","-c",  type=str, default="matchnet/code/ml/savedmodel/place_epoch85.pth", help="the path of resume models")
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
    radius = 3
    

    print("--------------start preparing data--------------")
    

    train_loader = get_placement_loader(kit_name, 
                                        dtype="test", 
                                        batch_size=batch_size, 
                                        use_color = use_color,
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=True,
                                        radius=radius,
                                        background_subtract=background_subtract)

    model = PlacementNet(num_channels=num_channels).to(device)
    criterion = losses.PlacementLoss(sample_ratio=sample_ratio, device=device,mean=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    start_epoch = -1
    if opt.resume:
        state_dict = torch.load(opt.checkpoint, map_location=device)
        model.load_state_dict(state_dict["model"])
        start_epoch = state_dict["epoch"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
    
    # train_epochs_loss = []
    # valid_epochs_loss = []

    one_epoch_step = len(train_loader)
    for epoch in tqdm(range(start_epoch +1, epochs)):
        model.train()
        train_epoch_loss = []

        for i, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            cuda_labels = []
            for j in range(len(labels)):
                cuda_labels.append(labels[j].to(device))

            output = model(imgs)
            optimizer.zero_grad()
            bi_cross_loss, dice_loss = criterion(output, cuda_labels,add_dice_loss=True)
            loss = bi_cross_loss + (5 * dice_loss)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())

            global_step = epoch * one_epoch_step + i
            writer.add_scalar("loss/step", loss.item(), global_step=global_step, walltime=None)
            writer.add_scalar("loss/bi_cross_loss", bi_cross_loss.item(), global_step=global_step, walltime=None)
            writer.add_scalar("loss/dice_loss", dice_loss.item(), global_step=global_step, walltime=None)
            writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], global_step=global_step, walltime=None)

            
            #print("epoch={}/{},{}/{}of train, loss={}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
            logger.warning("epoch = {}/{}, {}/{} of train, loss = {}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
        scheduler.step()
        writer.add_scalar("loss/epoch", np.mean(train_epoch_loss), global_step=epoch, walltime=None)

        save_ckpt(savepath,"place",epoch, model,optimizer,scheduler)
    writer.close()