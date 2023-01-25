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
sys.path.append(os.getcwd())
from matchnet.code.ml.dataloader import get_corr_loader
from matchnet.code.ml.models.correspondence import CorrespondenceNet
from matchnet.code.ml import losses

import warnings
warnings.filterwarnings("ignore")


# python form2fit/code/train_correspondence.py 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Form2Fit suction Module")
    parser.add_argument("--batchsize", type=int, default=1, help="The batchsize of the dataset.")
    parser.add_argument("--sample_ratio", type=int, default=3, help="The ratio of negative to positive labels.")
    parser.add_argument("--epochs", type=int, default=50, help="The number of training epochs.")
    parser.add_argument("--augment", action='store_true', help="(bool) Whether to apply data augmentation.")
    parser.add_argument("--background_subtract", type=tuple, default=None, help="apply mask.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--imgsize", type=list, default=[848,480], help="size of final image.")
    parser.add_argument("--root", type=str, default="", help="the path of dataset")
    parser.add_argument("--savepath", type=str, default="matchnet/code/ml/savedmodel/", help="the path of saved models")
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
    sample_ratio = 5
    radius = 1
    

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
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    print("----------------start training-----------------")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []

        for i, (imgs, labels, centers) in enumerate(train_loader):

            imgs = imgs.to(device)
            for j in range(len(labels)):
                labels[j] = labels[j].to(device)

            out_s, out_t = model(imgs,centers[0][0], centers[0][1])
            optimizer.zero_grad()
            loss = criterion(out_s, out_t, labels) #TODO:check output shape and split
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            #print("epoch={}/{},{}/{}of train, loss={}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
            print("epoch = {}/{}, {}/{} of train, loss = {}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        if (epoch % 10 == 0 and epoch != 0) or (epoch < 155 and epoch > 145):                            # 选择输出的epoch
            print("---------saving model for epoch {}----------".format(epoch))
            savedpath = savepath + 'coor_epoch' + str(epoch) + '.pth'
            torch.save(model.state_dict(), savedpath)

        if epoch + 1 == epochs:
            print("---------saving model for last epoch ----------")
            finalsavedpath = savepath + 'corr_final' + '.pth'
            torch.save(model.state_dict(), finalsavedpath)