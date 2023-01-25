import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from matchnet.code.ml.dataloader import PlacementDataset
from matchnet.code.ml.models.placement import PlacementNet
from matchnet.code.ml import losses

import warnings
warnings.filterwarnings("ignore")


# python form2fit/code/train_correspondence.py 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Form2Fit suction Module")
    parser.add_argument("--batchsize", type=int, default=1, help="The batchsize of the dataset.")
    parser.add_argument("--sample_ratio", type=int, default=3, help="The ratio of negative to positive labels.")
    parser.add_argument("--epochs", type=int, default=160, help="The number of training epochs.")
    parser.add_argument("--augment", action='store_true', help="(bool) Whether to apply data augmentation.")
    parser.add_argument("--background_subtract", type=tuple, default=(0.04, 0.047), help="apply mask.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--imgsize", type=list, default=[848,480], help="size of final image.")
    parser.add_argument("--root", type=str, default="", help="the path of dataset")
    parser.add_argument("--savepath", type=str, default="form2fit/code/ml/savedmodel/", help="the path of saved models")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = opt.batchsize
    epochs = opt.epochs
    root = ""
    savepath = opt.savepath
    testroot = "dataset/test"
    background_subtract = opt.background_subtract
    num_channels = 2
    sample_ratio = 5
    radius = 1

    print("--------------start preparing data--------------")
    

    train_loader = PlacementDataset.get_placement_loader(root, 
                                        dtype="test", 
                                        batch_size=batch_size, 
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=True,
                                        background_subtract=background_subtract)

    model = PlacementNet(num_channels=num_channels, num_descriptor=64, num_rotations=20).to(device)
    criterion = losses.CorrespondenceLoss(sample_ratio=sample_ratio, device=device)
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

        for i, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            for j in range(len(labels)):
                labels[j] = labels[j].cuda()

            output = model(imgs)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            #print("epoch={}/{},{}/{}of train, loss={}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
            if i % (len(train_loader)//2) == 0:
                print("epoch = {}/{}, {}/{} of train, loss = {}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
                train_epochs_loss.append(np.average(train_epoch_loss))

        if (epoch % 40 == 0 and epoch != 0) or (epoch < 155 and epoch > 145):                            # 选择输出的epoch
            print("---------saving model for epoch {}----------".format(epoch))
            savedpath = savepath + 'place_epoch' + str(epoch) + '.pth'
            torch.save(model.state_dict(), savedpath)

        if epoch + 1 == epochs:
            print("---------saving model for last epoch ----------")
            finalsavedpath = savepath + 'place_final' + '.pth'
            torch.save(model.state_dict(), finalsavedpath)