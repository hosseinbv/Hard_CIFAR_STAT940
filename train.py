# Try1: best results: VGG19 + Ls + BS=64 + epoch 400 + lr=0.0001.
# Try2: acc=0.801 semi-epoch ADAMW +SGD , VGG19, bs=128, lr-0.0001, epoch 630, one GPU
# Try3: My Best Try: wideResnet 28_10, ASAM, MiddleDataset, bs=128, lr=0.0001, epoch 193, rho=0.5, train_loss/(idx+1) < 0.51
# -------------------------------------------------------------------------------------------------------------------
# By Hossein Rajabzadeh (id: 20811573), Email: hossein.rajabzadeh@uwaterloo.ca
# Course name: STAT940 (Prof. A. Ghodsi)
# Feb 2023
#====================================================================================================================
# Note: In this work, I used the idea of sharpness aware minimizer (SAM), which costs double in computing the gradient and space complexity but worked well in this data challenge.
# SAM uses the first gradient to search a locality of weights and takes a direction that minimizes the worst case in that locality.
# Please see this ref for more details: https://arxiv.org/abs/2102.11600
# I also used the ASAM code in this github for ASAM minimizer: https://github.com/SamsungLabs/ASAM.git
# ===================================================================================================================
#  ==========================================My Best Acc on Kaggle leaderboard is 0.8266
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from timm.loss import LabelSmoothingCrossEntropy

from dataset import TrainDataset, ValDataset, TestDataset, Middle_Dataset
from models import *
import os
import argparse
import random
import glob
import numpy as np
import csv
from operator import itemgetter

os.environ["CUDA_VISIBLE_DEVICES"] ="7" # change it to your available gpu indices

# FOR validation
def val(model, val_loader, device, criterion):
    model.eval()
    total_val_loss = 0
    val_acc = 0
    # walk through all mini-batches
    # cnt = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            total_val_loss += val_loss.item()
            _, predicted = outputs.max(1)
            val_acc += predicted.eq(targets).sum().item()/len(targets)
            # cnt += 1
    total_val_loss = total_val_loss / (idx+1)
    val_acc = val_acc / (idx+1)
    return val_loss, val_acc


# this function make a prediction for TEST data and save the results into a CSV file.
def test(model, test_loader, device):
    model.eval()
    # walk through all mini-batches
    preds_idxs = []
    preds = []
    sel_test_data = []
    sel_test_labels =[]
    F = nn.Softmax(dim=1)
    with torch.no_grad():
        for idx, (inputs, IDS) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds.append(predicted.cpu().detach().numpy())
            preds_idxs.append(IDS)
            scores, indices = F(outputs).max(dim=1)
            ## SemiSupervised apprach: if the maximum value in softmax(logits) for a test data is higher than a threshold (e.g. 0.9), take this test sample for training and use the 
            ## the predicted label as its true label.
            for id in np.where((scores > 0.9).cpu().detach().numpy().astype('int32')==1)[0]:  # confidence threshold is 0.8 for adding a test sample into training...
                sel_test_data.append(IDS[id])
                sel_test_labels.append(indices[id].item())
    
    L = []
    for P, I in zip(preds, preds_idxs):
        for num, file in enumerate(I):
            img_id = int(os.path.split(file)[-1].split('.')[0])
            img_label = P[num].item()
            L.append([img_id, img_label])
    L = sorted(L, key=itemgetter(0))
    # write test predictions into a CSV file 
    with open('output.csv','w+') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(L)

    return preds, preds_idxs, sel_test_data, sel_test_labels

## one can use this function for saving the model.
def save_checkpoint(model, acc, epoch):
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')


# Training
def train(args, model, train_loader, val_loader, test_loader, device, optimizer, optimizer2, criterion, scheduler, scheduler2, real_testloader=None, train_transform=None,
            minimizer_1=None, minimizer_2=None):
    #iterate for each epoch
    prev_loss = 1000
    for epoch in range(args.max_epoch):
        sel_test_data, sel_test_labels = None, None
        model.train()
        train_loss = 0
        acc = 0
        # walk through all mini-batches
        ## We use adaptive SAM as the default optimizer, one can also use ADAMW as well
        for idx, (inputs, targets) in enumerate(train_loader): 
            inputs, targets = inputs.to(device), targets.to(device)
            if args.minimizer_1 == 'ASAM':
                # Ascent Step
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.mean().backward()
                minimizer_1.ascent_step()

                # Descent Step
                criterion(model(inputs), targets).mean().backward()
                minimizer_1.descent_step()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # optimizer steps after each step
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            acc += predicted.eq(targets).sum().item() / len(targets)

        
        print('epoch: ', epoch, 'Train Loss: %.4f | Train Acc: %.4f'% (train_loss/(idx+1), 100.*acc/(idx+1)))
        if train_loss/(idx+1) <= prev_loss and train_loss/(idx+1) < 0.51: # If train loss in this epoch < 0.51, then make test prediction and 
            # start the semi-supervised approach explained below. I am not sure about the exact value of this hyperparameter because I had many experiments and changed it a lot.
            # If you found it important, please email me and I do the experiments again and report you the exact value.
            prev_loss = train_loss/(idx+1)
            # save_checkpoint(model, 100.*acc/(idx+1), epoch)
            print('testing for loss=', prev_loss)
            _,_, sel_test_data, sel_test_labels = test(model, test_loader, device)
        
            if sel_test_data is not None: ## Start the semisupervised apprach if there is a test predction with a high logit value (high prediction confidence)

                # make an additional dataset only for semisupervised step
                additional_dataSet = Middle_Dataset(transform=train_transform, sel_test_data=sel_test_data, sel_test_labels=sel_test_labels)
                # build an additional loaders only for semisupervised step
                additional_loader = torch.utils.data.DataLoader(
                    additional_dataSet, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=False)

                for idx2, (inputs, targets) in enumerate(additional_loader): ## This for-loop is for the semisupervised approach
                    inputs, targets = inputs.to(device), targets.to(device)
                    if args.minimizer_2 == 'ASAM':  
                        outputs = model(inputs)
                        loss2 = criterion(outputs, targets)
                        loss2.mean().backward()
                        minimizer_2.ascent_step()

                        # Descent Step
                        criterion(model(inputs), targets).mean().backward()
                        minimizer_2.descent_step()
                    else:
                        optimizer2.zero_grad()
                        outputs = model(inputs)
                        loss2 = criterion(outputs, targets)
                        loss2.backward()
                        optimizer2.step()

                    train_loss += loss2.item()
                    _, predicted = outputs.max(1)
                    acc += predicted.eq(targets).sum().item() / len(targets)
                    
                   
                # scheduler steps after each epoch
                scheduler2.step()
                print('** Semi epoch: ', epoch, 'Train Loss: %.4f | Train Acc: %.4f'% (train_loss/(idx+1+idx2+1), 100.*acc/(idx+1+idx2+1)))
                del additional_dataSet, additional_loader
        scheduler.step()


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)


def main():
    parser = argparse.ArgumentParser(description='PyTorch stat950_DC1 Training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batches for training')
    parser.add_argument('--val_batch_size', type=int, default=4096, help='size of mini-batches for validation')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='ratio of trainSize')
    parser.add_argument('--max_epoch', type=int, default=800, help='maximum training epoch')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--do_eval', type=bool, default=False, help='perform an evaluation in each epoch of training')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int, help="patch for ViT")
    parser.add_argument('--size', type=int, default=32, help='size of img')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument("--minimizer_1", default='ASAM', type=str, help="ASAM or SAM.")
    parser.add_argument("--minimizer_2", default='ASAM', type=str, help="ASAM or SAM.")
    parser.add_argument("--rho", default=1.5, type=float, help="Rho for ASAM.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
    parser.add_argument('--droprate', default=0.1, type=float,
                    help='dropout probability (default: 0.0)')
    parser.add_argument('--train_img_dir', type=str, default='/home/hossein/projects/stat940/train/train/', help='the path containing the train imgs.')
    parser.add_argument('--test_img_dir', type=str, default='/home/hossein/projects/stat940/test/test/', help='the path containing the test imgs.')
    parser.add_argument('--train_annotations_file', type=str, default='/home/hossein/projects/stat940/train_labels.csv', help='the complete csv path of train labels.')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # set the device
    print('Device is: ', device)
    seed_everything(1) # set the seed as requested
    # Data ==> Preparing data..
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # this transform add no augmentation.
    train_simple_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # This is for VAL/TEST data
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    ## split data into train and Val, But I prefered to use all my data for training at the end.   <------------------- FOR VALIDATION
    # all_imgs_paths = sorted(glob.glob(args.train_img_dir+'*'))
    # shuffled_idxs = list(range(len(all_imgs_paths)))
    # random.shuffle(shuffled_idxs)
    # train_idxs = shuffled_idxs[:int(args.train_ratio*len(shuffled_idxs))]
    # val_idxs = shuffled_idxs[int(args.train_ratio*len(shuffled_idxs)):]

    # build datasets for train/val/test
    train_set = TrainDataset(img_dir=args.train_img_dir, annotations_file=args.train_annotations_file, transform=train_transform, simple_transform=train_simple_transform,
    train_idxs=None)
    ## uncomment the following line for Validation
    # val_set = ValDataset(img_dir=args.train_img_dir, transform=val_transform, annotations_file=args.train_annotations_file, val_idxs=val_idxs)
    test_set = TestDataset(img_dir=args.test_img_dir, transform=val_transform)
  
    # build loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    ## uncomment the following line and comment the one after for Val loader
    # val_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    val_loader = None
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    classes = ('deer', 'horse', 'car', 'truck', 'small mammal',
           'flower', 'tree', 'aquatic mammal', 'fish', 'ship')

    #==============================================================To be Removed
    # Please note that I tried all the follwing models, and one can uncomment each one of them if needed. But, wide-res-net with depth=28 and width=10 works well at the end
    # I just used one model, and didn't tried ensembels of deep networks. 

    #### MODELS
    # model = EfficientNetB0()
    # model = VGG('VGG19')
    model = WideResNet(args.layers, 10,
                            args.widen_factor, dropRate=args.droprate)
    # model = ShuffleNetV2(net_size=2.0)
    # model = ResNet152()
    # model = swin_t(window_size=args.patch,
    #             num_classes=10,
    #             downscaling_factors=(2,2,2,1))
    # model = ViT(
    #     image_size = args.size,
    #     patch_size = args.patch,
    #     num_classes = 10,
    #     dim = int(args.dimhead),
    #     depth = 6,
    #     heads = 8,
    #     mlp_dim = 512,
    #     dropout = 0.1,
    #     emb_dropout = 0.1)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    ## DEfining the objective/loss function
    # criterion = nn.CrossEntropyLoss()  # simple crossEntropy
    criterion = LabelSmoothingCrossEntropy(args.smoothing)
    
    # init optimizer SGD/ADAMW
    ## Please note that I used a semi-supervised approach for training. That is, after several epoches, I make a prediction over test data (no label data), and 
    ## accept the predicted labels for those which come with a high prediction logit scores and add them into my training data. I do this startegy for the following epoches as well.
    ## To do that, I made two optimizers/scheduler/minimizer, defining as follwos:

    optimizer2 = optim.SGD(model.parameters(), lr=args.lr * 0.1,
                        momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # init scheduler for auto-adjusting the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)
    minimizer_1 = eval(args.minimizer_1)(optimizer, model, rho=args.rho, eta=args.eta)
    minimizer_2 = eval(args.minimizer_2)(optimizer2, model, rho=args.rho, eta=args.eta)

    # call training stage
    train(args, model, train_loader, val_loader, test_loader, device, optimizer, optimizer2, criterion, scheduler,scheduler2, train_transform=train_transform,
                minimizer_1=minimizer_1, minimizer_2=minimizer_2)

    print('Finished...')
    

if __name__=="__main__":
    main()