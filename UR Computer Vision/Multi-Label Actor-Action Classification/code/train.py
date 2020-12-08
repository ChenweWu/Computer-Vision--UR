from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import *
import time
from utils.eval_metrics import Precision, Recall, F1
from tqdm import tqdm


def calculate_weigths_labels(dataloader, num_classes):
    z = np.zeros((num_classes,))
    tqdm_batch = tqdm(dataloader
    for sample in tqdm_batch:
        _, y = sample
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    return ret

def validate(model, criterion, ars):
    test_dataset = a2d_dataset.A2DDataset(val_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    total_loss = 0
    X = np.zeros((data_loader.__len__(), args.num_cls))
    Y = np.zeros((data_loader.__len__(), args.num_cls))

    model.eval()

    tbar = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, data in enumerate(tbar):
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            output = torch.sigmoid(model(images)).cpu().detach().numpy()
            target = labels.cpu().detach().numpy()
            X[batch_idx, :] = output
            Y[batch_idx, :] = target
        
    P = Precision(X, Y)
    R = Recall(X, Y)
    F = F1(X, Y)

    return total_loss / data_loader.__len__(), P, R, F



# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8) # you can make changes

    weight = calculate_weigths_labels(data_loader, 43)
    weight = torch.from_numpy(weight.astype(np.float32)).to(device)

    model = net(args).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


    model_name = args.model_name
    best_valid_loss = -1
    best_model = None

    if os.path.isfile(os.path.join(args.model_path, args.model_name + '.ckpt')):
        print("load model....")
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name + '.ckpt')))

    
    

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        model.train()
        train_loss = 0

        tbar = tqdm(data_loader)



        for i, data in enumerate(tbar):

            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            # print(torch.max(labels, 1)[1])

            # Forward, backward and optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log info
            if i % args.log_step == 0:
                tbar.set_description('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

    
            if (i + 1) % args.save_step == 0 and best_model is not None:
                torch.save(best_model.state_dict(), os.path.join(
                    args.model_path, model_name + '.ckpt'))

        valid_loss, P, R, F = validate(model, criterion, args)
        if P > best_valid_loss:
            best_valid_loss = P
            best_model = model

        print('Epoch: {} \t | train_loss: {} \t | validation_loss: {} \t | Precision: {} \t | Recall: {} \t | F1: {} \t'.format(epoch, train_loss / data_loader.__len__(), valid_loss, P, R, F))

        t2 = time.time()

    print("Finished, saving the best model...")
    torch.save(best_model.state_dict(), os.path.join(
                    args.model_path, model_name + '.ckpt'))
    print("Complete!")
    return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=40, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--model_name', type=str, default="net_final")
    args = parser.parse_args()
    # print(args)

    model = train(args)
