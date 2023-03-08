import argparse
import torch
import torchvision
from torch.utils.data import Dataset
from sphere_net import sphere4
from train import Trainer
import numpy as np
from PIL import Image
from LFWDataset import LFWDataset, LFWDataset_test


def parse_args():
    parser = argparse.ArgumentParser(description='Sphereface')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--bs', type=int, default=128, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # dataset
    train_dataset = LFWDataset(dataset_path='data/lfw', text_path='data/pairsDevTrain.txt')
    test_dataset = LFWDataset_test(dataset_path='data/lfw', text_path='data/pairsDevTest.txt')

    # model
    model = sphere4(classnum=train_dataset.name_set_length)

    # training data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    # test data
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    # trainer
    trainer = Trainer(model=model)

    # # model training
    # trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # # model evaluation
    trainer.eval(test_loader=test_loader)

    return


if __name__ == "__main__":
    main()
