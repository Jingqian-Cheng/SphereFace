import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from sphere_net import AngleLoss, sphere4
import numpy as np

from utils import AverageMeter


class Trainer:

    def __init__(self, model):
        self._model = model.to('cuda')

    def train(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training """

        optimizer = optim.Adam(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        criterion = AngleLoss().to('cuda') # use GPU
        # self._model.cuda()
        
        print("Start training...")
        for i in range(epochs):
            loss_track.reset()
            self._model.train()
            tik = time.time()
            for image, target in train_loader:
                image, target = image.to('cuda'), target.to('cuda')
                optimizer.zero_grad()
                output = self._model(image)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                loss_track.update(loss.item(), n=len(image))

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))
            # self.eval(test_loader) # evaluate the model after each loop
            
        print("Training completed, saving model to %s" % save_dir)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "sphere.pth"))
        return

    # calculate the similarity between two images and compare with the label. Looping through different thresholds to determine the best threshold with highest accuracy
    def eval(self, test_loader: DataLoader) -> float:
        self.load_model('./save/sphere.pth')
        self._model = self._model.to('cuda')
        self._model.eval()
        accuracies = []
        print("Evaluation starts")
        with torch.no_grad():
            for threshold in np.arange(0.0, 1.0, 0.1):
                print("Current threshold: " + str(threshold))
                correct = 0
                total = 0
                for image1, image2, target in test_loader:
                    image1, image2, target = image1.to('cuda'), image2.to('cuda'), target.to('cuda')
                    image1_feature = self._model(image1, return_feature=True) # get feature instead of predicted class
                    image2_feature = self._model(image2, return_feature=True)
                    cos_similarity = torch.cosine_similarity(image1_feature, image2_feature) # calculate the similarity of two images
                    total += len(image1) 
                    predict_as_same = (abs(cos_similarity) > threshold)
                    for i in range(0, len(predict_as_same)):
                        if predict_as_same[i].item() == target[i].item(): # when the predicted result is equal to the label
                            correct += 1
                accuracy = correct / total
                accuracies.append((threshold, accuracy))
                print("Current accuracy: " + str(accuracy))
            
        sorted_accuracies = sorted(accuracies, key=lambda x: -x[1]) # sort the tuple list based on accuracy in decreasing order
        best_threshold = sorted_accuracies[0][0]
        best_accuracy = sorted_accuracies[0][1]
        print('Best accuracy is %.5f, best threshold is: %.3f' %(best_accuracy, best_threshold))


    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        self.load_model('./save/sphere.pth')
        self._model.eval()
        with torch.no_grad():
            output = self._model(sample)
            _, predicted = torch.max(output, 1)
        print("Infer result: "+str(predicted.item()))
        return predicted.item()

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        model = sphere4()
        model.load_state_dict(torch.load(path))
        self._model = model
        return

