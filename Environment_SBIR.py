
from dataset_chairv2 import *
import time
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from Net_Basic_V1 import Net_Basic
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from RL_Networks import backbone_network, Policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi = torch.FloatTensor([math.pi]).to(device)
rgb_dir = './../chairV2'



class Environment():
    def __init__(self):

        '''
        #super(check_Reward, self).__init__()
        model_fixed = Net_Basic()
        model_fixed.to(device)
        model_fixed.load_state_dict(torch.load('./../chairV2/model_Best_Supervised.pth'))
        model_fixed.eval()

        backbone_Network = backbone_network()
        backbone_Network.load_state_dict(torch.load('./../chairV2/model_Best_Supervised.pth'), strict=False)
        backbone_Network.fix_backbone()
        backbone_Network.to(device)
        backbone_Network.eval()

        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        opt.coordinate = 'ChairV2_Coordinate'
        opt.roor_dir = rgb_dir
        opt.mode = 'Train'
        opt.Train = True
        opt.shuffle = False
        opt.nThreads = 1
        opt.batch_size = 1

        dataset_sketchy_train = CreateDataset_Sketchy(opt, on_Fly=True)
        dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                                   num_workers=int(opt.nThreads))

        self.Image_Array_Train = torch.FloatTensor().to(device)
        self.Sketch_Array_Train = []
        self.Image_Name_Train = []
        self.Sketch_Name_Train = []

        for i_batch, sanpled_batch in enumerate(dataloader_sketchy_train):
            sketch_feature_ALL = torch.FloatTensor().to(device)
            for data_sketch in sanpled_batch['sketch_img']:
                sketch_feature = backbone_Network(data_sketch.to(device))
                sketch_feature_ALL = torch.cat((sketch_feature_ALL, sketch_feature.detach()))
            self.Sketch_Name_Train.extend(sanpled_batch['sketch_path'])
            self.Sketch_Array_Train.append(sketch_feature_ALL.cpu())

            if sanpled_batch['positive_path'][0] not in self.Image_Name_Train:
                rgb_feature = model_fixed(sanpled_batch['positive_img'].to(device))
                self.Image_Array_Train = torch.cat((self.Image_Array_Train, rgb_feature.detach()))
                self.Image_Name_Train.extend(sanpled_batch['positive_path'])

            print('Train Image Feature Loading:', i_batch)

        parser = argparse.ArgumentParser()
        test_opt = parser.parse_args()
        test_opt.coordinate = 'ChairV2_Coordinate'
        test_opt.roor_dir = rgb_dir
        test_opt.mode = 'Test'
        test_opt.Train = False
        test_opt.shuffle = False
        test_opt.nThreads = 1
        test_opt.batch_size = 1

        dataset_sketchy_test = CreateDataset_Sketchy(test_opt, on_Fly=True)
        dataloader_sketchy_test = data.DataLoader(dataset_sketchy_test, batch_size=test_opt.batch_size,
                                                  shuffle=test_opt.shuffle,
                                                  num_workers=int(test_opt.nThreads))

        self.Image_Array_Test = torch.FloatTensor().to(device)
        self.Sketch_Array_Test = []
        self.Image_Name_Test = []
        self.Sketch_Name_Test = []

        for i_batch, sanpled_batch in enumerate(dataloader_sketchy_test):

            sketch_feature_ALL = torch.FloatTensor().to(device)
            for data_sketch in sanpled_batch['sketch_img']:
                sketch_feature = backbone_Network(data_sketch.to(device))
                sketch_feature_ALL = torch.cat((sketch_feature_ALL, sketch_feature.detach()))
            self.Sketch_Name_Test.extend(sanpled_batch['sketch_path'])
            self.Sketch_Array_Test.append(sketch_feature_ALL.cpu())


            if sanpled_batch['positive_path'][0] not in self.Image_Name_Test:
                rgb_feature = model_fixed(sanpled_batch['positive_img'].to(device))
                self.Image_Array_Test = torch.cat((self.Image_Array_Test, rgb_feature.detach()))
                self.Image_Name_Test.extend(sanpled_batch['positive_path'])

            print('Test Image Feature Loading:', i_batch)

        with open("Train.pickle", "wb") as f:
            pickle.dump((self.Image_Array_Train, self.Sketch_Array_Train, self.Image_Name_Train, self.Sketch_Name_Train), f)

        with open("Test.pickle", "wb") as f:
            pickle.dump((self.Image_Array_Test, self.Sketch_Array_Test, self.Image_Name_Test, self.Sketch_Name_Test), f)

'''
        with open("Train.pickle", "rb") as f:
            self.Image_Array_Train, self.Sketch_Array_Train, self.Image_Name_Train, self.Sketch_Name_Train = pickle.load(f)
        with open("Test.pickle", "rb") as f:
            self.Image_Array_Test, self.Sketch_Array_Test, self.Image_Name_Test, self.Sketch_Name_Test = pickle.load(f)

        self.policy_network = Policy().to(device)




    def get_reward(self, action, sketch_name):
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        position_query = self.Image_Name_Train.index(sketch_query_name)
        target_distance = F.pairwise_distance(F.normalize(action),
                                              self.Image_Array_Train[position_query])
        distance = F.pairwise_distance(F.normalize(action), self.Image_Array_Train)
        rank = distance.le(target_distance).sum()

        if rank.item() == 0:
            reward = 1.
        else:
            reward = 1. / rank.item()
        return reward


    def evaluate_RL(self, step_stddev):
        self.policy_network.eval()
        num_of_Sketch_Step = len(self.Sketch_Array_Test[0])
        avererage_area = []
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            #print('evaluate_RL running', i_batch)
            sketch_name = self.Sketch_Name_Test[i_batch]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)
            mean_rank = []

            for i_sketch in range(sanpled_batch.shape[0]):
                _, sketch_feature, _, _  = self.policy_network.select_action(sanpled_batch[i_sketch].unsqueeze(0).to(device))
                target_distance = F.pairwise_distance(F.normalize(sketch_feature), self.Image_Array_Test[position_query].unsqueeze(0))
                distance = F.pairwise_distance(F.normalize(sketch_feature), self.Image_Array_Test)
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))

        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        meanIOU = np.mean(avererage_area)

        return top1_accuracy, meanIOU


    def calculate_loss(self, log_probs, rewards, entropies):
        loss = 0
        gamma = 0.9
        for i in reversed(range(len(rewards))):
            #R = gamma ** (len(rewards) - i -1) * rewards[i]
            R =  rewards[i] # Flat Reward
            loss = loss - log_probs[i] * R #- 0.0001 * entropies[i]
        loss = loss / len(rewards)
        return loss









