from dataset_chairv2 import *
from Environment_SBIR import Environment
import torch.optim as optim
import torch.utils.data as data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn.utils as utils
import numpy as np
GAMMA = 0.9


def main_train(opt):

    dataset_sketchy_train = CreateDataset_Sketchy(opt, on_Fly=True)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads))
    top1_buffer = 0
    mean_IOU_buffer = 0
    SBIR_Environment = Environment()
    loss_buffer = []


    optimizer = optim.Adam(SBIR_Environment.policy_network.parameters(), lr=opt.lr)



    Top1_Song = [0]
    meanIOU_Song = []

    step_stddev = 1
    SBIR_Environment.policy_network.train()

    for epoch in range(opt.niter):

        if mean_IOU_buffer > 0.25 and optimizer.param_groups[0]['lr']== 0.001 :
            optimizer.param_groups[0]['lr'] = 0.0001
            print('Reduce Learning Rate')

        print('LR value : {}'.format(optimizer.param_groups[0]['lr']))
        for i, sanpled_batch in enumerate(SBIR_Environment.Sketch_Array_Train):

            entropies = []
            log_probs = []
            rewards = []

            for i_sketch in range(sanpled_batch.shape[0]):
                action_mean, sketch_anchor_embedding, log_prob, entropy = \
                    SBIR_Environment.policy_network.select_action(sanpled_batch[i_sketch].unsqueeze(0).to(device))
                reward = SBIR_Environment.get_reward(sketch_anchor_embedding, SBIR_Environment.Sketch_Name_Train[i])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)


            loss_single = SBIR_Environment.calculate_loss(log_probs, rewards, entropies)
            loss_buffer.append(loss_single)

            step_stddev += 1

            print('Epoch: {}, Iteration: {}, Loss: {}, REWARD: {}, Top1_Accuracy: {}, '
                  'mean_IOU: {}, step: {}'.format(epoch, i, loss_single.item(),
                                                         np.sum(rewards), top1_buffer, mean_IOU_buffer, step_stddev))

            if (i + 1) % 16 == 0: #[Update after every 16 images]
                optimizer.zero_grad()
                policy_loss = torch.stack(loss_buffer).mean()
                policy_loss.backward()
                utils.clip_grad_norm_(SBIR_Environment.policy_network.parameters(), 40)
                optimizer.step()
                loss_buffer = []

            if (i + 1) % opt.save_iter == 0:
                with torch.no_grad():
                    top1, mean_IOU = SBIR_Environment.evaluate_RL(step_stddev)
                    SBIR_Environment.policy_network.train()
                    print(top1, mean_IOU)
                    Top1_Song.append(top1)
                    meanIOU_Song.append(mean_IOU)

                print('Epoch: {}, Iteration: {}, Top1_Accuracy: {}, mean_IOU: {}'.format(epoch, i, top1, mean_IOU))

                if mean_IOU > mean_IOU_buffer:
                    if torch.cuda.device_count() > 1:
                        torch.save(SBIR_Environment.policy_network.module.state_dict(), 'model_BestRL.pth')
                    else:
                        torch.save(SBIR_Environment.policy_network.state_dict(), 'model_BestRL.pth')
                    mean_IOU_buffer = mean_IOU
                    if top1 > top1_buffer:
                        top1_buffer = top1
                    print(Top1_Song, meanIOU_Song)

                    print('Model Updated')


    print('ayan Kumar Bhunia')

    print(Top1_Song, meanIOU_Song)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = 'ChairV2_Coordinate'
    opt.roor_dir = './../chairV2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.batchsize = 1 #has to be one
    opt.nThreads = 4
    opt.lr = 0.001
    opt.niter = 2000
    opt.save_iter = 400
    opt.load_earlier = False
    main_train(opt)




