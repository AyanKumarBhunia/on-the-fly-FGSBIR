# Sketch Less for More: On-the-Fly Fine-Grained Sketch Based Image Retrieval, CVPR 2020.
**Ayan Kumar Bhunia**, Yongxin Yang, Timothy M. Hospedales, Tao Xiang, Yi-Zhe Song, “Sketch Less for More: On-the-Fly Fine-Grained Sketch Based Image Retrieval”, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020. 

## Abstract
Fine-grained sketch-based image retrieval (FG-SBIR) addresses the problem of retrieving a particular photo instance given a user's query sketch. Its widespread applicability is however hindered by the fact that drawing a sketch takes time, and most people struggle to draw a complete and faithful sketch. In this paper, we reformulate the conventional FG-SBIR framework to tackle these challenges, with the ultimate goal of retrieving the target photo with the least number of strokes possible. We further propose an on-the-fly design that starts retrieving as soon as the user starts drawing. To accomplish this, we devise a reinforcement learning based cross-modal retrieval framework that directly optimizes rank of the ground-truth photo over a complete sketch drawing episode. Additionally, we introduce a novel reward scheme that circumvents the problems related to irrelevant sketch strokes, and thus provides us with a more consistent rank list during the retrieval. We achieve superior early-retrieval efficiency over state-of-the-art methods and alternative baselines on two publicly available fine-grained sketch retrieval datasets.

## Framework

![Framework](Framework.jpg)
(a) A conventional FG-SBIR framework trained using triplet loss. (b) Our proposed reinforcement learning based framework that takes into account a complete sketch rendering episode. Key locks signifies particular weights are fixed during RL training.

## Illustrative Example

![Example](example.jpg)
Illustration of proposed on-the-fly framework's efficacy over a baseline FG-SBIR method trained with completed sketches only.  For this particular example, our method needs only 30% of the complete sketch to include the true match in the top-10 rank list, compared to 80% for the baseline. Top-5 photo images retrieved by either framework are shown here, in progressive sketch-rendering steps of 10%. The number at the bottom denotes the paired (true match) photo's rank at every stage.

~A basic Policy-Gradient (not PPO which is our final version) based sketch branch finetuning (via Reinforcement Learning) code is uploaded at this time to help any reader understand how our framework works. 

## Citation

If you find this article useful in your research, please consider citing:
```
@inproceedings{bhunia2020sketch,
  title={Sketch Less for More: On-the-Fly Fine-Grained Sketch Based Image Retrieval},
  author={Ayan Kumar Bhunia and Yongxin Yang and Timothy M. Hospedales and Tao Xiang and Yi-Zhe Song},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
