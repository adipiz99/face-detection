#!/bin/bash
#source ~/virtualenv/pt1.4_tf2.1/bin/activate
python /content/AgeTransGAN/test/main.py --img_size 1024 --group 10 --batch_size 16 --snapshot '/content/drive/MyDrive/ffhq_10group_910k.pt' --file /content/AgeTransGAN/test/img/1.jpg
# deactivate
