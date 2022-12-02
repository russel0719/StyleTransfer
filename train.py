# 0. Package Installation
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import time
from utils.imageloader import load_images
from utils.dataloader import load_data
from utils.normalize import batch_normalize
from utils.gram_matrix import gram_matrix
from model.VGG16 import VGG16
from model.TransformerNet import TransformerNet

def train(batch_size, num_epoch, learning_rate, content_weight, style_weight, log_interval, ckpt_dir):
    # 1. Hyperparameter Setting
    # batch_size         10
    # num_epoch          5
    # learning_rate      1e-4
    # content_weight     1e5
    # style_weight       1e10
    # log_interval       50
    # ckpt_dir           './checkpoints'
    
    # 2. Style Images and Train Data Loading
    style_data = load_images('./data/', 'summer', batch_size)
    print(style_data.shape)

    train_dataset, train_dataloader = load_data('./data/', batch_size)
    print(train_dataset[0][0].shape)
    
    # 3. Style Images Transform with Gram matrix
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    features_style = vgg(batch_normalize(style_data.to(device)))
    gram_style = [gram_matrix(y) for y in features_style]
    
    # 4. TransformerNet training with train data
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    for epoch in range(num_epoch):
        model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_dataloader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = batch_normalize(y)
            x = batch_normalize(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * loss_function(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += loss_function(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                msg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(msg)
                
        # 4.1 Save Model
        transformer.eval().cpu()
        ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_id + 1) + ".pth"
        print(str(epoch), "th checkpoint is saved!")
        ckpt_model_path = os.path.join(ckpt_dir, ckpt_model_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, ckpt_model_path)

        transformer.to(device).train()