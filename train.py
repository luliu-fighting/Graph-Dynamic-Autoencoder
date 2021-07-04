# --coding:utf-8--
import os
import time
import torch
import torch.nn as nn
from torch import optim
from model import GAElayer, GraphStackAE, construct_graph
from argparse import ArgumentParser
from util import get_train_data

def parse_args():
    # Setting parameters

    parser = ArgumentParser(description='Implement of Fault Detection with GDAE')
    parser.add_argument('--num_train_layer_epochs', type=int, default=80, help='epochs of layer training')
    parser.add_argument('--num_train_whole_epochs', type=int, default=30, help='epochs of whole training')

    args = parser.parse_args()
    return args

def train_layers(layers_list=None, layer=None, num_epoch=None, inputs=None, lr=0.005):

    for net in layers_list:
        net.cuda()

    optimizer = optim.Adam(layers_list[layer].parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    # train
    for epoch in range(num_epoch):
        # Freeze the parameters of all layers before the current layer - layer 0 has no predecessor layers
        if layer != 0:
            for index in range(layer):
                layers_list[index].lock_grad()
                # In addition to the freeze parameters, you should also set the output return method of the freeze layer
                layers_list[index].is_training_layer = False
        g, out = inputs
            # Forward calculation for the former (layer-1) frozen layer
        if layer != 0:
            for l in range(layer):
                g, out = layers_list[l]([g, out])

        # train
        g, pred = layers_list[layer]([g, out])
        optimizer.zero_grad()
        loss = criterion(pred, out)
        loss.backward()
        optimizer.step()
        print("Epoch %d | Loss: %.4f | learning rate: %.6f" % (epoch+1, loss, optimizer.param_groups[0]['lr']))


def train_whole(net=None, num_epoch=None, inputs=None, lr=0.0001):
    print(">> start training whole model")
    if torch.cuda.is_available():
        net.cuda()

    # Unfreezing of parameters frozen due to pre-trained monolayers
    for param in net.parameters():
        param.require_grad = True

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()

    # train
    for epoch in range(num_epoch):
        g, out = inputs
        pred = net([g, out])
        optimizer.zero_grad()
        loss = criterion(pred, out)
        loss.backward()
        optimizer.step()
        print("Epoch %d | Loss: %.4f | learning rate: %.6f" % (epoch + 1, loss, optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    start = time.time()
    args = parse_args()

    feats = [52, 52, 27]

    train_data = get_train_data(unit=list(range(52)))
    train_g = construct_graph(train_data, neighbor=5)
    train_g = train_g.to("cuda:0")
    train_g.edata['h_e'] = train_g.edata['h_e'].cuda()
    train_data = torch.Tensor(train_data)
    train_data = train_data.cuda()
    num_layers = 5

    encoder_1 = GAElayer(feats[0], feats[1], SelfTraining=True)
    encoder_2 = GAElayer(feats[1], feats[2], SelfTraining=True)
    decoder_1 = GAElayer(feats[2], feats[1], SelfTraining=True)
    decoder_2 = GAElayer(feats[1], feats[0], SelfTraining=True)

    layers_list = [encoder_1, encoder_2, decoder_1, decoder_2]

    for level in range(num_layers - 1):
        print("layer %d" % (level + 1))
        train_layers(layers_list=layers_list, layer=level, num_epoch=args.num_train_layer_epochs,
                     inputs=[train_g, train_data])

    net = GraphStackAE(layers_list=layers_list)
    net.cuda()
    train_whole(net=net, num_epoch=args.num_train_whole_epochs, inputs=[train_g, train_data])

    base_path = "./params/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    torch.save(net, os.path.join(base_path, "net.pkl"))
    torch.save(net.state_dict(), os.path.join(base_path, "net_params.pkl"))
    print("time span : %.4f s" % (time.time()-start))


