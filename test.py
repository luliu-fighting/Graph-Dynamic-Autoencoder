# --coding:utf-8--
import torch
import time
from util import get_train_data, get_test_data
from model import construct_graph
import scipy.io as sio
import os
if __name__=="__main__":
    start = time.time()

    net = torch.load("./params/net.pkl", map_location="cuda:0")

    train_data = get_train_data(unit=list(range(52)))
    test_data = get_test_data(unit=list(range(52)))
    train_g = construct_graph(train_data, neighbor=5)
    train_g = train_g.to("cuda:0")
    train_g.edata['h_e'] = train_g.edata['h_e'].cuda()
    train_data = torch.Tensor(train_data)
    train_data = train_data.to("cuda:0")
    test_g = construct_graph(test_data, neighbor=5)
    test_g = test_g.to("cuda:0")
    test_g.edata['h_e'] = test_g.edata['h_e'].cuda()
    test_data = torch.Tensor(test_data)
    test_data = test_data.to("cuda:0")

    _ = net([train_g, train_data])
    train_hidden_feature= net.hidden_feature.cpu().numpy()

    _ = net([test_g, test_data])
    test_hidden_feature = net.hidden_feature.cpu().numpy()

    mat = {'fea_train': train_hidden_feature.T, 'fea_test': test_hidden_feature.T}
    base_path = "ExtractFeature/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    sio.savemat(os.path.join(base_path, "feature_GDAE.mat"), mat)
    print("completed!")
    print("time span: %.4f s" % (time.time()-start))