import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from tqdm import tqdm, trange

torch.manual_seed(18)
random.seed(18)


def image_classification_test(loader, model, model_teacher, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in trange(len(loader['test'][0]), leave=False):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                outputs_teacher = []
                for j in trange(10, leave=False):
                    _, predict_out = model(inputs[j])
                    _, predict_out_teacher = model_teacher(inputs[j])

                    outputs.append(nn.Softmax(dim=1)(predict_out))
                    outputs_teacher.append(nn.Softmax(dim=1)(predict_out_teacher))
                outputs = sum(outputs)
                outputs_teacher = sum(outputs_teacher)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()

                    all_output_teacher = outputs_teacher.float().cpu()
                    all_label_teacher = labels.clone().detach().float()

                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

                    all_output_teacher = torch.cat((all_output_teacher, outputs_teacher.float().cpu()), 0)
                    all_label_teacher = torch.cat((all_label_teacher, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    _, predict_teacher = torch.max(all_output_teacher, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy_teacher = torch.sum(torch.squeeze(predict_teacher).float() == all_label_teacher).item() / float(all_label_teacher.size()[0])

    return accuracy, accuracy_teacher


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["call"](net_config["name"], **net_config["params"])
    base_network_teacher = net_config["call"](net_config["name"], **net_config["params_teacher"])
    base_network = base_network.cuda()
    base_network_teacher = base_network_teacher.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    Hloss = loss.Entropy()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])


    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in trange(config["num_iterations"], leave=False):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            base_network_teacher.train(False)
            temp_acc, temp_acc_teacher = image_classification_test(dset_loaders, \
                base_network, base_network_teacher, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network_teacher)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            log_str1 = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc_teacher)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            config["out_file"].write(log_str1+"\n")
            config["out_file"].flush()
            print(log_str)
            print(log_str1)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        base_network_teacher.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()

        # inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs_source1, inputs_source2 = network.Augmenter(inputs_source.numpy()), network.Augmenter(inputs_source.clone().detach().numpy())
        inputs_target1, inputs_target2 = network.Augmenter(inputs_target.numpy()), network.Augmenter(inputs_target.clone().detach().numpy())
        inputs_source1, labels_source = torch.from_numpy(inputs_source1).float().cuda(), labels_source.cuda()
        inputs_target1 = torch.from_numpy(inputs_target1).float().cuda()
        inputs_source2 = torch.from_numpy(inputs_source2).float().cuda()
        inputs_target2 = torch.from_numpy(inputs_target2).float().cuda()


        features_source, outputs_source = base_network(inputs_source1)
        features_target, outputs_target = base_network(inputs_target1)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out_src = nn.Softmax(dim=1)(outputs_source)
        softmax_out_tar = nn.Softmax(dim=1)(outputs_target)
        softmax_out = nn.Softmax(dim=1)(outputs)

        features_source2, outputs_source2 = base_network_teacher(inputs_source2)
        features_target2, outputs_target2 = base_network_teacher(inputs_target2)
        features_target = torch.cat((features_source2, features_target2), dim=0)
        outputs_target = torch.cat((outputs_source2, outputs_target2), dim=0)
        softmax_out_src_teacher = nn.Softmax(dim=1)(outputs_source2)
        softmax_out_tar_teacher = nn.Softmax(dim=1)(outputs_target2)
        softmax_out_teacher = nn.Softmax(dim=1)(outputs_target)
        
        if config['method'] == 'DANN+E':
            # ent_loss = loss.Entropy(softmax_out_tar)
            ent_loss = Hloss(outputs_target)
            dann_loss = loss.DANN(features, ad_net)
            transfer_loss = dann_loss + ent_loss
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
        loss.update_ema_variables(base_network, base_network_teacher, config["teacher_alpha"], i)
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='DANN+E', choices=['DANN', 'DANN+E'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/webcam_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/amazon_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=500000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--teacher_alpha', type=float, default=0.999, help="amount of weight for weighted average")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
    config["teacher_alpha"] = args.teacher_alpha
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"call": network.resnet_model, "name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "ema":False, "new_cls":True}, \
            "params_teacher":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "ema": True, "new_cls":True}} 
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 256

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":4}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 31
        config["network"]["params_teacher"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
        config["network"]["params_teacher"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
