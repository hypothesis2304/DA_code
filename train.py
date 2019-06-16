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
import utils
import string
import copy
from tqdm import tqdm, trange
from collections import OrderedDict
import torch.nn.functional as F
import get_confident_idx

torch.cuda.manual_seed(18)
torch.cuda.manual_seed_all(18)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
random.seed(18)
np.random.seed(18)

activation_student = OrderedDict()
def get_activation_student(name):
    def hook(self, input_, output):
        activation_student[name] =  output
    return hook

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
    print("Deep copy of model with margin as 1.0")
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
    base_network_teacher = copy.deepcopy(base_network).cuda()
    for param in base_network_teacher.parameters():
        param.detach_()
    # base_network_teacher = base_network_teacher.cuda()

    # print("check init: ", torch.equal(base_network.fc.weight, base_network_teacher.fc.weight))

    base_network.layer1[-1].relu = nn.ReLU()
    base_network.layer2[-1].relu = nn.ReLU()
    base_network.layer3[-1].relu = nn.ReLU()
    base_network.layer4[-1].relu = nn.ReLU()

    base_network_teacher.layer1[-1].relu = nn.ReLU()
    base_network_teacher.layer2[-1].relu = nn.ReLU()
    base_network_teacher.layer3[-1].relu = nn.ReLU()
    base_network_teacher.layer4[-1].relu = nn.ReLU()

    # print(base_network)

    for n, m in base_network.named_modules():
    	if n == 'layer1.2.bn3' or 'layer2.3.bn3' or 'layer3.5.bn3' or 'layer4.2.bn3':
    		m.register_forward_hook(get_activation_student(n))

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
    temperature = config["temperature"]

    for i in trange(config["num_iterations"], leave=False):
        global activation_student
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.eval()
            base_network_teacher.eval()
            temp_acc, temp_acc_teacher = image_classification_test(dset_loaders, \
                base_network, base_network_teacher, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network_teacher)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            log_str1 = "precision: {:.5f}".format(temp_acc_teacher)
            config["out_file"].write(log_str+ "\t"+log_str1 + "\t" + str(classifier_loss.item())+"\t" + str(dann_loss.item())+"\t" + str(ent_loss.item())+ "\t" + "\n")
            config["out_file"].flush()
            print("ent Loss: ", ent_loss.item())
            print("Dann loss: ", dann_loss.item())
            print("Classification Loss: ", classifier_loss.item())
            print(log_str)
            print(log_str1)
        # if i % config["snapshot_interval"] == 0:
        #     torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
        #         "iter_{:05d}_model.pth.tar".format(i)))

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

        inputs_source1, inputs_source2, inputs_target1, inputs_target2, labels_source = utils.get_copies(inputs_source, inputs_target, labels_source)

        margin = 1
        loss_alter = 0

        #### For source data

        features_source, outputs_source = base_network(inputs_source1)
        # features_source2, outputs_source2 = base_network(inputs_source2)

        feature1 = base_network_teacher.features1(inputs_source2)
        feature2 = base_network_teacher.features2(feature1)
        feature3 = base_network_teacher.features3(feature2)
        feature4 = base_network_teacher.features4(feature3)
        feature4_avg = base_network_teacher.avgpool(feature4)
        feature4_res = feature4_avg.view(feature4_avg.size(0), -1)
        features_source2 = base_network_teacher.bottleneck(feature4_res)
        outputs_source2 = base_network_teacher.fc(features_source2)

        loss_alter += loss.decision_boundary_transfer(activation_student['layer1.2.bn3'], feature1.detach(), margin)/(train_bs*activation_student['layer1.2.bn3'].size(1) * 8)
        loss_alter += loss.decision_boundary_transfer(activation_student['layer2.3.bn3'], feature2.detach(), margin)/(train_bs*activation_student['layer2.3.bn3'].size(1) * 4)
        loss_alter += loss.decision_boundary_transfer(activation_student['layer3.5.bn3'], feature3.detach(), margin)/(train_bs*activation_student['layer3.5.bn3'].size(1) * 2)
        loss_alter += loss.decision_boundary_transfer(activation_student['layer4.2.bn3'], feature4.detach(), margin)/(train_bs*activation_student['layer4.2.bn3'].size(1))

        ## For Target data
        ramp = utils.sigmoid_rampup(i, 100004)
        ramp_confidence = utils.sigmoid_rampup(5*i, 100004)

        features_target, outputs_target = base_network(inputs_target1)
        sample_selection_indices = get_confident_idx.confident_samples(base_network,
        inputs_target1, ramp_confidence, class_num)

        confident_targets = utils.subsample(outputs_target, sample_selection_indices)

        feature1_teacher = base_network_teacher.features1(inputs_target2)
        feature2_teacher = base_network_teacher.features2(feature1_teacher)
        feature3_teacher = base_network_teacher.features3(feature2_teacher)
        feature4_teacher = base_network_teacher.features4(feature3_teacher)
        feature4_teacher_avg = base_network_teacher.avgpool(feature4_teacher)
        feature4_teacher_res = feature4_teacher_avg.view(feature4_teacher_avg.size(0), -1)
        features_target2 = base_network_teacher.bottleneck(feature4_teacher_res)
        outputs_target2 = base_network_teacher.fc(features_target2)

        loss_alter += loss.decision_boundary_transfer(activation_student['layer1.2.bn3'], feature1_teacher.detach(), margin)/(train_bs*activation_student['layer1.2.bn3'].size(1) * 8)
        loss_alter += loss.decision_boundary_transfer(activation_student['layer2.3.bn3'], feature2_teacher.detach(), margin)/(train_bs*activation_student['layer2.3.bn3'].size(1) * 4)
        loss_alter += loss.decision_boundary_transfer(activation_student['layer3.5.bn3'], feature3_teacher.detach(), margin)/(train_bs*activation_student['layer3.5.bn3'].size(1) * 2)
        loss_alter += loss.decision_boundary_transfer(activation_student['layer4.2.bn3'], feature4_teacher.detach(), margin)/(train_bs*activation_student['layer4.2.bn3'].size(1))

        loss_alter = loss_alter/1000 ## May be multiply with 4 later in tests
        loss_alter = loss_alter.unsqueeze(0).unsqueeze(1)

        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out_src = nn.Softmax(dim=1)(outputs_source)
        softmax_out_tar = nn.Softmax(dim=1)(outputs_target)
        softmax_out = nn.Softmax(dim=1)(outputs)

        features_teacher = torch.cat((features_source2, features_target2), dim=0)
        outputs_teacher = torch.cat((outputs_source2, outputs_target2), dim=0)
        softmax_out_src_teacher = nn.Softmax(dim=1)(outputs_source2)
        softmax_out_tar_teacher = nn.Softmax(dim=1)(outputs_target2)
        softmax_out_teacher = nn.Softmax(dim=1)(outputs_teacher)

        if config['method'] == 'DANN+E':
            ent_loss = Hloss(confident_targets)
            dann_loss = loss.DANN(features, ad_net)
        elif config['method']  == 'DANN':
            dann_loss = loss.DANN(features, ad_net)
            # dann_loss = 0
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        # loss_KD = -(F.softmax(outputs_teacher/ temperature, 1).detach() *
        # 	        (F.log_softmax(outputs/temperature, 1) - F.log_softmax(outputs_teacher/temperature, 1).detach())).sum() / train_bs
        # print(loss_KD)
        # total_loss =  loss_alter #+ (config["ent_loss"] * ent_loss)

        total_loss =  dann_loss + classifier_loss + (ramp * ent_loss) #+ (config["ent_loss"] * ent_loss)
        total_loss.backward(retain_graph=True)
        optimizer.step()
        loss.update_ema_variables(base_network, base_network_teacher, config["teacher_alpha"], i)
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='DANN+E', choices=['DANN', 'DANN+E'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--sdpath', type=str, default='../data/office-home/Clipart.txt', help="The source dataset path list")
    parser.add_argument('--tdpath', type=str, default='../data/office-home/Product.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=2000, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=500000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=True, help="whether use random projection")
    parser.add_argument('--teacher_alpha', type=float, default=0.999, help="amount of weight for weighted average")
    parser.add_argument('--ent_loss', type=float, default=0.1, help="parameter to balance entropy loss")
    parser.add_argument('--temperature', type=int, default=3, help="Temperature parameter")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
    config["temperature"] = args.temperature
    config["teacher_alpha"] = args.teacher_alpha
    config["ent_loss"] = args.ent_loss
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir

    src = (args.sdpath.split('/'))[-1][0]
    tar = (args.tdpath.split('/'))[-1][0]

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    curr_path = os.getcwd()
    pathfile = os.path.join(curr_path,'snapshot/san/')
    os.chdir(pathfile)
    files = os.listdir(os.path.join(curr_path, pathfile))
    for file in files:
        if file == 'log.txt':
            dest =  str(args.dset) + '-' + src.upper() + '-' + tar.upper() + '.txt'
            os.rename(file, dest)
    os.chdir(curr_path)

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
    config["data"] = {"source":{"list_path":args.sdpath, "batch_size":12}, \
                      "target":{"list_path":args.tdpath, "batch_size":12}, \
                      "test":{"list_path":args.tdpath, "batch_size":4}}

    if config["dataset"] == "office":
        if ("amazon" in args.sdpath and "webcam" in args.tdpath) or \
           ("webcam" in args.sdpath and "dslr" in args.tdpath) or \
           ("webcam" in args.sdpath and "amazon" in args.tdpath) or \
           ("dslr" in args.sdpath and "amazon" in args.tdpath):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.sdpath and "dslr" in args.tdpath) or \
             ("dslr" in args.sdpath and "webcam" in args.tdpath):
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
    config["out_file"].write('\n')
    config["out_file"].write("Sigmoid rampup decrease from 1 - 0 then 1")
    config["out_file"].flush()
    train(config)
