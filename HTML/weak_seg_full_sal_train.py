# coding=utf-8
import pdb
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np
from datasets import VOC, Saliency
from datasets import palette as palette_voc
from evaluate_seg import evaluate_iou
from evaluate_sal import fm_and_mae
import json
import os
from jls_fcn import JLSFCN
from logger import Logger
import sys
from os import path
import glob
from PIL import Image
import pickle
import json
import os.path
import warnings
warnings.simplefilter("ignore", UserWarning)

image_size = 256
batch_size = 8
train_iters = 100001
c_output = 21
_num_show = 4
experiment_name = "test_19"
# experiment_name = "all_30"
# experiment_name = "penalty"
learn_rate = 1e-4
start_iter = 0

path_save_valid_voc = "output/validation/{}_voc".format(experiment_name)
if not os.path.exists(path_save_valid_voc): os.mkdir(path_save_valid_voc)

path_save_valid_voc_train = "output/validation/{}_voc_train_test".format(experiment_name)
if not os.path.exists(path_save_valid_voc_train): os.mkdir(path_save_valid_voc_train)

path_save_valid_sal = "output/validation/{}_sal".format(experiment_name)
if not os.path.exists(path_save_valid_sal): os.mkdir(path_save_valid_sal)

path_save_checkpoints = "output/checkpoints/{}".format(experiment_name)
if not os.path.exists(path_save_checkpoints): os.mkdir(path_save_checkpoints)

net = JLSFCN(c_output).cuda()
writer = Logger("output/logs/{}".format(experiment_name), 
        clear=True, port=8000, palette=palette_voc)


mean = torch.Tensor([0.485, 0.456, 0.406])[None, ..., None, None].cuda()
std = torch.Tensor([0.229, 0.224, 0.225])[None, ..., None, None].cuda()

voc_train_img_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/JPEGImages'
voc_train_gt_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAug'
voc_train_size_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeActual'
# voc_train_size_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeApprox'
# voc_train_size_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown'
voc_train_size_dir_2 = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown_2'

voc_val_img_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/JPEGImages'
voc_val_gt_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClass'
voc_val_size_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeActual'
# voc_val_size_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeApprox'
# voc_val_size_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown'

voc_train_split = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt'
voc_val_split = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

sal_train_img_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/saliency/DUT-train/images'
sal_train_gt_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/saliency/DUT-train/masks'

sal_val_img_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/saliency/ECSSD/images'
sal_val_gt_dir = '/home/s2bashar/Desktop/Thesis/jsws/datasets/saliency/ECSSD/masks'

sal_train_loader = torch.utils.data.DataLoader(
    Saliency(sal_train_img_dir, sal_train_gt_dir,
           crop=None, flip=True, rotate=10, size=image_size, training=True),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

sal_val_loader = torch.utils.data.DataLoader(
    Saliency(sal_val_img_dir, sal_val_gt_dir,
           crop=None, flip=False, rotate=None, size=image_size, training=False), 
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

voc_train_loader = torch.utils.data.DataLoader(
    VOC(voc_train_img_dir, voc_train_gt_dir, voc_train_size_dir, voc_train_split,
           crop=None, flip=True, rotate=10, size=image_size, training=True),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

# voc_train_loader_2 = torch.utils.data.DataLoader(
#     VOC(voc_train_img_dir, voc_train_gt_dir, voc_train_size_dir_2, voc_train_split,
#            crop=None, flip=True, rotate=10, size=image_size, training=True),
#     batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False)

voc_val_loader = torch.utils.data.DataLoader(
    VOC(voc_val_img_dir, voc_val_gt_dir, voc_val_size_dir, voc_val_split,
           crop=None, flip=False, rotate=None, size=image_size, training=False),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)


def val_sal():
    net.eval()
    with torch.no_grad():
        for it, (img, gt, batch_name, WW, HH) in tqdm(enumerate(sal_val_loader), desc='train'):
            img = (img.cuda()-mean)/std
            pred_seg, v_sal, _ = net(img)
            pred_seg = torch.softmax(pred_seg, 1)
            bg = pred_seg[:, :1]
            fg = (pred_seg[:, 1:]*v_sal[:, 1:]).sum(1, keepdim=True)
            fg = fg.squeeze(1)
            fg = fg*255
            for n, name in enumerate(batch_name):
                msk =fg[n]
                msk = msk.detach().cpu().numpy()
                w = WW[n]
                h = HH[n]
                msk = Image.fromarray(msk.astype(np.uint8))
                msk = msk.resize((w, h))
                msk.save('{}/{}.png'.format(path_save_valid_sal, name), 'PNG')
        fm, mae, _, _ = fm_and_mae(path_save_valid_sal, sal_val_gt_dir)
        net.train()
        return fm, mae


def val_voc():
    net.eval()
    with torch.no_grad():
        for it, (img, gt, size, batch_name, WW, HH) in tqdm(enumerate(voc_val_loader), desc='train'):
            gt_cls = gt[:, None, ...] == torch.arange(c_output)[None, ..., None, None]
            gt_cls = (gt_cls.sum(3).sum(2)>0).float().cuda()
            img = (img.cuda()-mean)/std
            batch_seg, _, _ = net(img)
            _, batch_seg = batch_seg.detach().max(1)
            for n, name in enumerate(batch_name):
                msk =batch_seg[n]
                msk = msk.detach().cpu().numpy()
                w = WW[n]
                h = HH[n]
                msk = Image.fromarray(msk.astype(np.uint8))
                msk = msk.convert('P')
                msk.putpalette(palette_voc)
                msk = msk.resize((w, h))
                msk.save('{}/{}.png'.format(path_save_valid_voc, name), 'PNG')
        miou = evaluate_iou(path_save_valid_voc, voc_val_gt_dir, c_output)
        net.train()
        return miou

def val_voc_train():
    net.eval()
    with torch.no_grad():
        for it, (img, gt, size, batch_name, WW, HH) in tqdm(enumerate(voc_train_loader), desc='train'):
            gt_cls = gt[:, None, ...] == torch.arange(c_output)[None, ..., None, None]
            gt_cls = (gt_cls.sum(3).sum(2)>0).float().cuda()
            img = (img.cuda()-mean)/std
            batch_seg, _, _ = net(img)
            _, batch_seg = batch_seg.detach().max(1)
            for n, name in enumerate(batch_name):
                msk =batch_seg[n]
                msk = msk.detach().cpu().numpy()
                w = WW[n]
                h = HH[n]
                msk = Image.fromarray(msk.astype(np.uint8))
                msk = msk.convert('P')
                msk.putpalette(palette_voc)
                msk = msk.resize((w, h))
                msk.save('{}/{}.png'.format(path_save_valid_voc_train, name), 'PNG')
        miou = evaluate_iou(path_save_valid_voc_train, voc_train_gt_dir, c_output)
        net.train()
        return miou

def get_start_iter():
    start_iters = []
    checkpoints = os.listdir('./output/checkpoints/' + experiment_name)
    if(len(checkpoints) == 0): return 0
    for checkpoint in checkpoints:
        start_iters.append(int(checkpoint.split('.')[0]))
    start_iters.sort()
    return(start_iters[-1])

def getAB(size):
    if size <= 0.1: return [0, 0.1]
    elif size > 0.1 and size <= 0.2: return [0.1, 0.2]
    elif size > 0.2 and size <= 0.3: return [0.2, 0.3]
    elif size > 0.3 and size <= 0.4: return [0.3, 0.4]
    elif size > 0.4 and size <= 0.5: return [0.4, 0.5]
    elif size > 0.5 and size <= 0.6: return [0.5, 0.6]
    elif size > 0.6 and size <= 0.7: return [0.6, 0.7]
    elif size > 0.7 and size <= 0.8: return [0.7, 0.8]
    elif size > 0.8 and size <= 0.9: return [0.8, 0.9]
    elif size > 0.9 and size <= 1  : return [0.9, 1]

def getPenalty(pred_cls_size_n, size_n, C = 21):
    penalty_n = torch.zeros_like(pred_cls_size_n)
    for i in range(C):
        # a, b = getAB(size_n[i])
        a, b = [0.15, 1]
        if pred_cls_size_n[i] < a:
            penalty_n[i] = a
        elif pred_cls_size_n[i] > b:
            penalty_n[i] = b
        else:
        	penalty_n[i] = pred_cls_size_n[i]
    return penalty_n

def reevaluate_size():
    print('starting')
    val_folder = '/home/s2bashar/Desktop/Thesis/jsws/output/validation/reevaluate_2_voc'
    train_folder = '/home/s2bashar/Desktop/Thesis/jsws/output/validation/reevaluate_2_voc_train_test'
    output_folder = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown'
    if not path.isdir(output_folder): os.makedirs(output_folder)

    filesValPng = glob.glob(val_folder + '/*.png')
    for i in range(len(filesValPng)):
        classCountActual = {
        	0: 0,
        	1: 0,  2: 0,  3: 0,  4: 0,  5: 0, 
        	6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
        	11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
        	16: 0, 17: 0, 18: 0, 19: 0, 20: 0
    	}
        outputPath = filesValPng[i].split('/')[-1].split('.')[0]
        im = np.asarray(Image.open(filesValPng[i]))
        imFlat = im.flatten()
    	
    	# Get size and count
        size = imFlat.shape[0]
        counts = np.unique(imFlat, return_counts = True)
        classPresent = counts[0]
        count = counts[1]

        # Update dictionary
        for i in range(len(classPresent)):
            if classPresent[i] == 255:
                continue
            classCountActual[classPresent[i]] = count[i] / size

        # Write to file
        file = open(output_folder + '/' + outputPath + '.pkl', 'wb')
        pickle.dump(classCountActual, file)
        file.close()

    filesTrainPng = glob.glob(train_folder + '/*.png')
    for i in range(len(filesTrainPng)):
        classCountActual = {
        	0: 0,
        	1: 0,  2: 0,  3: 0,  4: 0,  5: 0, 
        	6: 0,  7: 0,  8: 0,  9: 0,  10: 0,
        	11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
        	16: 0, 17: 0, 18: 0, 19: 0, 20: 0
    	}
        outputPath = filesTrainPng[i].split('/')[-1].split('.')[0]
        im = np.asarray(Image.open(filesTrainPng[i]))
        imFlat = im.flatten()
    	
    	# Get size and count
        size = imFlat.shape[0]
        counts = np.unique(imFlat, return_counts = True)
        classPresent = counts[0]
        count = counts[1]

        # Update dictionary
        for i in range(len(classPresent)):
            if classPresent[i] == 255:
                continue
            classCountActual[classPresent[i]] = count[i] / size

        # Write to file
        file = open(output_folder + '/' + outputPath + '.pkl', 'wb')
        pickle.dump(classCountActual, file)
        file.close()

    print('done')

def half():
    old_folder = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown'
    new_folder = '/home/s2bashar/Desktop/Thesis/jsws/datasets/segmentation/VOCdevkit/VOC2012/SegmentationClassAugSizeUnknown_new'

    files_old = glob.glob(old_folder + '/*.pkl')
    files_new = glob.glob(new_folder + '/*.pkl')
    files_len = min(len(files_old), len(files_new))

    if len(files_old) < len(files_new): files = files_old
    else: files = files_new

    for i in range(files_len):
        file_name = files[i].split('/')[-1].split('.')[0]
        count_old = pickle.load(open(old_folder + '/' + file_name + '.pkl', 'rb'))
        count_new = pickle.load(open(new_folder + '/' + file_name + '.pkl', 'rb'))
        for j in range(21):
            count_old[j] = (count_old[j] + count_new[j]) / 2
        file = open(old_folder + '/' + file_name + '.pkl', 'wb')
        pickle.dump(count_old, file)
        file.close()


def train():
    miou_list = []
    print(experiment_name)
    print("============================= TRAIN ============================")

    voc_train_iter = iter(voc_train_loader)
    voc_it = 0
    sal_train_iter = iter(sal_train_loader)
    sal_it = 0
    log = {'best_miou': 0, 'best_it_miou': 0, 
            'best_mae': 1000, 'best_it_mae':0, 'best_fm':0, 'best_it_fm':0}
    optimizer = torch.optim.Adam([{'params': net.parameters(), 
        'lr': learn_rate, 'betas':(0.95, 0.999)}])
    
    start_iter = get_start_iter()
    
    if start_iter > 0:
        net.load_state_dict(torch.load(os.path.join(
            path_save_checkpoints, "{}.pth".format(start_iter))))

    for i in range(start_iter, train_iters):
        if i % 2000 == 0:
            _lr = learn_rate / float(10**(i//2000))
            optimizer = torch.optim.Adam([{'params': net.parameters(), 
                'lr': _lr, 'betas':(0.95, 0.999)}])


        """loss 1 """
        if sal_it >= len(sal_train_loader):
            sal_train_iter = iter(sal_train_loader)
            sal_it = 0
        img_sal, gt_sal = sal_train_iter.next()
        sal_it += 1
        gt_sal = gt_sal[:, None, ...].cuda()
        gt_sal = gt_sal.squeeze(1).long()
        img_sal_raw = img_sal
        img_sal = (img_sal.cuda()-mean)/std

        pred_seg, v_sal, _ = net(img_sal)
        pred_seg = torch.softmax(pred_seg, 1)
        bg = pred_seg[:, :1]
        fg = (pred_seg[:, 1:]*v_sal[:, 1:]).sum(1, keepdim = True)
        pred_sal = torch.cat((bg, fg), 1)
        loss_sal = F.nll_loss(pred_sal, gt_sal)

        """loss 2 """
        if voc_it >= len(voc_train_loader):
            voc_train_iter = iter(voc_train_loader)
            voc_it = 0
        img_seg, gt_seg, size, _, __, ___ = voc_train_iter.next()

        voc_it += 1
        gt_cls = gt_seg[:, None, ...] == torch.arange(c_output)[None, ..., None, None]
        gt_cls = (gt_cls.sum(3).sum(2)>0).float().cuda()
        img_seg_raw = img_seg
        img_seg = (img_seg.cuda()-mean)/std
        pred_seg, _, seg32x = net(img_seg)

        
        pred_cls = pred_seg.mean(3).mean(2)
        pred_cls32x = seg32x.mean(3).mean(2)

        loss_cls = F.binary_cross_entropy_with_logits(pred_cls[:, 1:], gt_cls[:, 1:])+ F.binary_cross_entropy_with_logits(pred_cls32x[:, 1:], gt_cls[:, 1:])
        
        #Sharhad
        """loss 3_1 """
        
        """Try one, just buckets"""
        # weight = [1, 10, 19, 20, 21, 22, 25, 30, 40, 50, 100, 200, 500]

        # pred_seg_softmax = F.softmax(pred_seg)
        # pred_seg32x_softmax = F.softmax(seg32x)
        # pred_cls_size = pred_seg_softmax.mean(3).mean(2)
        # pred_cls32x_size = pred_seg32x_softmax.mean(3).mean(2)
        

        # N_c, C = pred_cls.shape
        # sum_N_c = 0
        # sum_N_c_32x = 0
        # for n in range(N_c):
        #     sum_C = torch.sum((pred_cls_size[n] - size[n].cuda()) ** 2)
        #     sum_C_32x = torch.sum((pred_cls32x_size[n] - size[n].cuda()) ** 2)

        #     sum_N_c += sum_C/(C - 1)
        #     sum_N_c_32x += sum_C_32x/(C - 1)

        # loss_size = 20 * (sum_N_c + sum_N_c_32x)/N_c
        """loss 3_1 """
        



        """loss 3_2 """
        weight = [1, 10, 20, 30, 40, 50, 100, 200, 500]

        pred_seg_softmax = F.softmax(pred_seg)
        pred_seg32x_softmax = F.softmax(seg32x)
        pred_cls_size = pred_seg_softmax.mean(3).mean(2) #t^hat
        pred_cls32x_size = pred_seg32x_softmax.mean(3).mean(2) #t^hat
        

        N_c, C = pred_cls.shape
        
        sum_N_c = 0
        sum_N_c_32x = 0
        for n in range(N_c):
            pentaly_n = getPenalty(pred_cls_size[n], size[n])
            sum_C = torch.sum((pred_cls_size[n] - pentaly_n.cuda()) ** 2)
            
            pentaly32x_n = getPenalty(pred_cls32x_size[n], size[n])
            sum_C_32x = torch.sum((pred_cls32x_size[n] - pentaly32x_n.cuda()) ** 2)

            sum_N_c += sum_C/(C - 1)
            sum_N_c_32x += sum_C_32x/(C - 1)
        loss_size = weight[2] * (sum_N_c + sum_N_c_32x)/N_c
        """loss 3_2 """





        #Sharhad

        # loss_size = F.mse_loss(F.softmax(pred_cls[:, 1:], 1), size[:, 1:].float().cuda()) + F.mse_loss(F.softmax(pred_cls32x[:, 1:], 1), size[:, 1:].float().cuda())
        loss = loss_sal + loss_size + loss_cls
        # loss = loss_size + loss_sal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """output """
        if i % 50 == 0:
            writer.add_scalar("sal_loss", loss_sal.item(), i)
            writer.add_scalar("cls_loss", loss_cls.item(), i)
            writer.add_scalar("size_loss", loss_size.item(), i)
            num_show = _num_show if img_seg.size(0) > _num_show else img_seg.size(0)
            img = img_seg_raw[-num_show:]
            writer.add_image('image_seg', torchvision.utils.make_grid(img), i)

            pred = gt_seg[-num_show:,None,...]
            pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
            pred = pred[0]
            writer.add_label('gt_seg', pred,i)
            writer.write_html()
            _, pred_label = pred_seg.max(1)
            pred = pred_label[-num_show:,None,...]
            pred = torchvision.utils.make_grid(pred.expand(-1, 3, -1,-1))
            pred = pred[0]
            writer.add_label('pred_seg', pred,i)
            writer.write_html()
            print("iter %d loss_sal %.4f; loss_cls %.4f; loss_size %4f"%(i, loss_sal.item(), loss_cls.item(), loss_size.item()))
            # print("iter %d loss_sal %.4f; loss_size %.4f"%(i, loss_sal.item(), loss_size.item()))
        """validation"""
        if i !=0 and i % 500 == 0:
            log[i] = {}
            save_dict = net.state_dict()
            torch.save(save_dict, "{}/{}.pth".format(path_save_checkpoints, i))
            
            miou = val_voc()
            writer.add_scalar("miou", miou, i)
            log[i]['miou'] = miou
            miou_list.append(miou)
            if miou > log['best_miou']:
                log['best_miou'] = miou
                log['best_it_miou'] = i
            print("validation: iter %d; miou %.4f; best %d:%.4f"%(i, miou, log['best_it_miou'], log['best_miou']))
            print(miou_list)

            # miou_train = val_voc_train()
            # print('training miou:', miou_train)


            fm, mae = val_sal()
            writer.add_scalar("mae", mae, i)
            writer.add_scalar("fm", fm, i)
            log[i]['mae'] = mae
            log[i]['fm'] = fm
            if mae < log['best_mae']:
                log['best_mae'] = mae
                log['best_it_mae'] = i
            if fm > log['best_fm']:
                log['best_fm'] = fm
                log['best_it_fm'] = i
            print("mae %.4f; best %d:%.4f"%(mae, log['best_it_mae'], log['best_mae']))
            print("fm %.4f; best %d:%.4f"%(fm, log['best_it_fm'], log['best_fm']))
            with open("output/{}.json".format(experiment_name), "w") as f:
                json.dump(log, f)

        # Reevaluating size after 2 epoch
        # if (i == 15000):
        # if (i != 0 and i % 1500 == 0):
        #     val_voc_train()
        #     reevaluate_size()
            # half()

        # if (i != start_iter and i % 15000 == 0):
        # 	sys.exit('Completed ' + str(i) + ' steps')





if __name__ == "__main__":
    train()
    #net.load_state_dict(torch.load("output/checkpoints/debug/500.pth"))
    #miou = val()
    #print(miou)
