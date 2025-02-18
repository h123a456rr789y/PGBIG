import os
import sys
#sys.path.append('/mnt/hdd4T/mtz_home/code/SmoothPredictionRelease/')
sys.path.append(os.path.abspath('./'))
from utils import totalcap3d as datasets
from model import stage_4
from utils.opt import Options
from utils import util
from utils import log
from utils import viz_totalcap_3d as viz
from utils import util as util

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.DatasetsSmooth(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=1)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d']*25.4))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d']*25.4))
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#100ms']*25.4))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]*25.4])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]*25.4])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]*25.4])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d'] < err_best:
                err_best = ret_valid['m_p3d'] 
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)

def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()


    seed=1
    util.set_seed(seed)
    #load model
    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'])

    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    acts = ["acting", "freestyle", "rom", "walking"]
    data_loader = {}
    for act in acts:
        dataset = datasets.Datasets(opt=opt, split=1, actions=act)
        data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=10,
                             pin_memory=True)
    #do test
    is_create = True
    avg_ret_log = []

    for act in acts:
        ret_test = run_model(net_pred, is_train=3, data_loader=data_loader[act], opt=opt,act=act)
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]*25.4])
            head = np.append(head, ['test_' + k])

        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)

def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data
def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None,act=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d = 0
    else:
        titles = (np.array(range(opt.output_n//3)) + 1)*100
        m_p3d = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array(list(range(0, 63)))
    # dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
    #                      26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    #                      46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
    #                      75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # # joints at same loc
    # joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    # index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    # joint_equal = np.array([13, 19, 22, 13, 27, 30])
    # index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (p3d) in enumerate(data_loader):
        batch_size, seq_n, joint_n = p3d.shape
        p3d_thres = torch.tensor([])
        for j in range(batch_size):
            if(p3d[j,:,:].sum()!=0):
                p3d_thres = torch.cat((p3d_thres, p3d[j,:,:]),0)
        
        p3d_thres= torch.reshape(p3d_thres,(-1,seq_n,joint_n))
        p3d=p3d_thres
        batch_size,_,_ = p3d.shape
        if (p3d.sum()==0):
            continue

        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d = p3d.float().to(opt.cuda_idx)

        smooth1 = smooth(p3d[:, :, dim_used],
                         sample_len=opt.kernel_size + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        smooth2 = smooth(smooth1,
                         sample_len=opt.kernel_size + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        smooth3 = smooth(smooth2,
                         sample_len=opt.kernel_size + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        input = p3d[:, :, dim_used].clone()

        p3d_sup_4 = p3d.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_3 = smooth1.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_2 = smooth2.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_1 = smooth3.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])

        p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = net_pred(input, input_n=in_n, output_n=out_n, itera=itera)


        p3d_out_4 = p3d.clone()[:, in_n:in_n + out_n]
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, seq_in:]
        # p3d_out_4[:, :, index_to_ignore] = p3d_out_4[:, :, index_to_equal]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, joint_n//3, 3])
        p3d = p3d.reshape([-1, in_n + out_n, joint_n//3, 3])
        
        #p3d_out = p3d_out_4.reshape([-1, out_n, joint_n])
        p3d_in = p3d.reshape([-1, in_n + out_n, joint_n])
        #p3d_in = np.array(p3d_in.cpu())
        # p3d_in = p3d_in.permute(0, 2, 1)
        # print(p3d_in.shape) 
        p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_3 = p3d_out_all_3.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_2 = p3d_out_all_2.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out = p3d_out_all_4.reshape([-1,in_n + out_n, joint_n])
        #p3d_out =  np.array(p3d_out.cpu())
        # print(p3d_out.shape) 
        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup_4, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3 - p3d_sup_3, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2 - p3d_sup_2, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup_1, dim=3))

            loss_all = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/4
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size


        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d = torch.mean(torch.norm(p3d[:, in_n:in_n + out_n] - p3d_out_4, dim=3))
            m_p3d += mpjpe_p3d.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d = torch.sum(torch.mean(torch.norm(p3d[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
            m_p3d += mpjpe_p3d.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

        # evlauation
        if is_train > 1:

            file_path ='./vis_result/{}/'.format(act)
            isExist = os.path.exists(file_path)
            if not isExist:
                os.mkdir(file_path)


            tested_batch= [0,5,11,17,23,31]
            
            for k in range(len(tested_batch)):
                fig3d = plt.figure()
                ax = fig3d.add_subplot(projection='3d') 
                figure_title = "_action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions(p3d_in[tested_batch[k], :, :], p3d_out[tested_batch[k], :, :], ax, figure_title,act,k+1,in_n)
                plt.close()

                fig2d = plt.figure( figsize=(50, 40))
                viz.plot_predictions_2d(p3d_in[tested_batch[k], :, :], p3d_out[tested_batch[k], :, :], figure_title,act,k+1,in_n)
                plt.close()
                plt.pause(1)
            break 
                                            


    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n 

    if is_train <= 1:
        ret["m_p3d"] = m_p3d / n 
    else:
        m_p3d = m_p3d / n
        for j in range(out_n//3):
            ret["#{:d}ms".format(titles[j])] = m_p3d[j*3+2]
    return ret

if __name__ == '__main__':

    option = Options().parse()

    if option.is_eval == False:
        main(opt=option)
    else:
        eval(option)
