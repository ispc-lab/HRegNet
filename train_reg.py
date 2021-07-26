import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset

from models.models import HRegNet
from models.losses import transformation_loss
from models.utils import set_seed

from tqdm import tqdm
import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='HRegNet')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--voxel_size', type=float, default=0.3)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--runname', type=str, default='')
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--augment', type=float, default=0.0)
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    parser.add_argument('--freeze_detector', action='store_true')
    parser.add_argument('--freeze_feats', action='store_true')
    parser.add_argument('--use_fps', action='store_true')
    parser.add_argument('--data_list', type=str, default='')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--pretrain_feats', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=1.0)
    
    return parser.parse_args()

def val_reg(args, net):
    if args.dataset == 'kitti':
        val_seqs = ['06','07']
        val_dataset = KittiDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    elif args.dataset == 'nuscenes':
        val_seqs = ['val']
        val_dataset = NuscenesDataset(args.root, val_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    else:
        raise('Not implemented')

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
    
    net.eval()
    total_loss = 0
    total_R_loss = 0
    total_t_loss = 0
    count = 0
    pbar = tqdm(enumerate(val_loader))
    with torch.no_grad():
        for i, data in pbar:
            src_points, dst_points, gt_R, gt_t = data
            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            ret_dict = net(src_points, dst_points)
            l_trans, l_R, l_t = transformation_loss(ret_dict['rotation'], ret_dict['translation'], gt_R, gt_t, args.alpha)
            total_loss += l_trans.item()
            total_R_loss += l_R.item()
            total_t_loss += l_t.item()
            count += 1

    total_loss = total_loss/count
    total_R_loss = total_R_loss/count
    total_t_loss = total_t_loss/count

    return total_loss, total_R_loss, total_t_loss

def test_reg(args, net):
    if args.dataset == 'kitti':
        test_seqs = ['08','09','10']
        test_dataset = KittiDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    elif args.dataset == 'nuscenes':
        test_seqs = ['test']
        test_dataset = NuscenesDataset(args.root, test_seqs, args.npoints, args.voxel_size, args.data_list, 0.0)
    else:
        raise('Not implemented')

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
    
    net.eval()
    total_loss = 0
    total_R_loss = 0
    total_t_loss = 0
    count = 0
    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for i, data in pbar:
            src_points, dst_points, gt_R, gt_t = data
            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            ret_dict = net(src_points, dst_points)
            l_trans, l_R, l_t = transformation_loss(ret_dict['rotation'], ret_dict['translation'], gt_R, gt_t, args.alpha)
            total_loss += l_trans.item()
            total_R_loss += l_R.item()
            total_t_loss += l_t.item()
            count += 1

    total_loss = total_loss/count
    total_R_loss = total_R_loss/count
    total_t_loss = total_t_loss/count

    return total_loss, total_R_loss, total_t_loss

def train_reg(args):

    if args.dataset == 'kitti':
        train_seqs = ['00','01','02','03','04','05']
        train_dataset = KittiDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif args.dataset == 'nuscenes':
        train_seqs = ['train']
        train_dataset = NuscenesDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    else:
        raise('Not implemented')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = HRegNet(args)
    net.feature_extraction.load_state_dict(torch.load(args.pretrain_feats))

    if args.use_wandb:
        wandb.watch(net)
    
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in tqdm(range(args.epochs)):
        net.train()
        count = 0
        total_loss = 0
        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            src_points, dst_points, gt_R, gt_t = data
            src_points = src_points.cuda()
            dst_points = dst_points.cuda()
            gt_R = gt_R.cuda()
            gt_t = gt_t.cuda()

            optimizer.zero_grad()
            ret_dict = net(src_points, dst_points)

            l_trans, l_R, l_t = transformation_loss(ret_dict['rotation'], ret_dict['translation'], gt_R, gt_t, args.alpha)
            loss = l_trans
            loss.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()

            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item()
                ))
        
        total_loss /= count
        total_val_loss, total_val_R, total_val_t = val_reg(args, net)
        total_test_loss, total_test_R, total_test_t = test_reg(args, net)
        if args.use_wandb:
            wandb.log({"train loss":total_loss, 
                       "val loss": total_val_loss, \
                       "val R": total_val_R, \
                       "val t":total_val_t, \
                       "test loss":total_test_loss,\
                       "test R":total_test_R,\
                       "test_t":total_test_t})
        
        print('\n Epoch {} finished. Training loss: {:.4f} Valiadation loss: {:.4f}'.\
            format(epoch+1, total_loss, total_val_loss))
        
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset + '_ckpt_'+args.runname)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        if total_loss < best_train_loss:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_train.pth'))
            best_train_loss = total_loss
            best_train_epoch = epoch + 1
        
        if total_val_loss < best_val_loss:
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_val.pth'))
            best_val_loss = total_val_loss
            best_val_epoch = epoch + 1
        
        print('Best train epoch: {} Best train loss: {:.4f} Best val epoch: {} Best val loss: {:.4f}'.format(
            best_train_epoch, best_train_loss, best_val_epoch, best_val_loss
        ))

        scheduler.step()

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(config=args, project='HRegNet', name=args.dataset+'_'+args.runname, dir=args.wandb_dir)
    train_reg(args)