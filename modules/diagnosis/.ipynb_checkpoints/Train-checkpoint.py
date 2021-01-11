import torch
from torch.utils.data import dataset
import pathlib
import random
import SimpleITK as sitk
import numpy as np
import logging
import shutil
import time
import h5py
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from args import Args
from Custom import CustomDataset
from CL_Model import UnetModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform:
    """
    Data Transformer for training models.
    """
    def __init__(self, use_seed=True, if_train=True):
        
        self.use_seed = use_seed
        self.if_train = if_train

    def __call__(self, t1, t1ce, flair, dwi, adc, seg, t1ce_mask, flair_mask, label, d_label): 

        t1_min = t1[t1ce_mask].min()
        t1_max = t1[t1ce_mask].max()
        t1 = np.multiply((t1 - t1_min)/(t1_max-t1_min), t1ce_mask.astype(np.float32))

        t1ce_min = t1ce[t1ce_mask].min()
        t1ce_max = t1ce[t1ce_mask].max()
        t1ce = np.multiply((t1ce - t1ce_min)/(t1ce_max-t1ce_min), t1ce_mask.astype(np.float32))
        
        flair_min = flair[flair_mask].min()
        flair_max = flair[flair_mask].max()
        flair = np.multiply((flair - flair_min)/(flair_max-flair_min), flair_mask.astype(np.float32))
        
        dwi_min = dwi[t1ce_mask].min()
        dwi_max = dwi[t1ce_mask].max()
        dwi = np.multiply((dwi - dwi_min)/(dwi_max-dwi_min), t1ce_mask.astype(np.float32))
        
        adc_min = adc[t1ce_mask].min()
        adc_max = adc[t1ce_mask].max()
        adc = np.multiply((adc - adc_min)/(adc_max-adc_min), t1ce_mask.astype(np.float32))
        
        seg1 = (seg == 1).astype(np.float32)
        seg2 = (seg == 2).astype(np.float32)
        seg3 = (seg == 3).astype(np.float32)
        
        label = label[0]
        d_label = d_label[0]
        
        input_vol = np.concatenate((t1[...,np.newaxis], t1ce[...,np.newaxis], flair[...,np.newaxis], dwi[...,np.newaxis], adc[...,np.newaxis], seg1[...,np.newaxis], seg2[...,np.newaxis], seg3[...,np.newaxis]), axis=3) #(x, y, z, 6)
        input_vol = np.transpose(input_vol,(3,0,1,2)) #(ch, x, y, z)
        
        return input_vol, label, d_label

def create_datasets(args):

    train_data = CustomDataset(
        root=args.data_path,
        transform=DataTransform(if_train=True),
        sample_rate=args.sample_rate,
        challenge='train',
        cv=args.cv
    )
    val_data = CustomDataset(
        root=args.data_path,
        transform=DataTransform(if_train=False),
        sample_rate=args.sample_rate,
        challenge='valid',
        cv=args.cv
    )
    return train_data, val_data

def create_data_loaders(args):

    train_data, val_data = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    acc_0 = []; acc_1 = [];
    d_acc_0=[]; d_acc_1=[]; d_acc_2=[]; d_acc_3=[]; d_acc_4=[]; d_acc_5=[]; d_acc_6=[]; d_acc_7=[];
    
    for iter, data in enumerate(tqdm(data_loader)):
        
        input_vol, target, d_target = data
        
        input_vol = input_vol.to(args.device)
        target = target.to(args.device)
        d_target = d_target.to(args.device)
        label = target.squeeze(0)

        randlist = [random.random() for _ in range(3)]
        
        if randlist[0] > 0.5:
            input_vol = torch.flip(input_vol, [2])
        if randlist[1] > 0.5:
            input_vol = torch.flip(input_vol, [3])
        if randlist[2] > 0.5:
            input_vol = torch.flip(input_vol, [4])
            
        output = model(input_vol)

        weights = [0.1]
        class_weights = torch.FloatTensor(weights).cuda()

        bceloss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

        loss = bceloss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m = torch.nn.Sigmoid()
        acc = torch.round(m(output)).item() == label.item()

        if target == 0:
            acc_0.append(int(acc))
        else:
            acc_1.append(int(acc))

        if d_target == 0:
            d_acc_0.append(int(acc))
        elif d_target == 1:
            d_acc_1.append(int(acc))
        elif d_target == 2:
            d_acc_2.append(int(acc))
        elif d_target == 3:
            d_acc_3.append(int(acc))
        elif d_target == 4:
            d_acc_4.append(int(acc))
        elif d_target == 5:
            d_acc_5.append(int(acc))
        elif d_target == 6:
            d_acc_6.append(int(acc))
        elif d_target == 7:
            d_acc_7.append(int(acc))

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            f'Loss = {loss.item():.4g} '
            f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    print('<Train Accuracy>')
    print('Non-Tumor Classification Accuracy: %.2f [ %d / %d]'%(sum(acc_0)/len(acc_0), sum(acc_0), len(acc_0)))
    print('    Tumor Classification Accuracy: %.2f [ %d / %d]'%(sum(acc_1)/len(acc_1), sum(acc_1), len(acc_1)))

    print('Glioma: %.2f [ %d / %d], Metas: %.2f [ %d / %d], PCNSL: %.2f [ %d / %d]'%(sum(d_acc_0)/len(d_acc_0), sum(d_acc_0), len(d_acc_0), sum(d_acc_1)/len(d_acc_1), sum(d_acc_1), len(d_acc_1), sum(d_acc_2)/len(d_acc_2), sum(d_acc_2), len(d_acc_2)))
    print('Vascul: %.2f [ %d / %d], Demyeli: %.2f [ %d / %d], Hemor: %.2f [ %d / %d], Infect: %.2f [ %d / %d], Ischem: %.2f [ %d / %d]'%(sum(d_acc_3)/len(d_acc_3), sum(d_acc_3), len(d_acc_3), sum(d_acc_4)/len(d_acc_4), sum(d_acc_4), len(d_acc_4), sum(d_acc_5)/len(d_acc_5), sum(d_acc_5), len(d_acc_5), sum(d_acc_6)/len(d_acc_6), sum(d_acc_6), len(d_acc_6), sum(d_acc_7)/len(d_acc_7), sum(d_acc_7), len(d_acc_7)))
    
    return avg_loss, time.perf_counter() - start_epoch

def evaluate(args, epoch, model, data_loader, writer):
    
    model.eval() #Applies No Dropout by calling model.eval()
    losses = [];
    accs = [];
    start = time.perf_counter()
    
    acc_0 = []; acc_1 = [];
    d_acc_0=[]; d_acc_1=[]; d_acc_2=[]; d_acc_3=[]; d_acc_4=[]; d_acc_5=[]; d_acc_6=[]; d_acc_7=[];
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input, target, d_target = data
            input = input.to(args.device)
            target = target.to(args.device) #target.shape = [1,1]
            d_target = d_target.to(args.device)
            label = target.squeeze(0)
            
            output = model(input) # output.shape = [8]
            
            weights = [0.2969]
            class_weights = torch.FloatTensor(weights).cuda()

            bceloss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
            
            loss = bceloss(output, target.squeeze(0))
        
            m = torch.nn.Sigmoid()
            acc = torch.round(m(output)).item() == label.item()
            
            losses.append(loss.item())
            accs.append(int(acc))
            
            if target == 0:
                acc_0.append(int(acc))
            else:
                acc_1.append(int(acc))
                
            if d_target == 0:
                d_acc_0.append(int(acc))
            elif d_target == 1:
                d_acc_1.append(int(acc))
            elif d_target == 2:
                d_acc_2.append(int(acc))
            elif d_target == 3:
                d_acc_3.append(int(acc))
            elif d_target == 4:
                d_acc_4.append(int(acc))
            elif d_target == 5:
                d_acc_5.append(int(acc))
            elif d_target == 6:
                d_acc_6.append(int(acc))
            elif d_target == 7:
                d_acc_7.append(int(acc))
                
        print('<Validation Accuracy>')
        print('Non-Tumor Classification Accuracy: %.2f [ %d / %d]'%(sum(acc_0)/len(acc_0), sum(acc_0), len(acc_0)))
        print('    Tumor Classification Accuracy: %.2f [ %d / %d]'%(sum(acc_1)/len(acc_1), sum(acc_1), len(acc_1)))

        print('Glioma: %.2f [ %d / %d], Metas: %.2f [ %d / %d], PCNSL: %.2f [ %d / %d]'%(sum(d_acc_0)/len(d_acc_0), sum(d_acc_0), len(d_acc_0), sum(d_acc_1)/len(d_acc_1), sum(d_acc_1), len(d_acc_1), sum(d_acc_2)/len(d_acc_2), sum(d_acc_2), len(d_acc_2)))
        print('Vascul: %.2f [ %d / %d], Demyeli: %.2f [ %d / %d], Hemor: %.2f [ %d / %d], Infect: %.2f [ %d / %d], Ischem: %.2f [ %d / %d]'%(sum(d_acc_3)/len(d_acc_3), sum(d_acc_3), len(d_acc_3), sum(d_acc_4)/len(d_acc_4), sum(d_acc_4), len(d_acc_4), sum(d_acc_5)/len(d_acc_5), sum(d_acc_5), len(d_acc_5), sum(d_acc_6)/len(d_acc_6), sum(d_acc_6), len(d_acc_6), sum(d_acc_7)/len(d_acc_7), sum(d_acc_7), len(d_acc_7)))
    
        writer.add_scalar('Val_Loss', np.mean(losses), epoch)
    return np.mean(losses), np.mean(accs), time.perf_counter() - start, 

def save_model(args, exp_dir, epoch, model, optimizer, val_loss, is_new_best, best_val_loss):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'exp_dir': exp_dir,
            'best_val_loss': best_val_loss
        },
        f=str(exp_dir)+'/'+ 'model_%d.pt'%epoch
    )
    if is_new_best:
        shutil.copyfile(str(exp_dir)+'/'+'model_%d.pt'%epoch, exp_dir / 'best_model.pt')

def build_model(args):
    model = UnetModel(
        in_chans=8,
        chans=args.num_chans,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint, model, optimizer

def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / args.sumpath)

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_val_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)
    
    torch.backends.cudnn.benchmark=True
    train_loader, val_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        val_loss, val_acc, val_time = evaluate(args, epoch, model, val_loader, writer)
        scheduler.step(epoch)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, val_loss, is_new_best, best_val_loss)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} AccVal = {val_acc:.4g} TrainTime = {train_time:.4f}s DevTime = {val_time:.4f}s',
        )
    writer.close()
    
def create_arg_parser():
    parser = Args()
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=16, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=1,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.98,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=10, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--sumpath', type=str, default='summary',
                        help='Which folder to save the event')
    parser.add_argument('--gpu', type=str, default='1', help='GPU Number')
    parser.add_argument('--cv', type=str, default='1', help='Cross Validation Fold')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    main(args)
