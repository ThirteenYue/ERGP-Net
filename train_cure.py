import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from my_dataset import DriveDataset
from src import ERGP
from torchvision.transforms import transforms
import time
import datetime
import random
import os
from torch.nn.utils import clip_grad_norm_
from torch import nn, optim

import matplotlib.pyplot as plt

# def seed_torch(seed=1):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# seed_torch()

# count number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_acc(pred,target):
    mask = target
    mask[mask>0] = 1
    mask[mask<1] = 0
    pred = torch.argmax(pred, dim=1)
    fenzi = pred[mask>0].eq(mask[mask>0].view_as(pred[mask>0])).sum().item()
    fenmu = mask[mask>0].numel()
    acc = fenzi / fenmu

    return acc

def train(data_loader, model, optimizer, criterion, reduce_schedule):
    model.train()
    running_loss = 0.0
    count = 0
    running_acc = 0.0

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()

        # with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()

        # lr_scheduler.step()
        # exponent_schedule.step()
        # cosine_schedule.step()
        # reduce_schedule.step(loss)
        lr = optimizer.param_groups[0]["lr"]

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

        acc = get_acc(outputs, labels)
        running_acc += acc

        if batch_idx % args.log_interval != 0:
            continue

        print('[{}/{} ({:0.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f} \tlr:{:.7f}'.format(
            batch_idx * len(inputs),
            len(data_loader.dataset),
            100. * batch_idx / len(data_loader),
            loss.item(), acc, lr))

    epoch_loss = running_loss / count

    epoch_acc = running_acc / (batch_idx + 1)

    print('[End of tra epoch]\t tra_Loss: {:0.5f}\t tra_Acc: {:.5f}'.format(epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc, lr


def val(data_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    count = 0
    running_acc = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            acc = get_acc(outputs, labels)
            running_acc += acc
            count += inputs.size(0)

    epoch_loss = running_loss / count
    epoch_acc = running_acc / (batch_idx + 1)
    print('[End of val epoch]\t val_Loss: {:0.5f}\t val_acc: {:.5f}'.format(epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc

def create_model(n_class):

    model = ERGP(in_ch=3, n_class=2, bilinear=True,)
    return model

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Tensorboard writer
    writer = SummaryWriter(args.log_dir)
    save_filename = args.model_dir
    n_class = 2
    batch_size = args.batch_size

    results_mat = "./results_mat/results{}-100.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transforms = {
        'train': transforms.Compose([
            # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            # transforms.Resize((img_size,img_size)),  # 随机长宽比裁剪  img_size 输出大小 scale 裁剪的倍数范围
            transforms.RandomAffine(10.),     # 仿射变换
            transforms.RandomRotation(13.),  # 随机旋转 -13 到 13
            transforms.RandomHorizontalFlip(p=0.3),  # 依概率p 随机水平翻转 默认0.5
            transforms.RandomVerticalFlip(p=0.3),  # 垂直 默认0.5
            transforms.ToTensor(),       # 归一化到0-1了 除以了255
        ]),
        'valid': transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    }
    # Data
    train_dataset = DriveDataset(args.data_folder,
                                 train=True,
                                 transforms=data_transforms['train'])

    val_dataset = DriveDataset(args.data_folder,
                               train=False,
                               transforms=data_transforms['valid'])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               )

    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               )
    # 统计数据
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # Model
    model = create_model(n_class = n_class)
    model.to(device)
    # optimizer = Adam(model.parameters(), lr=args.lr)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # 余弦退火
    # cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=40,eta_min=0.00001)
    # 指数衰减
    # exponent_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
    # 自适应衰减
    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                           verbose=False, threshold=1e-4, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-8)


    # 可视化学习率曲线
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         exponent_schedule.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    criterion = torch.nn.CrossEntropyLoss()

    x = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    lr_list =[]

    test_losses = []
    best_loss = np.inf
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print("================== Epoch: {} ==================".format(epoch))

        train_loss, train_acc, lr = train(train_loader, model, optimizer, criterion, reduce_schedule)
        val_loss, val_acc = val(valid_loader, model, criterion)

        reduce_schedule.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        # Logs
        # writer.add_scalar('loss/train/loss', train_loss, epoch)
        # writer.add_scalar('loss/test/loss', val_loss, epoch)
        # writer.add_scalar('acc/train/acc', train_acc, epoch)
        # writer.add_scalar('acc/val/acc', val_acc, epoch)
        # writer.add_scalar('lr',lr, epoch)

        with open(results_mat,"a")as f:
            f.write("t_loss:{:.4f}  t_acc:{:.4f}  v_loss:{:.4f}  v_acc:{:.4f}  lr:{:.7f} \n"
                    .format(train_loss,train_acc,val_loss,val_acc,lr))

        # matplotlib
        x.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        lr_list.append(lr)

        # loss曲线
        plt.clf()
        plt.plot(x, train_loss_list, 'r', lw=1)  # lw为曲线宽度
        plt.plot(x, val_loss_list, 'b', lw=1)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train_loss","val_loss"])
        plt.savefig('./cure/loss-b4.png')

        # acc曲线
        plt.clf()
        plt.plot(x, train_acc_list, 'r', lw=1)  # lw为曲线宽度
        plt.plot(x, val_acc_list, 'b', lw=1)
        plt.title("acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend(["train_acc","val_acc"])
        plt.savefig('./cure/acc-b4.png')

        # lr曲线
        plt.clf()
        plt.plot(x, lr_list, 'r', lw=1)  # lw为曲线宽度
        plt.title("lr")
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.legend(["train_lr"])
        plt.savefig('./cure/lr-b4.png')

        test_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss

            # Save model
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)

        # Early stopping
        if args.patience is not None and epoch > args.patience + 1:
            loss_array = np.array(test_losses)

            if all(loss_array[-args.patience:] - best_loss >
                   args.early_stopping_eps):
                break
    writer.close()
    print("Model saved at: {0}/best.pt".format(save_filename))
    print("# Parameters: {}".format(count_parameters(model)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    return


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Patchy VAE')

    # Dataset
    parser.add_argument('--dataset', type=str, default='lfw',
                        help='name of the dataset (default: lfw)')
    parser.add_argument('--data-folder', type=str, default='../Dataset',
                        help='name of the data folder (default: train)')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of image preprocessing (default: 1)')

    # Model
    parser.add_argument('--arch', type=str, default='patchy',
                        help='model architecture (default: patchy)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for Adam optimizer (default: 3e-4)')

    parser.add_argument('--betas', default=(0.9,0.999), type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Early Stopping
    parser.add_argument('--patience', type=int, default=None,
                        help='patience for early stopping (default: None)')
    parser.add_argument('--early-stopping-eps', type=int, default=1e-5,
                        help='patience for early stopping (default: 1e-5)')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='./scratch',
                        help='name of the output folder (default: ./scratch)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    # args.device = torch.device("cpu")

    # Slurm
    if 'SLURM_JOB_NAME' in os.environ and 'SLURM_JOB_ID' in os.environ:
        # running with sbatch and not srun
        if os.environ['SLURM_JOB_NAME'] != 'bash':
            args.output_folder = os.path.join(args.output_folder,
                                              os.environ['SLURM_JOB_ID'])
    else:
        args.output_folder = os.path.join(args.output_folder, str(os.getpid()))

    # Create logs and models folder if they don't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    log_dir = os.path.join(args.output_folder, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = os.path.join(args.output_folder, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.log_dir = log_dir
    args.model_dir = model_dir

    main()
