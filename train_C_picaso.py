import warnings
import numpy as np
warnings.filterwarnings("ignore")
from datetime import datetime
from dataset.dataset_Crescent_2 import Crescent as data

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.backends.cudnn as cudnn


from model import *
from utils.eval_utils import compute_metrics_anomaly, compute_accuracy_anomaly
#from utils.logger_utils import Logger
from config import options

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0
    best_auc = 0

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()

        train_loss = 0
        targets, outputs = [], []
        batch_id = -1

        train_dataset = data(dataframe='train_fold0.csv', num=np.random.randint(low=3, high=8), mode='train',
                             transform=transforms.Compose(augmentation1))
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                                  num_workers=options.workers, drop_last=False)

        for (img, target) in train_loader:
            global_step += 1
            batch_id += 1

            img = img.cuda()
            target = target.cuda()
            #img = img.view(-1, img.size()[2], img.size()[3], img.size()[4])
            img = img.to(dtype=torch.float)
            target = target.view(target.size()[0], -1).float()

            output = net(img)

            batch_loss = nn.BCEWithLogitsLoss()(output, target)

            targets += [target]
            outputs += [output]

            train_loss += batch_loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss /= batch_id
        train_aupr, train_auc = compute_metrics_anomaly(torch.cat(outputs), torch.cat(targets))

        train_acc = compute_accuracy_anomaly(torch.cat(outputs), torch.cat(targets))

        log_string(
            "epoch: {0}, step: {1}, global step: {2}, train_loss: {3:.4f}, train_acc: {4: .4f}, train_auc: {5: .4f}, train_aupr: {6: .4f}"
            .format(epoch + 1, batch_id + 1, global_step, train_loss, train_acc, train_auc, train_aupr))

        log_string('--' * 30)
        log_string('Evaluating at step #{}'.format(global_step))
        best_loss, best_auc = evaluate(best_loss=best_loss,
                                       best_acc=best_acc,
                                       best_auc=best_auc,
                                       global_step=global_step)
        net.train()


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_auc = kwargs['best_auc']
    global_step = kwargs['global_step']

    net.eval()

    """def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    net.apply(set_bn_eval)"""

    test_loss = 0
    targets, outputs = [], []

    test_dataset = data(dataframe='val_fold0.csv', num=np.random.randint(low=3, high=8), mode='test',
                        transform=transforms.Compose(augmentation2))

    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    with torch.no_grad():
        for batch_id, (img, target) in enumerate(test_loader):
            img, target = img.cuda(), target.cuda()
            img = img.to(dtype=torch.float)
            target = target.view(target.size()[0], -1).float()

            output = net(img)
            batch_loss = nn.BCEWithLogitsLoss()(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss.item()

        test_loss /= (batch_id + 1)
        test_aupr, test_auc = compute_metrics_anomaly(torch.cat(outputs), torch.cat(targets))
        test_acc = compute_accuracy_anomaly(torch.cat(outputs), torch.cat(targets))

        # check for improvement
        loss_str, auc_str = '', ''
        if test_loss <= best_loss:
            loss_str, best_loss = '(improved)', test_loss
        if test_auc >= best_auc:
            auc_str, best_auc = '(improved)', test_auc

            # save checkpoint model
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            save_path = os.path.join(model_dir, 'best_model.ckpt')  # .format(global_step))
            torch.save({
                'global_step': global_step,
                'loss': test_loss,
                'acc': test_acc,
                'auc': test_auc,
                'aupr': test_aupr,
                'save_dir': model_dir,
                'state_dict': state_dict},
                save_path)
            log_string('Model saved at: {}'.format(save_path))

        # display
        log_string("validation_loss: {0:.4f} {1}, validation_acc: {2:.02%}, validation_auc: {3:.02%}{4}, validation_aupr: {5:.02%}"
                   .format(test_loss, loss_str, test_acc, test_auc, auc_str, test_aupr))


        log_string('--' * 30)
        return best_loss, best_auc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of train procedure
    os.system('cp {}/train_C_picaso.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/model.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/set_modules.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    #avail_models = timm.list_models('densenet*',pretrained=True)
    net = net(512, 512, 8)
    #net = DeepSet('mean1', 512, 512)

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)
    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.BCEWithLogitsLoss() #
    optimizer = Adam(net.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    augmentation1 = [
        #transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=(-25, 25), shear=15),
        transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
        transforms.ColorJitter(saturation=(0, 2), hue=0.3),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    augmentation2 = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}'.
               format(options.epochs, options.batch_size))

    train()

