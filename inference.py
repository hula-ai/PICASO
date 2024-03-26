import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd

from model import *
from utils.eval_utils import compute_metrics_anomaly, compute_accuracy_anomaly
from utils.visualize_utils import visualize
from config import options
from dataset.dataset_Crescent import Crescent as data
#np.random.seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
options.load_model_path = './save/20230922_105540/models/best_model.ckpt'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


@torch.no_grad()
def evaluate(n):
    print(n)
    net.eval()

    """def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    net.apply(set_bn_eval)"""

    test_loss = 0
    targets, outputs = [], []

    test_dataset = data(dataframe='val_fold4.csv', num=np.random.randint(low=3, high=n), mode='test', #'val_fold0.csv'
                        transform=transforms.Compose(augmentation2))

    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    with torch.no_grad():
        for batch_id, (img, target) in enumerate(test_loader):
            img, target = img.cuda(), target.cuda()
            img = img.to(dtype=torch.float)
            target = target.view(target.size()[0], -1).float()
            output = net(img)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
        test_aupr, test_auc = compute_metrics_anomaly(torch.cat(outputs), torch.cat(targets))
        test_acc = compute_accuracy_anomaly(torch.cat(outputs), torch.cat(targets))

        # display
        log_string("validation_loss: {0:.4f}, validation_auc: {1:.02%}, validation_aupr: {2:.02%}, validation_acc: {3: .02%}"
                   .format(test_loss, test_auc, test_aupr, test_acc))

        output_all = torch.cat(outputs)
        target_all = torch.cat(targets)
        prediction_scores = torch.max(torch.sigmoid(output_all), dim=1)[0]  # .detach().cpu().numpy()
        true_labels = prediction_scores[torch.argmax(target_all, dim=1) == torch.argmax(output_all, dim=1)]
        true_labels = true_labels.detach().cpu().numpy()
        scores = pd.DataFrame({'fold_4': true_labels})
        scores.to_csv('scores4.csv')


    return test_aupr, test_auc, test_acc

    #################################
    # Grad Cam visualizer
    #################################
    # if options.gradcam:
    #     log_string('Generating Gradcam visualizations')
    #     iter_num = options.load_model_path.split('/')[-1].split('.')[0]
    #     img_dir = os.path.join(save_dir, 'imgs')
    #     if not os.path.exists(img_dir):
    #         os.makedirs(img_dir)
    #     viz_dir = os.path.join(img_dir, iter_num)
    #     if not os.path.exists(viz_dir):
    #         os.makedirs(viz_dir)
    #     visualize(net, test_loader, grad_cam_hooks, viz_dir)
    #     log_string('Images saved in: {}'.format(viz_dir))


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference_C_picaso.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    net = net(512, 512, 8)
    #net = DeepSet('max', 512, 512)

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    os.system('cp {}/dataset/dataset_Crescent.py {}'.format(BASE_DIR, save_dir))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    augmentation2 = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start Testing')
    evaluate(n=8)








