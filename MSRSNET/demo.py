import sys
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset1
import logging
import torch
from models.Models import DPCD
from tqdm import tqdm
import os
from models.MSRSCD import MSRSCD
from models.FCCDN import FCCDN

def train_net(dataset_dir,model_dir,inference_size=4,load_checkpoint=True):
    save_dir = dataset_dir + '/label1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 1. Create dataset

    # compute mean and std of train dataset to normalize train/val/test dataset
    t1_mean, t1_std = [0.27481827 ,0.27763784 ,0.16916931],[0.14421893 ,0.11112755 ,0.08477218]
    t2_mean, t2_std = [0.2794117 , 0.28640259 ,0.17313303], [0.14867193, 0.11280791 ,0.09009067]
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    test_dataset = BasicDataset1(t1_images_dir=dataset_dir+'/t1/',
                                t2_images_dir=dataset_dir+'/t2/',
                                train=False, **dataset_args)
    # 2. Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size= inference_size, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Using device {device}')
    net = DPCD()
    net.to(device=device)



    load_model = torch.load(model_dir, map_location=device)
    net.load_state_dict(load_model)

    net.eval()
    logging.info('SET model mode to test!')
    from torchvision import transforms as T
    with torch.no_grad():
        for batch_img1, batch_img2, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            cd_preds = net(batch_img1, batch_img2)
            cd_preds = cd_preds[0]
            cd_preds = torch.sigmoid(cd_preds)
            cd_preds = (cd_preds > 0.5).float()  # 大于阈值的设为1，其余为0
            # cd_preds = torch.round(torch.sigmoid(cd_preds))
            cd_preds = cd_preds.cpu()
            to_pil = T.ToPILImage()
            for i ,pred in enumerate(cd_preds):
                pred=to_pil(pred)
                # pred.save(f'./{dataset_name}/test/pred1/{name[i]}.png')
                pred.save(save_dir+'/'+name[i]+'.png')
            # clear batch variables from memory
            del batch_img1, batch_img2
    print('over')


if __name__ == '__main__':

    dataset_dir=r'G:\datasets\HN_1024_huadong\test'
    model_dir='./model_dir/best_f1score_epoch257.pth'
    try:
        train_net(dataset_dir=dataset_dir, model_dir=model_dir,load_checkpoint=False)
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
