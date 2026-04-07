'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''
import wandb
import os
import argparse
import yaml
import glob
from tqdm import trange
from osgeo import gdal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# let's import our own classes and functions!
from util import init_seed
from dataset import BleachDataset
from model import CustomResNet

import warnings
import rasterio
warnings.filterwarnings("ignore", module='rasterio')


def create_dataloader(cfg, split='train', eval=False):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = BleachDataset(cfg, split, eval=eval)

    dataLoader = DataLoader(
        dataset=dataset_instance,
        batch_size=cfg['batch_size'],
        shuffle=(split == 'train'),
        num_workers=cfg['num_workers']
    )
    return dataLoader


def load_model(cfg):
    '''
        Creates a model instance and loads the latest numbered model state weights.
        Also loads best validation checkpoint metadata if available.
    '''
    model_instance = CustomResNet(cfg['num_classes'], cfg['layers'])

    # Only match numbered checkpoints, not best.pt
    model_states = glob.glob('model_states/[0-9]*.pt')
    if len(model_states):
        model_epochs = [
            int(os.path.basename(m).replace('.pt', ''))
            for m in model_states
        ]
        start_epoch = max(model_epochs)

        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(f'model_states/{start_epoch}.pt', map_location='cpu')
        model_instance.load_state_dict(state['model'])
    else:
        print('Starting new model')
        start_epoch = 0

    # Load best checkpoint metadata if present
    best_oa_val = -1.0
    best_epoch = 0
    best_path = 'model_states/best.pt'
    if os.path.exists(best_path):
        best_state = torch.load(best_path, map_location='cpu')
        best_oa_val = best_state.get('oa_val', -1.0)
        best_epoch = best_state.get('best_epoch', 0)
        print(f'Found previous best model at epoch {best_epoch} with val OA: {100 * best_oa_val:.2f}%')

    return model_instance, start_epoch, best_oa_val, best_epoch


def save_model(cfg, epoch, model, stats):
    '''
        Save per-epoch checkpoint.
    '''
    os.makedirs('model_states', exist_ok=True)

    save_stats = stats.copy()
    save_stats['model'] = model.state_dict()

    torch.save(save_stats, f'model_states/{epoch}.pt')

    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


def save_best_model(cfg, epoch, model, stats, filename='best.pt'):
    '''
        Save best validation checkpoint separately.
    '''
    os.makedirs('model_states', exist_ok=True)

    best_stats = stats.copy()
    best_stats['model'] = model.state_dict()
    best_stats['best_epoch'] = epoch

    torch.save(best_stats, os.path.join('model_states', filename))

    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay']
    )
    return optimizer


def setup_scheduler(cfg, optimizer, start_epoch=0):
    '''
        Cosine annealing scheduler.
        T_max is the number of epochs over which LR decays from initial lr to eta_min.
    '''
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs'],
        eta_min=cfg.get('min_learning_rate', 1e-6)
    )

    # Advance scheduler if resuming
    for _ in range(start_epoch):
        scheduler.step()

    return scheduler


def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    device = cfg['device']

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    loss_total, oa_total = 0.0, 0.0

    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):
        data = data.to(device)
        labels = labels.long().to(device)

        prediction = model(data)

        optimizer.zero_grad()
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)
        oa = torch.mean((pred_label == labels).float())
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total / (idx + 1),
                100 * oa_total / (idx + 1)
            )
        )
        progressBar.update(1)

    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    device = cfg['device']
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    loss_total, oa_total = 0.0, 0.0

    progressBar = trange(len(dataLoader))

    with torch.no_grad():
        for idx, (data, labels) in enumerate(dataLoader):
            data = data.to(device)
            labels = labels.long().to(device)

            prediction = model(data)
            loss = criterion(prediction, labels)

            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total / (idx + 1),
                    100 * oa_total / (idx + 1)
                )
            )
            progressBar.update(1)

    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total


def main():
    wandb.init(
        project="coral_bleaching",
    )

    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    init_seed(cfg.get('seed', None))

    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    print(f"Train dataset length: {len(dl_train.dataset)}")
    print(f"Val dataset length:   {len(dl_val.dataset)}")

    model, current_epoch, best_oa_val, best_epoch = load_model(cfg)

    optim = setup_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optim, current_epoch)

    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)

        scheduler.step()
        current_lr = optim.param_groups[0]['lr']

        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'lr': current_lr
        }

        # Save regular epoch checkpoint
        save_model(cfg, current_epoch, model, stats)

        # Save best validation checkpoint
        if oa_val > best_oa_val:
            best_oa_val = oa_val
            best_epoch = current_epoch
            save_best_model(cfg, current_epoch, model, stats, filename='best.pt')
            print(f'New best model saved at epoch {current_epoch} with val OA: {100 * oa_val:.2f}%')

        wandb.log({
            **stats,
            'best_oa_val': best_oa_val,
            'best_epoch': best_epoch
        })

    print(f'Training complete. Best val OA: {100 * best_oa_val:.2f}% at epoch {best_epoch}')


if __name__ == '__main__':
    gdal.UseExceptions()
    main()