'''
    Training script for one leave-one-site-out fold.

    Expects <data_root>/{train,val,test}.csv (see scripts/make_splits.py):
      train : tiles from the 11 training sites (minus inner-val dates)
      val   : inner validation dates from the training sites -> picks best epoch
      test  : the held-out site -> scored once, with the best checkpoint

    Outputs go to <output_dir> (default: runs/<held-out site>/):
      config.yaml   config used for this fold (read by the analysis notebooks)
      last.pt       latest epoch (model + optimizer + scheduler) for resuming
      best.pt       best inner-val epoch, with its test metrics added at the end
      results.json  per-fold summary consumed by scripts/summarize_loso.py

    2022 Benjamin Kellenberger (original); adapted for coral LOSO.
'''
import os
import json
import argparse
import yaml
from tqdm import trange
import pandas as pd
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
warnings.filterwarnings("ignore", module='rasterio')

# wandb is optional: without it (or with wandb_mode: disabled) training still runs.
# The hasattr check guards against stale local wandb/ log folders being imported
# as a namespace package when the real library isn't installed.
try:
    import wandb
    if not hasattr(wandb, 'init'):
        raise ImportError
except ImportError:
    wandb = None


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


def get_test_site(data_root):
    '''
        Name of the held-out site, read from test.csv.
    '''
    sites = pd.read_csv(os.path.join(data_root, 'test.csv')).site.unique()
    assert len(sites) == 1, f'test.csv should contain exactly one site, found {list(sites)}'
    return sites[0]


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


def setup_scheduler(cfg, optimizer):
    '''
        Cosine annealing from learning_rate down to min_lr over num_epochs.
    '''
    return CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs'],
        eta_min=cfg.get('min_lr', 1e-6)
    )


def load_checkpoint(output_dir, model, optimizer, scheduler):
    '''
        Resume from <output_dir>/last.pt if it exists. Only this fold's own
        directory is searched, so a new fold never picks up another fold's weights.

        Returns (start_epoch, best_oa_val, best_epoch).
    '''
    last_path = os.path.join(output_dir, 'last.pt')
    if not os.path.exists(last_path):
        print('Starting new model')
        return 0, -1.0, 0

    state = torch.load(last_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    print(f'Resuming from epoch {state["epoch"]} '
          f'(best val OA so far: {100 * state["best_oa_val"]:.2f}% at epoch {state["best_epoch"]})')
    return state['epoch'], state['best_oa_val'], state['best_epoch']


def save_checkpoint(output_dir, filename, model, stats, **extra):
    '''
        Save model weights plus stats (and any extra state) to <output_dir>/<filename>.
    '''
    save_stats = stats.copy()
    save_stats.update(extra)
    save_stats['model'] = model.state_dict()
    torch.save(save_stats, os.path.join(output_dir, filename))


def run_epoch(cfg, dataLoader, model, optimizer=None, desc='Val '):
    '''
        One pass over dataLoader. Trains if an optimizer is given, otherwise
        evaluates without gradients.

        Returns (mean loss per sample, overall accuracy per sample).
    '''
    device = cfg['device']
    training = optimizer is not None

    model.to(device)
    model.train(training)

    criterion = nn.CrossEntropyLoss(reduction='sum')

    loss_sum, n_correct, n_total = 0.0, 0, 0

    progressBar = trange(len(dataLoader))
    with torch.set_grad_enabled(training):
        for batch in dataLoader:
            data = batch[0].to(device)
            labels = batch[1].long().to(device)

            prediction = model(data)
            loss = criterion(prediction, labels)

            if training:
                optimizer.zero_grad()
                (loss / labels.numel()).backward()
                optimizer.step()

            loss_sum += loss.item()
            n_correct += (torch.argmax(prediction, dim=1) == labels).sum().item()
            n_total += labels.numel()

            progressBar.set_description(
                '[{}] Loss: {:.2f}; OA: {:.2f}%'.format(
                    desc, loss_sum / n_total, 100 * n_correct / n_total
                )
            )
            progressBar.update(1)

    progressBar.close()
    return loss_sum / n_total, n_correct / n_total


def main():
    parser = argparse.ArgumentParser(description='Train one leave-one-site-out fold.')
    parser.add_argument('--config', help='Path to config file', default='configs/loso_resnet50.yaml')
    parser.add_argument('--data_root', help='Folder with train/val/test.csv (overrides config)', default=None)
    parser.add_argument('--output_dir', help='Where checkpoints/results go (default: runs/<test site>)', default=None)
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    if args.data_root is not None:
        cfg['data_root'] = os.path.abspath(args.data_root)

    test_site = get_test_site(cfg['data_root'])
    output_dir = args.output_dir or os.path.join(cfg.get('runs_dir', 'runs'), test_site)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Held-out test site: {test_site}')
    print(f'Output directory:   {output_dir}')

    init_seed(cfg.get('seed', None))

    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # config next to the checkpoints, so analysis notebooks load the right splits
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    use_wandb = wandb is not None and cfg.get('wandb_mode', 'online') != 'disabled'
    if not use_wandb:
        print('wandb logging disabled')
    else:
        wandb.init(
            project=cfg.get('wandb_project', 'coral_bleaching'),
            group=cfg.get('wandb_group', 'loso'),
            name=test_site,
            config=cfg,
            mode=cfg.get('wandb_mode', 'online'),
        )

    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    dl_test = create_dataloader(cfg, split='test')

    print(f"Train pairs: {len(dl_train.dataset)}")
    print(f"Val pairs:   {len(dl_val.dataset)}")
    print(f"Test pairs:  {len(dl_test.dataset)}")

    # move to device before building the optimizer so resumed optimizer state lands there too
    model = CustomResNet(cfg['num_classes'], cfg['layers']).to(cfg['device'])
    optim = setup_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optim)
    current_epoch, best_oa_val, best_epoch = load_checkpoint(output_dir, model, optim, scheduler)

    numEpochs = cfg['num_epochs']
    save_every = cfg.get('save_every', 0)
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = run_epoch(cfg, dl_train, model, optim, desc='Train')
        loss_val, oa_val = run_epoch(cfg, dl_val, model, desc='Val ')

        current_lr = optim.param_groups[0]['lr']
        scheduler.step()

        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'lr': current_lr,
            'epoch': current_epoch,
        }

        # Best checkpoint is selected on the inner val set only (never the test site)
        if oa_val > best_oa_val:
            best_oa_val = oa_val
            best_epoch = current_epoch
            save_checkpoint(output_dir, 'best.pt', model, stats, best_epoch=current_epoch)
            print(f'New best model saved at epoch {current_epoch} with val OA: {100 * oa_val:.2f}%')

        save_checkpoint(output_dir, 'last.pt', model, stats,
                        optimizer=optim.state_dict(), scheduler=scheduler.state_dict(),
                        best_oa_val=best_oa_val, best_epoch=best_epoch)
        if save_every and current_epoch % save_every == 0:
            save_checkpoint(output_dir, f'{current_epoch}.pt', model, stats)

        if use_wandb:
            wandb.log({
                **stats,
                'best_oa_val': best_oa_val,
                'best_epoch': best_epoch
            })

    # ── Final: score the held-out site with the last and the best-val checkpoints
    loss_test_last, oa_test_last = run_epoch(cfg, dl_test, model, desc='Test (last)')

    best_path = os.path.join(output_dir, 'best.pt')
    best_state = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model'])
    loss_test, oa_test = run_epoch(cfg, dl_test, model, desc='Test (best)')

    best_state.update({'loss_test': loss_test, 'oa_test': oa_test, 'test_site': test_site})
    torch.save(best_state, best_path)

    n_test_tiles = len(dl_test.dataset.meta)
    results = {
        'test_site': test_site,
        'best_epoch': best_state['best_epoch'],
        'oa_val': best_state['oa_val'],
        'oa_test': oa_test,
        'loss_test': loss_test,
        'oa_test_last_epoch': oa_test_last,
        'n_test_tiles': n_test_tiles,
        'n_test_pairs': len(dl_test.dataset),
        'num_epochs': numEpochs,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if use_wandb:
        wandb.summary.update({'oa_test': oa_test, 'oa_test_last_epoch': oa_test_last,
                              'best_epoch': best_state['best_epoch']})
        wandb.finish()

    print(f'Training complete. Best val OA: {100 * best_oa_val:.2f}% at epoch {best_epoch}')
    print(f'Held-out site {test_site}: test OA {100 * oa_test:.2f}% '
          f'(last epoch: {100 * oa_test_last:.2f}%)')


if __name__ == '__main__':
    main()
