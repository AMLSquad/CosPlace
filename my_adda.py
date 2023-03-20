"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from model.network import GeoLocalizationNet
from adda_utils import loop_iterable, set_requires_grad
from datasets.train_dataset import TrainDataset
from datasets.target_dataset import TargetDataset
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DatasetArgs:
    def __init__(self, dataset_folder, pseudo_target_folder = None):
        self.augmentation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_folder = dataset_folder
        self.pseudo_target_folder = pseudo_target_folder
        self.brightness = 0.7
        self.contrast = 0.7
        self.saturation = 0.7
        self.hue = 0.5

class avg2d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, source):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(source.shape[0], -1)
        return x

def main(args):

    a = avg2d().to(device)

    source_model = GeoLocalizationNet(args.backbone_name, args.fc_output_dim).to(device)
    source_model.load_state_dict(torch.load(args.model_file))
   
    ds_args = DatasetArgs(args.source_dataset_path)
    train_folder = os.path.join(args.source_dataset_path, "train")
    source_dataset = TrainDataset(ds_args, train_folder)

    
    target_model = GeoLocalizationNet(args.backbone_name, args.fc_output_dim).to(device)
    target_dataset = TargetDataset(args.target_dataset_path)

    batch_size = args.batch_size
    features_dim = args.features_dim

    source_model = source_model.to(device)
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    clf = source_model
    source_model = source_model.backbone

    target_model = target_model.to(device)
    target_model.load_state_dict(torch.load(args.model_file))
    target_model = target_model.backbone

    discriminator = nn.Sequential(
        nn.Linear(features_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    ).to(device)

    source_batch = (batch_size ) // 2
    target_batch = batch_size - source_batch

    source_loader = DataLoader(source_dataset, batch_size=source_batch,
                               shuffle=True, num_workers=8, pin_memory=True)
    
    
    target_loader = DataLoader(target_dataset, batch_size=target_batch,
                               shuffle=True, num_workers=8, pin_memory=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=5e-3)
    target_optim = torch.optim.Adam(target_model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        batch_iterator = enumerate(zip(source_loader, target_loader))

        total_loss = 0
        total_accuracy = 0
        for (source_x, _), (target_x, _) in batch_iterator:
            # Train discriminator
            #set_requires_grad(target_model, requires_grad=False)
            #set_requires_grad(discriminator, requires_grad=True)
                
            source_x, target_x = source_x[0].to(device), target_x[0].to(device)

            source_features = source_model(source_x)
            source_features = a(source_features, source_x)
            target_features = target_model(target_x)
            target_features = a(target_features, target_x)

            discriminator_x = torch.cat([source_features, target_features])

            label_src = torch.ones(source_x.size(0)).long().to(device)
            label_tgt = torch.zeros(target_x.size(0)).long().to(device)
            discriminator_y = torch.cat((label_src, label_tgt), 0)
            

            preds = discriminator(discriminator_x)
            loss = criterion(preds, discriminator_y)

            discriminator_optim.zero_grad()
            loss.backward()
            discriminator_optim.step()

            total_loss += loss.item()
            total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            #set_requires_grad(target_model, requires_grad=True)
            #set_requires_grad(discriminator, requires_grad=False)

            discriminator_optim.zero_grad()
            
            target_features = target_model(target_x)
            target_features = a(target_features, target_x)


            # flipped labels
            discriminator_y = torch.ones(target_x.shape[0], device=device)

            preds = discriminator(target_features)
            loss = criterion(preds, discriminator_y)

            target_optim.zero_grad()
            loss.backward()
            target_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations*args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

        # Create the full target model and save it
        clf.backbone = target_model
        if not os.path.exists("adda_target_model"):
            os.makedirs("adda_target_model")
        torch.save(clf.state_dict(), 'adda_target_model/adda.pt' + str(epoch))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
    arg_parser.add_argument('--backbone-name', default='resnet18')
    arg_parser.add_argument('--fc-output-dim', type=int, default=512)
    arg_parser.add_argument('--model_file')
    arg_parser.add_argument('--source-dataset_path')
    arg_parser.add_argument('--target-dataset_path')
    arg_parser.add_argument('--features-dim', type=int, default=512)
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--iterations', type=int, default=500)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--k-disc', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=10)
    args = arg_parser.parse_args()
    main(args)