from unet_models import UNet11
from unet_models import UNet16
from unet_models import UNet
from unet_models import AlbuNet
from unet_models import LinkNet34
from prepare_data import getdatasetready
from torch.optim import Adam
from validation import validation_binary
from torch.utils.data import DataLoader
from loss import LossBinary, LossMulti
import utils
import argparse
from pathlib import Path

num_classes = 1


model_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--train_crop_height', type=int, default=1024)
    arg('--train_crop_width', type=int, default=1280)
    arg('--val_crop_height', type=int, default=1024)
    arg('--val_crop_width', type=int, default=1280)
    arg('--type', type=str, default='binary', choices=['eyes'])
    arg('--model', type=str, default='UNet', choices=model_list.keys())

    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1

    if args.type == 'binary':
        loss = LossBinary(jaccard_weight=args.jaccard_weight)
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    train_loader = DataLoader(
            dataset= getdatasetready(),
            num_workers=args.workers,
            batch_size=args.batch_size
    )

    valid_loader = DataLoader(
        dataset=getdatasetready(),
        num_workers=args.workers,
        batch_size=args.batch_size
    )

    valid = validation_binary
    model = UNet11(num_classes=num_classes, pretrained=True)

    print(train_loader)
    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )

main()