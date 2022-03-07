import argparse
import os
import torch

from pprint import pformat
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from config import config
from core.augmentations import detection as detection_augmentations
from core.criterions import RotationAwareLoss
from core.datasets import readers
from core.models import models
from core.logging import configure_logger
from core.logging import TrainingMonitor

def main():
    logger = configure_logger()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", help="The pid to resume training.")
    parser.add_argument("--start", default=0, type=int, help="The epoch to resume training.")
    parser.add_argument("--export", default=False, help="The flag indicates exporting models.")
    args = parser.parse_args()

    # Create directories
    output_dir = f"{os.getpid()}" if args.pid is None else args.pid
    output_dir = os.path.join("result", output_dir)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")

    # Use GPU
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # Create model
    model = models[config.model.type][config.model.name]()
    # Use multiple GPUs for training
    model_p = nn.DataParallel(model, device_ids=[4,5,6,7])
    #logger.info(f"Using {torch.cuda.device_count()} GPUs")
    model_p = model_p.to(device)
    logger.info(f"Configuring model {model.config['name']}...")

    # Read datasets
    Reader = readers[config.dataset.reader]

    # Image augmentations
    if config.model.type == "detection":
        transform = transforms.Compose([
            detection_augmentations.Resize((model.config["image_size"], model.config["image_size"])),
            detection_augmentations.ToTensor(),
        ])
    elif config.model.type == "classification":
        transform = transforms.Compose([
            transforms.Resize((model.config["image_size"], model.config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Create dataset
    logger.info(f"Reading training data from {config.dataset.train_path} ({config.dataset.reader}) ...")
    train_dataset = Reader(config.dataset.train_path, transform=transform)
    logger.info(f"Classes: {', '.join(train_dataset.classes)}", extra={ "type": "DATASET" })
    logger.info(f"Training samples: {len(train_dataset)}", extra={ "type": "DATASET" })

    # Create dataloaders
    if config.model.type == "detection":
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=Reader.collate_fn)
    elif config.model.type == "classification":
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    logger.info(f"Configuring optimizer ...")
    # Set weight decay only on conv.weight
    parameters = []
    for key, value in model_p.named_parameters():
        if "conv.weight" in key:
            parameters += [{"params": value, "weight_decay": config.weight_decay}]
        else:
            parameters += [{"params": value, "weight_decay": 0.0}]

    # Create optimizer
    optimizer = SGD(parameters,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    if args.start > 0:
        logger.info(f"Resuming training from epoch {args.start} ...")
        checkpoint = torch.load(os.path.join(checkpoints_dir, f"ckpt-{args.start}.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.export:
            torch.onnx.export(
                model,
                Variable(torch.randn(1, 3, model.config["image_size"], model.config["image_size"])).to(device),
                f"{model.config['name']}.onnx",
                opset_version=11
            )
            return 0

    # Loss function
    if config.model.criterion == "rotation_aware_loss":
        criterion = RotationAwareLoss(model.config["image_size"])
    elif config.model.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    training_monitor = TrainingMonitor(
        os.path.join(output_dir, "monitor.jpg"),
        os.path.join(output_dir, "monitor.json"),
        start=args.start
    )
    training_monitor.init()
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.start, config.epochs):
        # Training loop
        model_p.train()
        for i, (images, targets) in enumerate(train_dataloader):
            #if i == 3: return 0
            images = images.to(device)
            predictions = model_p(images)
            if isinstance(targets, list):
                targets = [target.to(device) for target in targets]
            else:
                targets = targets.to(device)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                logger.info(f"Epoch[{epoch:03d}] Batch[{i:03d}]\t\tLoss: {loss.item():.5f}", extra={ "type": "TRAINING" })

        # Update training monitor
        training_monitor.update({ "loss": loss.item() })
        logger.info(f"Epoch[{epoch:03d}]\t\t\tLoss: {loss.item():.5f}", extra={ "type": "TRAINING" })

        # Save checkpoint
        torch.save({
            "model_state_dict": model_p.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

        }, os.path.join(checkpoints_dir, f"ckpt-{epoch}.tar"))


if __name__ == '__main__':
    main()
