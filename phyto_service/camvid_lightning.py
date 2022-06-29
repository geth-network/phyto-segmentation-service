import random
import os

import albumentations as albu
import cv2.cv2 as cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from sklearn.model_selection import ParameterGrid
from pytorch_lightning import loggers as pl_loggers, callbacks
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class PhytoModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes,
                 target_metric, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels,
            classes=out_classes, **kwargs
        )
        self.target_metric = target_metric
        self.lr = learning_rate
        # preprocessing parameters for image
        encoder_weights = kwargs.get("encoder_weights", "imagenet")
        params = smp.encoders.get_preprocessing_params(encoder_name,
                                                       encoder_weights)
        self.register_buffer("std",
                             torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean",
                             torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE,
                                           from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(),
                                               mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # # per image IoU means that we first calculate IoU score for each image
        # # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn,
        #                                       reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
           # f"{stage}_piiou": per_image_iou,
            f"{stage}_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', factor=0.85, verbose=True, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.target_metric
        }


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

        Args:
            images_dir (str): path to images folder
            masks_dir (str): path to segmentation masks folder
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline
                (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing
                (e.g. noralization, shape manipulation, etc.)

        """

    CLASSES = ['phyto']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.ids[0] = self.ids[-1]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id
                           in
                           self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in
                          self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in
                             classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True,
                         border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


if __name__ == '__main__':
    DATA_DIR = './PycharmProjects/conf/photoshop_dataset/final_dataset'

    x_train_dir = os.path.join(DATA_DIR, 'train/original')
    y_train_dir = os.path.join(DATA_DIR, 'train/masks')

    x_valid_dir = os.path.join(DATA_DIR, 'valid/original')
    y_valid_dir = os.path.join(DATA_DIR, 'valid/masks')

    dataset = Dataset(x_train_dir, y_train_dir, classes=['phyto'])

    augmented_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        classes=['phyto'],
    )

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['phyto']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    params = {
        "encoder": [
            "resnext50_32x4d",
            # "resnet50",
            "se_resnext50_32x4d",
            #"resnet18",
            #"resnet34",
            "resnet101",
            #"resnet152",
            #"timm-resnest14d",
            # "timm-resnest26d",
            # "timm-resnest50d",
            # "timm-resnest101e",
            # "timm-res2net50_26w_4s",
            # "timm-res2net101_26w_4s",
            # "timm-res2net50_26w_6s",
            # "timm-res2net50_26w_8s",
            # "timm-res2net50_48w_2s",
            # "timm-regnety_120",
            # "timm-regnety_080",
            # "timm-regnety_064",
            # "timm-regnety_040",
            # "timm-regnetx_120",
            # "timm-regnetx_160",
            # "timm-gernet_l",
            "se_resnext101_32x4d",
            # "se_resnext50_32x4d",
            # "se_resnet152",
            # "timm-skresnext50_32x4d",
            # "timm-skresnet34",
            # "densenet161",
            # "densenet201",
            "xception",
            "inceptionv4",
            "inceptionresnetv2",
            "efficientnet-b6",
            # "timm-mobilenetv3_large_100",
            # "timm-mobilenetv3_small_100",
            # "dpn98",
            # "dpn92",
            # "vgg19_bn",
            # "vgg19",
            # "vgg16_bn",
            # "vgg16"
        ],
        "lr": [0.00009, 0.00005, 0.0008, 0.0005, 0.0001]
    }
    grid_params = list(ParameterGrid(params))
    for iter_params in grid_params:
        encoder_name = iter_params.get('encoder')
        lr = iter_params.get('lr')
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name,
                                                             ENCODER_WEIGHTS)

        train_dataset = Dataset(
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        valid_dataset = Dataset(
            x_valid_dir,
            y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                  num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=4)
        monitor_metric = 'val_iou'

        model = PhytoModel("FPN", encoder_name, encoder_weights=ENCODER_WEIGHTS,
                           in_channels=3, out_classes=1,
                           target_metric=monitor_metric,
                           learning_rate=lr)
        model_name = f"{encoder_name}-{ENCODER_WEIGHTS}-{lr}"
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs-photoshop",
                                                 name=model_name)

        ckpt_callback = callbacks.ModelCheckpoint(
            monitor=monitor_metric,
            filename='model-{epoch:02d}-{val_iou:.5f}',
            save_top_k=2,
            mode='max',
            save_last=True,
            verbose=True,
        )
        early_callback = callbacks.EarlyStopping(
            monitor=monitor_metric,
            min_delta=0.001,
            patience=12,
            verbose=True,
            mode='max',
            check_finite=True,
            stopping_threshold=0.99999
        )

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=60,
            callbacks=[ckpt_callback, early_callback],
            logger=tb_logger
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        del trainer
        del model
        del train_loader
        del valid_loader
