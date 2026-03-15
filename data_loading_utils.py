import random
import torch
import webdataset as wds
import numpy as np
import json
import io
import os
import glob
import albumentations as A
import einops
from torch.utils.data import IterableDataset


#Some helper functions

def get_augmentations(config, image_size):
    augmentations = config["augmentations"]
    independend_aug = []
    for k, v in augmentations.items():
        if k == "RandomResizedCrop":
            aug = A.augmentations.RandomResizedCrop(
                height=image_size, width=image_size, p=v["p"]
            )
        elif k == "ColorJitter":
            aug = A.augmentations.ColorJitter(
                brightness=v["value"][0],
                contrast=v["value"][1],
                saturation=v["value"][2],
                hue=v["value"][3],
                p=v["p"],
            )
        elif k == "HorizontalFlip":
            aug = A.augmentations.HorizontalFlip(p=v["p"])
        elif k == "VerticalFlip":
            aug = A.augmentations.VerticalFlip(p=v["p"])
        elif k == "RandomRotation":
            aug = A.augmentations.Rotate(p=v["p"])
        elif k == "GaussianBlur":
            aug = A.augmentations.GaussianBlur(sigma_limit=v["value"], p=v["p"])
        elif k == "ElasticTransform":
            aug = A.augmentations.ElasticTransform(p=v["p"])
        elif k == "Cutout":
            aug = A.augmentations.CoarseDropout(p=v["p"])
        elif k == "GaussianNoise":
            aug = A.augmentations.GaussNoise(p=v["p"])
        elif k == "MultNoise":
            aug = A.augmentations.MultiplicativeNoise(p=v["p"])
        independend_aug.append(aug)
    return A.ReplayCompose(independend_aug)

def normalize(image_timeseries, config, statistics_path="statistics.json"):
    """
    Normalize only the channels present in the image, based on the config.
    Handles geomorphology and atmospheric channels in primary/secondary pairs.

    Args:
        image_timeseries (Tensor): shape [T, C, H, W]
        config (dict): includes 'geomorphology_channels' and 'atmospheric_channels'
        statistics_path (str): JSON file with per-channel stats

    Returns:
        Tensor: normalized image_timeseries
    """
    #statistics = json.load(open(statistics_path, "r"))
    statistics = {
        "insar_difference":{
            "mean":0.001489616346722796,
            "std":1.6176833645747521
        },
        "insar_coherence":{
            "mean":69.3472689222075,
            "std":76.50325808397666
        },
        "dem":{
            "mean":833.0587512223403,
            "std":977.504283216005
        },
        "total_column_water_vapour":{
            "mean":22.526107385139355,
            "std":12.85386775756665
        },
        "surface_pressure":{
            "mean":92966.61998575741,
            "std":9132.050564654619
        },
        "vertical_integral_of_temperature":{
            "mean":2396036.7821715856,
            "std":263503.6556952968
        }
    }
    means = []
    stds = []

    # 1. Geomorphology channels come first
    for ch in config["geomorphology_channels"]:
        if ch in statistics:
            means.append(statistics[ch]["mean"])
            stds.append(statistics[ch]["std"])
        else:
            print(f"Missing stats for geomorph channel: {ch}")

    # 2. Atmospheric channels in primary-secondary pairs
    for ch in config["atmospheric_channels"]:
        if ch in statistics:
            # Add twice: once for primary, once for secondary
            means.extend([statistics[ch]["mean"], statistics[ch]["mean"]])
            stds.extend([statistics[ch]["std"], statistics[ch]["std"]])
        else:
            print(f"Missing stats for atmospheric variable: {ch}")

    # Apply normalization over each channel
    for i in range(len(means)):
        image_timeseries[:, i, :, :] = (image_timeseries[:, i, :, :] - means[i]) / stds[i]

    return image_timeseries


def augment(augmentations, insar_timeseries, mask_timeseries):    
    """Augment the image with the specified augmentations."""
    if not isinstance(insar_timeseries, np.ndarray):
        insar_timeseries = insar_timeseries.numpy()
    if not isinstance(mask_timeseries, np.ndarray):
        mask_timeseries = mask_timeseries.numpy()
    timeseries_length = insar_timeseries.shape[0]

    # Rearrange bands and masks to match the expected format
    # Timestep is given as different channels
    bands = einops.rearrange(insar_timeseries, "t c h w -> h w (c t)")
    masks = einops.rearrange(mask_timeseries, "t c h w -> h w (c t)")

    transform = augmentations(image=bands, mask=masks)
    augmented_bands = transform["image"]
    augmented_masks = transform["mask"]

    # Split time (T) back from channel (C*T)
    augmented_bands = einops.rearrange(augmented_bands, "h w (c t) -> t c h w", t=timeseries_length)
    augmented_masks = einops.rearrange(augmented_masks, "h w (c t) -> t c h w", t=timeseries_length)

    # If not tensor convert to tensor
    if not isinstance(augmented_bands, torch.Tensor):
        augmented_bands = torch.tensor(augmented_bands)
    if not isinstance(augmented_masks, torch.Tensor):
        augmented_masks = torch.tensor(augmented_masks)

    return augmented_bands, augmented_masks


def create_webdataset_loaders(configs, repeat=False, resample_shards=False):
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])

    all_channels = [
        "insar_difference", "insar_coherence", "dem", 
        "primary_date_total_column_water_vapour", "secondary_date_total_column_water_vapour",
        "primary_date_surface_pressure", "secondary_date_surface_pressure",
        "primary_date_vertical_integral_of_temperature", "secondary_date_vertical_integral_of_temperature"
        ]

    def get_channel_indices(channel_list, all_channels, is_atmospheric=False):
        indices = []
        for channel in channel_list:
            if is_atmospheric:
                prim = f"primary_date_{channel}"
                sec = f"secondary_date_{channel}"
                if prim in all_channels:
                    indices.append(all_channels.index(prim))
                else:
                    print(f"Warning: {prim} not in all_channels")
                if sec in all_channels:
                    indices.append(all_channels.index(sec))
                else:
                    print(f"Warning: {sec} not in all_channels")
            else:
                if channel in all_channels:
                    indices.append(all_channels.index(channel))
                else:
                    print(f"Warning: {channel} not in all_channels")
        return indices
    
    geomorphology_indices = get_channel_indices(configs['geomorphology_channels'], all_channels)
    atmospheric_indices = get_channel_indices(configs['atmospheric_channels'], all_channels, is_atmospheric=True)

    #Define data loading pipeline for training. Can include augmentations etc.
    def get_patches(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            sample = torch.load(io.BytesIO(sample["sample.pth"]))

            if label.ndim == 3:
                label = label[:, None, :, :]

            if isinstance(label, dict):
                label = label["label"]

            image = image.reshape(configs['timeseries_length'], len(all_channels), configs['image_size'], configs['image_size'])
            label = label.reshape(configs['timeseries_length'], 1, configs['image_size'], configs['image_size'])
            
            # Select only the relevant geomorphology channels and atmospheric channels
            selected_channels = geomorphology_indices + atmospheric_indices
            image = image[:, selected_channels, :, :]  # Keep only the selected channels
            
            if configs["augment"] == True:
                data_augmentations = get_augmentations(configs, configs['image_size'])
                image, label = augment(data_augmentations, image, label)
            
            image = normalize(image, configs)

            if configs['task'] == 'segmentation':
                if configs['timeseries_length'] != 1:
                    if configs['mask_target'] == 'peak':
                        counts = torch.sum(label, dim=(2, 3))
                        label = label[torch.argmax(counts), :, :, :]
                    elif configs['mask_target'] == 'union':
                        label = torch.sum(label, dim=0)
                        label = torch.where(label > 0, 1, 0)
                    elif configs['mask_target'] == 'last':
                        label = label[-1, :, :, :]
                else:
                    label = label[-1, :, :, :] # Last channel
            else:
                if configs['mask_target'] == 'union':
                    label = torch.tensor(int(np.any(sample['label'])==1))
                elif configs['mask_target'] == 'last':
                    label = torch.tensor(int(sample['label'][-1]==1))

            image = image.reshape(configs['timeseries_length']*(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels'])), configs['image_size'], configs['image_size'])

            if configs['task'] == 'segmentation':
                label = label.reshape(configs['image_size'], configs['image_size'])

            yield (image, label, sample)
    
    #Define data loading pipeline for evaluation.

    def get_patches_eval(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            sample = torch.load(io.BytesIO(sample["sample.pth"]))
            if isinstance(label, dict):
                label = label["label"]

            image = image.reshape(configs['timeseries_length'], len(all_channels), configs['image_size'], configs['image_size'])
            label = label.reshape(configs['timeseries_length'], 1, configs['image_size'], configs['image_size'])

            # Select only the relevant geomorphology channels and atmospheric channels
            selected_channels = geomorphology_indices + atmospheric_indices
            image = image[:, selected_channels, :, :]  # Keep only the selected channels

            image = normalize(image, configs)
            
            if configs['task'] == 'segmentation':
                if configs['timeseries_length'] != 1:
                    if configs['mask_target'] == 'peak':
                        counts = torch.sum(label, dim=(2, 3))
                        label = label[torch.argmax(counts)]
                    elif configs['mask_target'] == 'last':
                        label = label[-1, :, :, :]
                    elif configs['mask_target'] == 'union':
                        label = torch.sum(label, dim=0)
                        label = torch.where(label > 0, 1, 0)
                else:
                    label = label[-1, :, :, :] # Last channel
            else:
                label = torch.tensor(int(np.any(sample['label'])==1))

            image = image.reshape(configs['timeseries_length']*(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels'])), configs['image_size'], configs['image_size'])

            if configs['task'] == 'segmentation':
                if configs['mask_target'] == 'all':
                    label = label.reshape(configs['timeseries_length'], configs['image_size'], configs['image_size'])
                else:
                    label = label.reshape(configs['image_size'], configs['image_size'])

            yield (image, label, sample)
    configs["webdataset_path"] = os.path.join(configs["webdataset_root"], str(configs['timeseries_length']))

    for mode in ["train", "val", "test"]:
        if mode == "train":
            if not os.path.isdir(os.path.join(configs["webdataset_path"], 'train_pos')) or not os.path.isdir(os.path.join(configs["webdataset_path"], 'train_neg')):
                raise RuntimeError(f"Webdataset missing for mode: {mode}")
                
        else:
            if not os.path.isdir(os.path.join(configs["webdataset_path"], mode)):
                raise RuntimeError(f"Webdataset missing for mode: {mode}")

    compress = configs.get("compress", False)
    ext = ".tar.gz" if compress else ".tar"

    max_train_pos_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train_pos", f"*{ext}")))[-1]
    max_train_pos_index = max_train_pos_shard.split("-train_pos-")[-1][:-4]
    max_train_neg_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train_neg", f"*{ext}")))[-1]
    max_train_neg_index = max_train_neg_shard.split("-train_neg-")[-1][:-4]
    max_train_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train_neg", f"*{ext}")))[-1]

    pos_train_shards = os.path.join(
        configs["webdataset_path"],
        "train_pos",
        "sample-train_pos-{000000.." + max_train_pos_index + "}"+ext,
    )
    neg_train_shards = os.path.join(
        configs["webdataset_path"],
        "train_neg",
        "sample-train_neg-{000000.." + max_train_neg_index + "}"+ext,
    )

    positives = wds.WebDataset(pos_train_shards, shardshuffle=True, resampled=False).shuffle(
        configs["webdataset_shuffle_size"]
    ).compose(get_patches)
    negatives = wds.WebDataset(neg_train_shards, shardshuffle=True, resampled=False).shuffle(
        configs["webdataset_shuffle_size"]
    ).compose(get_patches)

    #train_dataset = wds.RandomMix(datasets=[positives, negatives], probs=[1, 1])
    count_pos = len([iter(positives)])
    count_neg = len([iter(negatives)])
    train_dataset = RandomMix(datasets=[positives, negatives], probs=[1/count_pos, 1/count_neg])

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=configs["prefetch_factor"],
        persistent_workers=configs["persistent_workers"],
    ).shuffle(configs["webdataset_shuffle_size"]).batched(configs["batch_size"], partial=False)
    
    train_loader = (
        train_loader.unbatched()
        .shuffle(
            configs["webdataset_shuffle_size"],
            initial=configs["webdataset_initial_buffer"],
        )
        .batched(configs["batch_size"])
    )
    if repeat:
        train_loader = train_loader.repeat()

    max_val_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "val", f"*{ext}")))[-1]
    max_val_index = max_val_shard.split("-val-")[-1][:-4]
    val_shards = os.path.join(
        configs["webdataset_path"],
        "val",
        "sample-val-{000000.." + max_val_index + "}" + ext,
    )

    val_dataset = wds.WebDataset(val_shards, shardshuffle=False, resampled=False)
    val_dataset = val_dataset.compose(get_patches_eval)
    val_dataset = val_dataset.batched(configs["batch_size"], partial=True)

    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    max_test_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "test", f"*{ext}")))[-1]
    max_test_index = max_test_shard.split("-test-")[-1][:-4]
    test_shards = os.path.join(
        configs["webdataset_path"],
        "test",
        "sample-test-{000000.." + max_test_index + "}"+ext,
    )

    test_dataset = wds.WebDataset(test_shards, shardshuffle=False, resampled=False)
    test_dataset = test_dataset.compose(get_patches_eval)
    test_dataset = test_dataset.batched(configs["batch_size"], partial=True)

    test_loader = wds.WebLoader(
        test_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

class RandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, probs=None, longest=False):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return random_samples(sources, self.probs, longest=self.longest)


def random_samples(sources, probs=None, longest=False):
    """Yield samples randomly from multiple sources based on given probabilities.

    Args:
        sources (list): List of iterable sources to draw samples from.
        probs (list, optional): List of probabilities for each source. Defaults to None.
        longest (bool): If True, continue until all sources are exhausted. Defaults to False.

    Yields:
        Sample randomly selected from one of the sources.
    """
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = random.random()
        i = np.searchsorted(cum, r)

        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break