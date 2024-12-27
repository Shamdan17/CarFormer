from carformer.data import (
    SequenceDataset,
    PlantSequenceDataset,
    DatasetPreloader,
    InMemoryDatasetPreloader,
    AugmentableDatasetPreloader,
)
import os


def get_datasets(config, model=None, return_all=False, splits=["train", "val"]):
    if return_all:
        return _get_entire_dataset(
            config.data_dir,
            config.training,
            config.dataset.plant_data,
            preload=config.preload,
            preload_in_memory=config.preload_in_memory,
            wipe_cache=config.wipe_cache,
            augmentable=config.augmentable_preloader,
            cache_dir=config.cache_dir,
            model=model,
        )

    return _get_datasets(
        config.data_dir,
        config.training,
        config.dataset.plant_data,
        preload=config.preload,
        preload_in_memory=config.preload_in_memory,
        wipe_cache=config.wipe_cache,
        augmentable=config.augmentable_preloader,
        cache_dir=config.cache_dir,
        model=model,
        splits=splits,
    )


def _get_datasets(
    data_dir,
    train_cfg,
    is_plant,
    preload=False,
    preload_in_memory=False,
    wipe_cache=False,
    augmentable=False,
    cache_dir="",
    model=None,
    splits=["train", "val"],
):
    data_module = SequenceDataset if not is_plant else PlantSequenceDataset

    if "train" in splits:
        train_dataset = data_module(
            data_dir,
            train_cfg.splits.train,
            train_cfg,
        )
    else:
        train_dataset = None

    if "val" in splits:
        val_dataset = data_module(
            data_dir,
            train_cfg.splits.val,
            train_cfg,
        )
    else:
        val_dataset = None

    if preload:
        assert cache_dir != "", "Cache dir must be specified if preloading is enabled"

        preloader = (
            DatasetPreloader if not preload_in_memory else InMemoryDatasetPreloader
        )
        args = []

        if augmentable:
            preloader = AugmentableDatasetPreloader
            args.append(
                "{}.pt".format(model.get_preprocessed_cache_parametrized_dirname())
            )
        if train_dataset is not None:
            train_dataset = preloader(
                train_dataset,
                os.path.join(cache_dir, train_dataset.get_parametrized_dirname()),
                *args,
                wipe_cache=wipe_cache,
            )

        if val_dataset is not None:
            val_dataset = preloader(
                val_dataset,
                os.path.join(cache_dir, val_dataset.get_parametrized_dirname()),
                *args,
                wipe_cache=wipe_cache,
            )
        if train_dataset is not None:
            train_dataset.load_state()

        if val_dataset is not None:
            val_dataset.load_state()

    return train_dataset, val_dataset


def _get_entire_dataset(
    data_dir,
    train_cfg,
    is_plant,
    preload=False,
    preload_in_memory=False,
    wipe_cache=False,
    augmentable=False,
    cache_dir="",
    model=None,
):
    data_module = SequenceDataset if not is_plant else PlantSequenceDataset

    all_dataset = data_module(
        data_dir,
        "all",
        train_cfg,
    )

    if preload:
        assert cache_dir != "", "Cache dir must be specified if preloading is enabled"

        preloader = (
            DatasetPreloader if not preload_in_memory else InMemoryDatasetPreloader
        )

        args = []

        if augmentable:
            preloader = AugmentableDatasetPreloader
            args.append(
                "{}.pt".format(model.get_preprocessed_cache_parametrized_dirname())
            )

        all_dataset = preloader(
            all_dataset,
            os.path.join(cache_dir, all_dataset.get_parametrized_dirname()),
            *args,
            wipe_cache=wipe_cache,
        )

        all_dataset.load_state()

    return all_dataset
