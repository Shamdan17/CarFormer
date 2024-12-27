from torch.utils.data import DataLoader
from carformer.data import get_datasets
from tqdm import tqdm, trange
from carformer.wanderer import Wanderer
from carformer.backbone import gpt2
from carformer.utils import calculate_model_stats, as_rgb, unwrap_model
import torch
import wandb
from collections import defaultdict
from carformer.visualization.visutils import (
    save_bin_probs,
    visualize_input_from_batch,
    get_probs_and_gt,
    create_confusion_matrix,
)
from carformer.utils import WeightedDistributedSampler, seed_everything
import os
import numpy as np
import json
import sys
from carformer.config import config_init
from omegaconf import OmegaConf
from carformer.utils.distributed import ddp_setup, save_on_master, get_rank

import hydra

config_init()


@hydra.main(version_base="1.1", config_path="carformer/config", config_name="config")
def main(cfg):
    seed_everything(cfg.seed)

    args = cfg

    if cfg.gpus > 1 and args.multi_gpu_strategy == "ddp" and not args.cpu:
        ddp_setup(args)

        master_process = get_rank() == 0
    else:
        master_process = True

    if master_process:
        # Wandb
        wandb.init(
            project=args.logging.project,
            entity=args.logging.entity,
            mode=args.logging.mode,
        )
        if args.wandb_name != "":
            # Append wandb tag to run name
            wandb.run.name += f"-{args.wandb_name}"

        if args.wandb_tag:
            wandb.run.tags = [args.wandb_tag]

        wandb.config.update(OmegaConf.to_container(args, resolve=True))

    backbone_cnf = OmegaConf.to_container(args.backbone, resolve=True)
    all_cnf = OmegaConf.to_container(args, resolve=True)
    all_cnf.pop("backbone")
    backbone_cnf.update(all_cnf)

    config = gpt2.GPT2Config.from_dict(backbone_cnf)
    print(config)

    model = Wanderer(config)
    train_dataset, val_dataset = get_datasets(args, model)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model.to(device)

    if cfg.gpus > 1 and not args.cpu:
        print("Using multiple GPUs")
        if args.multi_gpu_strategy == "dp":
            model = torch.nn.DataParallel(model)
        elif args.multi_gpu_strategy == "ddp":
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[get_rank()],
                output_device=get_rank(),
                # find_unused_parameters=True,
            )

    calculate_model_stats(model)

    opt = getattr(torch.optim, args.hyperparams.optimizer.name)(
        model.parameters(), lr=args.hyperparams.lr, **args.hyperparams.optimizer.kwargs
    )

    # Scheduler from huggingface
    from transformers import get_scheduler

    n_epochs = args.hyperparams.num_epochs
    batch_size = args.hyperparams.batch_size

    scheduler = get_scheduler(
        cfg.hyperparams.scheduler.name, opt, **cfg.hyperparams.scheduler.kwargs
    )

    if args.training.weighted_sampling:
        weights = train_dataset.getweights()
    else:
        weights = None

    sampler = WeightedDistributedSampler(
        train_dataset,
        args.dataset.subsample_ratio,
        shuffle=True if not args.overfit else False,
        weights=weights,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    val_sampler = WeightedDistributedSampler(
        val_dataset,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
    )

    # folder = os.path.join("checkpoints", wandb.run.name if not args.debug else "debug")
    folder = cfg.save_dir

    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "train_predictions"), exist_ok=True)
    os.makedirs(os.path.join(folder, "val_predictions"), exist_ok=True)

    # Save the model config and training args
    config.save_pretrained(folder)
    # Save all training arguments
    with open(os.path.join(folder, "args.json"), "w") as f:
        json.dump(OmegaConf.to_yaml(args, resolve=True), f, indent=2)

    # Save the command used to run the training into the file {folder}/command.sh
    with open(os.path.join(folder, "command.sh"), "w") as f:
        f.write("python ")
        f.write(" ".join(sys.argv))

    t = trange(n_epochs, desc="Epoch", leave=True)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp, init_scale=2**11)

    best_val_loss = 1000000
    use_early_stopping = args.early_stopping
    early_stopping_counter = 0
    early_stopping_patience = args.early_stopping_patience
    early_stopping_criterion = "action_classification_loss"
    val_epoch_loss = -1
    embed_log = ""

    for epoch in t:
        # Update the sampler
        sampler.set_epoch(epoch)

        model.train()
        avg_loss_dct = defaultdict(float)
        conf_matrix_inps = []
        t.reset()
        for i, batch in enumerate(tqdm(train_loader, leave=False)):
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                _, preds, loss, pred_labels, preprocessed_inputs = model(
                    batch, return_labels=True, return_inputs=True
                )
                # Average the loss across the gpus if multiple gpus are used
                probs_gt = get_probs_and_gt(
                    preds["deinterleaved_outputs"],
                    pred_labels,
                    unwrap_model(model).quantization_offset_map,
                    unwrap_model(model).quantization_vocab_size_map,
                    [unwrap_model(model).action_quantizer],
                )
                conf_matrix_inps.append(probs_gt)

                loss = {k: v.mean() for k, v in loss.items()}

                if args.augmentable_preloader:
                    if "to_cache" in preprocessed_inputs and args.preload:
                        train_dataset.append_features(
                            batch, preprocessed_inputs["to_cache"]
                        )

            grad_scaler.scale(loss["loss"]).backward()

            grad_scaler.unscale_(opt)
            for param in model.parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(
                        torch.isinf(param.grad)
                    ):
                        print("NAN/INFs in gradients")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.hyperparams.max_grad_norm
            )

            grad_scaler.step(opt)
            grad_scaler.update()
            for k, v in loss.items():
                avg_loss_dct[k] += v.item()

        if args.preload:
            train_dataset.save_state()

        if master_process:
            # Save the confusion matrix
            for token_type in conf_matrix_inps[0].keys():
                for name in conf_matrix_inps[0][token_type].keys():
                    conf_labels = np.concatenate(
                        [x[token_type][name]["labels"] for x in conf_matrix_inps],
                        axis=0,
                    )
                    conf_pred_scores = np.concatenate(
                        [x[token_type][name]["preds"] for x in conf_matrix_inps], axis=0
                    )

                    wandb.log(
                        {
                            f"train_conf_matrix/{name}": create_confusion_matrix(
                                conf_pred_scores,
                                conf_labels,
                                conf_matrix_inps[0][token_type][name][
                                    "pred_val_labels"
                                ],
                                name,
                            ),
                            "epoch": epoch,
                        }
                    )

        scheduler.step()

        train_epoch_loss = avg_loss_dct["loss"] / len(train_loader)

        model.eval()

        # Log all losses
        if master_process:
            for k, v in avg_loss_dct.items():
                if not args.debug:
                    wandb.log(
                        {"train/loss/" + k: v / len(train_loader), "epoch": epoch}
                    )
            # Get batch from training set to log
            train_batch_to_log = next(iter(train_loader))

            # Evaluate training batch
            train_batch_to_log = {
                k: v.to(device) for k, v in train_batch_to_log.items()
            }
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    _, train_preds_to_log, loss, labels = model(
                        train_batch_to_log, return_labels=True
                    )

            if cfg.visualize and epoch % cfg.visualize_interval == 0:
                impath = visualize_input_from_batch(
                    train_batch_to_log,
                    1,
                    train_preds_to_log,
                    labels,
                    folder,
                    epoch,
                    unwrap_model(model),
                    "train",
                )

                wandb.log({"vis/Train BEV": [wandb.Image(impath)], "epoch": epoch})

        avg_loss_dct = defaultdict(float)
        selected_batch = np.random.randint(0, len(val_loader))
        preds_to_log = None
        batch_to_log = None
        conf_matrix_inps = []

        for i, batch in enumerate(tqdm(val_loader, leave=False)):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                with torch.no_grad():
                    if i == selected_batch:
                        _, preds, loss, labels, preprocessed_inputs = model(
                            batch, return_labels=True, return_inputs=True
                        )
                    else:
                        _, preds, loss, pred_labels, preprocessed_inputs = model(
                            batch, return_labels=True, return_inputs=True
                        )
                        probs_gt = get_probs_and_gt(
                            preds["deinterleaved_outputs"],
                            pred_labels,
                            unwrap_model(model).quantization_offset_map,
                            unwrap_model(model).quantization_vocab_size_map,
                            [unwrap_model(model).action_quantizer],
                        )
                        conf_matrix_inps.append(probs_gt)

                loss = {k: v.mean() for k, v in loss.items()}

                if args.augmentable_preloader:
                    if "to_cache" in preprocessed_inputs and args.preload:
                        val_dataset.append_features(
                            batch, preprocessed_inputs["to_cache"]
                        )

            for k, v in loss.items():
                if args.gpus > 1 and not args.cpu and args.multi_gpu_strategy == "ddp":
                    torch.distributed.all_reduce(v)
                    v /= args.gpus

                avg_loss_dct[k] += v.item()

            if i == selected_batch:
                preds_to_log = preds
                batch_to_log = batch

        # Save the confusion matrix
        if master_process:
            for token_type in conf_matrix_inps[0].keys():
                for name in conf_matrix_inps[0][token_type].keys():
                    conf_labels = np.concatenate(
                        [x[token_type][name]["labels"] for x in conf_matrix_inps],
                        axis=0,
                    )
                    conf_pred_scores = np.concatenate(
                        [x[token_type][name]["preds"] for x in conf_matrix_inps], axis=0
                    )

                    wandb.log(
                        {
                            f"val_conf_matrix/{name}": create_confusion_matrix(
                                conf_pred_scores,
                                conf_labels,
                                conf_matrix_inps[0][token_type][name][
                                    "pred_val_labels"
                                ],
                                name,
                            ),
                            "epoch": epoch,
                        }
                    )

        # Save last model
        path = os.path.join(folder, "last_model.pt")
        save_on_master(
            {
                "model": unwrap_model(model.state_dict()),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            },
            path,
        )

        # Log all losses
        if master_process:
            for k, v in avg_loss_dct.items():
                wandb.log({"val/loss/" + k: v / len(val_loader), "epoch": epoch})

            # Log predictions
            if cfg.visualize and epoch % cfg.visualize_interval == 0:
                impath = visualize_input_from_batch(
                    batch_to_log,
                    np.random.randint(0, len(batch_to_log["action"])),
                    preds_to_log,
                    labels,
                    folder,
                    epoch,
                    unwrap_model(model),
                    "val",
                )

                wandb.log({"vis/Validation BEV": [wandb.Image(impath)], "epoch": epoch})

        # Save model with best validation loss
        if avg_loss_dct[early_stopping_criterion] / len(val_loader) < best_val_loss:
            early_stopping_counter = 0
            best_val_loss = avg_loss_dct[early_stopping_criterion] / len(val_loader)
            overall_loss = avg_loss_dct["loss"] / len(val_loader)

            path = os.path.join(folder, "best_model.pt")

            save_on_master(
                {
                    "model": unwrap_model(model.state_dict()),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "grad_scaler": grad_scaler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "overall_loss": overall_loss,
                },
                path,
            )
        else:
            early_stopping_counter += 1

        if epoch % cfg.save_every == 0 and cfg.start_saving_epoch <= epoch:
            os.makedirs(os.path.join(folder, "epochs"), exist_ok=True)
            path = os.path.join(folder, "epochs", f"epoch_{epoch}.pt")
            save_on_master(
                {
                    "model": unwrap_model(model.state_dict()),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "overall_loss": overall_loss,
                },
                path,
            )

        if master_process:
            embed_log = unwrap_model(model).embedding_counter.print_stats()

            t.set_description(
                "[EP:{:0>3d}|TL:{:.2f}|VL:{:.2f}|{}]".format(
                    epoch + 1,
                    train_epoch_loss,
                    avg_loss_dct["loss"] / len(val_loader),
                    embed_log,
                )
            )

        if args.preload:
            val_dataset.save_state()

        if early_stopping_counter >= early_stopping_patience and use_early_stopping:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
