import torch
import yaml
import torch.nn as nn

from .encoders import StoSAVi
import numpy as np
import os
from carformer.data import from_object_level_vector
from carla_birdeye_view import (
    BirdViewProducerObjectLevelRenderer,
    PixelDimensions,
    BirdViewCropType,
)


class ObjectLevelBEVEncoder(nn.Module):
    def __init__(self, config):
        super(ObjectLevelBEVEncoder, self).__init__()
        # Load config
        # If config is a string, load from file, otherwise assume it's a dict
        if isinstance(config, str):
            with open(config, "r") as file:
                try:
                    config = yaml.safe_load(file)
                    self.config = config
                except yaml.YAMLError as exc:
                    print(exc)
                    raise exc

        self.config = config

        self.multi_head = config["encoder_params"].get("multiple_heads", False)

        modules = []
        for i in range(config["model_params"]["num_layers"]):
            modules.append(
                nn.Linear(
                    (
                        config["encoder_params"]["object_dims"]
                        if i == 0
                        else config["encoder_params"]["n_embd"]
                    ),
                    config["encoder_params"]["n_embd"],
                )
            )
            modules.append(nn.GELU())

        if self.multi_head:
            modules.append(
                nn.Linear(
                    config["encoder_params"]["n_embd"],
                    config["encoder_params"]["n_embd"]
                    * config["encoder_params"]["n_object_classes"],
                )
            )
        else:
            modules.append(
                nn.Linear(
                    config["encoder_params"]["n_embd"],
                    config["encoder_params"]["n_embd"],
                )
            )

        self.object_projection = nn.Sequential(*modules)

        # Keep constant in both cases
        pred_modules = []
        for i in range(config["model_params"]["decoder_layers"]):
            pred_modules.append(
                nn.Linear(
                    config["encoder_params"]["n_embd"],
                    config["encoder_params"]["n_embd"],
                )
            )
            pred_modules.append(nn.GELU())
        pred_modules.append(
            nn.Linear(
                config["encoder_params"]["n_embd"],
                config["encoder_params"]["object_dims"],
            )
        )

        self.object_prediction = nn.Sequential(*pred_modules)

        self.object_embeddings = nn.Embedding(
            config["encoder_params"]["n_object_classes"],
            config["encoder_params"]["n_embd"],
        )

        # We use a separate embedding from the main model, so the number of classes
        # we need to allocate is just 4 dummy embeddings for the number of object classes
        self.num_classes = 4
        self.latent_dim = config["encoder_params"]["object_dims"]
        self.width = (
            config["encoder_params"]["object_level_max_route_length"]
            + config["encoder_params"]["object_level_max_num_objects"]
        )
        self.dropout = config["encoder_params"].get("object_dropout", 0.0)
        self.padding_idx = 0

        self.renderer = BirdViewProducerObjectLevelRenderer(
            target_size=PixelDimensions(
                width=400,
                height=400,
            ),
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            pixels_per_meter=5,
        )

        # Mean and std are constants from the config. Register them as buffers
        self.register_buffer(
            "mean",
            torch.tensor(config["encoder_params"]["obj_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "std",
            torch.tensor(config["encoder_params"]["obj_std"], dtype=torch.float32),
        )

    def forward(self, x, type_ids):
        return self.encode(x, type_ids)

    def apply_random_mask(self, type_ids, target_type_ids=None):
        # type_ids: BxTxN, target_type_ids: BxTxN
        # Randomly mask out some of the objects, only if training
        if not self.training:
            if target_type_ids is None:
                return type_ids
            return type_ids, target_type_ids

        mask = torch.rand(type_ids.shape) < self.dropout
        type_ids[mask] = 0
        if target_type_ids is None:
            return type_ids
        target_type_ids[mask] = -100
        return type_ids, target_type_ids

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def encode(self, x, type_ids):
        # x = self.dropout(x.reshape(-1, x.shape[-1])).reshape(*(x.shape))
        x = self.normalize(x)

        if not self.multi_head:
            return self.object_projection(x) + self.object_embeddings(type_ids)

        expanded_projection = self.object_projection(x).reshape(
            *x.shape[:-1], -1, self.config["encoder_params"]["n_embd"]
        )
        gathered_embeddings = torch.gather(
            expanded_projection,
            -2,
            type_ids.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                -1,
                -1,
                self.config["encoder_params"]["n_embd"],
            ),
        ).squeeze(-2)

        return gathered_embeddings + self.object_embeddings(type_ids)

    def decode(self, x, *args, **kwargs):
        # raise NotImplementedError("ObjectLevelBEVEncoder does not support decoding")
        return self.object_prediction(x)
        # return args

    def interpret(self, x, type_ids, denormalize=False, renderer=None):
        # x: BxTxNxD or TxNxD
        # type_ids: BxTxN or TxN
        # denormalize: bool
        if renderer is None:
            renderer = self.renderer

        if denormalize:
            x = self.denormalize(x)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            type_ids = type_ids.unsqueeze(0)

        assert len(x.shape) == 4, "x size mismatch. Expected BxTxNxD or TxNxD"
        assert len(type_ids.shape) == 3, "type_id size mismatch, Expected BxTxN or TxN"

        x = x.cpu().numpy()
        type_ids = type_ids.cpu().numpy()

        # print(x, type_ids)

        imgs = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                objects = []
                for k in range(x.shape[2]):
                    vector_rep = from_object_level_vector(
                        x[i, j, k], max(type_ids[i, j, k], 0)
                    )

                    if vector_rep is not None:
                        if not "id" in vector_rep:
                            vector_rep["id"] = k
                        objects.append(vector_rep)

                # print(objects)

                img = renderer.as_rgb(objects, color_by_id=True)  # Shape: HxWxC

                imgs.append(img)

        imgs = np.stack(imgs, axis=0)

        return imgs.reshape(x.shape[0], x.shape[1], *imgs.shape[1:])


class ObjectLevelSlotsEncoder(nn.Module):
    def __init__(self, config):
        super(ObjectLevelSlotsEncoder, self).__init__()
        # Load config
        # If config is a string, load from file, otherwise assume it's a dict
        if isinstance(config, str):
            with open(config, "r") as file:
                try:
                    config = yaml.safe_load(file)
                    self.config = config
                except yaml.YAMLError as exc:
                    print(exc)
                    raise exc

        self.config = config

        self.slot_encoder = StoSAVi(
            config["savi"]["resolution"],
            config["savi"]["clip_len"],
            config["savi"]["slot_dict"],
            config["savi"]["enc_dict"],
            config["savi"]["dec_dict"],
            config["savi"]["pred_dict"],
            config["savi"]["loss_dict"],
            config["savi"]["eps"],
        )
        # TODO: Move requires grad elsewhere
        self.slot_encoder.requires_grad_(False)

        self.slot_projection = nn.Linear(
            config["savi"]["slot_dict"]["slot_size"],
            config["encoder_params"]["n_embd"],
        )

        self.slot_prediction = nn.Linear(
            config["encoder_params"]["n_embd"],
            config["savi"]["slot_dict"]["slot_size"],
        )

        self.slot_object_level_id = config.get("slot_object_level_id", 1)

        self.test_time_context = config["savi"].get("test_time_context", 0)

        if config["savi"]["checkpoint_path"] is not None:
            self.slot_encoder.load_state_dict(
                torch.load(config["savi"]["checkpoint_path"], map_location="cpu")[
                    "state_dict"
                ],
            )

        self.slot_encoder.eval()
        self.slot_encoder.requires_grad_(False)
        self.slot_encoder.testing = True

        modules = []
        for i in range(config["model_params"]["num_layers"]):
            modules.append(
                nn.Linear(
                    (
                        config["encoder_params"]["object_dims"]
                        if i == 0
                        else config["encoder_params"]["n_embd"]
                    ),
                    config["encoder_params"]["n_embd"],
                )
            )
            modules.append(nn.GELU())

        modules.append(
            nn.Linear(
                config["encoder_params"]["n_embd"],
                config["encoder_params"]["n_embd"],
            )
        )

        self.object_projection = nn.Sequential(*modules)

        # Keep constant in both cases
        pred_modules = []
        for i in range(config["model_params"]["decoder_layers"]):
            pred_modules.append(
                nn.Linear(
                    config["encoder_params"]["n_embd"],
                    config["encoder_params"]["n_embd"],
                )
            )
            pred_modules.append(nn.GELU())
        pred_modules.append(
            nn.Linear(
                config["encoder_params"]["n_embd"],
                config["encoder_params"]["object_dims"],
            )
        )

        self.object_prediction = nn.Sequential(*pred_modules)

        # Since we are currently not using object prediction heads, we mark them as requiring no grad
        self.object_prediction.requires_grad_(False)

        # We use a separate embedding from the main model, so the number of classes
        # we need to allocate is just 4 dummy embeddings for the number of object classes
        self.num_classes = 4
        self.latent_dim = config["encoder_params"]["object_dims"]
        self.width = (
            config["encoder_params"]["object_level_max_route_length"]
            + config["savi"]["slot_dict"]["num_slots"]
        )
        self.num_slots = config["savi"]["slot_dict"]["num_slots"]
        self.dropout = config["encoder_params"].get("object_dropout", 0.0)
        self.padding_idx = 0

        self.renderer = BirdViewProducerObjectLevelRenderer(
            target_size=PixelDimensions(
                width=400,
                height=400,
            ),
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            pixels_per_meter=5,
        )

        # Mean and std are constants from the config. Register them as buffers
        self.register_buffer(
            "mean",
            torch.tensor(config["encoder_params"]["obj_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "std",
            torch.tensor(config["encoder_params"]["obj_std"], dtype=torch.float32),
        )

    def forward(
        self, slots, x, object_level_ids, slots_embeds=None, return_targets=False
    ):
        return self.encode(slots, x, object_level_ids, slots_embeds, return_targets)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def encode(
        self, slots, x, object_level_ids, slots_embeds=None, return_targets=False
    ):
        if slots_embeds is None:
            encoded_slots = self.encode_slots(slots)
        else:
            encoded_slots = slots_embeds

        T = x.shape[1]
        if encoded_slots.shape[1] > T:
            target_slots = encoded_slots[:, -T:, :, :]
            encoded_slots = encoded_slots[:, :T, :, :]
        else:
            target_slots = encoded_slots

        encoded_slots = self.slot_projection(encoded_slots)

        x = self.normalize(x)
        x = self.object_projection(x)  # B, T, N, D

        out = torch.cat((encoded_slots, x), dim=-2)

        slot_object_level_ids = torch.ones(
            *encoded_slots.shape[:-1], dtype=object_level_ids.dtype, device=x.device
        )

        out_object_level_ids = torch.cat(
            (slot_object_level_ids, object_level_ids), dim=-1
        )

        if return_targets:
            return out, out_object_level_ids, target_slots
        else:
            return out, out_object_level_ids

    def encode_slots(self, slots, return_attn=False):
        with torch.no_grad():
            self.slot_encoder.eval()
            returned_dict = self.slot_encoder(slots, return_attn=return_attn)

            encoded_slots = returned_dict["post_slots"][
                :, self.test_time_context :, :, :
            ]

            if return_attn:
                return (
                    encoded_slots,
                    returned_dict["attns"][:, self.test_time_context :, ...],
                )

            return encoded_slots

    def decode(self, x, *args, **kwargs):
        B, _, D = x.shape
        slot_latents = x.reshape(B, -1, self.width, D)[
            :, :, : self.num_slots, :
        ]  # TODO: Fix for variable number of slots
        return self.slot_prediction(slot_latents)

    def interpret_slot_latents(self, slot_latents):
        slots = self.slot_prediction(slot_latents)

        return self.interpret_slots(slots)

    def interpret_slots(self, slots, return_dict=False):
        orig_shape = slots.shape

        if len(slots.shape) == 4:
            slots = slots.flatten(0, 1).detach()

        with torch.no_grad():
            self.slot_encoder.eval()
            recons, post_recons, post_masks, _ = self.slot_encoder.decode(slots)

        recons = recons.unflatten(0, orig_shape[:-2])

        if return_dict:
            return {
                "recons": recons,
                "post_recons": post_recons,
                "post_masks": post_masks,
            }
        return recons

    def interpret(self, encoded_latents, denormalize=False):
        slots_recons = self.interpret_slot_latents(
            encoded_latents[:, :, : self.num_slots, :]
        )
