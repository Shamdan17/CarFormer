import torch
import torch.nn.functional as F
import numpy as np
from carformer.backbone.gpt2 import GPT2ModelContinuous, GPT2LMHeadModel, GPT2Config
from carformer.backbone.generation_utils import (
    WandererLogitsProcessor,
    WandererBEVLogitsProcessor,
    ForcedTransitionFunction,
    FixedWidthTransitionFunction,
    WandererBEVTransitionFunction,
)
from carformer.utils import (
    interleave,
    TokenTypeIDs,
    deinterleave,
    move_padding_to_end,
    change_padding_to_ignore_index,
    normalize_angle_torch,
    get_block_masks_from_token_type_ids,
    flatten_dict,
)
from carformer.encoder import (
    ObjectLevelBEVEncoder,
    ObjectLevelSlotsEncoder,
)
from carformer.quantizer import KMeansQuantizer
from transformers import LogitsProcessorList
from carformer.utils.embedding_counter import EmbeddingCounter
from carformer.loss import focal_loss, softF1Loss
import hashlib
import os
import line_profiler


# Wanderer model
class Wanderer(torch.nn.Module):
    def __init__(self, config):
        super(Wanderer, self).__init__()
        self.cfg = config
        # TODO: Possibly decouple the backbone from the Wanderer model config better (e.g. config.backbone_config)
        self.backbone_config = config
        # Must be before GPT2 init because vocab size is set there
        self.init_model(self.cfg)
        if not config.training["quantized"]:
            self.backbone = GPT2ModelContinuous(self.backbone_config)
        else:
            self.backbone = GPT2LMHeadModel(self.backbone_config)

        if self.cfg.init_from_lm_ckp:
            from transformers import AutoModel

            init_model_dummy = AutoModel.from_pretrained(self.cfg.init_name_or_path)

            state_dict = init_model_dummy.state_dict()

            # Crop wte and wpe weights to match the current model
            state_dict["wte.weight"] = state_dict["wte.weight"][: self.cfg.vocab_size]
            state_dict["wpe.weight"] = state_dict["wpe.weight"][
                : self.cfg.max_position_embeddings
            ]

            if config.training["quantized"]:
                self.backbone.transformer.load_state_dict(state_dict, strict=False)
            else:
                self.backbone.load_state_dict(state_dict, strict=False)

        self.logits_processor = self.get_logits_processor()

        self.embedding_counter = self.get_embedding_counter()

        # self.action_loss = focal_loss(alpha=None, gamma=2.0)
        self.action_loss = F.cross_entropy

    def forward(
        self, input_dict, calculate_loss=True, return_labels=False, return_inputs=False
    ):
        backbone_inputs = self.prepare_inputs(
            input_dict, create_autoregressive_labels=calculate_loss
        )

        if calculate_loss:
            backbone_inputs, labels = backbone_inputs
        else:
            labels = None

        if self.cfg.training["quantized"]:
            self.embedding_counter(backbone_inputs["input_ids"])
        backbone_outputs = self.backbone(**backbone_inputs, output_hidden_states=True)

        if calculate_loss:
            # If we are calculating the loss, we know what the label type ids are
            label_type_ids = labels["label_type_ids"]
        else:
            if self.cfg.training["quantized"]:
                # Get label type ids from the output logits
                predictions = backbone_outputs["logits"]
                pred_tokens = torch.argmax(predictions, dim=-1)
                label_type_ids = self.get_label_type_ids_from_tokens(pred_tokens)

        deinterleaved_outputs = deinterleave(backbone_outputs["logits"], label_type_ids)

        actions_pred = None
        rewards_pred = None
        bev_pred = None
        state_pred = None
        goal_pred = None
        loss = None

        # Inputs to the decoding function is the deinterleaved outputs except for the BEV
        decoder_inputs = {
            key: value
            for key, value in deinterleaved_outputs.items()
            if key != TokenTypeIDs.BEV
        }

        if TokenTypeIDs.BEV in deinterleaved_outputs:
            if self.cfg.training["quantized"] and not self.cfg.training["object_level"]:
                bev_pred_latent = deinterleaved_outputs[TokenTypeIDs.BEV]
                bev_pred_latent = bev_pred_latent[
                    ...,
                    self.quantization_offset_map[
                        TokenTypeIDs.BEV
                    ] : self.quantization_offset_map[TokenTypeIDs.BEV]
                    + self.bev_encoder.num_classes,
                ]

                # Restrict the vocab to only EOS when label is EOS, only PAD when label is PAD, and only the quantized classes when label is neither
                # This is so that the decoder can understand the structure of the input
                if calculate_loss and self.cfg.training["tokenized_state"]:
                    quantized_label_bev = labels["quantized_label_bev"]
                    quantized_label_bev = move_padding_to_end(
                        quantized_label_bev,
                        padding_id=-100,
                        trim_padding=self.cfg.training["dynamic_batching"],
                    )
                    bev_pred_latent_shape = bev_pred_latent.shape

                    quantized_label_bev = quantized_label_bev.view(-1)
                    bev_pred_latent = bev_pred_latent.view(
                        -1, bev_pred_latent.shape[-1]
                    )

                    for token_id in [
                        self.bev_encoder.padding_idx,
                        self.bev_encoder.eos_idx,
                        self.bev_encoder.sep_idx,
                    ]:
                        label_token_id = (
                            -100
                            if token_id == self.bev_encoder.padding_idx
                            else token_id
                        )

                        indexing_mask = (quantized_label_bev == label_token_id)[
                            : bev_pred_latent.shape[0]
                        ]

                        bev_pred_latent[
                            indexing_mask,
                            :token_id,
                        ] = -torch.inf

                        bev_pred_latent[
                            indexing_mask,
                            token_id + 1 :,
                        ] = -torch.inf

                        bev_pred_latent[
                            ~indexing_mask,
                            token_id,
                        ] = -torch.inf

                    bev_pred_latent = bev_pred_latent.view(bev_pred_latent_shape)

                bev_pred_latent = torch.argmax(bev_pred_latent, dim=-1)
            else:
                if self.cfg.training["quantized"]:
                    # If quantized, we need to use the output hidden states of the last
                    # layer instead of the logits (Which will be vocabulary probabilities)
                    output_hidden_states = backbone_outputs["hidden_states"][-1]

                    input_type_ids = backbone_inputs["token_type_ids"]

                    # Deinterleave the hidden states
                    deinterleaved_hidden_states = deinterleave(
                        output_hidden_states, input_type_ids
                    )

                    bev_pred_latent = deinterleaved_hidden_states[TokenTypeIDs.BEV]
                else:
                    bev_pred_latent = self.bev_projector(
                        deinterleaved_outputs[TokenTypeIDs.BEV]
                    )

        decoder_inputs[TokenTypeIDs.BEV] = bev_pred_latent

        decoded_outputs = self.decode(decoder_inputs)

        if TokenTypeIDs.ACTION in decoded_outputs:
            actions_pred = decoded_outputs[TokenTypeIDs.ACTION]

        if TokenTypeIDs.REWARD in decoded_outputs:
            rewards_pred = decoded_outputs[TokenTypeIDs.REWARD]

        if TokenTypeIDs.BEV in decoded_outputs:
            bev_pred = decoded_outputs[TokenTypeIDs.BEV]

        if TokenTypeIDs.STATE in decoded_outputs:
            state_pred = decoded_outputs[TokenTypeIDs.STATE]

        if TokenTypeIDs.GOAL in decoded_outputs:
            goal_pred = decoded_outputs[TokenTypeIDs.GOAL]

        predictions = {
            "action": actions_pred,
            "reward": rewards_pred,
            "bev": bev_pred,
            "bev_latent": bev_pred_latent,
            "state": state_pred,
            "goal": goal_pred,
            "deinterleaved_outputs": deinterleaved_outputs,
        }

        # Handle waypoints if we have a GRU head
        # TODO: Refactor this into a separate function
        if "waypoints" in self.cfg.training["action_type"]:
            if self.cfg.training.get("waypoint_gru_head", False):
                output_hidden_states = backbone_outputs["hidden_states"][-1]
                waypoint_mask = backbone_inputs["waypoint_latent_mask"]

                waypoint_latents = output_hidden_states[
                    waypoint_mask, :
                ]  # (batch_size, num_steps, n_emdb)

                # If num_steps is 1, we need to add a dimension as it was squeezed by the indexing
                if len(waypoint_latents.shape) == 2:
                    waypoint_latents = waypoint_latents.unsqueeze(1)

                # Explicitly disable amp
                with torch.cuda.amp.autocast(False):
                    waypoint_latents = self.waypoint_head(
                        waypoint_latents
                    )  # (batch_size, num_steps, waypoint_size)

                    waypoint_latents_flags = backbone_inputs["waypoint_latent_flags"]

                    waypoint_latents_flags = waypoint_latents_flags.reshape(
                        *waypoint_latents.shape[:2], -1
                    )

                    waypoint_latents = torch.concat(
                        [waypoint_latents, waypoint_latents_flags], dim=-1
                    )

                    tps = backbone_inputs["waypoint_targetpts"].reshape(-1, 2)

                    T = waypoint_latents.shape[1]
                    waypoint_latents = waypoint_latents.reshape(
                        -1, waypoint_latents.shape[-1]
                    )
                    x = torch.zeros(
                        (*waypoint_latents.shape[:-1], 2),
                        device=waypoint_latents.device,
                        dtype=tps.dtype,
                    )

                    output_wps = []

                    for _ in range(self.cfg.training["num_waypoints"]):
                        x_in = torch.cat([x, tps], dim=-1)
                        waypoint_latents = self.waypoint_gru(x_in, waypoint_latents)
                        dx = self.waypoint_output(waypoint_latents)
                        x = x + dx
                        output_wps.append(x)

                output_wps = torch.stack(output_wps, dim=1)  # B * T x NWP x 2
                output_wps = output_wps.reshape(
                    -1, T, self.cfg.training["num_waypoints"], 2
                )
                # Subtract x offset from the waypoints
                output_wps[:, :, :, 0] -= 1.3

                predictions["waypoints"] = output_wps

        if calculate_loss:
            loss = self.calculate_loss(
                input_dict,
                backbone_inputs,
                labels,
                deinterleaved_outputs,
                predictions,
            )
            loss["loss_lm"] = backbone_outputs["loss"]

            return_values = [backbone_outputs["logits"], predictions, loss]

            if return_labels:
                return_values.append(labels)

            if return_inputs:
                return_values.append(backbone_inputs)

            return tuple(return_values)
        else:
            return backbone_outputs["logits"], predictions

    @torch.no_grad()
    def generate(self, input_dict, total_sequence_length, **kwargs):
        """
        Generate a sequence of tokens given an input dictionary.
        Args:
            input_dict (dict): Dictionary of inputs
        Returns:
            output_dict (dict): Dictionary of outputs
        """
        if self.logits_processor is None:
            self.logits_processor = self.get_logits_processor()

        self.logits_processor[0].set_max_transitions(total_sequence_length)

        self.eval()

        # Prepare inputs for the backbone model
        backbone_inputs = self.prepare_inputs(
            input_dict, create_autoregressive_labels=False
        )

        if "to_cache" in backbone_inputs:
            del backbone_inputs["to_cache"]

        if self.cfg.training["quantized"]:
            # Check that the input_ids are encountered in the embedding counter
            ood = self.embedding_counter.check_out_of_distribution(
                backbone_inputs["input_ids"]
            )
            if ood:
                print("Out of distribution token ids encountered at test time!")

            self.embedding_counter(backbone_inputs["input_ids"])
            if ood:
                # Print embedding stats
                self.embedding_counter.print_stats(verbose=True, print_counts=True)

        waypoint_latent_mask = backbone_inputs.pop("waypoint_latent_mask")
        # Set last mask index to True
        waypoint_latent_mask[:, -1] = True
        waypoint_latent_flags = backbone_inputs.pop("waypoint_latent_flags")
        waypoint_targetpts = backbone_inputs.pop("waypoint_targetpts")

        # Generate sequence of tokens
        generate_output_dict = self.backbone.generate(
            **backbone_inputs,
            logits_processor=self.logits_processor,
            max_length=1024,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **kwargs,
        )

        output_token_ids = generate_output_dict["sequences"]

        label_type_ids = self.get_label_type_ids_from_tokens(output_token_ids)

        # print("label_type_ids", label_type_ids.cpu().numpy().tolist())

        # for token_type_id in TokenTypeIDs:
        #     print(token_type_id, (label_type_ids == token_type_id).sum(-1))

        # Prepare outputs
        output_dict = deinterleave(output_token_ids, label_type_ids)

        if self.cfg.training["object_level"]:
            if not self.use_slots:
                # Replace object ids with predicted object latents
                vehicle_mask = (
                    input_dict["bevobjecttype"][input_dict["bevobjecttype"] != 0] == 1
                )

                vehicle_latents = generate_output_dict["hidden_states"][0][-1][
                    backbone_inputs["token_type_ids"] == 1, :
                ]

                vehicle_latents = vehicle_latents[vehicle_mask, :]

                output_dict[TokenTypeIDs.BEV] = vehicle_latents.unsqueeze(0)
            else:
                bev_latents = generate_output_dict["hidden_states"][0][-1][
                    backbone_inputs["token_type_ids"] == 1, :
                ]

                output_dict[TokenTypeIDs.BEV] = bev_latents.unsqueeze(0)

        output_dict = self.decode(output_dict)

        generate_output_dict["output_dict"] = output_dict

        if "waypoints" in self.cfg.training["action_type"]:
            if self.cfg.training.get("waypoint_gru_head", False):
                output_hidden_states = generate_output_dict["hidden_states"][0][-1]
                waypoint_mask = waypoint_latent_mask

                waypoint_latents = output_hidden_states[
                    waypoint_mask, :
                ]  # (batch_size, num_steps, n_emdb)

                # If num_steps is 1, we need to add a dimension as it was squeezed by the indexing
                if len(waypoint_latents.shape) == 2:
                    waypoint_latents = waypoint_latents.unsqueeze(1)

                # Explicitly disable amp
                with torch.cuda.amp.autocast(False):
                    waypoint_latents = self.waypoint_head(
                        waypoint_latents
                    )  # (batch_size, num_steps, waypoint_size)

                    waypoint_latents_flags = waypoint_latent_flags

                    waypoint_latents = torch.concat(
                        [waypoint_latents, waypoint_latents_flags], dim=-1
                    )

                    tps = waypoint_targetpts.reshape(-1, 2)

                    T = waypoint_latents.shape[1]
                    waypoint_latents = waypoint_latents.reshape(
                        -1, waypoint_latents.shape[-1]
                    )
                    x = torch.zeros(
                        (*waypoint_latents.shape[:-1], 2),
                        device=waypoint_latents.device,
                        dtype=tps.dtype,
                    )

                    output_wps = []

                    for _ in range(self.cfg.training["num_waypoints"]):
                        x_in = torch.cat([x, tps], dim=-1)
                        waypoint_latents = self.waypoint_gru(x_in, waypoint_latents)
                        dx = self.waypoint_output(waypoint_latents)
                        x = x + dx
                        output_wps.append(x)

                output_wps = torch.stack(output_wps, dim=1)  # B * T x NWP x 2
                output_wps = output_wps.reshape(
                    -1, T, self.cfg.training["num_waypoints"], 2
                )
                # Subtract x offset from the waypoints
                output_wps[:, :, :, 0] -= 1.3

                generate_output_dict["waypoints"] = output_wps

        return generate_output_dict

    @torch.no_grad()
    def generate_gru_only_optimized(self, input_dict, total_sequence_length, **kwargs):
        """
        Generate GRU actions only for faster decoding.
        Args:
            input_dict (dict): Dictionary of inputs
        Returns:
            output_dict (dict): Dictionary of outputs
        """
        if self.logits_processor is None:
            self.logits_processor = self.get_logits_processor()

        self.logits_processor[0].set_max_transitions(total_sequence_length)

        self.eval()

        # If bfloat 16 supported by GPU, use it
        # otherwise fp16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            # Prepare inputs for the backbone model
            backbone_inputs = self.prepare_inputs(
                input_dict, create_autoregressive_labels=False
            )
            results = {}
            results["backbone_inputs"] = {k: v for k, v in backbone_inputs.items()}

            if "to_cache" in backbone_inputs:
                to_cache = backbone_inputs["to_cache"]

            waypoint_latent_mask = backbone_inputs.pop("waypoint_latent_mask")
            # Set last mask index to True
            waypoint_latent_mask[:, -1] = True
            waypoint_latent_flags = backbone_inputs.pop("waypoint_latent_flags")
            waypoint_targetpts = backbone_inputs.pop("waypoint_targetpts")

            # Generate sequence of tokens
            generate_output_dict = self.backbone(
                **backbone_inputs, output_hidden_states=True
            )

            # output_token_ids = generate_output_dict["sequences"]

            # label_type_ids = self.get_label_type_ids_from_tokens(output_token_ids)

            # print("label_type_ids", label_type_ids.cpu().numpy().tolist())

            # for token_type_id in TokenTypeIDs:
            #     print(token_type_id, (label_type_ids == token_type_id).sum(-1))

            # Prepare outputs
            # output_dict = deinterleave(output_token_ids, label_type_ids)

            # if self.cfg.training["object_level"]:
            #     if not self.use_slots:
            #         # Replace object ids with predicted object latents
            #         vehicle_mask = (
            #             input_dict["bevobjecttype"][input_dict["bevobjecttype"] != 0] == 1
            #         )

            #         vehicle_latents = generate_output_dict["hidden_states"][0][-1][
            #             backbone_inputs["token_type_ids"] == 1, :
            #         ]

            #         vehicle_latents = vehicle_latents[vehicle_mask, :]

            #         output_dict[TokenTypeIDs.BEV] = vehicle_latents.unsqueeze(0)
            #     else:
            #         bev_latents = generate_output_dict["hidden_states"][0][-1][
            #             backbone_inputs["token_type_ids"] == 1, :
            #         ]

            #         output_dict[TokenTypeIDs.BEV] = bev_latents.unsqueeze(0)

            # output_dict = self.decode(output_dict)

            # generate_output_dict["output_dict"] = output_dict

            if "waypoints" in self.cfg.training["action_type"]:
                if self.cfg.training.get("waypoint_gru_head", False):
                    output_hidden_states = generate_output_dict["hidden_states"][-1]
                    waypoint_mask = waypoint_latent_mask

                    waypoint_latents = output_hidden_states[
                        waypoint_mask, :
                    ]  # (batch_size, num_steps, n_emdb)

                    # If num_steps is 1, we need to add a dimension as it was squeezed by the indexing
                    if len(waypoint_latents.shape) == 2:
                        waypoint_latents = waypoint_latents.unsqueeze(1)

                    # Explicitly disable amp
                    with torch.cuda.amp.autocast(False, dtype=dtype):
                        waypoint_latents = self.waypoint_head(
                            waypoint_latents
                        )  # (batch_size, num_steps, waypoint_size)

                        waypoint_latents_flags = waypoint_latent_flags

                        waypoint_latents = torch.concat(
                            [waypoint_latents, waypoint_latents_flags], dim=-1
                        )

                        tps = waypoint_targetpts.reshape(-1, 2)

                        T = waypoint_latents.shape[1]
                        waypoint_latents = waypoint_latents.reshape(
                            -1, waypoint_latents.shape[-1]
                        )
                        x = torch.zeros(
                            (*waypoint_latents.shape[:-1], 2),
                            device=waypoint_latents.device,
                            dtype=tps.dtype,
                        )

                        output_wps = []

                        for _ in range(self.cfg.training["num_waypoints"]):
                            x_in = torch.cat([x, tps], dim=-1)
                            waypoint_latents = self.waypoint_gru(x_in, waypoint_latents)
                            dx = self.waypoint_output(waypoint_latents)
                            x = x + dx
                            output_wps.append(x)

                    output_wps = torch.stack(output_wps, dim=1)  # B * T x NWP x 2
                    output_wps = output_wps.reshape(
                        -1, T, self.cfg.training["num_waypoints"], 2
                    )
                    # Subtract x offset from the waypoints
                    output_wps[:, :, :, 0] -= 1.3

                    results["waypoints"] = output_wps

        return results

    def decode(self, output_dict):
        """
        Decode a sequence of tokens given an output dictionary.
        Args:
            output_dict (dict): Dictionary of output token ids, indexed by token type
        Returns:
            output_dict (dict): Dictionary of output values, indexed by token type
        """

        for token_type_id, decoder in [
            (TokenTypeIDs.STATE, self.state_decoder),
            (TokenTypeIDs.GOAL, self.goal_decoder),
            (TokenTypeIDs.ACTION, self.action_decoder),
            (TokenTypeIDs.REWARD, self.reward_decoder),
            (TokenTypeIDs.BEV, self.bev_encoder.decode),
        ]:
            if token_type_id in output_dict:
                output_dict[token_type_id] = decoder(output_dict[token_type_id])

        return output_dict

    def prepare_inputs(self, input_dict, create_autoregressive_labels=False):
        """
        Prepare inputs for the backbone model.
        Args:
            input_dict (dict): Dictionary of inputs
            create_autoregressive_labels (bool): Whether to prepare labels for self-supervised autoregressive decoding or not
        Returns:
            backbone_inputs (dict): Dictionary of inputs for the backbone model
            labels (dict): Dictionary of labels for the backbone model (if create_autoregressive_labels is True)
        """
        input_dict = {k: v for k, v in input_dict.items()}

        input_dict = self.preprocess_input_dict(input_dict)

        goals = None
        if self.cfg.training["condition_on_goal"]:
            goals = input_dict["goal"]  # (batch_size, num_steps, goal_size)

        states = None
        if "state" in input_dict:
            states = input_dict["state"]  # (batch_size, num_steps, state_size)

        # if not "bev_latent" in input_dict:
        if self.cfg.training["object_level"]:
            if self.use_slots:
                # TODO: Accomodate not having route objects
                bev_latent, bev_object_level_ids, bev_targets = self.bev_encoder(
                    input_dict["bevslots"],
                    input_dict["bevobject"],
                    input_dict["bevobjecttype"],
                    input_dict.get(
                        "bevslotslatent", None
                    ),  # If we have precomputed slots latents, use them
                    return_targets=True,
                )

                input_dict["bevobjectlatent"] = bev_latent
                input_dict["bevobjecttype"] = bev_object_level_ids
                if self.cfg.training["use_future_vehicle_forcast"]:
                    input_dict["targetbevslotslatent"] = bev_targets
                input_dict["bev_latent"] = input_dict["bevobjecttype"]
            else:
                if "bevobjectlatent" in input_dict:
                    assert self.bev_encoder.dropout == 0.0, (
                        "Dropout in the BEV encoder is not supported when "
                        "using precomputed object level latents"
                    )
                else:
                    if "targetbevobjecttype" in input_dict:
                        (
                            input_dict["bevobjecttype"],
                            input_dict["targetbevobjecttype"],
                        ) = self.bev_encoder.apply_random_mask(
                            input_dict["bevobjecttype"],
                            input_dict["targetbevobjecttype"],
                        )
                    else:
                        input_dict["bevobjecttype"] = (
                            self.bev_encoder.apply_random_mask(
                                input_dict["bevobjecttype"]
                            )
                        )

                    bev_latent = self.bev_encoder(
                        input_dict["bevobject"], input_dict["bevobjecttype"]
                    )
                    input_dict["bevobjectlatent"] = bev_latent

                # Just have the type ids as latent for compatibility with the rest of the code
                input_dict["bev_latent"] = input_dict["bevobjecttype"]
        else:
            if "bev_latent" not in input_dict:
                bev = input_dict["bev"]
                bev_latent = self.bev_encoder(bev)
                if self.cfg.training["quantized"]:
                    bev_latent = (
                        bev_latent + self.quantization_offset_map[TokenTypeIDs.BEV]
                    )
                input_dict["bev_latent"] = bev_latent

                if "targetbev" in input_dict:
                    # Compute the latent for the target bev as well
                    target_bev = input_dict["targetbev"]
                    target_bev_latent = self.bev_encoder(target_bev)
                    if self.cfg.training["quantized"]:
                        target_bev_latent = target_bev_latent
                    input_dict["target_bev_latent"] = target_bev_latent

        # else:
        #     if self.cfg.training["quantized"] and not self.cfg.training["object_level"]:
        #         input_dict["bev_latent"] = (
        #             input_dict["bev_latent"]
        #             + self.quantization_offset_map[TokenTypeIDs.BEV]
        #         )

        if "bevobjectlatent" in input_dict:
            # Merge the time and width dimensions
            input_dict["bevobjectlatent"] = input_dict["bevobjectlatent"].reshape(
                input_dict["bevobjectlatent"].shape[0],  # Batch size
                -1,  # Num steps * num objects
                input_dict["bevobjectlatent"].shape[-1],  # Latent dim
            )

        bev = input_dict["bev_latent"]  # (batch_size, num_steps, bev_latent_dim)
        bev = bev.reshape(bev.shape[0], -1)  # (batch_size, num_steps*bev_latent_dim)
        actions = input_dict["action"]  # (batch_size, num_steps, action_size)
        rewards = input_dict["reward"]  # (batch_size, num_steps, reward_size)

        widths = {}
        widths["bev"] = self.bev_encoder.width

        # Encode
        # Note: This does not seem incremental decoding friendly
        #       Might need reworking or modification
        if self.cfg.training["condition_on_goal"]:
            # We will use categorical goals
            goals = self.goal_encoder(goals).to(bev.device)

            widths["goal"] = goals.shape[-1]

            # Squeeze to get rid of the possible extra dimension not removed by the embedding operation
            if len(goals.shape) == 4:
                goals = goals.squeeze(-2)  # (batch_size, num_steps, n_embd)
            # If quantized, flatten the last 2 dimensions
            if self.cfg.training["quantized"]:
                goals = goals.reshape(
                    goals.shape[0], -1
                )  # (batch_size, num_steps*n_embd)
        else:
            widths["goal"] = 1

        # Encode states
        if self.cfg.training["quantized"]:
            # Quantize the states
            states = self.state_quantizer.encode(states).to(
                bev.device
            )  # (batch_size, num_steps, state_size)
            actions = self.action_quantizer.encode(actions).to(
                bev.device
            )  # (batch_size, num_steps, action_size)
            rewards = self.reward_quantizer.encode(rewards).to(
                bev.device
            )  # (batch_size, num_steps, reward_size)
            widths["state"] = states.shape[-1]
            widths["action"] = actions.shape[-1]
            widths["reward"] = rewards.shape[-1]
            # Reshape to (batch_size, num_steps*width), add offset
            states = (
                states.reshape(states.shape[0], -1)
                + self.quantization_offset_map[TokenTypeIDs.STATE]
            )

            actions = (
                actions.reshape(actions.shape[0], -1)
                + self.quantization_offset_map[TokenTypeIDs.ACTION]
            )

            rewards = (
                rewards.reshape(rewards.shape[0], -1)
                + self.quantization_offset_map[TokenTypeIDs.REWARD]
            )

        else:
            # Encode the states
            states = self.state_encoder(states)  # (batch_size, num_steps, n_embd)
            actions = self.action_encoder(actions)  # (batch_size, num_steps, n_embd)
            rewards = self.reward_encoder(rewards)  # (batch_size, num_steps, n_embd)
            widths["state"] = 1
            widths["action"] = 1
            widths["reward"] = 1

        # Interleave the inputs to get the sequence as:
        # [goal0, state0, action0, reward0, goal1, state1, action1, reward1, ...]
        # This is the format that GPT2 expects
        # If config.goal_conditioning is "global", then the goal is not interleaved and is appended to the beginning of the sequence as a prefix
        if goals is None or self.cfg.training["goal_conditioning_type"] == "global":
            # No goal conditioning/conditioning is global (i.e. prefix)
            inputs, input_ids, attention_masks = interleave(
                (states, bev, actions, rewards),
                widths=[
                    widths["state"],
                    widths["bev"],
                    widths["action"],
                    widths["reward"],
                ],
            )

            if self.cfg.training["quantized"] and self.cfg.training["object_level"]:
                # Since only bev (object level) features are continuous, we interleave with other dummy states/actions/rewards
                dummy_emb = torch.zeros(
                    torch.zeros(
                        [1] ** (len(states.shape) + 1),
                        device=states.device,
                        dtype=input_dict["bevobjectlatent"].dtype,
                    )
                )
                bevobjectlatent_interleaved, _, _ = interleave(
                    (
                        dummy_emb.expand(*states.shape, self.backbone_config.n_embd),
                        input_dict["bevobjectlatent"],
                        dummy_emb.expand(*actions.shape, self.backbone_config.n_embd),
                        dummy_emb.expand(*rewards.shape, self.backbone_config.n_embd),
                    ),
                    widths=[
                        widths["state"],
                        widths["bevobjectlatent"],
                        widths["action"],
                        widths["reward"],
                    ],
                )
            # if self.cfg.training["object_level"] and self.cfg.fu
        elif self.cfg.training["goal_conditioning_type"] == "local":
            # Concatenate the goals to the beginning of every timestep
            inputs, input_ids, attention_masks = interleave(
                (goals, states, bev, actions, rewards),
                token_type_ids_mapping=[
                    TokenTypeIDs.GOAL,
                    TokenTypeIDs.STATE,
                    TokenTypeIDs.BEV,
                    TokenTypeIDs.ACTION,
                    TokenTypeIDs.REWARD,
                ],
                widths=[
                    widths["goal"],
                    widths["state"],
                    widths["bev"],
                    widths["action"],
                    widths["reward"],
                ],
            )
            if self.cfg.training["quantized"] and self.cfg.training["object_level"]:
                # Since only bev (object level) features are continuous, we interleave with other dummy states/actions/rewards
                dummy_emb = torch.zeros(
                    [1] * (len(states.shape) + 1),
                    device=states.device,
                    dtype=input_dict["bevobjectlatent"].dtype,
                )

                bevobjectlatent_interleaved, _, _ = interleave(
                    (
                        dummy_emb.expand(*goals.shape, self.backbone_config.n_embd),
                        dummy_emb.expand(*states.shape, self.backbone_config.n_embd),
                        input_dict["bevobjectlatent"],
                        dummy_emb.expand(*actions.shape, self.backbone_config.n_embd),
                        dummy_emb.expand(*rewards.shape, self.backbone_config.n_embd),
                    ),
                    widths=[
                        widths["goal"],
                        widths["state"],
                        widths["bev"],
                        widths["action"],
                        widths["reward"],
                    ],
                )

        else:
            raise ValueError(
                "Invalid goal_conditioning value: {}. Either disable goal conditioning or use a valid goal conditioning type".format(
                    self.cfg.training["goal_conditioning_type"]
                )
            )

        if (
            self.cfg.training["goal_conditioning_type"] == "global"
            and goals is not None
        ):
            # Add the goal to the beginning of the sequence as a prefix
            inputs = torch.cat([goals, inputs], dim=1)
            # Type ids are TokenTypeIDs.GOAL for the prefix
            # TODO: check shapes
            input_ids = torch.cat(
                [
                    torch.ones((1, goals.shape[1]), dtype=input_ids.dtype)
                    * TokenTypeIDs.GOAL,
                    input_ids,
                ],
                dim=1,
            ).to(states.device)

            if self.cfg.training["quantized"] and self.cfg.training["object_level"]:
                bevobjectlatent_interleaved = torch.cat(
                    [
                        dummy_emb.expand(*goals.shape, self.cfg.backbone_config.n_embd),
                        bevobjectlatent_interleaved,
                    ],
                    dim=1,
                )

        assert inputs.shape[1] == states.shape[1] + actions.shape[1] + rewards.shape[
            1
        ] + bev.shape[1] + (
            goals.shape[1] if goals is not None else 0
        ), "Inputs shape: {}, states shape: {}, bev shape: {}, actions shape: {}, rewards shape: {}, goals shape: {}".format(
            inputs.shape,
            states.shape,
            bev.shape,
            actions.shape,
            rewards.shape,
            goals.shape if goals is not None else None,
        )

        if self.cfg.training["quantized"] and (
            self.cfg.training["tokenized_state"] or self.cfg.training["object_level"]
        ):
            if self.cfg.training["object_level"]:
                (
                    inputs,
                    input_ids,
                    attention_masks,
                    bevobjectlatent_interleaved,
                ) = move_padding_to_end(
                    inputs,
                    token_type_ids=input_ids,
                    attention_mask=attention_masks,
                    inputs_embeds=bevobjectlatent_interleaved,
                    padding_id=self.bev_encoder.padding_idx,
                    trim_padding=self.cfg.training["dynamic_batching"],
                )
            else:
                inputs, input_ids, attention_masks = move_padding_to_end(
                    inputs,
                    token_type_ids=input_ids,
                    attention_mask=attention_masks,
                    padding_id=self.bev_encoder.padding_idx,
                    trim_padding=self.cfg.training["dynamic_batching"],
                )

        result_dict = {}
        if self.cfg.training.get("block_attn", False):
            block_attn_mask = get_block_masks_from_token_type_ids(
                input_ids,
                self.cfg.training.get("block_attn_token_type_ids", []),
            )
        else:
            block_attn_mask = None

        if "to_cache" in input_dict:
            result_dict["to_cache"] = input_dict["to_cache"]

        if create_autoregressive_labels:
            # For autoregressive decoding, we need to shift the inputs to the left by 1 to get the output
            # The first token is the goal and is not supposed to be used as a target
            result_dict["labels"] = inputs[:, 1:]
            result_dict["label_type_ids"] = input_ids[:, 1:]
            inputs = inputs[:, :-1]
            input_ids = input_ids[:, :-1]
            attention_masks = attention_masks[:, :-1]
            if block_attn_mask is not None:
                block_attn_mask = block_attn_mask[:, :-1]
            if self.cfg.training["quantized"] and self.cfg.training["object_level"]:
                result_dict["labels_embeds"] = bevobjectlatent_interleaved[:, 1:]
                bevobjectlatent_interleaved = bevobjectlatent_interleaved[:, :-1]

        result_dict["block_mask"] = block_attn_mask

        if not self.cfg.training["quantized"]:
            result_dict["inputs_embeds"] = inputs
        else:
            result_dict["input_ids"] = inputs
            if self.cfg.training["object_level"]:
                result_dict["inputs_embeds"] = bevobjectlatent_interleaved

                def merge_lambda(x):
                    return torch.logical_and(
                        x >= self.quantization_offset_map[TokenTypeIDs.BEV],
                        x
                        < self.quantization_offset_map[TokenTypeIDs.BEV]
                        + self.bev_encoder.num_classes,
                    )

                result_dict["merge_input_ids_and_embeds"] = True
                result_dict["merge_lambda"] = merge_lambda
            elif self.cfg.training.get("use_slots", False):

                def merge_lambda(x):
                    return torch.logical_and(
                        x >= self.quantization_offset_map[TokenTypeIDs.BEV],
                        x
                        < self.quantization_offset_map[TokenTypeIDs.BEV]
                        + self.bev_encoder.num_classes,
                    )

                result_dict["inputs_embeds"] = bevobjectlatent_interleaved
                result_dict["input_ids"] = input_ids
                result_dict["merge_input_ids_and_embeds"] = True
                result_dict["merge_lambda"] = merge_lambda
            # print(inputs)
            # print(inputs.shape)
        result_dict["token_type_ids"] = input_ids
        if "waypoints" in self.cfg.training["action_type"]:
            if self.cfg.training.get("waypoint_gru_head", False):
                # Create the waypoint latent mask
                # Waypoints are the outputs of the last token type id before the action
                waypoint_latent_mask = (
                    input_ids[:, 1:] == TokenTypeIDs.ACTION
                ).logical_and(input_ids[:, :-1] != TokenTypeIDs.ACTION)
                # Pad to right with False to match the shape of the inputs
                waypoint_latent_mask = F.pad(
                    waypoint_latent_mask,
                    (0, 1),
                    mode="constant",
                    value=False,
                )
                result_dict["waypoint_latent_mask"] = waypoint_latent_mask
                # Waypoint latent flags are the traffic light states
                light_idx = (
                    self.cfg.training["non_bev_state_type"].split("-").index("lights")
                )
                lights_flags = input_dict["state"][:, :, light_idx : light_idx + 1]
                result_dict["waypoint_latent_flags"] = lights_flags
                target_pts_idx = (
                    self.cfg.training["goal_type"].split("-").index("target_point")
                )
                result_dict["waypoint_targetpts"] = input_dict["goal"][
                    :, :, target_pts_idx : target_pts_idx + 2
                ]

        result_dict["attention_mask"] = attention_masks

        if create_autoregressive_labels:
            labels = {}

            labels["labels"] = result_dict["labels"]
            labels["label_type_ids"] = result_dict["label_type_ids"]

            if "waypoints" in self.cfg.training["action_type"]:
                if self.cfg.training.get("waypoint_gru_head", False):
                    waypoint_idx = (
                        self.cfg.training["action_type"].split("-").index("waypoints"),
                        self.cfg.training["action_type"].split("-").index("waypoints")
                        + 8,
                    )
                    labels["label_waypoints"] = input_dict["action"][
                        :, :, waypoint_idx[0] : waypoint_idx[1]
                    ]

            # Ground truth (not embedded labels)
            if self.cfg.training["condition_on_goal"]:
                labels["label_goal"] = input_dict["goal"][:, 1:]
                labels["label_state"] = input_dict["state"]
            else:
                labels["label_state"] = input_dict["state"][:, 1:]

            if self.cfg.training["object_level"]:
                labels["label_bevobjectlatent"] = input_dict["bevobjectlatent"]
                if self.cfg.training["use_future_vehicle_forcast"]:
                    if self.cfg.training.get("use_slots", False):
                        labels["label_bevslotslatent"] = input_dict[
                            "targetbevslotslatent"
                        ]
                    else:
                        B, T, _, D = input_dict["targetbevobject"].shape
                        (
                            labels["label_bevobjecttype"],
                            labels["label_bevobject"],
                            (labels["input_bevobject"],),
                        ) = move_padding_to_end(
                            input_dict["targetbevobjecttype"].reshape(B, -1),
                            inputs_embeds=input_dict["targetbevobject"].reshape(
                                B, -1, D
                            ),
                            other_tensors=(input_dict["bevobject"].reshape(B, -1, D),),
                            padding_id=-100,
                            trim_padding=self.cfg.training["dynamic_batching"],
                        )
            elif self.cfg.training.get("use_slots", False):
                raise NotImplementedError
            else:
                labels["label_bev"] = input_dict["bev"]
                if "target_bev_latent" in input_dict:
                    labels["label_target_bev_latent"] = input_dict["target_bev_latent"]
            labels["label_bev_latent"] = input_dict["bev_latent"]

            if self.cfg.training["quantized"]:
                if self.cfg.training["condition_on_goal"]:
                    labels["quantized_label_goal"] = goals[:, 1:]
                    labels["quantized_label_state"] = states[:, :]
                else:
                    labels["quantized_label_state"] = states[:, 1:]

                labels["quantized_label_bev"] = bev.clone()
                if self.cfg.training["tokenized_state"]:
                    change_padding_to_ignore_index(
                        labels["quantized_label_bev"],
                        padding_idx=self.bev_encoder.padding_idx,
                    )
                labels["quantized_label_action"] = actions
                labels["quantized_label_reward"] = rewards

            labels["label_action"] = input_dict["action"]
            labels["label_reward"] = input_dict["reward"]

            return result_dict, labels

        # TODO: check device
        return result_dict

    def preprocess_input_dict(self, input_dict):
        """
        Preprocess the input dictionary by performing deterministic operations that are the same everytime.
        Every run of this function with the same input_dict should return the same output_dict.
        This way, we can cache the output_dict and reuse it for multiple runs, or even delegate it to the dataloader.
        Args:
            input_dict (dict): Dictionary of inputs
        Returns:
            input_dict (dict): Dictionary of inputs
        """

        if self.cfg.training["object_level"]:
            if self.cfg.training.get("use_slots", False):
                if "bevslotslatent" not in input_dict:
                    if self.cfg.training.get("perceive_slots", False):
                        perceived_slots = self.bev_perception(
                            input_dict["bevslotspercept"]
                        )
                        # print(input_dict["bevslots"].shape)
                        # input_dict["bevslotsorig"] = input_dict["bevslots"]
                        input_dict["bevslots"] = perceived_slots
                    bev_slots_latent = self.bev_encoder.encode_slots(
                        input_dict["bevslots"]
                    )
                    input_dict["bevslotslatent"] = bev_slots_latent
                    if not "to_cache" in input_dict:
                        input_dict["to_cache"] = {}

                    input_dict["to_cache"]["bevslotslatent"] = bev_slots_latent
        else:
            if "bev_latent" not in input_dict:
                bev = input_dict["bev"]
                bev_latent = self.bev_encoder(bev)
                if self.cfg.training["quantized"]:
                    bev_latent = (
                        bev_latent + self.quantization_offset_map[TokenTypeIDs.BEV]
                    )
                input_dict["bev_latent"] = bev_latent

                if "targetbev" in input_dict:
                    # Compute the latent for the target bev as well
                    target_bev = input_dict["targetbev"]
                    target_bev_latent = self.bev_encoder(target_bev)
                    if self.cfg.training["quantized"]:
                        target_bev_latent = target_bev_latent
                    input_dict["target_bev_latent"] = target_bev_latent
                else:
                    target_bev_latent = None

                if not "to_cache" in input_dict:
                    input_dict["to_cache"] = {}
                    input_dict["to_cache"]["bev_latent"] = bev_latent
                    if target_bev_latent is not None:
                        input_dict["to_cache"]["target_bev_latent"] = target_bev_latent

        return input_dict

    def get_preprocessed_cache_parametrized_dirname(self):
        """
        Get a unique preprocessing directory name that is unique to the parameters used in preprocessing, such
        as the bev encoder and checkpoint path.
        """
        backbone_conf = self.cfg.training["encoder_backbone"]

        backbone_conf = flatten_dict(backbone_conf)

        object_level = self.cfg.training["object_level"]

        use_slots = self.cfg.training.get("use_slots", False)

        backbone_conf_str = "_".join(
            ["{}={}".format(k, v) for k, v in backbone_conf.items()]
        )

        conf_hash = hashlib.md5(
            "{}-{}-{}".format(
                backbone_conf_str,
                object_level,
                use_slots,
            ).encode("utf-8")
        ).hexdigest()

        return "{}".format(conf_hash[:10])

    def calculate_loss(
        self, input_dict, inputs, labels, deinterleaved_outputs, predictions
    ):
        # Calculate loss
        loss_dict = {}
        loss = 0

        # deinterleaved_labels = deinterleave(inputs["labels"], inputs["label_type_ids"])

        if (
            TokenTypeIDs.ACTION in deinterleaved_outputs
            and deinterleaved_outputs[TokenTypeIDs.ACTION].numel() > 0
        ):
            param_dict = (
                self.loss_params["action"]
                if "action" in self.loss_params
                else self.loss_params["default"]
            )

            if "reconstruction" in param_dict:
                # MSE Loss
                loss_dict["action_reconstruction_loss"] = F.mse_loss(
                    predictions["action"]
                    .view(labels["label_action"].shape)
                    .to(labels["label_action"].device),
                    labels["label_action"],
                )
                loss += (
                    param_dict["reconstruction"]
                    * loss_dict["action_reconstruction_loss"]
                )

            if "gru_reconstruction" in param_dict:
                # L1 Loss for GRU predictions
                loss_dict["action_gru_reconstruction_loss"] = F.l1_loss(
                    predictions["waypoints"].reshape(labels["label_waypoints"].shape),
                    labels["label_waypoints"],
                )
                loss += (
                    param_dict["gru_reconstruction"]
                    * loss_dict["action_gru_reconstruction_loss"]
                )

                # l1 loss for regular waypoint preds
                loss_dict["action_wp_reconstruction_loss"] = F.l1_loss(
                    predictions["action"]
                    .view(labels["label_action"].shape)
                    .to(labels["label_action"].device),
                    labels["label_action"],
                )

            if "latent_mse" in param_dict:
                # Not implemented
                raise NotImplementedError("Latent MSE loss not implemented for actions")

            if "classification" in param_dict:
                # Classification Loss
                action_logits = deinterleaved_outputs[TokenTypeIDs.ACTION][
                    ...,
                    self.quantization_offset_map[
                        TokenTypeIDs.ACTION
                    ] : self.quantization_offset_map[TokenTypeIDs.ACTION]
                    + self.action_quantizer.num_classes,
                ]

                # loss_dict["action_classification_loss"] = F.cross_entropy(
                #     action_logits.reshape(-1, self.action_quantizer.num_classes),
                #     labels["quantized_label_action"].reshape(-1)
                #     - self.quantization_offset_map[TokenTypeIDs.ACTION],
                # )
                # TODO: Do this better
                loss_dict["action_classification_loss"] = self.action_loss(
                    action_logits.reshape(-1, self.action_quantizer.num_classes),
                    labels["quantized_label_action"].reshape(-1)
                    - self.quantization_offset_map[TokenTypeIDs.ACTION],
                )

                loss += (
                    param_dict["classification"]
                    * loss_dict["action_classification_loss"]
                )

            if "softf1" in param_dict:
                loss_dict["action_softf1_loss"] = softF1Loss(
                    action_logits.reshape(-1, self.action_quantizer.num_classes),
                    labels["quantized_label_action"].reshape(-1)
                    - self.quantization_offset_map[TokenTypeIDs.ACTION],
                )

                loss += param_dict["softf1"] * loss_dict["action_softf1_loss"]
        if (
            TokenTypeIDs.REWARD in deinterleaved_outputs
            and deinterleaved_outputs[TokenTypeIDs.REWARD].numel() > 0
        ):
            param_dict = (
                self.loss_params["reward"]
                if "reward" in self.loss_params
                else self.loss_params["default"]
            )

            if "reconstruction" in param_dict:
                # MSE Loss
                loss_dict["reward_reconstruction_loss"] = F.mse_loss(
                    predictions["reward"], labels["label_reward"]
                )
                loss += (
                    param_dict["reconstruction"]
                    * loss_dict["reward_reconstruction_loss"]
                )

            if "latent_mse" in param_dict:
                # Not implemented
                raise NotImplementedError("Latent MSE loss not implemented for rewards")

            if "classification" in param_dict:
                # Classification Loss
                reward_logits = deinterleaved_outputs[TokenTypeIDs.REWARD][
                    ...,
                    self.quantization_offset_map[
                        TokenTypeIDs.REWARD
                    ] : self.quantization_offset_map[TokenTypeIDs.REWARD]
                    + self.reward_quantizer.num_classes,
                ]

                loss_dict["reward_classification_loss"] = F.cross_entropy(
                    reward_logits.reshape(-1, self.reward_quantizer.num_classes),
                    labels["quantized_label_reward"].reshape(-1)
                    - self.quantization_offset_map[TokenTypeIDs.REWARD],
                )
                loss += (
                    param_dict["classification"]
                    * loss_dict["reward_classification_loss"]
                )

        if (
            TokenTypeIDs.STATE in deinterleaved_outputs
            and deinterleaved_outputs[TokenTypeIDs.STATE].numel() > 0
        ):
            param_dict = (
                self.loss_params["state"]
                if "state" in self.loss_params
                else self.loss_params["default"]
            )

            if "reconstruction" in param_dict:
                # MSE Loss
                loss_dict["state_reconstruction_loss"] = F.mse_loss(
                    predictions["state"], labels["label_state"]
                )
                # loss_dict["bev_reconstruction_loss"] = F.mse_loss(
                #     predictions["bev"], labels["label_bev"]
                # )
                loss += param_dict["reconstruction"] * (
                    loss_dict["state_reconstruction_loss"]
                    # + loss_dict["bev_reconstruction_loss"]
                )

            if "latent_mse" in param_dict:
                loss_dict["state_latent_mse_loss"] = F.mse_loss(
                    predictions["bev_latent"], labels["label_bev_latent"]
                )
                loss += param_dict["latent_mse"] * loss_dict["state_latent_mse_loss"]

            if "classification" in param_dict:
                state_logits = deinterleaved_outputs[TokenTypeIDs.STATE][
                    ...,
                    self.quantization_offset_map[
                        TokenTypeIDs.STATE
                    ] : self.quantization_offset_map[TokenTypeIDs.STATE]
                    + self.state_quantizer.num_classes,
                ]

                # Categorical Loss
                loss_dict["state_classification_loss"] = F.cross_entropy(
                    state_logits.reshape(-1, self.state_quantizer.num_classes),
                    labels["quantized_label_state"].reshape(-1)
                    - self.quantization_offset_map[TokenTypeIDs.STATE],
                )
                loss += (
                    param_dict["classification"]
                    * loss_dict["state_classification_loss"]
                )

            if "forecast" in param_dict and param_dict["forecast"] is not None:
                if self.use_slots:
                    bev_logits = predictions["bev"]

                    targets = labels["label_bevslotslatent"]

                    forecast_loss = F.mse_loss(
                        bev_logits,
                        targets,
                    )

                    loss_dict["bev_slots_forecast_loss"] = forecast_loss

                    loss += (
                        param_dict["forecast"] * loss_dict["bev_slots_forecast_loss"]
                    )
                elif self.cfg.training["object_level"]:
                    bev_obj_logits = predictions["bev"]

                    input_objs = labels["input_bevobject"]
                    target_objs = labels["label_bevobject"]

                    target_offsets = self.bev_encoder.normalize(
                        input_objs
                    ) - self.bev_encoder.normalize(target_objs)

                    angle_offsets = normalize_angle_torch(input_objs - target_objs)

                    target_offsets[:, :, -2] = angle_offsets[:, :, -2]

                    # Apply tanh to preds and targets
                    # bev_obj_logits = torch.tanh(bev_obj_logits)
                    # target_offsets = torch.tanh(target_offsets)

                    target_obj_mask = labels["label_bevobjecttype"] == 1

                    # Trim T dim of logits to match input_objs
                    bev_obj_logits = bev_obj_logits[:, : input_objs.shape[1], :]

                    # Categorical Loss
                    forecast_loss = F.mse_loss(
                        bev_obj_logits,
                        target_offsets,
                        reduction="none",
                    )

                    # Mask out non-objects
                    forecast_loss = forecast_loss * target_obj_mask.unsqueeze(-1)

                    # Clip mse loss elementwise to avoid exploding gradients, max loss per attribute element
                    # is 2^2 = 4. This is uselesss since we are using tanh
                    forecast_loss = torch.clamp(forecast_loss, max=2**2)

                    # Sum over all objects
                    per_attribute_forecast_loss = forecast_loss.reshape(-1, 6).sum(
                        dim=0
                    ) / (target_obj_mask.sum() + 1e-4)

                    # # If any of the losses are nan, start ipdb debugger
                    # if torch.isnan(per_attribute_forecast_loss).any():

                    for i, attribute in enumerate(
                        ["x", "y", "extent_x", "extent_y", "yaw", "vel"]
                    ):
                        loss_dict[
                            "bev_object_forecast_loss_per_attribute/" + attribute
                        ] = per_attribute_forecast_loss[i]

                    loss_dict["bev_object_forecast_loss"] = (
                        per_attribute_forecast_loss.sum()
                    )

                    loss += (
                        param_dict["forecast"] * loss_dict["bev_object_forecast_loss"]
                    )
                else:
                    target_bev_latent = labels["label_target_bev_latent"]
                    bev_logits = deinterleaved_outputs[TokenTypeIDs.BEV][
                        ...,
                        self.quantization_offset_map[
                            TokenTypeIDs.BEV
                        ] : self.quantization_offset_map[TokenTypeIDs.BEV]
                        + self.bev_encoder.num_classes,
                    ]

                    # Categorical Loss
                    loss_dict["bev_forecast_loss"] = F.cross_entropy(
                        bev_logits.reshape(-1, self.bev_encoder.num_classes),
                        target_bev_latent.reshape(-1),
                    )

                    loss += param_dict["forecast"] * loss_dict["bev_forecast_loss"]

        if (
            TokenTypeIDs.GOAL in deinterleaved_outputs
            and deinterleaved_outputs[TokenTypeIDs.GOAL].numel() > 0
        ):
            param_dict = (
                self.loss_params["goal"]
                if "goal" in self.loss_params
                else self.loss_params["default"]
            )

            if "reconstruction" in param_dict:
                # MSE Loss
                loss_dict["goal_reconstruction_loss"] = F.mse_loss(
                    predictions["goal"], labels["label_goal"]
                )
                loss += (
                    param_dict["reconstruction"] * loss_dict["goal_reconstruction_loss"]
                )

            if "classification" in param_dict:
                # Categorical Loss
                goal_logits = deinterleaved_outputs[TokenTypeIDs.GOAL][
                    ...,
                    self.quantization_offset_map[
                        TokenTypeIDs.GOAL
                    ] : self.quantization_offset_map[TokenTypeIDs.GOAL]
                    + self.cfg.num_goal_classes,
                ]

                loss_dict["goal_classification_loss"] = F.cross_entropy(
                    goal_logits.reshape(-1, self.cfg.num_goal_classes),
                    labels["quantized_label_goal"].reshape(-1)
                    - self.quantization_offset_map[TokenTypeIDs.GOAL],
                )
                loss += (
                    param_dict["classification"] * loss_dict["goal_classification_loss"]
                )

            if "latent_mse" in param_dict:
                # Not implemented
                raise NotImplementedError("Latent MSE loss not implemented for goals")

        if (
            TokenTypeIDs.BEV in deinterleaved_outputs
            and deinterleaved_outputs[TokenTypeIDs.BEV].numel() > 0
        ):
            param_dict = (
                self.loss_params["bev"]
                if "bev" in self.loss_params
                else self.loss_params["default"]
            )

            # TODO: Check if this is correct
            if "reconstruction" in param_dict:
                # MSE Loss
                bev = predictions["bev"]
                label = (
                    labels["label_bev"]
                    if not self.cfg.training["object_level"]
                    else labels["label_bevobjectlatent"]
                )
                loss_dict["bev_reconstruction_loss"] = F.mse_loss(bev, label)
                loss += (
                    param_dict["reconstruction"] * loss_dict["bev_reconstruction_loss"]
                )

            if "latent_mse" in param_dict:
                # Not implemented
                raise NotImplementedError("Latent MSE loss not implemented for bev")

            if "classification" in param_dict:
                # Categorical Loss
                bev_logits = deinterleaved_outputs[TokenTypeIDs.BEV][
                    ...,
                    self.quantization_offset_map[
                        TokenTypeIDs.BEV
                    ] : self.quantization_offset_map[TokenTypeIDs.BEV]
                    + self.bev_encoder.num_classes,
                ]

                # Move padding to end
                label_bev_quant = labels["quantized_label_bev"]
                label_bev_quant = move_padding_to_end(
                    label_bev_quant,
                    padding_id=-100,
                    trim_padding=self.cfg.training["dynamic_batching"],
                )

                loss_dict["bev_classification_loss"] = F.cross_entropy(
                    bev_logits.reshape(-1, self.bev_encoder.num_classes),
                    (
                        label_bev_quant.reshape(-1)
                        - self.quantization_offset_map[TokenTypeIDs.BEV]
                    ),  # [:bev_logits.numel() // bev_logits.shape[-1]],
                )
                # print((labels["quantized_label_bev"].reshape(-1)- self.quantization_offset_map[TokenTypeIDs.BEV])[:bev_logits.numel() // bev_logits.shape[-1]])
                loss += (
                    param_dict["classification"] * loss_dict["bev_classification_loss"]
                )

        loss_dict["loss"] = loss
        return loss_dict

    def init_model(self, config):
        self.quantization_offset_map = {}
        self.quantization_vocab_size_map = {}
        config.quantization_vocab_size_map = self.quantization_vocab_size_map
        config.quantization_offset_map = self.quantization_offset_map
        num_quantized_vectors = 0
        self.use_slots = False

        if config.training["object_level"]:
            if config.training.get("use_slots", False):
                self.use_slots = True
                self.bev_encoder = ObjectLevelSlotsEncoder(
                    config.training["encoder_backbone"]
                )
            else:
                self.bev_encoder = ObjectLevelBEVEncoder(
                    config.training["encoder_backbone"]
                )
            # Create decoder if needed (quantized)
            # if config.training["quantized"]:
            # if self.cfg.training["use_future_vehicle_forcast"]:
            #     # self.bev_decoder = self.bev_encoder
            #     # pass
            # else:
            # self.bev_projector = lambda x: self.bev_encoder.decode(x)
            # self.bev_decoder = torch.nn.Identity()
        elif config.training["quantized"]:
            # self.bev_encoder = VQBEVEncoder(config.training["encoder_backbone"])
            if config.training["tokenized_state"]:
                self.bev_encoder = TokenizedVQBEVDecoder(
                    config.training["encoder_backbone"]
                )
            else:
                # self.bev_encoder = CompressedVQBEVEncoder(
                #     config.training["encoder_backbone"]
                # )
                self.bev_encoder = VQBEVEncoder(config.training["encoder_backbone"])
        else:
            self.bev_encoder = BEVEncoder(config.training["encoder_backbone"])

        if config.training.get("perceive_slots", False):
            self.bev_perception = SimpleBEV(config.training["perception_backbone"])
            # TODO: Move requires grad elsewhere
            self.bev_perception.requires_grad_(False)

        if config.training["quantized"]:
            # Set up quantizers
            self.action_quantizer = KMeansQuantizer.from_file(
                config.training["action_quantizer_path"]
            )
            self.reward_quantizer = KMeansQuantizer.from_file(
                config.training["reward_quantizer_path"]
            )
            self.state_quantizer = KMeansQuantizer.from_file(
                config.training["state_quantizer_path"]
            )
            for type_name in ["action", "reward", "state"]:
                attribute_names = config.training[type_name + "_type"]
                if attribute_names is not None:
                    attribute_names = attribute_names.split("-")
                    # Exclude any bev attributes
                    attribute_names = [
                        name for name in attribute_names if not "bev" in name
                    ]

                    # Expand attribute names by number of repeats
                    # If attribute name is target_point, it is expanded to target_point_x, target_point_y
                    # If attribute name is waypoints, it is expanded to waypoint_x_1, waypoint_y_1, ..., waypoint_x_4, waypoint_y_4
                    expanded_attribute_names = []
                    for name in attribute_names:
                        if name == "target_point":
                            expanded_attribute_names += [
                                name + "_x",
                                name + "_y",
                            ]
                        elif name == "waypoints":
                            expanded_attribute_names += [
                                name + "_1_x",
                                name + "_1_y",
                                name + "_2_x",
                                name + "_2_y",
                                name + "_3_x",
                                name + "_3_y",
                                name + "_4_x",
                                name + "_4_y",
                            ]
                        else:
                            expanded_attribute_names.append(name)

                    attribute_names = expanded_attribute_names

                    attribute_names = sorted(attribute_names)

                    quantizer = getattr(self, type_name + "_quantizer")
                    if getattr(quantizer, "attribute_names", None) is None:
                        # print(
                        #     "Setting attribute names for {} to {}".format(
                        #         type_name, attribute_names
                        #     )
                        # )
                        quantizer.set_attribute_names(attribute_names)

            self.quantization_offset_map[TokenTypeIDs.BEV] = num_quantized_vectors
            num_quantized_vectors += self.bev_encoder.num_classes  # 512
            self.quantization_vocab_size_map[TokenTypeIDs.BEV] = (
                self.bev_encoder.num_classes
            )

            self.quantization_offset_map[TokenTypeIDs.ACTION] = num_quantized_vectors
            num_quantized_vectors += self.action_quantizer.num_classes
            self.quantization_vocab_size_map[TokenTypeIDs.ACTION] = (
                self.action_quantizer.num_classes
            )

            self.quantization_offset_map[TokenTypeIDs.REWARD] = num_quantized_vectors
            num_quantized_vectors += self.reward_quantizer.num_classes
            self.quantization_vocab_size_map[TokenTypeIDs.REWARD] = (
                self.reward_quantizer.num_classes
            )

            self.quantization_offset_map[TokenTypeIDs.STATE] = num_quantized_vectors
            num_quantized_vectors += self.state_quantizer.num_classes
            self.quantization_vocab_size_map[TokenTypeIDs.STATE] = (
                self.state_quantizer.num_classes
            )

            self.action_decoder = self.action_quantizer.get_decoder_lambda(
                offset=self.quantization_offset_map[TokenTypeIDs.ACTION]
            )
            self.reward_decoder = self.reward_quantizer.get_decoder_lambda(
                offset=self.quantization_offset_map[TokenTypeIDs.REWARD]
            )
            self.state_decoder = self.state_quantizer.get_decoder_lambda(
                offset=self.quantization_offset_map[TokenTypeIDs.STATE]
            )

            if "waypoints" in self.cfg.training["action_type"]:
                if self.cfg.training.get("waypoint_gru_head", False):
                    self.use_waypoint_gru_head = True
                    self.waypoint_head = torch.nn.Linear(
                        self.cfg.n_embd,
                        self.cfg.training["waypoint_gru_hidden_size"],
                    )
                    self.waypoint_gru = torch.nn.GRUCell(
                        4,
                        self.cfg.training["waypoint_gru_hidden_size"] + 1,
                    )
                    self.waypoint_output = torch.nn.Linear(
                        self.cfg.training["waypoint_gru_hidden_size"] + 1, 2
                    )
                else:
                    self.use_waypoint_gru_head = False

            if self.cfg.training["condition_on_goal"]:
                self.quantization_offset_map[TokenTypeIDs.GOAL] = num_quantized_vectors

                if self.cfg.training["goal_quantizer_path"] is None:
                    num_quantized_vectors += config.training["num_goal_classes"]
                    self.quantization_vocab_size_map[TokenTypeIDs.GOAL] = (
                        config.training["num_goal_classes"]
                    )

                    def goal_encoder(x):
                        return x + self.quantization_offset_map[TokenTypeIDs.GOAL]

                    # self.goal_encoder = (
                    #     lambda x: x + self.quantization_offset_map[TokenTypeIDs.GOAL]
                    # )
                    self.goal_encoder = goal_encoder

                    def decode_goal(x):
                        # If x is long or int tensor return it as is
                        if x.dtype in [torch.long, torch.int]:
                            return x - self.quantization_offset_map[TokenTypeIDs.GOAL]
                        else:
                            return torch.argmax(
                                x[
                                    ...,
                                    self.quantization_offset_map[
                                        TokenTypeIDs.GOAL
                                    ] : self.quantization_offset_map[TokenTypeIDs.GOAL]
                                    + self.cfg.num_goal_classes,
                                ],
                                dim=-1,
                            )

                    self.goal_decoder = decode_goal
                else:
                    # Load from path like other quantizers, create encoder and decoder normally
                    self.goal_quantizer = KMeansQuantizer.from_file(
                        config.training["goal_quantizer_path"]
                    )
                    num_quantized_vectors += self.goal_quantizer.num_classes
                    self.cfg.num_goal_classes = self.goal_quantizer.num_classes
                    self.quantization_vocab_size_map[TokenTypeIDs.GOAL] = (
                        self.goal_quantizer.num_classes
                    )

                    self.goal_encoder = (
                        lambda x: self.goal_quantizer.encode(x)
                        + self.quantization_offset_map[TokenTypeIDs.GOAL]
                    )
                    self.goal_decoder = self.goal_quantizer.get_decoder_lambda(
                        offset=self.quantization_offset_map[TokenTypeIDs.GOAL]
                    )

            self.quantization_offset_map[TokenTypeIDs.EOS] = num_quantized_vectors
            self.sequence_eos_id = num_quantized_vectors

            # For generation purposes
            self.backbone_config.eos_token_id = self.sequence_eos_id
            self.backbone_config.pad_token_id = self.sequence_eos_id

            num_quantized_vectors += 1
            self.quantization_vocab_size_map[TokenTypeIDs.EOS] = 1

        else:
            self.action_encoder = torch.nn.Linear(
                config.action_size, self.backbone_config.n_embd
            )

            self.reward_encoder = torch.nn.Linear(
                config.reward_size, self.backbone_config.n_embd
            )

            # TODO: handle when no bev is included
            self.state_encoder = torch.nn.Linear(
                config.state_size + self.bev_encoder.latent_dim,
                self.backbone_config.n_embd,
            )

            self.action_decoder = torch.nn.Linear(
                self.backbone_config.n_embd, config.action_size
            )
            self.reward_decoder = torch.nn.Linear(
                self.backbone_config.n_embd, config.reward_size
            )
            self.bev_projector = torch.nn.Linear(
                self.backbone_config.n_embd, self.bev_encoder.latent_dim
            )
            self.state_decoder = torch.nn.Linear(
                self.backbone_config.n_embd, config.state_size
            )

            if self.cfg.training["condition_on_goal"]:
                # Continuous goal
                if self.cfg.training["goal_continuous"]:
                    self.goal_encoder = torch.nn.Linear(
                        config.goal_size, self.backbone_config.n_embd
                    )
                    self.goal_decoder = torch.nn.Linear(
                        self.backbone_config.n_embd, config.goal_size
                    )
                else:
                    # We will use categorical goals
                    self.goal_encoder = torch.nn.Embedding(
                        config.num_goal_classes, self.backbone_config.n_embd
                    )
                    self.goal_decoder = torch.nn.Linear(
                        self.backbone_config.n_embd, config.num_goal_classes
                    )

        if config.training["quantized"]:
            self.backbone_config.vocab_size = num_quantized_vectors
            self.logits_processor = None
        else:
            self.backbone_config.vocab_size = len(TokenTypeIDs)
            print("Set vocab size to {}".format(self.backbone_config.vocab_size))

        self.loss_params = config.training["loss_params"]

    def get_logits_processor(self):
        # If we are quantized, we need to create a custom logits processor
        if not self.cfg.training["quantized"]:
            return LogitsProcessorList([])
        if self.cfg.training["tokenized_state"]:
            # Create a BEV processor to enforce valid BEV generation
            bev_processor = WandererBEVLogitsProcessor(
                self.quantization_offset_map[TokenTypeIDs.BEV], self.bev_encoder
            )
            finegrained_logit_processor_map = {}
            finegrained_logit_processor_map[TokenTypeIDs.BEV] = bev_processor
        else:
            # TODO: Add object level BEV processor
            finegrained_logit_processor_map = {}

        token_index_mapping_dict = {}
        for token_type, token_id_begin in self.quantization_offset_map.items():
            if token_type == TokenTypeIDs.EOS:
                continue
            token_index_mapping_dict[token_type] = [
                token_id_begin,
                token_id_begin + self.quantization_vocab_size_map[token_type],
            ]

        # We always end with rewards
        transition_end_id = TokenTypeIDs.REWARD
        transition_end_width = len(self.reward_quantizer)

        # (GOAL) -> STATE -> BEV -> ACTION -> REWARD
        transition_states = {
            TokenTypeIDs.STATE: TokenTypeIDs.BEV,
            TokenTypeIDs.BEV: TokenTypeIDs.ACTION,
            TokenTypeIDs.ACTION: TokenTypeIDs.REWARD,
            TokenTypeIDs.REWARD: (
                TokenTypeIDs.GOAL
                if self.cfg.training["condition_on_goal"]
                and not self.cfg.training["goal_conditioning_type"] == "global"
                else TokenTypeIDs.STATE
            ),
        }

        if self.cfg.training["condition_on_goal"]:
            transition_states[TokenTypeIDs.GOAL] = TokenTypeIDs.STATE

        transition_functions = {
            TokenTypeIDs.STATE: FixedWidthTransitionFunction(
                TokenTypeIDs.STATE,
                transition_states[TokenTypeIDs.STATE],
                len(self.state_quantizer),
            ),
            TokenTypeIDs.ACTION: FixedWidthTransitionFunction(
                TokenTypeIDs.ACTION,
                transition_states[TokenTypeIDs.ACTION],
                len(self.action_quantizer)
                * (4 if "waypoints" in self.cfg.training["action_type"] else 1),
            ),
            TokenTypeIDs.REWARD: FixedWidthTransitionFunction(
                TokenTypeIDs.REWARD,
                transition_states[TokenTypeIDs.REWARD],
                len(self.reward_quantizer),
            ),
        }

        if self.cfg.training["tokenized_state"]:
            transition_functions[TokenTypeIDs.BEV] = WandererBEVTransitionFunction(
                TokenTypeIDs.BEV,
                transition_states[TokenTypeIDs.BEV],
                self.bev_encoder.eos_idx,
            )
        elif self.cfg.training["object_level"]:
            transition_functions[TokenTypeIDs.BEV] = ForcedTransitionFunction(
                TokenTypeIDs.BEV,
                transition_states[TokenTypeIDs.BEV],
                self.bev_encoder.width,
            )
        else:
            transition_functions[TokenTypeIDs.BEV] = FixedWidthTransitionFunction(
                TokenTypeIDs.BEV,
                transition_states[TokenTypeIDs.BEV],
                self.bev_encoder.width,
            )

        if self.cfg.training["condition_on_goal"]:
            if self.cfg.training["goal_conditioning_type"] == "global":
                # TODO: I will not handle multi class goal because it should not be predicted to begin with
                transition_functions[TokenTypeIDs.GOAL] = FixedWidthTransitionFunction(
                    TokenTypeIDs.GOAL,
                    transition_states[TokenTypeIDs.GOAL],
                    1,
                )
            else:
                transition_functions[TokenTypeIDs.GOAL] = FixedWidthTransitionFunction(
                    TokenTypeIDs.GOAL,
                    transition_states[TokenTypeIDs.GOAL],
                    1,
                )

        # Need to wrap in a list because the default is a single processor
        return LogitsProcessorList(
            [
                WandererLogitsProcessor(
                    token_index_mapping_dict,
                    transition_functions,
                    self.sequence_eos_id,
                    self.bev_encoder.padding_idx,
                    finegrained_logit_processor_map,
                    transition_end_id=transition_end_id,
                    transition_end_width=transition_end_width,
                )
            ]
        )

    def get_label_type_ids_from_tokens(self, tokens):
        assert self.cfg.training["quantized"]

        return self.backbone.get_label_type_ids_from_tokens(tokens)
        # label_type_ids = tokens.clone()
        # for token_type, token_id_begin in self.quantization_offset_map.items():
        #     token_id_end = token_id_begin + self.quantization_vocab_size_map[token_type]
        #     label_type_ids[
        #         torch.logical_and(tokens >= token_id_begin, tokens < token_id_end)
        #     ] = token_type

        # return label_type_ids

    def get_embedding_counter(self):
        if self.cfg.training["quantized"]:
            return EmbeddingCounter(embedding_layer=self.backbone.transformer.wte)
        else:
            return EmbeddingCounter(embedding_layer=self.backbone.wte)

    @staticmethod
    def from_pretrained(path, epoch="best"):
        assert os.path.exists(path), "Path {} does not exist".format(path)

        config_path = os.path.join(path, "config.json")

        assert os.path.exists(config_path), "Config path {} does not exist".format(
            config_path
        )

        config = GPT2Config.from_pretrained(config_path)

        # Fix quantizer paths if needed
        import inspect, carformer

        carformer_path = os.path.dirname(inspect.getfile(carformer))
        # Go one layer up
        carformer_path = os.path.dirname(carformer_path)

        config.training["action_quantizer_path"] = os.path.join(
            carformer_path, config.training["action_quantizer_path_rel"]
        )
        config.training["goal_quantizer_path"] = os.path.join(
            carformer_path, config.training["goal_quantizer_path_rel"]
        )
        config.training["state_quantizer_path"] = os.path.join(
            carformer_path, config.training["state_quantizer_path_rel"]
        )
        config.training["reward_quantizer_path"] = os.path.join(
            carformer_path, config.training["reward_quantizer_path_rel"]
        )

        if "savi" in config.training["encoder_backbone"]:
            # It will already be in the checkpoint, do not load
            config.training["encoder_backbone"]["savi"]["checkpoint_path"] = None

        if "encoder_params" in config.training:
            config.training["encoder_backbone"]["encoder_params"][
                "checkpoint_path"
            ] = None

        model = Wanderer(config)

        if epoch in ["last", "best"]:
            checkpoint_path = os.path.join(path, "{}_model.pt".format(epoch))
        else:
            checkpoint_path = os.path.join(path, "epochs/epoch_{}.pt".format(epoch))

        checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
        # Legacy compatibility
        keys = list(checkpoint.keys())

        for k in keys:
            if "module." in k:
                checkpoint[k.replace("module.", "")] = checkpoint.pop(k)

        model.load_state_dict(checkpoint)

        return model
