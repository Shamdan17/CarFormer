from transformers import StoppingCriteria
from transformers import LogitsProcessor
import torch
import warnings

# A set of utils for wanderer generation


# Might not be usable.
# Instead: integrate into the logits processor (Done)
class WandererFixedLengthStoppingCriteria(StoppingCriteria):
    """
    A stopping criteria for wanderer generation. It stops when the the max number of transitions is reached.
    """

    def __init__(
        self,
        stop_token_begin_idx,
        stop_token_max_idx,
        max_transitions,
        width=1,
        reduce="all",
    ):
        self.stop_token_begin_idx = stop_token_begin_idx
        self.stop_token_max_idx = stop_token_max_idx
        self.max_transitions = max_transitions
        self.width = width
        self.reduce = reduce

    def __call__(self, input_ids, scores):
        # Count the number of tokens in the input_ids that are in the range [stop_token_begin_idx, stop_token_max_idx]
        num_tokens = (
            torch.logical_and(
                input_ids >= self.stop_token_begin_idx,
                input_ids < self.stop_token_max_idx,
            )
        ).sum(axis=-1)
        # Check if all elements in the batch have reached the max number of transitions which is max_transitions * width
        mask = num_tokens >= self.max_transitions * self.width

        if self.reduce == "all":
            return mask.all()
        else:
            return mask


class WandererTransitionStateMachine:
    """
    A state machine for wanderer generation. It keeps track of the status of the current generation type (GOAL/STATE/BEV/ACTION/REWARD) and whether we should transition to the next state.
    """

    def __init__(
        self, transition_function_map, stopping_criteria=None, pad_token_id=-1
    ) -> None:
        self.transition_function_map = transition_function_map
        # Terminal state transition function
        self.stopping_criteria = stopping_criteria
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids, token_types, ignore_idx=-100):
        # Get the current state
        current_state = token_types[:, -1]
        # If the current state at an index is the pad token, find the last non-pad token for that index
        # Avoid iterating if none of the states are pad tokens
        if torch.any(input_ids[:, -1] == self.pad_token_id):
            # Display warning to user
            warnings.warn(
                "A pad token was found in the input_ids. This is probably due to one of two things:\n1) The input has not been trimmed of padding tokens.\n2) You are using a batch size larger than 1 during generation. Although this works, it can result in wasted computation due to the padding contributing to the attention operation input length.",
                UserWarning,
            )
            for i, (state, inp_id) in enumerate(zip(current_state, input_ids[:, -1])):
                if inp_id == self.pad_token_id:
                    last_non_pad_token = (
                        token_types.shape[-1]
                        - (input_ids[i, :].flip(0) == self.pad_token_id)
                        .int()
                        .argmin()
                        .item()
                        - 1
                    )

                    current_state[i] = token_types[i, last_non_pad_token]

        # print("Current state: ", current_state)

        next_state = current_state.clone()
        if self.stopping_criteria is not None:
            terminate_mask = self.stopping_criteria(input_ids, None)
        else:
            terminate_mask = torch.zeros_like(current_state, dtype=torch.bool)

        for i, state in enumerate(current_state):
            if state == ignore_idx:
                continue
            if terminate_mask[i]:
                next_state[i] = ignore_idx
                continue
            transition_function = self.transition_function_map[state.item()]
            # Get the last contiguous sequence of input ids for which the state is the same as the current state
            last_contiguous_index = (
                -(token_types[i, :].flip(0) == state).int().argmin().item()
            )
            last_contiguous_sequence = input_ids[i, last_contiguous_index:]
            # Call the transition function
            next_state[i] = transition_function(last_contiguous_sequence)

        return next_state


class WandererLogitsProcessor(LogitsProcessor):
    """
    A logits processor for wanderer generation. It constrains the next token to be in the range of the next token type.
    The next token type is determined by the state machine.
    """

    def __init__(
        self,
        token_index_mapping_dict,
        state_machine_transition_function_map,
        eos_token_id,
        pad_token_id,
        finegrained_logit_processor_map=None,
        max_transitions=1,
        transition_end_id=-1,
        transition_end_width=-1,
    ):
        self.token_index_mapping_dict = token_index_mapping_dict
        self.eos_token_id = eos_token_id
        self.finegrained_logit_processor_map = finegrained_logit_processor_map
        if max_transitions > 0:
            assert (
                transition_end_id != -1 and transition_end_width > 0
            ), "Transition end id and width must be specified if max_transitions is specified."

            self.stopping_criteria = WandererFixedLengthStoppingCriteria(
                token_index_mapping_dict[transition_end_id][0],
                token_index_mapping_dict[transition_end_id][1],
                max_transitions,
                width=transition_end_width,
                reduce="None",
            )
        else:
            self.stopping_criteria = None

        self.state_machine = WandererTransitionStateMachine(
            state_machine_transition_function_map, self.stopping_criteria, pad_token_id
        )

    def set_max_transitions(self, max_transitions):
        if self.stopping_criteria:
            self.stopping_criteria.max_transitions = max_transitions

    def __call__(self, input_ids, scores):
        # Map the input ids to the token type
        token_types = input_ids.clone()
        for token_type, token_index_list in self.token_index_mapping_dict.items():
            token_types[
                torch.logical_and(
                    input_ids >= token_index_list[0], input_ids < token_index_list[1]
                )
            ] = token_type
        # Make eos tokens have token type -100 so that they are ignored by the state machine
        token_types[input_ids == self.eos_token_id] = -100

        # Get the next state from the state machine
        next_state = self.state_machine(input_ids, token_types)

        for i, next_state_idx in enumerate(next_state):
            if next_state_idx == -100:
                # Make everything except the eos token have a probability of 0
                scores[i, : self.eos_token_id] = float("-inf")
                scores[i, self.eos_token_id + 1 :] = float("-inf")
                scores[i, self.eos_token_id] = 0
                continue

            # Otherwise, constrain the logits to be in the range of the next state
            scores[
                i, : self.token_index_mapping_dict[next_state_idx.item()][0]
            ] = float("-inf")
            scores[
                i, self.token_index_mapping_dict[next_state_idx.item()][1] :
            ] = float("-inf")

            if (
                self.finegrained_logit_processor_map is None
                or not next_state_idx.item() in self.finegrained_logit_processor_map
            ):
                continue

            # Get the finegrained logits processor for the next state, specifically for BEV currently
            logits_processor = self.finegrained_logit_processor_map[
                next_state_idx.item()
            ]

            logits_processor(input_ids[i], scores[i])

        return scores


class ForcedTransitionFunction:
    """
    A temporary transition function for wanderer generation. It always transitions to the next state.
    This is used currently as a placeholder until object level states are properly implemented with a valid EOS
    """

    def __init__(self, current_state, next_state, width):
        self.current_state = current_state
        self.next_state = next_state
        self.width = width

    def __call__(self, last_contiguous_sequence):
        return self.next_state


class FixedWidthTransitionFunction:
    """
    A transition function for wanderer generation. It ensures that the next token type is always the same as the current token type.
    """

    def __init__(self, current_state, next_state, width):
        self.current_state = current_state
        self.next_state = next_state
        self.width = width

    def __call__(self, last_contiguous_sequence):
        # last_contiguous_sequence: [seq_len]
        if last_contiguous_sequence.shape[0] >= self.width:
            return self.next_state
        else:
            return self.current_state


class WandererBEVTransitionFunction:
    """
    A transition function for wanderer generation. It simply signals a transition if the last token is an EOS token.
    """

    def __init__(self, current_state, next_state, eos_token):
        self.current_state = current_state
        self.next_state = next_state
        self.eos_token = eos_token

    def __call__(self, last_contiguous_sequence):
        # last_contiguous_sequence: [seq_len]
        if last_contiguous_sequence[-1] == self.eos_token:
            return self.next_state
        else:
            return self.current_state


class WandererBEVLogitsProcessor:
    """
    A logits processor for wanderer generation. It ensures BEV generation is always valid.
    Specifically used for variable length BEV generation as a result of tokenization.
    """

    def __init__(self, logit_offset, bev_encoder):
        self.logit_offset = logit_offset
        self.width = bev_encoder.dim_size
        self.height = bev_encoder.dim_size
        self.token_lengths = bev_encoder.token_lengths
        self.next_line_char = bev_encoder.sep_idx
        self.bev_end_char = bev_encoder.eos_idx
        self.pad_char = bev_encoder.padding_idx
        self.vocab_size = bev_encoder.num_classes

    def __call__(self, input_ids, scores):
        # Input_ids: [seq_len]
        # Scores: [vocab_size]

        # Get the index of the last EOS token in the input_ids if the input_ids contain an EOS token
        if (input_ids == self.bev_end_char).sum().item() == 0:
            last_eos_idx = -1
        else:
            last_eos_idx = (
                (input_ids == self.bev_end_char).nonzero(as_tuple=False)[-1, -1].item()
            )

        # Get the input_ids after the last EOS token that are between the logit offset and the logit offset + vocab size
        input_ids = input_ids[last_eos_idx + 1 :]

        input_ids = input_ids[
            torch.logical_and(
                input_ids >= self.logit_offset,
                input_ids < self.logit_offset + self.vocab_size,
            )
        ]

        cur_line = (input_ids == self.next_line_char).sum().item()

        # Get the index of the last next line token in the input_ids if the input_ids contain a next line token
        if cur_line == 0:
            last_next_line_idx = -1
        else:
            last_next_line_idx = (
                (input_ids == self.next_line_char)
                .nonzero(as_tuple=False)[-1, -1]
                .item()
            )

        # Get the input_ids after the last next line token that are between the logit offset and the logit offset + vocab size
        input_ids = input_ids[last_next_line_idx + 1 :]

        cur_length = 0
        for token in input_ids:
            if token == self.pad_char:
                continue
                # raise ValueError(
                #     "Padding token should not be in the input_ids. Something went wrong."
                # )
            cur_length += self.token_lengths[token.item() - self.logit_offset]

        if cur_length < self.width:
            # If the current length is less than the width, then we can add any token except the special tokens
            scores[: self.logit_offset] = float("-inf")
            scores[self.logit_offset + self.vocab_size :] = float("-inf")
            # Mask out the special tokens
            scores[self.logit_offset + self.bev_end_char] = float("-inf")
            scores[self.logit_offset + self.next_line_char] = float("-inf")
            scores[self.logit_offset + self.pad_char] = float("-inf")
        else:
            # If the current length is equal to the width, then we can only add the next line token
            scores[:] = float("-inf")
            # scores[: self.logit_offset] = float("-inf")
            # scores[self.logit_offset + self.vocab_size :] = float("-inf")
            # # Mask out the pad token
            # scores[self.logit_offset + self.pad_char] = float("-inf")
            # Mask out all voc
            # if the current line is equal to the height, then we can only add the end token
            if cur_line == self.height - 1:
                scores[self.logit_offset + self.bev_end_char] = 0
                # scores[self.logit_offset + self.next_line_char] = float("-inf")
            else:
                # scores[self.logit_offset + self.bev_end_char] = float("-inf")
                scores[self.logit_offset + self.next_line_char] = 0


class WandererObjectLevelLogitsProcessor:
    """
    A logits processor for wanderer generation. It ensures object level generation outputs are always valid.
    Specifically used for variable length object generation.
    """

    def __init__(self, logit_offset, bev_encoder):
        self.logit_offset = logit_offset
        self.width = bev_encoder.dim_size
        self.height = bev_encoder.dim_size
        self.token_lengths = bev_encoder.token_lengths
        self.next_line_char = bev_encoder.sep_idx
        self.bev_end_char = bev_encoder.eos_idx
        self.pad_char = bev_encoder.padding_idx
        self.vocab_size = bev_encoder.num_classes

    def __call__(self, input_ids, scores):
        # Input_ids: [seq_len]
        # Scores: [vocab_size]

        # Get the index of the last EOS token in the input_ids if the input_ids contain an EOS token
        if (input_ids == self.bev_end_char).sum().item() == 0:
            last_eos_idx = -1
        else:
            last_eos_idx = (
                (input_ids == self.bev_end_char).nonzero(as_tuple=False)[-1, -1].item()
            )

        # Get the input_ids after the last EOS token that are between the logit offset and the logit offset + vocab size
        input_ids = input_ids[last_eos_idx + 1 :]

        input_ids = input_ids[
            torch.logical_and(
                input_ids >= self.logit_offset,
                input_ids < self.logit_offset + self.vocab_size,
            )
        ]

        cur_line = (input_ids == self.next_line_char).sum().item()

        # Get the index of the last next line token in the input_ids if the input_ids contain a next line token
        if cur_line == 0:
            last_next_line_idx = -1
        else:
            last_next_line_idx = (
                (input_ids == self.next_line_char)
                .nonzero(as_tuple=False)[-1, -1]
                .item()
            )

        # Get the input_ids after the last next line token that are between the logit offset and the logit offset + vocab size
        input_ids = input_ids[last_next_line_idx + 1 :]

        cur_length = 0
        for token in input_ids:
            if token == self.pad_char:
                continue
                # raise ValueError(
                #     "Padding token should not be in the input_ids. Something went wrong."
                # )
            cur_length += self.token_lengths[token.item() - self.logit_offset]

        if cur_length < self.width:
            # If the current length is less than the width, then we can add any token except the special tokens
            scores[: self.logit_offset] = float("-inf")
            scores[self.logit_offset + self.vocab_size :] = float("-inf")
            # Mask out the special tokens
            scores[self.logit_offset + self.bev_end_char] = float("-inf")
            scores[self.logit_offset + self.next_line_char] = float("-inf")
            scores[self.logit_offset + self.pad_char] = float("-inf")
        else:
            # If the current length is equal to the width, then we can only add the next line token
            scores[:] = float("-inf")
            # scores[: self.logit_offset] = float("-inf")
            # scores[self.logit_offset + self.vocab_size :] = float("-inf")
            # # Mask out the pad token
            # scores[self.logit_offset + self.pad_char] = float("-inf")
            # Mask out all voc
            # if the current line is equal to the height, then we can only add the end token
            if cur_line == self.height - 1:
                scores[self.logit_offset + self.bev_end_char] = 0
                # scores[self.logit_offset + self.next_line_char] = float("-inf")
            else:
                # scores[self.logit_offset + self.bev_end_char] = float("-inf")
                scores[self.logit_offset + self.next_line_char] = 0


class AllGasNoBrakesProcessor:
    """
    A logits processor for wanderer generation. It disables the brakes and straight steering tokens
    """

    def __init__(
        self,
        logit_offset,
        brake_token_idx=-1,
        straight_steering_idx=-1,
        max_activations=-1,
    ):
        self.logit_offset = logit_offset
        self.brake_token_idx = brake_token_idx
        self.straight_steering_idx = straight_steering_idx
        self.max_activations = int(max_activations)

    def __call__(self, input_ids, scores):
        if self.max_activations != 0:
            # print("Number of activations left: ", self.max_activations)
            if self.max_activations > 0:
                self.max_activations -= 1
        else:
            return
        if self.brake_token_idx != -1:
            scores[self.logit_offset + self.brake_token_idx] = float("-inf")
        if self.straight_steering_idx != -1:
            scores[self.logit_offset + self.straight_steering_idx] = float("-inf")


class ProperActionsProcessor:
    """
    A logits processor for wanderer generation. Ensures that the actions are valid.
    Odd actions are steering actions and even actions are throttle actions.
    """

    def __init__(self, logit_offset, quantizer):
        self.logit_offset = logit_offset
        self.quantizer = quantizer
        self.vocab_size = self.quantizer.num_classes
        self.width = len(self.quantizer)

    def __call__(self, input_ids, scores):
        # Input_ids: [seq_len]
        # Scores: [vocab_size]
        # Get number of actions in the input_ids
        num_actions = (
            torch.logical_and(
                input_ids >= self.logit_offset,
                input_ids < self.logit_offset + self.vocab_size,
            )
            .sum()
            .item()
        )

        action_start_idx, action_end_idx = self.quantizer.get_dim_boundaries(
            num_actions % self.width
        )

        scores[self.logit_offset : self.logit_offset + action_start_idx] = float("-inf")
        scores[
            self.logit_offset + action_end_idx : self.logit_offset + self.vocab_size
        ] = float("-inf")

        # if num_actions % 2 == 0:
        # If the current length is even, then the next action should be a throttle action
        # scores[self.logit_offset : self.logit_offset + action_start_idx] = float(
        #     "-inf"
        # )
        # scores[
        #     self.logit_offset + action_end_idx : self.logit_offset + self.vocab_size
        # ] = float("-inf")
        # else:
        #     # If the current length is odd, then the next action should be a steering action
        #     scores[self.logit_offset : self.logit_offset + action_bin_size] = float(
        #         "-inf"
        #     )

        # Faster alternative, but not very readable
        # scores[self.logit_offset + ((num_actions + 1)% 2) * action_bin_size : self.logit_offset + ((num_actions +1)% 2 + 1) * action_bin_size] = float("-inf")
