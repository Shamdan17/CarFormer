# A pytorch module with buffers that counts the number of times each embedding is used

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCounter(nn.Module):
    def __init__(self, embedding_size=None, embedding_layer=None):
        super().__init__()
        if embedding_layer is not None:
            embedding_size = embedding_layer.weight.shape[0]
        self.register_buffer("training_embedding_counts", torch.zeros(embedding_size))
        self.register_buffer("validation_embedding_counts", torch.zeros(embedding_size))
        self.embedding_layer = embedding_layer

    # x is a batch of indices
    # Must count them because adding by indexing results in unexpected behavior
    def forward(self, x):
        if self.training:
            self.training_embedding_counts += torch.bincount(
                x.flatten(), minlength=self.training_embedding_counts.shape[0]
            )
        else:
            self.validation_embedding_counts += torch.bincount(
                x.flatten(), minlength=self.validation_embedding_counts.shape[0]
            )

    def get_training_embedding_counts(self):
        return self.training_embedding_counts

    def get_validation_embedding_counts(self):
        return self.validation_embedding_counts

    def reset_training_embedding_counts(self):
        self.training_embedding_counts.zero_()

    def reset_validation_embedding_counts(self):
        self.validation_embedding_counts.zero_()

    def reset(self):
        self.reset_training_embedding_counts()
        self.reset_validation_embedding_counts()

    def check_out_of_distribution(self, x=None):
        if self.training:
            print(
                "Warning: checking out of distribution in training mode, returning False"
            )
            return False
        else:
            # If x is not none, check if the bin count of x is non-zero at any index where the training bin count is zero
            if x is not None:
                return torch.any(
                    torch.bincount(
                        x.flatten(), minlength=self.validation_embedding_counts.shape[0]
                    )
                    * (self.training_embedding_counts == 0)
                )
            else:
                # If x is none, check if the validation bin count is non-zero at any index where the training bin count is zero
                return torch.any(
                    self.validation_embedding_counts
                    * (self.training_embedding_counts == 0)
                )

    def print_stats(
        self, verbose=False, print_counts=False, offset_map=None, quant_vocab_size=None
    ):
        training_unused_counts = torch.sum(self.training_embedding_counts == 0).item()
        val_unused_counts = torch.sum(self.validation_embedding_counts == 0).item()
        train_but_not_val_cnts = torch.sum(
            (self.training_embedding_counts > 0)
            * (self.validation_embedding_counts == 0)
        ).item()
        val_but_not_train_cnts = torch.sum(
            (self.training_embedding_counts == 0)
            * (self.validation_embedding_counts > 0)
        ).item()

        if print_counts:
            if verbose:
                print(
                    "Training embedding counts: \n",
                    self.training_embedding_counts.clone().detach().cpu().numpy(),
                )
                print(
                    "Validation embedding counts: \n",
                    self.validation_embedding_counts.clone().detach().cpu().numpy(),
                )

                if not (offset_map is None or quant_vocab_size is None):
                    for k in offset_map:
                        size = quant_vocab_size[k]
                        offset = offset_map[k]
                        print(
                            "Training embedding counts for {}: \n".format(k),
                            self.training_embedding_counts[offset : offset + size]
                            .clone()
                            .detach()
                            .cpu()
                            .numpy(),
                        )
                        print(
                            "Validation embedding counts for {}: \n".format(k),
                            self.validation_embedding_counts[offset : offset + size]
                            .clone()
                            .detach()
                            .cpu()
                            .numpy(),
                        )

            # Number unused embeddings in training
            print(
                "Number unused embeddings in training: ",
                training_unused_counts,
            )
            # Number unused embeddings in validation
            print(
                "Number unused embeddings in validation: ",
                val_unused_counts,
            )

            # Number of embeddings used in training but not validation
            print(
                "Number of embeddings used in training but not validation: ",
                train_but_not_val_cnts,
            )
            # Number of embeddings used in validation but not training
            print(
                "Number of embeddings used in validation but not training: ",
                val_but_not_train_cnts,
            )

        # Formatted string of the 4 metrics above
        result = "[T:{}|V:{}|T/V:{}|V/T:{}]".format(
            training_unused_counts,
            val_unused_counts,
            train_but_not_val_cnts,
            val_but_not_train_cnts,
        )

        return result
