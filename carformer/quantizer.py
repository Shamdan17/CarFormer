from sklearn.cluster import KMeans
import numpy as np
import torch

# TODO: Go back to importing this
from carformer.utils import to_numpy, convert_numpy_to_type
import sys
import os


sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# A generic quantizer. Two arguments are required:
# num_classes: the number of classes to quantize to
# quantization_level: the level of quantization. "whole" means the entire input is quantized to a single index. "dim" means each dimension is quantized to a single index.
class Quantizer:
    """
    A generic quantizer. Two arguments are required:

    ...
    Attributes
    ----------
    num_classes : int
        The number of classes to quantize to.
    quantization_level : str
        The level of quantization. "whole" means the entire input is quantized to a single index. "dim" means each dimension is quantized to a single index.

    Methods
    -------
    train(iterable)
        Train the quantizer on the given iterable.
    encode(x)
        Encode the given input.
    decode(x)
        Decode the given input. Typically maps to centroids.
    """

    def __init__(
        self,
        num_classes,
        quantization_level="whole",
        quantization_sizes=None,
        attribute_names=None,
    ):
        self.num_classes = num_classes
        self.quantization_level = quantization_level
        self.quantization_sizes = quantization_sizes
        if quantization_sizes is not None:
            assert (
                sum(quantization_sizes) == num_classes
            ), "Sum of quantization sizes must equal num_classes"
            self.quantization_sizes_cumsum = np.cumsum(quantization_sizes)
        else:
            self.quantization_sizes_cumsum = None

        self.attribute_names = attribute_names

    def train(self, iterable):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError


class KMeansQuantizer(Quantizer):
    """
    A k-means quantizer. Two arguments are required:

    ...
    Attributes
    ----------
    num_classes : int
        The number of classes to quantize to.
    quantization_level : str
        The level of quantization. "whole" means the entire input is quantized to a single index. "dim" means each dimension is quantized to a single index.
    method : str
        The method to use for kmeans. Can be "offline" or "online"
        "offline" means the entire dataset is used to train the kmeans model
        "online" means the kmeans model is trained on a single batch at a time, not currently implemented

    Methods
    -------
    train(iterable)
        Train the quantizer on the given iterable.
    encode(x)
        Encode the given input.
    decode(x)
        Decode the given input. Typically maps to centroids.
    """

    def __init__(
        self,
        num_classes,
        quantization_level="whole",
        quantization_sizes=None,
        attribute_names=None,
        method="offline",
        disable_looping=False,
    ):
        super().__init__(
            num_classes, quantization_level, quantization_sizes, attribute_names
        )
        if method == "online":
            raise NotImplementedError

        self.set_attribute_names(attribute_names)

        self.method = method
        self.disable_looping = disable_looping

    def train(self, iterable):
        data = np.concatenate([np.array(x) for x in iterable], axis=0)
        data = data.reshape(-1, data.shape[-1])

        if self.quantization_level == "whole":
            self.kmeans = KMeans(n_clusters=self.num_classes)
            self.kmeans.fit(data)
        elif self.quantization_level == "dim":
            # num_classes should be divisble by data.shape[1]
            if self.quantization_sizes is None:
                assert (
                    self.num_classes % data.shape[1] == 0
                ), f"num_classes {self.num_classes} must be divisible by data dimension {data.shape[1]}, data shape: {data.shape}"

                self.kmeans = [
                    KMeans(n_clusters=self.num_classes // data.shape[1])
                    for _ in range(data.shape[1])
                ]
            else:
                assert (
                    len(self.quantization_sizes) == data.shape[1]
                ), f"quantization_sizes must be of length {data.shape[1]}, got {len(self.quantization_sizes)}"

                self.kmeans = [
                    KMeans(n_clusters=self.quantization_sizes[i])
                    for i in range(data.shape[1])
                ]

            for i in range(data.shape[1]):
                self.kmeans[i].fit(data[:, i].reshape(-1, 1))

            self.concatenated_centroids = np.concatenate(
                [kmeans.cluster_centers_ for kmeans in self.kmeans], axis=0
            )

    def encode(self, x):
        x = to_numpy(x)
        if self.quantization_level == "whole":
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            # Return early if x is empty
            if x.size == 0:
                return torch.zeros(original_shape[:-1], dtype=torch.long)
            indices = torch.tensor(self.kmeans.predict(x))
            return indices.reshape(original_shape[:-1])
        elif self.quantization_level == "dim":
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            indices = torch.zeros(x.shape, dtype=torch.long)
            # Return early if x is empty
            if indices.numel() == 0:
                return indices.reshape(original_shape)

            for i in range(original_shape[-1]):
                if i > len(self.kmeans) and self.disable_looping:
                    raise ValueError(
                        f"Dimension {i} is greater than number of kmeans models {len(self.kmeans)}"
                    )

                kmeans_idx = i % len(self.kmeans)

                quantization = self.kmeans[kmeans_idx].predict(x[:, i].reshape(-1, 1))
                if self.quantization_sizes_cumsum is not None:
                    indices[:, i] = torch.tensor(quantization) + (
                        self.quantization_sizes_cumsum[kmeans_idx - 1]
                        if kmeans_idx > 0
                        else 0
                    )
                else:
                    indices[:, i] = torch.tensor(
                        quantization
                    ) + kmeans_idx * self.num_classes // len(self.kmeans)

            return indices.reshape(original_shape)
        else:
            raise ValueError("Invalid quantization level")

    def decode(self, x):
        x = to_numpy(x, dtype=int)
        if self.quantization_level == "whole":
            return self.kmeans.cluster_centers_[x]
        elif self.quantization_level == "dim":
            return self.concatenated_centroids[x].squeeze(-1)
        else:
            raise ValueError("Invalid quantization level")

    def decode_with_offset(self, x, offset):
        return self.decode(x - offset)

    # Dim width getter
    def get_dim_width(self, dim):
        if self.quantization_level == "whole":
            return self.kmeans.n_clusters
        elif self.quantization_level == "dim":
            if not self.disable_looping:
                return self.kmeans[dim % len(self.kmeans)].n_clusters
            return self.kmeans[dim].n_clusters

    # Get dim boundaries
    def get_dim_boundaries(self, dim):
        if self.quantization_level == "whole":
            return (0, self.kmeans.n_clusters)
        elif self.quantization_level == "dim":
            if not self.disable_looping:
                dim = dim % len(self.kmeans)
            if self.quantization_sizes_cumsum is not None:
                return (
                    self.quantization_sizes_cumsum[dim - 1] if dim > 0 else 0,
                    self.quantization_sizes_cumsum[dim],
                )
            else:
                return (
                    dim * self.num_classes // len(self.kmeans),
                    (dim + 1) * self.num_classes // len(self.kmeans),
                )

    def set_attribute_names(self, attribute_names):
        # If none, return early
        if attribute_names is None:
            self.attribute_names = None
            return

        # If string, convert to list
        if isinstance(attribute_names, str):
            attribute_names = [attribute_names]

        self.attribute_names = attribute_names

    def get_attribute_name(self, dim, return_none_if_none=False):
        if self.attribute_names is None:
            return None if return_none_if_none else f"dim{dim}"
        else:
            return self.attribute_names[dim]

    def validate_attribute_names(self):
        if self.quantization_level == "whole":
            # Make sure attribute_names is a string or None. A size 1 list is also allowed
            assert (
                self.attribute_names is None
                or isinstance(self.attribute_names, str)
                or (
                    len(self.attribute_names) == 1
                    and isinstance(self.attribute_names[0], str)
                    and isinstance(self.attribute_names, list)
                )
            ), "attribute_names must be one string or None if quantization_level is whole"
        elif self.quantization_level == "dim":
            # Make sure attribute_names is a list of strings or None
            assert self.attribute_names is None or isinstance(
                self.attribute_names, list
            ), "attribute_names must be a list of strings or None if quantization_level is dim"
            if self.attribute_names is not None and self.quantization_sizes is not None:
                assert len(self.attribute_names) == len(
                    self.quantization_sizes
                ), "attribute_names must be of length quantization_sizes if both are not None"

    def get_decoder_lambda(self, offset=0, return_type="torch"):
        def argmax(x):
            # If x is a long tensor, return it minus the offset
            if x.dtype == torch.long or x.dtype == torch.int:
                return x - offset
            # Argmax of offset:offset + num_classes
            x = x[..., offset : offset + self.num_classes]
            return torch.argmax(x, dim=-1)

        return lambda x: convert_numpy_to_type(self.decode(argmax(x)), return_type)

    # If quantization level is whole, sort by distance to origin
    # If quantization level is dim, sort by value in ascending order for each dimension
    def get_centroid_sorting_indices(self, dim=None):
        if self.quantization_level == "whole":
            return np.argsort(np.linalg.norm(self.kmeans.cluster_centers_, axis=-1))
        elif self.quantization_level == "dim":
            if dim is None:
                raise ValueError("Must specify dim for dim quantization level")
            if not self.disable_looping:
                dim = dim % len(self.kmeans)
            return np.argsort(self.kmeans[dim].cluster_centers_.squeeze(-1))

    def get_centroids(self, dim=None):
        if self.quantization_level == "whole":
            return self.kmeans.cluster_centers_
        elif self.quantization_level == "dim":
            if dim is None:
                return self.concatenated_centroids
            else:
                return self.kmeans[dim].cluster_centers_.squeeze(-1)
        else:
            raise ValueError("Invalid quantization level")

    # Static method to load object from .npy
    @staticmethod
    def from_file(path, verbose=False):
        obj = np.load(path, allow_pickle=True).item()
        # if "concatenated_centoids" is an attribute, rename to concatenated_centroids
        if hasattr(obj, "concatenated_centoids"):
            obj.concatenated_centroids = obj.concatenated_centoids
        if not hasattr(obj, "quantization_sizes"):
            if verbose:
                print(
                    "WARNING: quantization_sizes not found in quantizer, setting to None"
                )
            obj.quantization_sizes = None
        if not hasattr(obj, "quantization_sizes_cumsum"):
            if verbose:
                print(
                    "WARNING: quantization_sizes_cumsum not found in quantizer, setting to None"
                )
            obj.quantization_sizes_cumsum = None
        if not hasattr(obj, "attribute_names"):
            if verbose:
                print(
                    "WARNING: attribute_names not found in quantizer, setting to None"
                )
            obj.set_attribute_names(None)
        # Backwards compatibility
        if not hasattr(obj, "disable_looping"):
            if verbose:
                print(
                    "WARNING: disable_looping not found in quantizer, setting to False"
                )
            obj.disable_looping = False
        return obj

    # Save object to .npy
    def save(self, path):
        np.save(path, self)

    # Length of the decoder
    def __len__(self):
        if self.quantization_level == "whole":
            return 1
        elif self.quantization_level == "dim":
            return len(self.kmeans)


class KMeansDummyQuantizer(Quantizer):
    """
    A k-means dummy quantizer. Quantizes everything to 0 and decodes everything to 0. Two arguments are required:

    ...
    Attributes
    ----------
    num_classes : int
        The number of classes to quantize to.
    quantization_level : str
        The level of quantization. "whole" means the entire input is quantized to a single index. "dim" means each dimension is quantized to a single index.
    method : str
        The method to use for kmeans. Can be "offline" or "online"
        "offline" means the entire dataset is used to train the kmeans model
        "online" means the kmeans model is trained on a single batch at a time, not currently implemented

    Methods
    -------
    train(iterable)
        Train the quantizer on the given iterable.
    encode(x)
        Encode the given input.
    decode(x)
        Decode the given input. Typically maps to centroids.
    """

    def __init__(
        self,
        num_classes,
        quantization_level="whole",
        quantization_sizes=None,
        attribute_names=None,
        method="offline",
        disable_looping=False,
    ):
        super().__init__(
            num_classes, quantization_level, quantization_sizes, attribute_names
        )
        if method == "online":
            raise NotImplementedError

        self.set_attribute_names(attribute_names)

        self.method = method
        self.disable_looping = disable_looping
        self.prints_left = 5

    def train(self, iterable):
        data = np.concatenate([np.array(x) for x in iterable], axis=0)
        data = data.reshape(-1, data.shape[-1])

        if self.quantization_level == "whole":
            self.kmeans = KMeans(n_clusters=self.num_classes)
            self.kmeans.fit(data)
        elif self.quantization_level == "dim":
            # num_classes should be divisble by data.shape[1]
            if self.quantization_sizes is None:
                assert (
                    self.num_classes % data.shape[1] == 0
                ), f"num_classes {self.num_classes} must be divisible by data dimension {data.shape[1]}, data shape: {data.shape}"

                self.kmeans = [
                    KMeans(n_clusters=self.num_classes // data.shape[1])
                    for _ in range(data.shape[1])
                ]
            else:
                assert (
                    len(self.quantization_sizes) == data.shape[1]
                ), f"quantization_sizes must be of length {data.shape[1]}, got {len(self.quantization_sizes)}"

                self.kmeans = [
                    KMeans(n_clusters=self.quantization_sizes[i])
                    for i in range(data.shape[1])
                ]

            for i in range(data.shape[1]):
                self.kmeans[i].fit(data[:, i].reshape(-1, 1))

            self.concatenated_centroids = np.concatenate(
                [kmeans.cluster_centers_ for kmeans in self.kmeans], axis=0
            )

    def encode(self, x):
        x = to_numpy(x)
        if self.quantization_level == "whole":
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            # Return early
            if self.prints_left > 0:
                print(
                    f"WARNING: Using dummy quantizer. Returning zeros. {self.prints_left} prints left."
                )
                print("Input shape:", x.shape)
                print("Input: ", x)
                print("Quantized shape: ", original_shape[:-1])
                print("Quantized: ", torch.zeros(original_shape[:-1], dtype=torch.long))
                self.prints_left -= 1

            return torch.zeros(original_shape[:-1], dtype=torch.long)

        elif self.quantization_level == "dim":
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            indices = torch.zeros(x.shape, dtype=torch.long)
            # Return early if x is empty
            if self.prints_left > 0:
                print(
                    f"WARNING: Using dummy quantizer. Returning zeros. {self.prints_left} prints left."
                )
                print("Input shape:", x.shape)
                print("Input: ", x)
                print("Quantized shape: ", original_shape)
                print("Quantized: ", torch.zeros(original_shape, dtype=torch.long))
                self.prints_left -= 1
            return indices.reshape(original_shape)
        else:
            raise ValueError("Invalid quantization level")

    def decode(self, x):
        x = to_numpy(x, dtype=int)
        if self.quantization_level == "whole":
            return self.kmeans.cluster_centers_[x]
        elif self.quantization_level == "dim":
            return self.concatenated_centroids[x].squeeze(-1)
        else:
            raise ValueError("Invalid quantization level")

    def decode_with_offset(self, x, offset):
        return self.decode(x - offset)

    # Dim width getter
    def get_dim_width(self, dim):
        if self.quantization_level == "whole":
            return self.kmeans.n_clusters
        elif self.quantization_level == "dim":
            if not self.disable_looping:
                return self.kmeans[dim % len(self.kmeans)].n_clusters
            return self.kmeans[dim].n_clusters

    # Get dim boundaries
    def get_dim_boundaries(self, dim):
        if self.quantization_level == "whole":
            return (0, self.kmeans.n_clusters)
        elif self.quantization_level == "dim":
            if not self.disable_looping:
                dim = dim % len(self.kmeans)
            if self.quantization_sizes_cumsum is not None:
                return (
                    self.quantization_sizes_cumsum[dim - 1] if dim > 0 else 0,
                    self.quantization_sizes_cumsum[dim],
                )
            else:
                return (
                    dim * self.num_classes // len(self.kmeans),
                    (dim + 1) * self.num_classes // len(self.kmeans),
                )

    def set_attribute_names(self, attribute_names):
        # If none, return early
        if attribute_names is None:
            self.attribute_names = None
            return

        # If string, convert to list
        if isinstance(attribute_names, str):
            attribute_names = [attribute_names]

        self.attribute_names = attribute_names

    def get_attribute_name(self, dim, return_none_if_none=False):
        if self.attribute_names is None:
            return None if return_none_if_none else f"dim{dim}"
        else:
            return self.attribute_names[dim]

    def validate_attribute_names(self):
        if self.quantization_level == "whole":
            # Make sure attribute_names is a string or None. A size 1 list is also allowed
            assert (
                self.attribute_names is None
                or isinstance(self.attribute_names, str)
                or (
                    len(self.attribute_names) == 1
                    and isinstance(self.attribute_names[0], str)
                    and isinstance(self.attribute_names, list)
                )
            ), "attribute_names must be one string or None if quantization_level is whole"
        elif self.quantization_level == "dim":
            # Make sure attribute_names is a list of strings or None
            assert self.attribute_names is None or isinstance(
                self.attribute_names, list
            ), "attribute_names must be a list of strings or None if quantization_level is dim"
            if self.attribute_names is not None and self.quantization_sizes is not None:
                assert len(self.attribute_names) == len(
                    self.quantization_sizes
                ), "attribute_names must be of length quantization_sizes if both are not None"

    def get_decoder_lambda(self, offset=0, return_type="torch"):
        def argmax(x):
            # If x is a long tensor, return it minus the offset
            if x.dtype == torch.long or x.dtype == torch.int:
                return x - offset
            # Argmax of offset:offset + num_classes
            x = x[..., offset : offset + self.num_classes]
            return torch.argmax(x, dim=-1)

        return lambda x: convert_numpy_to_type(self.decode(argmax(x)), return_type)

    # If quantization level is whole, sort by distance to origin
    # If quantization level is dim, sort by value in ascending order for each dimension
    def get_centroid_sorting_indices(self, dim=None):
        if self.quantization_level == "whole":
            return np.argsort(np.linalg.norm(self.kmeans.cluster_centers_, axis=-1))
        elif self.quantization_level == "dim":
            if dim is None:
                raise ValueError("Must specify dim for dim quantization level")
            if not self.disable_looping:
                dim = dim % len(self.kmeans)
            return np.argsort(self.kmeans[dim].cluster_centers_.squeeze(-1))

    def get_centroids(self, dim=None):
        if self.quantization_level == "whole":
            return self.kmeans.cluster_centers_
        elif self.quantization_level == "dim":
            if dim is None:
                return self.concatenated_centroids
            else:
                return self.kmeans[dim].cluster_centers_.squeeze(-1)
        else:
            raise ValueError("Invalid quantization level")

    # Static method to load object from .npy
    @staticmethod
    def from_file(path, verbose=False):
        obj = np.load(path, allow_pickle=True).item()

        # if "concatenated_centoids" is an attribute, rename to concatenated_centroids
        if hasattr(obj, "concatenated_centoids"):
            obj.concatenated_centroids = obj.concatenated_centoids
        if not hasattr(obj, "quantization_sizes"):
            if verbose:
                print(
                    "WARNING: quantization_sizes not found in quantizer, setting to None"
                )
            obj.quantization_sizes = None
        if not hasattr(obj, "quantization_sizes_cumsum"):
            if verbose:
                print(
                    "WARNING: quantization_sizes_cumsum not found in quantizer, setting to None"
                )
            obj.quantization_sizes_cumsum = None
        if not hasattr(obj, "attribute_names"):
            if verbose:
                print(
                    "WARNING: attribute_names not found in quantizer, setting to None"
                )
            obj.set_attribute_names(None)
        # Backwards compatibility
        if not hasattr(obj, "disable_looping"):
            if verbose:
                print(
                    "WARNING: disable_looping not found in quantizer, setting to False"
                )
            obj.disable_looping = False
        if not hasattr(obj, "prints_left"):
            if verbose:
                print("WARNING: prints_left not found in quantizer, setting to 5")

            obj.prints_left = 5

        dmmy = KMeansDummyQuantizer(
            obj.num_classes,
            obj.quantization_level,
            obj.quantization_sizes,
            obj.attribute_names,
            obj.method,
            obj.disable_looping,
        )
        dmmy.kmeans = obj.kmeans
        dmmy.concatenated_centroids = obj.concatenated_centroids
        dmmy.prints_left = obj.prints_left
        dmmy.quantization_sizes_cumsum = obj.quantization_sizes_cumsum
        obj = dmmy

        return obj

    # Save object to .npy
    def save(self, path):
        np.save(path, self)

    # Length of the decoder
    def __len__(self):
        if self.quantization_level == "whole":
            return 1
        elif self.quantization_level == "dim":
            return len(self.kmeans)
