# A wrapper around a dataset
# First, it loads batches, 1 instance at a time, then writes it into a disk cache
# Future calls to load batches will load from the cache instead of the original dataset
# This is to avoid the overhead of loading from disk every time

from skit import DatasetPreloader, InMemoryDatasetPreloader, AugmentableDatasetPreloader
