"""Stub module for cosyvoice.dataset.processor references in YAML configs.

The YAML config files reference training-only data processing functions via
'!name:cosyvoice.dataset.processor.xxx'. These are never called during inference,
but HyperPyYAML must be able to resolve them when loading the config.

This module provides no-op stubs so the YAML loader succeeds without the full
training dataset pipeline.
"""


def parquet_opener(*args, **kwargs):
    pass


def tokenize(*args, **kwargs):
    pass


def filter(*args, **kwargs):
    pass


def resample(*args, **kwargs):
    pass


def truncate(*args, **kwargs):
    pass


def compute_fbank(*args, **kwargs):
    pass


def compute_f0(*args, **kwargs):
    pass


def parse_embedding(*args, **kwargs):
    pass


def shuffle(*args, **kwargs):
    pass


def split_speech(*args, **kwargs):
    pass


def gen_duration_target(*args, **kwargs):
    pass


def duration_length_regulator(*args, **kwargs):
    pass


def sort(*args, **kwargs):
    pass


def batch(*args, **kwargs):
    pass


def padding(*args, **kwargs):
    pass
