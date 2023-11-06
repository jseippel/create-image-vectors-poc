"""Prefect Flow for creating image vectors."""
import os
from collections.abc import Generator
from typing import Any, cast

import boto3
import pandas as pd
import PIL.Image
import torch
from PIL import Image
from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner
from project_package import T, config, constants
from s3fs import S3FileSystem
from transformers import (
    AutoImageProcessor,
    BatchFeature,
    CLIPImageProcessor,
    CLIPModel,
    PreTrainedModel,
    TensorType,
)


@task
def load_images_from_s3(paths: list[str]) -> list[PIL.Image.Image]:
    """Load a batch of image files from S3 as PIL Images.

    Performance could be greatly improved with multithreading.

    Args:
        paths: List of filenames to load, as S3 URIs

    Returns:
        a list of PIL images
    """
    s3fs = S3FileSystem()

    images = []
    for path in paths:
        with s3fs.open(path, "rb") as f:
            images.append(Image.open(f).copy())

    return images


@task
def preprocess(
    images: list[PIL.Image.Image], image_processor: CLIPImageProcessor
) -> BatchFeature:
    """Preprocess a batch of images with the given processor.

    Preprocessing may include resizing, center cropping, normalization, etc.
    See: https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/clip#transformers.CLIPImageProcessor

    Also works given a single image.

    Args:
        images: One or more images to preprocess
        image_processor: A HuggingFace image processor. We're using CLIP specifically,
            but this should be easily modifiable if needed.

    Returns:
        A batch of type torch.FloatTensor
    """
    return image_processor(images=images, return_tensors=TensorType.PYTORCH)


@task
def compute_embeddings(batch: BatchFeature, model: CLIPModel) -> torch.FloatTensor:
    """Compute the embeddings for a batch of input.

    Args:
        batch: A batch of preprocessed images as pytorch tensors
        model: The model to use for prediction

    Returns:
        Pytorch tensor of embeddings.
    """
    with torch.no_grad():
        return model.get_image_features(**batch)


@task
def get_model(model_name: str | os.PathLike) -> PreTrainedModel:
    """Get the model to use for embedding.

    This gets its own task to help with efficiency of Dask graph, avoiding moving the
    large model among workers repeatedly.

    Args:
        model_name: Name of pretrained model in HuggingFace model hub, or path to a
            locally-stored model

    Returns:
        A pretrained model to use for embedding.
    """
    return CLIPModel.from_pretrained(model_name)


@task
def get_image_processor(model_name: str | os.PathLike) -> AutoImageProcessor:
    """Get the image processor to use for preprocessing.

    During inference, we should match the preprocessing used during training.

    Args:
        model_name: Name of pretrained model in HuggingFace model hub, or path to a
            locally-stored model

    Returns:
        An image processor to use for preprocessing.
    """
    return AutoImageProcessor.from_pretrained(model_name)


@flow(task_runner=DaskTaskRunner())
def create_image_vectors(
    input_image_uris: list[str],
    batch_size: int = 16,
    profile: str | None = None,
) -> None:
    """Create vector embeddings of images and store them in S3.

    Args:
        input_image_uris: List os S3 URIs of images to process
        batch_size: Number of images to process per end-to-end batch
        profile: The name of a locally-configured AWS profile, if you would like to
            provide credentials that way during development.
    """
    boto3_session = boto3.Session(profile_name=profile)

    model_name = constants.IMAGE_EMBEDDING_BASE_MODEL_NAME
    model = get_model.submit(model_name)
    image_processor: CLIPImageProcessor = get_image_processor.submit(model_name)

    def chunker(seq: list, size: int) -> Generator[T | list[T], Any, None]:
        # candidate for extraction as general util function
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    image_batches = [
        load_images_from_s3.submit(uris_batch)
        for uris_batch in chunker(input_image_uris, 16)
    ]
    tensor_batches = [
        preprocess.submit(image_batch, image_processor)
        for image_batch in image_batches
    ]
    embeddings = [
        compute_embeddings.submit(tensor_batch, model)
        for tensor_batch in tensor_batches
    ]

    return embeddings


if __name__ == "__main__":
    create_image_vectors(
        input_image_uris=[
            "s3://bucket/some/image.jpg",
            "s3://bucket/some/other/image.jpg",
        ],
        batch_size=16,
    )
