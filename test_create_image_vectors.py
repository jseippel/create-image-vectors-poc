import io
from unittest.mock import MagicMock, Mock, patch

import boto3
import pandas as pd
import PIL.Image
import torch
from mypy_boto3_s3 import S3Client
from pandas.testing import assert_frame_equal
from transformers import BatchFeature, TensorType

from src.flows.create_image_vectors import (
    compute_embeddings,
    create_image_vectors,
    create_manifest_file_in_s3,
    get_image_processor,
    get_model,
    load_images_from_s3,
    preprocess,
    select_images_metadata,
)


class TestCreateImageVectors:
    @patch("src.flows.create_image_vectors.get_model")
    @patch("src.flows.create_image_vectors.get_image_processor")
    @patch("src.flows.create_image_vectors.load_images_from_s3")
    @patch("src.flows.create_image_vectors.preprocess")
    @patch("src.flows.create_image_vectors.compute_embeddings")
    def test_flow(  # noqa: PLR0913
        self,
        mock_get_model: MagicMock,
        mock_get_image_processor: MagicMock,
        mock_load_images_from_s3: MagicMock,
        mock_preprocess: MagicMock,
        mock_compute_embeddings: MagicMock,
    ) -> None:
        """It runs successfully with mocked tasks.

        This is abusing patching, but I'm leaving it with a noqa for now instead of
            taking time to figure out a better way. It should ultimately be modified to
            patch less.
        One blocker to that is getting the mock s3 to work cleanly, which may require
            altering the imported config.

        Since so much is mocked out, this mainly serves as a sanity test for the code
            gluing the tasks together within the flow and that the task runner works.

        Args:
            mock_select_images_metadata: mock task
            mock_create_manifest_file_in_s3: mock task
            mock_get_model: mock task
            mock_get_image_processor: mock task
            mock_load_images_from_s3: mock task
            mock_preprocess: mock task
            mock_compute_embeddings: mock task
        """
        create_image_vectors(
            input_image_uris=["s3://bucket/some/image.jpg", "s3://bucket/some/other/image.jpg"],
            batch_size=16
        )


class TestLoadImagesFromS3:
    def test_all_files_loaded(self, mock_s3_client: S3Client) -> None:
        """It successfully loads a given list of files that exist.

        Args:
            mock_s3_client: S3Client mocked with moto
        """
        test_bucket = "test-bucket"
        test_keys = [f"test{x}.jpg" for x in range(3)]
        mock_s3_client.create_bucket(Bucket=test_bucket)
        # preload test bucket with test image files
        for key in test_keys:
            img_bytes_io = io.BytesIO()
            PIL.Image.new("RGB", (1, 1)).save(img_bytes_io, format="jpeg")
            img_bytes = img_bytes_io.getvalue()
            mock_s3_client.put_object(Bucket=test_bucket, Key=key, Body=img_bytes)

        load_images_from_s3.fn([f"s3://{test_bucket}/{key}" for key in test_keys])


class TestPreprocess:
    def test_preprocess(self) -> None:
        """It calls the provided image processor with the provided images.

        Just verifying the passed-in processor is called appropriately.
        """
        images = [PIL.Image.new("RGB", (32, 32))]
        image_processor = Mock()
        preprocess.fn(images, image_processor)

        image_processor.assert_called_once_with(
            images=images, return_tensors=TensorType.PYTORCH
        )


class TestComputeEmbeddings:
    def test_compute_embeddings(self) -> None:
        """It calls the provided model's get_image_features with the provided batch.

        Just verifying the passed-in model is used appropriately.
        """
        batch = BatchFeature(
            data={"pixel_values": torch.ones([32, 3, 224, 224])},
            tensor_type=TensorType.PYTORCH,
        )
        model = Mock()
        compute_embeddings.fn(batch, model)

        model.get_image_features.assert_called_once_with(**batch)


class TestGetModel:
    @patch("src.flows.create_image_vectors.CLIPModel.from_pretrained")
    def test_get_model(self, mock_from_pretrained: MagicMock) -> None:
        """It uses the HuggingFace API as expected.

        Args:
            mock_from_pretrained: Mock of the HuggingFace from_pretrained function that fetches the model
        """
        test_model = "test"
        get_model.fn(test_model)

        mock_from_pretrained.assert_called_with(test_model)


class TestGetImagePreprocessor:
    @patch("src.flows.create_image_vectors.AutoImageProcessor.from_pretrained")
    def test_get_image_preprocessor(self, mock_from_pretrained: MagicMock) -> None:
        """It uses the HuggingFace API as expected.

        Args:
            mock_from_pretrained: Mock of the HuggingFace from_pretrained function that fetches the processor
        """
        test_model = "test"
        get_image_processor.fn(test_model)

        mock_from_pretrained.assert_called_with(test_model)
