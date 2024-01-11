"""WeedAnomaly Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the WeedAnomaly dataset.

Code Reference:
Code is based on the mvtec.py of the anomalib repository

"""

import logging
from collections.abc import Sequence
from pathlib import Path
import os

import albumentations as A  # noqa: N812
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import (
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    get_transforms,
)
from anomalib.utils.types import TaskType

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES = (
    "RumexWeeds",
    "CottonCanopy",
    "AerialWheat",
)


def make_weedanomaly_dataset(
    root: str | Path,
    split: str,
    split_list: str | Path,
    extensions: Sequence[str] | None = None,

) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/label/image_filename.png

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | filename.png  | ground_truth/defect/filename_mask.png | 1           |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+

    Args:
        root (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.
    
    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

        >>> root = Path('./WeedAnomaly')
        >>> category = 'RumexWeeds'
        >>> path = root / category
        >>> path
        PosixPath('WeedAnomaly/RumexWeeds')

        >>> samples = make_weedanomaly_dataset(path, split='train')

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS
    
    with open(split_list, "r") as f:
        samples_list = f.readlines()[::-1]
        samples_list = [s.strip() for s in samples_list]

    for i, sample in enumerate(samples_list):
        sample = sample.split("/")
        samples_list[i] = (str(root), split, sample[0], sample[1])
    
    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])


    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "crop"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "crop"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)


    # assign mask paths to anomalous test images
    if os.path.isdir(str(root) + "/masks"):
        samples.loc[
            (samples.label_index == LabelName.ABNORMAL),
            "mask_path",
        ] = samples.path + "/masks/" + samples.image_path.replace(".jpg", ".png")

        samples.loc[
            (samples.label_index == LabelName.NORMAL),
            "mask_path",
        ] = ""

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.label + "/" + samples.image_path

    return samples


class WeedAnomalyDataset(AnomalibDataset):
    """WeedAnomaly dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/WeedAnomaly``.
        category (str): Sub-category of the dataset, e.g. 'RumexWeeds'
            Defaults to ``RumexWeeds``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.

    Examples:
        .. code-block:: python

            from anomalib.data.image.mvtec import MVTecDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = MVTecDataset(
                task="classification",
                transform=transform,
                root='./datasets/MVTec',
                category='zipper',
            )
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image'])

        When the task is segmentation, the dataset will also contain the mask:

        .. code-block:: python

            dataset.task = "segmentation"
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        The image is a torch tensor of shape (C, H, W) and the mask is a torch tensor of shape (H, W).

        .. code-block:: python

            print(dataset[0]["image"].shape, dataset[0]["mask"].shape)
            # Output: (torch.Size([3, 256, 256]), torch.Size([256, 256]))

    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str = "./datasets/WeedAnomaly",
        category: str = "RumexWeeds",
        split: str | Split | None = None,
        split_list: Path | str | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.split = split
        self.split_list = f"{self.root_category}/dataset_splits/{split_list}"

    def _setup(self) -> None:
        self.samples = make_weedanomaly_dataset(self.root_category, self.split, split_list=self.split_list, extensions=IMG_EXTENSIONS)


class WeedAnomaly(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/WeedAnomaly"``.
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
            Defaults to ``"RumexWeeds"``.
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to ``(256, 256)``.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
            Defaults to ``None``.
        normalization (InputNormalizationMethod | str): Normalization method to be applied to the input images.
            Defaults to ``InputNormalizationMethod.IMAGENET``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
            Defaults to ``TaskType.SEGMENTATION``.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing during training.
            Defaults to ``None``.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defualts to ``None``.

    Examples:
        To create an MVTec AD datamodule with default settings:

        >>> datamodule = MVTec()
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To change the category of the dataset:

        >>> datamodule = MVTec(category="cable")

        To change the image and batch size:

        >>> datamodule = MVTec(image_size=(512, 512), train_batch_size=16, eval_batch_size=8)

        MVTec AD dataset does not provide a validation set. If you would like
        to use a separate validation set, you can use the ``val_split_mode`` and
        ``val_split_ratio`` arguments to create a validation set.

        >>> datamodule = MVTec(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.1)

        This will subsample the test set by 10% and use it as the validation set.
        If you would like to create a validation set synthetically that would
        not change the test set, you can use the ``ValSplitMode.SYNTHETIC`` option.

        >>> datamodule = MVTec(val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.2)

    """

    def __init__(
        self,
        root: Path | str = "./datasets/WeedAnomaly",
        category: str = "RumexWeeds",
        image_size: int | tuple[int, int] = (256, 256),
        center_crop: int | tuple[int, int] | None = None,
        normalization: InputNormalizationMethod | str = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        train_split_list: str = "train.txt",
        val_split_list: str = "val.txt",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = Path(category)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = WeedAnomalyDataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            split_list=train_split_list,
            root=root,
            category=category,
        )
        self.test_data = WeedAnomalyDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            split_list=val_split_list,
            root=root,
            category=category,
        )


if __name__ == "__main__":
    root = Path("/home/ronja/data/WeedAnomaly")
    category = "RumexWeeds"
    ds = WeedAnomalyDataset(root=root, category=category, split="train", split_list="test.txt", task="classification", transform = get_transforms(image_size=256))
    ds._setup()
