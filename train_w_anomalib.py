# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Patchcore, EfficientAd
from anomalib.engine import Engine
from pytorch_lightning.loggers import CSVLogger
import logging
from weed_anomaly.data_classes.weed_anomaly import WeedAnomaly
from anomalib.utils.types import TaskType

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

task = TaskType.SEGMENTATION


# Initialize the datamodule, model and engine
# datamodule = MVTec(root="/home/ronja/data/mvtec", category="carpet")
datamodule = WeedAnomaly(root="/home/ronja/data/WeedAnomaly", category="RumexWeeds", task=task, train_split_list="test.txt", val_split_list="val.txt") #, train_batch_size=1, eval_batch_size=1)
model = Patchcore(input_size=(256,256))
# model = EfficientAd()

csv_logger = CSVLogger("logs", name="patchcore")
engine = Engine(image_metrics=["AUROC"], log_every_n_steps=70, check_val_every_n_epoch=1, max_steps= 5000, logger=csv_logger, task=task)

# Train the model
engine.fit(datamodule=datamodule, model=model)