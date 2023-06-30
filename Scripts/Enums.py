from enum import Enum


class Pytorch(Enum):
	DEFAULT_BATCH = 1
	NORMAL_BATCH = 4


class DatasetsNames(Enum):
	TRAIN_FILE = 'HHD_AGE_TRAIN'
	VALIDATION_FILE = 'HHD_AGE_VAL'
	TEST_FILE = 'HHD_AGE_TEST'
	MODEL_NAME = 'HHD_AGE_EfficientNet B4'



