from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Training Basic Settings
# -----------------------------------------------------------------------------
_C.NB_EPOCHS = 1
_C.BATCH_SIZE = 16
_C.KFOLD = 4
_C.WORKERS = 1
_C.MULTI_PRO = False
_C.CUDA_DEVICE = "cuda:0"
_C.CUDA_VISIBLE_DEVICE = "0"

# -----------------------------------------------------------------------------
# IO Settings
# -----------------------------------------------------------------------------
_C.pretrain_path = 'D:/Data/PreTrain/imagenet_models/densenet169_weights_tf.h5'
_C.data_dir = 'D:/Data/imageprocessing/tf'
_C.csv_dir = 'D:/Data/imageprocessing/MTLmetadata.csv'
_C.OUTPUT_DIR = './result'
_C.model_name = 'D169_attention_'   # ['D169_', 'D169_attention_']

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
_C.MTL_weight = 0.5
_C.img_rows, _C.img_cols = 224, 224
_C.color_type = 3
