import os
import torch

# 设定 GPU 编号（如不需要可注释掉）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 删除视觉特征（你已经不需要了）
# visual_features_list = list(range(55, 91))
# VISUAL_DIM = len(visual_features_list)
VISUAL_DIM = 0  # 显式设为 0

acoustic_features_list = list(range(0, 60))  # 声学特征索引
ACOUSTIC_DIM = len(acoustic_features_list)

HCF_DIM = 4
LANGUAGE_DIM = 768

VISUAL_DIM_ALL = 91   # 虽然不使用，但模型结构中有时需保留占位
ACOUSTIC_DIM_ALL = 81

H_MERGE_SENT = 768
DATASET_LOCATION = "./dataset/"  # 确保你的 ur_funny.pkl/mustard.pkl 在这个目录
SEP_TOKEN_ID = 3