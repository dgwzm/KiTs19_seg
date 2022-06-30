import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser,ViShow
from utils.error_logger import ErrorLogger
import os
from models import get_model

def train(json_path):

    # Load options
    json_opts = json_file_to_pyobj(json_path)
    print(json_opts)
    #torch.cuda.set_device(arguments.local_rank)
    #torch.distributed.init_process_group(backend='nccl')


if __name__ == '__main__':
    json_path=r"D:\torch_keras_code\KiTs19_seg\Seg-Net\configs\kits_2d_unet.json"
    train(json_path)
