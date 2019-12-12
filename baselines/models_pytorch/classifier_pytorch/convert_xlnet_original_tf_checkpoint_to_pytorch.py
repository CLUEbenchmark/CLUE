"""Convert XLNET checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch

from transformers import (CONFIG_NAME, WEIGHTS_NAME,
                        XLNetConfig,
                        XLNetLMHeadModel,
                        load_tf_weights_in_xlnet)

import logging
logging.basicConfig(level=logging.INFO)

def convert_xlnet_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path):
    # Initialise PyTorch model
    config = XLNetConfig.from_json_file(bert_config_file)

    model = XLNetLMHeadModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)

    # Save pytorch-model
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
    print("Save PyTorch model to {}".format(os.path.abspath(pytorch_weights_dump_path)))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print("Save configuration file to {}".format(os.path.abspath(pytorch_config_dump_path)))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the TensorFlow checkpoint path.")
    parser.add_argument("--xlnet_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained XLNet model. \n"
                               "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the folder to store the PyTorch model or dataset/vocab.")
    args = parser.parse_args()

    convert_xlnet_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                        args.xlnet_config_file,
                                        args.pytorch_dump_folder_path)
