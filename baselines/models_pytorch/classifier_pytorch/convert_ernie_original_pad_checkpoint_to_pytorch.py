#!/usr/bin/env python
# encoding: utf-8
"""
File Description: https://github.com/nghuyong/ERNIE-Pytorch
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2020/7/14
"""
import collections
import os
import json
import shutil
import paddle.fluid.dygraph as D
import torch
from paddle import fluid


# downloading paddlepaddle model
# ERNIE1.0: https://ernie-github.cdn.bcebos.com/model-ernie1.0.1.tar.gz and unzip
# ERNIE-tiny: https://ernie-github.cdn.bcebos.com/model-ernie_tiny.1.tar.gz and unzip
# ERNIE2.0 https://ernie-github.cdn.bcebos.com/model-ernie2.0-en.1.tar.gz and unzip
# ERNIE large https://ernie-github.cdn.bcebos.com/model-ernie2.0-large-en.1.tar.gz and unzip

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'word_emb.weight': "bert.embeddings.word_embeddings.weight",
        'pos_emb.weight': "bert.embeddings.position_embeddings.weight",
        'sent_emb.weight': "bert.embeddings.token_type_embeddings.weight",
        'ln.weight': 'bert.embeddings.LayerNorm.gamma',
        'ln.bias': 'bert.embeddings.LayerNorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'encoder_stack.block.{i}.attn.q.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'encoder_stack.block.{i}.attn.q.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'encoder_stack.block.{i}.attn.k.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'encoder_stack.block.{i}.attn.k.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'encoder_stack.block.{i}.attn.v.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'encoder_stack.block.{i}.attn.v.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'encoder_stack.block.{i}.attn.o.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'encoder_stack.block.{i}.attn.o.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'encoder_stack.block.{i}.ln1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'encoder_stack.block.{i}.ln1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'encoder_stack.block.{i}.ffn.i.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'encoder_stack.block.{i}.ffn.i.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'encoder_stack.block.{i}.ffn.o.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'encoder_stack.block.{i}.ffn.o.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'encoder_stack.block.{i}.ln2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'encoder_stack.block.{i}.ln2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'mlm.weight': 'cls.predictions.transform.dense.weight',
            'mlm.bias': 'cls.predictions.transform.dense.bias',
            'mlm_ln.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'mlm_ln.bias': 'cls.predictions.transform.LayerNorm.beta',
            'mlm_bias': 'cls.predictions.bias'
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'ernie_config.json'), 'rt', encoding='utf-8'))
    config['layer_norm_eps'] = 1e-5
    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'wt', encoding='utf-8'), indent=4)
    print('=' * 20 + 'save vocab file' + '=' * 20)
    shutil.copyfile(os.path.join(input_dir, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))
    print('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'saved_weights'))
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'weight' in weight_name:
            if 'encoder_stack' in weight_name or 'pooler' in weight_name or 'mlm.' in weight_name:
                weight_value = weight_value.transpose()
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    extract_and_convert('./model-ernie1.0.1', './convert')