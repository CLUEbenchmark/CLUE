# chineseGLUE_pytorch

**详细信息见于https://github.com/chineseGLUE/chineseGLUE**

## 代码目录说明

```text
├── chineseGLUEdatasets   #　存放数据
|  └── inews　　　
|  └── lcqmc　
|  └── ...
├── metrics　　　　　　　　　# metric计算
|  └── glue_compute_metrics.py　　　
├── outputs              # 模型输出保存
|  └── inews_output
|  └── lcqmc_output　
|  └── ...
├── prev_trained_model　# 预训练模型
|  └── albert_base
|  └── bert-wwm
|  └── ...
├── processors　　　　　# 数据处理
|  └── glue.py
|  └── ...
├── tools　　　　　　　　#　通用脚本
|  └── progressbar.py
|  └── ...
├── transformers　　　# 模型
|  └── modeling_albert.py
|  └── modeling_bert.py
|  └── ...
├── convert_albert_original_tf_checkpoint_to_pytorch.py　#　模型文件转换
├── run_classifier.py       # 主程序
├── run_classifier_inews.sh   #　任务运行脚本
```
### 依赖模块

- pytorch=1.1.0
- boto3=1.9
- regex
- sacremoses
- sentencepiece

### 运行

1. 若下载对应tf模型权重，则运行转换脚本，比如转换`albert_base_tf`:
```python
python convert_albert_original_tf_checkpoint_to_pytorch.py \
      --tf_checkpoint_path=./prev_trained_model/albert_base_tf \
      --bert_config_file=./prev_trained_model/albert_base_tf/albert_config_base.json \
      --pytorch_dump_path=./prev_trained_model/albert_base/pytorch_model.bin
```
**注意**: 当转换完模型之后，需要在对应的文件夹内存放`config.json`和`vocab.txt`文件

2. 直接运行对应任务sh脚本，如：

```shell
sh run_classifier_inews.sh
```
### 模型列表

```
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # xlnet_base xlnet_mid xlnet_large
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # roberta_base roberta_wwm roberta_wwm_ext roberta_wwm_large_ext
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # albert_tiny albert_base albert_large albert_xlarge
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
}
```
**注意**: bert ernie bert_wwm bert_wwwm_ext等模型只是权重不一样，而模型本身主体一样，因此参数`model_type=bert`其余同理。

## 基线结果

**说明**：

1. 目前结果大体上跟tf差不多，但是有+-0.4%上下波动，可能时由于参数不同等原因造成

2. 增加collate_fn，对每一个batch进行动态长度padding

### Tnews文本分类任务

| 模型 | 开发集(Dev) | 测试集(Test) | 训练参数 |
| :------- | :---------: | :---------: | :---------: |
| albert_tiny | 86.89 | 87.02 | epoch=5,length=128,lr=1e-4 |
| albert_base | 88.42 | 88.26 | epoch=5,length=128,lr=1e-4 |
| bert_base | 89.8 | 89.77 | epoch=4,length=128,lr=2e-5 |
| ernie_base | 89.99 | 89.90 | epoch=4,length=128,lr=2e-5 |
| xlnet_base | 89.44 | 89.59 | epoch=4,length=128,lr=2e-5 |
| bert_wwm_ext | 89.83 | 89.80 | epoch=4,length=128,lr=2e-5 |

### Lcqmc语义相似度匹配

| 模型 | 开发集(Dev) | 测试集(Test) | 训练参数 |
| :------- | :---------: | :---------: | :---------: |
| albert_base | 87.8 | 86.6 | epoch=5,length=128,lr=1e-4 |
| bert_base | 89.4 | 86.9 | epoch=4,length=128,lr=2e-5 |
| ernie_base | 89.8 | 87.1 | epoch=4,length=128,lr=2e-5 |
| bert_wwm | 89.0 | 87.2 | epoch=4,length=128,lr=2e-5 |
| bert_wwm_ext | 89.3 | 87.1 | epoch=4,length=128,lr=2e-5 |

### Inews 互联网情感分析

| 模型 | 开发集(Dev) | 测试集(Test) | 训练参数 |
| :------- | :---------: | :---------: | :---------: |
| bert_base | 85.1 | 84.5 | epoch=4,length=512,lr=2e-5 |
| ernie_base | 85.9 | 84.7 | epoch=4,length=512,lr=2e-5 |
| xlnet_base | 85.1 | 84.5 | epoch=4,length=512,lr=2e-5 |
| bert_wwm | 85.7 | 85.1 | epoch=4,length=512,lr=2e-5 |
| bert_wwm_ext | 85.4 | 85.8 | epoch=4,length=512,lr=2e-5 |
| robertta_wwm_ext | 84.5 | 84.9 | epoch=4,length=512,lr=2e-5 |




