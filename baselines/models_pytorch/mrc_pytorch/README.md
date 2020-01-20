# CLUE MRC pytorch

**详细信息见于https://github.com/chineseGLUE/chineseGLUE**

## 代码目录说明

```text
├── mrc_data   #　存放数据
|  └── CHID　　　
|  └── DRCD　
|  └── CMRC2018
|  └── ...
├── preprocess          # 预处理和评测
├── check_points        # 预训练模型和模型输出保存
|  └── prev_trained_model
|  └── CHID　　　
|  └── DRCD　
|  └── CMRC2018
|  └── ...
├── tools　　　　　　　　#　通用脚本
├── convert_tf_checkpoint_to_pytorch.py　#　模型文件转换
├── pytorch_modeling.py　#　模型文件
├── run_mrc.py       # 训练主程序
├── test_mrc.py   #　测试主程序
├── run_mrc_xxxx.sh   #　任务运行脚本

```
### 依赖模块

- pytorch==1.2.0 通过测试

### 运行

1. 下载对应tf模型权重到./check_points/prev_trained_model/对应模型名文件夹中，则运行转换脚本:
```
python convert_tf_checkpoint_to_pytorch.py \
      --tf_checkpoint_path=.check_points/prev_trained_model/roberta_wwm_ext_large/bert_model.ckpt \
      --bert_config_file=.check_points/prev_trained_model/roberta_wwm_ext_large/bert_config.json \
      --pytorch_dump_path=.check_points/prev_trained_model/roberta_wwm_ext_large/pytorch_model.pth
```
**注意**: 当转换完模型之后，需要在对应的文件夹内存放`bert_config.json`和`vocab.txt`文件

2. 直接运行对应任务sh脚本，如：

```shell
sh run_mrc_cmrc2018.sh
```
**注意**: 请根据需求调整参数和路径，测试命令中init_restore_dir请根据需要调整。如果init_restore_dir结尾为.pth .bin .pt则会直接读取，如果是目录，则会读取目录下唯一的一个权重文件。(超过一个会返回异常)





