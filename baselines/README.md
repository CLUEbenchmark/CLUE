<!--1. 数据集整体下载，解压到glue文件夹里-->
<!--  ```cd glue ```-->
<!--  ```wget https://storage.googleapis.com/chineseglue/chineseGLUEdatasets.v0.0.1.zip```-->

<!--    注：lcqmc数据集，请从<a href="http://icrc.hitsz.edu.cn/info/1037/1146.htm">这里</a>申请或搜索网络-->

<!--2. 训练模型-->
<!--  将预训练模型下载解压到对应的模型中prev_trained_model文件夹里-->
<!--  a. albert_xlarge-->
<!--  https://github.com/brightmart/albert_zh-->
<!--  b. bert-->
<!--  https://github.com/google-research/bert  -->
<!--  c. bert-wwm-ext  -->
<!--  https://github.com/ymcui/Chinese-BERT-wwm  -->
<!--  d. ernie  -->
<!--  https://github.com/ArthurRizar/tensorflow_ernie  -->
<!--  e. roberta    -->
<!--  https://github.com/brightmart/roberta_zh  -->
<!--  f. xlnet  -->
<!--  https://github.com/ymcui/Chinese-PreTrained-XLNet  -->
<!--  修改run_classifier.sh指定模型路径  -->
<!--  运行各个模型文件夹下不同任务对应的run_classifier_*.sh即可。如跑xnl运行： -->
<!--  ```sh run_classifier_xnli.sh```-->

1. 一键运行

    我们为您提供了可以“一键运行”的脚本来辅助您更快的在指定模型上运行特定任务。  
    
    以在 Bert 模型上运行“BQ 智能客服问句匹配”任务为例，您可以直接在 chineseGLUE/baselines/models/**bert**/ 下运行 run_classifier_**bq**.sh 脚本。

    ```bash
    cd chineseGLUE/baselines/models/bert/
    sh run_classifier_bq.sh
    ```
    
    该脚本将会自动下载“BQ 智能客服问句匹配”数据集（保存在chineseGLUE/baselines/glue/chineseGLUEdatasets/**bq**/ 文件夹下）和Bert模型（保存在 chineseGLUE/baselines/models/bert/prev_trained_model/ 下）。
    
    如果您想在其他模型上执行其他的任务，只需要在对应模型的目录下找到对应任务的执行脚本（ run_classifier_**??**.sh ），即可直接运行。
    
    注: 在pytorch中阅读理解(MRC)的baseline中，所有模型的(包括bert,roberta,albert但是不包含xlnet)baseline是整合的。请通过调整MODEL_NAME和BERT_DIR来调整使用不同的模型，注意MODEL_NAME如果包含"albert"则会构建albert模型。
    
2. 测试效果


    #### AFQMC 蚂蚁语义相似度 Ant Semantic Similarity (Accuracy)：
    |         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
    | :-------------------: | :----------: | :-----------: | :--------------------------------: |
    |     ALBERT-xlarge     |    -     |     -   | batch_size=16, length=128, epoch=3 |
    |      ALBERT-tiny      |    69.13%     |    69.92%    | batch_size=16, length=128, epoch=3 |
    |       BERT-base       |    74.16%     |     73.70%  | batch_size=16, length=128, epoch=3 |
    |   BERT-wwm-ext-base   |    73.74%     |      74.07%   | batch_size=16, length=128, epoch=3 |
    |      ERNIE-base       |         74.88% |      73.83%    | batch_size=16, length=128, epoch=3 |
    |     RoBERTa-large     |         |       74.02%   | batch_size=16, length=128, epoch=3 |
    |       XLNet-mid       |     70.73%    |   70.50%       | batch_size=16, length=128, epoch=3 |
    |    RoBERTa-wwm-ext    |   74.30%      |      74.04%       | batch_size=16, length=128, epoch=3 |
    | RoBERTa-wwm-large-ext |  74.92% |  76.55% | batch_size=16, length=128, epoch=3 |

    #### TNEWS' 头条新闻分类 Toutiao News Classification (Accuracy)：
    |         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
    | :-------------------: | :----------: | :-----------: | :--------------------------------: |
    |     ALBERT-xlarge     |    -     |         | batch_size=16, length=128, epoch=3 |
    |      ALBERT-tiny      |    -     |       53.35%   | batch_size=16, length=128, epoch=3 |
    |       BERT-base       |    -     |     55.58%    | batch_size=16, length=128, epoch=3 |
    |   BERT-wwm-ext-base   |         |    56.84%      | batch_size=16, length=128, epoch=3 |
    |      ERNIE-base       |         |     58.23%     | batch_size=16, length=128, epoch=3 |
    |     RoBERTa-large     |         |      57.05%    | batch_size=16, length=128, epoch=3 |
    |       XLNet-mid       |         |      56.24%    | batch_size=16, length=128, epoch=3 |
    |    RoBERTa-wwm-ext    |         |      56.86%       | batch_size=16, length=128, epoch=3 |
    | RoBERTa-wwm-large-ext |   | 58.61%  | batch_size=16, length=128, epoch=3 |

    #### IFLYTEK' 长文本分类 Long Text Classification (Accuracy)：
    |         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
    | :-------------------: | :----------: | :-----------: | :--------------------------------: |
    |     ALBERT-xlarge     |    61.94     |     61.34     | batch_size=32, length=128, epoch=3 |
    |      ALBERT-tiny      |    44.83     |     44.62     | batch_size=32, length=256, epoch=3 |
    |       BERT-base       |    63.57     |     63.48     | batch_size=32, length=128, epoch=3 |
    |   BERT-wwm-ext-base   |    63.83     |     63.75     | batch_size=32, length=128, epoch=3 |
    |      ERNIE-base       |    61.75     |     61.80     | batch_size=24, length=256, epoch=3 |
    |     RoBERTa-large     |    63.80     |     63.91     | batch_size=32, length=128, epoch=3 |
    |       XLNet-mid       |    60.16     |     60.04     | batch_size=16, length=128, epoch=3 |
    |    RoBERTa-wwm-ext    |    64.18     |       -       | batch_size=16, length=128, epoch=3 |
    | RoBERTa-wwm-large-ext | ***65.19***  |  ***65.10***  | batch_size=32, length=128, epoch=3 |

    #### CMNLI 中文自然语言推理 Chinese Multi-Genre NLI (Accuracy)：
    | 模型 | matched | mismatched |  训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base	| 79.39 | 79.76 | batch=32, length=128, epoch=3 lr=2e-5 |
    | BERT-wwm-ext-base	|81.41 |80.67|	batch=32, length=128, epoch=3 lr=2e-5 |
    | ERNIE-base	|79.65 | 80.70 | batch=32, length=128, epoch=3 lr=2e-5 |
    | ALBERT-xxlarge	|- | - | - |
    | ALBERT-tiny	|72.71 | 72.72 | batch=32, length=128, epoch=3 lr=2e-5 |
    | RoBERTa-large	| - | - | - |
    | xlnet-mid	|78.15 |76.93 | batch=16, length=128, epoch=3 lr=2e-5 |
    | RoBERTa-wwm-ext	|81.09 | 81.38 | batch=32, length=128, epoch=3 lr=2e-5  |
    | RoBERTa-wwm-large-ext	|***83.4*** | ***83.42*** | batch=32, length=128, epoch=3 lr=2e-5  |

    #### XNLI 自然语言推理  Natural Language Inference (Accuracy)：
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xlarge | 79.6 | 78.7 |batch_size=64, length=128, epoch=2 |
    | BERT-base | 77.80 | 77.80 | batch_size=64, length=128, epoch=2 |
    | BERT-wwm-ext-base | 79.4 | 78.7 | batch_size=64, length=128, epoch=2 |
    | ERNIE-base  | 79.7  |78.6 | batch_size=64, length=128, epoch=2 |
    | RoBERTa-large |***80.2*** |79.9 | batch_size=64, length=128, epoch=2 |
    | XLNet-mid | 79.2 | 78.7 | batch_size=64, length=128, epoch=2 |
    | RoBERTa-wwm-ext | 79.56 | 79.28 | batch_size=64, length=128, epoch=2 |
    | RoBERTa-wwm-large-ext | ***80.20*** | ***80.04*** | batch_size=16, length=128, epoch=2 |

    注：ALBERT-xlarge，在XNLI任务上训练暂时还存在有问题

    COPA TODO 

    #### WSC Winograd模式挑战中文版  The Winograd Schema Challenge,Chinese Version：
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xxlarge |    |    |  |
    | ALBERT-tiny |  57.7(52.9)  |  58.5(52.1)  | lr=1e-4, batch_size=8, length=128, epoch=50   |
    | BERT-base | 59.6（56.7)  | 62.0（57.9）  |  lr=2e-5, batch_size=8, length=128, epoch=50 |
    | BERT-wwm-ext-base | 59.4(56.7) |  61.1(56.2) | lr=2e-5, batch_size=8, length=128, epoch=50   |
    | ERNIE-base  | 58.1(54.9)| 60.8(55.9) | lr=2e-5, batch_size=8, length=128, epoch=50   |
    | RoBERTa-large | 68.6(58.7)  | 72.7(63.6)  | lr=2e-5, batch_size=8, length=128, epoch=50   |
    | XLNet-mid | 60.9(56.8）  |  64.4(57.3） | lr=2e-5, batch_size=8, length=128, epoch=50   |
    | RoBERTa-wwm-ext | 67.2(57.7)  | 67.8(63.5)  | lr=2e-5, batch_size=8, length=128, epoch=50   |
    | RoBERTa-wwm-large-ext |69.7(64.5) |  74.6(69.4) | lr=2e-5, batch_size=8, length=128, epoch=50   |

    #### CSL 关键词识别  Keyword Recognition (Accuracy)：
    |         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
    | :-------------------: | :----------: | :-----------: | :--------------------------------: |
    |     ALBERT-xlarge     |    80.23     |     80.29     | batch_size=16, length=128, epoch=5 |
    |     ALBERT-tiny     |             |     74.56     | batch_size=16, length=128, epoch=5 |
    |       BERT-base       |              |     80.23     | batch_size=4, length=256, epoch=5  |
    |   BERT-wwm-ext-base   |    80.60     |     81.00     | batch_size=4, length=256, epoch=5  |
    |      ERNIE-base       |    79.43     |     79.10     | batch_size=4, length=256, epoch=5  |
    |     RoBERTa-large     |    81.87     |     81.36     | batch_size=4, length=256, epoch=5  |
    |       XLNet-mid       |    82.06     |     81.26     | batch_size=4, length=256, epoch=3  |
    |    RoBERTa-wwm-ext    |    80.67     |     80.63     | batch_size=4, length=256, epoch=5  |
    | RoBERTa-wwm-large-ext |    82.17     |     82.13     | batch_size=4, length=256, epoch=5  |

    #### DRCD 繁体阅读理解 Reading Comprehension for Traditional Chinese (F1, EM)：
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base |F1:92.30 EM:86.60 | F1:91.46 EM:85.49 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | BERT-wwm-ext-base |F1:93.27 EM:88.00 | F1:92.63 EM:87.15 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ERNIE-base  |F1:92.78 EM:86.85 | F1:92.01 EM:86.03 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ALBERT-large  |F1:93.90 EM:88.88 | F1:93.06 EM:87.52 |  batch=32, length=512, epoch=3 lr=2e-5 warmup=0.05 |
    | ALBERT-xlarge |F1:94.63 EM:89.68 | F1:94.70 EM:89.78 |  batch_size=32, length=512, epoch=3 lr=2.5e-5 warmup=0.06 |
    | ALBERT-tiny |F1:81.51 EM:71.61 | F1:80.67 EM:70.08 |  batch=32, length=512, epoch=3 lr=2e-4 warmup=0.1 |
    | RoBERTa-large |F1:94.93 EM:90.11 | F1:94.25 EM:89.35 |  batch=32, length=256, epoch=2 lr=3e-5 warmup=0.1|
    | xlnet-mid |F1:92.08 EM:84.40 | F1:91.44 EM:83.28 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | RoBERTa-wwm-ext |F1:94.26 EM:89.29 | F1:93.53 EM:88.12 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1|
    | RoBERTa-wwm-large-ext |***F1:95.32 EM:90.54*** | ***F1:95.06 EM:90.70*** | batch=32, length=512, epoch=2 lr=2.5e-5 warmup=0.1 |

    #### CMRC2018 阅读理解 Reading Comprehension for Simplified Chinese (F1, EM)：
    | 模型 | 开发集（dev) | 测试集（test) |  训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base	|F1:85.48 EM:64.77 | F1:87.17 EM:69.72 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | BERT-wwm-ext-base	|F1:86.68 EM:66.96 |F1:88.78 EM:73.23|	batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ERNIE-base	|F1:87.30 EM:66.89 | F1:89.62 EM:73.32 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ALBERT-large	| F1:87.86 EM:67.75 |F1:90.17 EM:73.66| epoch3, batch=32, length=512, lr=2e-5, warmup=0.05 |
    | ALBERT-xlarge	| F1:88.66 EM:68.90 |F1:90.92 EM:75.22| epoch3, batch=32, length=512, lr=2e-5, warmup=0.1 |
    | ALBERT-tiny	| F1:73.95 EM:48.31 |F1:75.73 EM:53.68| epoch3, batch=32, length=512, lr=2e-4, warmup=0.1 |
    | RoBERTa-large	| F1:88.61 EM:69.94 |F1:90.94 EM:76.11| epoch2, batch=32, length=256, lr=3e-5, warmup=0.1 |
    | xlnet-mid	|F1:85.63 EM:65.31 | F1:86.09 EM:66.51 | epoch2, batch=32, length=512, lr=3e-5, warmup=0.1 |
    | RoBERTa-wwm-ext	|F1:87.28 EM:67.89 | F1:89.74 EM:73.89 | epoch2, batch=32, length=512, lr=3e-5, warmup=0.1 |
    | RoBERTa-wwm-large-ext	|***F1:89.42 EM:70.59*** | ***F1:91.56 EM:76.58*** | epoch2, batch=32, length=512, lr=2.5e-5, warmup=0.1 |

    #### CHID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test (Accuracy)：
    | 模型 | 开发集（dev) | 测试集（test) |  训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base	|82.20 | 82.04 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | BERT-wwm-ext-base	|83.36 |82.9 |	batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | ERNIE-base	|82.46 | 82.28 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | ALBERT-xlarge	| 79.44 |79.55 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | ALBERT-tiny	| 43.47 |43.53 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | RoBERTa-large	| 85.31 |84.50 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | xlnet-mid	|83.76 | 83.47 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | RoBERTa-wwm-ext	|83.78 | 83.62 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
    | RoBERTa-wwm-large-ext	|***85.81*** | ***85.37*** | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
