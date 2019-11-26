# ChineseGLUE
Language Understanding Evaluation benchmark for Chinese: datasets, baselines, pre-trained models, corpus and leaderboard

中文语言理解测评基准，包括代表性的数据集、基准(预训练)模型、语料库、排行榜。  

我们会选择一系列有一定代表性的任务对应的数据集，做为我们测试基准的数据集。这些数据集会覆盖不同的任务、数据量、任务难度。

中文任务基准测评(ChineseGLUE)-排行榜 Leaderboard
---------------------------------------------------------------------
#####  排行榜会定期更新                     数据来源: https://github.com/CLUEbenchmark/CLUE

#### 分类任务(v1版本,正式版)

| 模型   | Score  | 参数    | LCQMC'  | TNEWS'  | IFYTEK'   | CMNLI-m  | CMNLI-mm  | XNLI     | COPA | WSC | CSL  |
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>        | - | 108M | 74.89% | 55.58%  | 60.29% | 79.39%  | 79.76% | 77.8%  | 57.40% | -  | -      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a>      | - | 108M  | 76.72% | 56.84%  | 59.43% | 81.41% | 80.67% | 78.7%  | 61.4%  | -  | -      |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>         | - | 108M  | 77.94% | 58.23% | 58.96% | 79.65% | 80.70% | 78.6%  | -  | -  | -      |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a>      | - | 334M  | 77.57% | 57.05%  | 62.55% | -  |  - | 79.9%   | - | -   | -       |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>  | - | 200M | - | 56.24% | 57.85% | 78.15%  | 76.93%   | 78.7% | -      | -   | -     |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xxlarge</a>      | - | 59M   | -  | - | - | - | - | - | -  | -  | -     |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a>        | - | 4M |76.57% | 53.35% | 36.18% | 72.71% | 72.72%  | 69.5% | -  | -   | -     |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>   | - | 108M  | 77.43% | 56.86% | 60.31% | 81.09% | 81.38% | 79.3%  | 63.6%  | - | -      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | - | 330M | **78.02%** | **58.61%** | 62.98% | **83.4%** | **83.42%** | **80.0%** | 59.40% | - | - |


注：' 代表对原数据集筛选后获得，数据集与原数据集不同；TNEWS:文本分类(Acc)；LCQMC:语义相似度(Acc)；XNLI/MNLI:自然语言推理(Acc),CMNLI-m:chinese-MNLI-matched，CMNLI-mm:chinese-MNLI-mismatched；

DRCD & CMRC2018:抽取式阅读理解(F1, EM)；CHID:成语多分类阅读理解(Acc)；BQ:智能客服问句匹配(Acc)；MSRANER:命名实体识别(F1)；iFLYTEK:长文本分类(Acc)；

Score是通过计算1-9数据集得分平均值获得；


#### 阅读理解任务

| 模型 | Score | 参数 | DRCD | CMRC2018 | CHID |
| :----:| :----: | :----: | :----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>	| 79.08 | 108M | 85.49 	| 69.72 | 82.04 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a> | - | 108M | 87.15 | 73.23 | - |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>	| 80.54 | 108M | 86.03 | 73.32 | 82.28 |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a> | 83.32 | 334M 	| 89.35 | 76.11 | 84.5 |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>	| - | 209M | 83.28 | 66.51  | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xlarge</a> | - | 59M | 89.78 | 75.22 | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xxlarge</a> | - | - | - | - | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a> | 55.73 | 1.8M | 70.08 | 53.68 | 43.53 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>  | 81.88 | 108M  | 88.12 | 73.89 | 83.62 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | ***84.22*** | 330M |	***90.70*** |	***76.58*** | ***85.37*** |

注：阅读理解上述指标中F1和EM共存的情况下，取EM为最终指标
