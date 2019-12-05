#!/bin/bash
#########################################################################
# File Name: copa_eval_dev.sh
# Author: Junyi Li
# Personal page: dukeenglish.github.io
# Created Time: 21:39:07 2019-12-02
#########################################################################
'''
因为copa任务比较特殊，所以特地提供了一个额外的评估脚本。如果想要测试获取
和网络上公布一致的dev结果，请将dev.json作为test.json进行预测，将预测结果
进行评测，使用下面的命令。
如果不进行评测，则可以利用第二条命令生成submit格式的test结果
'''
python eval_copa.py copa_output/test_result.tsv
python covert_test.py copa_output/test_result.tsv

