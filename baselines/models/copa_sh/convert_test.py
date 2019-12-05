#coding:utf-8
import sys
import json

test_file=sys.argv[1]
predict_label = []
tmp = []
for line in open(test_file, 'r').readlines():
    ss = line.strip().split('\t')
    if len(ss) == 2:
        tmp.append(ss[1])
    else:
        print ('wrong format!!!: ' + line.strip())

i = 0
while(i < len(tmp)-1):
    if tmp[i] >= tmp[i+1]:
        predict_label.append(str(0))
    else:
        predict_label.append(str(1))
    i += 2
print ("predict_label size: " + str(len(predict_label)))

res = {}
for idx, label in enumerate(predict_label):
    res['id'] = idx
    res['label'] = label
    print(json.dumps(res, ensure_ascii=False))


