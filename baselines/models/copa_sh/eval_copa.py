#coding:utf-8
import sys

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


golden_file = 'dev_label.txt'
golden_label=[]
for line in open(golden_file, 'r').readlines():
    ss = line.strip().split('\t')
    if len(ss) == 2:
        golden_label.append(ss[1])
    else:
        print ('wrong format!!!: ' + line.strip())

print ('golden_label size: ' + str(len(golden_label)))
correct_count = 0
wrong_count = 0
for i in range(0, len(golden_label)):
    if golden_label[i] == predict_label[i]:
        correct_count += 1
    else:
        wrong_count += 1
print ("correct_count: " + str(correct_count))
print ("wrong_count: " + str(wrong_count))
print ("precision: " + str( correct_count * 1.0 / len(golden_label)))



