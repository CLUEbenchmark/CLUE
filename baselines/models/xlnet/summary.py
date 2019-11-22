# -*- coding: utf-8 -*-
'''
print summary
'''
from __future__ import print_function
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pdb
import os
import math
import numpy as np
import collections
from prettytable import PrettyTable

def print_summary():
	lscmd = os.popen('ls '+sys.argv[1]+'/result.*').read()
	result_list = lscmd.split()
	num_args = len(result_list)
	assert num_args==2 or num_args==3

	dev_input_file = open(sys.argv[1]+'/result.dev', 'rb')
	test_input_file = open(sys.argv[1]+'/result.test', 'rb')
	if num_args==2:
		print_table = PrettyTable(['#','DEV-AVG','DEV-EM','DEV-F1','TEST-AVG','TEST-EM','TEST-F1','FILE'])
	elif num_args==3:
		chl_input_file = open(sys.argv[1]+'/result.challenge', 'rb')
		print_table = PrettyTable(['#','DEV-AVG','DEV-EM','DEV-F1','TEST-AVG','TEST-EM','TEST-F1','CHL-AVG','CHL-EM','CHL-F1','FILE'])

	# style set
	print_table.align['FILE'] = 'l'
	print_table.float_format = '2.3'

	# data fill
	dev_avg = []
	dev_em = []
	dev_f1 = []
	dev_file = []
	for dline in dev_input_file.readlines():
		dline = dline.strip()
		if re.search('^{', dline):
			ddict = json.loads(dline)
			dev_avg.append(float(ddict['AVERAGE']))
			dev_em.append(float(ddict['EM']))
			dev_f1.append(float(ddict['F1']))
			dev_file.append(ddict['FILE'])

	test_avg = []
	test_em = []
	test_f1 = []
	test_file = []
	for dline in test_input_file.readlines():
		dline = dline.strip()
		if re.search('^{', dline):
			ddict = json.loads(dline)
			test_avg.append(float(ddict['AVERAGE']))
			test_em.append(float(ddict['EM']))
			test_f1.append(float(ddict['F1']))
			test_file.append(ddict['FILE'])

	if num_args==3:
		chl_avg = []
		chl_em = []
		chl_f1 = []
		chl_file = []
		for dline in chl_input_file.readlines():
			dline = dline.strip()
			if re.search('^{', dline):
				ddict = json.loads(dline)
				chl_avg.append(float(ddict['AVERAGE']))
				chl_em.append(float(ddict['EM']))
				chl_f1.append(float(ddict['F1']))
				chl_file.append(ddict['FILE'])

	# print
	if num_args == 2:
		min_len = min(len(dev_avg),len(test_avg))
		for k in range(min_len):
			print_table.add_row([k+1, dev_avg[k], dev_em[k], dev_f1[k], test_avg[k], test_em[k], test_f1[k], dev_file[k]])
	elif num_args == 3:
		min_len = min(len(dev_avg),len(test_avg),len(chl_avg))
		for k in range(min_len):
			print_table.add_row([k+1, dev_avg[k], dev_em[k], dev_f1[k], test_avg[k], test_em[k], test_f1[k], chl_avg[k], chl_em[k], chl_f1[k], dev_file[k]])

	if len(sys.argv)==3:
		sk = sys.argv[2].upper()
		print('sort key detected: {}'.format(sk))
		print(print_table.get_string(sortby=sk, reversesort=True))
	else:
		print(print_table)
	

	if num_args == 2:
		summary_table = PrettyTable(['#','DEV-AVG','DEV-EM','DEV-F1','TEST-AVG','TEST-EM','TEST-F1','FILE'])
		summary_table.add_row(["M", np.max(dev_avg), np.max(dev_em), np.max(dev_f1), 
								  np.max(test_avg), np.max(test_em), np.max(test_f1),"-"])
		summary_table.add_row(["A", np.mean(dev_avg), np.mean(dev_em), np.mean(dev_f1), 
								  np.mean(test_avg), np.mean(test_em), np.mean(test_f1),"-"])
		summary_table.add_row(["D", np.std(dev_avg), np.std(dev_em), np.std(dev_f1), 
								  np.std(test_avg), np.std(test_em), np.std(test_f1),"-"])
	elif num_args == 3:
		summary_table = PrettyTable(['#','DEV-AVG','DEV-EM','DEV-F1','TEST-AVG','TEST-EM','TEST-F1','CHL-AVG','CHL-EM','CHL-F1','FILE'])
		summary_table.add_row(["M", np.max(dev_avg), np.max(dev_em), np.max(dev_f1), 
								  np.max(test_avg), np.max(test_em), np.max(test_f1),
								  np.max(chl_avg), np.max(chl_em), np.max(chl_f1), "-"])
		summary_table.add_row(["A", np.mean(dev_avg), np.mean(dev_em), np.mean(dev_f1), 
								  np.mean(test_avg), np.mean(test_em), np.mean(test_f1),
								  np.mean(chl_avg), np.mean(chl_em), np.mean(chl_f1), "-"])
		summary_table.add_row(["D", np.std(dev_avg), np.std(dev_em), np.std(dev_f1), 
								  np.std(test_avg), np.std(test_em), np.std(test_f1),
								  np.std(chl_avg), np.std(chl_em), np.std(chl_f1), "-"])
	# style set
	summary_table.align['FILE'] = 'l'
	summary_table.float_format = '2.3'
	print(summary_table)
	return 0




if __name__ == '__main__':
	print_summary()

