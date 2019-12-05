# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-01 22:28:41
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-02 18:36:50
# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import csv
import os
import six

import tensorflow as tf


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_txt(cls, input_file):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = f.readlines()
      lines = []
      for line in reader:
        lines.append(line.strip().split("_!_"))
      return lines

  @classmethod
  def _read_json(cls, input_file):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = f.readlines()
      lines = []
      for line in reader:
        lines.append(json.loads(line.strip()))
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def _create_examples(self, lines, set_type):
    """See base class."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(line['premise'])
      text_b = convert_to_unicode(line['hypo'])
      label = convert_to_unicode(line['label']) if set_type != 'test' else 'contradiction'
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


# class TnewsProcessor(DataProcessor):
#     """Processor for the MRPC data set (GLUE version)."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "toutiao_category_train.txt")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "toutiao_category_dev.txt")), "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "toutiao_category_test.txt")), "test")
#
#     def get_labels(self):
#         """See base class."""
#         labels = []
#         for i in range(17):
#             if i == 5 or i == 11:
#                 continue
#             labels.append(str(100 + i))
#         return labels
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = convert_to_unicode(line[3])
#             text_b = None
#             label = convert_to_unicode(line[1])
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


class TnewsProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    labels = []
    for i in range(17):
      if i == 5 or i == 11:
        continue
      labels.append(str(100 + i))
    return labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(line['sentence'])
      text_b = None
      label = convert_to_unicode(line['label']) if set_type != 'test' else "100"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


# class iFLYTEKDataProcessor(DataProcessor):
#     """Processor for the iFLYTEKData data set (GLUE version)."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "train.txt")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "test.txt")), "test")
#
#     def get_labels(self):
#         """See base class."""
#         labels = []
#         for i in range(119):
#             labels.append(str(i))
#         return labels
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = convert_to_unicode(line[1])
#             text_b = None
#             label = convert_to_unicode(line[0])
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


class iFLYTEKDataProcessor(DataProcessor):
  """Processor for the iFLYTEKData data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    labels = []
    for i in range(119):
      labels.append(str(i))
    return labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(line['sentence'])
      text_b = None
      label = convert_to_unicode(line['label']) if set_type != 'test' else "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class AFQMCProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(line['sentence1'])
      text_b = convert_to_unicode(line['sentence2'])
      label = convert_to_unicode(line['label']) if set_type != 'test' else '0'
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class CMNLIProcessor(DataProcessor):
  """Processor for the CMNLI data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "train.json"), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "dev.json"), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "test.json"), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples_json(self, file_name, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    lines = open(file_name, "r", encoding="utf-8")
    index = 0
    for line in lines:
      line_obj = json.loads(line)
      index = index + 1
      guid = "%s-%s" % (set_type, index)
      text_a = convert_to_unicode(line_obj["sentence1"])
      text_b = convert_to_unicode(line_obj["sentence2"])
      label = convert_to_unicode(line_obj["label"]) if set_type != 'test' else 'neutral'

      if label != "-":
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class CslProcessor(DataProcessor):
  """Processor for the CSL data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(" ".join(line['keyword']))
      text_b = convert_to_unicode(line['abst'])
      label = convert_to_unicode(line['label']) if set_type != 'test' else '0'
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


# class InewsProcessor(DataProcessor):
#   """Processor for the MRPC data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "train.txt")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     labels = ["0", "1", "2"]
#     return labels
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       text_a = convert_to_unicode(line[2])
#       text_b = convert_to_unicode(line[3])
#       label = convert_to_unicode(line[0]) if set_type != "test" else '0'
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
#
# class THUCNewsProcessor(DataProcessor):
#   """Processor for the THUCNews data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "train.txt")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     labels = []
#     for i in range(14):
#       labels.append(str(i))
#     return labels
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0 or len(line) < 3:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       text_a = convert_to_unicode(line[3])
#       text_b = None
#       label = convert_to_unicode(line[0])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
# class LCQMCProcessor(DataProcessor):
#   """Processor for the internal data set. sentence pair classification"""
#
#   def __init__(self):
#     self.language = "zh"
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
#     # dev_0827.tsv
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#     # return ["-1","0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     print("length of lines:", len(lines))
#     for (i, line) in enumerate(lines):
#       # print('#i:',i,line)
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       try:
#         label = convert_to_unicode(line[2])
#         text_a = convert_to_unicode(line[0])
#         text_b = convert_to_unicode(line[1])
#         examples.append(
#             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       except Exception:
#         print('###error.i:', i, line)
#     return examples
#
#
# class JDCOMMENTProcessor(DataProcessor):
#   """Processor for the internal data set. sentence pair classification"""
#
#   def __init__(self):
#     self.language = "zh"
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "jd_train.csv"), ",", "\""), "train")
#     # dev_0827.tsv
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "jd_dev.csv"), ",", "\""), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "jd_test.csv"), ",", "\""), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["1", "2", "3", "4", "5"]
#     # return ["-1","0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     print("length of lines:", len(lines))
#     for (i, line) in enumerate(lines):
#       # print('#i:',i,line)
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       try:
#         label = convert_to_unicode(line[0])
#         text_a = convert_to_unicode(line[1])
#         text_b = convert_to_unicode(line[2])
#         examples.append(
#             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       except Exception:
#         print('###error.i:', i, line)
#     return examples
#
#
# class BQProcessor(DataProcessor):
#   """Processor for the internal data set. sentence pair classification"""
#
#   def __init__(self):
#     self.language = "zh"
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
#     # dev_0827.tsv
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#     # return ["-1","0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     print("length of lines:", len(lines))
#     for (i, line) in enumerate(lines):
#       # print('#i:',i,line)
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       try:
#         label = convert_to_unicode(line[2])
#         text_a = convert_to_unicode(line[0])
#         text_b = convert_to_unicode(line[1])
#         examples.append(
#             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       except Exception:
#         print('###error.i:', i, line)
#     return examples
#
#
# class MnliProcessor(DataProcessor):
#   """Processor for the MultiNLI data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
#         "dev_matched")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["contradiction", "entailment", "neutral"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, convert_to_unicode(line[0]))
#       text_a = convert_to_unicode(line[8])
#       text_b = convert_to_unicode(line[9])
#       if set_type == "test":
#         label = "contradiction"
#       else:
#         label = convert_to_unicode(line[-1])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
#
# class MrpcProcessor(DataProcessor):
#   """Processor for the MRPC data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       text_a = convert_to_unicode(line[3])
#       text_b = convert_to_unicode(line[4])
#       if set_type == "test":
#         label = "0"
#       else:
#         label = convert_to_unicode(line[0])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
#
# class ColaProcessor(DataProcessor):
#   """Processor for the CoLA data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       # Only the test set has a header
#       if set_type == "test" and i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       if set_type == "test":
#         text_a = convert_to_unicode(line[1])
#         label = "0"
#       else:
#         text_a = convert_to_unicode(line[3])
#         label = convert_to_unicode(line[1])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#     return examples

class WSCProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["true", "false"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = convert_to_unicode(line['text'])
      text_a_list = list(text_a)
      target = line['target']
      query = target['span1_text']
      query_idx = target['span1_index']
      pronoun = target['span2_text']
      pronoun_idx = target['span2_index']

      assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))
                    ] == pronoun, "pronoun: {}".format(pronoun)
      assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)

      if pronoun_idx > query_idx:
        text_a_list.insert(query_idx, "_")
        text_a_list.insert(query_idx + len(query) + 1, "_")
        text_a_list.insert(pronoun_idx + 2, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
      else:
        text_a_list.insert(pronoun_idx, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
        text_a_list.insert(query_idx + 2, "_")
        text_a_list.insert(query_idx + len(query) + 2 + 1, "_")

      text_a = "".join(text_a_list)

      if set_type == "test":
        label = "true"
      else:
        label = line['label']

      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class COPAProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @classmethod
  def _create_examples_one(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid1 = "%s-%s" % (set_type, i)
#         try:
      if line['question'] == 'cause':
        text_a = convert_to_unicode(line['premise'] + '原因是什么呢？' + line['choice0'])
        text_b = convert_to_unicode(line['premise'] + '原因是什么呢？' + line['choice1'])
      else:
        text_a = convert_to_unicode(line['premise'] + '造成了什么影响呢？' + line['choice0'])
        text_b = convert_to_unicode(line['premise'] + '造成了什么影响呢？' + line['choice1'])
      label = convert_to_unicode(str(1 if line['label'] == 0 else 0)) if set_type != 'test' else '0'
      examples.append(
          InputExample(guid=guid1, text_a=text_a, text_b=text_b, label=label))
#         except Exception as e:
#             print('###error.i:',e, i, line)
    return examples

  @classmethod
  def _create_examples(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      i = 2 * i
      guid1 = "%s-%s" % (set_type, i)
      guid2 = "%s-%s" % (set_type, i + 1)
#         try:
      premise = convert_to_unicode(line['premise'])
      choice0 = convert_to_unicode(line['choice0'])
      label = convert_to_unicode(str(1 if line['label'] == 0 else 0)) if set_type != 'test' else '0'
      #text_a2 = convert_to_unicode(line['premise'])
      choice1 = convert_to_unicode(line['choice1'])
      label2 = convert_to_unicode(
          str(0 if line['label'] == 0 else 1)) if set_type != 'test' else '0'
      if line['question'] == 'effect':
        text_a = premise
        text_b = choice0
        text_a2 = premise
        text_b2 = choice1
      elif line['question'] == 'cause':
        text_a = choice0
        text_b = premise
        text_a2 = choice1
        text_b2 = premise
      else:
        print('wrong format!!')
        return None
      examples.append(
          InputExample(guid=guid1, text_a=text_a, text_b=text_b, label=label))
      examples.append(
          InputExample(guid=guid2, text_a=text_a2, text_b=text_b2, label=label2))
#         except Exception as e:
#             print('###error.i:',e, i, line)
    return examples
