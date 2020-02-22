# Contributing guidelines

[中文版本](https://github.com/CLUEbenchmark/CLUE/tree/master/CONTRIBUTING_ZH.md)

## Pull Request Rules

Before sending your pull requests, make sure you followed this list.

- Read contributing guidelines.
- Check if my changes are consistent with the guidelines.
- Changes are consistent with the [Coding Style].
- Run Unit Tests.

## How to become a contributor and submit your own code

Welcome any contribution and whatever it is, even just a typo. Please raise your question via issue or email us privately. We take care both the documents and codes equally. So, just do it as long as you follow our rules.

### Where shall I start

----

If this is your first time to touch CLUE, then we suggest you start from solving issues or our minimal tasks. 

If you are already familiar with this project and you are definitely feel comfortable with NLP related problems, please raise what you want to do via issue or email, and follow the workflow below. 

WELCOME!

### **Github WorkFLOW**

---
We take branch "master" as our main branch, which means it is not wise to develop new feature directly on it. We encourage you to create your own branch and create a PR for your contribution, after you complete it.

Here's the workflow：

1. fort it into you github
2. clone it into your local machine. 
3. create a new branch and code on it
4. push you code into YOUR git
5. create a PR

If you have huge modification, please make sure there is a coressbonding issue on our main res.


```
Describe what this PR does / why we need it
Does this pull request fix one issue?
Describe how you did it
Describe how to verify it
Special notes for reviews 
[copied from https://github.com/alibaba/Sentinel/blob/master/.github/PULL_REQUEST_TEMPLATE.md]
```
After you create you PR, we will assign one or two reviewer for you PR.


### Create Issue/PR

---
We use Github Issue and Pull Request to manage/track problems.

If you find any little bug or typo, or you have new ideas about this project, you could create an issue. 

If you want to contribute code, please the workflow above. If you have big modification about this project or you want to reconstruct this project ,PLEASE create an issue or email us (chineseGLUE@163.com) firstly.


## Security Problems

If you find there is any serious bug about security, please contact us via chineseGLUE@163.com privately. PLEASE DO NOT publish any security problem via ANY public way, including issue. Thank you very much.

### Contribution guidelines and standards

Before sending your pull request for [review](https://github.com/tensorflow/tensorflow/pulls), make sure your changes are consistent with the guidelines and follow the TensorFlow coding style.

#### General guidelines and philosophy for contribution

- Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.
- Keep API compatibility in mind when you change code. Reviewers of your pull request will comment on any API compatibility issues.
- When you contribute a new feature to CLUE, the maintenance burden is (by default) transferred to the CLUE team. This means that the benefit of the contribution must be compared against the cost of maintaining the feature.
- As every PR requires several CPU/GPU hours of CI testing, we discourage submitting PRs to fix one typo, one warning,etc. We recommend fixing the same issue at the file level at least (e.g.: fix all typos in a file, fix all compiler warning in a file, etc.)

### Code review

All the code will need to be reviewed.

#### License

Include a license at the top of new files.

Python Liscense:

```
# Copyright 2020 The CLUE Authors. All Rights Reserved.
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
# =============================================================================
```

#### Python coding style

Use `pylint` to check your Python changes. To install `pylint` and check a file with `pylint` against TensorFlow's custom style definition:

We encourage PEP-8.

```
pip install pylint
pylint myfile.py
```

Note `pylint ` should run from the top level directory.

#### Running unit tests

We encourage you to send your PR with your test case. Then, the review process will be quick.

# Community

## Contact

### Email

Please contact us via [chineseGLUE@163.com](mailto:chineseGLUE@163.com).

### Gitter

Gitter room: https://github.com/CLUEbenchmark



All the things above, we refer to：[Sentinel]([https://github.com/alibaba/Sentinel/wiki/%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97](https://github.com/alibaba/Sentinel/wiki/开源贡献指南)) and [Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md). Thanks for their wisdom.