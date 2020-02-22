# Contributing guidelines

## Pull Request Rules

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines]().
- Read and follow [Code of Conduct]().
- Check if my changes are consistent with the [guidelines]().
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

我们使用 `master` 分支作为我们的开发分支，这代表它是不稳定的分支。每个版本区间（如 0.1.x）都会创建一个 release 分支（如 `release-0.1`）作为稳定的发布分支。每发布一个新版本都会将其合并到对应的 release 分支并打上对应的 tag。

下面是开源贡献者常用的工作流（workflow）：

1. 将仓库 fork 到自己的 GitHub 下
2. 将 fork 后的仓库 clone 到本地
3. 创建新的分支，在新的分支上进行开发操作（**请确保对应的变更都有测试用例或 demo 进行验证**）
4. 保持分支与远程 master 分支一致（通过 `fetch` 和 `rebase` 操作）
5. 在本地提交变更（**注意 commit log 保持简练、规范**），**注意提交的 email 需要和 GitHub 的 email 保持一致**
6. 将提交 push 到 fork 的仓库下
7. 创建一个 pull request (PR)

提交 PR 的时候请参考。在进行较大的变更的时候请确保 PR 有一个对应的 Issue。

```
Describe what this PR does / why we need it
Does this pull request fix one issue?
Describe how you did it
Describe how to verify it
Special notes for reviews 
[copied from https://github.com/alibaba/Sentinel/blob/master/.github/PULL_REQUEST_TEMPLATE.md]
```

在提交 PR 后，系统会自动运行持续集成，请确保所有的 CI 均为 pass 状态。一切就绪后，我们会为 PR 分配一个或多个 reviewer。Reviewer 会对提交的代码进行 review。

在合并 PR 的时候，请把多余的提交记录都 squash 成一个。最终的提交信息需要保证简练、规范。

### Create Issue/PR

---

我们使用 GitHub Issues 以及 Pull Requests 来管理/追踪问题。

如果您发现了文档中有表述错误，或者代码发现了 BUG，或者希望开发新的特性，或者希望提建议，可以创建一个 Issue。请参考 Issue 模板中对应的指导信息来完善 Issue 的内容，来帮助我们更好地理解您的 Issue。

如果您想要贡献代码，您可以参考上面的 [GitHub 工作流]，提交对应的 PR。若是对当前开发版本进行提交，则目标分支为 `master`。如果您的 PR 包含非常大的变更，比如模块的重构或者添加新的组件，请**务必先提出相关 issue，发起详细讨论，达成一致后再进行变更**，并为其编写详细的文档来阐述其设计、解决的问题和用途。注意一个 PR 尽量不要过于大。如果的确需要有大的变更，可以将其按功能拆分成多个单独的 PR。

## 报告安全问题

特别地，若您发现 CLUE 及其生态项目中有任何的安全漏洞（或潜在的安全问题），请第一时间通过邮箱[chineseGLUE@163.com私下联系我们。在对应代码修复之前，**请不要将对应安全问题对外披露，也不鼓励公开提 issue 报告安全问题**。

### Contribution guidelines and standards

Before sending your pull request for [review](https://github.com/tensorflow/tensorflow/pulls), make sure your changes are consistent with the guidelines and follow the TensorFlow coding style.

#### General guidelines and philosophy for contribution

- Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.
- Keep API compatibility in mind when you change code. Reviewers of your pull request will comment on any API compatibility issues.
- When you contribute a new feature to CLUE, the maintenance burden is (by default) transferred to the CLUE team. This means that the benefit of the contribution must be compared against the cost of maintaining the feature.
- As every PR requires several CPU/GPU hours of CI testing, we discourage submitting PRs to fix one typo, one warning,etc. We recommend fixing the same issue at the file level at least (e.g.: fix all typos in a file, fix all compiler warning in a file, etc.)

### Code review

所有的代码都需要经过 committer 进行 review。以下是我们推荐的一些原则：

- 可读性：代码遵循我们的开发规约，重要代码需要有详细注释和文档
- 优雅性：代码简练、复用度高，有着完善的设计
- 测试：重要的代码需要有完善的测试用例（单元测试、集成测试），对应的衡量标准是测试覆盖率

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

# 社区

## 联系我们

### 邮件组

如果您有任何问题与建议，请通过邮箱[chineseGLUE@163.com](mailto:chineseGLUE@163.com)联系我们。

### Gitter

我们的 Gitter room: https://github.com/CLUEbenchmark



以上贡献者模版参考自：[Sentinel]([https://github.com/alibaba/Sentinel/wiki/%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97](https://github.com/alibaba/Sentinel/wiki/开源贡献指南)) and [Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)