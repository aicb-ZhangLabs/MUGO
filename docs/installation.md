# Installation

MUGO is available on PyPI and can be installed via pip.

## Stable Release

```bash
pip install mugo
```

Development Version
To install the latest version from source:

Bash

git clone [https://github.com/anonymous/mugo.git](https://github.com/anonymous/mugo.git)
cd mugo
pip install -e .
Requirements
Python >= 3.8

PyTorch >= 2.0

Pandas, NumPy


#### 4. `docs/advanced.md` (画饼专区 - 扩展性)
```markdown
# Extending MUGO

MUGO is designed to be modular. You can easily extend it to support new genomic models or custom objective functions.

## Adding a Custom Model

Inherit from `mugo.models.BaseOracle` and implement the `forward` method:

```python
import torch
from mugo.models import BaseOracle

class MyCustomModel(BaseOracle):
    def __init__(self, weights_path):
        super().__init__()
        self.model = load_pretrained(weights_path)
    
    def forward(self, one_hot_seq):
        # Your custom logic here
        return self.model(one_hot_seq)
Custom Loss Functions
Define any differentiable function that takes model outputs and returns a scalar loss:

Python

def specific_TF_binding_loss(pred_track, target_indices):
    """
    Maximize binding at specific genomic bins.
    """
    return -torch.sum(pred_track[:, target_indices])
```


#### 5. `docs/tutorials/repro.md` (Paper 复现)
*(记得在 docs 下新建 `tutorials` 文件夹)*
```markdown
# Reproducing Paper Results

This section describes how to reproduce the experiments presented in our KDD paper.

## Dataset Preparation

The scripts assume you have access to `hg38.fa` and the corresponding GTF annotations.

## Running the Optimization

To replicate the **Whole Blood** optimization task using the **Borzoi** backbone:

```bash
# This script is located in the 'src' directory of the repository
python src/train_model/MVP_multi_head.py --tissue blood --k 20

```


### Part 2: 编译与部署流程 (一条龙)

因为你的代码是在服务器（SSH: Voyager）上，而浏览器在你的本地电脑上，所以流程稍微多一步“下载”。

#### 1. 在服务器上编译
在项目根目录下运行：
```bash
mkdocs build