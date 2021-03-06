## DADER: Domain Adaptation for Deep Entity Resolution

![python](https://img.shields.io/badge/python-3.6.5-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.7.1-brightgreen)

Entity resolution (ER) is a core problem of data integration. The state-of-the-art (SOTA) results on ER are achieved by deep learning (DL) based methods, trained with a lot of labeled matching/non-matching entity pairs. This may not be a problem when using well-prepared benchmark datasets. Nevertheless, for many real-world ER applications, the situation changes dramatically, with a painful issue to collect large-scale labeled datasets. In this paper, we seek to answer: If we have a well-labeled source ER dataset, can we train a DL-based ER model for target dataset, without any labels or with a few labels? This is known as domain adaptation (DA), which has achieved great successes in computer vision and natural language processing, but is not systematically studied for ER. Our goal is to systematically explore the benefits and limitations of a wide range of DA methods for ER. To this purpose, we develop a DADER (Domain Adaptation for Deep Entity Resolution) framework that significantly advances ER in applying DA. We define a space of design solutions for the three modules of DADER, namely Feature Extractor, Matcher, and Feature Aligner. We conduct so far the most comprehensive experimental study to explore the design space and compare different choices of DA for ER. We provide guidance for selecting appropriate design solutions based on extensive experiments.

<!-- <img src="figure/architecture.png" width="820" /> -->

This repository contains the implementation code of six representative methods of [DADER]: MMD, K-order, GRL, InvGAN, InvGAN+KD, ED.

<!-- <img src="figure/designspace.png" width="700" /> -->


## DataSets
The dataset format is <entity1,entity2,label>. See [Hugging Face](https://huggingface.co/datasets/RUC-DataLab/ER-dataset) for details.

<!-- <img src="figure/dataset.png" width="700" /> -->


## Quick Start
Step 1: Requirements
- Before running the code, please make sure your Python version is 3.6.5 and cuda version is 11.1. Then install necessary packages by :
- `pip install dader`

- If Pytorch is not installed automatically, you can install it using the following command:
- `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

Step 2: Run Example

    ```python
    #!/usr/bin/env python3
    from dader import data, model

    # load datasets
    X_src, y_src = data.load_data(path='source.csv')
    X_tgt, X_tgt_val, y_tgt, y_tgt_val = data.load_data(path='target.csv', valid_rate = 0.1)


    # load model
    aligner = model.Model(method = 'invgankd', architecture = 'Bert')
    # train & adapt
    aligner.fit(X_src, y_src, X_tgt, X_tgt_val, y_tgt_val, batch_size = 16, ada_max_epoch=20)
    # predict                    
    y_prd = aligner.predict(X_tgt)
    # evaluate
    eval_result = aligner.eval(X_tgt, y_prd, y_tgt)

    ```

