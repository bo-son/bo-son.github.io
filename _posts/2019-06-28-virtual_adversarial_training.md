---
published: true
layout: single
title: "Adversarial Training Methods for Semi-supervised Text Classification"
path: "2019-06-28-virtual_adversarial_training"
use_math: true
category: "Paper reading"
tags: 
    - "adversarial training"
---

[Miyato et al. (ICLR 2017)](https://arxiv.org/abs/1605.07725)

두 가지를 적용하였다.

1) Discrete data의 특징상 input을 직접 perturb하지 않고 (normalized) 임베딩을 perturb한다.

2) Original embedding과 perturbed embedding의 KLD를 loss로 사용하는 virtual adversarial training을 사용한다. Label을 필요로 하지 않아 semi-supervised로 훈련될 수 있다.

<!--more-->

## How?

### Normalized embedding perturbation

Perturbation의 norm이 $\epsilon$ 으로 제한되므로, 모델이 임베딩 norm을 크게 학습하여 perturbation을 무시하는 상황을 방지하기 위해 normalized embedding을 사용한다. $\mathbf{s}_k$ 가 k번째 토큰의 임베딩이라고 할 때,

$$
\mathbf{\bar{s}}_k = \frac{\mathbf{s}_k - \mathcal{E}(\mathbf{s})}{\sqrt{\mathrm{Var(\mathbf{s})}}}
$$


### Virtual adversarial training ([Miyato et al., 2016](https://arxiv.org/abs/1507.00677)) 

#### Adversarial training

$$
\mathcal{L}_{\mathrm{adv}}(\theta) = -\frac{1}{N}\sum_{n=1}^N \log p(y_n | \mathbf{\bar{s}}_n + \mathbf{r}_{\mathrm{adv},n})
$$

Loss에 label y가 포함되어 있음을 확인할 수 있다. 

원래 worst case perturbation
$\mathbf{r}_{\mathrm{adv}}=\mathrm{argmin}\log p(y|\mathbf{\bar{s}+r})$
여야 하지만,
intractable하므로 다음과 같이 근사된다.

$$
\mathbf{r}_{\mathrm{adv}}=-\epsilon\frac{\mathbf{g}}{\|\mathbf{g}\|_2} \\
\mathbf{g}=\nabla_\mathbf{\bar{s}}\log p(y|\mathbf{\bar{s}})
$$


#### Virtual adversarial training

$$
\mathcal{L}_{\mathrm{v-adv}}=\frac{1}{N}\sum_{n=1}^{N}\,\mathrm{KL}[\,p(\cdot|\mathbf{\bar{s}}_n)\,\|\,p(\cdot|\mathbf{\bar{s}}_n + \mathbf{r}_{\mathrm{v-adv, n}})\,]
$$

NLL 대신 original embedding과 perturbed embedding의 KLD를 loss로 사용한다. Label y가 loss에 사용되지 않음을 확인할 수 있다. 따라서 N에는 labeled, unlabeled example이 모두 포함될 수 있다 (semi-supervised).

원래 worst case perturbation
$\mathbf{r}_{\mathrm{v-adv}}=\mathrm{argmax}\,\mathrm{KL}$
이어야 하지만, 마찬가지로 다음처럼 근사한다 (Note the positive sign!).
$\mathbf{d}$ 가 랜덤 벡터일 때,

$$
\mathbf{r}_{\mathrm{adv}}=\epsilon\frac{\mathbf{g}}{\|\mathbf{g}\|_2} \\
\mathbf{g}=\nabla_{\mathbf{\bar{s}+d}}\,\mathrm{KL}[\,p(\cdot|\mathbf{\bar{s}})\,\|\,p(\cdot | \mathbf{\bar{s}+d})\,]
$$


## 실험

* 여러 종류의 classification dataset (IMDB, Elec, Rotten Tomatoes, DBpedia, RCV1)에 대해 진행.
* LSTM, biLSTM을 base 모델로 사용.
* Perturbation은 임베딩 dropout 후 적용.
* NLL, adversarial / virtual adversarial loss 모두 virtual adversarial training을 했을 때 낮게 유지되었다. Baseline과 adversarial training은 labeled data만 사용할 수 있어 overfit.
* Adversarial, virtual adversarial을 같이 적용했을 때 성능이 좋은 데이터셋이 많았다. (perturbation norm constraint $\epsilon$ 은 shared)