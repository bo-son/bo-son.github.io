---
published: true
layout: single
title: "Cooperative Learning of Disjoint Syntax and Semantics"
path: "2019-05-07-disjoint_syntax_semantics"
use_math: true
category: "Paper reading"
tags: 
    - "unsupervised parsing"
---

[Havrylov et al. (NAACL 2019)](https://arxiv.org/abs/1902.09393)

별도의 파서 (syntax)와 합성 함수 (semantics) 모델을 동시에 훈련시킬 때 발생하는 coadaptation 문제를 해결하기 위해서, Gumbel Tree-LSTM에 SCT와 PPO를 더하고 downstream task에 대해 실험하였다.

<!--more-->

## 배경 지식

### Coadaptation problem

Latent parser $p_\phi(\cdot|x)$ 는 인풋 문장에 대한 트리의 확률분포이고, 합성함수(compositional function)
$f_\theta(x,z)$ 는 트리를 따라 문장의 representation $y$ 를 만든다. 둘은 아래 objective에 대해 동시에 훈련된다.

$$
min_{\theta, \phi}\;\mathcal{L}(\theta, \phi) = \frac{1}{N}\sum_{(x,y)}\mathbb{E}_{z \sim p_\phi(z|x)}[\,\mathrm{logreg}(f_\theta(x,z), y)\,]
$$

이 식은 $\theta$ 에 대해서는 미분 가능하지만 $\phi$ 에 대해서는 아니므로, 파서에 대한 discrete optimization이 필요하다. [RL-SPINN]()의 경우 파서는 REINFORCE로 훈련하고 합성함수는 gradient descent로 훈련하였다. 그런데 이렇게 하면 파서의 학습이 합성함수에 비해 크게 느리므로, 초기의 트리 형태와 비슷한 것들로만 exploration이 제한되어 left-branching 트리만 생기는 등, suboptimal한 파싱 전략으로 수렴해버리는 문제가 생긴다. 이를 coadaptation이라 한다. 



## 모델

파서는 syntax, 합성함수는 semantic을 담당하는 별도의 모델로 놓는다 (sharing 없음). Coadaptation 문제는 gradient variance가 낮은 semantic 모듈이 syntax 모듈의 gradient를 방향을 압도해버리는 문제로도 해석된다.

[Gumbel Tree-LSTM]()에서 파싱은 end-to-end로 훈련되므로 coadaptation 문제를 피할 수 있다. 하지만 ST Gumbel-Softmax로 근사를 하면서 gradient estimate에 bias가 생긴다 (empirically works though). 이 논문은 unbiased objective function을 유지하기 위하여 Gumbel Tree-LSTM의 훈련 방식을 수정하였다. 

### Objective

엔트로피 $\mathcal{H}$ 를 추가하였다. 엔트로피가 커야 loss가 낮아지므로, suboptimal한 파싱 전략으로 수렴해버리는 것을 예방한다. 

$$
min_{\theta, \phi}\;\mathcal{L}(\theta,\phi) - \lambda\sum_x \mathcal{H}(z|x)
$$


### Parser training with SCT

파서를 $-\mathcal{L}$ 을 reward로 받는 RL agent로 간주하여 REINFORCE로 훈련한다. Policy는 K개의 merging action으로 구성되는 이진 트리들에 대한 확률분포로 나타낸다. $\mathbf{r}^k = (\mathbf{r}_0^k, \dots, \mathbf{r}_{K-k}^k)$ 를 k번째 층에서의 노드 K-k개의 representation 벡터라고 할 때, 다음이 성립한다.

$$
p_\phi (z|x) = \prod_{k=0}^K \pi_\phi (a_k|\mathbf{r}^k) \\
\nabla_\phi\mathcal{L}(\theta, \phi) = \text{logreg}(f_\theta(x,z), y)\:\nabla_\phi\log p_\phi(z|x)
$$

분산을 줄이기 위해 baseline을 사용하는데, 일반적으로 쓰는 $b(t) = r\nabla_\phi \log p_\phi(z|x)$ (recent average x score 의 moving average)는 인풋에 따라 체계적으로 발생하는 reward의 차이를 포착하지 못하는 문제가 있다 (예: 긴 문장은 파싱이 어려워서 체계적으로 더 낮은 reward를 발생시키기 때문에, 최적화 결과 긴 시퀀스에 대한 action이 무시될 수 있다). 
따라서 input-dependent한 self-critical training의 baseline $b(t, x) = r_{\theta, \phi}(x) \nabla_\phi \log p_\phi(z|x)$을 사용한다.
$r_{\theta, \phi}$ 는 테스트 때 얻은 보상이다. 훈련 중 gradient fluctuation의 문제가 있어서, 실제로는 reward를 normalize하여 unit range로 제한하여 사용한다.

**Greedy decoding.**	테스트 때 parse tree는 본래
$\hat{z} = \mathrm{argmax}p_\phi(z|x)$ 이지만, 이 계산은 exponential하므로 greedy decoding을 하여 얻은 액션
$\hat{a}_k = \mathrm{argmax}\pi_{\phi}(a_k | \hat{\mathbf{r}}^k)$ 의 시퀀스로 대체한다.

### PPO

BPST로 훈련되는 합성함수에 비해서 파서의 gradient variance가 더 크고 최적화가 어렵다. 따라서 PPO를 사용하여 정책의 변화를 제한한다. 파서에 대해 SGD 스텝을 여러 번 밟아서 합성함수와 비슷한 페이스로 훈련될 수 있게 한다.



## 실험

**NOTE.**	성능상의 개선이 real data에서는 명확하지 않다. (PTB-style) Constituency parsing이 목표가 아니기 때문에 downstream task의 성능이 중요한데, 이렇게 까다로운 과정을 거치고 얻는 게 없어 보인다. 

#### 1. Toy data: ListOps

* Mathematical expression parsing
* RL-SPINN과 Gumbel Tree-LSTM은 일반 LSTM보다도 잘 하지 못하는데, 이 모델은 거의 완벽하게 함.
* SCT baseline이나 PPO를 빼면 variance가 크고 성능이 낮음.
* 파서와 합성함수의 파라미터를 sharing했을 때 성능이 크게 떨어짐을 확인.
* Hyperparameter에 예민함.
* 파서가 만드는 final representation이 grammatical한 수식의 경우 다들 비슷한데, ungrammatical한 것들끼리는 차이가 큰 것으로 보아 '문법적으로 맞음'을 제대로 표상했다고 보았다.

#### 2. Real data: SNLI, MNLI, SST

* 오리지널 Gumbel Tree-LSTM에 비해 큰 차이가 없거나 더 낮음.
* 반절의 run에서 left-branching으로 collapse함. 
  * 논문에서 명확히 밝히진 않았으나 파싱 결과/성능의 variance가 큰 것 같은데, 이는 오리지널 Gumbel Tree-LSTM에서도 나타나는 특징 [Williams et al., 2017](https://arxiv.org/abs/1709.01121)