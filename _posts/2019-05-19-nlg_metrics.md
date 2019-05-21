---
published: true
layout: single
title: "Automatic Evaluation Metrics for NLG"
path: "2019-05-19-nlg_metrics"
use_math: true
category: "Topic"
---

NLG (대부분 NMT)에서 사용되는 주요 evaluation metric들을 정리하였습니다. (MEWR를 제외하고는) output 시퀀스가 reference 시퀀스와 얼마나 오버랩되는가를 측정합니다.



<!--more-->

**NOTE.** [Novikova et al., 2017](https://www.aclweb.org/anthology/D17-1238) 은 automated NLG metric 21가지에 대한 비판으로, 모델을 평가하는 레벨에서는 괜찮으나 문장 레벨에서는 신뢰할 수 없다는 분석을 내놓았다. (추가 예정) 


### BLEU

$$
BLEU = \min(1, \frac{|\mathbf{y}|}{|\mathbf{r}|})(\prod_{i=1}^N \mathrm{precision}_i)^\frac{1}{N}
$$

* N-gram precision: output에 등장하는 n-gram (1~N-gram)들이 reference에 등장하는 n-gram들과 얼마나 겹치는지 측정한다.
* Clipping: reference에 등장하긴 하지만, output에 불필요하게 반복적으로 등장하는 n-gram들의 경우에는 correct prediction 횟수를 reference에 나타난 횟수로만 clipping한다. 
* Brevity penalty: 모델 입장에서는 짧은 output을 내놓는 것이 유리하다 (penalty 받을 확률이 적어지므로). 이를 보정하기 위해 output length가 reference length에 비해 짧을 경우 penalty를 준다. 
* Corpus-level metric: sentence-level로 측정하고 코퍼스에 평균을 내면 과대 측정된다.
* Caveat: reference의 수, brevity penalty의 계산에 쓰이는 reference의 선정, N의 크기, smoothing 등이 파라미터로 작용하지만 이들이 보고되지 않는 경우가 많아 주의해야 하며, 전처리 (tokenization, UNK handling)으로부터 영향을 받는다 ([Post, 2018](https://arxiv.org/pdf/1804.08771.pdf)).

  

#### 문제점

* 어휘 층위 이상의 의미를 고려하지 않는다. n-gram evaluation의 고질적인 문제.
* 통사 구조를 고려하지 않는다. 이로 인해 의미의 오류 및 fluency 문제가 생긴다.
* 복잡한 형태론을 가진 언어들에는 적합하지 않다. Reference 시퀀스의 surface morphology를 정확히 복구해낼 확률이 떨어지기 때문이다 (의미가 유지된다고 해도).
* 인간의 판단과 상관이 낮다는 연구들이 많다.
  * [Reiter, 2018](https://aclweb.org/anthology/J18-3002) : NMT를 모델 레벨에서 평가하는 데만 유효하고, 개별 텍스트를 평가하기에는 부적절하다.
  * [Sulem et al., 2018](https://aclweb.org/anthology/D18-1081) : Text simplificaton에서 사용하기에 부적절하고, 짧은 문장에 penalty를 주므로 부적 상관을 보이기까지 한다.



### NIST

* Weighted n-gram precision: Frequency가 낮은 n-gram일수록 정보성이 높으므로 weight를 높게 받는다. BLEU에서 기능어를 match/miss하는 것이 많은 비중을 차지해버리는 문제를 예방하지만, 동의어도 penalize하는 문제는 더 심해진다. 
* Brevity penalty의 수정: 사소한 길이 변화는 전체 점수에 영향을 별로 미치지 않도록 수정한다.



### METEOR

$$
METEOR = (1 - \text{fraction penalty}) \frac{P\cdot R}{\alpha P + (1-\alpha)R} \\
\text{fraction penalty} = \gamma\left(\frac{\text{#overlapping chunk}}{\text{#overlapping unigram}}\right)^\beta
$$

* Harmonic mean of unigram precision and recall: recall의 weight이 더 크다.
* Fraction penalty : 인접한 matching이 많으면 (=less fractioning) chunk는 적어지므로 penalty가 감소한다.
* WordNet을 사용하여 동의어/paraphrase를 고려한다.
* 단어의 표면형 대신 stem을 비교하여 개별 언어의 특징들을 반영한다.
* Sentence-level metric
* $\alpha, \beta, \gamma$ 는 fine-tuning하는 것이 적절 (원래 $\alpha=0.9, \beta=3, \gamma=0.5$)



### ROUGE

* **Summarization**의 주요 metric.
* Precision 대신 recall을 사용: reference가 output에 의해 얼마나 잘 포착되었는가를 평가하는 셈이다.
* Variants
  * ROUGE-N : N-gram overlap (보통 N=1,2)
  * ROUGE-L : Longest common subsequence (LCS)를 비교한다. 비연속적인 matching을 포착할 수 있다.
  * ROUGE-W : Weighted LCS; 연속적인 LCS에 높은 weight을 부여한다.
  * ROUGE-S: Skip-bigram; 모든 연속적/비연속적 pair들 (ordering은 유지)
  * ROUGE-SU: Skip-bigram + unigram

* Caveat: recall을 사용하므로, 긴 문장이 유리하다.

  

### TER (Translation Error Rate), TER-plus

* Output을 reference와 정확하게 일치하게 만들려면 얼마나 많은 edit (단어 삭제, 삽입, 대치, 순서 변경)을 거쳐야 하는지를 측정한다.
* Edit의 정의에서 paraphrase, stemming, 동의어 등을 고려할 수 있다.



### LEPOR

* Enhanced length penalty : output의 길이가 reference의 길이보다 길거나 짧으면 penalty.
* N-gram position difference penalty
* Weighted harmonic mean of P, R (unigrams)
* 형태론적으로 복잡한 언어들에 더 적합함을 표방한다.
* Variants
  * hLEPOR : Length penalty와 position difference penalty에도 weighted harmonic mean을 적용
  * nLEPOR : n-gram P, R
  * POS 정보를 사용하여 POS 층위에서도 점수를 구하고, 단어 층위의 점수와 weighted combination



### RIBES

* Word order에 기반한 rank correlation coefficient를 사용
  * Spearman's $\rho$ : rank의 distance
  * Kendall's $\tau$ : rank의 방향 차이
* Word boundary에 의존하지 않으므로 띄어쓰기가 없는 중국어, 일본어에 적합할 수 있다.
* Caveat: 여전히 인간의 판단과 상관이 낮다 ([Tan et al., 2015](https://www.aclweb.org/anthology/W15-5009))



### STM (Subtree Metric)

$$
STM = \frac{1}{D}\sum_{n=1}^D \frac{\sum_{t \in \text{output subtrees}_n} \text{count}_\text{clip}(t)}{\sum_{t \in \text{output subtrees}_n} \text{count}(t)}
$$

* 단순히 surface order 대신 parse를 비교하여 fluency를 포착하기 유리하다.
* Depth 1 (single node) ~D 까지의 output subtree들이 output에 나온 횟수 대비 reference에 나온 clipped 횟수를 비교함
* HWCM (Headword Chain Metric): Dependency tree에 사용하는 variant. Subtree를 headword로만 교체한다.

  

### MEWR

* Reference translation에 의존하지 않는다. ("Machine translation Evaluation without Reference Texts")
* Fidelity score : 우선, source 언어의 단어 임베딩을 target 언어로 맵핑하는 matrix를 MSE로 훈련시킨다. 그 후 source 언어의 문장을 위의 맵핑에 거쳐 얻은 임베딩과, target 문장 임베딩 사이의 cosine similarity에 기반하여 계산한다.
* Fluency: Perplexity의 역수.
* 인간의 판단과 상관이 높음을 표방..한다.
* Caveat: frequent한 (=low perplexity) phrase가 통사적으로 틀리게 조합되어 실제로는 fluent하지 않은 문장을 만드는 경우가 많다. (또한, 논문이 공개되지 않고 워크샵 자료만 있어 정확한 확인이 불가함)

