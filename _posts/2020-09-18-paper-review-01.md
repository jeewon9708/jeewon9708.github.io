---
layout: post
title:  "Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs"
date:   2020-09-17T16:25:52-23:00
author: Jeewon Chae
categories: Paper_review
use_math: true
---


Deepak Nathani∗ Jatin Chauhan∗ Charu Sharma∗ Manohar Kaul Department of Computer Science and Engineering, IIT Hyderbad {deepakn1019,chauhanjatin100,charusharma1991}@gmail.com, mkaul@iith.ac.in

해당 블로그 포스트는 ACL 2019 에 게재된 [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://www.aclweb.org/anthology/P19-1466/) 논문 정리 및 분석하기 위해 작성되었습니다.

### Knowledge graphs(KGs)와 relaton prediction이란?

Knowledge graphs(KGs)는 Knowledge bases(KBs)를 directed graph로 표현한 것을 의미합니다. 이 때 KBs는 AI가 사용될 분야에 대해 축적한 전문 지식 그리고 문제 해결에 필요한 사실과 규칙 등이 저장되어 있는 데이터베이스입니다. 즉, KGs는 아래의 그림과 같이 KBs의 entity를 node로, relation을 edge로 표현한 자료구조입니다. 

![Image_1](https://user-images.githubusercontent.com/22410209/93541718-2173cc80-f992-11ea-9985-7c518a48d745.JPG)

위의 그림에서 London, capital_of, United Kingdom과 같이 2개의 entity와 하나의 relation를 triple(London, capital_of, United Kingdom) 이라고 표현합니다.

KGs finds는 다양한 application에서 사용됩니다. 그러나 일반적으로 missing relation의 문제를 갖고 있고 이에 따라 주어진 triple이 valid한지 아닌지를 예측하는 knowledge base completion(relation prediction) 분야의 연구가 활발해졌습니다. 

### Relation prediction의 기존 연구

state-of-the-art relation prediction 방법은 knowledge embedding based model이라고 알려져있습니다. knowledge embedding based model은 아래의 네가지 방법으로 나누어집니다.

*아래의 예시 설명들은 간단한 정리를 위함이니 더 궁금하신 점이 있으면 링크를 통해 해당 논문을 읽으실 수 있습니다.*

1. Compositional models

   Ex) RESCAL, NTN, HOLE
   RESCAL과 NTN은 tensor product를 사용해서 많은 상호관계를 capture한다. 그러나 필요한 파라미터가 많고 따라서 계산이 번거로운 단점이 있습니다. 이러한 단점을 극복하기 위해 HOLE은 더 효율적이고 scalable한 compositional representation을 고안했는데 이는 entity embeddings의 circular correlation을 사용하는 것입니다. 

2. Translational models

   Ex) TRANSE, DISTMULT, ComplEx -> 상대적으로 simple
   TransE는 relation의 head와 tail entity 사이의 translation operation을 고려하였고  DISTMULT는 bilinear diagonal model이라는 NTN과 TransE에서 사용된 bilinear objective의 특별한 케이스를 사용하여 embedding을 학습합니다. DISTMULT는 entity relation을 모델링하기 위해 weighted dot product를 사용하였다. ComplEx는 complex embedding을 사용하고 Hermitian dot product를 대신 사용하는 방법으로 DISTMULT를 generalize하였습니다. 

   이러한 translational model은 빠르고 상대적으로 필요한 파라미터의 개수가 적고 train하기 쉽다는 장점이 있지만  less expressive하다는 단점이 있습니다. 

3. CNN-based model

   Ex) ConvE, ConvKB

   ConvE는 relation을 예측하기 위해 embedding에 2-D convolution을 사용하고 이는 1) Convolution layer 2) fully connected projection layer 3) 최종 예측을 위한 inner product layer로 이루어져있습니다. Global relationship을 찾기 위해 Multiple filter를 사용함에 따라 여러 다른 feature map이 생기고 이 feature map들을 concatenation한 것이 하나의 input triple을 표현합니다.  이런 모델들은 파라미터의 개수가 상대적으로 적다는 장점이 있지만 triple간의 관계를 고려하지 않은 채 예측한다는 단점이 있습니다

4. Graph-based model

   Ex) R-GCN

   R-GCN은 GCN을 relational data로 확장한 것입니다. Convolution 연산을 각 entity의 이웃노드에도 적용하고 각 이웃노드마다 동일한 가중치를 부여합니다. 이 그래프 기반 모델은 CNN 기반 모델에 비해 성능이 낮습니다. 

즉, 위에서 확인할 수 있듯이 현재 존재하는 방법들은 아래와 같은 문제점이 있습니다. 

* Translational model과 CNN-based model 모두 각각의 triple을 독립적으로 처리하기 때문에 KG의 해당 entity 주변에 내재적으로 존재하는 중요하고 잠재력있는 relation들을 고려하지 못합니다. 

* Graph-based model에 있는 GCN의 경우에는 1-hop neighborhood에 있는 가장 관련있는 node feature에만 집중하였습니다.

*  R-GCN의 경우에는 GCN보다 발전하여 multi-hop 이웃으로 확장하였지만 entity의 feature에만 집중할 뿐 각각의 entity와 relation의 feature를 서로 독립되고 별개인 방법으로 간주하는 방법입니다. 

따라서 본 논문에서 제안하는 graph attention model은 전체적으로 KG에 주어진 entity에 대해서 multi-hop과 n-hop neighborhood에 있는 의미가 유사한 relation에도 attention을 두는 방법으로 모델을 구성하였습니다. 



### Graph Attention Networks(GATs)

본 논문에서 제안한 모델을 이해하기에 앞서 위에서 설명한 Graph-based model의 예시였던 GCN의 단점을 해결하기 위하여 나온 GAT에 대해 알아보도록 하겠습니다.

Graph Convolutional networks(GCNs) 는 entity의 이웃노드에게서 정보를 수집하는데 이 때 부여하는 가중치가 모두 동일합니다. 이러한 단점을 해결하기 위해 나온 Graph attention networks(GATs)는 노드의 이웃 노드에게 서로 다른 중요도를 부여합니다. 즉, 각각의 중요도를 계산한 후 가중치를 다르게 부여하는 것입니다. 

먼저, 한 레이어의 노드의 input feature set은  $x= \{\vec{x_1}, \vec{x_2}, \vec{x_3}, ..., \vec{x_N}\}$ 이고 layer를 지나고 나면 $x'= \{\vec{x_1'}, \vec{x_2'}, \vec{x_3'}, ..., \vec{x_N'}\}$ 을 output으로 만듭니다. 이 때 $x$ 와 $x'$ 는 모두 entity $e_i$의 embedding이고 N은 노드 개수입니다. 따라서 하나의 GAT 레이어는 아래의 식으로 설명될 수 있습니다. 

$$e_{ij}= {a( W \vec{x_i},W \vec{x_j})}$$

이때 $(e_{ij})$는 edge $(e_i,e_j)$의 attention value를 의미합니다. Attention value는 각 edge의 feature들이 시작 노드인 $e_i$에 대한 중요도입니다. $W$는 input feature를 더 큰 차원을 가진 output feature공간으로 매핑하기 위한 선형변환이고 $a$는 attention function으로 직접 정하는 함수입니다. 여기서 relative attention $\alpha_{ij}$ 는 이웃에 있는 모든 value에 대해 softmax function을 이용하여 계산한 값입니다. 아래는 output embedding을 구하는 과정입니다.

$${\vec{x_i'} = \sigma(\sum\limits_{j\in\mathbb{N_i}} \alpha_{ij}W\vec{x_j})}$$

그런데, GAT는 learning precoess를 안정화시키기 위해 multi-head attention을 사용합니다. (자세한 내용은 [여기](https://arxiv.org/abs/1706.03762)를 클릭하세요) multi-head attention process는 $K$개의 attention head를 합치는 것으로 이루어지는데 구체적인 식은 아래와 같습니다. 이 때, $||$ 는 합치는 과정을 의미하고 $\sigma$는 비선형 함수를 의미하고 $\alpha_{ij}^k$는 정규화된 edge $(e_i,e_j)$의 계수를 의미하는데 이는 k-th attention mechanism으로 계산됩니다. 그리고 $W^k$는 k-th attention mechanism의 선형 변환 행렬을 의미합니다. 
$${\vec{(x_i')}} = {\vert\vert}_{k=1}^K \sigma(\sum\limits_{j\in\mathbb{N_i}} \alpha_{ij}^kW\vec{x_j})$$
마지막으로, 최종 레이어에서는 output embedding이 평균을 구하는 것으로 계산되기 때문에 아래와 같이 연산됩니다. 
$$ \\
{\vec{x_i'}} = \sigma({1\over{K}}\sum_{k=1}^K
{\sum\limits_{j\in\mathbb{N_i}}} \alpha_{ij}^kW^k\vec{x_j}) \\
$$


GAT는 성공적이었지만 KG의 중요한 파트인 relation feature에 대해서는 고려하지 않았기 때문에 KGs에는 적합하지 않았습니다. KG에서는 entity가 어떤 relation과 관계되어지는지에 따라서 다른 역할이 부여되기 때문입니다. 

![그림 2](https://user-images.githubusercontent.com/22410209/93541748-3d776e00-f992-11ea-9bc6-8f4b1034867a.JPG)



예를 들어, 위의 그림에서 보면 Christopher Nolan은 두개의 서로 다른 triple에 나타나는데 하나의 역할은 brother이고 하나의 역할은 director입니다. 



### 본 논문이 제안하는 New model

위와 같은 문제를 해결하고자 본 논문에서는 relation과 neighboring node feature를 모두 고려하는 attention mechanism에서의 embedding 접근을 제안합니다. 먼저 하나의 attentional 레이어를 만들고 모델 구축을 시작합니다. GAT와 유사하게 이 framework는 특정 attention mechanism에 더욱 강력합니다. 아래의 그림은 본 모델의 전체 flow를 나타낸 그림입니다.

![Image_4](https://user-images.githubusercontent.com/22410209/93541770-4cf6b700-f992-11ea-8d25-c65d58175007.JPG)

제안하는 모델의 각 layer는 2개의 embedding 행렬을 input으로 가집니다. 

* Entity embeddings $H \in R^{N_e \times T}$ 

  $N_e$ 는 전체 entity 개수, $T$ 는 각 entity embedding의 feature dimension

* relation embeddings $G \in R^{N_r \times P}$  

  $N_r$ 는 전체 relation 개수, $P$ 는 각 relation embedding의 feature dimension

그리고  $H' \in R^{N_e \times T'}$ 와  $G' \in R^{N_r \times P'}$ 을 output으로 내보냅니다. 

entity $e_i$의 새로운 embedding을 얻으려면 $e_i$가 속한 triple가 학습됩니다. 이러한 embedding을 아래 식에서 볼 수 있듯이 entity의 concatenation과 특정 triple $t_{ij}^k=(e_i,r_k,e_j)$ 의 relation feature vector 사이의 선형변환을 수행하는 것으로 학습합니다. 
$$
{\vec{c_{ijk}}} = W_1[\vec{h_i} || \vec{h_j} ||\vec{g_k}]
$$
위의 식에서 $\vec{c_{ijk}}$ 은 triple  $t_{ij}^k$의 벡터형이고 $\vec{h_i}$ , $\vec{h_j}$ ,  $\vec{g_k}$ 은 각각 entity $e_i$, $e_j$와 relation 의 embedding입니다. 그리고 $W_1$은 선형변환 행렬입니다. 

GAT와 유사하게 각 triple  $t_{ij}^k$의 중요도를  $b_{ijk}$ 라고 할 때 아래와 같이 계산한다. 이 때, 가중치 행렬인 $W_2$ 와 LeakyReLU의 계산으로 triple의 중요도를 계산한다. 
$$
b_{ijk} = LeakyReLU(W_2c_{ijk})
$$
아래의 식과 같이 Relative attention value $\alpha_{ijk}$를 얻기 위해 GAT와 유사하게 softmax를  $b_{ijk}$에 적용하여 구합니다. 
$$
\alpha_{ijk}=softmax_{jk}(b_{ijk}) = \frac{exp(b_{ijk})}{\sum\limits_{n\in N_i}\sum\limits_{r\in R_{in}}exp(b_{inr})}
$$
위 식에서 $N_i$는 entity $e_i$의 이웃이고 $R_{ij}$는 $e_i$와 $e_j$를 연결하는 relation의 집합입니다. 

Entity  $e_i$의 새로운 embedding은  attention value에 의해 가중치가 표현된 각 triple의 sum이고 아래와 같이 구할 수 있습니다. 
$$
\vec{h_i'} = \sigma(\sum\limits_{j\in N_i}\sum\limits_{k\in R_{ij}}\alpha_{ijk}\vec{c_{ijk}})
$$
본 모델도 GAT에서 언급했던 학습 과정을 안정화시키고 더 많은 이웃에 대한 정보를 포함하기 위해서 사용되는 multi-head attention으로 구현합니다. 본질적으로 $M$개의 서로 독립적인 attention mechanism이 embedding을 계산하고 합쳐질 때 아래와 같은 식으로 연산됩니다. 이 과정이 그림 4에서 graph attention layer라고 표시된 부분입니다. 			

$$
\vec{h_i'} =\mathbin\Vert_{m=1}^M \sigma(\sum\limits_{j\in N_i}\alpha_{ijk}^m\vec{c_{ijk}^m})
$$


그리고 relation embedding 행렬인 G에 대해서도 선형변환을 가중치있는 행렬 $W^R$을 통해 진행하면 아래의 식과 같이 output relation embedding을 구할 수 있습니다. 이 때 $W^R \in R^{T \times T'}$ 이고 이때의 $T'$은 output relation embedding의 차원입니다. 
$$
G'=G.W^R
$$
GAT와 유사하게 본 모델에서도 마지막 layer에서는 concatenation 대신에 평균치를 사용합니다. 그래서 마지막 layer를 구할 때 사용되는 식은 아래와 같습니다. 
$$
\vec{h_i'} = \sigma(\frac{1}{M}\sum_{m=1}^M\sum\limits_{j\in N_i}\sum\limits_{k\in R_{ij}}\alpha_{ijk}^m\vec{c_{ijk}^m})
$$
그러나 새로운 embedding을 학습하면서 entity는 그들의 최초의 embedding 정보를 잃게됩니다. 그래서 이를 해결하기 위해, 해당 모델에서는 input entity embedding인 $H^i$를 가중치 행렬 $W^E \in R^{T^i \times T^f}$을 사용해서 선형적으로 변환하여 $H^t$를 얻습니다. 이 때, $T^i$ 는 최초의 entity embedding의 차원이고 $T^f$는 마지막 entity embedding의 차원입니다. 아래의 식과 같이 이 최초 embedding 정보를 마지막 attentional 레이어에서 얻은 entity embedding에 더합니다.  이 때, $H^f \in R^{N_e \times T^f}$입니다. 
$$
H''=W^EH^t + H^f
$$
본 모델의 설계에서는 edge의 개념을 n-hop 이웃에 대해 두개의 entity 사이의 보조 relation을 활용하여 directed path로 확장하였습니다. 이는 모든 경로에 있는 relation의 embedding에 대한 합이라고 볼 수 있습니다. 본 모델은 반복적으로 멀리 있는 이웃의 정보까지 모두 축적합니다. 

### ![Image_2](https://user-images.githubusercontent.com/22410209/93541748-3d776e00-f992-11ea-9bc6-8f4b1034867a.JPG)

그림 2에서 보면 모델의 첫 레이어에서는 모든 entity가 direct in-flowing neighbors로부터 오는 정보를 받아들입니다. 그리고 두번째 layer에는 U.S가 entity Burack Obama, Ethan Horvath, Chevrolet, Washington D.C에서 정보를 받는데 이미 이러한 entity들은 그들의 이웃인 Michelle Obama, Samuel L.Jackson과 같은 entity의 정보를 갖고 있습니다. 일반적으로 n 레이어 모델의 경우 들어오는 정보는 n-hop 이웃까지 모두 축적됩니다. 이러한 새로운 entity embeddings를 알기 위한 집합 과정과  n-hop 이웃사이의 보조 edge는 그림 2에서 확인할 수 있다. 즉, 매번 주 반복때마다 entity embedding을 매번 일반화된 GAT 레이어를 수행한 후와 first layer 전에 일반화하는 것입니다. 



### 어떻게 학습시킬 것인가

##### Train objective

Translational scoring function(소개된 논문은 [여기](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)를 참고하세요)을 사용하는데 이는 다음을 만족하는 embedding으로부터 학습합니다. 주어진 valid triple 에 대해서 $t_{ij}^k = (e_i, r_k, e_j)$에 대하여 조건 $\vec{h_i} + \vec{g_k} \approx \vec{h_j}$가 만족해야합니다. 

Ex) $e_j$가 $e_i$의 relation $r_k$로 연결된 가장 가까운 이웃

구체적으로, L1-norm dissimilarity를 줄이기 위해 entity와 relation embeddings를 학습하는 것입니다. L1-norm dissimilarity를 $d_{t_{ij}} = ||\vec{h_i} + \vec{g_k} - \vec{h_j}||$ 로 측정할 수 있습니다.

모델을 학습시킬 때에 hinge-loss를 사용하였는데 이는 아래의 수식입니다.
$$
L(\Omega) = \sum\limits_{t_{ij} \in S}\sum\limits_{t_{ij}'\in S} max\{d_{t_{ij}'}-d_{t_{ij}} + \gamma,0\}
$$
위의 식에서 $\gamma >0 $는 hyper-parameter이고 $S$는 valid triple의 집합, $S'$은 invalid한 triple의 집합입니다.

##### Decoder

본 모델은 ConvKB를 decoder로 사용한다. Convolution layer의 목표는 각 dimension에 있는 triple  $t_{ij}^k $ 의 global embedding 속성을 분석하는 것과 모델의 변하는 성질을 일반화하는 것에 있습니다. Score function은 여러개의 feature map으로 아래와 같이 표현될 수 있습니다.
$$
f(t_{ij}^k) = (||_{m=1}^\Omega RELU([\vec{h_i},\vec{g_k},\vec{h_j}] * w^m)) \cdot W\\ \ \\
\Omega: filter 개수 \quad*: convolution\ 연산 \\ W \in R^{\Omega k \times 1}: triple의 \ final score를\ 계산하기\ 위한\ 선형\ 변환
$$
이 모델은 soft-margin loss를 사용하여 학습되고 아래와 같습니다.


$$
L = \sum\limits_{t_{ij}^k}\in\{S\cup S'\}}log(1+exp(l_{t_{ij}^k} \cdot f(t_{ij}^k))) + {\lambda \over 2}||W||^2_2
\\
where \quad l_{t_{ij}^k} = \left\{ \begin{array}{ll}
         1 & \mbox{if $t_{ij}^k \in S$};\\
        -1 & \mbox{if $t_{ij}^k \in S'$}.\end{array} \right. 
$$


### 실험 및 결과

##### 데이터셋

본 논문에서는 총 5가지의 벤치마크 데이터셋을 사용하여 모델을 평가하였습니다.

* WN18RR [ Dettmers et al., 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366)
* FB15k-237 [Toutanova et al., 2015](https://www.aclweb.org/anthology/D15-1174/)
* NELL-995  [Xiong et al., 2017](https://www.aclweb.org/anthology/D17-1060/)
* Unified Medical Language Systems (UMLS)[Kok and Domingos, 2007](https://dl.acm.org/doi/10.1145/1273496.1273551)
* Alyawarra Kinship [(Lin et al., 2018). ](https://www.aclweb.org/anthology/D18-1362/)

![table_1](https://user-images.githubusercontent.com/22410209/93541878-9515d980-f992-11ea-9c62-1bad9d505571.JPG)

##### 학습 방법

먼저, Triple안에 있는 head나 tail entity를 invalid한 entity로 바꿀 때마다 두개의 invalid한 triple의 집합을 생성합니다. 그리고 head와 tail에 대한 성능이 달라지지 않게 하기 위해서 생성한 두 개의 집합에서 같은 수의 invalid triples을 랜덤으로 고릅니다. Entity 와 relation embedding은 [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)로 생성하였습니다.

Train할 때에는 본 논문에서는 2가지 step의 과정을 거쳤습니다. 먼저 그래프의 entity와 relation에 대한 정보를 encoding하기 위해 일반화된 GAT(기존 GAT과 다르게 n-hop 이웃까지 고려한 모델)를 학습합니다. 그리고 나서 ConvKB와 같은 디코더모델로 relation prediction을 위한 학습을 진행합니다. 또한 위에서 설명했듯이 보조 edge를 사용해서 sparse 그래프에 있는 이웃 정보도 더 많이 취합한다. Optimizer로 Adam을 사용했고 initial learning rate는 0.001로 학습하였습니다. 마지막 레이어의 Entity, relation embedding 는 모두 200으로 맞추었습니다. 



##### 평가 방법

Relation prediction의 목적은 $e_i$나 $e_j$가 없어졌을 때에도  triple $(e_i, r_k, e_j)$를 예측하는 것입니다.  즉, $(r_k, e_j)$만 있는 상태에서 $e_i$를 예측하거나 $(e_i, r_k)$만 있는 상태에서 $e_j$를 예측하는 것입니다. 따라서 실험 시에 각 entity $e_i$에 대해서 N-1개의 손상된 triple을 생성했고 그러한 triple마다 score를 부여하였습니다. 그리고 이 score를 증가하는 순서로 정렬하였을 때 올바른 triple에 대한 rank를 얻는 방식으로 평가하였습니다. 

평가 시 report mean reciprocal rank (MRR), mean rank(MR) 그리고 top N rank에 속하는 올바른 entity의 비율(N=1,3,10)을 사용하였습니다.



##### 결과 및 분석

![table_2](https://user-images.githubusercontent.com/22410209/93541902-a3fc8c00-f992-11ea-8959-99c25792a937.JPG)
![table_3](https://user-images.githubusercontent.com/22410209/93541905-a52db900-f992-11ea-9b96-bfdaa9a6990d.JPG)

위의 실험표를 보면 FB15k-237에는 5개의 항목 모두, WN18RR에서는 2개의 항목이 다른 대조모델에 비해 가장 성능이 좋은 것을 확인할 수 있습니다. 

* Attention values vs Epoch

  또한 본 논문에서는 epoch가 늘어남에 따른 attention의 분포에 대한 실험도 진행하였는데 결과는 아래와 같습니다.
  
  ![image_5](https://user-images.githubusercontent.com/22410209/93541927-b676c580-f992-11ea-9be5-bab432e5872b.JPG)
  
  초기 단계의 학습 과정 떄에는 attention이 랜덤하게 분포한 것을 확인할 수 있는 반면, 학습을 더 진행 할수록 만든 모델로 인해 이웃에 있는 정보를 모으고 직접 연결된 이웃에는 더 많은 attention이, 멀리 있는 이웃에는 상대적으로 적은 attention이 분포하는 것을 확인할 수 있습니다. 
  
* PageRank 분석

  또한 본 논문에서는 복잡하고 숨겨진 multi-hop relation들이 밀집도가 높은 그래프에서 더 간결하게 찾을 수 있다고 가정하고 이를 확인하였습니다. Mean PageRank와DistMult에 따른 MRR의 증가의 상관관계를 분석하였습니다. 그리고 분석 결과, r=0.808일 경우에 강한 상관관계를 찾았습니다. 

  ![table_4](https://user-images.githubusercontent.com/22410209/93541951-c55d7800-f992-11ea-94f1-45b1953e7f0f.JPG)

  위의 테이블을 보면PageRank의 증가가 있을 때, MRR값도 증가하는 것을 보여줍니다.  결과를 보면 WN18RR의 경우에는 증가하지 않는 것을 볼 수 있는데 이 이유를 본 논문에서는  WNI18RR은 아주 sparse하고 계층적인 구조를 갖고 있기 때문이라고 설명합니다. 이러한 구조는 본 모델에서 사용하는 top-down 재귀 방법에서 정보를 캡쳐하지 않기 때문에 증가하지 않는 것으로 분석합니다. 이 부분이 본 모델의 한계점 및 단점이라고 생각합니다. 
