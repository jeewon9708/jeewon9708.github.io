---
layout: post
title:  "KBAT paper의 KG-ReEval paper를 통한 분석"
date:   2020-11-17T16:25:52-23:00
author: Jeewon Chae
categories: Paper_review
use_math: true
---



해당 블로그 포스트는 ACL 2019 에 게재된 [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://www.aclweb.org/anthology/P19-1466/) 논문의 심층 분석을 위하여 작성되었습니다. ACL 2020논문 [A Re-evaluation of Knowledge Graph Completion Methods](https://arxiv.org/pdf/1911.03903v3.pdf)을 활용한 분석입니다.  각 논문의 실험 코드는 [KBAT github](https://github.com/deepakn97/relationPrediction), [KG-ReEval github](https://github.com/svjan5/kg-reeval)를 참고하시면 됩니다. 

### KG-ReEval 논문

본 논문은 많은 지식 그래프 completion, 즉 relation prediction에 대한 연구 성능에 대한 의문을 제시합니다. 이전 연구들의 성능이 매우 좋다고 발표되었으나 이는 부적절한 평가 프로토콜 때문이라고 주장합니다. 또한, 데이터셋에 따라 결과값이 매우 다른점을 지적합니다. 

아래와 같이 다시 실험하였을 때의 결과가 이전 연구의 논문에서 발표된 결과와 다르다고 주장합니다. 

먼저 relation prediction은 아래의 그림과 같이 entity(node) 2개가 주어졌을 때 주변 환경, 그래프를 보고 그 관계를 예측하는 일입니다. 

![image-20201120190746832](C:\Users\jeewo\AppData\Roaming\Typora\typora-user-images\image-20201120190746832.png)

이 relation prediction을 평가하기 위해서는 (e1,e2,r)이 있을 때 (e1,e2)로부터 r이 될 수 있는 모든 경우 (T')에 대해 score를 매기고 이를 sorting하여 올바른 r을 예측합니다.

이 때, 논문에서 지적한 부분은 **"score가 같은 triplet에 대해 어떻게 처리할까"**에 대한 부분이었습니다. 

먼저 현재 사용되고 있는 방법은 TOP 방법으로 valid한 triplet를 sort할 때 맨 앞으로 두어 같은 score값이 있을 때 언제나 100%의 확률로 valid한 결과로 평가되어왔습니다. 아래의 그림과 같이 같은 score인 c를 가진 여러 경우에도 t가 선택되는 것을 볼 수 있습니다.

![image-20201120191334827](C:\Users\jeewo\AppData\Roaming\Typora\typora-user-images\image-20201120191334827.png)

즉, 만약에 점수가 같은 triplet이 여러개 있을 경우 어떠한 triplet을 정할 것인지가 중요한데 위의 몇몇 연구들에서는 이 점을 제대로 짚고 넘어가지 않았습니다. 그래서 KG-ReEvaluation 논문에서는 정답 후보인 T'에서 valid한 triplet을 항상 가장 앞에 있는 triplet에 둘지, 가장 뒤에 둘지 아니면 랜덤으로 고를지에 따라 다른 결과가 나오는 것을 위의 표를 통해 보여주었습니다.

Top을 고를 때에는 상대적으로 성능이 더 좋게 나오는데 이는 이 방법이 모델을 엄밀하게 평가하지 않기 때문입니다. 이 방법은 서로 다른 triplet임에도 같은 점수를 주는 bias를 가진 모델에게 적절한 advantage를 줍니다. 반대로 Bottom을 고를 때에는 inference time에 있는 모델에게 불공정합니다. 왜냐하면 이 방법은 모델이 multiple triplets에게 같은 점수를 주는 것에 부적절한 처벌을 주기 떄문입니다. 예를 들어 많은 triplet이 올바른 correct triple과 같은 점수를 갖는 경우 correct triplet이 가장 낮은 rank를 갖는 것이 가능하다는 것입니다. 

그래서 논문에서는 결론적으로 Random이 가장 좋은 평가 테크닉이라고 판단하고 이를 통하여 비교하였습니다.

![res](https://user-images.githubusercontent.com/22410209/99682052-2fdd8200-2ac2-11eb-9244-a4587d50ce76.JPG)

위에서 사용한 Top, Bottom, Random은 아래와 같습니다.

~~~
T': candidate set

• TOP: the correct triplet is inserted in the beginning of T'

• BOTTOM: the correct triplet is inserted at the end of T'

• RANDOM: the correct triplet is placed randomly in T'

~~~

위의 결과를 보면 이전 포스팅에서 분석한 논문의 결과값과 다른 것을 확인할 수 있습니다.



### 재실험

위와 같이 결과값이 서로 다르기 때문에 재실험을 진행하였습니다. 이전 포스팅에서는 FB15k-237 데이터셋이 지나치게 크고 실험시간이 지나치게 오래 걸려 진행하지 못했으나 이번 포스팅에서는 비교를 위해 진행하였습니다.

1. **KBAT 논문 FB15k-237 실험 결과**

   * 실제 실험 결과

   ![kbat_150](https://user-images.githubusercontent.com/22410209/99739191-891fd280-2b0f-11eb-80df-fde01da3dc7c.JPG)



* 논문의 실험 결과

  ![kbat_paper_res](https://user-images.githubusercontent.com/22410209/99684149-792ed100-2ac4-11eb-9cfd-1426337737f4.JPG)



* 실험을 재연한  결과

![fff](https://user-images.githubusercontent.com/22410209/99739228-9fc62980-2b0f-11eb-97b2-82a40f5734fb.png)

논문의 실험 결과값이  발표된 값과 모두 유사한 것을 알 수 있습니다.



2. **두번째 KG-ReEval 논문에서 제시한 방법으로 실험**
   * 실제로 실험 결과

![ffffff222222](https://user-images.githubusercontent.com/22410209/99739843-0ac43000-2b11-11eb-8012-5472aab2e595.png)

* 실험 재연 결과 표

![fff222](https://user-images.githubusercontent.com/22410209/99739844-0c8df380-2b11-11eb-8f16-57bcedd05fe5.png)





3. **위의 두 실험의 결과 차이**
   * 비교 표

![compare](https://user-images.githubusercontent.com/22410209/99740232-e6b51e80-2b11-11eb-9cbe-041f83c35672.png)



* 비교 차트

  ![image-20201120193936634](https://user-images.githubusercontent.com/22410209/99791170-99b06700-2b68-11eb-84fa-a49e761202ae.png)

  

  *MR: 작을수록 성능이 좋은 것*

  *MRR: 클수록 성능이 좋은 것*

  *Hits@N: 클수록 성능이 좋은 것*

  ![res](https://user-images.githubusercontent.com/22410209/99682052-2fdd8200-2ac2-11eb-9244-a4587d50ce76.JPG)						

위의 제가 실험한 두 표를 보면 논문에서 발표한  표와 완벽하게 일치하지는 않지만 Re-Evaluation 논문에서 제안한 방법으로 실험했을 경우 더 성능이 좋지 않다는 점은 확실합니다.



### 결과 분석

##### Data Leakage

먼저, KG-ReEvaluation 논문에서는 기존의 KBAT 논문에서 진행한 실험 코드에 data leakage가 있다고 지적합니다. 

*Data leakage*란 training 데이터 밖에서 유입된 정보가 모델을 만드는데 사용되는 것을 의미합니다. 이로 인해 완전히 잘못된 예측 모델이 만들어지거나 오버피팅된 결과를 낳기도 합니다. 

머신러닝에서 가장 중요하고 또 어려운 부분은 학습 시 갖고 있는 데이터(train dataset)으로부터 모델을 학습시켜 갖고 있지 않은 데이터(test dataset)에 대한 예측을 "잘"하는 일인데 이 때 갖고 있을 수 없는 데이터를 실험 시 추가하였을 경우 데이터 누수가 발생하여 의미없는 실험 결과를 가집니다. 

위와 같은 실수를 하지 않기 위해서는 train data, validation data, test data에 대한 이해가 필요합니다.

먼저 train data와 validaton data는 training 과정에서 사용합니다. 이 때 train data로 학습을 진행하고 training 과정에서 중간에 모델이 잘 학습하고 있는지를 검증하기 위해 validation data를 사용합니다. 

학습이 끝나고 나면 실제로 처음보는, 한번도 보지 못한 데이터에 대한 평가를 진행하기 위해 test data가 필요합니다. 이 데이터가 최종적으로 모델 평가 지표가 됩니다. 

기존 KBAT 코드를 보면 train batch에 validation dataset을 이용하기 때문에 데이터누수가 있다고 여겨 이를 해결한 아래와 같은 코드로 실험을 진행한 것입니다. 

![image-20201120183208867](https://user-images.githubusercontent.com/22410209/99791243-b2b91800-2b68-11eb-8ea9-844d4b2bc7c8.png)



따라서 기존 KBAT 논문의 실험에 문제가 있다는 것을 밝혔습니다.



### 결론

다음 포스팅에서는 KBAT 논문에 실험에 위와 같은 data leakage가 있다 하더라도 KBAT의 아이디어대로라면 성능이 더 좋아져야하는데 왜 더 좋아지지 않았는지에 대해 알아보도록 하겠습니다.