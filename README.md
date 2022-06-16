# SEKT
SEKT : A Spacing-Effect based model for Knowledge Tracing

## Summary

![image](https://user-images.githubusercontent.com/49023717/173782680-8f3f5490-5f36-43b4-9248-d9fa5a4d33dc.png)

Using the forgetting curve considered in the spacing effect, SEKT train to predict higher than the original prediction value of the sentence whenever the user repeats the sentence.

Spacing Effect에서 고려하는 망각곡선을 사용하여, 사용자가 같은 품사구성의 문장을 반복할때마다 모델이 도출한 해당 문장의 원래 예측값보다 높게 예측하도록 학습한다. 

<br/>

## Datasets

original data:
> http://sharedtask.duolingo.com/2018.html
> 
> https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO


processed data:
> https://drive.google.com/file/d/1Owf4vvmdx9voQ9am11tso1OA186HKgGa/view?usp=sharing

<br/>

## Custom Loss Function to reflect Spacing Effect

The fact that the Spacing Effect occurred by repeating the same part-of-speech composition means that the probability of correcting the currently repeated sentence is higher than when practicing a sentence with a composition that has not been seen before. Therefore, if the composition of the current sentence to be predicted is a sentence that the user has practiced before, the probability of matching is higher and reflected in the learning.

같은 품사구성의 문장을 반복하여 Spacing Effect 가 일어났다는 것은, 현재 반복하고 있는 문장에 대해 맞출확률이 이전에 본적이 없는 구성의 문장을 연습할 때보다 높다라는 것이다. 따라서 현재 예측하고자 하는 문장의 구성이 이전에 사용자가 연습한 적이 있는 문장이라면, 맞출확률을 더 높게 보고 학습에 반영했다. 

### New Sentence Predict Value (Add)

![image](https://user-images.githubusercontent.com/49023717/174065251-ce5d8d35-8e13-47e2-80c4-542b3c76631d.png)

Sentence prediction value = Probability originally predicted by the model + Probability originally predicted by the model * Weight (based on forgetting curve)

Considering how many times the sentence is repeated, the weight is assigned as 1/2 for the second practiced sentence and 1/4 for the third practiced sentence. (A negative power of 2 as much as the number of sentence repetitions. 2^-(repeat num - 1) )

문장 예측값 = 모델이 원래 예측한 확률 + 모델이 원래 예측한 확률 * 가중치(망각곡선기반) 으로 정의했고, 여기서 가중치란 문장이 몇 번째 반복해서 나았는지를 고려하여, 두 번째 연습하는 문장이라면 1/2, 세 번째 연습하는 문장이라면 1/4로 반복횟수만큼 2의 마이너스 제곱수를 할당했다. 

### Total Loss : POS Loss + Sentence Loss

![image](https://user-images.githubusercontent.com/49023717/174065263-f81c08a6-6cd2-49d9-972c-b4a5ce013bd3.png)

POS(Part Of Speech) Loss is Loss comparing the predicted value for each token. (part of speech) : used in DKT
Sentence Loss is the sum of the Loss obtained by multiplying the error between the newly defined prediction value and the target value by the weight.
Finally, SEKT is created that adds the loss per part of speech and loss per sentence and reflects it in learning.

품사의 예측값과 실제값의 차이를 반영한 로스는 기존 DKT에서 사용된 Binary Cross Entropy를 그대로 따르고, 새롭게 정의한 예측값과 타겟값 간의 오차를 가중치만큼 곱한 값을 모두 더해 문장 로스로 저장한다. 최종적으로 기존 품사별 로스와 문장별 로스를 더해 학습에 반영하는 모델이 만들어진다. 

<br/>

## How to train

`!python3 train.py --model_name=SEKT --dataset_name=DUOLINGO+`

<br/>

## Result

|model|AUC|
|:---:|:---:|
|DKT|0.685±0.06|
|SEKT|0.720±0.01|

<br/>

## Demo (Part-Of-Speech Recommendation System)

> GitHub https://github.com/Nayeon12/Duolingo-Personalized-Part-of-Speech-Recommendation-System

> Demo video https://www.youtube.com/watch?v=rD7dqu7ULUs

