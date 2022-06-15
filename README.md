# SEKT
SEKT : A Spacing-Effect based model for Knowledge Tracing

![image](https://user-images.githubusercontent.com/49023717/173782680-8f3f5490-5f36-43b4-9248-d9fa5a4d33dc.png)

Using the forgetting curve considered in the spacing effect, the model was trained to predict higher than the original prediction value whenever the user repeats the sentence.

## Datasets

origin:
> http://sharedtask.duolingo.com/2018.html
> 
> https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO


processed data:
> https://drive.google.com/file/d/1Owf4vvmdx9voQ9am11tso1OA186HKgGa/view?usp=sharing



## How to train

`!python3 train.py --model_name=SEKT --dataset_name=DUOLINGO+`

## Result

|model|AUC|
|:---:|:---:|
|DKT|0.685±0.06|
|SEKT|0.720±0.01|
