# Cost-Effective Interactive Attention Learning with Neural Attention Processes
This is the **TensorFlow implementation** for the paper "Cost-Effective Interactive Attention Learning with Neural Attention Processes (**ICML 2020**) : https://arxiv.org/abs/2006.05419

## Abstract
<p align="center">
<image width="950", height="400" src="/images/ial_concept_figure.png">
  
We propose a novel interactive learning framework which we refer to as Interactive Attention Learning (IAL), in which the human supervisors interactively manipulate the allocated attentions, to correct the model's behaviour by updating the attention-generating network. However, such a model is prone to overfitting due to scarcity of human annotations, and requires costly retraining. Moreover, it is almost infeasible for the human annotators to examine attentions on tons of instances and features. We tackle these challenges by proposing a sample-efficient attention mechanism and a cost-effective reranking algorithm for instances and features. First, we propose Neural Attention Processes (NAPs), which is an attention generator that can update its behaviour by incorporating new attention-level supervisions without any retraining. Secondly, we propose an algorithm which prioritizes the instances and the features by their negative impacts, such that the model can yield large improvements with minimal human feedback. We validate IAL on various time-series datasets from multiple domains (healthcare, real-estate, and computer vision) on which it significantly outperforms baselines with conventional attention mechanisms, or without cost-effective reranking, with substantially less retraining and human-model interaction cost.

__Contribution of this work__
- We propose a __novel interactive learning framework__ which iteratively updates the model by interacting with the human supervisor via the generated attentions.
- To minimize the retraining cost, we propose a __novel probabilistic attention mechanism__ which sampleefficiently incorporates new attention-level supervisions on-the-fly without retraining and overfitting.
- To minimize human supervision cost, we propose an __efficient instance and feature reranking__ algorithm, that prioritizes them based on their negative impacts on the prediction, measured either by uncertainty, influence function, or counterfactual estimation.
- We validate our model on __five real-world datasets__ with binary, multi-label classification, and regression tasks, and show that our model obtains significant improvements over baselines with substantially less retraining and human feedback cost.

__Structure of Neural Attention Processes (NAP)__
- NAP can incorporate new labeled instances into the context to immediately change its attention-generating behaviour without retraining via amortization, which also allows the annotator to see the effect of his/her annotation on the prediction on the fly.
<p align="center">
<image width="820", height="210" src="/images/nap.png">

__Structure of Cost-Effective Reranking Algorithm (CER)__
- CER prioritizes them based on their negative impacts on the prediction, measured either by uncertainty, influence function, or counterfactual estimation.
<p align="center">
<image width="800", height="300" src="/images/cer.png">


## Prerequisites
- Python 3.5
- Tensorflow 1.14.0
- CUDA 10.0
- cudnn 7.6.5

Install the python packages:
```
$ pip install flask
$ pip install opencv-python
$ pip install pyqt5
```


### Data Preparation
Go to the folder of each dataset (i.e. ```data/EHR```, ```data/Finance```, or ```data/Squat```) to check. All datasets have been already preprocessed.

## Run

__Risk Prediction Tasks (Heart Failure/Cerebral Infarction/CardioVascular disease)__
```
Cerebral Infarction
- Initial Round & Execute uncertainty-counterfactual combination
$ python train.py

- Online Annotation Interface (Flask)
$ python ./hil_medical_annotator/main.py

- Further Round (Retraining)
$ python retrain.py
```

## Online Annotation Interface
<img align="middle" width="800" src="/images/EHR.gif">

## Run - (Healthcare) Risk Prediction Tasks
```
- Online Annotation Interface (Flask)
$ python ./hil_medical_annotator/main.py
```

<img align="middle" width="800" src="/images/Squat.gif">

## Run - (Fitness) Squat Posture Correction Tasks
```
- Initial Round Sampling
$ cd ./hil_squat_annotator/model
$ python Sampling.py

- Online Annotation Interface (PyQt5)
$ cd ..
$ python hil_keypoint.py
```

## Results 
### 1. Risk Prediction Tasks
The results in the main paper (Final AUC over five training rounds):
|       | Heart Failure | Cerebral Infartion | CardioVascular Disease (CVD)|
| ------| ---------------- | ----------------- | ------------------ |
| Random-UA  | 0.6231 ± 0.03       | 0.6491 ± 0.01        | 0.6112 ± 0.02    |
| Random-NAP | 0.6414 ± 0.01       | 0.6674 ± 0.02        | 0.6284 ± 0.01    | 
| AILA       | 0.6363 ± 0.03       | 0.6602 ± 0.03        | 0.6193 ± 0.02| 
| IAL-NAP    | __0.6612 ± 0.02__   | __0.6892 ± 0.03__    | __0.6371 ± 0.02__|   

__IAL-NAP Combinations__
| Instance-level  | Feature-level               | Heart Failure      | Cerebral Infartion   | CardioVascular Disease|
| ------  | ------        | ---------------- | -----------------   | ------------------   | 
| Influence Function  | Uncertainty          | 0.6563 ± 0.01       |  0.6821 ± 0.02       | 0.6308 ± 0.02     |
| Influence Function  | Influence Function   | 0.6514 ± 0.02       |  0.6825 ± 0.01       | 0.6329 ± 0.03     | 
| Influence Function  | Counterfactual       | 0.6592 ± 0.02       |  __0.6921 ± 0.03__   | __0.6379 ± 0.02__ | 
| Uncertainty         | Counterfactual       | __0.6612 ± 0.01__   |  0.6892 ± 0.03       | 0.6371 ± 0.02     |   
 

### 2. Real Estate Forecasting Task (Forecasting housing sales transaction price)
|            | Mean-Percentage error |
| ------     | ----------------      |
| Random-UA  | 0.2222 ± 0.04         |
| Random-NAP | 0.2061 ± 0.01         | 
| AILA       | 0.2119 ± 0.01         |
| IAL-NAP    | __0.1835 ± 0.02__     |  

__IAL-NAP Combinations__
| Instance-level      | Feature-level        | Mean-Percentage error| 
| ------              | ------               | ----------------     | 
| Influence Function  | Uncertainty          | 0.1921 ± 0.01        |  
| Influence Function  | Influence Function   | 0.1865 ± 0.02        | 
| Influence Function  | Counterfactual       | 0.1863 ± 0.02        | 
| Uncertainty         | Counterfactual       | __0.1835 ± 0.01__    | 


### 3. Squat Pose Correction Task
|            | Mean Accuracy         |
| ------     | ----------------      |
| Random-UA  | 0.8521 ± 0.02         |
| Random-NAP | 0.8525 ± 0.01         | 
| AILA       | 0.8425 ± 0.01         |
| IAL-NAP    | __0.8689 ± 0.01__     |  


__IAL-NAP Combinations__
| Instance-level      | Feature-level        | Mean Accuracy       | 
| ------              | ------               | ----------------    | 
| Influence Function  | Uncertainty          | 0.8712 ± 0.01       |  
| Influence Function  | Influence Function   | 0.8632 ± 0.01       | 
| Influence Function  | Counterfactual       | 0.8682 ± 0.01       | 
| Uncertainty         | Counterfactual       | __0.8689 ± 0.02__   |



### Retraining time to retrain examples of human annotation
```
*** Retraining time ***
- Heart Failure
                 s=1       s=2       s=3      s=4 
 Random-UA    31.1532s  34.5223s  39.2324s  38.2094s
   AILA       43.2324s  42.2102s  45.4364s  47.1129s
 Random-NAP    9.2445s   8.2309s   9.2320s   9.1083s
  IAL-NAP      9.2309s   8.3693s   9.1129s   9.0324s

- Cerebral Infarction
                 s=1       s=2       s=3       s=4 
 Random-UA    63.3984s  50.3209s  49.1896s  50.9103s
   AILA       49.8931s  45.2804s  60.0425s  58.3929s
 Random-NAP   22.5792s  18.4052s  18.9384s  17.9374s
  IAL-NAP     18.3982s  19.1423s  18.7834s  16.8199s

- CardioVascular Disease (CVD)
                 s=1       s=2       s=3       s=4 
 Random-UA    74.2351s  75.5424s  77.9324s  78.2088s
   AILA       46.2524s  47.2396s  69.2441s  73.2692s
 Random-NAP   21.7324s  27.5324s  31.6341s  28.2392s
  IAL-NAP     29.8324s  28.2334s  29.7326s  27.2044s
  
- Real Estate Forecasting
                 s=1        s=2        s=3        s=4 
 Random-UA    239.2379s  236.5408s  237.9478s  239.2818s
   AILA       228.2123s  261.5464s  241.9364s  162.7389s
 Random-NAP   148.7354s  163.5324s  169.1341s  162.3813s
  IAL-NAP     164.8328s  163.1334s  147.5326s  150.9381s

- Squat Posture
                 s=1       s=2       s=3       s=4 
 Random-UA    32.1194s  27.0938s  32.9482s  32.2984s
   AILA       25.8931s  24.9374s  24.1850s  23.3081s
 Random-NAP    7.2324s   8.9034s   8.2984s   8.9374s
  IAL-NAP      7.2183s   7.2314s   6.1254s   8.2109s
```

### Mean Response Time (mean-RT) of human labeling
```
*** Mean Response Time (mean-RT) ***
- Heart Failure
                 s=1        s=2        s=3       s=4 
 Random-NAP   204.2236s  198.9244s  179.8526s  174.4985s
  IAL-NAP     155.9324s  150.8924s  131.9224s  139.9074s

- Cerebral Infarction
                 s=1        s=2        s=3        s=4 
 Random-NAP   184.2939s  182.5246s  189.5029s  179.8127s
  IAL-NAP     141.2843s  128.8344s  132.5524s  129.3053s

- CardioVascular Disease (CVD)
                 s=1        s=2        s=3        s=4 
 Random-NAP   250.3955s  239.8921s  226.2995s  231.4734s
  IAL-NAP     192.2392s  173.5641s  171.3423s  165.9254s

- Real Estate Forecasting
                 s=1        s=2        s=3        s=4 
 Random-NAP   377.6324s  319.8921s  316.2O95s  289.5034s
  IAL-NAP     299.2941s  251.5634s  243.3423s  240.8254s
  
- Squat Posture
                  s=1       s=2         s=3       s=4 
 Random-NAP   124.2324s  131.2324s  128.2324s  114.2324s
  IAL-NAP      96.2973s   81.7391s   80.2393s   78.2924s

```

## Qaulitative Analysis
### Visualization of Attention Weights
<img align="center" width="800" src="/images/qualitative_analysis.png">

| Feature      | Meaning      |
| ------     | ----------------      |
|  __Age__     | Age                   |                
|  __HDL__     | High-densitylipoproteins holesterol  |   
|  __Smoking__ | Whether currently smokes a cigarette  |
|  __SysBP__   | Systolic blood pressure  |
|  __LDL__     | Low-density lipoprotein cholesterol.    | 

- Visualization of attentions for a selected patient on CardioVascular Disease (CVD) prediction task. Contribution indicates the extent to which each individual feature affects the onset of CVD in 1 year. 


## Citation
If you found the provided code useful, please cite our work.
```
@inproceedings{heo2020cost,
  title={Cost-Effective Interactive Attention Learning with Neural Attention Processes},
  author={Jay Heo and Junhyeon Park and Hyewon Jeong and Kwang Joon Kim, and Juho Lee and Eunho Yang and Sung Ju  Hwang},
  booktitle={ICML},
  year={2020}
}
```

