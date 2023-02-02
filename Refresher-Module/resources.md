### Intro to ML

1. Why do we prefer parameters with small norms for regularization?
   References:
   https://towardsdatascience.com/visualizing-regularization-and-the-l1-and-l2-norms-d962aa769932
   https://arunm8489.medium.com/an-overview-on-regularization-f2a878507eae
   Another intuition: For classification, we want to maximize y.log(h(θ)), i.e., maximize θ'.x; where (θ' = transpose of θ). Hence, if the norm of θ is high,
   θ'x will be high and hence the model will not generalize well. Hence, we prefer smaller normed θ's.

2. On what factors does the selection of a metric depends?
   References:
   https://levelup.gitconnected.com/a-practical-guide-to-choosing-the-right-metric-for-evaluating-machine-learning-models-ae3712bd01e9
   https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

### Intro to PyTorch

1. Transfer Learning Notebook(Finetuning a Pretrained model in pytorch):-
   https://github.com/LeanManager/PyTorch_Image_Classifier/blob/master/Image_Classifier_Project.ipynb

2. Resource regarding Convolution and Pooling:-
   https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/

3. Link for the dataset used in the tutorial (Kindly access the link via your IIITD account):-
   https://drive.google.com/drive/folders/1ZQMgH0byHeeIbCKfO-GBU6ImqY4_SYbB?usp=sharing

4. Original and Complete dataset Link:-
   https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
