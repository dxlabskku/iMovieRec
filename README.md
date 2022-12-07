# iMovieRec: a hybrid movie recommendation method based on user-image-item mode
Movie repository is to supplement the paper "iMovieRec: a hybrid movie recommendation method based on user-image-item mode".

## Abstract
Among the recommendation models, the collaborative filtering recommendation system is the most frequently used method and shows high accuracy performance. However, this method has limitations in improving additional performance due to the problem of sparse rating matrix. To compensate for this problem, many studies have expected performance improvements by adding external and internal features as an input. Because user information is difficult to use due to privacy issues, various graph features and item information are recently considered. In this work, we proposed iMovieRec, a hybrid movie recommendation method using a user-image-item model, by utilizing both collaborative filtering models and graph features. The experimental results with two bench-marking datasets indicate that iMovieRec is more efficient than other recommendation models, while effects of image features are limited and varied in other models. 


## Overview of our framework
<img src="https://user-images.githubusercontent.com/43632309/105990739-43baeb00-60e6-11eb-8117-a12310ccc655.png" width="480" height="303">
<strong>Figure 1 : iMovieRec model</strong>
<br>
<img src="https://user-images.githubusercontent.com/43632309/105991281-effcd180-60e6-11eb-8cd4-b2420b0329c4.png" width="613" height="168">
<strong>Figure 2 : Image-feature embedding to traditional matrix factorization models</strong>


## Clone
```
git clone https://bit.ly/2YEix9V
```


## Dataset
MovieLens 100K : https://grouplens.org/datasets/movielens/100k/

MovieLens 1M : https://grouplens.org/datasets/movielens/1m/


## Model Description
* graphrec_image.py : this file provides training GraphRec model adding image features.
* image_processing.py : this file provides an extraction of the image features in the GraphRec by inserting them into the CNN.
* models_mf.py : this file provides a set of basic matrix factorization models.
* preprocess.py : this file proivdes conducting label encoding of user and item columns.
* train_test_mf.py : this file provides training and testing for basic matrix factorization models.
* util.py : this file provides supplements for graphrec_image.py
