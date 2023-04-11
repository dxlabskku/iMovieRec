# iMovieRec: a hybrid movie recommendation method based on user-image-item mode
Movie repository is to supplement the paper "iMovieRec: a hybrid movie recommendation method based on user-image-item mode".

## Abstract
We propose iMovieRec, a hybrid movie recommendation method that employs an image-user-item model, which utilizes both CF models and graph features. The purpose of this model is to efficiently learn the interactions between users and items and the key features of the poster images using single layer neural networks and matrix factorization. In particular, we consider various types of graph architectures to determine the graph structure that would express the relationship between users and items. The experimental results obtained using two benchmarking datasets indicate that iMovieRec is more efficient than the other recommendation models, which exhibit limited and varied image feature effects. In addition, we make both our datasets and the iMovieRec model publicly available.


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
