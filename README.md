## DeepFake Image Detection

### Problem Statement

With the advent of Generative Adversarial Network (GAN) and other deep learning based DeepFake techniques, the immediate challenge we face as a community is how to assess the validity of online material be it machine learning derived images or videos. We are faced with an unprecedented potential for an extreme violation of basic human rights along with a fundamental unavoidable change in how humans interact socially. We have already seen evidence of maligning and manipulation of news headlines, medical (dis)information along with abuse of individual privacy. The goal of this proposed project is to use an online image database to effectively detect DeepFake images. This paper focuses on use of convolutional neural networks for classification of true versus fake images obtained from a large online database. We aimed to compare three different convolutional neural networks:

1. VGGFace16
2. DenseNet-121
3. 3 Custom CNN Architectures. 

Future work would include the use of unsupervised clustering methods / auto-encoders explore if true versus fake images cluster separately and also to add transparency and interpretability to our models by use of CNN visualization methods. 

For all the models we extracted the output of the last layer before classification to see if the vectors are representative of the images. The vectors have dimensions of 512, 2048, 1024 for custom, VGGFace and DenseNet models. This was entirely too large, therefore, we used principal component analysis (PCA) to keep points that contributed the most in terms of variability. By running PCA, we were able to retain 50 principal components. We then used support vector machine (SVM) with polynomial kernel to classify the retained components into the two classes (real or fake). We also looked at the PCA visualization and found that most architectures were able to learn the differentiating patterns of real vs. fake images and distinct clusters could be seen against the first 2 principle components.

> Detailed analysis, performance metrics and inferences are provided in the report.

### Dataset Source

Two datasets from Kaggle were taken and combined together. Since the datasets are too large they are not pushed to the repository. Please download the following datasets:

1. https://www.kaggle.com/xhlulu/140k-real-and-fake-faces
2. https://www.kaggle.com/ciplab/real-and-fake-face-detection

Once downloaded create a folder called `combined-real-and-fake-faces/combined-real-vs-fake`. Within this folder should be 3 subfolders: `train`, `valid` and `test`. Combine the second dataset with the first one.

### How to Run?

The notebooks within the specific arch. are indepedent and can be run parallely. Each of these notebooks will save the `.h5` models. Place all the saved models in the folder called `models`. 

> Please note that pretrained models could not be uploaded due to file size restrictions.

Once the models are saved run the `performance-eval` notebook to see the performance comparision of the various models and extracting the last layer of the network which gives the vector representation of the images which was learnt by the model.

After this run the `pca_svm` notebook to look at the PCA visualizations and perform classification using SVM.