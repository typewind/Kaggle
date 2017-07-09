# Introduction
3 different solutions for the [Data Science London + Scikit-learn](https://www.kaggle.com/c/data-science-london-scikit-learn).

# How to use
0. Find the `line 10` to change the file path
1. Find the `line 82-88`
2. Choose the function by add/delete the `#`
3. If you need to test the DNN, delete the `#` at `line 7-8`. Reading Keras model is very slow so I commented them
4. Run the code

# Score:
- GMM+Random Forest: 0.99218 
- Simple SVC: 0.93443 (n_components=12)
- DNN:0.85544

# Analysis
Please read my corresponding blog: [Notes of Data Science London + Scikit-learn](https://typewind.github.io/2017/07/05/k-london-notes/)

# Reference
1. [Tutorials - Data Science London + Scikit-learn](https://www.kaggle.com/c/data-science-london-scikit-learn/visualization/1091) by William Cukierski. Very good introduction of common functions (PCA, SVM, cross-validation, etc., in sklearn). I love this cheat sheet from it. Just Perfect. I should print it out or set it as a desktop XD.
2. [Achieve 99% by GMM](http://nbviewer.jupyter.org/gist/luanjunyi/6632d4c0f92bc30750f4)
3. [Sklearn: Gaussian Mixture Model](http://scikit-learn.org/stable/modules/mixture.html)
