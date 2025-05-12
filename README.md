# Handwritten Number Recognition
Machine Learning Model trained from scratch to recognize handwritten numbers.

## Machine Learning Pipeline
Raw Data -> Data Preprocessing -> Exploratory Data Analysis -> Feature Selection -> Model Construction -> Model Evaluation -> Model Deployment on new data.

### Dataset Selection (Raw Data):
[![MNIST Dataset](https://img.shields.io/badge/MNIST-Dataset-blue?style=flat&logo=readthedocs)](https://docs.ultralytics.com/datasets/classify/mnist/)
---

## Running the flask for development
Install flask first
```
conda install flask
```
or
```
pip install flask
```

Then, run app.py (assuming you have all dependencies: scikit-learn, tensorflow-macos, )
```
python app.py
```

It will usually run in local port 3000. You will find a message similar to:
```
 * Running on all addresses (0.0.0.0)
 * Running on http://120.1.1.3:3000
 * Running on http://10.8.4.3:3000
```
go to that link, and voila! You can test the logistic regression model working!

