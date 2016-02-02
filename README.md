# Will_Bill_Solve_It
Hacker Earth Machine Learning Competition

# Data Preparation

* Merge user's submission log with user and problem details
* Created new features like num_attempts at a problem, only took 200,000 examples from around 400,000 available.


# Feature Engineering

* Created some features related to user's problem solving capability, how hard a problem is and used rest of the features specified in the examples
* One hot encoding of user's skills
* Label Encoding categorical Features

# Modelling

* Trained Logistic Regression Model, Random Forest Classifier, Extra Trees Classifier, SGD and XGBoost Model
* But final model was an Extreme Gradient Boosting Model with parameters that produced the best accuracy on the cv set


