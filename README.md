# Spot insulting online comments
## Data challenge at Télécom ParisTech, in June 2016.
- **Goal**: Being able to state whether an online comment is insulting or not using supervised learning algorithms.
- One constraint of this challenge is not to use external machine learning libraries such as sklearn.
- Repository contains training and test datasets (**data** folder), and a notebook (**insulting_comments.ipynb**) that generates a prediction file (**y_pred.txt**), using preprocessing from **data_transform.py** and stochastic gradient descent from **optimisation.py**.
-With a proper hyper-parameter tuning, an accuracy close to 80% can be reached. 
