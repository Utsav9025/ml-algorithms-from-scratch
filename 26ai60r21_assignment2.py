
# Assignment - 2 : [20 Marks]

## 1. Import required libraries and load the load_Wine Dataset from scikit-learn[MARKS 0]

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

"""## 2.Exploratory Data Analysis[2 Marks]

Perform the following analyses.

(a) Check
- Missing values
- Data types

(b) Visualize Class distribution

(c) Draw Correlation Heatmap

(d) Plot Histograms

(e) Generate Pair Plot

(f) Write your observations for each visualization.
"""

wine = load_wine()
X = wine.data
y = wine.target

df = pd.DataFrame(X,columns = wine.feature_names)
df["target"] = y

df.info()
df.shape

print("Columns containing Non-Null values : ")
df.isnull().sum()

sns.countplot(x="target", data=df)
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Class Distribution")
plt.show()

corr = df.corr()
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(corr,annot=False,cmap="coolwarm",linewidths=0.5)
plt.tight_layout()
plt.show()

print(corr)

df.hist(figsize=(12,14),bins = 20,edgecolor = "black")
plt.tight_layout()
plt.show()

selected = ["flavanoids","od280/od315_of_diluted_wines","total_phenols","proline"]
sns.pairplot(df[selected])
plt.show()

"""### Observations

**1. Class Distribution**
* The three classes have different numbers of samples.
* However, the class distribution is reasonably balanced.
  
**2. Correlation Matrix**
* total_phenols and flavanoids are strongly positively correlated.
* flavanoids and od280/od315_of_diluted_wines also show strong positive correlation.

## 3. Perform Feature Scaling and One-Hot Encoding[1 Marks]

- Use StandardScaler

- Compare Original features with Scaled features

- Display Mean , Standard deviation

- Convert the target labels into one-hot vectors.

- Display the encoded labels.
"""

X = df.drop("target",axis = 1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)

fig, splot = plt.subplots(2, 1, figsize=(10,8))
sns.boxplot(data=X, ax=splot[0])
splot[0].set_title("Original Features", fontsize=16)
splot[0].tick_params(axis='x', rotation=45)
sns.boxplot(data=X_scaled, ax=splot[1])
splot[1].set_title("Scaled Features", fontsize=16)
splot[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

print("Mean of all the features\n")
print(X_scaled.mean())

print("\nStandard Deviation of all the features\n")
print(X_scaled.std())

encoder = OneHotEncoder(sparse_output=False)
y = df[["target"]]
y_encoded = encoder.fit_transform(y)
print(y_encoded)

"""## 5. Split the data into training, validation, and test sets (70%-15%-15%). [Marks 1]"""

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled,y_encoded,test_size=0.30,random_state=42,shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,random_state=42,shuffle=True)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

"""# 6. Implement Elastic-Net Softmax Regression from Scratch [6 Marks]

In this step, you will implement an **Elastic-Net Regularized Softmax Regression** model from scratch using **Gradient Descent**. **Do not use any machine learning library** (e.g., Scikit-learn, TensorFlow, or PyTorch) for model training.

Your implementation must include the following components.

---

## (a) Softmax Function

Implement the **Softmax activation function** to convert the raw logits into class probabilities.

The Softmax function should satisfy the following properties:

- The probability of each class lies between **0 and 1**.
- The sum of probabilities for each sample should be **equal to 1**.

---

## (b) Cross-Entropy Loss

Implement the **Multiclass Cross-Entropy Loss** to measure the difference between the predicted probabilities and the true class labels.

Your implementation should compute the average loss over all training samples.

---

## (c) L1 Regularization (Lasso)

Extend the loss function by adding an **L1 Regularization** term.

The L1 penalty is defined as

$$
L_{L1}=\lambda_1\sum_{i,j}|W_{ij}|
$$

where

- $W$ is the weight matrix.
- $\lambda_1$ is the L1 regularization parameter.

---

## (d) L2 Regularization (Ridge)

Further extend the loss function by adding an **L2 Regularization** term.

The L2 penalty is defined as

$$
L_{L2}=\frac{\lambda_2}{2}\sum_{i,j}W_{ij}^{2}
$$

where

- $W$ is the weight matrix.
- $\lambda_2$ is the L2 regularization parameter.

---

## (e) Gradient of L1 Regularization

Implement the gradient of the L1 regularization term.

The gradient is given by

$$
\frac{\partial |W|}{\partial W}=\operatorname{sign}(W)
$$

---

## (f) Gradient of L2 Regularization

Implement the gradient of the L2 regularization term.

The gradient is given by

$$
\frac{\partial}{\partial W}\left(\frac{\lambda_2}{2}\|W\|_2^2\right)=\lambda_2W
$$

---

## (g) Gradient Descent Optimization

Implement the complete **Gradient Descent** algorithm for training the model.

During each training epoch, your implementation must perform the following steps:

1. Compute the logits.
2. Apply the Softmax function.
3. Compute the Cross-Entropy Loss.
4. Add the L1 and L2 regularization terms.
5. Compute the gradients of the weights and bias.
6. Update the model parameters using Gradient Descent.
7. Store the **Training Loss**.
8. Compute and store the **Validation Loss**.

---

## Expected Outputs

After completing this step, your implementation should:

- Successfully train an Elastic-Net Softmax Regression model.
- Store the **training loss** after every epoch.
- Store the **validation loss** after every epoch.
- Learn the optimal weight matrix and bias vector.

> **Note**
>
> - You must implement the complete algorithm **from scratch**.
> - Do **not** use any built-in machine learning library for model training.
> - Only **NumPy** may be used for numerical computations.
"""

#a) softmax
def softmax(z):
    z_s = z - np.max(z, axis=1)[:, None]
    exp_z = np.exp(z_s)
    return exp_z / np.sum(exp_z, axis=1)[:, None]

#b) Cross-Entropy-Loss
def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def l1_penalty(W, lambda1):
    return lambda1 * np.sum(np.abs(W))

def l2_penalty(W,lambda2):
    return (lambda2 / 2) * np.sum(W**2)

def gradient_weights(X, Y, P):
    N = X.shape[0]
    return (X.T @ (P - Y)) / N

def gradient_bias(Y, P):
    N = Y.shape[0]
    return np.sum(P - Y, axis=0) / N

def l1_gradient(W, lambda1):
    return lambda1 * np.sign(W)

def l2_gradient(W, lambda2):
    return lambda2 * W

features = X_train.shape[1]
classes = y_train.shape[1]

W = np.zeros((features, classes))
b = np.zeros(classes)

"""# 7. Train the Model [1 Mark]

Train the **Elastic-Net Softmax Regression** model using the **Gradient Descent** algorithm.

---

## Training Configuration

Train the model for **1000 epochs**.

Select appropriate values for the following hyperparameters:

- **Learning Rate ($\alpha$)**
- **L1 Regularization Parameter ($\lambda_1$)**
- **L2 Regularization Parameter ($\lambda_2$)**

You may experiment with different values of these hyperparameters to achieve better model performance.
"""

learning_rate = 0.01
epochs = 1000
lambda1 = 0.001
lambda2 = 0.001

train_losses = []
val_losses = []

for epoch in range(epochs):
####1)logits
    logits = X_train @ W + b

####2)applying Softmax
    probs = softmax(logits)

####3)Computing cross-entropy loss
    ce_loss = cross_entropy_loss(y_train, probs)

####4)regularization
    l1_loss = l1_penalty(W, lambda1)
    l2_loss = l2_penalty(W, lambda2)
    total_loss = ce_loss + l1_loss + l2_loss
####5)Gradients
    grad_W_ce = (X_train.T @ (probs - y_train)) / X_train.shape[0]
    grad_W_l1 = l1_gradient(W, lambda1)
    grad_W_l2 = l2_gradient(W, lambda2)
    t_grad_W = grad_W_ce + grad_W_l1 + grad_W_l2
    grad_b = (np.sum(probs - y_train, axis=0)/ X_train.shape[0])
####6)Updating Parametes
    W = W - learning_rate * t_grad_W
    b = b - learning_rate * grad_b

####7)storing training loss
    train_losses.append(total_loss)

####8)computing validation loss + storing
    val_logits = X_val @ W + b
    val_probs = softmax(val_logits)
    val_ce = cross_entropy_loss(y_val, val_probs)
    val_l1 = l1_penalty(W, lambda1)
    val_l2 = l2_penalty(W, lambda2)
    val_loss = val_ce + val_l1 + val_l2
    val_losses.append(val_loss)

"""
## 8. Plot Learning Curve[1 Mark]

- Plot Training Loss and Validation Loss

- Discuss Underfitting and Overfitting"""

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.show()

"""Underfitting: The model does not show significant underfitting because both training and validation losses decrease during training and reach relatively low values.

Overfitting: The model also does not show significant overfitting because the training and validation loss curves remain close to each other. There is no large increase in validation loss while training loss continues to decrease.

Overall: The learning curves indicate that the model has a good fit to the data, with no clear signs of severe underfitting or overfitting.

## 9. Predict on test set[1 Mark]
 - display actual and predicted class
"""

test_logits = X_test @ W + b
test_probs = softmax(test_logits)
y_pred = np.argmax(test_probs, axis=1)
y_actual = np.argmax(y_test, axis=1)
print("Actual Class    Predicted Class")
for actual, predicted in zip(y_actual, y_pred):
    print(f"{actual:<15} {predicted}")

"""## 10. Evaluate the Model[1 Mark]

- compute Accuracy, precision, recall, f1_score
"""

accuracy = accuracy_score(y_actual, y_pred)
precision = precision_score(y_actual, y_pred, average="macro")
recall = recall_score(y_actual, y_pred, average="macro")
f1 = f1_score(y_actual, y_pred, average="macro")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("Actual   :", y_actual)
print("Predicted:", y_pred)
print("Correct predictions:",np.sum(y_actual == y_pred))
print("Total predictions:",len(y_actual))

"""## 11. Plot the confusion matrix, Classification report and interpret the results.[1 Marks]"""

cm = confusion_matrix(y_actual, y_pred)
print(cm)

print(classification_report(y_actual, y_pred))

"""The confusion matrix shows that all test samples are correctly classified with no misclassification between the three classes,the classification report also shows precision, recall, and f1-score of 1 for all classes,that means the model achieved perfect classification performance on the test set.

## 12. Perform the Sklearn Implementation and Compare the both model.[1 Marks]
"""

y_train_c = np.argmax(y_train, axis=1)
y_test_c = np.argmax(y_test, axis=1)

sklearn_model = LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.5,max_iter=1000,random_state=35)
sklearn_model.fit(X_train, y_train_c)

y_pred_skl = sklearn_model.predict(X_test)

accuracy1 = accuracy_score(y_test_c, y_pred_skl)
precision1 = precision_score(y_test_c, y_pred_skl, average="macro")
recall1 = recall_score(y_test_c, y_pred_skl, average="macro")
f11 = f1_score(y_test_c, y_pred_skl, average="macro")

print("Sklearn Model")
print("Accuracy :", accuracy1)
print("Precision:", precision1)
print("Recall   :", recall1)
print("F1 Score :", f11)

print("Comparison")
print("--------------------------------")
print("Metric       From Scratch   Sklearn")
print("Accuracy        ", accuracy, "       ", accuracy1)
print("Precision       ", precision, "       ", precision1)
print("Recall          ", recall, "       ", recall1)
print("F1 Score        ", f1, "       ", f11)

"""# 13: Effect of Regularization[3 Marks]

In this section, analyze the effect of different regularization techniques on the performance of the **Softmax Regression** model.

Train the model using the following three configurations and compare the results.

---

## Case 1: No Regularization

Train the model without any regularization.

**Hyperparameters**

- L1 Regularization ($\lambda_1$) = **0**
- L2 Regularization ($\lambda_2$) = **0**

This serves as the baseline model.

---

## Case 2: L2 Regularization (Ridge)

Train the model using **only L2 Regularization**.

**Hyperparameters**

- L1 Regularization ($\lambda_1$) = **0**
- L2 Regularization ($\lambda_2$) = **0.01**

Observe how L2 regularization affects the model performance and the learning curves.

---

## Case 3: Elastic-Net Regularization

Train the model using **both L1 and L2 Regularization**.

**Hyperparameters**

- L1 Regularization ($\lambda_1$) = **0.01**
- L2 Regularization ($\lambda_2$) = **0.01**

Observe the combined effect of L1 and L2 regularization on the model.

---

## Compare the Following Performance Metrics

For each configuration, compute and compare:

- Accuracy
- Precision
- Recall
- F1-Score
- Training Loss
- Validation Loss

Present your results in the following table.

| Model | Accuracy | Precision | Recall | F1-Score | Training Loss | Validation Loss |
|--------|---------:|----------:|--------:|---------:|--------------:|----------------:|
| No Regularization | | | | | | |
| L2 Regularization | | | | | | |
| Elastic-Net Regularization | | | | | | |

---


"""

def train_model(X_train, y_train, X_val, y_val, lambda1, lambda2):
    W = np.zeros((X_train.shape[1], y_train.shape[1]))
    b = np.zeros(y_train.shape[1])

    # Store losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
####1)logits
        logits = X_train @ W + b

####2)applying Softmax
        probs = softmax(logits)

####3)Computing cross-entropy loss
        ce_loss = cross_entropy_loss(y_train, probs)

####4)regularization
        l1_loss = l1_penalty(W, lambda1)
        l2_loss = l2_penalty(W, lambda2)
        total_loss = ce_loss + l1_loss + l2_loss
####5)Gradients
        grad_W_ce = (X_train.T @ (probs - y_train)) / X_train.shape[0]
        grad_W_l1 = l1_gradient(W, lambda1)
        grad_W_l2 = l2_gradient(W, lambda2)
        t_grad_W = grad_W_ce + grad_W_l1 + grad_W_l2
        grad_b = (np.sum(probs - y_train, axis=0)/ X_train.shape[0])
####6)Updating Parametes
        W = W - learning_rate * t_grad_W
        b = b - learning_rate * grad_b

####7)storing training loss
        train_losses.append(total_loss)

####8)computing validation loss + storing
        val_logits = X_val @ W + b
        val_probs = softmax(val_logits)
        val_ce = cross_entropy_loss(y_val, val_probs)
        val_l1 = l1_penalty(W, lambda1)
        val_l2 = l2_penalty(W, lambda2)
        val_loss = val_ce + val_l1 + val_l2
        val_losses.append(val_loss)

    return W,b,train_losses,val_losses

####1)No Regularization
W_no, b_no, train_no, val_no = train_model(X_train, y_train,X_val, y_val,0, 0)
test_probs_no = softmax(X_test @ W_no + b_no)
y_pred_no = np.argmax(test_probs_no, axis=1)

####2)L2 -Reg
W_l2, b_l2, train_l2, val_l2 = train_model(X_train, y_train,X_val, y_val,0, 0.01)
test_probs_l2 = softmax(X_test @ W_l2 + b_l2)
y_pred_l2 = np.argmax(test_probs_l2, axis=1)

####3)Elastic-net
W_en, b_en, train_en, val_en = train_model(X_train, y_train,X_val, y_val,0.01, 0.01)
test_probs_en = softmax(X_test @ W_en + b_en)
y_pred_en = np.argmax(test_probs_en, axis=1)

####All - Metrices
def metrics(y_actual, y_pred):
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred, average="macro")
    recall = recall_score(y_actual, y_pred, average="macro")
    f1 = f1_score(y_actual, y_pred, average="macro")
    return accuracy, precision, recall, f1

metrics_no = metrics(y_actual, y_pred_no)
metrics_l2 = metrics(y_actual, y_pred_l2)
metrics_en = metrics(y_actual, y_pred_en)

results = pd.DataFrame({
    "Model": [
        "No Regularization",
        "L2 Regularization",
        "Elastic-Net Regularization"
    ],

    "Accuracy": [
        metrics_no[0],
        metrics_l2[0],
        metrics_en[0]
    ],

    "Precision": [
        metrics_no[1],
        metrics_l2[1],
        metrics_en[1]
    ],

    "Recall": [
        metrics_no[2],
        metrics_l2[2],
        metrics_en[2]
    ],

    "F1-Score": [
        metrics_no[3],
        metrics_l2[3],
        metrics_en[3]
    ],

    "Training Loss": [
        train_no[-1],
        train_l2[-1],
        train_en[-1]
    ],

    "Validation Loss": [
        val_no[-1],
        val_l2[-1],
        val_en[-1]
    ]
})

print(results)

"""## 14. Analysis[1 Marks]

Based on the experimental results, answer the following questions.

1. Which regularization technique achieved the highest classification accuracy?

2. How did L2 regularization affect the training and validation losses compared to the model without regularization?

3. What impact did adding L1 regularization have on the overall model performance?

4. Which model showed the best generalization performance on the validation and test datasets?

5. Which regularization technique would you recommend for this dataset? Justify your answer using the obtained results.

1) All three models achieved the same classification accuracy of 1.00(100%) on the test dataset.therefore no model had a higher accuracy than the others.
2) L2 regularization increased the training loss from 0.0949 to 0.1237 and the validation loss from 0.1284 to 0.1563.thus L2 increased the loss but still maintained the same 100% test accuracy.
3) Adding L1 regularization increased the training loss to 0.2288 and validation loss to 0.2539.however the classification metrics remained unchanged at 100%.thus L1 increased the regularization penalty without improving the classification performance on this dataset.
4) The no regularization model showed the best performance based on the obtained losses with the lowest training loss (0.0949) and validation loss (0.1237) while also achieving 100% test accuracy.
5) Based on the obtained results i would recommend no regularization for this dataset.all three models achieved 100% test accuracy but the unregularized model had the lowest training and validation losses.therefore regularization did not provide any performance improvement in this experiment.
"""

