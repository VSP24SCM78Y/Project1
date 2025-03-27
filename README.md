# Project 1  
**Course:** CS584 - Machine Learning  
**Instructor:** Steve Avsec  

## Team Members  
- Rohith Kukkadapu – A20554359  
- Vinay Yerram – A20554778  
- Rama Muni Reddy Bandi – A20554387  
- Kumar Sri Pavan Veeramallu – A20539662  


---

## Linear Regression with LASSO Regularization via Homotopy & RecLasso (Online Updates)

---

### 🔹 What does the model you have implemented do and when should it be used?

This project implements **LASSO (Least Absolute Shrinkage and Selection Operator)** regression using the **Homotopy method**, which efficiently traces the solution path as the regularization parameter (lambda) decreases. This allows for real-time observation of how features enter the model — a concept inspired by the LARS algorithm. Additionally, the model supports **RecLasso**, an online extension of LASSO which updates the model efficiently when new data arrives, without retraining from scratch.

It should be used when:
- You need a sparse regression model that selects features automatically
- You have high-dimensional data or collinearity among features
- You want to understand how the model evolves as regularization changes
- You are working with streaming data and want to update the model incrementally (RecLasso)

---

### 🔹 How did you test your model to determine if it is working reasonably correctly?

- Used **unit tests** (via `pytest`) to verify:
  - Basic functionality of `.fit()` and `.predict()` methods
  - That the model produces a **sparse solution** with collinear data
  - That predictions are numeric, well-formed, and not NaNs
- Tested `RecLassoModel` using `.partial_fit()` to ensure it updates coefficients without breaking existing behavior
- Used synthetic test cases and CSV datasets like `collinear_data.csv` and `small_test.csv`
- Verified results visually with a **lambda vs coefficient path plot** to ensure features behave as expected

---

### 🔹 What parameters have you exposed to users of your implementation in order to tune performance?

- No explicit hyperparameters are required to run the Homotopy method, as the solution path is calculated automatically.
- However, exposed internal thresholds that control algorithm behavior:
  - `tol` – Tolerance to stop iterations when the maximum correlation falls below this value
  - `max_iter` – Maximum number of iterations (defaulted to min(n-1, p))

For `RecLassoModel`, updates are handled automatically per incoming data point via `.partial_fit(x_new, y_new)`.

---

### 🔹 Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

- **Numerical instability**: In some rare steps of Homotopy, the denominator in the gamma update may approach zero, leading to potential division warnings. This was addressed using safe-threshold checks.
- **Highly non-linear data**: As this is a linear model, it assumes linear relationships and won't perform well on data with complex non-linear interactions.
- **Extremely high-dimensional data (p >> n)**: While LASSO is built for this, our implementation does not yet use techniques like coordinate descent or warm-started solvers that scale better in very large feature spaces.
- **No categorical feature support**: Assumes input data is already numeric and preprocessed. Given more time, we would add encoders or handle pandas DataFrames directly.

---

###  How to Run the Project

####  Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

####  Run the Models

##### Run Homotopy-based LASSO

Run the Models
Run Homotopy-based LASSO

```python
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

model = LassoHomotopyModel()
result = model.fit(X, y)
predictions = result.predict(X)
```

##### Run Online RecLasso Update

```python
from LassoHomotopy.model.RecLasso import RecLassoModel

model = RecLassoModel()
model.fit(X, y)

# Simulate new data point
x_new = ...
y_new = ...
model.partial_fit(x_new, y_new)
```

---

#### 📊 View Visualizations (Optional)

```bash
cd demo
jupyter notebook LassoHomotopy_Demo_Annotated.ipynb
```

You’ll see:
- Coefficient paths as lambda decreases
- Entry points for each feature
- Feature count over time
- RecLasso online updates in action

---

#### 🧪 Run Tests

```bash
cd LassoHomotopy/tests
pytest -s
```

This runs:
- Unit tests for LASSO and RecLasso
- Sparse recovery on collinear data
- Prediction sanity checks
