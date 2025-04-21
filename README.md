# DISCLAIMER:
# CHAT GPT and Tutor help were used to complete this project.

# REPORT IN README

# Alphabet Soup Foundation: Neural Network Performance Analysis

## Project Purpose

The Alphabet Soup Foundation seeks to enhance the efficiency of its funding process by leveraging machine learning to predict whether funded organizations will achieve success. Using a historical dataset of over 34,000 entries, this project implemented and iteratively refined a deep learning model to serve as a predictive tool.

---

## Dataset Preparation

- **Objective**: Predict the binary outcome: `IS_SUCCESSFUL` (1 = successful, 0 = unsuccessful).

- **Predictor Features**: The dataset contained application metadata including:
  - Funding request amount (`ASK_AMT`), use cases (`USE_CASE`), sector (`AFFILIATION`), income tiers (`INCOME_AMT`), and other organizational characteristics.

- **Data Cleaning Steps**:
  - Removed non-informative identifiers: `EIN`, `NAME`
  - Encoded categorical variables using `pd.get_dummies()`
  - Consolidated rare categorical values into an `Other` category to reduce noise
  - Transformed features using `StandardScaler()` to normalize input ranges

---

## Model Development

### Initial Architecture

- Input layer matched the number of engineered features
- First Hidden Layer: 80 neurons, ReLU activation
- Second Hidden Layer: 30 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation
- Baseline Accuracy: ~72%

### Optimization Phases

1. **Increased Depth and Width**:
   - 3 hidden layers with 100, 30, and 10 nodes respectively
   - Combination of ReLU and Sigmoid activations
   - Resulted in modest improvement to ~74%

2. **Adjusted Activation Strategy**:
   - Used sigmoid in deeper layers to control overfitting and handle class imbalance
   - Helped extract more signal from minority patterns

3. **Input Feature Refinement** (Documented Only):
   - More aggressive binning of categorical variables
   - Dropped low-variance and redundant features
   - Log transformation of highly skewed `ASK_AMT`

### Model Architecture Summary

Visual showing the number of parameters and layer configuration:

![Model Architecture](Images/model_summary.png)

---

## Evaluation Summary

- Best Achieved Accuracy: ~75%
- Validation: Accuracy consistent across multiple runs
- Drivers of Improvement: Deeper layers, refined activation functions, and cleaner input data

### Training Accuracy and Loss Over Epochs

This plot shows the learning curve over time, indicating how the model's accuracy improved and loss decreased as it was trained:

![Training Accuracy and Loss](Images/training_accuracy_loss.png)

### Evaluation Output from Best Model (Opitmized_Code.ipynb)

The following output confirms that the best-performing model achieved ~74% accuracy:

![Evaluation Output](Images/evaluation_output.png)

---

## Recommendation: Alternative Model

While the neural network achieved acceptable performance, a tree-based model such as **XGBoost** or **Random Forest** may offer several advantages:

- Handles categorical variables more efficiently
- Less sensitive to feature scaling and skewed data
- Provides better explainability via feature importance
- Often yields strong performance with minimal hyperparameter tuning

This alternative should be considered if the goal includes both prediction and transparency.

---

## Files Generated

- `AlphabetSoupCharity.h5` - initial model  
- `AlphabetSoupCharity_Optimization.h5` - optimized model  
- `Starter_Code.ipynb` - training notebook  
- `Opitmized_Code.ipynb` - optimization notebook  

