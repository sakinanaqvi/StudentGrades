# Student Grade Prediction

This project aims to predict the final grade (`G3`) of students using various features such as study time, family relationships, and past grades (`G1` and `G2`). The model is built using **Linear Regression** and is evaluated using metrics like **Mean Absolute Error (MAE)** and **R-squared (R²)**.

## Project Overview

In this project, we use a dataset of student performance to build a regression model that predicts the final grade (`G3`). The model leverages several features of the students, such as:

- `studytime`: Weekly study time (1-4 scale)
- `failures`: Number of past failures (0-1 scale)
- `famrel`: Family relationships (1-5 scale)
- `freetime`: Free time (1-5 scale)
- `goout`: Socializing with friends (1-5 scale)
- `sex_M`: Gender (1 for male, 0 for female, based on one-hot encoding)
- `address_U`: Address type (1 for urban, 0 for rural, based on one-hot encoding)
- `G1`: First period grade
- `G2`: Second period grade

### Key Features

- **Linear Regression Model**: We train a linear regression model to predict the final grade (`G3`) using the features mentioned above.
- **Evaluation Metrics**: We evaluate the model using:
  - **Mean Absolute Error (MAE)**: Measures the average error between predicted and actual values.
  - **R-squared (R²)**: Measures the proportion of variance in the target variable explained by the features.

### Performance Metrics

The model's performance is evaluated on a test set, and the following metrics were obtained:

- **Mean Absolute Error**: `1.6466656197147518`
- **R-squared**: `0.7241341236974019`

These results indicate that the model explains about **72%** of the variance in the final grades and has an average prediction error of **1.65** points.

## Installation and Setup

### Requirements

- Python 3.x
- `pandas` library
- `scikit-learn` library
- `matplotlib` library (optional, for visualization)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-grades.git
   cd student-grade-prediction
