# 📊 Sales Prediction Using Python

## 📌 Project Overview

Sales Prediction Using Python is a machine learning project that predicts product sales based on advertising budgets spent on TV, Radio, and Newspaper. It uses a Random Forest Regression model and provides a web application built with Streamlit.

## 🎯 Objectives

- Analyze the relationship between advertising expenditure and sales.
- Explore the dataset using data visualization techniques.
- Train a machine learning model to predict sales.
- Evaluate the model using regression metrics.
- Deploy an interactive web application for sales prediction.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```text
sales-prediction-web/
├── app.py
├── Advertising.csv
├── requirements.txt
├── Sales_Prediction.ipynb
└── README.md
```

## 📊 Dataset

The project uses `Advertising.csv`, containing the following columns:

- **TV:** TV advertising budget
- **Radio:** Radio advertising budget
- **Newspaper:** Newspaper advertising budget
- **Sales:** Product sales

The dataset contains 200 records and four columns.

## 🤖 Machine Learning Model

A Random Forest Regressor is trained using TV, Radio, and Newspaper advertising budgets as input features and Sales as the target variable.

The project also explores data correlations, advertising-versus-sales relationships, and feature importance.

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sales-prediction-web.git
cd sales-prediction-web
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 3. Run the web application

```bash
python3 -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## 🌐 Deployment

The Streamlit web application can be deployed using Streamlit Community Cloud.

## 📈 Features

- Advertising budget input fields
- Sales prediction
- Advertising-versus-sales scatter plots
- Correlation heatmap
- Random Forest feature importance visualization
- Dataset preview and descriptive statistics

## 👨‍💻 Author

**Yashwanth R**

## 📄 License

This project is intended for educational and internship purposes.
