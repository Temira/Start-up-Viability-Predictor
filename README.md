# 🚀 Temira's Black Box VC Investing Algorithm (TOP SECRET!)

## 📚 Table of Contents
- [🌟 Project Overview](#-project-overview)
- [🔥 Objectives](#-objectives)
- [💻 Technologies Used](#-technologies-used)
- [🚀 Getting Started](#-getting-started)
- [🎉 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [💼 Business Applications](#-business-applications)
- [🤔 How Did I Pick Random Forest?](#-how-did-i-pick-random-forest)
- [📝 License](#-license)
- [📫 Contact](#-contact)


## 🌟 Project Overview
Welcome to **Temira's Black Box VC Investing Algorithm**! This project aims to predict the viability and future success of start-ups using a machine learning model. The goal is to provide validation for investing in start-ups, boosting confidence in investment decisions. While the model alone may not justify an investment, it serves as a valuable tool for explaining investment choices to Limited Partners (LPs) and other stakeholders. 📈

## 🔥 Objectives
- Analyze a dataset of nearly **50,000 start-ups** and their funding processes to determine their current status (exist or not).
- Cross-reference critical life-cycle events with Consumer Price Index (CPI) data to enhance prediction accuracy.
- Develop an explainable model that can assist in investment discussions and decision-making.

## 💻 Technologies Used
- Python 3.x 🐍
- Pandas for data manipulation 📊
- NumPy for numerical calculations ➗
- Matplotlib for data visualization 📈
- Jupyter Notebook for code execution 📝

## 🚀 Getting Started

### 📦 Prerequisites
Make sure you have Python 3.x and the necessary libraries installed. You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib
```

### 🔄 Cloning the Repository
To get a copy of this project, clone the repository using the following command:

```bash
git clone https://github.com/temira/startup-viability-predictor.git
```

### 📂 Navigating to the Project Directory
Change into the project directory:

```bash
cd startup-viability-predictor
```

## 🎉 Usage
1. **Import Libraries**: Start by importing all necessary libraries in your Jupyter Notebook or Python script.

   ```python
   import pandas as pd
   import datetime as dt
   import numpy as np
   %matplotlib inline
   ```

2. **Load Dataset**: Import the start-up dataset, which includes funding processes and current existence status.

   ```python
   investments_url = "investments_VC.csv"
   df = pd.read_csv(investments_url, encoding="latin1")
   ```

3. **Data Preprocessing**: Clean and format the data to prepare it for model training. This includes dropping unnecessary columns and handling missing values.

   ```python
   # Fix column names and drop unnecessary columns
   df['market'] = df[' market ']
   df['funding_total_usd'] = df[' funding_total_usd ']
   df = df.drop([...])  # Drop columns as needed
   df = df.dropna(subset=['status', 'funding_total_usd', 'founded_at'])
   df = df.drop(df[df.funding_total_usd == '-'].index)
   ```

4. **Feature Engineering**: Create relevant features such as funding dates for model training.

   ```python
   df['fund1'] = df['first_funding_at'].str.slice(0, 7)
   df['fundLast'] = df['last_funding_at'].str.slice(0, 7)
   df['found'] = df['founded_at'].str.slice(0, 7)
   ```

5. **Model Training**: Use the cleaned dataset to train your machine learning model.

## 📁 Project Structure
```plaintext
startup-viability-predictor/
├── README.md                       # Documentation file
├── Temira's Black Box VC Investing Algorithm (TOP SECRET!) (5).ipynb # Main project script
├── investments_VC.csv              # Dataset of start-ups
└── LICENSE                         # License file
```

## 💼 Business Applications

**Applications of this Machine Learning Final Project**

The goal of my project is to use data to predict the viability or future success of a start-up. The business purpose of my project is to provide validation for investing in start-ups. While I do not anticipate that my model alone will be enough to justify an investment, it can help support investment decisions and boost confidence in the outcome. The model aims to be explainable, which can assist in clarifying the investment rationale to Limited Partners (LPs) or other stakeholders accountable for the investment.

I utilized a dataset of almost **50,000 start-ups**, including their funding processes and current existence status. This information was cross-referenced with the CPIs on the dates of important "life-cycle" events of the start-ups, such as their founding date, the date they raised their first round of funding, and the date they raised their last round of funding. Although the model is far from perfect, further refinements to address the inherent imbalance in the dataset are planned for future iterations.

## 🤔 How Did I Pick Random Forest?

Random Forest is an example of a "bagging" algorithm. This means that the different trees in the forest run in parallel to each other. This type of algorithm is particularly effective for datasets that are imbalanced.

In this context, there are significantly more examples of "operating" startups than "closed" startups in the training/test dataset. This imbalance does not accurately reflect how startups behave in the real world. Therefore, using this bagging algorithm is my attempt to mitigate this issue.

To further enhance the model's performance, I used a large number of estimators (i.e., trees) in order to prevent overfitting, which was a significant challenge during development.


## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact
For any inquiries or feedback, please reach out to:

- **Name**: Temira Koenig
- **GitHub**: [temira](https://github.com/temira)
