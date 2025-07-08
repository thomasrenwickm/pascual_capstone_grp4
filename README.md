# Pascual Capstone (Group 4) – MSc in Business Analytics and Data Science (IE University)

## 🧪 Project Overview

This repository contains the source code developed for the **Capstone** of the **Master's Degree in Business Analytics and Data Science at IE University** (2024–2025). The project was completed in collaboration with **Pascual**, one of Spain's leading Consumer Packaged Goods (CPG) companies.


Our goal:  
To **optimize client visit frequency and order alignment** using machine learning, helping Pascual reduce logistical and client visit costs while maintaining service quality.

The project required applying advanced analytics, segmentation, optimization, and generative AI techniques to raw transactional data from Pascual’s clients.

---

## 🎯 Objective

Pascual’s HR (restaurants & bars) and AR (small convenience shops) channels suffer from high-frequency, low-efficiency client visits and small orders that erode margins.

The objective of this capstone project was to:

- **Align monthly visit frequency with order behavior** at the client level.
- Reduce redundant promoter visits and unnecessary deliveries.
- Maintain service quality while improving operational efficiency.
- Provide a transparent, explainable solution powered by an LLM.

---

## 🛠️ Tech Stack

| Tool / Library           | Purpose                                      |
|--------------------------|----------------------------------------------|
| **Python**               | Data processing and model development        |
| **Pandas, NumPy**        | Data manipulation and transformation         |
| **Scikit-Learn**         | Machine learning modeling                    |
| **SciPy (L-BFGS-B)**     | Optimization for cost-efficient solutions    |
| **Jupyter Notebook**     | Development and experimentation environment  |
| **Streamlit**            | Web app interface for end-user interaction   |
| **LangChain**            | Prompt management and LLM orchestration      |
| **Google Gemini Flash**  | Natural language explanations via API        |
| **GitHub**               | Version control and collaboration            |


---

## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `model.ipynb` | Notebook which includes EDA, data cleaning, segmentation, optimization model building, simulation, and evaluation. This notebook is the analytical heart of the project. |
| `main.py` | Streamlit front-end logic with Gemini LLM integration. Allows users to input a client ID and receive a detailed, human-readable explanation of the optimization. |
| `optimized_df.csv` | Final output table. Contains one row per client with both pre- and post-optimization values (visits, orders, efficiency, costs, segment, etc.). Serves as input to the LLM tool. |
| `requirements.txt` | List of Python dependencies to replicate the environment locally. |
| `README.md` | This file. Overview of the project, objectives, tools used, and how to interpret the repository structure. |

---

## 💬 Explainability with LLM

A key requirement from Pascual was transparency: the LLM tool helps clearly communicate what changes were made at the client level.

We built a lightweight Streamlit interface powered by **Google Gemini Flash** (via LangChain) to offer natural language explanations for each client’s optimized contact strategy.

- 🔎 Just input a **Client ID**, and the app retrieves the corresponding row from the output table.
- 🤖 The LLM generates a clear, structured explanation detailing:
  - Client’s original state
  - Optimization suggestions
  - Expected impact
  - Estimated time to benefit

This ensures the model’s decisions are **interpretable, justifiable, and easy to communicate** to business stakeholders.

---

## 🚀 Try the App

You can explore the client-level optimization results directly in your browser using our interactive Streamlit app. Simply enter a client ID to view the suggested visit and order strategy.

👉 **Access the app here:**  
[https://pascualcapstonegrp4.streamlit.app/](https://pascualcapstonegrp4-4zqzlnrrisregnxmrmqvsm.streamlit.app/)

---

## 👨‍🏫 Academic Context

This project was completed for the **Capstone** of the Master's Degree in Business Analytics and Data Science at **IE University**.

It demonstrates the application of machine learning, optimization, and AI explainability techniques to a real-world, high-impact business problem in the CPG industry.

---

## 👥 Team

- Abdullah Alshaarawi  
- Joel James Alarde  
- Hiromitsu Fujiyama  
- Sanjo Joy  
- Thomas Arturo Renwick Morales
