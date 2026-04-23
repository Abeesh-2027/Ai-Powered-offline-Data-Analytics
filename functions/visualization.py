import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Analytics Engine", layout="wide")

def setup_chart(ax, title, x_label=None, y_label=None):
    ax.set_title(title, fontsize=10)
    if x_label:
        ax.set_xlabel(x_label, fontsize=8)
    if y_label:
        ax.set_ylabel(y_label, fontsize=8)

    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.tick_params(axis='y', labelsize=7)


def get_kpis(df, column):
    return {
        "total": round(df[column].sum(), 2),
        "average": round(df[column].mean(), 2),
        "max": df[column].max(),
        "min": df[column].min(),
        "std": round(df[column].std(), 2)
    }


def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def line_chart(df, x, y):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df[x].head(50), df[y].head(50))
    setup_chart(ax, "Trend", x, y)
    return fig

def bar_chart(df, x, y):
    fig, ax = plt.subplots(figsize=(6, 3))
    temp = df.sort_values(by=y, ascending=False).head(10)
    ax.bar(temp[x], temp[y])
    setup_chart(ax, "Comparison", x, y)
    return fig

def pie_chart(df, col):
    fig, ax = plt.subplots()
    data = df[col].value_counts().head(5)
    ax.pie(data, labels=data.index, autopct='%1.1f%%')
    ax.set_title("Share")
    return fig

def histogram(df, col):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df[col], bins=15)
    setup_chart(ax, "Distribution")
    return fig

def scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(df[x], df[y])
    setup_chart(ax, "Relationship", x, y)
    return fig

def boxplot(df, col):
    fig, ax = plt.subplots()
    ax.boxplot(df[col])
    ax.set_title("Outliers")
    return fig

def heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5,3))
    cax = ax.matshow(corr)
    plt.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Correlation Heatmap")
    return fig


def trend_summary(df, col):
    if df[col].iloc[-1] > df[col].iloc[0]:
        return "Increasing 📈"
    elif df[col].iloc[-1] < df[col].iloc[0]:
        return "Decreasing 📉"
    return "Stable ➖"

def detect_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])

def smart_summary(df, col):
    return {
        "kpi": get_kpis(df, col),
        "trend": trend_summary(df, col),
        "outliers": detect_outliers(df, col)
    }


def suggest_chart(df, x, y):
    if x and y:
        return "Scatter"
    elif y:
        return "Histogram"
    return "Bar"


st.title("🚀 AI Analytics Engine")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    num_cols = get_numeric_columns(df)

    if len(num_cols) == 0:
        st.warning("No numeric columns found")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox("X Column", df.columns)

    with col2:
        y = st.selectbox("Y Column", num_cols)

    st.subheader("📊 KPI Dashboard")

    kpi = get_kpis(df, y)
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total", kpi["total"])
    k2.metric("Average", kpi["average"])
    k3.metric("Max", kpi["max"])
    k4.metric("Min", kpi["min"])
    k5.metric("Std Dev", kpi["std"])

    st.subheader("🧠 Smart Analysis")

    summary = smart_summary(df, y)

    st.info(f"Trend: {summary['trend']}")
    st.warning(f"Outliers Detected: {summary['outliers']}")

    st.subheader("📈 Visualization")

    chart_type = st.selectbox("Select Chart", [
        "Auto", "Line", "Bar", "Pie", "Histogram",
        "Scatter", "Box", "Heatmap"
    ])

    if chart_type == "Auto":
        chart_type = suggest_chart(df, x, y)

    if chart_type == "Line":
        fig = line_chart(df, x, y)

    elif chart_type == "Bar":
        fig = bar_chart(df, x, y)

    elif chart_type == "Pie":
        fig = pie_chart(df, x)

    elif chart_type == "Histogram":
        fig = histogram(df, y)

    elif chart_type == "Scatter":
        fig = scatter(df, x, y)

    elif chart_type == "Box":
        fig = boxplot(df, y)

    elif chart_type == "Heatmap":
        fig = heatmap(df)

    st.pyplot(fig)

    st.subheader("⬇ Export Clean Data")
    st.download_button("Download CSV", df.to_csv(index=False), "clean_data.csv")