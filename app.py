import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import tempfile
import ollama
import re

st.set_page_config(page_title="AI Analytics Pro MAX", layout="wide")

def ask_ai(df, question):
    sample = df.sample(min(30, len(df))).to_csv(index=False)
    prompt = f"""
Dataset:
{sample}

Answer clearly.

Question: {question}
"""
    res = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )
    return res['message']['content']

def dataset_summary(df):
    rows = df.shape[0]
    cols = df.shape[1]
    duplicates = df.duplicated().sum()
    missing = df.isnull().sum().sum()
    return rows, cols, duplicates, missing

# ✅ SMART DATE CHECK
def is_valid_date(val):
    if pd.isna(val): return False
    val = str(val).strip()
    patterns = [
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
        r"^\d{4}[-/][A-Za-z]{3}[-/]\d{1,2}$",
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
    ]
    return any(re.match(p, val) for p in patterns)

# ✅ DATE FIX → ALWAYS YYYY-MM-DD
def fix_dates(df):
    date_cols = []
    for col in df.columns:
        if "date" in col.lower():
            date_cols.append(col)

            def convert(x):
                if pd.isna(x): return ""
                if is_valid_date(x):
                    dt = pd.to_datetime(x, errors='coerce', dayfirst=True)
                    if pd.notna(dt):
                        return dt.strftime("%Y-%m-%d")
                return x

            df[col] = df[col].apply(convert)

    return df, date_cols

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "before_dates" not in st.session_state:
    st.session_state.before_dates = None
if "after_dates" not in st.session_state:
    st.session_state.after_dates = None
if "date_cols" not in st.session_state:
    st.session_state.date_cols = []

st.title("🚀 AI Analytics Pro MAX")

file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:
    st.session_state.raw_df = pd.read_csv(file)

if st.session_state.raw_df is not None:

    raw_df = st.session_state.raw_df

    st.subheader("📦 Dataset Overview")
    rows, cols, duplicates, missing = dataset_summary(raw_df)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows", rows)
    s2.metric("Columns", cols)
    s3.metric("Duplicates", duplicates)
    s4.metric("Missing Values", missing)

    st.subheader("📊 RAW DATA")
    tab1, tab2 = st.tabs(["Preview", "Full Data"])

    with tab1:
        st.dataframe(raw_df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(raw_df, use_container_width=True)

    if st.button("🚀 Clean Dataset"):
        df = raw_df.copy()

        date_preview_before = {}
        for col in df.columns:
            if "date" in col.lower():
                date_preview_before[col] = df[col].head(10)

        df.drop_duplicates(inplace=True)

        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # ✅ DATE FIX
        df, date_cols = fix_dates(df)

        # ✅ TEXT FIX
        for col in df.columns:
            if df[col].dtype == "object":

                def fix_text(x):
                    if pd.isna(x): return ""
                    x = str(x).strip()

                    if "@" in x:
                        return x.lower()

                    if re.search(r"[A-Za-z]+\d+|\d+[A-Za-z]+", x):
                        return x

                    return x.title()

                df[col] = df[col].apply(fix_text)

        # ✅ FINAL: NO "None"
        df = df.fillna("")

        date_preview_after = {}
        for col in date_cols:
            date_preview_after[col] = df[col].head(10)

        st.session_state.clean_df = df
        st.session_state.filtered_df = None
        st.session_state.before_dates = date_preview_before
        st.session_state.after_dates = date_preview_after
        st.session_state.date_cols = date_cols

        st.success("✅ Dataset Cleaned!")

# ---------------- REMAINING CODE UNCHANGED ----------------
if st.session_state.clean_df is not None:

    clean_df = st.session_state.clean_df

    st.subheader("🧹 CLEANED DATA")
    tab1, tab2 = st.tabs(["Preview", "Full Data"])

    with tab1:
        st.dataframe(clean_df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(clean_df, use_container_width=True)

    st.subheader("🔍 Advanced Filter")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        col = st.selectbox("Column", clean_df.columns)

    if clean_df[col].dtype in [np.int64, np.float64]:

        with f2:
            condition = st.selectbox("Condition", ["=", ">", "<", ">=", "<="])
        with f3:
            value = st.number_input("Value")
        with f4:
            apply = st.button("Apply")

        if apply:
            if condition == "=":
                filtered = clean_df[clean_df[col] == value]
            elif condition == ">":
                filtered = clean_df[clean_df[col] > value]
            elif condition == "<":
                filtered = clean_df[clean_df[col] < value]
            elif condition == ">=":
                filtered = clean_df[clean_df[col] >= value]
            elif condition == "<=":
                filtered = clean_df[clean_df[col] <= value]

            st.session_state.filtered_df = filtered

    else:

        with f2:
            condition = st.selectbox("Condition", ["contains", "equals"])
        with f3:
            value = st.text_input("Value")
        with f4:
           with f4:
               st.write("")  # small vertical spacing
               apply = st.button("Apply", use_container_width=True)

        if apply:
            if condition == "contains":
                filtered = clean_df[clean_df[col].astype(str).str.contains(value, case=False, na=False)]
            else:
                filtered = clean_df[clean_df[col].astype(str) == value]

            st.session_state.filtered_df = filtered

    df = st.session_state.filtered_df if st.session_state.filtered_df is not None else clean_df

    st.subheader("🎯 ACTIVE DATASET")
    tab1, tab2 = st.tabs(["Preview", "Full Data"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df, use_container_width=True)

    st.subheader("📊 KPI Dashboard")

    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 0:
        col_kpi = st.selectbox("Select Column", num_cols)

        total = df[col_kpi].sum()
        avg = df[col_kpi].mean()
        mx = df[col_kpi].max()
        mn = df[col_kpi].min()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", round(total, 2))
        k2.metric("Average", round(avg, 2))
        k3.metric("Max", mx)
        k4.metric("Min", mn)

    st.subheader("📈 Advanced Visualization")

    v1, v2, v3 = st.columns(3)

    with v1:
        chart = st.selectbox("Chart Type", [
            "Line", "Bar", "Histogram", "Pie", "Scatter", "Box", "Area"
        ])

    with v2:
        x = st.selectbox("X-axis", df.columns)

    with v3:
        y = st.selectbox("Y-axis", num_cols)

    fig, ax = plt.subplots(figsize=(7, 3))

    if chart == "Line":
        ax.plot(df[x].head(50), df[y].head(50), marker='o')
    elif chart == "Bar":
        ax.bar(df[x].head(20).astype(str), df[y].head(20))
    elif chart == "Histogram":
        ax.hist(df[y], bins=15)
    elif chart == "Pie":
        pie_data = df.groupby(x)[y].sum().head(5)
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    elif chart == "Scatter":
        ax.scatter(df[x], df[y])
    elif chart == "Box":
        ax.boxplot(df[y])
    elif chart == "Area":
        ax.fill_between(range(len(df[y].head(50))), df[y].head(50))

    if chart not in ["Pie", "Box"]:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    st.subheader("🧠 AI Insights")

    if st.button("✨ Generate Insights"):
        st.success(ask_ai(df, "Give insights and trends"))

    st.subheader("💬 Ask AI")

    q = st.text_input("Ask anything")

    if q:
        st.info(ask_ai(df, q))

   