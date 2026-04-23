import pandas as pd
import numpy as np
import re

class Config:
    DATE_FORMAT = "%Y-%m-%d"
    REMOVE_DUPLICATES = True
    REMOVE_NEGATIVE = True

def normalize_nulls(df):
    return df.replace(
        ["None", "none", "NULL", "null", "nan", "N/A", ""],
        np.nan
    )


def is_valid_date(val):
    """
    Allow only clean date patterns
    Reject mixed strings like 'abc123x'
    """
    if pd.isna(val):
        return False

    val = str(val).strip()

    patterns = [
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",       
        r"^\d{4}[-/][A-Za-z]{3}[-/]\d{1,2}$",    
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"         
    ]

    for p in patterns:
        if re.match(p, val):
            return True

    return False

def clean_dates(df):
    date_cols = []

    for col in df.columns:
        if "date" in col.lower():
            date_cols.append(col)

            def fix_date(x):
                if is_valid_date(x):
                    try:
                        dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
                        if pd.notna(dt):
                            return dt.strftime(Config.DATE_FORMAT)
                    except:
                        return x
                return x  # keep original if not valid date

            df[col] = df[col].apply(fix_date)

    return df, date_cols

def clean_text_columns(df):
    for col in df.columns:

        if df[col].dtype == "object":

            if "email" in col.lower():
                # Email → full lowercase
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

            else:
                # Only apply title if NOT mixed alphanumeric
                def safe_title(x):
                    x = str(x).strip()

                    # skip mixed values like A12X
                    if re.search(r"[A-Za-z]+\d+|\d+[A-Za-z]+", x):
                        return x

                    return x.title()

                df[col] = df[col].apply(safe_title)

    return df

def clean_numeric(df):
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def remove_negative(df):
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        df = df[df[col] >= 0]

    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def final_touch(df):
    return df.fillna("")

def clean_data(df):

    df = df.copy()

    df = normalize_nulls(df)

    df, date_cols = clean_dates(df)

    df = clean_text_columns(df)

    df = clean_numeric(df)

    if Config.REMOVE_NEGATIVE:
        df = remove_negative(df)

    if Config.REMOVE_DUPLICATES:
        df = remove_duplicates(df)

    df = final_touch(df)

    return df

if __name__ == "__main__":

    file = input("Enter CSV file path: ")
    df = pd.read_csv(file)

    cleaned_df = clean_data(df)

    print("✅ Cleaning Completed Successfully!\n")
    print(cleaned_df.head())

    cleaned_df.to_csv("cleaned_output.csv", index=False)
    print("\n📁 Saved as cleaned_output.csv")