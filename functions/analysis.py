import numpy as np

def generate_insights(df):
    insights = []
    insights.append("📊 DATASET STORY ANALYSIS\n")

    # Basic Info
    insights.append(f"Total Rows: {df.shape[0]}")
    insights.append(f"Total Columns: {df.shape[1]}\n")

    num_cols = df.select_dtypes(include='number').columns

    if len(num_cols) == 0:
        return "No numeric data available for analysis."

    for col in num_cols:
        data = df[col].dropna()

        if len(data) < 2:
            continue

        insights.append(f"\n🔎 Column: {col}")

        mean = data.mean()
        max_val = data.max()
        min_val = data.min()
        std = data.std()

        insights.append(f"• Average value is {round(mean,2)}")
        insights.append(f"• Highest value is {max_val}")
        insights.append(f"• Lowest value is {min_val}")
        insights.append(f"• Volatility (std dev): {round(std,2)}")

        # Trend detection (smarter)
        trend = data.iloc[-1] - data.iloc[0]
        growth_pct = ((data.iloc[-1] - data.iloc[0]) / (abs(data.iloc[0]) + 1e-9)) * 100

        if trend > 0:
            insights.append(f"• Trend: Increasing 📈 ({round(growth_pct,2)}%)")
        elif trend < 0:
            insights.append(f"• Trend: Decreasing 📉 ({round(growth_pct,2)}%)")
        else:
            insights.append("• Trend: Stable")

        # Outlier detection (IQR method)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

        if len(outliers) > 0:
            insights.append(f"⚠️ {len(outliers)} outliers detected")

        # Volatility insight
        if std > mean * 0.5:
            insights.append("• High variability detected")

        # Business-style storytelling
        direction = "growing" if trend > 0 else "declining"
        insights.append(
            f"👉 Insight: {col} is {direction} with average around {round(mean,2)} "
            f"and peak reaching {max_val}."
        )

    if len(num_cols) > 1:
        insights.append("\n🔗 RELATIONSHIPS BETWEEN VARIABLES")

        corr = df[num_cols].corr()

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = corr.iloc[i, j]

                if abs(val) > 0.75:
                    strength = "Strong"
                elif abs(val) > 0.5:
                    strength = "Moderate"
                else:
                    continue

                insights.append(
                    f"• {strength} relationship between {num_cols[i]} and {num_cols[j]} "
                    f"(Correlation: {round(val,2)})"
                )

    insights.append("\n📌 FINAL SUMMARY")

    best_col = df[num_cols].mean().idxmax()
    worst_col = df[num_cols].mean().idxmin()

    insights.append(f"• Best performing metric: {best_col}")
    insights.append(f"• Lowest performing metric: {worst_col}")
    insights.append("• Dataset shows actionable patterns")
    insights.append("• Optimization opportunity detected in low-performing areas")

    return "\n".join(insights)