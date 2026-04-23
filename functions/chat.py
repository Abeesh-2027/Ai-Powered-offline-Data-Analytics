import ollama

def ask_ai(df, question):

    # Convert dataset to text (limit for speed)
    sample_data = df.head(50).to_csv(index=False)

    prompt = f"""
You are a professional data analyst.

Dataset (CSV):
{sample_data}

Instructions:
- Answer ONLY based on dataset
- Be clear and correct
- Do calculations if needed
- If not found, say "Not in dataset"

Question:
{question}
"""

    response = ollama.chat(
        model='llama3',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']