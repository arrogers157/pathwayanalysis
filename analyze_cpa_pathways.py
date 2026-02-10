import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

def main():
    # Load dataset
    csv_file = 'Alternative CPA Pathways Survey_December 31, 2025_09.45.csv'
    # Row 0 is header, Row 1 is Question Text, Row 2 is ImportId
    # We want Row 0 as header, skip Row 1 and Row 2.
    df = pd.read_csv(csv_file, header=0, skiprows=[1, 2])

    # Mapping Dictionary based on inspection
    col_map = {
        'Q23': 'Years_Experience', # Q19 in task
        'Q29': 'CPA_Likelihood',   # Q29 in task
        'Q6': 'Perception',        # Q6 in task
        'Q46': 'Work_20hrs_Acc',   # Q16 in task
        'Q47': 'Work_CPA_Firm',    # Q17 in task
        'Q50': 'Reasoning'         # Q50 in task
    }

    # Select only relevant columns
    df_relevant = df[list(col_map.keys())].copy()
    df_relevant.rename(columns=col_map, inplace=True)

    # --- Data Cleaning ---

    # 1. Years Experience (Q23) -> Numeric
    # Use regex to extract numeric value from strings like "5 years", "10+"
    # Also handle float-like strings "2.5".
    # Regex: Extract digits, optional decimal point and more digits.
    # Note: If regex fails (no match), it returns NaN.
    # Pattern: extract (\d+(\.\d+)?) -> this captures 2.5 or 5.
    # The output of extract is a DataFrame if multiple groups. We want just the match.
    # Actually, simpler: extract(r'(\d+\.?\d*)') should work for 5, 10, 2.5.
    df_relevant['Years_Experience'] = df_relevant['Years_Experience'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # 2. CPA Likelihood (Q29) -> 1-5 Scale
    likelihood_map = {
        'Very likely': 5,
        'Somewhat likely': 4,
        'Neither likely nor unlikely': 3,
        'Somewhat unlikely': 2,
        'Very unlikely': 1
    }
    df_relevant['CPA_Likelihood_Score'] = df_relevant['CPA_Likelihood'].map(likelihood_map)

    # 3. Perception (Q6) -> 1-5 Scale
    perception_map = {
        'Very Positive': 5,
        'Somewhat Positive': 4,
        'Neutral': 3,
        'Somewhat Negative': 2,
        'Very Negative': 1
    }
    df_relevant['Perception_Score'] = df_relevant['Perception'].map(perception_map)

    # --- Analysis Goals ---

    output_lines = []
    output_lines.append("--- Analysis Results ---\n")

    # 1. Correlation: Years Experience vs CPA Likelihood
    # Filter valid rows
    corr_df = df_relevant.dropna(subset=['Years_Experience', 'CPA_Likelihood_Score'])
    correlation = corr_df['Years_Experience'].corr(corr_df['CPA_Likelihood_Score'])
    output_lines.append(f"1. Correlation between Years Experience and CPA Likelihood: {correlation:.4f}\n")

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(data=corr_df, x='Years_Experience', y='CPA_Likelihood_Score', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Relationship between Work Experience and CPA Likelihood')
    plt.xlabel('Years of Work Experience')
    plt.ylabel('CPA Likelihood (1-5 Scale)')
    plt.savefig('scatter_plot.png')
    plt.close()

    # 2. Segmentation
    # Q17 (Work_CPA_Firm) Yes/No
    # Q16 (Work_20hrs_Acc) Yes/No
    # Compare mean CPA Likelihood (Q29) AND Perception (Q6) - Updated requirement

    output_lines.append("2. Segmentation Analysis (Mean CPA Likelihood & Perception):\n")

    # CPA Firm
    cpa_firm_stats = df_relevant.groupby('Work_CPA_Firm')[['CPA_Likelihood_Score', 'Perception_Score']].agg(['mean', 'count'])
    output_lines.append("   - Working in CPA Firm (Q47/Q17):")
    output_lines.append(cpa_firm_stats.to_string())
    output_lines.append("\n")

    # 20+ Hours Accounting
    acc_work_stats = df_relevant.groupby('Work_20hrs_Acc')[['CPA_Likelihood_Score', 'Perception_Score']].agg(['mean', 'count'])
    output_lines.append("   - Working 20+ Hours in Accounting (Q46/Q16):")
    output_lines.append(acc_work_stats.to_string())
    output_lines.append("\n")

    # 3. Experience Brackets
    # 0 years, 1-2 years, 3-5 years, 5+ years

    def get_bracket(x):
        if pd.isna(x):
            return np.nan
        # Round to nearest integer for bracket classification if needed,
        # or just use ranges directly on float.
        # Task implied integer years ("5 years", "10+").
        # If x is 2.5, where does it go? 1-2 or 3-5?
        # Usually 1-2 means [1, 2]. 3-5 means [3, 5].
        # 2.5 is > 2 and < 3.
        # I'll use simple binning:
        # 0: < 0.5 (effectively 0)
        # 1-2: 0.5 <= x <= 2.5
        # 3-5: 2.5 < x <= 5.5
        # 5+: > 5.5
        # Wait, simple integer check:
        r = round(x)
        if r == 0:
            return '0 years'
        elif r <= 2:
            return '1-2 years'
        elif r <= 5:
            return '3-5 years'
        else:
            return '5+ years'

    df_relevant['Experience_Bracket'] = df_relevant['Years_Experience'].apply(get_bracket)

    # Order brackets for plotting
    bracket_order = ['0 years', '1-2 years', '3-5 years', '5+ years']
    df_relevant['Experience_Bracket'] = pd.Categorical(df_relevant['Experience_Bracket'], categories=bracket_order, ordered=True)

    bracket_stats = df_relevant.groupby('Experience_Bracket', observed=True)[['CPA_Likelihood_Score', 'Perception_Score']].mean()
    output_lines.append("3. Experience Brackets Means:\n")
    output_lines.append(bracket_stats.to_string())
    output_lines.append("\n")

    # Bar Chart for Mean CPA Likelihood
    plt.figure(figsize=(10, 6))
    sns.barplot(x=bracket_stats.index, y=bracket_stats['CPA_Likelihood_Score'], order=bracket_order)
    plt.title('Mean CPA Likelihood by Experience Bracket')
    plt.xlabel('Experience Bracket')
    plt.ylabel('Mean CPA Likelihood Score')
    plt.ylim(1, 5) # Scale is 1-5
    plt.savefig('bar_chart.png')
    plt.close()

    # 4. Qualitative Synergy (Q50)
    # Most frequent keywords for "5+ years" vs "0 years"
    output_lines.append("4. Qualitative Synergy (Top 10 Keywords in Reasoning):\n")

    stopwords = set(['i', 'to', 'the', 'and', 'a', 'of', 'it', 'is', 'in', 'that', 'for', 'my', 'have', 'with', 'on', 'be', 'this', 'as', 'but', 'not', 'are', 'or', 'so', 'me', 'am', 'was', 'at', 'would', 'more', 'an', 'if', 'pathway', 'cpa', 'license', 'year', 'experience', 'work', 'hours', 'credit', 'credits', '150'])

    def get_top_keywords(text_series, n=10):
        all_text = ' '.join(text_series.dropna().astype(str).tolist()).lower()
        # Remove punctuation
        all_text = re.sub(r'[^\w\s]', '', all_text)
        words = all_text.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        return Counter(filtered_words).most_common(n)

    # 0 Years
    reasons_0 = df_relevant[df_relevant['Experience_Bracket'] == '0 years']['Reasoning']
    top_0 = get_top_keywords(reasons_0)
    output_lines.append("   - 0 Years Experience Group:")
    output_lines.append(str(top_0))
    output_lines.append("\n")

    # 5+ Years
    reasons_5plus = df_relevant[df_relevant['Experience_Bracket'] == '5+ years']['Reasoning']
    top_5plus = get_top_keywords(reasons_5plus)
    output_lines.append("   - 5+ Years Experience Group:")
    output_lines.append(str(top_5plus))
    output_lines.append("\n")

    # Write summary to file
    with open('summary_report.txt', 'w') as f:
        f.writelines(output_lines)

    print("Analysis complete. Outputs generated: scatter_plot.png, bar_chart.png, summary_report.txt")

if __name__ == "__main__":
    main()
