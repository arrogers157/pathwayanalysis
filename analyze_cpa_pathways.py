import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

def main():
    # Load dataset
    csv_file = 'Alternative CPA Pathways Survey_December 31, 2025_09.45.csv'
    # Row 0 is header, Row 1 is Question Text, Row 2 is ImportId metadata
    # We skip rows 1 and 2 to get clean data starting from row 3.
    # Note: Qualtrics CSVs typically have 3 header rows. The task description mentions Row 2+ = Data,
    # but inspection confirms Row 2 contains ImportId metadata which should be skipped.
    df = pd.read_csv(csv_file, header=0, skiprows=[1, 2])

    # Mapping Dictionary based on inspection of file content vs task description.
    # Task says Q19 for Experience, but file has Q23 for "Years of Work Experience".
    # Task says Q16 for 20+ hours, but file has Q46.
    # Task says Q17 for CPA Firm, but file has Q47.
    # Task says Q6 for Perception, Q29 for Likelihood, Q50 for Reasoning - these match.
    col_map = {
        'Q23': 'Years_Experience', # Matches Q19 in task description
        'Q29': 'CPA_Likelihood',   # Matches Q29 in task description
        'Q6': 'Perception',        # Matches Q6 in task description
        'Q46': 'Work_20hrs_Acc',   # Matches Q16 in task description
        'Q47': 'Work_CPA_Firm',    # Matches Q17 in task description
        'Q50': 'Reasoning'         # Matches Q50 in task description
    }

    # Select only relevant columns
    # Use errors='ignore' in case columns are missing to prevent crash, though specific ones are needed.
    # Better to check if they exist.
    missing_cols = [c for c in col_map.keys() if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns from dataset: {missing_cols}")
        # Only map existing columns
        col_map = {k: v for k, v in col_map.items() if k in df.columns}

    df_relevant = df[list(col_map.keys())].copy()
    df_relevant.rename(columns=col_map, inplace=True)

    # --- Data Cleaning ---

    # 1. Years Experience (Q23) -> Numeric
    # Use regex to extract numeric value from strings like "5 years", "10+".
    # Also handle float-like strings "2.5".
    if 'Years_Experience' in df_relevant.columns:
        df_relevant['Years_Experience'] = df_relevant['Years_Experience'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # 2. CPA Likelihood (Q29) -> 1-5 Scale
    # Task requested "Extremely likely" -> 5. File contains "Very likely". Mapping both to 5.
    likelihood_map = {
        'Extremely likely': 5,
        'Very likely': 5,
        'Somewhat likely': 4,
        'Neither likely nor unlikely': 3,
        'Somewhat unlikely': 2,
        'Very unlikely': 1,
        'Extremely unlikely': 1
    }
    if 'CPA_Likelihood' in df_relevant.columns:
        df_relevant['CPA_Likelihood_Score'] = df_relevant['CPA_Likelihood'].map(likelihood_map)

    # 3. Perception (Q6) -> 1-5 Scale
    # Task requested "Extremely Positive" -> 5. File contains "Very Positive". Mapping both to 5.
    perception_map = {
        'Extremely Positive': 5,
        'Very Positive': 5,
        'Somewhat Positive': 4,
        'Neutral': 3,
        'Neither positive nor negative': 3,
        'Somewhat Negative': 2,
        'Very Negative': 1,
        'Extremely Negative': 1
    }
    if 'Perception' in df_relevant.columns:
        df_relevant['Perception_Score'] = df_relevant['Perception'].map(perception_map)

    # --- Analysis Goals ---

    output_lines = []
    output_lines.append("--- Analysis Results ---\n")

    # 1. Correlation: Years Experience vs CPA Likelihood
    if 'Years_Experience' in df_relevant.columns and 'CPA_Likelihood_Score' in df_relevant.columns:
        # Filter valid rows
        corr_df = df_relevant.dropna(subset=['Years_Experience', 'CPA_Likelihood_Score'])
        if not corr_df.empty:
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
        else:
            output_lines.append("1. Correlation: Insufficient data.\n")

    # 2. Segmentation
    output_lines.append("2. Segmentation Analysis (Mean CPA Likelihood & Perception):\n")

    # CPA Firm
    if 'Work_CPA_Firm' in df_relevant.columns:
        # Normalize Yes/No
        df_relevant['Work_CPA_Firm'] = df_relevant['Work_CPA_Firm'].fillna('No') # Treat NaN as No for simplicity or exclude
        cpa_firm_stats = df_relevant.groupby('Work_CPA_Firm')[['CPA_Likelihood_Score', 'Perception_Score']].agg(['mean', 'count'])
        output_lines.append("   - Working in CPA Firm (Q47/Q17):")
        output_lines.append(cpa_firm_stats.to_string())
        output_lines.append("\n")

    # 20+ Hours Accounting
    if 'Work_20hrs_Acc' in df_relevant.columns:
        df_relevant['Work_20hrs_Acc'] = df_relevant['Work_20hrs_Acc'].fillna('No')
        acc_work_stats = df_relevant.groupby('Work_20hrs_Acc')[['CPA_Likelihood_Score', 'Perception_Score']].agg(['mean', 'count'])
        output_lines.append("   - Working 20+ Hours in Accounting (Q46/Q16):")
        output_lines.append(acc_work_stats.to_string())
        output_lines.append("\n")

    # 3. Experience Brackets
    if 'Years_Experience' in df_relevant.columns:
        def get_bracket(x):
            if pd.isna(x):
                return np.nan
            # Simple binning logic
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
    output_lines.append("4. Qualitative Synergy (Top 10 Keywords in Reasoning):\n")

    stopwords = set(['i', 'to', 'the', 'and', 'a', 'of', 'it', 'is', 'in', 'that', 'for', 'my', 'have', 'with', 'on', 'be', 'this', 'as', 'but', 'not', 'are', 'or', 'so', 'me', 'am', 'was', 'at', 'would', 'more', 'an', 'if', 'pathway', 'cpa', 'license', 'year', 'experience', 'work', 'hours', 'credit', 'credits', '150'])

    def get_top_keywords(text_series, n=10):
        all_text = ' '.join(text_series.dropna().astype(str).tolist()).lower()
        # Remove punctuation
        all_text = re.sub(r'[^\w\s]', '', all_text)
        words = all_text.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        return Counter(filtered_words).most_common(n)

    if 'Experience_Bracket' in df_relevant.columns and 'Reasoning' in df_relevant.columns:
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
