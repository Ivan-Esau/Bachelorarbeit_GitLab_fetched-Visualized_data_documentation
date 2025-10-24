"""
Generate Visual Tables for All Comparison Diagrams

Creates PNG images of comprehensive statistical tables showing:
- Type A vs Type B statistics
- Differences and variance analysis
- Standard deviation and variance prominently displayed
- Visual formatting for easy interpretation

Output:
- One table image per comparison diagram
- Saved to visualizations/comparisons/tables/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("white")

def create_styled_table(data, title, filename, column_widths=None):
    """
    Create a professionally styled table as PNG

    Args:
        data: List of lists containing table data
        title: Title for the table
        filename: Output filename
        column_widths: Optional list of column width ratios
    """
    fig, ax = plt.subplots(figsize=(16, len(data) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Add title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Create table
    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     colWidths=column_widths)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Header styling
    for i in range(len(data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#34495E')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('white')

    # Data row styling
    for i in range(1, len(data)):
        for j in range(len(data[0])):
            cell = table[(i, j)]

            # Alternate row colors
            if i % 2 == 0:
                cell.set_facecolor('#ECF0F1')
            else:
                cell.set_facecolor('white')

            # Highlight specific columns
            if 'Difference' in str(data[0][j]) or 'Variance' in str(data[0][j]):
                cell.set_facecolor('#FEF5E7')

            cell.set_edgecolor('#BDC3C7')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Table saved: {filename}")


def generate_project_duration_table():
    """Generate table for project duration comparison"""
    df = pd.read_csv('visualizations/comparisons/project_duration_a_vs_b_statistics.csv')

    # Extract summary statistics
    type_a = df[df['Type'] == 'A']
    type_b = df[df['Type'] == 'B']

    # Calculate statistics
    a_durations = type_a['Duration (minutes)'].dropna().head(10)
    b_durations = type_b['Duration (minutes)'].dropna().head(10)

    data = [
        ['Metric', 'Type A', 'Type B', 'Difference (A-B)', 'Relative Diff (%)'],
        ['Count (projects)', f'{len(a_durations)}', f'{len(b_durations)}', '-', '-'],
        ['Mean (minutes)', f'{a_durations.mean():.1f}', f'{b_durations.mean():.1f}',
         f'{a_durations.mean() - b_durations.mean():+.1f}',
         f'{((a_durations.mean() - b_durations.mean()) / b_durations.mean() * 100):+.1f}%'],
        ['Median (minutes)', f'{a_durations.median():.1f}', f'{b_durations.median():.1f}',
         f'{a_durations.median() - b_durations.median():+.1f}',
         f'{((a_durations.median() - b_durations.median()) / b_durations.median() * 100):+.1f}%'],
        ['Std Dev (minutes)', f'{a_durations.std():.1f}', f'{b_durations.std():.1f}',
         f'{a_durations.std() - b_durations.std():+.1f}', '-'],
        ['Variance (minutes²)', f'{a_durations.var():.1f}', f'{b_durations.var():.1f}',
         f'{a_durations.var() - b_durations.var():+.1f}', '-'],
        ['Min (minutes)', f'{a_durations.min():.1f}', f'{b_durations.min():.1f}',
         f'{a_durations.min() - b_durations.min():+.1f}', '-'],
        ['Max (minutes)', f'{a_durations.max():.1f}', f'{b_durations.max():.1f}',
         f'{a_durations.max() - b_durations.max():+.1f}', '-'],
        ['Range (minutes)', f'{a_durations.max() - a_durations.min():.1f}',
         f'{b_durations.max() - b_durations.min():.1f}',
         f'{(a_durations.max() - a_durations.min()) - (b_durations.max() - b_durations.min()):+.1f}', '-'],
        ['CV (Coef. of Variation)', f'{(a_durations.std() / a_durations.mean() * 100):.1f}%',
         f'{(b_durations.std() / b_durations.mean() * 100):.1f}%', '-', '-']
    ]

    output_dir = Path('visualizations/comparisons/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    create_styled_table(data,
                       'Project Duration Comparison: Type A vs Type B\nOverall Project Timeline (First to Last Commit)',
                       output_dir / 'project_duration_comparison_table.png')


def generate_issue_duration_table():
    """Generate table for issue duration comparison"""
    df = pd.read_csv('visualizations/comparisons/issue_duration_a_vs_b_statistics.csv')

    # Overall statistics
    overall = df[df['Issue'] == 'OVERALL'].iloc[0]

    data = [
        ['Metric', 'Type A', 'Type B', 'Difference (A-B)', 'Relative Diff (%)'],
        ['Count (branches)', f'{int(overall["Type A Count"])}', f'{int(overall["Type B Count"])}',
         f'{int(overall["Type A Count"]) - int(overall["Type B Count"]):+d}', '-'],
        ['Mean (minutes)', f'{overall["Type A Mean (minutes)"]:.1f}', f'{overall["Type B Mean (minutes)"]:.1f}',
         f'{overall["Difference Mean (A-B)"]:+.1f}',
         f'{(overall["Difference Mean (A-B)"] / overall["Type B Mean (minutes)"] * 100):+.1f}%'],
        ['Median (minutes)', f'{overall["Type A Median (minutes)"]:.1f}', f'{overall["Type B Median (minutes)"]:.1f}',
         f'{overall["Difference Median (A-B)"]:+.1f}',
         f'{(overall["Difference Median (A-B)"] / overall["Type B Median (minutes)"] * 100):+.1f}%'],
        ['Std Dev (minutes)', f'{overall["Type A Std (minutes)"]:.1f}', f'{overall["Type B Std (minutes)"]:.1f}',
         f'{overall["Type A Std (minutes)"] - overall["Type B Std (minutes)"]:+.1f}', '-'],
        ['Variance (minutes²)', f'{overall["Type A Std (minutes)"]**2:.1f}', f'{overall["Type B Std (minutes)"]**2:.1f}',
         f'{overall["Type A Std (minutes)"]**2 - overall["Type B Std (minutes)"]**2:+.1f}', '-'],
        ['Min (minutes)', f'{overall["Type A Min (minutes)"]:.1f}', f'{overall["Type B Min (minutes)"]:.1f}',
         f'{overall["Type A Min (minutes)"] - overall["Type B Min (minutes)"]:+.1f}', '-'],
        ['Max (minutes)', f'{overall["Type A Max (minutes)"]:.1f}', f'{overall["Type B Max (minutes)"]:.1f}',
         f'{overall["Type A Max (minutes)"] - overall["Type B Max (minutes)"]:+.1f}', '-'],
        ['CV (Coef. of Variation)',
         f'{(overall["Type A Std (minutes)"] / overall["Type A Mean (minutes)"] * 100):.1f}%',
         f'{(overall["Type B Std (minutes)"] / overall["Type B Mean (minutes)"] * 100):.1f}%', '-', '-']
    ]

    output_dir = Path('visualizations/comparisons/tables')
    create_styled_table(data,
                       'Issue Duration Comparison: Type A vs Type B\nIndividual Branch Development Time',
                       output_dir / 'issue_duration_comparison_table.png')


def generate_issue_duration_per_issue_table():
    """Generate detailed per-issue duration table"""
    df = pd.read_csv('visualizations/comparisons/issue_duration_a_vs_b_statistics.csv')

    # Filter out OVERALL row
    df = df[df['Issue'] != 'OVERALL']

    data = [['Issue', 'Type A Mean', 'Type A Std', 'Type B Mean', 'Type B Std',
             'Difference Mean', 'Difference %', 'Variance Ratio (B/A)']]

    for _, row in df.iterrows():
        var_ratio = (row['Type B Std (minutes)']**2) / (row['Type A Std (minutes)']**2) if row['Type A Std (minutes)'] > 0 else 0
        data.append([
            row['Issue'],
            f'{row["Type A Mean (minutes)"]:.1f}m',
            f'{row["Type A Std (minutes)"]:.1f}m',
            f'{row["Type B Mean (minutes)"]:.1f}m',
            f'{row["Type B Std (minutes)"]:.1f}m',
            f'{row["Difference Mean (A-B)"]:+.1f}m',
            f'{(row["Difference Mean (A-B)"] / row["Type B Mean (minutes)"] * 100):+.1f}%',
            f'{var_ratio:.2f}x'
        ])

    output_dir = Path('visualizations/comparisons/tables')
    create_styled_table(data,
                       'Per-Issue Duration Analysis\nMean Duration and Variability Comparison',
                       output_dir / 'issue_duration_per_issue_table.png',
                       column_widths=[0.12, 0.13, 0.12, 0.13, 0.12, 0.14, 0.12, 0.12])


def generate_merge_success_table():
    """Generate table for merge success comparison"""
    df = pd.read_csv('visualizations/comparisons/merge_success_a_vs_b_statistics.csv')

    # Get totals
    total_a = df[df['Project'] == 'TOTAL'][df['Type'] == 'A'].iloc[0]
    total_b = df[df['Project'] == 'TOTAL'][df['Type'] == 'B'].iloc[0]

    data = [
        ['Metric', 'Type A', 'Type B', 'Difference (A-B)', 'Relative Diff (%)'],
        ['Total Branches', f'{int(total_a["Total Branches"])}', f'{int(total_b["Total Branches"])}',
         f'{int(total_a["Total Branches"]) - int(total_b["Total Branches"]):+d}', '-'],
        ['Total Merges', f'{int(total_a["Total Merges"])}', f'{int(total_b["Total Merges"])}',
         f'{int(total_a["Total Merges"]) - int(total_b["Total Merges"]):+d}',
         f'{((total_a["Total Merges"] - total_b["Total Merges"]) / total_b["Total Merges"] * 100):+.1f}%'],
        ['Valid Merges', f'{int(total_a["Valid Merges"])}', f'{int(total_b["Valid Merges"])}',
         f'{int(total_a["Valid Merges"]) - int(total_b["Valid Merges"]):+d}',
         f'{((total_a["Valid Merges"] - total_b["Valid Merges"]) / total_b["Valid Merges"] * 100):+.1f}%'],
        ['Invalid Merges', f'{int(total_a["Invalid Merges"])}', f'{int(total_b["Invalid Merges"])}',
         f'{int(total_a["Invalid Merges"]) - int(total_b["Invalid Merges"]):+d}', '-'],
        ['Success Rate (%)', f'{total_a["Success Rate (%)"]:.1f}%', f'{total_b["Success Rate (%)"]:.1f}%',
         f'{total_a["Success Rate (%)"] - total_b["Success Rate (%)"]:+.1f}%',
         f'{((total_a["Success Rate (%)"] - total_b["Success Rate (%)"]) / total_b["Success Rate (%)"] * 100):+.1f}%'],
        ['Branch Success Rate (%)', f'{total_a["Branch Success Rate (%)"]:.1f}%', f'{total_b["Branch Success Rate (%)"]:.1f}%',
         f'{total_a["Branch Success Rate (%)"] - total_b["Branch Success Rate (%)"]:+.1f}%',
         f'{((total_a["Branch Success Rate (%)"] - total_b["Branch Success Rate (%)"]) / total_b["Branch Success Rate (%)"] * 100):+.1f}%'],
        ['Open MRs', f'{int(total_a["Open MRs"])}', f'{int(total_b["Open MRs"])}',
         f'{int(total_a["Open MRs"]) - int(total_b["Open MRs"]):+d}', '-']
    ]

    output_dir = Path('visualizations/comparisons/tables')
    create_styled_table(data,
                       'Merge Success Comparison: Type A vs Type B\nMerge Request Quality and Completion Rates',
                       output_dir / 'merge_success_comparison_table.png')


def generate_pipeline_success_table():
    """Generate table for pipeline success comparison"""
    df = pd.read_csv('visualizations/comparisons/pipeline_success_a_vs_b_statistics.csv')

    # Get totals
    total_a = df[df['Project'] == 'TOTAL'][df['Type'] == 'A'].iloc[0]
    total_b = df[df['Project'] == 'TOTAL'][df['Type'] == 'B'].iloc[0]

    data = [
        ['Pipeline State', 'Type A (%)', 'Type B (%)', 'Difference (A-B)'],
        ['Total Pipelines', f'{int(total_a["Total Pipelines"])}', f'{int(total_b["Total Pipelines"])}',
         f'{int(total_a["Total Pipelines"]) - int(total_b["Total Pipelines"]):+d}'],
        ['Both Success (Build+Test)', f'{total_a["Both Success (%)"]:.1f}%', f'{total_b["Both Success (%)"]:.1f}%',
         f'{total_a["Both Success (%)"] - total_b["Both Success (%)"]:+.1f}%'],
        ['Build Only (Test Failed)', f'{total_a["Build Only (%)"]:.1f}%', f'{total_b["Build Only (%)"]:.1f}%',
         f'{total_a["Build Only (%)"] - total_b["Build Only (%)"]:+.1f}%'],
        ['Build Failed', f'{total_a["Build Failed (%)"]:.1f}%', f'{total_b["Build Failed (%)"]:.1f}%',
         f'{total_a["Build Failed (%)"] - total_b["Build Failed (%)"]:+.1f}%'],
        ['Canceled', f'{total_a["Canceled (%)"]:.1f}%', f'{total_b["Canceled (%)"]:.1f}%',
         f'{total_a["Canceled (%)"] - total_b["Canceled (%)"]:+.1f}%']
    ]

    # Calculate variability across projects
    projects_a = df[(df['Type'] == 'A') & (df['Project'] != 'TOTAL')]
    projects_b = df[(df['Type'] == 'B') & (df['Project'] != 'TOTAL')]

    data.append(['', '', '', ''])
    data.append(['Variability Analysis - Both Success (%)', 'Type A', 'Type B', 'Difference (A-B)'])

    metric = 'Both Success (%)'

    # Calculate statistics
    mean_a = projects_a[metric].mean()
    mean_b = projects_b[metric].mean()
    std_a = projects_a[metric].std()
    std_b = projects_b[metric].std()
    variance_a = std_a ** 2
    variance_b = std_b ** 2
    min_a = projects_a[metric].min()
    min_b = projects_b[metric].min()
    max_a = projects_a[metric].max()
    max_b = projects_b[metric].max()
    var_ratio = variance_b / variance_a if variance_a > 0 else 0

    # Add rows
    data.append(['Mean', f'{mean_a:.1f}%', f'{mean_b:.1f}%', f'{mean_a - mean_b:+.1f}%'])
    data.append(['Std Deviation', f'{std_a:.1f}%', f'{std_b:.1f}%', f'{std_a - std_b:+.1f}%'])
    data.append(['Variance (Std²)', f'{variance_a:.1f}', f'{variance_b:.1f}', f'{variance_a - variance_b:+.1f}'])
    data.append(['Min', f'{min_a:.1f}%', f'{min_b:.1f}%', f'{min_a - min_b:+.1f}%'])
    data.append(['Max', f'{max_a:.1f}%', f'{max_b:.1f}%', f'{max_a - max_b:+.1f}%'])
    data.append(['Range (Max-Min)', f'{max_a - min_a:.1f}%', f'{max_b - min_b:.1f}%',
                 f'{(max_a - min_a) - (max_b - min_b):+.1f}%'])
    data.append(['Variance Ratio (B/A)', '-', '-', f'{var_ratio:.2f}x'])

    output_dir = Path('visualizations/comparisons/tables')
    create_styled_table(data,
                       'Pipeline Success Comparison: Type A vs Type B\nBuild and Test Outcome Distribution',
                       output_dir / 'pipeline_success_comparison_table.png')


def generate_correlation_table():
    """Generate table for infrastructure correlation comparison"""
    df = pd.read_csv('visualizations/comparisons/duration_correlation_statistics.csv')

    row_a = df[df['Metric'].str.contains('Type A|Correlation.*A', na=False)]
    row_b = df[df['Metric'].str.contains('Type B|Correlation.*B', na=False)]

    # Extract values (this CSV has a different structure)
    data = [
        ['Metric', 'Type A', 'Type B', 'Difference (A-B)'],
        ['Number of Projects', '10', '10', '0'],
        ['Avg Duration (seconds)',
         f'{df.iloc[2, 1]:.1f}' if len(df) > 2 else 'N/A',
         f'{df.iloc[2, 2]:.1f}' if len(df) > 2 else 'N/A',
         f'{df.iloc[2, 3]:+.1f}' if len(df) > 2 else 'N/A'],
        ['Median Duration (seconds)',
         f'{df.iloc[3, 1]:.1f}' if len(df) > 3 else 'N/A',
         f'{df.iloc[3, 2]:.1f}' if len(df) > 3 else 'N/A',
         f'{df.iloc[3, 3]:+.1f}' if len(df) > 3 else 'N/A'],
        ['Avg Cancellation Rate (%)',
         f'{df.iloc[4, 1]:.1f}%' if len(df) > 4 else 'N/A',
         f'{df.iloc[4, 2]:.1f}%' if len(df) > 4 else 'N/A',
         f'{df.iloc[4, 3]:+.1f}%' if len(df) > 4 else 'N/A'],
        ['Branch Success Rate (%)',
         f'{df.iloc[6, 1]:.1f}%' if len(df) > 6 else 'N/A',
         f'{df.iloc[6, 2]:.1f}%' if len(df) > 6 else 'N/A',
         f'{df.iloc[6, 3]:+.1f}%' if len(df) > 6 else 'N/A'],
        ['Correlation (Duration vs Success)',
         f'{df.iloc[8, 1]:.3f}' if len(df) > 8 else 'N/A',
         f'{df.iloc[8, 2]:.3f}' if len(df) > 8 else 'N/A',
         f'{df.iloc[8, 3]:+.3f}' if len(df) > 8 else 'N/A'],
        ['R² (Duration vs Success)',
         f'{df.iloc[9, 1]:.3f}' if len(df) > 9 else 'N/A',
         f'{df.iloc[9, 2]:.3f}' if len(df) > 9 else 'N/A',
         f'{df.iloc[9, 3]:+.3f}' if len(df) > 9 else 'N/A']
    ]

    output_dir = Path('visualizations/comparisons/tables')
    create_styled_table(data,
                       'Infrastructure Correlation Analysis: Type A vs Type B\nRelationship Between Duration and Success Rate',
                       output_dir / 'correlation_comparison_table.png')


def generate_coverage_table():
    """Generate table for coverage comparison"""
    df = pd.read_csv('visualizations/comparisons/coverage_per_branch_a_vs_b_statistics.csv')

    # Get overall statistics
    overall = df[df['Issue'] == 'OVERALL'].iloc[0]

    data = [
        ['Metric', 'Type A', 'Type B', 'Difference (A-B)'],
        ['Overall Mean Coverage (%)', f'{overall["Type A Mean (%)"]:.1f}%', f'{overall["Type B Mean (%)"]:.1f}%',
         f'{overall["Difference (A-B)"]:+.1f}%'],
        ['Overall Median Coverage (%)', f'{df[df["Issue"] == "MEDIAN"]["Type A Mean (%)"].values[0]:.1f}%' if 'MEDIAN' in df['Issue'].values else 'N/A',
         f'{df[df["Issue"] == "MEDIAN"]["Type B Mean (%)"].values[0]:.1f}%' if 'MEDIAN' in df['Issue'].values else 'N/A',
         f'{df[df["Issue"] == "MEDIAN"]["Difference (A-B)"].values[0]:+.1f}%' if 'MEDIAN' in df['Issue'].values else 'N/A']
    ]

    # Calculate variability per issue
    issues_df = df[(df['Issue'] != 'OVERALL') & (df['Issue'] != 'MEDIAN') & (df['Issue'] != '')]

    if len(issues_df) > 0:
        data.append(['', '', '', ''])
        data.append(['Variability Analysis', 'Type A Std', 'Type B Std', 'Avg Difference'])

        for _, row in issues_df.iterrows():
            data.append([
                row['Issue'],
                f'{row["Type A Std Dev"]:.1f}%',
                f'{row["Type B Std Dev"]:.1f}%',
                f'{row["Difference (A-B)"]:+.1f}%'
            ])

    output_dir = Path('visualizations/comparisons/tables')
    create_styled_table(data,
                       'Test Coverage Comparison: Type A vs Type B\nPer-Issue Coverage Analysis',
                       output_dir / 'coverage_comparison_table.png')


def generate_all_tables():
    """Generate all comparison tables"""
    print("\n" + "="*80)
    print("GENERATING VISUAL COMPARISON TABLES")
    print("="*80 + "\n")

    print("1. Project Duration Table...")
    generate_project_duration_table()

    print("2. Issue Duration Table (Overall)...")
    generate_issue_duration_table()

    print("3. Issue Duration Table (Per Issue)...")
    generate_issue_duration_per_issue_table()

    print("4. Merge Success Table...")
    generate_merge_success_table()

    print("5. Pipeline Success Table...")
    generate_pipeline_success_table()

    print("6. Infrastructure Correlation Table...")
    generate_correlation_table()

    print("7. Test Coverage Table...")
    generate_coverage_table()

    print("\n" + "="*80)
    print("ALL COMPARISON TABLES GENERATED")
    print("="*80)
    print("\nTables saved to: visualizations/comparisons/tables/")
    print("\nGenerated tables:")
    print("  - project_duration_comparison_table.png")
    print("  - issue_duration_comparison_table.png")
    print("  - issue_duration_per_issue_table.png")
    print("  - merge_success_comparison_table.png")
    print("  - pipeline_success_comparison_table.png")
    print("  - correlation_comparison_table.png")
    print("  - coverage_comparison_table.png")
    print()


if __name__ == '__main__':
    generate_all_tables()
