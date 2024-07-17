# ======================================dashboard utility functions here========================================
import matplotlib.pyplot as plt #type:ignore
import seaborn as sns #type:ignore
import pandas as pd #type:ignore


def reading_cleaning(df):
    df.drop_duplicates(inplace=True)
    cols = df.columns.tolist()
    df.columns = [x.lower() for x in cols]

    return df

def employee_important_info(df):
    # Average satisfaction level
    average_satisfaction = df['satisfaction_level'].mean()
    # Department-wise average satisfaction level
    department_satisfaction = df.groupby('department')['satisfaction_level'].mean()
    # Salary-wise average satisfaction level
    salary_satisfaction = df.groupby('salary')['satisfaction_level'].mean()

    # Employees who left
    left_employees = len(df[df['left'] == 1])
    # Employees who stayed
    stayed_employees = len(df[df['left'] == 0])

    return average_satisfaction, department_satisfaction, salary_satisfaction, left_employees, stayed_employees

def plots(df, col):
    values = df[col].unique()
    plt.figure(figsize=(15, 10))

    explode = [0.1 if len(values) > 1 else 0] * len(values)
    plt.pie(df[col].value_counts(), explode=explode, startangle=40, autopct='%1.1f%%', shadow=True)
    labels = [f'{value} ({col})' for value in values]
    plt.legend(labels=labels, loc='upper right', fontsize=12)
    plt.title(f"Distribution of {col}", fontsize=16, fontweight='bold')

    plt.savefig('static/'+ col + '.png')
    plt.close()


def distribution(df, col):
    values = df[col].unique()
    plt.figure(figsize=(15, 10))
    sns.countplot(x=df[col], hue='left', palette='Set1', data=df)
    labels = [f"{val} ({col})" for val in values]
    plt.legend(labels=labels, loc="upper right", fontsize=12)
    plt.title(f"Distribution of {col}", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90)
    plt.savefig('static/' + col + '_distribution.png')
    plt.close()

def comparison(df, x, y):
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y, hue='left', data=df, ci=None)
    plt.title(f'{x} vs {y}', fontsize=16, fontweight='bold')
    plt.savefig('static/' + 'comparison.png')
    plt.close()


def corr_with_left(df):
    df_encoded = pd.get_dummies(df)
    correlations = df_encoded.corr()['left'].sort_values()[:-1]
    colors = ['skyblue' if corr >= 0 else 'salmon' for corr in correlations]
    plt.figure(figsize=(15, 10))
    correlations.plot(kind='barh', color=colors)
    plt.title('Correlation with Left', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/correlation.png')
    plt.close()


def histogram(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # Create a grid of 1 row and 2 columns

    # Plot the first histogram
    sns.histplot(data=df, x=col, hue='left', bins=20, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}", fontsize=16, fontweight='bold')

    # Plot the second histogram
    sns.kdeplot(data=df, x='satisfaction_level', y='last_evaluation', hue='left', shade=True, ax=axes[1])
    axes[1].set_title("Kernel Density Estimation", fontsize=16, fontweight='bold')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig('static/' + col + '_histogram.png')
    plt.close()
