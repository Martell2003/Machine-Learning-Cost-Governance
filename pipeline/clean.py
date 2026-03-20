"""
pipeline/clean.py

Reads raw period data from PostgreSQL, applies three cleaning steps,
and writes the cleaned data back. Also flags low-quality projects.

Run after ingest.py:
    python pipeline/clean.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine


def load_periods():
    """Load all periods from the database into a DataFrame."""
    query = """
        SELECT 
            p.period_id,
            p.project_id,
            p.period_number,
            p.planned_value,
            p.earned_value,
            p.actual_cost,
            pr.budget_at_completion
        FROM periods p
        JOIN projects pr ON p.project_id = pr.project_id
        ORDER BY p.project_id, p.period_number
    """
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} period rows across {df['project_id'].nunique()} projects")
    return df


def forward_fill(df):
    """
    For each project, fill missing EV, PV and AC values by carrying
    forward the last known value, then backfill any remaining.
    """
    before = df[['earned_value', 'planned_value', 'actual_cost']].isna().sum().sum()

    df = df.sort_values(['project_id', 'period_number'])
    
    # Forward fill within each project
    df[['earned_value', 'planned_value', 'actual_cost']] = (
        df.groupby('project_id')[['earned_value', 'planned_value', 'actual_cost']]
        .transform(lambda x: x.ffill())
    )
    
    # Backfill any remaining nulls (usually first period)
    df[['earned_value', 'planned_value', 'actual_cost']] = (
        df.groupby('project_id')[['earned_value', 'planned_value', 'actual_cost']]
        .transform(lambda x: x.bfill())
    )

    after = df[['earned_value', 'planned_value', 'actual_cost']].isna().sum().sum()
    filled = before - after
    print(f"  Forward-fill: filled {filled} missing values ({before} before → {after} remaining)")
    return df


def cap_outliers(df):
    """
    For each project, cap AC values that are extreme outliers using the
    IQR method. Values above Q3 + 1.5 * IQR are capped at that threshold.
    """
    capped_count = 0

    def cap_project_ac(group):
        nonlocal capped_count
        q1 = group['actual_cost'].quantile(0.25)
        q3 = group['actual_cost'].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr

        outliers = (group['actual_cost'] > upper_limit).sum()
        capped_count += outliers

        group['actual_cost'] = group['actual_cost'].clip(upper=upper_limit)
        return group

    df = df.groupby('project_id').apply(cap_project_ac).reset_index(drop=True)
    print(f"  Outlier capping: capped {capped_count} AC values across all projects")
    return df


def flag_quality(df):
    """
    For each project, calculate what percentage of periods have valid EV data.
    Projects below 80% completeness are flagged as 'fail'.
    """
    results = []

    for project_id, group in df.groupby('project_id'):
        total_periods = len(group)
        valid_ev = group['earned_value'].notna().sum()
        completeness = round((valid_ev / total_periods) * 100, 1) if total_periods > 0 else 0
        flag = 'pass' if completeness >= 80 else 'fail'

        results.append({
            'project_id': project_id,
            'completeness': completeness,
            'quality_flag': flag
        })

    quality_df = pd.DataFrame(results)

    passed = (quality_df['quality_flag'] == 'pass').sum()
    failed = (quality_df['quality_flag'] == 'fail').sum()
    print(f"  Quality flags: {passed} passed, {failed} failed (threshold: 80% completeness)")

    if failed > 0:
        failed_projects = quality_df[quality_df['quality_flag'] == 'fail']['project_id'].tolist()
        print(f"    Failed project IDs: {failed_projects}")

    return quality_df


def save_cleaned_periods(df):
    """Update the cleaned EV, PV and AC values back into the periods table."""
    clean_subset = df[['period_id', 'planned_value', 'earned_value', 'actual_cost']]
    clean_subset.to_sql(
        name='periods_cleaned_temp',
        con=engine,
        if_exists='replace',
        index=False
    )

    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE periods
            SET
                planned_value = t.planned_value,
                earned_value  = t.earned_value,
                actual_cost   = t.actual_cost
            FROM periods_cleaned_temp t
            WHERE periods.period_id = t.period_id
        """))
        conn.execute(text("DROP TABLE IF EXISTS periods_cleaned_temp"))
        conn.commit()

    print(f"  Saved {len(df)} cleaned period rows back to database")


def save_quality_flags(quality_df):
    """Add quality_flag and completeness_pct columns to projects table."""
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE projects
            ADD COLUMN IF NOT EXISTS quality_flag VARCHAR(10) DEFAULT 'pass'
        """))
        conn.execute(text("""
            ALTER TABLE projects
            ADD COLUMN IF NOT EXISTS completeness_pct NUMERIC(5,1) DEFAULT 100.0
        """))
        conn.commit()

    with engine.connect() as conn:
        for _, row in quality_df.iterrows():
            conn.execute(
                text("""
                    UPDATE projects
                    SET quality_flag = :flag,
                        completeness_pct = :completeness
                    WHERE project_id = :project_id
                """),
                {
                    'flag': row['quality_flag'],
                    'completeness': row['completeness'],
                    'project_id': row['project_id']
                }
            )
        conn.commit()

    print(f"  Quality flags saved to projects table")


def print_summary(df, quality_df):
    """Print a clean summary of what the cleaning process produced."""
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Total projects        : {df['project_id'].nunique()}")
    print(f"  Total periods         : {len(df)}")
    print(f"  Remaining nulls (EV)  : {df['earned_value'].isna().sum()}")
    print(f"  Remaining nulls (PV)  : {df['planned_value'].isna().sum()}")
    print(f"  Remaining nulls (AC)  : {df['actual_cost'].isna().sum()}")
    print()

    print("  Per-project quality:")
    print(f"  {'Project ID':<14} {'Periods':<10} {'Completeness':<16} {'Flag'}")
    print(f"  {'-'*14} {'-'*10} {'-'*16} {'-'*6}")
    for _, row in quality_df.sort_values('project_id').iterrows():
        pid = int(row['project_id'])
        comp = row['completeness']
        flag = row['quality_flag']
        total = len(df[df['project_id'] == pid])
        marker = "  ← EXCLUDED from training" if flag == 'fail' else ""
        print(f"  {pid:<14} {total:<10} {comp:<16} {flag}{marker}")

    print("=" * 60)


def run_cleaning():
    """Run the complete cleaning pipeline"""
    print("\nStarting data cleaning pipeline...\n")

    df = load_periods()

    print("\nApplying cleaning steps:")
    df = forward_fill(df)
    df = cap_outliers(df)

    quality_df = flag_quality(df)

    print("\nSaving results:")
    save_cleaned_periods(df)
    save_quality_flags(quality_df)

    print_summary(df, quality_df)
    print("\nCleaning complete. Run pipeline/split.py next.")


if __name__ == "__main__":
    run_cleaning()