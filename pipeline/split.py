"""
pipeline/split.py

Assigns each project to either the training set or the test set.
Uses a 70/30 stratified split by project size so that small, medium
and large projects are proportionally represented in both sets.

Only projects that passed the quality gate in clean.py are included.
Failed projects are excluded entirely.
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import engine


# ── STEP 1: Load projects from PostgreSQL ─────────────────────────────────────

def load_projects():
    """
    Load all projects from the database.
    Only include projects that passed the quality gate.
    """
    query = """
        SELECT
            project_id,
            project_name,
            budget_at_completion,
            project_size,
            total_periods,
            quality_flag,
            completeness_pct
        FROM projects
        ORDER BY project_id
    """
    df = pd.read_sql(query, engine)

    total    = len(df)
    passed   = (df['quality_flag'] == 'pass').sum()
    failed   = (df['quality_flag'] == 'fail').sum()

    print(f"Loaded {total} projects — {passed} passed quality gate, {failed} excluded")

    if failed > 0:
        excluded = df[df['quality_flag'] == 'fail']['project_name'].tolist()
        print(f"  Excluded: {excluded}")

    # Only work with projects that passed
    df = df[df['quality_flag'] == 'pass'].reset_index(drop=True)
    return df


# ── STEP 2: Stratified 70/30 split by project size ────────────────────────────

def stratified_split(df, train_ratio=0.70, random_seed=42):
    """
    Split projects into training and test sets.
    Stratified by project_size so each size group (small, medium, large)
    is represented proportionally in both sets.

    Why stratify by size:
    A project with a £500k budget behaves very differently from one with
    a £50m budget. If all large projects ended up in training and the
    test set only had small projects, the model would never be evaluated
    on the kind of project it needs to handle.

    Why 70/30:
    With 40+ projects, 70/30 gives roughly 28 training and 12 test projects
    — enough to train a stable Random Forest while keeping a meaningful
    held-out evaluation set (Batselier and Vanhoucke, 2015).
    """
    np.random.seed(random_seed)

    train_ids = []
    test_ids  = []

    size_groups = df.groupby('project_size')

    print(f"\nSplitting by project size (seed={random_seed}):")
    print(f"  {'Size':<10} {'Total':<8} {'Train':<8} {'Test'}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

    for size, group in size_groups:
        ids      = group['project_id'].tolist()
        n_total  = len(ids)
        n_train  = max(1, round(n_total * train_ratio))
        n_test   = n_total - n_train

        # Shuffle within each size group before splitting
        shuffled = np.random.permutation(ids).tolist()
        train_ids.extend(shuffled[:n_train])
        test_ids.extend(shuffled[n_train:])

        print(f"  {size:<10} {n_total:<8} {n_train:<8} {n_test}")

    # Handle edge case — if a size group has only 1 project, put it in training
    # and note that this size will not be represented in test
    total_train = len(train_ids)
    total_test  = len(test_ids)
    actual_ratio = round(total_train / (total_train + total_test) * 100, 1)

    print(f"\n  Total: {total_train + total_test} projects")
    print(f"  Train: {total_train} ({actual_ratio}%)")
    print(f"  Test:  {total_test} ({100 - actual_ratio}%)")

    return train_ids, test_ids


# ── STEP 3: Save the split flags to PostgreSQL ────────────────────────────────

def save_split(train_ids, test_ids):
    """
    Update the train_test_flag column in the projects table.
    Training projects get 'train', test projects get 'test'.
    """
    with engine.connect() as conn:
        # Set all passing projects to train first
        conn.execute(text("""
            UPDATE projects
            SET train_test_flag = 'train'
            WHERE quality_flag = 'pass'
        """))

        # Then set test projects
        if test_ids:
            conn.execute(
                text("""
                    UPDATE projects
                    SET train_test_flag = 'test'
                    WHERE project_id = ANY(:ids)
                """),
                {"ids": test_ids}
            )

        # Mark failed projects clearly
        conn.execute(text("""
            UPDATE projects
            SET train_test_flag = 'excluded'
            WHERE quality_flag = 'fail'
        """))

        conn.commit()

    print(f"\n  Saved split flags to database")
    print(f"  {len(train_ids)} projects marked 'train'")
    print(f"  {len(test_ids)} projects marked 'test'")


# ── STEP 4: Verify the split in the database ──────────────────────────────────

def verify_split():
    """
    Read back the split from the database and confirm everything
    looks correct before moving to feature engineering.
    """
    query = """
        SELECT
            project_id,
            project_name,
            project_size,
            budget_at_completion,
            total_periods,
            train_test_flag,
            quality_flag,
            completeness_pct
        FROM projects
        ORDER BY train_test_flag, project_size, project_id
    """
    df = pd.read_sql(query, engine)
    return df


# ── STEP 5: Summary report ─────────────────────────────────────────────────────

def print_summary(df):
    """Print the final split breakdown clearly."""
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    for flag in ['train', 'test', 'excluded']:
        group = df[df['train_test_flag'] == flag]
        if group.empty:
            continue
        print(f"\n  {flag.upper()} SET ({len(group)} projects):")
        print(f"  {'ID':<6} {'Name':<35} {'Size':<10} {'Periods':<10} {'BAC'}")
        print(f"  {'-'*6} {'-'*35} {'-'*10} {'-'*10} {'-'*15}")
        for _, row in group.iterrows():
            pid     = int(row['project_id'])
            name    = str(row['project_name'])[:33]
            size    = str(row['project_size'])
            periods = int(row['total_periods']) if row['total_periods'] else 0
            bac     = f"{row['budget_at_completion']:,.0f}" if row['budget_at_completion'] else 'N/A'
            print(f"  {pid:<6} {name:<35} {size:<10} {periods:<10} {bac}")

    # Size distribution check
    train_df = df[df['train_test_flag'] == 'train']
    test_df  = df[df['train_test_flag'] == 'test']

    if not test_df.empty:
        print(f"\n  Size distribution check:")
        print(f"  {'Size':<12} {'Train':<8} {'Test':<8} {'Train %'}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
        for size in ['small', 'medium', 'large']:
            n_train = (train_df['project_size'] == size).sum()
            n_test  = (test_df['project_size']  == size).sum()
            total   = n_train + n_test
            pct     = round(n_train / total * 100, 0) if total > 0 else 0
            print(f"  {size:<12} {n_train:<8} {n_test:<8} {pct}%")

    print("=" * 70)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_split():
    print("\nStarting train/test split...\n")

    # Load quality-passed projects
    projects_df = load_projects()

    if projects_df.empty:
        print("ERROR: No projects available for splitting.")
        print("Make sure clean.py has been run and at least one project passed the quality gate.")
        return

    if len(projects_df) < 3:
        print(f"WARNING: Only {len(projects_df)} project(s) available.")
        print("A minimum of 3 projects is recommended for a meaningful split.")
        print("All projects will be assigned to training until more data is added.")
        # Assign everything to train if too few projects
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE projects SET train_test_flag = 'train'
                WHERE quality_flag = 'pass'
            """))
            conn.commit()
        print("All projects set to 'train'. Re-run split.py after adding more data.")
        return

    # Run the split
    train_ids, test_ids = stratified_split(projects_df)

    # Save to database
    save_split(train_ids, test_ids)

    # Verify and report
    final_df = verify_split()
    print_summary(final_df)

    print("\nSplit complete. Run features/engineer.py next.")


if __name__ == "__main__":
    run_split()