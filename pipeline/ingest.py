import openpyxl
import pandas as pd
from sqlalchemy import text
import os
import sys

# This lets ingest.py find config.py in the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW
from pipeline.db import engine

def extract_project_info(wb, filename):
    """Pull top-level project facts from the Baseline Schedule sheet."""
    ws = wb['Baseline Schedule']
    # Row 3 is the top-level project summary row
    row = list(ws.iter_rows(min_row=3, max_row=3, values_only=True))[0]
    return {
        'project_name':          filename.replace('.xlsx', ''),
        'budget_at_completion':  row[13],   # Total Cost column
        'total_periods':         None,       # filled in after periods are counted
        'train_test_flag':       'train',    # default — you will update some to 'test' later
        'project_size':          'medium'    # default — update based on BAC
    }

def extract_periods(wb):
    """Pull PV, EV, AC for every tracking period sheet."""
    records = []
    for sheet_name in wb.sheetnames:
        if not sheet_name.startswith('TP'):
            continue
        ws = wb[sheet_name]
        period_num  = int(sheet_name.replace('TP', ''))
    
        # Row 5 (index 4) is always the project-level summary row
        ev = ws.cell(5, 23).value
        pv = ws.cell(5, 24).value
        ac = ws.cell(5, 19).value
        if ev is None and pv is None and ac is None:
            continue
        records.append({
            'period_number': period_num,
            #'status_date':   status_date,
            'planned_value': pv,
            'earned_value':  ev,
            'actual_cost':   ac
        })
    return pd.DataFrame(records).sort_values('period_number').reset_index(drop=True)

def save_project(project_info):
    """Insert one project record and return the generated project_id."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO projects
                    (project_name, budget_at_completion, total_periods,
                     train_test_flag, project_size)
                VALUES
                    (:project_name, :budget_at_completion, :total_periods,
                     :train_test_flag, :project_size)
                RETURNING project_id
            """),
            project_info
        )
        conn.commit()
        return result.fetchone()[0]

def save_periods(periods_df, project_id):
    """Insert all period rows for a project."""
    periods_df['project_id'] = project_id
    periods_df.to_sql(
        name='periods',
        con=engine,
        if_exists='append',
        index=False
    )
    print(f"  Saved {len(periods_df)} periods for project_id {project_id}")

def ingest_file(filepath):
    """Full ingestion pipeline for one Excel file."""
    filename = os.path.basename(filepath)
    print(f"Ingesting: {filename}")

    wb = openpyxl.load_workbook(filepath, data_only=True)

    # 1. Extract project info
    project_info = extract_project_info(wb, filename)

    # 2. Extract periods
    periods_df = extract_periods(wb)
    project_info['total_periods'] = len(periods_df)

    # 3. Save to database
    project_id = save_project(project_info)
    save_periods(periods_df, project_id)

    print(f"  Done — project_id: {project_id}, periods: {len(periods_df)}")
    return project_id

def ingest_all():
    """Loop through every Excel file in data/raw/ and ingest it."""
    raw_folder = DATA_RAW
    files = [f for f in os.listdir(raw_folder) if f.endswith('.xlsx')]

    if not files:
        print("No Excel files found in data/raw/")
        return

    print(f"Found {len(files)} files to ingest\n")
    for filename in sorted(files):
        filepath = os.path.join(raw_folder, filename)
        try:
            ingest_file(filepath)
        except Exception as e:
            print(f"  ERROR on {filename}: {e}")

    print("\nIngestion complete.")

if __name__ == "__main__":
    ingest_all()