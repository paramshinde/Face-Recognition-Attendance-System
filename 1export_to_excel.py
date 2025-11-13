import sqlite3
import pandas as pd
import os

# Connect to the SQLite database
db_path = 'E:/Final Year-1/attendance.db'
conn = sqlite3.connect(db_path)

# Read all records from the attendance table
df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)

# Debug output
print(f"[DEBUG] Fetched {len(df)} rows from database.")
print(df.head())

# Ensure Excel file is saved to correct location
excel_path = 'E:/Final Year-1/attendance.xlsx'
df.to_excel(excel_path, index=False)

print(f"Attendance exported to '{excel_path}' successfully.")

conn.close()
