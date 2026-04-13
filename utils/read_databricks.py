# Read Data from a SQL Warehouse
# Connects to a Databricks SQL Warehouse from outside Databricks (e.g. your laptop)
# using either:
#   - OAuth U2M (browser-based login, no token needed), or
#   - PAT (Personal Access Token, prompted at runtime)
#
# Prerequisites:
#   pip install databricks-sql-connector pandas
#
# Required before running:
#   1. Get connection details from:
#      SQL Warehouses → <your warehouse> → Connection Details tab
#   2. Fill in server_hostname and http_path below
#   3. For PAT: generate a token at User Settings → Developer → Access Tokens

import getpass

import pandas as pd
from databricks import sql

server_hostname = "adb-3159846276042543.3.azuredatabricks.net"
http_path = "/sql/1.0/warehouses/8248ab4890b97e06"

catalog = "dbw_datascience_tst_weu_001"
schema = "default"
table = "ow_101001_waterstand_20250723_115411"
query = f"SELECT * FROM {catalog}.{schema}.{table} LIMIT 100"

# --- Authenticate with PAT, or fall back to OAuth if left blank ---
# pat = getpass.getpass("Enter your Databricks Personal Access Token (leave blank for OAuth browser login): ")
pat = input("Enter your Databricks Personal Access Token (leave blank for OAuth browser login): ")

if pat:
    connect_kwargs = dict(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=pat,
    )
else:
    print("No token provided — using OAuth U2M (a browser window will open).")
    connect_kwargs = dict(
        server_hostname=server_hostname,
        http_path=http_path,
        auth_type="databricks-oauth",
    )

with sql.connect(**connect_kwargs) as connection:
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

df = pd.DataFrame(rows, columns=columns)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(df.head())
