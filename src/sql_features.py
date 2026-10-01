from pathlib import Path
import sqlite3
import pandas as pd

def build_feature_table() -> pd.DataFrame:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(Path("sql/schema.sql").read_text())
        pd.read_csv("data/customers.csv").to_sql("customers", conn, if_exists="append", index=False)
        pd.read_csv("data/customer_monthly_usage.csv").to_sql("customer_monthly_usage", conn, if_exists="append", index=False)
        sql = Path("sql/features.sql").read_text().strip().rstrip(";")
        conn.executescript(f"CREATE VIEW churn_features AS {sql};")
        df = pd.read_sql_query("SELECT * FROM churn_features", conn)
        Path("data").mkdir(exist_ok=True)
        df.to_csv("data/model_features.csv", index=False)
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    print(build_feature_table().shape)
