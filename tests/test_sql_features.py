from pathlib import Path

def test_sql_feature_sql_has_window_functions():
    sql = Path(__file__).resolve().parents[1] / "sql" / "features.sql"
    text = sql.read_text().upper()
    assert "LAG(" in text
    assert "OVER (" in text
    assert "ROWS BETWEEN 2 PRECEDING" in text
