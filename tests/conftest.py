import sqlite3

import pytest
from fastapi.testclient import TestClient

try:
    import duckdb
except ImportError:
    duckdb = None

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_duckdb(tmp_path):
    if duckdb is not None:
        db_path = tmp_path / "test.duckdb"
        con = duckdb.connect(str(db_path))

        # Create mock schema and data
        con.execute("CREATE TABLE ports (port_id VARCHAR, port_name VARCHAR, country_code VARCHAR)")
        con.execute("INSERT INTO ports VALUES ('PORT_TEST', 'Test Port', 'TC')")

        con.execute(
            "CREATE TABLE port_activity (port_id VARCHAR, date DATE, daily_port_calls INT, global_chokepoint_transit INT, active_disasters_7d INT)"
        )
        con.execute("INSERT INTO port_activity VALUES ('PORT_TEST', '2024-01-01', 100, 50, 0)")

        yield str(db_path)
        con.close()
    else:
        db_path = tmp_path / "test.db"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE ports (port_id TEXT, port_name TEXT, country_code TEXT)")
        con.execute("INSERT INTO ports VALUES ('PORT_TEST', 'Test Port', 'TC')")

        con.execute(
            "CREATE TABLE port_activity (port_id TEXT, date TEXT, daily_port_calls INT, global_chokepoint_transit INT, active_disasters_7d INT)"
        )
        con.execute("INSERT INTO port_activity VALUES ('PORT_TEST', '2024-01-01', 100, 50, 0)")
        con.commit()

        yield str(db_path)
        con.close()
