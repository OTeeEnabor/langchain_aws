from pathlib import Path
import sys
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

import psycopg2

# append project directory - change to cwd
PROJECT_PATH = Path.cwd().parent
sys.path.append(str(PROJECT_PATH))

from config.config_ import load_config
from config.connect import connect

def create_tables():
    # load database configurations
    config = load_config()

    # connect to database
    conn = connect(config)

    # create tables lanchaineco database
    commands = (
        """
        CREATE TABLE IF NOT EXISTS economic_indicators (
                Date TEXT PRIMARY KEY,
                UNRATE REAL,
                PAYEMS REAL,
                ICSA REAL,
                CIVPART REAL,
                INDPRO REAL)
        """,
        """
        CREATE TABLE IF NOT EXISTS yield_curve_prices (
                Date TEXT PRIMARY KEY,
                DGS1MO REAL,
                DGS3MO REAL,
                DGS6MO REAL,
                DGS1 REAL,
                DGS2 REAL,
                DGS3 REAL,
                DGS5 REAL,
                DGS7 REAL,
                DGS10 REAL,
                DGS20 REAL,
                DGS30 REAL
            )
        """,
        """
            CREATE TABLE IF NOT EXISTS production_data (
                Date TEXT PRIMARY KEY,
                SAUNGDPMOMBD REAL,
                ARENGDPMOMBD REAL,
                IRNNGDPMOMBD REAL,
                SAUNXGO REAL,
                QATNGDPMOMBD REAL,
                KAZNGDPMOMBD REAL,
                IRQNXGO REAL,
                IRNNXGO REAL,
                KWTNGDPMOMBD REAL,
                IPN213111S REAL,
                PCU213111213111 REAL,
                DPCCRV1Q225SBEA REAL
            )
        """,
        """
            CREATE TABLE IF NOT EXISTS business_cycles (
                id SERIAL PRIMARY KEY,
                Peak_Month TEXT,
                Trough_Month TEXT,
                Start_Date TEXT,
                End_Date TEXT,
                Phase TEXT
            )
        """,
    )

    index_commands = (
        "CREATE INDEX IF NOT EXISTS idx_econ_date ON economic_indicators (Date)",
        "CREATE INDEX IF NOT EXISTS idx_yield_date ON yield_curve_prices (Date)",
        "CREATE INDEX IF NOT EXISTS idx_prod_date ON production_data (Date)",
    )

    with conn.cursor() as cur:
        for command in commands:
            cur.execute(command)
        for command in index_commands:
            cur.execute(command)

    # commit transaction
    conn.commit()
    # close connection
    conn.close()
    print("Tables created successfully")


def insert_business_cycle_data():
    # load database configurations
    config = load_config()

    # connect to database
    conn = connect(config)

    business_cycles = [
        {
            "peak": "1999-03-01",
            "trough": "2001-03-01",
            "start": "1999-03-01 00:00:00",
            "end": "2001-03-01 00:00:00",
            "phase": "Expansion",
        },
        {
            "peak": "2001-03-01",
            "trough": "2001-11-01",
            "start": "2001-03-01 00:00:00",
            "end": "2001-11-01 00:00:00",
            "phase": "Contraction",
        },
        {
            "peak": "2001-11-01",
            "trough": "2007-12-01",
            "start": "2001-11-01 00:00:00",
            "end": "2007-12-01 00:00:00",
            "phase": "Expansion",
        },
        {
            "peak": "2007-12-01",
            "trough": "2009-06-01",
            "start": "2007-12-01 00:00:00",
            "end": "2009-06-01 00:00:00",
            "phase": "Contraction",
        },
        {
            "peak": "2020-02-01",
            "trough": "2020-04-01",
            "start": "2009-06-01 00:00:00",
            "end": "2020-02-01 00:00:00",
            "phase": "Expansion",
        },
        {
            "peak": "2020-02-01",
            "trough": "2020-04-01",
            "start": "2020-02-01 00:00:00",
            "end": "2020-04-01 00:00:00",
            "phase": "Contraction",
        },
        {
            "peak": "2021-12-01",
            "trough": "2022-03-31",
            "start": "2020-04-01 00:00:00",
            "end": "2022-03-11 00:00:00",
            "phase": "Expansion",
        },
    ]
    insert_statement = """
                INSERT INTO business_cycles (Peak_Month, Trough_Month, Start_Date, End_Date, Phase)
                VALUES (%s, %s, %s, %s, %s)
            """
    with conn.cursor() as curr:
        for cycle in business_cycles:
            curr.execute(
                insert_statement,
                (
                    cycle["peak"],
                    cycle["trough"],
                    cycle["start"],
                    cycle["end"],
                    cycle["phase"],
                ),
            )
        print("Business cycle data inserted successfully.")
    # commit transaction
    conn.commit()
    # close connection
    conn.close()

if __name__ == "__main__":
    create_tables()
    insert_business_cycle_data()