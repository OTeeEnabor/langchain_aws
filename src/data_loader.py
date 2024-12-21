from pathlib import Path
import sys
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

import psycopg2

# append project directory - change to cwd
PROJECT_PATH = Path.cwd()
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
class DataLoader:
     # load database configurations
    config = load_config()

    # connect to database
    conn = connect(config)
    
    def __init__(self, db_name="langchainecodata"):
        self.db_name = db_name
        self.economic_indicators_tickers = [
            "UNRATE",
            "PAYEMS",
            "ICSA",
            "CIVPART",
            "INDPRO",
        ]
        self.yield_curve_tickers = [
            "DGS1MO",
            "DGS3MO",
            "DGS6MO",
            "DGS1",
            "DGS2",
            "DGS3",
            "DGS5",
            "DGS7",
            "DGS10",
            "DGS20",
            "DGS30",
        ]
        self.production_data_tickers = [
            "SAUNGDPMOMBD",
            "ARENGDPMOMBD",
            "IRNNGDPMOMBD",
            "SAUNXGO",
            "DPCCRV1Q225SBEA",
            "QATNGDPMOMBD",
            "KAZNGDPMOMBD",
            "IRQNXGO",
            "IRNNXGO",
            "KWTNGDPMOMBD",
            "IPN213111S",
            "PCU213111213111",
        ]
    
    def clean_data(self, data):
        """Function to clean data, remove leading/trailing single quotes, and convert to numeric"""
        cleaned_data = data.map(lambda x: x.strip("'") if isinstance(x, str) else x)
        cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce")
        return cleaned_data
    def fetch_and_insert_data(self, tickers, table_name):
        start_date = "2000-12-31"
        end_date = datetime.now().strftime("%Y-%m-%d")
        try:
            # fetch data for the tickers between the specified date
            data = web.DataReader(tickers, "fred", start_date, end_date)
            # perform quadratic interpolation and perfrom backward and forward fill respectively
            data = data.interpolate(method="quadratic").bfill().ffill()
            # resample the data frame to daily and perfrom forward and backward fill
            data = data.resample("D").ffill().bfill()

            # clean the data to handle format issues
            data = self.clean_data(data)
            # convert the index (date) to the correct format
            data.index = pd.to_datetime(data.index).strftime("%Y-%m-%d %H:%M:%S")
            
            # save the file to csv
            temp_df = Path.cwd()/"data"/"tmp"/"tmp_df.csv"#../data/tmp/tmp_df.csv"
            # save data to csv
            data.to_csv(temp_df, index_label="Date", header=False)
            # open the file
            f = open(temp_df, "r")
            with self.conn.cursor() as curr:
                curr.copy_from(f, table_name, sep=",")
                self.conn.commit()   
            print(f"Data inserted into {table_name} table")

        except Exception as error:
            print(f"Failed to fetch and insert the data: {error}")






def main():
    create_tables()
    insert_business_cycle_data()
    #
    loader = DataLoader()
    table_names = ["yield_curve_prices","production_data"]#["economic_indicators","yield_curve_prices","production_data"]
    tickers = [loader.yield_curve_tickers, loader.production_data_tickers]#[loader.economic_indicators_tickers, loader.yield_curve_tickers, loader.production_data_tickers]
    for i,table in enumerate(table_names):
        loader.fetch_and_insert_data(tickers[i], table)
    loader.conn.close()
if __name__ == "__main__":
    main()