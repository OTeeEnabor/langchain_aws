import os
from pathlib import Path
import pprint
import sys
from dotenv import load_dotenv

PROJECT_PATH = Path.cwd()#.parent
print(PROJECT_PATH)

sys.path.append(str(PROJECT_PATH))

from config.config_ import load_config
from config.connect import connect

load_dotenv()

def test_db_connection():
    config = load_config()
    # connect to database
    try:
        conn = connect(config)
        print(conn)
    except Exception as e:
        print(f"Following error occurred - {e}")

    else:
        if conn.is_connected():
            conn.close()
            print("Connection closed")


test_db_connection()