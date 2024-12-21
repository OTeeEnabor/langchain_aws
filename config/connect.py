import psycopg2

from .config_ import load_config

def connect(config):
    """
    Connect to Postgresql server

    :param config(dict): dictionary containing information to connect to PostgreSQL server

    returns
    conn (psycopg2 connection object)
    
    """
    try:
        # connect to the server
        with psycopg2.connect(**config) as conn:
            # print connected to server
            print("Connected to PostgreSQL server")
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

if __name__ == "__main__":
    config = load_config()
    connect(config)