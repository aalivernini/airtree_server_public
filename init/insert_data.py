'init airtree database'

import dotenv
from pymongo import MongoClient
import pandas as pd
import os

class DD(dict):  # @@2
    """dot.notation access to dictionary attributes (pickable) """
    __getattr__ = dict.__getitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __setattr__(self, name, value): self.__setitem__(name, value)
    def __delattr__(self, name):self.__delitem__(name)
    def __init__(self, *args, **kwargs ): dict.__init__(self, *args, **kwargs )


class MongoDb:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def get_database(self, db_name='airtree'):
        client = MongoClient(self.connection_string)
        mdb = client[db_name]
        return mdb

    def insert_data(self, collection, data):
        collection.insert_one(data)

    def insert_param_db(self, csv_path):
        mdb = self.get_database()
        collection = mdb['param_db']
        d1 = get_dictionary(csv_path)
        collection.insert_many(d1)

    def insert_param_cst(self, csv_path):
        mdb = self.get_database()
        collection = mdb['param_cst']
        d1 = get_dictionary(csv_path)
        d1 = [{k: v for k, v in d0.items() if v == v} for d0 in d1]
        d3 = dict()
        for d0 in d1:
            d3[d0['parameter']] =  d0['value']
        collection.insert_one(d3)

    def insert_atm_coords(self, csv_path):
        mdb = self.get_database()
        collection = mdb['atm_coords']
        d1 = get_dictionary(csv_path)
        collection.insert_many(d1)

    def delete_atm(self, posix_time: int):
        ''' delete all atm data with time > posix_time '''
        mdb = self.get_database('atm')
        # list all collections
        collections = mdb.list_collection_names()
        for c1 in collections:
            collection = mdb[c1]
            query = {'time': {'$gte': posix_time}}
            collection.delete_many(query)

    def insert_atm(self, atm_csv_dir):
        mdb = self.get_database('atm')

        # get csv paths in input directory
        for fname in os.listdir(atm_csv_dir):
            if fname.endswith('.csv'):
                # create subcollection
                name = fname[:-4]
                collection = mdb[name]

                # get dictionary list from csv
                path = os.path.join(atm_csv_dir, fname)
                d1 = get_dictionary(path)
                collection.insert_many(d1)

    def define_settings(self):
        mdb = self.get_database('atm')
        atm1 = list(mdb['558059'].find({}))
        if not atm1:
            raise Exception('No atm data found')
        df = pd.DataFrame.from_dict(atm1)
        df = df.sort_values(by=['time'])
        time_max = int(df['time'].max())
        time_min = int(df['time'].min())
        mdb = self.get_database('airtree')
        mdb['settings'].drop()
        collection = mdb['settings']
        collection.insert_one(
            {'atm_time_start': time_min,
             'atm_time_end': time_max,
             'version': '1'}
        )

    def reset_db(self):
        'drop existing tables'
        db_air = self.get_database('airtree')
        for collection in ['param_db', 'param_cst', 'atm_coords', 'settings']:
            db_air[collection].drop()
        db_atm = self.get_database('atm')
        id_atm2 = db_atm.list_collection_names()
        for collection in id_atm2:
            db_atm[collection].drop()



class Connection:
    def __init__(self, config: dict):
        self.config = config

    def init_db(self):
        config = self.config
        dir0 = config['data_dir']

        # define paths
        p_param_db = os.path.join(dir0, 'param_db.csv')
        p_param_cst = os.path.join(dir0, 'param_cst.csv')
        p_coords = os.path.join(dir0, 'atm_coords.csv')
        p_atm = os.path.join(dir0, 'atm_csv')
        if not os.path.exists(p_atm):
            print(f'path {p_atm} does not exist')

        user = config['mongo_user']
        passw = config['mongo_pass']
        connection_string = f"mongodb://{user}:{passw}@127.0.0.1:27017/"

        mdbc = MongoDb(connection_string)

        # drop existing input tables
        mdbc.reset_db()

        # insert param_db
        mdbc.insert_param_db(p_param_db)

        # # insert param_cst
        mdbc.insert_param_cst(p_param_cst)

        # insert atm coords
        mdbc.insert_atm_coords(p_coords)

        # insert atm data
        mdbc.insert_atm(p_atm)

        # init settings
        mdbc.define_settings()


def get_dictionary(path):
    df = pd.read_csv(path)
    dictionary = df.to_dict('records')
    return dictionary


def main(config: dict):
    # GET CONNECTION
    conn = Connection(config)
    conn.init_db()

    # INSERT ATM DATA
    print('atm data are inserted')


if __name__ == "__main__":
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    config = dotenv.dotenv_values(env_path)
    main(config)
