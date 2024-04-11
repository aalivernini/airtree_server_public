import flask
from flask import Flask, request, jsonify, stream_with_context
import gzip
import json
from pymongo import MongoClient
from urllib.request import urlopen
from asgiref.wsgi import WsgiToAsgi
import time, datetime
import dotenv


# ----------------------------------------------------------------------------
# .env data manager
class AirWebData:
    def __init__(self):
        config         = dotenv.dotenv_values('.env')
        mongo_user     = config['mongo_user']
        mongo_password = config['mongo_pass']
        self._air_key  = config['AIRTREE_KEY']
        self._db_connection = f"mongodb://{mongo_user}:{mongo_password}@dbmongo.airtree:27017/"

    @property
    def air_key(self):
        return self._air_key

    @property
    def db_connection(self):
        return self._db_connection

air = AirWebData()

# ----------------------------------------------------------------------------
VERSION = 3
app = Flask(__name__)
app.config['db_connection'] = air.db_connection
app.config['air_key'] = air.air_key

# ----------------------------------------------------------------------------
# MONGODB
class MongoDb:
    @staticmethod
    def get_database():
        connection_string = app.config['db_connection']
        client = MongoClient(connection_string)
        mdb = client['airtree']
        return mdb

    @staticmethod
    def insert_data(collection, data):
        collection.insert_one(data)
# ----------------------------------------------------------------------------


logger = None



def get_time_now_posix():
    d = datetime.datetime.now()
    return int(time.mktime(d.timetuple()))

def get_altitude(lat, lon):
    'get altitude from opentopodata'
    url = "https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}".format(lat=lat, lon=lon)
    response = urlopen(url)
    data_json = json.loads(response.read())
    altitude = data_json['results'][0]['elevation']
    return altitude

@app.route('/ping', methods=['GET'])
def ping():
    print(app.config['air_key'], flush=True)
    result = {
            'api_version': VERSION,
            'service': 'ping',
            'isOK': True,
            'error': {
                'coderr': 0,
                'deserr': ""
                }
            }
    return jsonify(result)


@app.route('/set_delivered', methods=['PATCH'])
def set_delivered():
    api_key = request.args.get('api_key', type = str)
    if api_key != app.config['air_key']:
        return "wrong api key"
    id_project = request.args.get('id_project', type = str)
    id_user = request.args.get('id_user', type = str)
    mdb        = MongoDb.get_database()
    collection = mdb['project_status']
    collection.update_one({
        'id_user': id_user,
        'id_project': id_project,
        }, {'$set': {'work_status': 5}})
    result = {
            'isOK': True,
            'error': {
                'coderr': 0,
                'deserr': ""
                }
            }
    print("update_project_status")
    return jsonify(result)



@app.route('/get_settings', methods=['GET'])
def get_settings():
    api_key = request.args.get('api_key', type = str)
    if api_key != app.config['air_key']:
        return "wrong api key"
    mdb        = MongoDb.get_database()
    collection = mdb['settings']

    result_a = collection.find_one({})
    result_a.pop('_id', None)

    if result_a:
        result = json.dumps(result_a)
        return result


# todo: implement user id
@app.route('/get_work_status', methods=['GET'])
def get_work_status():
    api_key = request.args.get('api_key', type = str)
    if api_key != app.config['air_key']:
        return "-1"
    id_project = request.args.get('id_project', type = str)
    mdb        = MongoDb.get_database()
    collection = mdb['project_status']
    status     = -1  # default error code
    try:
        result_a = collection.find_one({'id_project': id_project})
        if result_a:
            print('res_a ok')
            result_b = result_a['work_status']
            if result_b:
                print('res_b ok')
                status = result_b
    except:
        pass
    return str(status)



def stream_generator(data):
    size = len(data)
    chunk_size = 4096
    for i in range(0, size, chunk_size):
        yield data[i:i+chunk_size]


# todo: implement user id
@app.route('/get_result', methods=['GET'])
def get_result():
    print("getting_result")
    api_key = request.args.get('api_key', type = str)
    if api_key != app.config['air_key']:
        return "wrong api key"
    id_user = request.args.get('id_user', type = str)
    id_project = request.args.get('id_project', type = str)

    # check if results are available
    mdb        = MongoDb.get_database()
    collection = mdb['to_user']
    status     = -1  # default error code
    result_a = collection.find_one({
        'id_user': id_user,
        'id_project': id_project,
    })
    if not result_a:
        return str(status)
    result_a.pop('_id', None)
    result_b = json.dumps(result_a)
    result_c = gzip.compress(result_b.encode('utf-8'))
    print("responding with results")
    return app.response_class(stream_generator(result_c), mimetype='application/gzip')


# TODO
# rapid conformity check of input before inserting in db
@app.route('/post_project', methods=['POST'])
def post_project():
    api_key = request.args.get('api_key', type = str)
    if not api_key:
        return "no api key"
    if api_key != app.config['air_key']:
        return "wrong api key"
    data = bytes()
    chunk_size = 4096
    while True:
        chunk = request.input_stream.read(chunk_size)
        data += chunk
        if len(chunk) == 0:
            break
    try:
        data3 = json.loads(gzip.decompress(data))
        data3['id_project'] = data3['project']['id_project']
        data3['id_user'] = data3['project']['id_user']
        data3['project']['altitude'] = get_altitude(data3['project']['lat'], data3['project']['lon'])

        # delete any previous project data with same id
        mdb = MongoDb.get_database()
        collection2 = [mdb['project_status'], mdb['from_user'], mdb['to_user'], mdb['job']]
        for collection in collection2:
            collection.delete_many({
                'id_user': data3['id_user'],
                'id_project': data3['id_project'],
            })

        # insert project data
        collection = mdb['from_user']
        MongoDb.insert_data(collection, data3)

        # insert project status (used by workers and for app update)
        collection = mdb['project_status']
        status = {
                'id_user'  : data3['id_user'],
                'id_project'  : data3['id_project'],
                'private_project' : data3['project']['private_project'],
                'post_time' :  get_time_now_posix(),
                'work_status' : 0  # 0 : in queue, 1 : processing, 2 : done
                }
        MongoDb.insert_data(collection, status)
        return "upload ok"

    except Exception as e:
        return "upload error: " + str(e)

wsgi = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

