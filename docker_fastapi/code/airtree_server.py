import json
import time
import datetime
from urllib.request import urlopen
from functools import lru_cache
import gzip
import dotenv
import streaming_form_data as sfd
import pymongo
import pydantic
from pydantic_settings import BaseSettings
from starlette.requests import ClientDisconnect
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request, HTTPException, status
from models import FromUser

MAX_FILE_SIZE = 1024 * 1024 * 1024 * 4  # = 4GB
MAX_REQUEST_BODY_SIZE = MAX_FILE_SIZE + 1024

app = FastAPI()


# ----------------------------------------------------------------------------
# .ENV DATA MANAGER

class AirWebData:
    def __init__(self):
        config         = dotenv.dotenv_values('.env')
        mongo_user     = config['mongo_user']
        mongo_password = config['mongo_pass']
        self._air_key  = config['AIRTREE_KEY']
        self._db_connection = (
            f"mongodb://{mongo_user}:{mongo_password}@dbmongo.airtree:27017/"
        )

    @property
    def air_key(self):
        return self._air_key

    @property
    def db_connection(self):
        return self._db_connection


# ----------------------------------------------------------------------------
# SETTINGS

class Settings(BaseSettings):
    db_connection: str
    air_key: str


@lru_cache
def get_settings():
    air = AirWebData()
    settings = Settings(
        db_connection = air.db_connection,
        air_key       = air.air_key
    )
    return settings


# ----------------------------------------------------------------------------
# MONGODB

class MongoDb:
    @staticmethod
    def get_database():
        settings = get_settings()
        connection_string = settings.db_connection
        client = pymongo.MongoClient(
            connection_string,
            serverSelectionTimeoutMS=300
        )
        mdb = client['airtree']
        return mdb

    @staticmethod
    def insert_data(collection, data):
        collection.insert_one(data)

    @staticmethod
    def insert_internal_log(msg: str):
        mdb = MongoDb.get_database()
        client = mdb['internal_log']
        data = {
            "time": datetime.datetime.now(tz=datetime.timezone.utc),
            "log":           msg
        }
        client.insert_one(data)

    @staticmethod
    def ping():
        settings = get_settings()
        connection_string = settings.db_connection
        client = pymongo.MongoClient(
            connection_string,
            serverSelectionTimeoutMS=300
        )
        try:
            client.admin.command('ping')
        except pymongo.errors.ServerSelectionTimeoutError:
            print("Server not available")


# ----------------------------------------------------------------------------
# UTILS

def get_time_now_posix():
    d = datetime.datetime.now()
    return int(time.mktime(d.timetuple()))


def stream_generator(data):
    size = len(data)
    chunk_size = 4096
    for i in range(0, size, chunk_size):
        yield data[i:i + chunk_size]


def get_altitude(lat, lon):
    'get altitude from opentopodata'
    # ------------------------------
    url = f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}"
    with urlopen(url) as response:
        data_json = json.loads(response.read())
    altitude = data_json['results'][0]['elevation']
    return altitude


# ----------------------------------------------------------------------------
class MaxBodySizeException(Exception):
    def __init__(self, body_len: str):
        self.body_len = body_len


class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(body_len=str(self.body_len))


def err_upload_project(exc: Exception):
    try:
        raise exc
    except ClientDisconnect:
        print("Client Disconnected")
    except MaxBodySizeException:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail='Maximum request body size limit'
        )
    # except sfd.validators.ValidationError:
    #     raise HTTPException(
    #         status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    #         detail=f'Maximum file size limit ({MAX_FILE_SIZE} bytes) exceeded'
    #     )
    except pydantic.ValidationError:
        raise HTTPException(
            status_code = 422,
            detail      = "Item not validated"
        )
    except Exception as err:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f'Error uploading the file: {err}'
        )


def check_api(api_key: str | None):
    settings = get_settings()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Airtree API key is missing'
        )
    if api_key != settings.air_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Airtree API key is wrong'
        )


def err_db(exc: Exception):
    try:
        raise exc
    except pymongo.errors.ServerSelectionTimeoutError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'error server db is down: {err}'
        )

    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'error server db: {err}'
        )


def db_upload_project(data3: dict):
    try:
        data3['id_project'] = data3['project']['id_project']
        data3['id_user'] = data3['project']['id_user']
        data3['project']['altitude'] = get_altitude(
            data3['project']['lat'],
            data3['project']['lon']
        )

        # delete any previous project data with same id
        mdb = MongoDb.get_database()
        collection2 = [
            mdb['project_status'],
            mdb['from_user'],
            mdb['to_user'],
            mdb['job'],
        ]
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
        air_status = {
            'id_user':         data3['id_user'],
            'id_project':      data3['id_project'],
            'private_project': data3['project']['private_project'],
            'post_time':       get_time_now_posix(),
            'work_status':     0  # 0: in queue, 1: processing, 2: done
        }
        MongoDb.insert_data(collection, air_status)
    except Exception as exc:
        err_db(exc)


@app.get('/get-settings')
def get_air_settings(request: Request):
    api_key = request.query_params.get('api_key')
    check_api(api_key)
    try:
        mdb        = MongoDb.get_database()
        collection = mdb['settings']

        result_a = collection.find_one({})
        result_a.pop('_id', None)

        if result_a:
            result = json.dumps(result_a)
            return result
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='No settings found'
        )
    except Exception as exc:
        err_db(exc)


@app.get('/get-work-status')
def get_work_status(request: Request):
    api_key = request.query_params.get('api_key')
    check_api(api_key)

    id_user = request.query_params.get('id_user')
    id_project = request.query_params.get('id_project')

    mdb        = MongoDb.get_database()
    collection = mdb['project_status']
    air_status = -1  # default error code
    try:
        result_a = collection.find_one({
            'id_user': id_user,
            'id_project': id_project,
        })
        if result_a:
            print('res_a ok')
            result_b = result_a['work_status']
            if result_b:
                print('res_b ok')
                air_status = result_b
    except Exception as exc:
        err_db(exc)
    return str(air_status)


@app.patch('/set-delivered')
async def set_delivered(request: Request):
    api_key = request.query_params.get('api_key')
    id_project = request.query_params.get('id_project')
    id_user = request.query_params.get('id_user')

    # CHECK API KEY
    check_api(api_key)

    # CHECK DATA
    if not id_project or not id_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Missing id_project or id_user'
        )

    # UPDATE DB
    try:
        mdb = MongoDb.get_database()
        collection = mdb['project_status']
        collection.update_one(
            {
                'id_user': id_user,
                'id_project': id_project,
            },
            {
                '$set':
                {'work_status': 5}
            }
        )
    except Exception as exc:
        err_db(exc)

    # RETURN MSG
    feedback = {"detail": "Successfuly updated"}
    return feedback


# TODO: check
@app.get('/get-result')
def get_result(request: Request):
    api_key = request.query_params.get('api_key')
    id_project = request.query_params.get('id_project')
    id_user = request.query_params.get('id_user')

    # CHECK API KEY
    check_api(api_key)

    # CHECK DATA
    if not id_project or not id_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Missing id_project or id_user'
        )

    # CHECK IF RESULTS ARE AVAILABLE
    mdb        = MongoDb.get_database()
    collection = mdb['to_user']
    air_status     = -1  # default error code
    result_a = collection.find_one({
        'id_user': id_user,
        'id_project': id_project,
    })
    if not result_a:
        return str(air_status)
    result_a.pop('_id', None)
    result_b = json.dumps(result_a)
    result_c = gzip.compress(result_b.encode('utf-8'))
    return StreamingResponse(
        stream_generator(result_c),
        media_type='application/gzip'
    )


@app.post('/post-project')
async def upload_project(request: Request):
    # CHECK API KEY
    api_key = request.query_params.get('api_key')
    check_api(api_key)

    # PARSE DATA
    body_validator = MaxBodySizeValidator(MAX_REQUEST_BODY_SIZE)
    target_file = sfd.targets.ValueTarget()
    try:
        parser = sfd.StreamingFormDataParser(headers=request.headers)
        parser.register('file', target_file)

        async for chunk in request.stream():
            body_validator(chunk)
            parser.data_received(chunk)

    except Exception as exc:
        print(exc)
        err_upload_project(exc)

    print("File received")

    data_memfile = gzip.decompress(target_file.value).decode()
    try:
        prj = FromUser.parse_raw(data_memfile)
    except Exception as exc:
        print('--------------------')
        print('data_memfile:', data_memfile)
        print('--------------------')
        err_upload_project(exc)

    # STORE DATA
    db_upload_project(prj.dict())

    # RETURN MSG
    feedback = {"message": "Successfuly uploaded"}
    return feedback


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
