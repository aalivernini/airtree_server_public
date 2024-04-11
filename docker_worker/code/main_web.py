import numpy.typing  # keep
from pymongo import MongoClient
import time
import polars as pl
import haversine
import bson
from airtree import Airtree
import numpy as np
import time, datetime
import dotenv


CONFIG = dotenv.dotenv_values('.env')
MONGO_USER = CONFIG['mongo_user']
MONGO_PASSWORD = CONFIG['mongo_pass']

PI = 3.141592

def get_time_now_posix():
    d = datetime.datetime.now()
    return int(time.mktime(d.timetuple()))


class MongoDb:
    @staticmethod
    def get_database(db_name='airtree'):
        connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@dbmongo.airtree:27017/"
        client = MongoClient(connection_string)
        mdb = client[db_name]
        return mdb

    @staticmethod
    def insert_data(collection, data):
        collection.insert_one(data)

    @staticmethod
    def get_project(id_user, id_project):
        mdb        = MongoDb.get_database()
        collection = mdb['from_user']
        project    = collection.find_one({
            'id_user': id_user,
            'id_project': id_project,
        })
        if not project:
            raise Exception('Project not found')
        return project

    @staticmethod
    def reset_data():
        mdb = MongoDb.get_database()
        mdb['from_user'].drop()
        mdb['job'].drop()
        mdb['project_status'].drop()
        mdb['to_user'].drop()


class Atm:
    @staticmethod
    def get_id(project):
        mdb = MongoDb.get_database()
        # get coordinates of stations
        collection = mdb['atm_coords'].find({})
        df_station = pl.DataFrame(list(collection))
        df_station = df_station.drop('_id')
        # compute the distance between the project and each station
        coord1 = (project['lat'], project['lon'])
        distance = df_station.map_rows(
            lambda x: haversine.haversine((x[1], x[2],), coord1)
        )
        df_station = df_station.with_columns(
            dist = distance.to_series()
        )
        # select the closest station
        station = df_station.filter(pl.col('dist') == pl.col('dist').min())
        station = str(station['fid'][0])
        return station


    @staticmethod
    def get_data(atm_id, project):
        mdb = MongoDb.get_database('atm')
        collection = mdb[atm_id].find({'time': {'$gte': project['start_date'], '$lt': project['end_date']}})
        df = pl.DataFrame(list(collection))
        return df.sort('time')


class Job:
    @staticmethod
    def add_many(jobs):
        mdb = MongoDb.get_database()
        collection = mdb['job']
        collection.insert_many(jobs)

    @staticmethod
    def get_one(status=0):
        mdb = MongoDb.get_database()
        collection = mdb['job']
        jobs = collection.find({'status' : status}).sort([( '$natural', -1 )]).limit(1) # sorted from most recently inserted document to oldest
        jobs = [x for x in jobs]
        if not jobs:
            return
        job = jobs[0]
        return job

    @staticmethod
    def get_many(query_dict):
        mdb = MongoDb.get_database()
        collection = mdb['job']
        jobs = collection.find(query_dict)
        jobs = [x for x in jobs]
        return jobs

    @staticmethod
    def set(_id, status=None, result=None):
        if not status and not result:
            raise Exception('No data to update')
        mdb = MongoDb.get_database()
        collection = mdb['job']
        set3 = {}
        if status:
            set3.update({'status': status})
        if result:
            set3.update({'result': result})
        collection.update_one({'_id': _id}, {'$set': set3})


# project status
# 0 project from user input
# 1 lock
# 2 data for airtree are ready
# 3 lock
# 4 project results are ready
# 5? project results are sent to user
class ProjectStatus:
    @staticmethod
    def get(query_dict):
        mdb = MongoDb.get_database()
        collection = mdb['project_status']
        jobs = collection.find(query_dict).sort([( '$natural', -1 )] ).limit(1) # sorted from most recently inserted document to oldest
        jobs = [x for x in jobs]
        if jobs:
            return jobs[0]

    @staticmethod
    def get_one(query_dict):
        mdb = MongoDb.get_database()
        collection = mdb['project_status']
        job = collection.find_one(query_dict)
        if job:
            return job

    @staticmethod
    def set(id_user, id_project, status, detailed_work_status=None):
        mdb = MongoDb.get_database()
        collection = mdb['project_status']
        if detailed_work_status:
            collection.update_one({
                'id_user': id_user,
                'id_project': id_project,
            }, {'$set': {'work_status': status, 'detailed_work_status': detailed_work_status}})
        else:
            collection.update_one({
                'id_user': id_user,
                'id_project': id_project,
            }, {'$set': {'work_status': status}})


class AirtreeManager:
    @staticmethod
    def reduce_runs(project):
        # airtree runs are performed on a reduced number of inputs
        # user data are aggregated by diameter classes and by species
        # aggregated data are reduced by mean values

        # merge all user green data in one list
        user_data = []
        user_data.extend(project['point'])
        user_data.extend(project['line'])
        user_data.extend(project['polygon_data'])

        # keep only the fields of interest
        fields = ['id', 'id_species', 'diameter', 'height', 'crown_height', 'crown_diameter', 'lai']
        shared_data = [{key: x[key] for key in fields} for x in user_data]
        shared_df = pl.DataFrame(shared_data)

        # create the column diameter_class with a step of 10 cm
        shared_df = shared_df.with_columns(
                diameter_class = shared_df.map_rows(lambda x: int((x[2]+5)/10)*10).to_series()
                )

        # group by id_species and diameter_class and compute the mean values,
        # the id string is converted to a list of strings
        # diameters <5 cm are grouped in the class 0
        grouped_df = shared_df.group_by([shared_df['id_species'], shared_df['diameter_class']])
        grouped_df = grouped_df.agg([
            pl.col('id').explode().alias('id'),
            pl.col('diameter').mean().alias('diameter'),
            pl.col('height').mean().alias('height'),
            pl.col('crown_height').mean().alias('crown_height'),
            pl.col('crown_diameter').mean().alias('crown_diameter'),
            pl.col('lai').mean().alias('lai'),
        ])
        result = grouped_df.to_dicts()
        return result

    @staticmethod
    def cleanup_project(id_user, id_project):
        mdb = MongoDb.get_database()
        table_to_clean = ['from_user', 'to_user', 'project_status', 'job']
        myquery = {
            "id_project": id_project,
            "id_user": id_user,
        }
        for tb1 in table_to_clean:
            collection = mdb[tb1]
            collection.delete_many(myquery)
        mdb.close()


    @staticmethod
    def cleanup_results():
        print('cleanup_results')
        mdb = MongoDb.get_database()

        # clean up private projects that are downloaded
        status2 = mdb['project_status'].find({'work_status': 5, 'private_project': 1})
        for st1 in status2:
            AirtreeManager.cleanup_project(st1['id_user'], st1['id_project'])

        # clean up private projects older than 1 week
        time_lapse_max = 7*24*3600 # 1 week
        status2 = mdb['project_status'].find({'private_project': 1})
        for st1 in status2:
            time_max = st1['post_time'] + time_lapse_max
            time_now = get_time_now_posix()
            if time_now > time_max:
                AirtreeManager.cleanup_project(st1['id_user'], st1['id_project'])

        # clean up completed jobs
        status2 = mdb['project_status'].find({'work_status': {'$gte': 4}})
        for st1 in status2:
            print(st1)
            collection = mdb['job']
            myquery = {
                "id_user": st1['id_user'],
                "id_project": st1['id_project'],
            }
            collection.delete_many(myquery)

    @staticmethod
    def get_canopy2(reduced_data):
        canopy2 = []
        mdb          = MongoDb.get_database()
        id_job       = 0

        # !@ develop here
        param_cst = mdb['param_cst'].find_one()
        if not param_cst:
            raise ValueError('Canopy constants not found')
        param_cst = dict(param_cst)


        for r1 in reduced_data:
            id_job += 1
            param1 = mdb['param_db'].find_one({'index': r1['id_species']})
            if param1:
                par1 = dict(param1)
            else:
                print('Species not found')
                continue
            par1.pop('_id')
            par1.update(param_cst)   # add canopy constants
            par1.update({
                'dbh': r1['diameter'],
                'height': r1['height'],
                'lai': r1['lai'],
                'crownwidth': r1['crown_diameter'],
                })
            par1['depth'] = r1['height'] - r1['crown_height']

            canopy2.append({
                'id'      : r1['id'],
                'canopy'  : par1,
                })
        return canopy2



    @staticmethod
    def get_soil_texture(id_soil):
        # return [%sand, %silt, %clay]
        # 1: "mean",
        # 2: "sandy",
        # 3: "silty",
        # 4: "clay",
        match id_soil:
            case 1:
                return [30, 30, 40]
            case 2:
                return [60, 20, 20]
            case 3:
                return [20, 60, 20]
            case 4:
                return [20, 20, 60]
            case _:
                return [30, 30, 40]



    @staticmethod
    def run_airtree():
        job = Job.get_one()
        if not job:
            return 1 # no jobs
        Job.set(job['_id'], 1)         # lock work before processing
        atm     = Atm.get_data(job['atm_id'], job['project'])
        id_user = job['id_user']
        project = job['project']
        canopy  = job['canopy']['canopy']
        project['mode'] = 'web'
        airtree = Airtree(project, canopy, atm)
        result  = airtree.airtree()

        # TODO manage errors

        # save results and update work status
        Job.set(job['_id'], 2, result) # set work status to done

        # update detailed_work_status
        query3    = {
            'id_user': job['id_user'],
            'id_project': job['id_project'],
        }
        prjStatus = ProjectStatus.get(query3)
        # TODO Error here when uploading the project after the first time
        if not prjStatus:
            return 1
        detailed_work_status  = prjStatus['detailed_work_status']
        job_id = str(job['_id'])
        detailed_work_status[job_id] = 1
        id_project = job['id_project']
        ProjectStatus.set(id_user, id_project, 2, detailed_work_status) # set project status to done


    @staticmethod
    def pre_process():
        query3 = {'work_status': 0}
        pst = ProjectStatus.get(query3)
        if not pst:
            return 1
        id_project = pst['id_project']
        id_user = pst['id_user']
        ProjectStatus.set(id_user, id_project, 1) # lock project preprocessing
        user_project     = MongoDb.get_project(id_user, id_project)
        airtree_project  = user_project['project']
        sand, silt, clay = AirtreeManager.get_soil_texture(airtree_project['id_soil_texture'])
        airtree_project.update({
            'sand' : sand,
            'silt' : silt,
            'clay' : clay,
            })


        reduced_data = AirtreeManager.reduce_runs(user_project)
        atm_id = Atm.get_id(airtree_project)

        canopy2 = AirtreeManager.get_canopy2(reduced_data)
        job2 = []
        job_id2 = []
        for canopy1 in canopy2:
            _id = bson.ObjectId()
            job_id2.append(_id)
            job1 = {
                    '_id'        : _id,
                    'id_user'    : id_user,
                    'id_project' : id_project,
                    'project'    : airtree_project,
                    'canopy'     : canopy1,
                    'atm_id'     : atm_id,
                    'status'     : 0,
                    }
            job2.append(job1)
        Job.add_many(job2)
        detailed_work_status = {}
        for _id in job_id2:
            detailed_work_status[str(_id)] = 0
        ProjectStatus.set(id_user, id_project, 2, detailed_work_status) # the project is ready for airtree runs
        return 0

    @staticmethod
    def get_project_df(id_user, id_project):
        project  = MongoDb.get_project(id_user, id_project)
        data2 = []
        pt2 = project['point']
        pt2 = [dict(pt, **{'type': 0}) for pt in pt2]
        line2 = project['line']
        line2 = [dict(line, **{'type': 1}) for line in line2]
        polygon2 = project['polygon_data']
        polygon2 = [dict(polygon, **{'type': 2}) for polygon in polygon2]
        data2.extend(pt2)
        data2.extend(line2)
        data2.extend(polygon2)
        df = pl.DataFrame(data2)
        # print('get_project_df')
        # print(df)
        return df



    @staticmethod
    def build_results():
        query3 = {'work_status': 2}
        pst = ProjectStatus.get(query3)  # return only one
        if not pst:
            return 1 # no jobs
        work_status = pst['detailed_work_status']
        for key, value in work_status.items():
            if value == 0:
                return 1 # jobs are not ready
        id_project = pst['id_project']
        id_user = pst['id_user']

        # all jobs are ready
        ProjectStatus.set(id_user, id_project, 3) # lock project while building results

        # retrieve user data
        project_df = AirtreeManager.get_project_df(id_user, id_project)
        result2 = []
        job2 = Job.get_many({
            'id_project': id_project,
            'id_user': id_user,
        })
        for job1 in job2:
            for id_data in job1['canopy']['id']:
                row = project_df.filter(project_df['id'] == id_data).to_dicts()[0]
                out1 = job1['result']
                out1['id'] = id_data
                out1['id_species'] = row['id_species']

                match row['type']:
                    case 0: # point
                        canopy_area = row['crown_diameter']**2 * (PI / 4)
                    case 1: # line
                        canopy_area = row['crown_diameter']**2 * (PI / 4) * row['tree_number']
                    case 2: # polygon
                        canopy_area = row['area'] * (row['percent_area'] / 100) * (row['percent_cover'] / 100)
                    case _:
                        raise ValueError('unknown type')

                for key, value in out1['total'].items():
                    out1['total'][key] = value * canopy_area

                for key, value in out1['time_series'].items():
                    if key == 'time':
                        continue
                    np_value = np.array(value)
                    out1['time_series'][key] = (np_value * canopy_area).tolist()
                    out1['canopy_area'] = canopy_area

                result2.append(out1)

        out = {
                'id_project' : id_project,
                'id_user'    : job2[0]['project']['id_user'],
                'data'       : result2,
                'last_update': get_time_now_posix(),
                'start_date' : job2[0]['project']['start_date'],
                'end_date'   : job2[0]['project']['end_date'],
                }
        mdb        = MongoDb.get_database()
        collection = mdb['to_user']
        collection.insert_one(out)
        ProjectStatus.set(id_user, id_project, 4) # results are ready to download
        # ProjectStatus.set(id_user, id_project, 5) # is set by user after downloading the results

        # TODO: update settings with total of points, lines and polygons
        return 0



def main():
    cleanup_lapse = 60 * 30  # [seconds] half an hour
    cleanup_time = get_time_now_posix() + cleanup_lapse
    while True:
        # 0) periodically cleanup the database
        if get_time_now_posix() > cleanup_time:
            cleanup_time = get_time_now_posix() + cleanup_lapse
            AirtreeManager.cleanup_results()


        # 1) check for project with work_status == 3; set work_status = 4 (lock)
        # ... produce results and set work_status = 4 (results ready)
        check = AirtreeManager.build_results()
        if not check:
            continue

        # 2) check for project with work_status == 2; if job has job_status 0; set job_status = 1 (lock);
        # ... do airtree run; set job_status = 2 (airtree result ready);
        # ... if all jobs have job_status ==2: set work status =3

        check = AirtreeManager.run_airtree()
        if not check:
            continue


        # 3) check for project with work_status == 0; set work_status to 1 (lock); prepare data for airtree runs; set work_status to 2
        check = AirtreeManager.pre_process()
        if not check:
            continue

        # sleep if no work
        print('ralaxing 2 sec...')
        time.sleep(2)


if __name__ == "__main__":
    main()





