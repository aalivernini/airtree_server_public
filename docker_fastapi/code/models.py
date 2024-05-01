from pydantic import BaseModel  # pylint: disable = no-name-in-module




class AirResponse(BaseModel):
    msg: str

class Project(BaseModel):
    id_project:      str
    id_user:         str
    name:            str
    location:        str
    description:     str
    last_update:     int
    start_date:      int
    end_date:        int
    private_project: int
    irrigation:      int
    status:          int
    lat:             float
    lon:             float
    id_soil_texture: int
    has_data:        int
    id_int:          int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id_project": "660d6d28ce3c6d4a18dad374",
                    "id_user": "660cf45b4b2929754cebc36c",
                    "name": "Worshop Tecnoverde",
                    "location": "Roma",
                    "description": "Azienda Crea",
                    "last_update": 1713090888,
                    "start_date": 1640995200,
                    "end_date": 1672531200,
                    "private_project": 0,
                    "irrigation": 0,
                    "status": 3,
                    "lat": 41.90915217341299,
                    "lon": 12.360916990109203,
                    "id_soil_texture": 1,
                    "has_data": 1,
                    "id_int": 1,
                },
            ]
        }
    }

class Green(BaseModel):
    id:             str
    id_project:     str
    id_user:        str
    last_update:    int
    id_species:     int
    diameter:       int
    height:         float
    crown_height:   float
    crown_diameter: float
    lai:            float
    truth:          int
    id_int:         int



class Point(Green):
    lat:            float
    lon:            float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "6Item60d6d40ce3c6d4a18dad375",
                    "id_project": "660d6d28ce3c6d4a18dad374",
                    "id_user": "660cf45b4b2929754cebc36c",
                    "lat": 41.91124799107433,
                    "lon": 12.364137650571926,
                    "last_update": 1712993330,
                    "id_species": 366,
                    "diameter": 51,
                    "height": 16.1,
                    "crown_height": 9,
                    "crown_diameter": 9,
                    "lai": 4.2,
                    "truth": 1,
                    "id_int": 1
                },
            ]
        }
    }


class Line(Green):
    tree_number: int
    length: float
    json_geometry: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "660d6da3ce3c6d4a18dad376",
                    "id_project": "660d6d28ce3c6d4a18dad374",
                    "id_user": "660cf45b4b2929754cebc36c",
                    "json_geometry": "{\"type\":\"LineString\",\"coordinates\":[[12.363942175505994,41.91112848814507],[12.363740610190355,41.91115810252318],[12.363449257674839,41.91123017993781]]}",
                    "last_update": 1712981864,
                    "id_species": 366,
                    "diameter": 51,
                    "height": 16,
                    "crown_height": 9,
                    "crown_diameter": 10,
                    "lai": 4.2,
                    "truth": 1,
                    "tree_number": 8,
                    "length": 42.40782035627868,
                    "id_int": 1
                },
            ]
        }
    }


class PolygonGeometry(BaseModel):
    id:             str
    id_project:     str
    id_user:        str
    json_geometry:  str
    last_update:    int
    truth:          int
    id_int:         int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "660e86c480d4fc1aef937667",
                    "id_project": "660d6d28ce3c6d4a18dad374",
                    "id_user": "660cf45b4b2929754cebc36c",
                    "json_geometry": "{\"type\":\"Polygon\",\"coordinates\":[[12.359872888505391,41.90830838269898],[12.360001027651803,41.90824996543629],[12.359718989160722,41.90800982284815],[12.359512538948715,41.908145719090875],[12.359618230213977,41.90821358129054],[12.35971880151742,41.90816685824359]]}",
                    "last_update": 1712228243,
                    "truth": 1,
                    "id_int": 4
                },
            ]
        }
    }


class PolygonData(Green):
    id_geometry:   str
    percent_area:  int
    percent_cover: int
    area:          float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "660d6e1dce3c6d4a18dad378",
                    "id_project": "660d6d28ce3c6d4a18dad374",
                    "id_user": "660cf45b4b2929754cebc36c",
                    "id_geometry": "660d6de3ce3c6d4a18dad377",
                    "last_update": 1712156189,
                    "id_species": 91,
                    "diameter": 25,
                    "height": 16,
                    "crown_height": 3,
                    "crown_diameter": 5,
                    "lai": 3,
                    "truth": 1,
                    "percent_area": 100,
                    "percent_cover": 50,
                    "area": 2736.9019689032225,
                    "id_int": 1
                },
            ]
        }
    }


class UploadData(BaseModel):
    api_key: str
    id_project: str
    id_user: str
    project_data: bytes



class FromUser(BaseModel):
    project: Project
    point: list[Point]
    line: list[Line]
    polygon_geometry: list[PolygonGeometry]
    polygon_data: list[PolygonData]
