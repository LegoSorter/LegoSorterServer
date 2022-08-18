import argparse
from loguru import logger

import fiftyone as fo
from threading import Event

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from lego_sorter_server.database import Crud, Models, Schemas
from lego_sorter_server.database.Database import SessionLocal, engine
from lego_sorter_server.server import Server
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig

Models.Base.metadata.create_all(bind=engine)

app = FastAPI()

brickCategoryConfig = {}

server_elements = {}

event = [Event()]


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick_category_config", "-c", help='.json file with brick-category mapping specification',
                        type=str, required=False)
    args = parser.parse_args()
    brickCategoryConfig["brickCategoryConfig"] = BrickCategoryConfig(args.brick_category_config)
    server_elements["elements"] = Server.run(brickCategoryConfig["brickCategoryConfig"], event[0])
    # name = "testDB7"
    # dataset = fo.load_dataset(name)
    db = SessionLocal()
    server_fiftyone_port = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_fiftyone_port").one_or_none()
    if server_fiftyone_port is None:
        server_fiftyone_port = Models.DBConfiguration(option="server_fiftyone_port", value="5151")
        db.add(server_fiftyone_port)
        db.commit()
        db.refresh(server_fiftyone_port)

    server_fiftyone_address = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == "server_fiftyone_address").one_or_none()
    if server_fiftyone_address is None:
        server_fiftyone_address = Models.DBConfiguration(option="server_fiftyone_address", value="0.0.0.0")
        db.add(server_fiftyone_address)
        db.commit()
        db.refresh(server_fiftyone_address)
    fo.launch_app(port=int(server_fiftyone_port.value), remote=True, address=server_fiftyone_address.value)  # , address="192.168.11.189"
    db.close()


origins = ["*"]
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    # "http://localhost:8080",

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)










@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/sessions/", response_model=list[Schemas.DBSession])
def read_sessions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    sessions = Crud.get_sessions(db, skip=skip, limit=limit)
    return sessions


@app.get("/sessions_fo/")
def read_sessions_fo():
    sessions = fo.list_datasets()
    return sessions

@app.get("/sessions_with_items/", response_model=list[Schemas.DBSessionWithItems])
def read_sessions_with_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    sessions = Crud.get_sessions(db, skip=skip, limit=limit)
    return sessions


@app.get("/start_session/{name}")
def start_session(name: str, db: Session = Depends(get_db)):
    sessions = Crud.get_session(db, name=name)
    if (sessions is not None):
        if(name in fo.list_datasets()):
            dataset = fo.load_dataset(name)
        else:
            dataset = fo.Dataset.from_dir(
                labels_path=sessions.path,
                dataset_type=fo.types.VOCDetectionDataset,
                name=name,
            )
        return True
    return False


@app.get("/ref_session/{name}")
def refresh_session(name: str, db: Session = Depends(get_db)):
    sessions = Crud.get_session(db, name=name)
    if(sessions is not None):
        if(fo.dataset_exists(name)):
            fo.delete_dataset(name)
        dataset = fo.Dataset.from_dir(
            # data_path=data_path,
            labels_path=sessions.path,
            dataset_type=fo.types.VOCDetectionDataset,
            name=name,
        )
        return True
    return False


@app.get("/stop_session/{name}")
def delete_session(name: str, db: Session = Depends(get_db)):
    sessions = Crud.get_session(db, name=name)
    if(sessions is not None):
        if(fo.dataset_exists(name)):
            fo.delete_dataset(name)
            return True
    return False


@app.patch("/configurations/", response_model=Schemas.DBConfiguration)
def update_conf(conf: Schemas.DBConfigurationCreate, db: Session = Depends(get_db)):
    conf = Crud.update_config(db, option=conf.option, value=conf.value)
    return conf


@app.get("/configurations/{option}", response_model=Schemas.DBConfiguration)
def read_conf(option: str, db: Session = Depends(get_db)):
    conf = Crud.get_config(db, option=option)
    return conf


@app.get("/configurations/", response_model=list[Schemas.DBConfiguration])
def read_conf(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    conf = Crud.get_configs(db, skip=skip, limit=limit)
    return conf


@app.get("/stop/")
def stop_server():
    if len(server_elements) > 0:
        Server.stop(*server_elements["elements"], event[0])
        server_elements.clear()
        event[0] = Event()
        return {True}

    return {False}


@app.get("/start/")
def start_server():
    if len(server_elements) == 0:
        server_elements["elements"] = Server.run(brickCategoryConfig["brickCategoryConfig"], event[0])
        return {True}

    return {False}


@app.get("/restart/")
def restart_server():
    if len(server_elements) > 0:
        Server.stop(*server_elements["elements"], event[0])
        server_elements.clear()
        event[0] = Event()
    server_elements["elements"] = Server.run(brickCategoryConfig["brickCategoryConfig"], event[0])

    return {True}
