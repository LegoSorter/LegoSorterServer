from sqlalchemy.orm import Session

from . import Models, Schemas


def get_config(db: Session, option: str):
    return db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == option).first()


def get_configs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Models.DBConfiguration).offset(skip).limit(limit).all()


def update_config(db: Session, option: str, value:str):
    conf = db.query(Models.DBConfiguration).filter(Models.DBConfiguration.option == option).first()
    if conf is not None:
        conf.value = value
        db.commit()
    return conf


def get_session(db: Session, name: str):
    return db.query(Models.DBSession).filter(Models.DBSession.name == name).first()


def get_sessions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Models.DBSession).offset(skip).limit(limit).all()


def get_sessions_with_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Models.DBSession).offset(skip).limit(limit).all()
