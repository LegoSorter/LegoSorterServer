from typing import Union

from pydantic import BaseModel


class DBConfigurationBase(BaseModel):
    option: str
    value: str


class DBConfigurationCreate(DBConfigurationBase):
    pass


class DBConfiguration(DBConfigurationBase):

    class Config:
        orm_mode = True


class DBImageResultBase(BaseModel):
    label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float


class DBImageResultCreate(DBImageResultBase):
    pass


class DBImageResult(DBImageResultBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class DBImageBase(BaseModel):
    filename: str
    image_width: int
    image_height: int
    VOC_exist: bool


class DBImageCreate(DBImageBase):
    pass


class DBImage(DBImageBase):
    id: int
    owner_id: int

    results: list[DBImageResult] = []

    class Config:
        orm_mode = True


class DBSessionBase(BaseModel):
    name: str
    path: str


class DBSessionCreate(DBSessionBase):
    pass


class DBSession(DBSessionBase):
    id: int

    class Config:
        orm_mode = True


class DBSessionWithItems(DBSessionBase):
    id: int
    items: list[DBImage] = []

    class Config:
        orm_mode = True
