# coding: utf-8
from sqlalchemy import Column, Float, ForeignKey, Integer, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from lego_sorter_server.database.Database import Base


class DBConfiguration(Base):
    __tablename__ = 'dbconfiguration'

    option = Column(Text, primary_key=True, index=True)
    value = Column(Text, nullable=False)


class DBSession(Base):
    __tablename__ = 'dbsession'

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False, unique=True)
    path = Column(Text, nullable=False)

    items = relationship('DBImage', back_populates="owner")


class DBImage(Base):
    __tablename__ = 'dbimage'

    id = Column(Integer, primary_key=True)
    owner_id = Column(ForeignKey('dbsession.id'), nullable=False, index=True)
    filename = Column(Text, nullable=False, unique=True)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    VOC_exist = Column(Boolean, nullable=False)

    owner = relationship('DBSession', back_populates="items")
    results = relationship('DBImageResult', back_populates="owner")


class DBImageResult(Base):
    __tablename__ = 'dbimageresult'

    id = Column(Integer, primary_key=True)
    owner_id = Column(ForeignKey('dbimage.id'), nullable=False, index=True)
    label = Column(Text, nullable=False)
    x_min = Column(Integer, nullable=False)
    y_min = Column(Integer, nullable=False)
    x_max = Column(Integer, nullable=False)
    y_max = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)

    owner = relationship('DBImage', back_populates="results")
