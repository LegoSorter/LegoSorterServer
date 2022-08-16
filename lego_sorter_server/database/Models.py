from peewee import *

db = SqliteDatabase('lego_sorter_server/database/LegoSorterServerDB.db')


class DBConfiguration(Model):
    option = TextField(unique=True, primary_key=True)
    value = TextField()

    class Meta:
        database = db  # This model uses the "LegoSorterDB.db" database.


class DBSession(Model):
    name = TextField(unique=True)
    path = TextField()

    class Meta:
        database = db  # This model uses the "LegoSorterDB.db" database.


class DBImage(Model):
    owner = ForeignKeyField(DBSession, backref='images')
    filename = TextField(unique=True)
    image_width = IntegerField()
    image_height = IntegerField()
    VOC_exist = BooleanField()

    class Meta:
        database = db  # This model uses the "LegoSorterDB.db" database.


class DBImageResult(Model):
    owner = ForeignKeyField(DBImage, backref='results')
    label = TextField()
    x_min = IntegerField()
    y_min = IntegerField()
    x_max = IntegerField()
    y_max = IntegerField()
    score = FloatField()

    class Meta:
        database = db  # This model uses the "LegoSorterDB.db" database.