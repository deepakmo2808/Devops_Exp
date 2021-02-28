import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_restful import Resource,Api
from Fekz.Algorithm.views import Text_Algo,Image_Algo,Text_Algo_head


app = Flask(__name__)



app.config['SECRET_KEY'] = 'ajsbcjajh12g7675&%&^@%&#@^(@HVGVGKFDSI^&%^@%(%^#@^T(@^^%^#$'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

Migrate(app,db)

from Fekz.Authentication.views import Authentication
from Fekz.statistics.views import statistics

app.register_blueprint(Authentication,url_prefix='/')
app.register_blueprint(statistics,url_prefix='/')


api = Api(app)
api.add_resource(Text_Algo_head, '/Algorithm/text_head')
api.add_resource(Text_Algo, '/Algorithm/text')
api.add_resource(Image_Algo, '/Algorithm/image')