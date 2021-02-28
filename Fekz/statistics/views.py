from flask import url_for,render_template,redirect,request,Blueprint
from flask_restful import Resource

statistics = Blueprint('statistics',__name__,template_folder='templates/statistics')

@statistics.route('/dashboard')
def Dashboard():
    return render_template('dashboard.html')