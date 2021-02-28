from flask import Blueprint,render_template,redirect,url_for



Authentication = Blueprint('Authentication',__name__,template_folder='templates/Authentication')

@Authentication.route('/')
def login():
    return render_template('login.html')

@Authentication.route('/signup')
def signup():
    return render_template('signup.html')

@Authentication.route('/forgot')
def forgot():
    return render_template('forgot.html')

