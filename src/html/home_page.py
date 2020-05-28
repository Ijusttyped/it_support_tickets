from flask import Flask, render_template, request
from flask_httpauth import HTTPBasicAuth


app = Flask(__name__)
auth = HTTPBasicAuth()


users = {
    "TestUser": "password",
}


@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/get_results", methods=['GET', 'POST'])
# @auth.login_required
def get_results():
    if request.method == "POST":
        test = request.form
    print(test)
    return "Hello"
    # return render_template("secrets.html", title=auth.username())
