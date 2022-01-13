from flask import Flask, render_template, request
from flask import jsonify
import requests
from attendance_using_Image import *

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return 'This is Homepage!'


@app.route("/attendance", methods=['GET','POST'])
def attendance():
    text=make_attendance()
    return text

    
if __name__=="__main__":
    app.run(debug=True)