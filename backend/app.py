from flask import Flask,request,jsonify,send_file,url_for
from flask_cors import CORS
import base64
from PIL import Image
import requests
from threading import Thread
import numpy as np
import os
from ImageProcessing import analyseImage
#from app4_rn import predict_image_classification_sample 

app = Flask(__name__)
CORS(app)
count = 1

@app.route("/classification",methods = ["POST"])
def classification():
    global count
    req = request.get_json(silent=False, force=True)
    photo = req['photo']
    image=open('image.jpeg','wb')
    image.write(base64.b64decode(photo.split(',')[1]))
    image.close()
    res = analyseImage('image.jpeg')
    print(res)
    # foldername = "fish_tank"+str(count)
    foldername = "fish_tank1"
    # os.mkdir(foldername)
    for item in res:
        filename = foldername+"/"+f"{item[0]}.txt"
        content = str(item[1])
        file = open(filename, "w")
        file.write(content)
        
        # 关闭文件
        file.close()
    count += 1
    return jsonify("finished !")
    #classification = ["clean recyclables", "dirty recyclables", "non recyclables"]
    #return jsonify(classification[np.argmax(res)])

if __name__ == '__main__':
    app.run(host="192.168.137.13", port=5000, debug=True)
    # app.run(host="0.0.0.0", port=5000, debug=True)
# flask run --host=0.0.0.0