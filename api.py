import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, jsonify
from pathlib import Path
import base64
from io import BytesIO
from flask_cors import CORS, cross_origin
import json
from json import dumps
import bson
import os
import re
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from flask_cors import CORS, cross_origin
connection_string: str = os.environ.get("mongodb+srv://minhkietpeople:IvptaLQLohowN4aE@cluster0-minhkietwebeco.ovhz23k.mongodb.net/?retryWrites=true&w=majority")
mongo_client: MongoClient = MongoClient(connection_string)
database: Database = mongo_client.get_database("test")
collection = Collection = database.get_collection("products")

app = Flask(__name__)
CORS(app)
# Read image features
fe = FeatureExtractor()
features = []
img_paths = []


for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(feature_path.stem)
   # print("khoi dau", features)
features = np.array(features)
img_paths = np.array(img_paths)


@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            products= []
            file = re.sub('^data:image/.+;base64,', '', request.json['query_img'])
            image = base64.b64decode(file)
            img = Image.open(BytesIO(image))
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") +"find.jpg"
            img = img.convert('RGB')
            img.save(uploaded_img_path)
            query = fe.extract(img)
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            ids = np.argsort(dists)[:30]  # Top 30 results
            for id in ids :
                products.append(img_paths[id])
            x={"data": products, "status": "OK" }
            s=json.dumps(x)
            return s
        except  Exception:
             return jsonify("error")
    if request.method == 'GET':
        return {"data" :"OK"}
   

if __name__=="__main__":
    app.run(debug=True)
    
