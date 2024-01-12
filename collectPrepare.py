import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import base64
import re
from io import BytesIO
from flask_cors import CORS, cross_origin
import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from flask_cors import CORS, cross_origin
mongo_client: MongoClient = MongoClient("mongodb+srv://minhkietpeople:IvptaLQLohowN4aE@cluster0-minhkietwebeco.ovhz23k.mongodb.net/?retryWrites=true&w=majority")
database: Database = mongo_client.get_database("test")
collection = Collection = database.get_collection("products")
prods =[]
products= list(collection.find())
for product in products :
   ob = {"id": product['_id'], "image" : product['image']}
   prods.append(ob)
# Read image features
# fe = FeatureExtractor()
# features = []
# img_paths = []

# for feature_path in prods:
    
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./static/ImageSearch") / (feature_path.stem + ".jpg"))
#    # print("khoi dau", features)
# features = np.array(features)
# img_paths = np.array(img_paths)


if __name__ == '__main__':
    fe = FeatureExtractor()

    for prod in prods:
        # print(img_path)  # e.g., ./static/img/xxx.jpg
        file = re.sub('^data:image/.+;base64,', '', prod['image'])
        file = base64.b64decode(file)
        img = Image.open(BytesIO(file))
        img = img.convert('RGB')
        feature = fe.extract(img)

        print ("thong so", feature)
        feature_path = Path("./static/feature") / (str(prod['id']) + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
   