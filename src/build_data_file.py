#
# build a data file (joblib format)  from the source files
#


import pandas as pd 
import glob
import os
import joblib

SOURCE_PATH=os.path.join (os.path.dirname(__file__), "../sources/")
OUT_FILE = os.path.join (os.path.dirname(__file__), "../models/model_data.joblib")


sources=[]

for file in glob.glob (os.path.join (SOURCE_PATH, "*.txt")):
    with open (file, 'r') as f:
        tmp = f.readlines()

    sources.extend (tmp)

print(f"{len(sources)} lines read from sources.")

df = pd.DataFrame (sources)
joblib.dump (df, OUT_FILE)
