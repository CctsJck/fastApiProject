import os
from os import remove
from os import path
from fer import Video
from fer import FER
import pandas as pd
from fastapi import FastAPI, File, UploadFile


app = FastAPI()
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    with open(file.filename, "wb") as myfile:
        myfile.write(await file.read())
        myfile.close()
        fer(file.filename)
        emotions = dataAnalysis()
    eliminar(file.filename)
    return emotions

def fer(video:str):
    video = Video(video)
    detector = FER(mtcnn=True)
    raw_data = video.analyze(detector, display=True)
    df = video.to_pandas(raw_data)

def eliminar(fileName:str):
    directorio_actual = os.getcwd()
    ruta = os.path.join(directorio_actual, "output")
    if path.exists(fileName):
        remove(fileName)
    if path.exists("data.csv"):
        remove("data.csv")
    if path.exists("output\\images.zip"):
        remove("output\\images.zip")
    if path.exists("output\\"+fileName[:-4]+"_output.mp4"):
        remove("output\\"+fileName[:-4]+"_output.mp4")
    if path.exists(ruta):
        os.rmdir(ruta)

def dataAnalysis():
    df = pd.read_csv('data.csv', header=0)
    data = {"angry":df['angry0'].mean()*100,
            "disgust":df['disgust0'].mean() * 100,
            "fear":df['fear0'].mean() * 100,
            "happy":df['happy0'].mean() * 100,
            "neutra":df['neutral0'].mean() * 100,
            "sad":df['sad0'].mean() * 100,
            "surprise":df['surprise0'].mean() * 100}
    return data

