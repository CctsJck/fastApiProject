from fer import Video
from fer import FER
from fastapi import FastAPI, File, UploadFile


app = FastAPI()
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    with open(file.filename, "wb") as myfile:
        myfile.write(await file.read())
        myfile.close()
        fer(file.filename)
    return {"filename": file.filename}

def fer(video:str):
    video = Video(video)
    detector = FER(mtcnn=True)
    raw_data = video.analyze(detector, display=True)
    df = video.to_pandas(raw_data)
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

