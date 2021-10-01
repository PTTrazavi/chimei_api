import os
import json
import base64

with open("temp.json") as f:
    results = json.load(f)
    # decode base64 image string
    end_diastole = base64.b64decode(results[0]['end_diastole'])
    end_systole = base64.b64decode(results[0]['end_systole'])
    cam = base64.b64decode(results[0]['cam'])

with open("end_diastole.png", "wb") as fh:
    fh.write(end_diastole)

with open("end_systole.png", "wb") as fh:
    fh.write(end_systole)

with open("cam.png", "wb") as fh:
    fh.write(cam)
