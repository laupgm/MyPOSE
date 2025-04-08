# LIVE CAMS
#  https://github.com/AzwadFawadHasan/Public_MotionJPEG_Sources
#  http://insecam.org/en/bytag/Street/?page=2

# VIDEOS
#  https://mixkit.co/

from ultralytics import YOLO
import cv2
from PIL import Image
import base64


import os

from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, Response, url_for, render_template_string, send_from_directory


model = YOLO("yolo11s-pose.pt") # very good model


#VIDEO_SOURCE = "../videos/mixkit-cargo-ship-arriving-to-container-terminal-30979-hd-ready.mp4"
VIDEO_SOURCE = "./video/wheelchair-exercises.mp4"
#VIDEO_SOURCE = "./video/sports-runner.mp4"

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# HTML templates as raw strings
upload_page = '''
<!doctype html>
<title>Upload an Image</title>
<h1>Upload a picture</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
'''

display_page = '''
<!doctype html>
<title>Image Uploaded</title>
<h1>Here is your image:</h1>

<img src="data:image/jpeg;base64,{{ filename }}" alt="Image"/>

<!-- <img src="{{ url_for('uploaded_file', filename=filename) }}"> -->

<!-- {{ url_for('uploaded_file', filename=filename) }} -->

<br><br>
<a href="{{ url_for('upload') }}">Upload another</a>
'''


## VIDEO

def get_frame():

    print("-- opening ", VIDEO_SOURCE)

    camera_port=0
    camera=cv2.VideoCapture(VIDEO_SOURCE) #this makes a web cam object

    while True:
        success, frame = camera.read()
        if success:
            results = model(frame)

            print(results)

            annotated_frame = results[0].plot()
            #cv2.imshow("YOLO Inference", annotated_frame)
            imgencode = cv2.imencode('.jpg', annotated_frame)[1]
            #stringData=imgencode.tostring()
            stringData=imgencode.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: charset=utf-8; text/plain\r\n\r\n'+stringData+b'\r\n')
    del (camera)


## IMAGE

def process_image(img):

    print("-- opening ", img)
    frame = Image.open(img)

    results = model(frame)
    annotated_frame = results[0].plot()
    #cv2.imshow("YOLO Inference", annotated_frame)
    imgencode = cv2.imencode('.jpg', annotated_frame)[1]
    return base64.b64encode(imgencode).decode('utf-8')


##


@app.route('/vid')
def vid():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            #
            processed_img = process_image(filepath)
            #
            return render_template_string(display_page, filename=processed_img)
            #return render_template_string(display_page, filename=filename)
    return render_template_string(upload_page)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run(host='localhost',port=9999, debug=True, threaded=True)

