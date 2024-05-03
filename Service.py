# Import required libraries
import flask
from flask import Flask, render_template,request 
import numpy as np
import pandas as pd
import os
import warnings
from ultralytics import YOLO
import pandas as pd
import io
import cv2
import numpy as np
from PIL import Image
# filter warnings
warnings.filterwarnings('ignore')
import logging
import random
import tempfile
from time import time

#dreate log file
logging.basicConfig(filename='log/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Load Model weights
weight_g='E:/Projects/Workplace_Safety/Counting_persons/runs/detect/yolov8n_augmented/weights/best.pt'
model_g= YOLO(weight_g)

#Define Image size
image_size = 640


def get_coordinates_graphic(images_path):
    classes = ['Persona']
    
    
    image_file_path=images_path
    image = cv2.imread(image_file_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (image_size, image_size)) 
    
    class_list=[]
    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]
    conf=[]
        
    results = model_g.predict(image, imgsz=image_size)
    for r in results:        
        boxes = r.boxes
        for box in boxes:
            x = box.xyxy[0].cpu().numpy()
            xmin.append(x[0])
            ymin.append(x[1])
            xmax.append(x[2])
            ymax.append(x[3])
            class_list.append(classes[int(box.cls.cpu().numpy()[0])])
            conf.append(box.conf.cpu().numpy()[0])
            if "no" not in classes[int(box.cls.cpu().numpy()[0])]:
                cv2.rectangle(image, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (0, 255, 0), 2)
                #cv2.putText(image, str(classes[int(box.cls.cpu().numpy()[0])]), (int(x[0]), int(x[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:    
                cv2.rectangle(image, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (0, 0, 255), 2)
                #cv2.putText(image, str(classes[int(box.cls.cpu().numpy()[0])]), (int(x[0]), int(x[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
    cv2.imwrite('static/images/output_image.png', image)
    #cv2.imshow('Rectangles Image', image)
    #cv2.waitKey(0)
    df=pd.DataFrame({"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,"class":class_list,"conf":conf})
    
    return class_list



#initiate flask
app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['OUTPUT_FOLDER'] = 'static/images'

def str_generator(key, value):
    if isinstance(value, str):
        return key +': ' + value
    elif key == 'bio':
        return key + ':\n' + '\n'.join(value)
    else:
        return key + ': ' + str(value)



@app.route('/')
def index():
    return render_template('index_old.html', display_text=None, input_image=None, output_image=None)


@app.route('/Counting_Person', methods = ['POST'])
def defect_detection():    
    if flask.request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
            
        if file.filename == '':
            return "No selected file"

   
        try:
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            data = get_coordinates_graphic(full_path)
                    
            my_dict = {i:str(data.count(i)) for i in data}
            print(type(my_dict))
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_image.png')
            random_value = random.randint(1, 100000)
            timestamp = int(time())
            
            display_text = ""
            for key, value in my_dict.items():
                display_text += (''.join(str_generator(key, value)) + '<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;')
                    
            display_text = "<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Detected Objects: <br><br><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+ display_text +'</h3>'
            print(display_text)
            return render_template('index_old.html', display_text= display_text, input_image=full_path, output_image=output_image_path, random=random_value, timestamp=timestamp)
            #return "Predicted class: "+ str(my_dict)
        except Exception as e:
            logging.exception("Exception occurred")
            return " "
       
        
if __name__ == '__main__':
    app.run("127.0.0.1",5012,debug=True)
    
    
    