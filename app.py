from ultralytics import YOLO
from flask import request, Response, Flask, render_template
from PIL import Image
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file",
    passes it through YOLOv8 object detection
    network and returns an array of bounding boxes
    and kidney stone detection status.
    :return: a JSON object with "boxes" and "kidney_stones_detected" keys
    """
    buf = request.files["image_file"]
    boxes, kidney_stones_detected = detect_objects_on_image(Image.open(buf.stream))
    response_data = {
        "boxes": boxes,
        "kidney_stones_detected": kidney_stones_detected
    }
    return Response(
        json.dumps(response_data),
        mimetype='application/json'
    )


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes, along with the kidney stone detection status.
    :param buf: Input image file stream
    :return: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..], kidney_stones_detected (bool)
    """
    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    kidney_stones_detected = False
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
        if result.names[class_id] == "KIDNEY_STONE":
            kidney_stones_detected = True
    return output, kidney_stones_detected

if __name__ == '__main__':
    app.run()
