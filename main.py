from flask import Flask, render_template, Response, request
from camera import VideoCamera
from detect_mask_image import detect_mask
import cv2

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route("/")
def index():
    return render_template("about.html")


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/main")
def main():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen(VideoCamera()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/imagecapture")
def imagecapture():
    return render_template("imagecapture.html")


@app.route("/livecapture")
def livecapture():
    return render_template("livecapture.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    img = request.files["image"]
    img.save("static/image.jpg")

    image = cv2.imread("static/image.jpg")
    prediction_image = detect_mask(image)
    cv2.imwrite("static/prediction.jpg", prediction_image)
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
