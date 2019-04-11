from threading import Thread
from get_prediction import get_prediction
import cv2
import time
from keras.models import load_model

our_model = load_model('neural_network/fire_detector_model.h5')
SECONDS_BY_PHOTO = 10


class CameraVision:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.cam_live = True

    def start_cam(self):
        self.cam_live = True
        Thread(target=self.show_camera).start()

    def show_camera(self):
        while self.cam_live:
            _, frame = self.capture.read()
            cv2.imshow('FireDetector', frame)
            key = cv2.waitKey(1)
            if (key == 27) or (cv2.getWindowProperty('FireDetector', 1) < 1):
                self.cam_live = False
        cv2.destroyWindow("FireDetector")
        self.cam_live = False

    def save_check_photo(self):
        flag, frame = self.capture.read()
        cv2.imwrite('check_photo/check.png', frame)

    def send_check_photo_after(self):
        while True and self.cam_live:
            time.sleep(SECONDS_BY_PHOTO)
            self.save_check_photo()
            # Place for prediction function

    def send_photo(self):
        x = Thread(target=self.send_check_photo_after)
        x.daemon = True
        x.start()

if __name__ == '__main__':
    camera = CameraVision()
    camera.start_cam()
    camera.send_photo()