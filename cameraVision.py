from keras.models import load_model
from threading import Thread
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import time


our_model = load_model('neural_network/fire_detector_model.h5')
our_model._make_predict_function()
SECONDS_BY_PHOTO = 10
PATH_TO_PHOTO = 'check_photo/check.jpg'
USER_CHAT_ID = 433227293
SEND_MESSAGE_URL = 'https://api.telegram.org/bot800792656:AAF3UcFpElvjeG3q3b-Q9JjRVSEn_c_Y6JE/'\
                   'sendMessage?chat_id={0}&text=PREDICTION:{1}'


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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(PATH_TO_PHOTO, frame)

    def send_check_photo_after(self):
        while True and self.cam_live:
            time.sleep(SECONDS_BY_PHOTO)
            self.save_check_photo()
            self.make_prediction()

    def send_photo(self):
        x = Thread(target=self.send_check_photo_after)
        x.daemon = True
        x.start()

    @staticmethod
    def make_prediction():
        # CLASS 0 - FIRE
        # CLASS 1 - NOT FIRE
        test_image = load_img(PATH_TO_PHOTO, target_size=(150, 150, 3))
        test_image = img_to_array(test_image)/255
        test_image = np.expand_dims(test_image, axis=0)
        result = 1 - float("{:.2f}".format(our_model.predict(test_image)[0][0]))
        if result > 0.5:
            pass
            # call function to send message
        return



if __name__ == '__main__':
    x = CameraVision()
    x.make_prediction()