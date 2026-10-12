import cv2
from PyQt5 import QtGui


class ImageDetector:

    def __init__(self, detector):
        self.detector = detector

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            return None, 0, 0

        annotated_frame, wei_dai, yi_dai = self.detector.detect_frame(frame)

        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QtGui.QImage(
            rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888
        )

        return qt_image, wei_dai, yi_dai