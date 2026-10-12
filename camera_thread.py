import os
import sys

# 1. 强行将当前文件所在的绝对目录加到最高优先级
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)

import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from ultralytics import YOLO

# 2. 尝试从绝对路径导入，如果失败则尝试相对导入
try:
  from ui_utils import draw_custom_boxes
except ImportError:
  from .ui_utils import draw_custom_boxes


class DetectionThread(QtCore.QThread):
  change_pixmap_signal = QtCore.pyqtSignal(QtGui.QImage)
  update_count_signal = QtCore.pyqtSignal(int, int)

  def __init__(self, model_path="best.pt"):
    super().__init__()
    self.running = False
    self.model_path = model_path
    self.source = 0
    self.conf_threshold = 0.6
    try:
      self.model = YOLO(self.model_path)
    except Exception as e:
      print(f"模型未找到: {e}")
      self.model = None

  def set_source(self, source):
    self.source = source

  def set_conf(self, conf):
    self.conf_threshold = float(conf)

  def run(self):
    if self.model is None:
      return

    # 单张图片处理
    if isinstance(self.source, str) and self.source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):
      frame = cv2.imread(self.source)
      if frame is not None:
        results = self.model(frame, conf=self.conf_threshold)
        wei_dai_count, yi_dai_count = 0, 0
        if len(results[0].boxes) > 0:
          cls_list = results[0].boxes.cls.cpu().numpy().tolist()
          wei_dai_count = cls_list.count(0)
          yi_dai_count = cls_list.count(1)

        annotated_frame = draw_custom_boxes(frame, results[0].boxes)
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QtGui.QImage(
            rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888
        )

        self.change_pixmap_signal.emit(qt_image)
        self.update_count_signal.emit(wei_dai_count, yi_dai_count)
      return

    # 视频/摄像头处理
    self.cap = cv2.VideoCapture(self.source)
    if not self.cap.isOpened():
      return

    fps = self.cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or isinstance(self.source, int):
      fps = 30
    frame_delay = int(1000 / fps)

    self.running = True
    while self.running and self.cap.isOpened():
      start_time = time.time()
      ret, frame = self.cap.read()
      if not ret:
        break

      results = self.model(frame, conf=self.conf_threshold)
      wei_dai_count, yi_dai_count = 0, 0

      if len(results[0].boxes) > 0:
        cls_list = results[0].boxes.cls.cpu().numpy().tolist()
        wei_dai_count = cls_list.count(0)
        yi_dai_count = cls_list.count(1)

      annotated_frame = draw_custom_boxes(frame, results[0].boxes)
      rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
      h, w, ch = rgb_image.shape
      qt_image = QtGui.QImage(
          rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888
      )

      if self.running:
        self.change_pixmap_signal.emit(qt_image)
        self.update_count_signal.emit(wei_dai_count, yi_dai_count)

      if isinstance(self.source, str):
        elapsed_time = int((time.time() - start_time) * 1000)
        sleep_time = max(1, frame_delay - elapsed_time)
        self.msleep(sleep_time)

    if hasattr(self, "cap") and self.cap and self.cap.isOpened():
      self.cap.release()

  def stop(self):
    self.running = False
    if hasattr(self, "cap") and self.cap and self.cap.isOpened():
      self.cap.release()
    self.wait()