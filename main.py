# -*- coding: utf-8 -*-
"""
安全帽智能检测系统 —— 主程序入口
=================================
职责：DLL 环境准备、检测线程、主窗口、启动流程。

相关模块：
    database.py      MySQL 数据访问层
    ui_dialogs.py    登录框 / 历史记录弹窗 / 可点击缩略图
    diagnostics.py   自检与视频链路诊断（--selftest / --videotest）
    main_window.ui   主窗口界面定义
"""

import os
import sys
import time
import traceback
from datetime import datetime

# ============================================================================
# 崩溃现场取证：尽早开启 faulthandler。
# 原生崩溃（例如 Qt 内部 0xc0000005 访问违例）不会走 Python 异常钩子，
# 因此不会留下 error.log，表现为"无声闪退"。faulthandler 在收到
# SIGSEGV/SIGABRT/SIGFPE 等信号时，会把所有线程的 Python 调用栈
# 直接写入 crash.log，从而定位崩溃发生在哪一步。
# ============================================================================
try:
    import faulthandler as _faulthandler

    _crash_dir = (
        os.path.dirname(os.path.abspath(sys.executable))
        if getattr(sys, "frozen", False)
        else os.path.dirname(os.path.abspath(__file__))
    )
    _crash_fp = open(os.path.join(_crash_dir, "crash.log"), "wb", buffering=0)
    _faulthandler.enable(file=_crash_fp, all_threads=True)
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================================
# PyInstaller 打包后 DLL 搜索路径修复（必须放在 import torch 之前）
# torch 依赖大量 DLL（c10.dll 等），打包后默认搜索路径不包含 _internal\torch\lib，
# 会导致 "WinError 1114 动态链接库(DLL)初始化例程失败"。
# ============================================================================
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    _torch_lib = os.path.join(_base, "torch", "lib")

    # 1) 把关键 DLL 目录置于 PATH 最前
    _dll_dirs = [
        _torch_lib,
        _base,
        os.path.join(_base, "torch"),
        os.path.join(_base, "cv2"),
        os.path.join(_base, "PyQt5", "Qt5", "bin"),
    ]
    os.environ["PATH"] = os.pathsep.join(
        [d for d in _dll_dirs if os.path.isdir(d)]
    ) + os.pathsep + os.environ.get("PATH", "")

    # 2) Python 3.8+ 必须显式 add_dll_directory 才能让 C 扩展找到依赖 DLL
    for _d in _dll_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except (AttributeError, OSError):
                pass

    # 3) 兜底：用 ctypes 按依赖顺序显式预加载 torch 的核心 DLL，
    #    即便 add_dll_directory 未生效也能保证 c10/torch_cpu 正常初始化
    try:
        import ctypes
        for _name in ("libiomp5md.dll", "uv.dll", "shm.dll",
                      "c10.dll", "torch_cpu.dll", "torch_global_deps.dll",
                      "torch.dll", "torch_python.dll"):
            _p = os.path.join(_torch_lib, _name)
            if os.path.exists(_p):
                try:
                    ctypes.WinDLL(_p)
                except OSError:
                    pass
    except Exception:
        pass

# ==== 关键：torch 必须在 cv2 / numpy / PyQt5 之前导入 ====
# 根因：PyQt5 的 Qt5\bin 目录自带旧版 msvcp140.dll，与 torch 的 c10.dll
#       所需的 MSVC 运行库版本冲突，导致 "WinError 1114 DLL 初始化例程失败"。
#       （cv2 / PyQt5 是"帮凶"——它们把错误版本的 DLL 带进搜索范围；
#         ultralytics 是"报警人"——它只是第一个去 import torch 的包。）
# 对策：① 此处提前导入 torch，锁定正确的 DLL 加载顺序；
#       ② 配合 main.spec 中剔除 Qt 目录下的旧版 MSVC 运行库（治本），双重保障。
import torch  # noqa: F401  （必须位于 cv2/PyQt5 之前）

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from ultralytics import YOLO

from database import DatabaseManager
from ui_dialogs import HistoryDialog, LoginDialog

# ---------------------------------------------------------------------------
# 资源目录（只读）：打包后为 _MEIPASS（即 _internal 目录），存放 ui / 模型 / 字体
# 数据目录（可写）：打包后为 exe 所在目录，存放 snapshot 抓拍图等运行期产物
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BASE_DIR


# ---------------------------------------------------------------------------
# Qt 消息转发：把 Qt 内部的 warning/fatal 信息落盘到 qt.log。
# Qt 在即将崩溃前往往会先打印 critical/fatal 级别的警告（例如平台插件、
# 图像格式、OpenGL 上下文等），这些信息默认写到 stderr，
# 而打包成无控制台窗口的程序没有 stderr，会被直接丢弃。
# ---------------------------------------------------------------------------
def _qt_message_handler(_mode, _ctx, _msg):
    try:
        with open(os.path.join(APP_DIR, "qt.log"), "a", encoding="utf-8") as _fh:
            _fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {_msg}\n")
    except Exception:
        pass


try:
    QtCore.qInstallMessageHandler(_qt_message_handler)
except Exception:
    pass


def resource_path(name):
    """把相对资源名解析为绝对路径（兼容打包后的 _internal 目录）"""
    if os.path.isabs(name):
        return name
    p = os.path.join(BASE_DIR, name)
    return p if os.path.exists(p) else name


def _font_path(name):
    """优先使用程序自带字体，其次使用系统字体"""
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        return p
    return name  # 回退到系统字体查找


# 辅助函数：使用 PIL 在 OpenCV 图像上绘制中文文本
def draw_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(_font_path("simhei.ttf"), font_size, encoding="utf-8")
    except OSError:
        try:
            font = ImageFont.truetype(_font_path("msyh.ttc"), font_size, encoding="utf-8")
        except OSError:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_custom_boxes(frame, boxes):
    """自定义绘制中文检测框和标签"""
    if len(boxes) == 0:
        return frame

    annotated_frame = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        # 0: 未戴, 1: 已戴
        if cls_id == 0:
            label = f"未戴 {conf:.2f}"
            color = (255, 0, 0)  # BGR 蓝色
        else:
            label = f"已戴 {conf:.2f}"
            color = (255, 255, 0)  # BGR 青/黄色

        # 画框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=2)

        # 画标签背景
        label_size = len(label) * 15
        cv2.rectangle(
            annotated_frame,
            (x1, max(0, y1 - 25)),
            (x1 + label_size, y1),
            color,
            thickness=-1,
        )

        # 写中文
        annotated_frame = draw_chinese_text(
            annotated_frame,
            label,
            (x1 + 2, max(0, y1 - 23)),
            font_size=16,
            color=(255, 255, 255),
        )

    return annotated_frame


class DetectionThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QImage)
    update_count_signal = QtCore.pyqtSignal(int, int)

    def __init__(self, model_path="best.pt"):
        super().__init__()
        self.running = False
        # 将相对模型路径解析为绝对路径（打包后模型位于 _internal 目录）
        self.model_path = resource_path(model_path)
        self.source = 0
        self.conf_threshold = 0.6  # 默认置信度阈值
        self.cap = None
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"模型未找到: {e}")
            self.model = None

    def set_source(self, source):
        self.source = source

    def set_conf(self, conf):
        """动态设置置信度阈值"""
        self.conf_threshold = float(conf)

    def run(self):
        if self.model is None:
            return

        # 如果 source 是图片路径，直接进行单张图片检测（已加入内存安全保护）
        if isinstance(self.source, str) and self.source.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
        ):
            frame = cv2.imread(self.source)
            if frame is not None:
                frame = frame.copy()
                results = self.model(frame, conf=self.conf_threshold)
                wei_dai_count, yi_dai_count = 0, 0
                if len(results[0].boxes) > 0:
                    cls_list = results[0].boxes.cls.cpu().numpy().tolist()
                    wei_dai_count = cls_list.count(0)
                    yi_dai_count = cls_list.count(1)

                annotated_frame = draw_custom_boxes(frame, results[0].boxes)
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                rgb_image = np.ascontiguousarray(rgb_image)
                h, w, ch = rgb_image.shape
                qt_image = QtGui.QImage(
                    rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888
                ).copy()

                self.change_pixmap_signal.emit(qt_image)
                self.update_count_signal.emit(wei_dai_count, yi_dai_count)
            return

        # 如果 source 是摄像头 (0) 或视频文件。
        # 注意：VideoCapture 的创建、读取、释放全部在本线程内完成，
        #       绝不能从 GUI 线程调用 release()，否则会与 read() 竞争导致崩溃。
        try:
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
                rgb_image = np.ascontiguousarray(rgb_image)
                h, w, ch = rgb_image.shape
                qt_image = QtGui.QImage(
                    rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888
                ).copy()

                if self.running:
                    self.change_pixmap_signal.emit(qt_image)
                    self.update_count_signal.emit(wei_dai_count, yi_dai_count)

                if isinstance(self.source, str):
                    elapsed_time = int((time.time() - start_time) * 1000)
                    sleep_time = max(1, frame_delay - elapsed_time)
                    self.msleep(sleep_time)
        finally:
            # 无论正常结束还是异常退出，都由本线程负责释放
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def stop(self):
        """请求停止。只置标志位，由工作线程自己释放 VideoCapture 并退出，
        避免 GUI 线程 release() 一个正被 read() 使用的对象（数据竞争）。"""
        self.running = False
        if self.isRunning():
            self.wait(3000)


class MainWindow(QtWidgets.QMainWindow):
    logout_signal = QtCore.pyqtSignal()

    def __init__(self, db_manager, user_role="operator"):
        super().__init__()
        self.db = db_manager
        self.user_role = user_role
        self.last_alarm_time = 0.0  # 初始化防抖时间戳

        loadUi(resource_path("main_window.ui"), self)

        self.setWindowTitle(
            f"安全帽智能检测系统 [当前登录角色: {self.user_role.upper()}]"
        )

        if hasattr(self, "label_show"):
            self.label_show.setScaledContents(True)
            self.label_show.setAlignment(QtCore.Qt.AlignCenter)

        # 动态生成“切换账号”按钮
        if not hasattr(self, "btn_logout") or self.btn_logout is None:
            self.btn_logout = QtWidgets.QPushButton("切换账号 / 注销", self)

            if hasattr(self, "btn_history") and self.btn_history.parent():
                parent_layout = self.btn_history.parent().layout()
                if parent_layout:
                    idx = parent_layout.indexOf(self.btn_history)
                    if idx != -1:
                        parent_layout.insertWidget(idx + 1, self.btn_logout)

        # 绑定按钮逻辑
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.select_video)
        self.btn_image.clicked.connect(self.select_image)
        if hasattr(self, "btn_start"):
            self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop.clicked.connect(self.stop_detection)

        if hasattr(self, "btn_history"):
            self.btn_history.clicked.connect(self.show_history)

        if hasattr(self, "btn_logout"):
            self.btn_logout.clicked.connect(self.handle_logout)

        # RBAC 权限控制逻辑
        if hasattr(self, "slider_conf"):
            self.slider_conf.setRange(10, 95)
            self.slider_conf.setValue(60)
            self.slider_conf.valueChanged.connect(self.on_conf_changed)

            if self.user_role == "operator":
                self.slider_conf.setEnabled(False)
                if hasattr(self, "label_conf_title"):
                    self.label_conf_title.setText("置信度 (仅Admin可调):")

        self.detection_thread = DetectionThread(model_path="best.pt")
        self.detection_thread.change_pixmap_signal.connect(self.update_image)
        self.detection_thread.update_count_signal.connect(self.update_counts)
        self.current_source = 0

    def handle_logout(self):
        reply = QtWidgets.QMessageBox.question(
            self,
            "切换账号",
            "确定要注销当前账号并返回登录界面吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.stop_detection()
            self.logout_signal.emit()
            self.close()

    def on_conf_changed(self, value):
        conf = value / 100.0
        self.detection_thread.set_conf(conf)
        if hasattr(self, "label_conf_val"):
            self.label_conf_val.setText(f"{conf:.2f}")

    def show_history(self):
        dialog = HistoryDialog(self.db, user_role=self.user_role, parent=self)
        dialog.exec_()

    def start_camera(self):
        self.stop_detection()
        self.current_source = 0
        self.detection_thread.set_source(self.current_source)
        self.detection_thread.start()

    def select_video(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_name:
            self.stop_detection()
            self.current_source = file_name
            self.detection_thread.set_source(self.current_source)
            self.detection_thread.start()

    def select_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.png *.jpeg *.bmp)"
        )
        if file_name:
            self.stop_detection()
            self.current_source = file_name
            self.detection_thread.set_source(self.current_source)
            self.detection_thread.start()

    def start_detection(self):
        if not self.detection_thread.isRunning():
            self.detection_thread.set_source(self.current_source)
            self.detection_thread.start()

    def stop_detection(self):
        if self.detection_thread.isRunning():
            self.detection_thread.stop()
            QtCore.QThread.msleep(50)

    def update_image(self, qt_img):
        scaled_pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(
            self.label_show.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.label_show.setPixmap(scaled_pixmap)

    def update_counts(self, wei_dai, yi_dai):
        if hasattr(self, "edit_head"):
            self.edit_head.setText(f"未戴数量: {wei_dai}")
        if hasattr(self, "edit_helmet"):
            self.edit_helmet.setText(f"已戴数量: {yi_dai}")

        if wei_dai > 0:
            now = time.time()
            if now - self.last_alarm_time >= 2.0:
                self.last_alarm_time = now

                # 抓拍图保存到程序所在目录的 snapshot 子目录（可写，不依赖工作目录）
                snap_dir = os.path.join(APP_DIR, "snapshot")
                os.makedirs(snap_dir, exist_ok=True)
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(snap_dir, f"alarm_{time_str}.jpg")

                pixmap = self.label_show.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    pixmap.toImage().save(snap_path)

                # 写库必须容错：PyQt5 中槽函数抛出未捕获异常会直接终止进程
                # （表现为"无声闪退"），所以这里降级为提示并继续检测。
                try:
                    self.db.add_alarm_record(wei_dai, yi_dai, snap_path)
                    print(f"[告警成功] 违规截图已保存至: {snap_path}，记录已写入 MySQL 数据库！")
                except Exception as e:
                    print(f"[告警写库失败] {type(e).__name__}: {e}（检测继续运行）")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


# --- 全局异常捕获：任何未处理异常写入日志，避免程序静默闪退 ---
def _excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        log_dir = os.path.join(os.path.expanduser("~"), "HelmetDetection")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "error.log"), "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n{msg}\n")
    except Exception:
        pass
    # 尽力弹出图形化提示
    try:
        # 持有引用，避免 QApplication 被回收后 Qt 对象失效
        _app_ref = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        QtWidgets.QMessageBox.critical(
            None, "程序运行错误",
            f"程序遇到未处理的错误：\n\n{exc_value}\n\n详细日志已写入:\n"
            f"{os.path.join(os.path.expanduser('~'), 'HelmetDetection', 'error.log')}",
        )
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook


# --- 主程序循环：支持注销后重新弹出登录框 ---
if __name__ == "__main__":
    # 诊断入口（正常使用不会触发，实现见 diagnostics.py）
    if "--selftest" in sys.argv:
        from diagnostics import run_selftest
        sys.exit(run_selftest(APP_DIR, resource_path, _font_path, MainWindow))

    if "--videotest" in sys.argv:
        from diagnostics import run_videotest
        sys.exit(run_videotest(
            APP_DIR, resource_path, MainWindow,
            sys.argv[sys.argv.index("--videotest") + 1:], DatabaseManager,
        ))

    app = QtWidgets.QApplication(sys.argv)

    # 数据库连接容错：连不上 MySQL 时给出友好提示，而不是直接闪退
    try:
        db_manager = DatabaseManager(password="root")
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            None,
            "数据库连接失败",
            "无法连接到本地 MySQL 数据库，程序无法启动。\n\n"
            "请确认：\n"
            "1. 已安装并启动 MySQL 服务（默认端口 3306）；\n"
            "2. 数据库用户为 root，密码为 root；\n\n"
            f"错误详情：{e}",
        )
        sys.exit(1)

    while True:
        login_dialog = LoginDialog(db_manager)
        if login_dialog.exec_() != QtWidgets.QDialog.Accepted:
            break

        window = MainWindow(db_manager=db_manager, user_role=login_dialog.user_role)
        restart_flag = [False]

        def on_logout():
            restart_flag[0] = True

        window.logout_signal.connect(on_logout)
        window.show()
        app.exec_()

        if not restart_flag[0]:
            break

    sys.exit(0)
