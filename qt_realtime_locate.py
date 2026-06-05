#!/usr/bin/env python3
"""Native Qt webcam demo for LocateAnything-3B MLX 4-bit."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import transformers
import transformers.processing_utils as transformers_processing_utils
from mlx_vlm.models.locateanything.image_processing_locateanything import (
    LocateAnythingImageProcessor,
)
from mlx_vlm.models.locateanything.processing_locateanything import (
    LocateAnythingProcessor,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_path, load_model, prepare_inputs


MODEL_ID = "mlx-community/LocateAnything-3B-4bit"
transformers.LocateAnythingProcessor = LocateAnythingProcessor
transformers.LocateAnythingImageProcessor = LocateAnythingImageProcessor
transformers_processing_utils.transformers_module.LocateAnythingProcessor = LocateAnythingProcessor
transformers_processing_utils.transformers_module.LocateAnythingImageProcessor = LocateAnythingImageProcessor


def build_prompt(task: str, query: str) -> str:
    if task == "detect":
        categories = "</c>".join(part.strip() for part in query.split(",") if part.strip())
        return f"Locate all the instances that matches the following description: {categories}."
    if task == "ground-single":
        return f"Locate a single instance that matches the following description: {query}."
    if task == "ground-multi":
        return f"Locate all the instances that match the following description: {query}."
    if task == "gui-box":
        return f"Locate the region that matches the following description: {query}."
    if task in {"point", "gui-point"}:
        return f"Point to: {query}."
    return query


def parse_boxes(answer: str, width: int, height: int) -> list[dict[str, float]]:
    boxes = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            {
                "x1": x1 / 1000 * width,
                "y1": y1 / 1000 * height,
                "x2": x2 / 1000 * width,
                "y2": y2 / 1000 * height,
            }
        )
    return boxes


def parse_points(answer: str, width: int, height: int) -> list[dict[str, float]]:
    points = []
    for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
        x, y = [int(group) for group in match.groups()]
        points.append({"x": x / 1000 * width, "y": y / 1000 * height})
    return points


class CameraWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index
        self._running = False

    @Slot()
    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Could not open camera index {self.camera_index}.")
            self.finished.emit()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        while self._running:
            ok, frame_bgr = cap.read()
            if ok:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
            QThread.msleep(25)

        cap.release()
        self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False


class InferenceWorker(QObject):
    result_ready = Signal(str, object, object, float, int)
    status = Signal(str)
    error = Signal(str)
    loaded = Signal()

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._closing = False

    def _load(self) -> None:
        if self.model is not None:
            return
        if self._closing:
            return
        self.status.emit("Loading MLX 4-bit model...")
        model_path = get_model_path(self.model_id)
        if self._closing:
            return
        self.model = load_model(model_path)
        if self._closing:
            return
        self.processor = LocateAnythingProcessor.from_pretrained(model_path)
        if self._closing:
            return
        self.status.emit("Model ready")
        self.loaded.emit()

    @Slot()
    def load(self) -> None:
        try:
            self._load()
        except Exception as exc:
            if not self._closing:
                self.error.emit(str(exc))

    @Slot()
    def stop(self) -> None:
        self._closing = True

    @Slot(object, str, str, str, int, int)
    def infer(
        self,
        frame_rgb: np.ndarray,
        task: str,
        query: str,
        generation_mode: str,
        max_tokens: int,
        inference_width: int,
    ) -> None:
        try:
            if self._closing:
                return
            self._load()
            if self._closing:
                return
            assert self.model is not None
            assert self.processor is not None

            width = int(frame_rgb.shape[1])
            height = int(frame_rgb.shape[0])
            image = Image.fromarray(frame_rgb)
            if inference_width > 0 and width > inference_width:
                inference_height = max(1, int(height * inference_width / width))
                image = image.resize((inference_width, inference_height), Image.Resampling.BILINEAR)
                model_width, model_height = image.size
            else:
                model_width, model_height = width, height

            prompt_text = build_prompt(task, query)
            prompt = apply_chat_template(self.processor, self.model.config, prompt_text, num_images=1)
            inputs = prepare_inputs(self.processor, images=[image], prompts=prompt)
            input_ids = inputs.pop("input_ids")
            inputs.pop("attention_mask", None)

            start = time.perf_counter()
            tokens = self.model.pbd_generate(
                input_ids,
                generation_mode=generation_mode,
                max_tokens=max_tokens,
                **inputs,
            )
            if self._closing:
                return
            elapsed = time.perf_counter() - start
            answer = self.processor.decode(tokens, skip_special_tokens=False)
            boxes = parse_boxes(answer, model_width, model_height)
            points = parse_points(answer, model_width, model_height)
            if model_width != width or model_height != height:
                scale_x = width / model_width
                scale_y = height / model_height
                for box in boxes:
                    box["x1"] *= scale_x
                    box["x2"] *= scale_x
                    box["y1"] *= scale_y
                    box["y2"] *= scale_y
                for point in points:
                    point["x"] *= scale_x
                    point["y"] *= scale_y
            self.result_ready.emit(answer, boxes, points, elapsed, len(tokens))
        except Exception as exc:
            if not self._closing:
                self.error.emit(str(exc))


class MainWindow(QMainWindow):
    request_inference = Signal(object, str, str, str, int, int)

    def __init__(self, camera_index: int, model_id: str):
        super().__init__()
        self.setWindowTitle("LocateAnything Live")
        self.resize(1180, 760)

        self.latest_frame: np.ndarray | None = None
        self.latest_boxes: list[dict[str, float]] = []
        self.latest_points: list[dict[str, float]] = []
        self.inference_busy = False
        self.camera_running = False
        self.using_uploaded_image = False
        self.model_ready = False
        self.closing = False
        self.shutdown_started = False

        self.preview = QLabel("Camera stopped")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(760, 520)
        self.preview.setStyleSheet("background:#101214;color:#aeb4bb;border-radius:6px;")

        self.query = QLineEdit("red square, blue circle")
        self.task = QComboBox()
        self.task.addItems(["detect", "point", "ground-single", "ground-multi", "gui-box", "gui-point"])
        self.mode = QComboBox()
        self.mode.addItems(["fast", "hybrid", "slow"])
        self.mode.setCurrentText("hybrid")
        self.inference_width = QComboBox()
        self.inference_width.addItems(["128", "192", "256", "384", "512", "640", "0"])
        self.inference_width.setCurrentText("256")
        self.max_tokens = QComboBox()
        self.max_tokens.addItems(["64", "128", "256", "512"])
        self.max_tokens.setCurrentText("64")

        self.start_button = QPushButton("Start camera")
        self.upload_button = QPushButton("Upload image")
        self.pause_check = QCheckBox("Pause")
        self.run_button = QPushButton("Run now")
        self.run_button.setEnabled(False)
        self.run_button.setVisible(False)
        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setMinimumHeight(160)
        self.status = QLabel("Ready")

        controls = QFormLayout()
        controls.addRow("Prompt", self.query)
        controls.addRow("Task", self.task)
        controls.addRow("Mode", self.mode)
        controls.addRow("Inference width", self.inference_width)
        controls.addRow("Max tokens", self.max_tokens)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.upload_button)
        button_row.addWidget(self.pause_check)
        button_row.addWidget(self.run_button)

        side = QVBoxLayout()
        title = QLabel("LocateAnything Live")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        side.addWidget(title)
        side.addLayout(controls)
        side.addLayout(button_row)
        side.addWidget(QLabel("Latest result"))
        side.addWidget(self.answer)
        side.addStretch()
        side.addWidget(self.status)

        root = QHBoxLayout()
        root.addWidget(self.preview, 3)
        root.addLayout(side, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.camera_index = camera_index
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None

        self.infer_thread = QThread(self)
        self.infer_worker = InferenceWorker(model_id)
        self.infer_worker.moveToThread(self.infer_thread)
        self.request_inference.connect(self.infer_worker.infer)
        self.infer_worker.result_ready.connect(self.on_result)
        self.infer_worker.status.connect(self.status.setText)
        self.infer_worker.error.connect(self.on_infer_error)
        self.infer_worker.loaded.connect(self.on_model_loaded)
        self.infer_thread.started.connect(self.infer_worker.load)
        self.infer_thread.start()

        self.start_button.clicked.connect(self.toggle_camera)
        self.upload_button.clicked.connect(self.upload_image)
        self.pause_check.toggled.connect(self.on_pause_toggled)
        self.run_button.clicked.connect(self.run_once)

    def toggle_camera(self) -> None:
        if self.camera_running:
            if self.camera_worker is not None:
                self.camera_worker.stop()
            if self.camera_thread is not None:
                self.camera_thread.quit()
                self.camera_thread.wait(1500)
            self.camera_running = False
            self.start_button.setText("Start camera")
            self.status.setText("Camera stopped")
        else:
            self.using_uploaded_image = False
            self.latest_boxes = []
            self.latest_points = []
            self.camera_thread = QThread(self)
            self.camera_worker = CameraWorker(self.camera_index)
            self.camera_worker.moveToThread(self.camera_thread)
            self.camera_thread.started.connect(self.camera_worker.run)
            self.camera_worker.frame_ready.connect(self.on_frame)
            self.camera_worker.error.connect(self.on_error)
            self.camera_worker.finished.connect(self.camera_thread.quit)
            self.camera_thread.start()
            self.camera_running = True
            self.start_button.setText("Stop camera")
            self.status.setText("Camera running")
            if self.model_ready and not self.pause_check.isChecked():
                QTimer.singleShot(0, self.maybe_infer)

    def upload_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            QMessageBox.warning(self, "Image error", f"Could not open image:\n{path}")
            return
        if self.camera_running:
            self.toggle_camera()
        self.using_uploaded_image = True
        self.latest_boxes = []
        self.latest_points = []
        self.latest_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.pause_check.setChecked(True)
        self.status.setText(f"Loaded image: {Path(path).name}")
        self.draw_preview()

    def on_pause_toggled(self, paused: bool) -> None:
        self.run_button.setVisible(paused)
        self.run_button.setEnabled(paused and self.model_ready)
        if not paused and self.model_ready and self.latest_frame is not None:
            QTimer.singleShot(0, self.maybe_infer)

    @Slot(object)
    def on_frame(self, frame_rgb: np.ndarray) -> None:
        self.latest_frame = frame_rgb
        self.draw_preview()
        if (
            self.model_ready
            and self.camera_running
            and not self.using_uploaded_image
            and not self.pause_check.isChecked()
            and not self.inference_busy
        ):
            QTimer.singleShot(0, self.maybe_infer)

    def draw_preview(self) -> None:
        if self.latest_frame is None:
            return
        frame = np.ascontiguousarray(self.latest_frame)
        height, width, channels = frame.shape
        image = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        scale_x = pixmap.width() / width
        scale_y = pixmap.height() / height
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#00e0a4"), 3))
        for box in self.latest_boxes:
            painter.drawRect(
                int(box["x1"] * scale_x),
                int(box["y1"] * scale_y),
                int((box["x2"] - box["x1"]) * scale_x),
                int((box["y2"] - box["y1"]) * scale_y),
            )
        painter.setPen(QPen(QColor("#ff3366"), 4))
        for point in self.latest_points:
            x = int(point["x"] * scale_x)
            y = int(point["y"] * scale_y)
            painter.drawEllipse(x - 8, y - 8, 16, 16)
            painter.drawLine(x - 18, y, x + 18, y)
            painter.drawLine(x, y - 18, x, y + 18)
        painter.end()
        self.preview.setPixmap(pixmap)

    def maybe_infer(self) -> None:
        self._start_inference(force=False)

    def run_once(self) -> None:
        self._start_inference(force=True)

    def _start_inference(self, force: bool) -> None:
        if self.closing:
            return
        if not self.model_ready:
            self.status.setText("Model is still loading...")
            return
        if self.inference_busy or self.latest_frame is None:
            return
        if self.pause_check.isChecked() and not force:
            return
        self.inference_busy = True
        self.status.setText("Running inference...")
        self.request_inference.emit(
            self.latest_frame.copy(),
            self.task.currentText(),
            self.query.text().strip(),
            self.mode.currentText(),
            int(self.max_tokens.currentText()),
            int(self.inference_width.currentText()),
        )

    @Slot(str, object, object, float, int)
    def on_result(self, answer: str, boxes: list[dict[str, float]], points: list[dict[str, float]], elapsed: float, tokens: int) -> None:
        if self.closing:
            return
        self.latest_boxes = boxes
        self.latest_points = points
        tps = tokens / elapsed if elapsed > 0 else 0
        self.answer.setPlainText(answer)
        self.status.setText(f"{elapsed:.2f}s | {tokens} tokens | {tps:.2f} tok/s")
        self.inference_busy = False
        self.draw_preview()
        if (
            self.camera_running
            and not self.using_uploaded_image
            and not self.pause_check.isChecked()
        ):
            QTimer.singleShot(0, self.maybe_infer)

    @Slot()
    def on_model_loaded(self) -> None:
        if self.closing:
            return
        self.model_ready = True
        self.run_button.setEnabled(self.pause_check.isChecked())
        if self.latest_frame is not None and not self.pause_check.isChecked():
            QTimer.singleShot(0, self.maybe_infer)

    @Slot(str)
    def on_infer_error(self, message: str) -> None:
        if self.closing:
            return
        self.inference_busy = False
        self.status.setText("Inference error")
        self.answer.setPlainText(message)

    @Slot(str)
    def on_error(self, message: str) -> None:
        if self.closing:
            return
        self.status.setText(message)
        QMessageBox.warning(self, "Camera error", message)

    def shutdown(self) -> None:
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.closing = True
        self.start_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.pause_check.setEnabled(False)
        self.run_button.setEnabled(False)
        if self.status is not None:
            self.status.setText("Closing...")

        try:
            self.request_inference.disconnect(self.infer_worker.infer)
        except RuntimeError:
            pass
        try:
            self.infer_worker.result_ready.disconnect(self.on_result)
            self.infer_worker.status.disconnect(self.status.setText)
            self.infer_worker.error.disconnect(self.on_infer_error)
            self.infer_worker.loaded.disconnect(self.on_model_loaded)
        except RuntimeError:
            pass
        self.infer_worker.stop()

        if self.camera_worker is not None:
            self.camera_worker.stop()
            try:
                self.camera_worker.frame_ready.disconnect(self.on_frame)
                self.camera_worker.error.disconnect(self.on_error)
            except RuntimeError:
                pass
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.quit()
            if not self.camera_thread.wait(2000):
                self.camera_thread.terminate()
                self.camera_thread.wait(1000)
        self.infer_thread.quit()
        if not self.infer_thread.wait(250):
            # If MLX is inside a load/generate call, let process teardown clean it up
            # instead of force-terminating the thread mid-kernel.
            pass

    def closeEvent(self, event) -> None:
        self.shutdown()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--screenshot", type=Path, help="Save a screenshot of the app window and exit.")
    parser.add_argument("--screenshot-delay", type=float, default=1.5)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(camera_index=args.camera, model_id=args.model)
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    if args.screenshot:
        def save_screenshot() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pixmap = window.grab()
            pixmap.save(str(args.screenshot))
            app.quit()

        QTimer.singleShot(int(args.screenshot_delay * 1000), save_screenshot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
