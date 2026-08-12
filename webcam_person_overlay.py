"""Detect a person and show a non-activating warning overlay on Windows.

Install:
    py -m pip install opencv-python ultralytics pillow

Run:
    py webcam_person_alt_tab.py

Choose a detected camera when prompted, then activate your game or any other
application. Draw a rectangle around the part of the camera image you want to
monitor, then press Enter or Space. When a person is detected, a click-through
warning image appears in the top-right corner without taking focus from a game.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
from pathlib import Path
import platform
import tkinter as tk
import time

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SW_SHOWNOACTIVATE = 4
SW_HIDE = 0
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a warning overlay when the webcam detects a person."
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Skip the menu and use this webcam index",
    )
    parser.add_argument(
        "--reset-after",
        type=float,
        default=1.0,
        help="Person must be absent this long before another trigger (default: 1)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the camera preview (this window can take focus)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO person confidence from 0 to 1 (default: 0.35)",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Skip region selection and detect people in the entire camera frame",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "detection_screenshots",
        help="Folder for detection screenshots (default: detection_screenshots beside the script)",
    )
    parser.add_argument(
        "--overlay-image",
        type=Path,
        default=Path(__file__).resolve().parent / "warning_overlay.png",
        help="PNG image displayed when a person is detected",
    )
    parser.add_argument(
        "--overlay-width",
        type=int,
        default=150,
        help="Overlay width in pixels (default: 360)",
    )
    parser.add_argument(
        "--overlay-margin",
        type=int,
        default=20,
        help="Distance from the top and right screen edges (default: 20)",
    )
    parser.add_argument(
        "--test-overlay",
        action="store_true",
        help="Show the overlay for 5 seconds without opening the camera",
    )
    return parser.parse_args()


class WarningOverlay:
    """Click-through, always-on-top image that never activates itself."""

    def __init__(self, image_path: Path, width: int, margin: int) -> None:
        if not image_path.is_file():
            raise SystemExit(f"Overlay image was not found: {image_path}")
        if width <= 0:
            raise SystemExit("--overlay-width must be greater than 0.")
        if margin < 0:
            raise SystemExit("--overlay-margin cannot be negative.")

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        transparent_color = "#010203"
        self.root.configure(background=transparent_color)
        self.root.attributes("-transparentcolor", transparent_color)

        source = Image.open(image_path).convert("RGBA")
        height = max(1, round(source.height * width / source.width))
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.label = tk.Label(
            self.root,
            image=self.photo,
            background=transparent_color,
            borderwidth=0,
            highlightthickness=0,
        )
        self.label.pack()

        screen_width = self.root.winfo_screenwidth()
        self.x = max(0, screen_width - width - margin)
        self.y = margin
        self.width = width
        self.height = height
        self.root.geometry(f"{width}x{height}+{self.x}+{self.y}")
        self.root.update_idletasks()
        self.root.update()

        user32 = ctypes.windll.user32
        # winfo_id() is Tk's inner child window. Windows topmost/show styles
        # must be applied to its actual top-level wrapper window.
        inner_hwnd = self.root.winfo_id()
        self.hwnd = user32.GetParent(inner_hwnd) or inner_hwnd
        style = user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, style)
        self.visible = False

    def show(self) -> None:
        user32 = ctypes.windll.user32
        if not self.visible:
            user32.SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                self.x,
                self.y,
                self.width,
                self.height,
                SWP_NOACTIVATE,
            )
            user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
            self.visible = True
        else:
            # Reassert topmost because some borderless games periodically
            # reorder their windows.
            user32.SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        self.process_events()

    def hide(self) -> None:
        if self.visible:
            ctypes.windll.user32.ShowWindow(self.hwnd, SW_HIDE)
            self.visible = False
        self.process_events()

    def process_events(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self.root.destroy()


def find_available_cameras(max_index: int = 10) -> list[int]:
    """Return camera indexes that can provide a frame."""
    available: list[int] = []
    print("Scanning for cameras...")
    for index in range(max_index):
        test_camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if test_camera.isOpened():
            ok, _ = test_camera.read()
            if ok:
                available.append(index)
        test_camera.release()
    return available


def choose_camera() -> int:
    """Scan for cameras and ask the user to choose one."""
    cameras = find_available_cameras()
    if not cameras:
        raise SystemExit("No working cameras were detected.")

    print("\nAvailable cameras:")
    for index in cameras:
        print(f"  [{index}] Camera {index}")

    while True:
        answer = input("\nEnter the camera number to use: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            print("Please enter one of the camera numbers shown above.")
            continue
        if selected in cameras:
            return selected
        print("That camera is unavailable. Choose a number from the list.")


def select_detection_region(camera: cv2.VideoCapture) -> tuple[int, int, int, int]:
    """Let the user draw the camera region in which people will be detected."""
    frame = None
    for _ in range(30):
        ok, candidate = camera.read()
        if ok:
            frame = candidate
        time.sleep(0.01)

    if frame is None:
        raise SystemExit("Could not read a frame for region selection.")

    window_name = "Select detection area, then press Enter or Space"
    print("\nDraw a box around the area to monitor.")
    print("Press Enter or Space to confirm, or C to use the full camera frame.")
    x, y, width, height = map(
        int,
        cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False),
    )
    cv2.destroyWindow(window_name)

    # selectROI returns a zero-sized rectangle when the selection is cancelled.
    if width <= 0 or height <= 0:
        frame_height, frame_width = frame.shape[:2]
        print("No area selected; using the full camera frame.")
        return 0, 0, frame_width, frame_height

    print(f"Detection area selected: x={x}, y={y}, width={width}, height={height}")
    return x, y, width, height


def make_annotated_frame(
    frame,
    person_boxes,
    detection_bounds: tuple[int, int, int, int],
):
    """Return a copy of the frame marked with the ROI and detected people."""
    annotated = frame.copy()
    x1, y1, x2, y2 = detection_bounds
    color = (0, 255, 0)

    cv2.putText(
        annotated,
        "PERSON DETECTED",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(
        annotated,
        "DETECTION AREA",
        (x1 + 5, max(20, y1 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    for box_x1, box_y1, box_x2, box_y2 in person_boxes.xyxy.cpu().numpy().astype(int):
        cv2.rectangle(
            annotated,
            (box_x1 + x1, box_y1 + y1),
            (box_x2 + x1, box_y2 + y1),
            color,
            2,
        )
    return annotated


def save_detection_screenshot(frame, screenshot_dir: Path) -> Path | None:
    """Save a timestamped detection image and return its path if successful."""
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        screenshot_path = screenshot_dir / f"person_detected_{timestamp}.jpg"
        if cv2.imwrite(str(screenshot_path), frame):
            return screenshot_path
        print("Warning: OpenCV could not save the detection screenshot.")
    except OSError as error:
        print(f"Warning: Could not save detection screenshot: {error}")
    return None


def main() -> None:
    args = parse_args()
    if platform.system() != "Windows":
        raise SystemExit("This program currently supports Windows only.")

    if args.test_overlay:
        overlay = WarningOverlay(args.overlay_image, args.overlay_width, args.overlay_margin)
        print("Showing the overlay for 5 seconds...")
        try:
            end_time = time.monotonic() + 5.0
            while time.monotonic() < end_time:
                overlay.show()
                time.sleep(0.02)
        finally:
            overlay.close()
        print("Overlay test finished.")
        return

    camera_index = args.camera if args.camera is not None else choose_camera()
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise SystemExit(f"Could not open webcam {camera_index}.")

    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence must be greater than 0 and no more than 1.")

    if args.full_frame:
        ok, frame = camera.read()
        if not ok:
            raise SystemExit("Could not read a frame from the camera.")
        frame_height, frame_width = frame.shape[:2]
        detection_region = (0, 0, frame_width, frame_height)
    else:
        detection_region = select_detection_region(camera)

    region_x, region_y, region_width, region_height = detection_region
    overlay = WarningOverlay(args.overlay_image, args.overlay_width, args.overlay_margin)

    print("Loading YOLO person detector...")
    detector = YOLO("yolov8n.pt")

    armed = True
    absent_since: float | None = None

    if args.preview:
        print("Watching for a person. Press Q in the preview window or Ctrl+C to stop.")
    else:
        print("Watching for a person in the background. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                overlay.process_events()
                time.sleep(0.02)
                continue

            frame_height, frame_width = frame.shape[:2]
            x1 = max(0, min(region_x, frame_width - 1))
            y1 = max(0, min(region_y, frame_height - 1))
            x2 = max(x1 + 1, min(region_x + region_width, frame_width))
            y2 = max(y1 + 1, min(region_y + region_height, frame_height))
            detection_frame = frame[y1:y2, x1:x2]

            # Run YOLO only inside the user-selected region. COCO class 0 is person.
            result = detector.predict(
                source=detection_frame,
                classes=[0],
                conf=args.confidence,
                imgsz=640,
                verbose=False,
            )[0]
            person_boxes = result.boxes
            person_found = len(person_boxes) > 0
            now = time.monotonic()
            annotated_detection = None

            if person_found:
                absent_since = None
                overlay.show()
                if armed:
                    annotated_detection = make_annotated_frame(
                        frame,
                        person_boxes,
                        (x1, y1, x2, y2),
                    )
                    screenshot_path = save_detection_screenshot(
                        annotated_detection,
                        args.screenshot_dir,
                    )
                    armed = False
                    print("Person detected: warning overlay shown.")
                    if screenshot_path is not None:
                        print(f"Screenshot saved: {screenshot_path}")
            else:
                if absent_since is None:
                    absent_since = now
                elif not armed and now - absent_since >= args.reset_after:
                    overlay.hide()
                    armed = True
                    print("Person no longer detected: warning overlay hidden.")
                else:
                    overlay.process_events()

            if args.preview:
                if person_found:
                    if annotated_detection is not None:
                        preview_frame = annotated_detection
                    else:
                        preview_frame = make_annotated_frame(
                            frame,
                            person_boxes,
                            (x1, y1, x2, y2),
                        )
                else:
                    preview_frame = frame.copy()
                    cv2.putText(
                        preview_frame,
                        "Watching...",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )
                    cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.imshow("Person detector - Q to quit", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        overlay.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
