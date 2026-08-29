import cv2
import time


class StreamReader:
    def __init__(self, source, reconnect_delay=5):
        self.source = self._parse_source(source)
        self.reconnect_delay = reconnect_delay
        self.cap = None

    def _parse_source(self, source):
        if str(source).strip() == "0":
            return 0

        try:
            return int(source)
        except ValueError:
            return source

    def connect(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)

        if isinstance(self.source, str):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                time.sleep(self.reconnect_delay)
                return None

        success, frame = self.cap.read()

        if not success:
            self.cap.release()
            self.cap = None
            time.sleep(self.reconnect_delay)
            return None

        return frame

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None