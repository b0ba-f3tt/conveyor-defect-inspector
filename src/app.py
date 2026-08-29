import csv
import json
import os
from datetime import datetime

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from stream_reader import StreamReader
from qa_engine import QAEngine
from vlm_provider import QwenVLMProvider


load_dotenv()


RTSP_URL = os.getenv("RTSP_URL", "0")
MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    "models/yolov8n.pt"
)

LINE_Y = int(
    os.getenv("LINE_Y", "350")
)

CONF_THRESHOLD = float(
    os.getenv("CONF_THRESHOLD", "0.45")
)

DATA_DIR = os.getenv(
    "DATA_DIR",
    "./data"
)

VLM_PROVIDER = os.getenv(
    "VLM_PROVIDER",
    "qwen_api"
)

VLM_BASE_URL = os.getenv(
    "VLM_BASE_URL",
    "http://localhost:11434/v1"
)

VLM_MODEL_NAME = os.getenv(
    "VLM_MODEL_NAME",
    "qwen2.5vl:7b"
)


PLANK_CLASSES = {
    "plank",
    "wood",
    "wooden_plank",
    "wooden plank",
    "board",
    "wood_board",
    "wood board"
}


os.makedirs(
    DATA_DIR,
    exist_ok=True
)


SNAPSHOT_DIR = os.path.join(
    DATA_DIR,
    "defect_snapshots"
)

os.makedirs(
    SNAPSHOT_DIR,
    exist_ok=True
)


LOG_FILE = os.path.join(
    DATA_DIR,
    "continuous_count_log.csv"
)

SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "live_count_summary.json"
)


def initialize_log():
    if os.path.exists(LOG_FILE):
        return

    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Plank_ID",
            "Timestamp",
            "Status",
            "Defect_Types",
            "Running_Total_Count",
            "Running_Defect_Count",
            "Running_Clean_Count",
            "Defect_Snapshot_Path"
        ])


def default_summary():
    return {
        "last_updated": None,
        "running_total_count": 0,
        "running_clean_count": 0,
        "running_defective_count": 0,
        "defect_rate_percentage": 0.0,
        "last_counted_plank_id": None,
        "last_status": None
    }


def save_summary(summary):
    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )


def load_summary():
    defaults = default_summary()

    if not os.path.exists(SUMMARY_FILE):
        save_summary(defaults)
        return defaults

    try:
        with open(
            SUMMARY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            save_summary(defaults)
            return defaults

        for key, value in defaults.items():
            if key not in data:
                data[key] = value

        return data

    except (
        json.JSONDecodeError,
        OSError
    ):
        save_summary(defaults)
        return defaults


def write_log(
    plank_id,
    status,
    defects,
    snapshot_path,
    summary
):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            plank_id,
            timestamp,
            status,
            defects,
            summary["running_total_count"],
            summary["running_defective_count"],
            summary["running_clean_count"],
            snapshot_path or ""
        ])


def save_plank_snapshot(
    frame,
    box,
    plank_id
):
    x1, y1, x2, y2 = map(
        int,
        box
    )

    height, width = frame.shape[:2]

    padding = 20

    x1 = max(
        0,
        x1 - padding
    )

    y1 = max(
        0,
        y1 - padding
    )

    x2 = min(
        width,
        x2 + padding
    )

    y2 = min(
        height,
        y2 + padding
    )

    plank_crop = frame[
        y1:y2,
        x1:x2
    ]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"plank_{plank_id:04d}_"
        f"{timestamp}.jpg"
    )

    path = os.path.join(
        SNAPSHOT_DIR,
        filename
    )

    success = cv2.imwrite(
        path,
        plank_crop
    )

    if not success:
        raise RuntimeError(
            f"Could not save snapshot: {path}"
        )

    return path


def update_summary(
    summary,
    plank_id,
    status
):
    summary[
        "running_total_count"
    ] += 1

    if status == "DEFECTIVE":

        summary[
            "running_defective_count"
        ] += 1

    elif status == "CLEAN":

        summary[
            "running_clean_count"
        ] += 1

    total = summary[
        "running_total_count"
    ]

    defective = summary[
        "running_defective_count"
    ]

    if total > 0:

        summary[
            "defect_rate_percentage"
        ] = round(
            (
                defective / total
            ) * 100,
            2
        )

    summary[
        "last_updated"
    ] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    summary[
        "last_counted_plank_id"
    ] = plank_id

    summary[
        "last_status"
    ] = status

    save_summary(summary)


def get_center(box):
    x1, y1, x2, y2 = box

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )

    return center_x, center_y


def is_plank(class_name):
    return (
        class_name.strip().lower()
        in PLANK_CLASSES
    )


def inspect_plank_with_qwen(
    qa_engine,
    snapshot_path
):
    question = """
You are inspecting a wooden plank for manufacturing defects.

Analyze ONLY the wooden plank visible in the image.

Determine whether the plank is:

CLEAN

or

DEFECTIVE

Look for visible defects including:
- cracks
- knots
- splits
- holes
- rot
- deformation
- surface damage
- broken edges
- other visible wood defects

Do not classify people, clothing, walls, furniture, lighting,
shadows, background objects, or camera artifacts as defects.

Return ONLY this format:

STATUS: CLEAN

or:

STATUS: DEFECTIVE
DEFECTS: crack, knot

If the image does not clearly show a wooden plank, return:

STATUS: UNKNOWN
DEFECTS: none
"""

    try:

        response = (
            qa_engine.vlm_provider.ask(
                question,
                snapshot_path
            )
        )

        return response

    except Exception as error:

        print(
            f"Qwen inspection failed: {error}"
        )

        return None


def parse_qwen_result(response):
    if not response:
        return "UNKNOWN", []

    text = response.lower()

    if "status: defective" in text:

        status = "DEFECTIVE"

    elif "status: clean" in text:

        status = "CLEAN"

    elif "status: unknown" in text:

        status = "UNKNOWN"

    else:

        status = "UNKNOWN"

    defects = []

    if status == "DEFECTIVE":

        for line in response.splitlines():

            if "defects:" in line.lower():

                defect_text = line.split(
                    ":",
                    1
                )[1].strip()

                if (
                    defect_text
                    and defect_text.lower()
                    != "none"
                ):

                    defects = [
                        item.strip()
                        for item in
                        defect_text.split(",")
                        if item.strip()
                    ]

                break

    return status, defects


def resolve_model_path():
    model_path = MODEL_WEIGHTS

    if os.path.isabs(model_path):
        return model_path

    if os.path.exists(model_path):
        return model_path

    possible_paths = [
        os.path.join(
            "/app",
            model_path
        ),
        os.path.join(
            "/app/models",
            os.path.basename(model_path)
        )
    ]

    for path in possible_paths:

        if os.path.exists(path):
            return path

    return model_path


def main():
    initialize_log()

    summary = load_summary()

    print(
        "Loading YOLO model..."
    )

    model_path = resolve_model_path()

    model = YOLO(
        model_path
    )

    print(
        f"YOLO model loaded: "
        f"{model_path}"
    )

    print(
        "YOLO classes available:"
    )

    for class_id, class_name in model.names.items():
        print(
            f"  {class_id}: {class_name}"
        )

    vlm_provider = None

    if VLM_PROVIDER == "qwen_api":

        vlm_provider = QwenVLMProvider(
            VLM_BASE_URL,
            VLM_MODEL_NAME
        )

        print(
            "Qwen VLM configured:"
        )

        print(
            f"  Model: "
            f"{VLM_MODEL_NAME}"
        )

        print(
            f"  API: "
            f"{VLM_BASE_URL}"
        )

    else:

        print(
            "Qwen VLM is disabled."
        )

    qa_engine = QAEngine(
        DATA_DIR,
        vlm_provider
    )

    stream = StreamReader(
        RTSP_URL
    )

    print(
        "Starting video stream..."
    )

    if not stream.connect():

        print(
            "Unable to connect to "
            "video source."
        )

        print(
            "The application will "
            "continue retrying."
        )

    counted_ids = set()

    previous_positions = {}

    while True:

        frame = stream.read()

        if frame is None:
            continue

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        annotated_frame = frame.copy()

        cv2.line(
            annotated_frame,
            (0, LINE_Y),
            (
                annotated_frame.shape[1],
                LINE_Y
            ),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            "COUNTING LINE",
            (20, LINE_Y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        if (
            results
            and results[0].boxes is not None
        ):

            boxes = results[0].boxes

            for box in boxes:

                if box.conf is None:
                    continue

                confidence = float(
                    box.conf[0]
                )

                if (
                    confidence
                    < CONF_THRESHOLD
                ):
                    continue

                coordinates = (
                    box.xyxy[0]
                    .tolist()
                )

                center_x, center_y = (
                    get_center(
                        coordinates
                    )
                )

                track_id = None

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                class_id = int(
                    box.cls[0]
                )

                class_name = (
                    model.names.get(
                        class_id,
                        str(class_id)
                    )
                )

                plank = is_plank(
                    class_name
                )

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )

                if plank:

                    box_color = (
                        0,
                        255,
                        0
                    )

                else:

                    box_color = (
                        150,
                        150,
                        150
                    )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                if track_id is not None:

                    label = (
                        f"ID {track_id} | "
                        f"{label}"
                    )

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2
                )

                if not plank:
                    continue

                if track_id is None:
                    continue

                previous_y = (
                    previous_positions.get(
                        track_id
                    )
                )

                previous_positions[
                    track_id
                ] = center_y

                if previous_y is None:
                    continue

                crossed = (
                    previous_y < LINE_Y
                    and center_y >= LINE_Y
                )

                if not crossed:
                    continue

                if track_id in counted_ids:
                    continue

                counted_ids.add(
                    track_id
                )

                plank_id = (
                    summary[
                        "running_total_count"
                    ] + 1
                )

                print(
                    f"Plank #{plank_id} "
                    f"crossed counting line."
                )

                snapshot_path = (
                    save_plank_snapshot(
                        frame,
                        coordinates,
                        plank_id
                    )
                )

                print(
                    f"Saved plank image: "
                    f"{snapshot_path}"
                )

                status = "UNKNOWN"

                defects = []

                if vlm_provider is not None:

                    print(
                        f"Sending plank "
                        f"#{plank_id} to "
                        f"Qwen2.5-VL..."
                    )

                    response = (
                        inspect_plank_with_qwen(
                            qa_engine,
                            snapshot_path
                        )
                    )

                    status, defects = (
                        parse_qwen_result(
                            response
                        )
                    )

                    print(
                        f"Qwen result | "
                        f"Plank #{plank_id} | "
                        f"{status} | "
                        f"{defects or 'None'}"
                    )

                else:

                    print(
                        "Qwen VLM is not "
                        "configured."
                    )

                if status == "DEFECTIVE":

                    defect_snapshot_path = (
                        snapshot_path
                    )

                else:

                    defect_snapshot_path = None

                    if os.path.exists(
                        snapshot_path
                    ):

                        os.remove(
                            snapshot_path
                        )

                update_summary(
                    summary,
                    plank_id,
                    status
                )

                write_log(
                    plank_id,
                    status,
                    (
                        ";".join(defects)
                        if defects
                        else "None"
                    ),
                    (
                        defect_snapshot_path
                        or ""
                    ),
                    summary
                )

        cv2.putText(
            annotated_frame,
            (
                f"Total: "
                f"{summary['running_total_count']}"
            ),
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            (
                f"Defective: "
                f"{summary['running_defective_count']}"
            ),
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            (
                f"Clean: "
                f"{summary['running_clean_count']}"
            ),
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            (
                f"Defect Rate: "
                f"{summary['defect_rate_percentage']:.2f}%"
            ),
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "CCTV Plank Inspection",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    stream.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()