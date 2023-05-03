import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils import (
    approximate_polygon,
    filter_polygons_by_area,
    mask_to_polygons,
    polygon_to_xyxy,
)


def load_yolo_txt_annotations(path: Union[str, Path], image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=' ', ndmin=2)

    if data.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.array([], dtype=int)

    class_id = data[:, 0].astype(int)
    x_center = data[:, 1] * image_shape[1]
    y_center = data[:, 2] * image_shape[0]
    width = data[:, 3] * image_shape[1]
    height = data[:, 4] * image_shape[0]

    # Convert (x_center, y_center, width, height) to (x1, y1, x2, y2) format
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    xyxy = np.column_stack([x1, y1, x2, y2])

    return xyxy, class_id

def detections_to_yolo_txt(detections: Detections, image_shape: Tuple[int, int]) -> str:
    # Convert the bounding box coordinates from (x1, y1, x2, y2) to (x_center, y_center, width, height)
    x_center = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
    y_center = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
    width = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    height = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    
    # Normalize the bounding box coordinates
    normalized_coords = np.column_stack([
        detections.class_id,
        x_center / image_shape[1],
        y_center / image_shape[0],
        width / image_shape[1],
        height / image_shape[0],
    ])

    lines = []
    for row in normalized_coords:
        lines.append(" ".join(map(str, row)))

    return "\n".join(lines) + '\n'
