import os
import json
import orjson
import numpy as np
import supervision as sv
import cv2
from PIL import Image



# Masks


def get_detections(frame_data, frame, scale=None):
    anns = frame_data.get('annotations') or []
    shape = frame.shape if isinstance(frame, np.ndarray) else (frame.height, frame.width)

    # extract annotation data
    xyxy = np.array([
        np.asarray(d['bounding_box']) if 'bounding_box' in d else 
        seg2box(d['segments'], scale) if 'segments' in d else 
        np.zeros(4)
        for d in anns
    ])
    mask = None
    if any('segments' in d for d in anns):
        mask = np.array([
            seg2mask(d['segments'], shape, scale) if 'segments' in d else 
            np.zeros(shape[:2], dtype=bool)
            for d in anns
        ], dtype=bool).reshape(-1, *shape[:2])
    confidence = [d.get('confidence', 1) for d in anns]
    class_id = [d['class_id'] for d in anns]
    track_id = [d.get('track_id', -1) for d in anns]
    labels = [d['name'] for d in anns]

    # create detections
    detections = sv.Detections(
        xyxy=xyxy.reshape(-1, 4),
        mask=mask,
        class_id=np.array(class_id, dtype=int),
        tracker_id=np.array(track_id, dtype=int),
        confidence=np.array(confidence),
        data={'labels': np.array(labels, dtype=str)},
    )
    return detections


def get_detections_h5(g, frame_ids, frame, scale=None):
    detections = []
    segments = g['segments']
    names = g['names']
    class_ids = g['class_ids']
    track_ids = g['track_ids']
    confidences = g['confidences']
    frame_index = g['frame_index']
    for fid in frame_ids:
        idxs = np.where(frame_index[()] == fid)[0]
        detections.append(get_detections({
            'annotations': [
                {
                    'segments': orjson.loads(segments[i]),
                    'name': names[i].decode(),
                    'class_id': int(class_ids[i]),
                    'track_id': int(track_ids[i]),
                    'confidence': float(confidences[i]),
                } for i in idxs
            ]
        }, frame, scale))
    return detections


TARGET = np.array([456, 256])
VISOR_SCALE = np.array([854, 480])
SCALE = list(VISOR_SCALE / TARGET)
def prepare_poly(points, scale=None):
    X = np.array([x for x in points]).reshape(-1, 2)
    if scale is not False:
        scale = SCALE if scale is None or scale is True else scale
        X = X / np.asarray(scale)
    return X.astype(int)


def seg2poly(segments, scale=None, min_points=3):
    xs = [prepare_poly(x, scale) for x in segments]
    return [x for x in xs if len(x) >= min_points]


def seg2box(segments, scale=None):
    X = np.concatenate(seg2poly(segments, scale) or [np.zeros((0, 2))])
    if X.shape[0] == 0:
        return np.array([0, 0, 0, 0])
    return np.concatenate([np.min(X, axis=0), np.max(X, axis=0)])


def seg2mask(segments, shape, scale=None):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    xs = seg2poly(segments, scale)
    if len(xs):
        mask = cv2.fillPoly(mask, xs, [255])
    return mask > 0




ma = sv.MaskAnnotator(opacity=0.4)
# pa = sv.PolygonAnnotator()
# ba = sv.BoxCornerAnnotator()
ba = sv.BoundingBoxAnnotator(thickness=1)
la = sv.LabelAnnotator(text_position=sv.Position.CENTER, text_scale=0.45, text_padding=2)
def draw_detections(image, detections, det_index, boxes_only=False):
    labels = detections.data['labels']
    color_lookup = np.array([det_index.index(labels[i]) for i in range(len(detections))])
    image = np.array(image)[:, :, ::-1]
    image = image.copy()
    is_ma = detections.mask.any((1,2)) if detections.mask is not None and not boxes_only else np.zeros(len(detections), dtype=bool)
    is_ba = ~is_ma & detections.xyxy.any(1)
    image = ma.annotate(scene=image, detections=detections[is_ma], custom_color_lookup=color_lookup[is_ma])
    image = ba.annotate(scene=image, detections=detections[is_ba], custom_color_lookup=color_lookup[is_ba])
    image = la.annotate(scene=image, detections=detections, labels=color_lookup.astype(str), custom_color_lookup=color_lookup)
    return Image.fromarray(image[:, :, ::-1])
