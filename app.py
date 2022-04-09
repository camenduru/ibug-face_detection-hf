#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import urllib.request
from typing import Union

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'face_detection')

from ibug.face_detection import RetinaFacePredictor, S3FDPredictor

REPO_URL = 'https://github.com/ibug-group/face_detection'
TITLE = 'ibug-group/face_detection'
DESCRIPTION = f'This is a demo for {REPO_URL}.'
ARTICLE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--face-score-slider-step', type=float, default=0.05)
    parser.add_argument('--face-score-threshold', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_model(
        model_name: str, threshold: float,
        device: torch.device) -> Union[RetinaFacePredictor, S3FDPredictor]:
    if model_name == 's3fd':
        model = S3FDPredictor(threshold=threshold, device=device)
    else:
        model_name = model_name.replace('retinaface_', '')
        model = RetinaFacePredictor(
            threshold=threshold,
            device=device,
            model=RetinaFacePredictor.get_model(model_name))
    return model


def detect(image: np.ndarray, model_name: str, face_score_threshold: float,
           detectors: dict[str, nn.Module]) -> np.ndarray:
    model = detectors[model_name]
    model.threshold = face_score_threshold

    # RGB -> BGR
    image = image[:, :, ::-1]
    preds = model(image, rgb=False)

    res = image.copy()
    for pred in preds:
        box = np.round(pred[:4]).astype(int)

        line_width = max(2, int(3 * (box[2:] - box[:2]).max() / 256))
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0),
                      line_width)

        if len(pred) == 15:
            pts = pred[5:].reshape(-1, 2)
            for pt in np.round(pts).astype(int):
                cv2.circle(res, tuple(pt), line_width, (0, 255, 0), cv2.FILLED)

    return res[:, :, ::-1]


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model_names = [
        'retinaface_mobilenet0.25',
        'retinaface_resnet50',
        's3fd',
    ]
    detectors = {
        name: load_model(name,
                         threshold=args.face_score_threshold,
                         device=device)
        for name in model_names
    }

    func = functools.partial(detect, detectors=detectors)
    func = functools.update_wrapper(func, detect)

    image_path = pathlib.Path('selfie.jpg')
    if not image_path.exists():
        url = 'https://raw.githubusercontent.com/peiyunh/tiny/master/data/demo/selfie.jpg'
        urllib.request.urlretrieve(url, image_path)
    examples = [[image_path.as_posix(), model_names[1], 0.8]]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(model_names,
                            type='value',
                            default='retinaface_resnet50',
                            label='Model'),
            gr.inputs.Slider(0,
                             1,
                             step=args.face_score_slider_step,
                             default=args.face_score_threshold,
                             label='Face Score Threshold'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
