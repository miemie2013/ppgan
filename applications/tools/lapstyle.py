#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import os
import sys

sys.path.insert(0, os.getcwd())
from ppgan.apps import LapStylePredictor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_path",
                        type=str,
                        help="path to content image")

    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model weight path")

    parser.add_argument(
        "--style",
        type=str,
        default='starrynew',
        help=
        "if weight_path is None, style can be chosen in 'starrynew', 'circuit', 'ocean' and 'stars'"
    )

    parser.add_argument("--style_image_path",
                        type=str,
                        default=None,
                        help="path to style image")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = LapStylePredictor(output=args.output_path,
                                  style=args.style,
                                  weight_path=args.weight_path)
    predictor.run(args.content_img_path, args.style_image_path)
