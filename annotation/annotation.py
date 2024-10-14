# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from image_to_annotations import image_to_annotations


def run(img_fn, char_anno_dir):

    image_to_annotations(img_fn, char_anno_dir)