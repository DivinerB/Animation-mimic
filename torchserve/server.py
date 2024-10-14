from flask import Flask, request, jsonify
import torch
import base64
import mmcv
import os
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_bottom_up_pose_model, 
                         inference_top_down_pose_model, init_pose_model)
from mmpose.models.detectors import AssociativeEmbedding, TopDown

app = Flask(__name__)

# Initialize the MMdetHandler for detection
class MMdetHandler:
    def __init__(self, model_dir):
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location)
        
        self.config_file = os.path.join(model_dir, 'detector_config.py')
        self.checkpoint = os.path.join(model_dir, 'detector_model.pth')  # Adjust for your detection model
        
        self.model = init_detector(self.config_file, self.checkpoint, self.device)

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)
        return images

    def inference(self, data):
        results = []
        for image in data:
            result = inference_detector(self.model, image)
            results.append(result)
        return results

    def postprocess(self, data):
        output = []
        for image_index, image_result in enumerate(data):
            output.append([])
            bbox_result = image_result if not isinstance(image_result, tuple) else image_result[0]
            for class_index, class_result in enumerate(bbox_result):
                class_name = self.model.CLASSES[class_index]
                for bbox in class_result:
                    bbox_coords = bbox[:-1].tolist()
                    score = float(bbox[-1])
                    if score >= 0.5:  # Set your score threshold
                        output[image_index].append({
                            'class_name': class_name,
                            'bbox': bbox_coords,
                            'score': score
                        })
        return output

# Initialize the MMPoseHandler for pose estimation
class MMPoseHandler:
    def __init__(self, model_dir):
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location)
        
        self.config_file = os.path.join(model_dir, 'pose_config.py')
        self.checkpoint = os.path.join(model_dir, 'pose_model.pth')  # Adjust for your pose model
        
        self.model = init_pose_model(self.config_file, self.checkpoint, self.device)

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)
        return images

    def inference(self, data):
        if isinstance(self.model, TopDown):
            return self._inference_top_down_pose_model(data)
        elif isinstance(self.model, AssociativeEmbedding):
            return self._inference_bottom_up_pose_model(data)
        else:
            raise NotImplementedError(f'Model type {type(self.model)} is not supported.')

    def _inference_top_down_pose_model(self, data):
        results = []
        for image in data:
            preds, _ = inference_top_down_pose_model(self.model, image, person_results=None)
            results.append(preds)
        return results

    def _inference_bottom_up_pose_model(self, data):
        results = []
        for image in data:
            preds, _ = inference_bottom_up_pose_model(self.model, image)
            results.append(preds)
        return results

    def postprocess(self, data):
        output = [[{'keypoints': pred['keypoints'].tolist()} for pred in preds] for preds in data]
        return output

# Create instances of both handlers
detector_handler = MMdetHandler('model-store')  # Use the model directory for detection
pose_handler = MMPoseHandler('model-store')  # Use the model directory for pose estimation

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get input data from request
        input_data = request.files['image'].read()
        
        # Preprocess the image
        images = detector_handler.preprocess([{'data': base64.b64encode(input_data).decode('utf-8')}])
        
        # Run detection inference
        results = detector_handler.inference(images)
        
        # Postprocess results
        output = detector_handler.postprocess(results)
        
        return jsonify(output[0])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pose', methods=['POST'])
def pose():
    try:
        # Get input data from request
        input_data = request.files['image'].read()
        
        # Preprocess the image
        images = pose_handler.preprocess([{'data': base64.b64encode(input_data).decode('utf-8')}])
        
        # Run pose estimation inference
        results = pose_handler.inference(images)
        
        # Postprocess results
        output = pose_handler.postprocess(results)
        
        return jsonify(output[0])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run():
    app.run(debug=True)