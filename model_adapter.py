import os
import torch
import dtlpy as dl
from mmdet.apis import init_detector, inference_detector
import logging
import subprocess

logger = logging.getLogger('MMYolo')


@dl.Package.decorators.module(description='Model Adapter for mmlabs yolo model',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class MMYolo(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        device = self.model_entity.configuration.get('device', None)
        config_name = self.model_entity.configuration.get('model_name', 'yolox_s_fast_8xb8-300e_coco')
        config_file = self.model_entity.configuration.get('config_file', 'yolox_s_fast_8xb8-300e_coco.py')
        checkpoint_file = self.model_entity.configuration.get('checkpoint_file',
                                                              'yolox_s_fast_8xb8-300e_coco_20230213_142600-2b224d8b.pth')

        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            logger.info("Downloading MMYolo artifacts")
            download_status = subprocess.Popen(f'mim download mmyolo --config {config_name} --dest .',
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               shell=True)
            download_status.wait()
            (out, err) = download_status.communicate()
            if download_status.returncode != 0:
                raise Exception(f'Failed to download MMYolo artifacts: {err}')
            logger.info(f"MMYolo artifacts downloaded successfully, Loading Model {out}")

        logger.info("MMYolo artifacts downloaded successfully, Loading Model")
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model on device {device}")
        self.model = init_detector(config_file, checkpoint_file, device=device)
        logger.info("Model and Classes Loaded Successfully")

    def predict(self, batch, **kwargs):
        logger.info(f"Predicting on batch of {len(batch)} images")
        confidence_thr = self.model_entity.configuration.get('confidence_thr', 0.4)
        batch_annotations = list()
        for image in batch:
            image_annotations = dl.AnnotationCollection()
            detections = inference_detector(self.model, image).pred_instances

            detections = detections[detections.scores >= confidence_thr]

            for det in detections:
                left, top, right, bottom = det.bboxes[0]
                image_annotations.add(annotation_definition=dl.Box(top=top,
                                                                   left=left,
                                                                   bottom=bottom,
                                                                   right=right,
                                                                   label=self.model_entity.labels[det.labels]),
                                      model_info={'name': self.model_entity.name,
                                                  'model_id': self.model_entity.id,
                                                  'confidence': det.scores})
            batch_annotations.append(image_annotations)
            logger.info(f"Found {len(image_annotations)} annotations in image")
        return batch_annotations
