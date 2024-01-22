import os

import dtlpy as dl
from mmdet.apis import init_detector, inference_detector
import logging

logger = logging.getLogger('MMYolo')


@dl.Package.decorators.module(description='Model Adapter for mmlabs yolo model',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class MMYolo(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        config_name = 'yolox_s_fast_8xb8-300e_coco'
        config_path = os.path.join(local_path, config_name)
        if not os.path.exists(config_path):
            os.system(f'mim download mmyolo --config {config_name} --dest {config_path}')

        files = os.listdir(config_path)
        for file in files:
            if file.endswith(".pth"):
                checkpoint_file = os.path.join(config_path, file)
            elif file.endswith(".py"):
                config_file = os.path.join(config_path, file)

        logger.info("MMYolo artifacts downloaded successfully, Loading Model")
        self.model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
        self.labels = self.model.dataset_meta.get('classes')
        logger.info("Model and Classes Loaded Successfully")

    def predict(self, batch, **kwargs):
        logger.info(f"Predicting on batch of {len(batch)} images")
        batch_annotations = list()
        for image in batch:
            image_annotations = dl.AnnotationCollection()
            detections = inference_detector(self.model, image).pred_instances
            detections = detections[detections.scores >= 0.4]

            for det in detections:
                left, top, right, bottom = det.bboxes[0]
                box = dl.Box(top=top, left=left, bottom=bottom, right=right, label=self.labels[det.labels])
                image_annotations.add(annotation_definition=box,
                                      model_info={'name': self.model_entity.name,
                                                  'model_id': self.model_entity.id,
                                                  'confidence': det.scores})
            batch_annotations.append(image_annotations)
            logger.info(f"Found {len(image_annotations)} annotations in image")
        return batch_annotations
