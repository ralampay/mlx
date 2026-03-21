from mlx.features.object_detection.ultralytics.training import train_object_detection


class TrainObjDetect:
    def __init__(self, config):
        self.config = config

    def execute(self):
        return train_object_detection(self.config)
