from mlx.core.aws.commands import (
    AwsTrainingService,
    GetAwsTrainingStatus,
    ResumeAwsTraining,
    StopAwsTraining,
    SubmitAwsTraining,
    WatchAwsTraining,
)


class SubmitAwsImageClassificationTraining(SubmitAwsTraining):
    pass


class ResumeAwsImageClassificationTraining(ResumeAwsTraining):
    pass


class GetAwsImageClassificationTrainingStatus(GetAwsTrainingStatus):
    pass


class WatchAwsImageClassificationTraining(WatchAwsTraining):
    pass


class StopAwsImageClassificationTraining(StopAwsTraining):
    pass


__all__ = [
    "AwsTrainingService",
    "GetAwsImageClassificationTrainingStatus",
    "ResumeAwsImageClassificationTraining",
    "StopAwsImageClassificationTraining",
    "SubmitAwsImageClassificationTraining",
    "WatchAwsImageClassificationTraining",
]
