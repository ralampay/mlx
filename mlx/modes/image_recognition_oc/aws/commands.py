from mlx.core.aws.commands import (
    GetAwsTrainingStatus,
    ResumeAwsTraining,
    StopAwsTraining,
    SubmitAwsTraining,
    WatchAwsTraining,
)


class SubmitAwsImageOneClass(SubmitAwsTraining):
    pass


class ResumeAwsImageOneClass(ResumeAwsTraining):
    pass


class GetAwsImageOneClassStatus(GetAwsTrainingStatus):
    pass


class WatchAwsImageOneClass(WatchAwsTraining):
    pass


class StopAwsImageOneClass(StopAwsTraining):
    pass


__all__ = [
    "GetAwsImageOneClassStatus",
    "ResumeAwsImageOneClass",
    "StopAwsImageOneClass",
    "SubmitAwsImageOneClass",
    "WatchAwsImageOneClass",
]
