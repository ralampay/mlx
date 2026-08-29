from mlx.core.aws.commands import GetAwsTrainingStatus, ResumeAwsTraining, SubmitAwsTraining, WatchAwsTraining


class SubmitAwsVideoAnomalyBatchTraining(SubmitAwsTraining):
    pass


class ResumeAwsVideoAnomalyBatchTraining(ResumeAwsTraining):
    pass


class GetAwsVideoAnomalyBatchStatus(GetAwsTrainingStatus):
    pass


class WatchAwsVideoAnomalyBatchTraining(WatchAwsTraining):
    pass


__all__ = [
    "GetAwsVideoAnomalyBatchStatus",
    "ResumeAwsVideoAnomalyBatchTraining",
    "SubmitAwsVideoAnomalyBatchTraining",
    "WatchAwsVideoAnomalyBatchTraining",
]
