from __future__ import annotations

from time import sleep
from typing import Callable, Optional, Protocol

from mlx.core.exceptions import MLXAbort, MLXUserError

from mlx.modes.object_detection.aws.models import (
    AwsInfrastructure,
    AwsTrainingStatus,
    AwsTrainingStopResult,
    AwsTrainingSubmission,
)


class AwsTrainingService(Protocol):
    def prepare_infrastructure(self) -> AwsInfrastructure:
        ...

    def submit(
        self,
        infrastructure: AwsInfrastructure,
        **kwargs,
    ) -> AwsTrainingSubmission:
        ...

    def resume(self, old_job_name: str) -> AwsTrainingSubmission:
        ...

    def status(self, job_name: str) -> AwsTrainingStatus:
        ...

    def stop(self, job_name: str, *, config_path: str) -> AwsTrainingStopResult:
        ...


class SubmitAwsObjectDetectionTraining:
    def __init__(self, service: AwsTrainingService) -> None:
        self.service = service

    def execute(self) -> AwsTrainingSubmission:
        infrastructure = self.service.prepare_infrastructure()
        return self.service.submit(infrastructure)


class ResumeAwsObjectDetectionTraining:
    def __init__(self, service: AwsTrainingService, job_name: str) -> None:
        self.service = service
        self.job_name = job_name

    def execute(self) -> AwsTrainingSubmission:
        return self.service.resume(self.job_name)


class GetAwsObjectDetectionTrainingStatus:
    def __init__(self, service: AwsTrainingService, job_name: str) -> None:
        self.service = service
        self.job_name = job_name

    def execute(self) -> AwsTrainingStatus:
        return self.service.status(self.job_name)


class WatchAwsObjectDetectionTraining:
    def __init__(
        self,
        service: AwsTrainingService,
        job_name: str,
        *,
        poll_interval: float = 30.0,
        on_status: Optional[Callable[[AwsTrainingStatus], None]] = None,
        wait: Callable[[float], None] = sleep,
    ) -> None:
        self.service = service
        self.job_name = job_name
        self.poll_interval = poll_interval
        self.on_status = on_status
        self.wait = wait

    def execute(self) -> AwsTrainingStatus:
        if self.poll_interval <= 0:
            raise MLXUserError("--poll-interval must be greater than zero.")
        try:
            while True:
                status = self.service.status(self.job_name)
                if self.on_status is not None:
                    self.on_status(status)
                if status.terminal:
                    return status
                self.wait(self.poll_interval)
        except KeyboardInterrupt as exc:
            raise MLXAbort() from exc


class StopAwsObjectDetectionTraining:
    def __init__(
        self,
        service: AwsTrainingService,
        job_name: str,
        *,
        config_path: str,
    ) -> None:
        self.service = service
        self.job_name = job_name
        self.config_path = config_path

    def execute(self) -> AwsTrainingStopResult:
        return self.service.stop(self.job_name, config_path=self.config_path)
