from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mlx.core.exceptions import MLXUserError


class AwsSessionConfig(Protocol):
    profile: str | None
    region: str | None


@dataclass(frozen=True)
class AwsClientBundle:
    session: Any
    region: str
    s3: Any
    ecr: Any
    iam: Any
    sts: Any
    sagemaker: Any
    cloudwatch: Any
    client_error: type[Exception]
    boto_error: type[Exception]

    @classmethod
    def create(cls, config: AwsSessionConfig) -> "AwsClientBundle":
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise MLXUserError(
                "AWS training requires boto3. Install MLX with the 'aws' extra."
            ) from exc
        try:
            session = boto3.Session(profile_name=config.profile, region_name=config.region)
        except (BotoCoreError, ValueError) as exc:
            raise MLXUserError(f"Unable to initialize the AWS session: {exc}") from exc
        region = session.region_name
        if not region:
            raise MLXUserError(
                "No AWS region is configured. Set aws.region or configure a default AWS region."
            )
        return cls(
            session=session,
            region=region,
            s3=session.client("s3"),
            ecr=session.client("ecr"),
            iam=session.client("iam"),
            sts=session.client("sts"),
            sagemaker=session.client("sagemaker"),
            cloudwatch=session.client("cloudwatch"),
            client_error=ClientError,
            boto_error=BotoCoreError,
        )


__all__ = ["AwsClientBundle", "AwsSessionConfig"]
