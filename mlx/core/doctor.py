"""Environment configuration report without provider initialization."""

from collections.abc import Mapping


class InspectEnvironment:
    """Report configuration presence, never credential values or validity."""

    def __init__(self, environment: Mapping[str, str]):
        self.environment = environment

    def execute(self) -> dict:
        optional_variables = (
            ("AWS_ACCESS_KEY_ID", "AWS access key when using environment credentials."),
            ("AWS_SECRET_ACCESS_KEY", "AWS secret key paired with AWS_ACCESS_KEY_ID."),
            ("AWS_SESSION_TOKEN", "Required with temporary AWS environment credentials."),
            ("AWS_PROFILE", "AWS profile; explicit --profile takes precedence."),
            ("AWS_DEFAULT_REGION", "AWS region fallback when not configured explicitly."),
            ("EARTHDATA_TOKEN", "Preferred authentication for protected SAR downloads; not needed for search."),
            ("EARTHDATA_USERNAME", "Earthdata username fallback, paired with EARTHDATA_PASSWORD."),
            ("EARTHDATA_PASSWORD", "Earthdata password fallback; .netrc is also supported."),
            ("MLX_SAR_OUTPUT_DIR", "SAR output directory; CLI --output takes precedence."),
            ("MLX_SAR_CACHE_DIR", "Verified SAR download cache directory."),
            ("MLX_SAR_DOWNLOAD_PROCESSES", "Concurrent SAR downloads; default 1."),
            ("MLX_SAR_REQUEST_TIMEOUT", "SAR request timeout in seconds; default 60."),
            ("MLX_SAR_MAX_RETRIES", "Transient SAR request retries; default 3."),
        )
        return {
            "required_variables": [],
            "summary": "No environment variables are required for ordinary local MLX workflows.",
            "notes": (
                "AWS/S3 workflows need credentials and a region, which can also come from "
                "AWS configuration files or workload roles. Unset variables are not errors. "
                "Protected SAR downloads need Earthdata authentication; public searches do not. "
                "Status indicates non-empty values only; credentials are not validated."
            ),
            "optional_variables": [
                {
                    "name": name,
                    "status": "set" if self.environment.get(name) else "unset",
                    "description": description,
                }
                for name, description in optional_variables
            ],
        }
