"""ASF HTTP policy and bounded retries, isolated from workflow commands."""
import time
from urllib.parse import urlsplit

from mlx.core.exceptions import MLXUserError


def trusted_host(host):
    return bool(host) and (host == "asf.alaska.edu" or host.endswith(".asf.alaska.edu")
                           or host == "earthdata.nasa.gov" or host.endswith(".earthdata.nasa.gov"))


def validate_download_url(url):
    try:
        parsed = urlsplit(url or "")
        port = parsed.port
    except ValueError:
        raise MLXUserError("Malformed ASF download URL; refresh the catalogue metadata.") from None
    if (parsed.scheme != "https" or not trusted_host(parsed.hostname)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or port not in (None, 443)):
        raise MLXUserError("Refusing download: expected an unsigned HTTPS ASF/Earthdata catalogue URL without credentials.")


def create_session(asf, options, *, public=False):
    # A local subclass retains the library's auth/redirect behavior while setting
    # timeouts on every request and refusing credential-bearing external requests.
    class RestrictedSession(asf.ASFSession):
        def send(self, request, **kwargs):
            parsed = urlsplit(request.url)
            trusted = trusted_host(parsed.hostname)
            signed_storage = bool(parsed.hostname and parsed.hostname.endswith(".amazonaws.com"))
            if parsed.scheme != "https" or not (trusted or signed_storage):
                raise MLXUserError("ASF redirected to an unsupported host; no credentials were transmitted.")
            if not trusted:
                request.headers.pop("Authorization", None)
                request.headers.pop("Cookie", None)
            kwargs["timeout"] = options.timeout
            response = super().send(request, **kwargs)
            # Raise before ASF converts all 4xx statuses to authentication errors.
            response.raise_for_status()
            return response

    session = RestrictedSession()
    session.trust_env = False  # No accidental .netrc/proxy credentials on redirects.
    return session


def retry_call(operation, options, *, sleep=time.sleep, description="ASF request"):
    from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout, HTTPError
    for attempt in range(options.max_retries + 1):
        try:
            return operation()
        except (ChunkedEncodingError, ConnectionError, Timeout, HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise MLXUserError("ASF authorization failed (401/403). Check Earthdata credentials and ASF application authorization.") from None
            transient = status is None or status == 429 or status >= 500
            if not transient or attempt == options.max_retries:
                raise MLXUserError(f"{description} failed after {attempt + 1} attempt(s). Check connectivity, rate limits, and service availability.") from None
            delay = min(2 ** attempt, 60)
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 300))
            sleep(delay)
        except MLXUserError:
            raise
        except Exception:
            raise MLXUserError(f"{description} failed. Check ASF availability, input filters, and Earthdata authorization.") from None
