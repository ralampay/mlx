"""Explicit Earthdata authentication; secrets never enter workflow results."""
from dataclasses import dataclass, field
from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True, repr=False)
class EarthdataCredentials:
    token: str | None = field(default=None, repr=False)
    username: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)


def authenticate(session, credentials, netrc_lookup=None):
    try:
        if credentials.token:
            return session.auth_with_token(credentials.token)
        if credentials.username and credentials.password:
            return session.auth_with_creds(credentials.username, credentials.password)
        if netrc_lookup is None:
            from requests.utils import get_netrc_auth
            netrc_lookup = get_netrc_auth
        pair = netrc_lookup("https://urs.earthdata.nasa.gov", raise_errors=True)
        if pair and all(pair):
            return session.auth_with_creds(*pair)
    except Exception:
        # Provider exceptions may contain request bodies. Never expose their text.
        raise MLXUserError("Earthdata authentication failed. Check token expiry, credentials, and .netrc permissions.") from None
    raise MLXUserError("Protected downloads require EARTHDATA_TOKEN, both EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or an Earthdata .netrc entry.")
