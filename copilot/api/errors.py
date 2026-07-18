class ManifestNotConfiguredError(Exception):
    pass


class ManifestFileNotFoundError(Exception):
    pass


class InvalidManifestError(Exception):
    pass


class DatabaseNotConfiguredError(RuntimeError):
    pass