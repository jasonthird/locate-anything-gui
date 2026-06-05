"""Image-only compatibility stub for LocateAnything on macOS.

The upstream processor imports decord for video helpers. Apple Silicon does not
have a decord 0.6.0 wheel, and this test harness only sends images.
"""


class VideoReader:  # pragma: no cover - only here to fail clearly on video use
    def __init__(self, *args, **kwargs):
        raise ImportError("decord is unavailable on Apple Silicon in this harness; video input is not supported.")


def cpu(index=0):  # pragma: no cover
    return ("cpu", index)
