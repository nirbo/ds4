"""Minimal Metal device probe for the Nemotron Mojo kernel experiment."""

from std.gpu.host import DeviceContext


def main() raises:
    with DeviceContext(api="metal") as ctx:
        print("nemotron mojo device:", ctx.name())
        print("nemotron mojo api:", ctx.api())
