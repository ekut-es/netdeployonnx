import asyncio
import logging
import os
import sys
from pathlib import Path

import click

import netdeployonnx
from netdeployonnx.config import DEFAULT_CONFIG_FILE, load_config


@click.group()
def main():
    pass


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


@main.command()
@click.option(
    "--listen", type=str, help="IP and port to listen on (e.g., 0.0.0.0:5000)"
)
@click.option(
    "--configfile",
    type=click.Path(exists=True, dir_okay=False, writable=True),
    required=False,
)
@click.option(
    "--log_level",
    type=click.Choice(LOG_LEVELS.keys(), case_sensitive=False),
    default="INFO",
    required=False,
)
def server(listen, configfile, log_level):
    logging.basicConfig(level=LOG_LEVELS[log_level.upper()])
    if configfile is None:
        configfile = DEFAULT_CONFIG_FILE
    else:
        configfile = os.path.abspath(configfile)

    config = load_config(configfile)

    if listen:
        host, port = config.parse_hostport(listen)
        config.server.host = host
        config.server.port = port

    netdeployonnx.server.listen(config)


@main.command()
@click.option(
    "--connect", type=str, help="IP and port to connect to (e.g., 127.0.0.1:5000)"
)
@click.option(
    "--experiments",
    is_flag=True,
    help="run experiments instead of single deploy",
)
@click.option(
    "--configfile",
    type=click.Path(exists=True, dir_okay=False, writable=True),
    help="config file, like netdeployonnx.yaml",
    required=False,
)
@click.option(
    "--no-flash",
    is_flag=True,
    help="allows to annotate the onnx file with metadataprop __reflash",
)
@click.argument(
    "networkfile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(__file__).parent.parent / "test" / "data" / "cifar10_short.onnx",
    required=False,
    # help="just a .onnx file",
)
@click.option(
    "--log_level",
    type=click.Choice(LOG_LEVELS.keys(), case_sensitive=False),
    default="INFO",
    required=False,
)
def client(
    connect: str,
    configfile: str,
    experiments: bool,
    networkfile: Path,
    no_flash: bool,
    log_level: str,
):
    logging.basicConfig(level=LOG_LEVELS[log_level.upper()])
    if configfile is None:
        configfile = DEFAULT_CONFIG_FILE
    else:
        configfile = os.path.abspath(configfile)

    config = load_config(configfile)

    if connect:
        host, port = config.parse_hostport(connect)
        config.client.host = host
        config.client.port = port

    netdeployonnx.client.connect(
        config,
        networkfile,
        run_experiments=experiments,
        no_flash=no_flash,
    )


@main.command()
def cli(tty="/dev/ttyUSB0", debug=False, timeout=5):
    from scripts import embedded_cli

    asyncio.run(embedded_cli.asyncmain(sys.argv, tty, debug=debug, timeout=timeout))


if __name__ == "__main__":
    main()
