import os

import click

import netdeployonnx
from netdeployonnx.config import DEFAULT_CONFIG_FILE, load_config


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--listen", type=str, help="IP and port to listen on (e.g., 0.0.0.0:5000)"
)
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, writable=True),
    required=False,
)
def server(listen, configfile):
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
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, writable=True),
    required=False,
)
def client(connect, configfile):
    if configfile is None:
        configfile = DEFAULT_CONFIG_FILE
    else:
        configfile = os.path.abspath(configfile)

    config = load_config(configfile)

    if connect:
        host, port = config.parse_hostport(connect)
        config.client.host = host
        config.client.port = port

    netdeployonnx.client.connect(config)


if __name__ == "__main__":
    main()
