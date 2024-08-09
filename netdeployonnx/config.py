import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_FILE = "netdeployonnx.yaml"


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(
        default=28329, ge=1024, le=65535, description="Server port number"
    )


class ClientConfig(BaseModel):
    host: str = Field(default="localhost", description="Client host address")
    port: int = Field(
        default=28329, ge=1024, le=65535, description="Client port number"
    )


class DeviceConfig(BaseModel):
    name: str = Field(default="MAX78000", description="Device name")
    class_name: str = Field(default="DummyDevice")
    communication_port: str | None = Field(
        default="/dev/ttyACM0", description="Communication port"
    )
    energy_port: str | None = Field(
        default="/dev/ttyACM1", description="Energy Info port"
    )


class LoggingConfig(BaseModel):
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    file: str = Field(default="log.log", description="Logging file name")


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    client: ClientConfig = ClientConfig()
    logging: LoggingConfig = LoggingConfig()
    devices: list[DeviceConfig] = [DeviceConfig(name="Dummy")]

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        config_dict = {}
        for field_name, field in self.model_fields.items():
            field_value = getattr(self, field_name)
            if isinstance(field_value, list):
                config_dict[field_name] = [item.model_dump() for item in field_value]
            else:
                config_dict[field_name] = field_value.model_dump()
        return config_dict


def parse_host_port(
    arg: str, default_host: str = "0.0.0.0", default_port: int = 28329
) -> tuple[str, int]:
    # Split the argument into host and port parts
    import ipaddress

    parts = arg.split(":")
    if len(parts) > 2:
        host = str(ipaddress.ip_address(":".join(parts[:-1]) or default_host))
        port = int(parts[-1]) or default_port
    elif len(parts) == 2:
        try:
            host = str(ipaddress.ip_address(parts[0] or default_host)) or parts[0]
        except ValueError:  # TODO: check if a value error really comes up
            host = parts[0]
        port = int(parts[-1]) or default_port
    else:
        host = ":".join(parts) or default_host
        port = default_port

    return host, port


def load_config(config_file: Path) -> AppConfig:
    if not os.path.exists(config_file):
        create_default_config(config_file)

    config = AppConfig().to_dict()
    with open(config_file) as f:
        config_update = yaml.safe_load(f)
        if config_update is None:
            config_update = create_default_config(config_file)
        config.update(config_update)

    return AppConfig.from_dict(config)


def create_default_config(config_file: Path) -> AppConfig:
    config = AppConfig()
    default_config = config.to_dict()
    with open(config_file, "w") as f:
        yaml.dump(default_config, f)
    return default_config


# Example usage:
if __name__ == "__main__":
    config_file = "config.yaml"
    config = load_config(config_file)
    print("Loaded configuration:", config)
