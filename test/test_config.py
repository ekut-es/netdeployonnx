#
# Copyright (c) 2024 netdeployonnx contributors.
#
# This file is part of netdeployonx.
# See https://github.com/ekut-es/netdeployonnx for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import pydantic
import pytest
import yaml

from netdeployonnx.config import create_default_config, load_config, parse_host_port


@pytest.fixture(scope="module")
def config_file(tmpdir_factory):
    # Create a temporary directory and file path for testing
    tmpdir = tmpdir_factory.mktemp("configs")
    config_path = tmpdir.join("config.yaml")
    return str(config_path)


def test_create_default_config(config_file):
    # Create default config
    create_default_config(config_file)

    # Check if config file exists and is non-empty
    assert os.path.exists(config_file)
    assert os.path.getsize(config_file) > 0


def test_load_config_default(config_file):
    # Create default config
    create_default_config(config_file)

    # Load config and check its contents
    config = load_config(config_file)
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 28329
    assert config.logging.level == "INFO"
    assert config.logging.file == "log.log"


def test_load_config_custom(tmpdir):
    # Create a custom config file for testing
    custom_config_file = tmpdir.join("custom_config.yaml")
    with open(custom_config_file, "w") as f:
        yaml.dump(
            {
                "server": {"host": "localhost", "port": 5000},
                "logging": {"level": "DEBUG", "file": "custom.log"},
            },
            f,
        )

    # Load custom config and check its contents
    config = load_config(str(custom_config_file))
    assert config.server.host == "localhost"
    assert config.server.port == 5000
    assert config.logging.level == "DEBUG"
    assert config.logging.file == "custom.log"


def test_validate_config_invalid(tmpdir):
    # Create an invalid config file for testing
    invalid_config_file = tmpdir.join("invalid_config.yaml")
    with open(invalid_config_file, "w") as f:
        yaml.dump(
            {
                "server": {
                    "host": "localhost",
                    "port": "invalid_port",
                },  # port should be integer
                "logging": {
                    "level": "INVALID_LEVEL",
                    "file": 123,
                },  # level should be in enum and file should be string
            },
            f,
        )

    # Load invalid config and validate it (expect validation error)
    with pytest.raises(
        pydantic.ValidationError, match="2 validation errors for AppConfig.*"
    ):
        load_config(str(invalid_config_file))


@pytest.mark.parametrize(
    "input_str, expected_host, expected_port",
    [
        ("localhost:5000", "localhost", 5000),
        ("127.0.0.1:8000", "127.0.0.1", 8000),
        ("::1:9000", "::1", 9000),  # IPv6 address with port
        ("example.com:12345", "example.com", 12345),
        ("missing-port", "missing-port", 28329),  # Default values
        (":28329", "0.0.0.0", 28329),  # Default values
    ],
)
def test_parse_host_port(input_str, expected_host, expected_port):
    host, port = parse_host_port(input_str)
    assert expected_host == host
    assert expected_port == port
