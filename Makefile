PYTHON = python3
PROTOC = protoc
PYTHON_GRPC = $(PYTHON) -m grpc_tools.protoc
PROTO_PATH = .
DEST_FOLDER = common
PROTO_FILE = $(PROTO_PATH)/device.proto
OUTPUT_DIR = ./netdeployonnx/$(DEST_FOLDER)
GRPC_PY_FILE = $(OUTPUT_DIR)/device_pb2_grpc.py

all: $(OUTPUT_DIR)/device_pb2.py $(OUTPUT_DIR)/device_pb2_grpc.py

device_pb2.py: $(PROTO_PATH)/$(PROTO_FILE)
	@echo "Generating Python protobuf files..."
	@cd $(PROTO_PATH) && $(PROTOC) --python_out=$(DEST_FOLDER) $(PROTO_FILE)

device_pb2_grpc.py: $(PROTO_PATH)/$(PROTO_FILE)
	@echo "Generating Python gRPC files..."
	@cd $(PROTO_PATH) && $(PYTHON_GRPC) --proto_path=common --python_out=$(DEST_FOLDER) --grpc_python_out=$(DEST_FOLDER) $(PROTO_FILE)

$(OUTPUT_DIR)/device_pb2.py $(OUTPUT_DIR)/device_pb2_grpc.py: $(PROTO_FILE)
	@mkdir -p $(OUTPUT_DIR)
	@$(PYTHON_GRPC) -I$(PROTO_PATH) --python_out=$(OUTPUT_DIR) --grpc_python_out=$(OUTPUT_DIR) $(PROTO_FILE)
# Patch the import statement in device_pb2_grpc.py
	@sed -i.bak 's/import device_pb2 as device__pb2/import netdeployonnx.common.device_pb2 as device__pb2/g' $(GRPC_PY_FILE)

clean:
	@echo "Cleaning..."
	rm -f $(OUTPUT_DIR)/device_pb2.py $(OUTPUT_DIR)/device_pb2_grpc.py