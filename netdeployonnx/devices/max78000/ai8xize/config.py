from pydantic import AliasChoices, BaseModel, Field


class AI8XizeConfigLayer(BaseModel):
    """The AI8XizeConfigLayer class is used to define the configuration for a single layer in the AI8XizeConfig class.
    https://github.com/analogdevicesinc/ai8x-synthesis/blob/424b4577c1ac7095fbf1ded442443c00e8d7572f/README.md

    """  # noqa: E501

    name: str = Field(
        default="",
        json_schema_extra={"optional": True},
        description="""`name` assigns a name to the current layer. By default, layers are referenced by their sequence number. Using  `name`, they can be referenced using a string.  * ... *Example:
`name: parallel_1_2`        """,  # noqa: E501
    )

    sequence: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""This key allows overriding of the processing sequence. The default is  `0` for the first layer, or the previous layer’s sequence + 1 for other layers. `sequence` numbers may have gaps. The software will sort layers by their numeric value, with the lowest value first.         """,  # noqa: E501
    )

    next_sequence: str = Field(
        default="",
        json_schema_extra={"optional": True},
        description="""On MAX78000, layers are executed sequentially. On MAX78002, this key can optionally be used to specify the next layer (by either using an integer number or a name).  `stop` terminates execution. Example:
`next_sequence: final`        """,  # noqa: E501
    )

    processors: int = Field(
        description="""`processors` specifies which processors will handle the input data. The processor map must match the number of input channels, and the input data format. For example, in CHW format, processors must be attached to different data memory instances. `processors` is specified as a 64-bit hexadecimal value. Dots (‘.’) and a leading ‘0x’ are ignored. * ... *Example for three processors 0, 4, and 8:
`processors: 0x0000.0000.0000.0111`Example for four processors 0, 1, 2, and 3:
`processors: 0x0000.0000.0000.000f`        """,  # noqa: E501
    )

    output_processors: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`output_processors` specifies which data memory instances and 32-bit word offsets to use for the layer’s output data. When not specified, this key defaults to the next layer’s  `processors`, or, for the last layer, to the global  `output_map`.  `output_processors` is specified as a 64-bit hexadecimal value. Dots (‘.’) and a leading ‘0x’ are ignored.         """,  # noqa: E501
    )

    out_offset: int = Field(
        # default=0, # dont set default value, so we have to specify it
        json_schema_extra={"optional": True},
        description="""`out_offset` specifies the relative offset inside the data memory instance where the output data should be written to. When not specified,  `out_offset` defaults to  `0`. See also  . Example:
`out_offset: 0x2000`        """,  # noqa: E501
    )

    in_offset: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`in_offset` specifies the offset into the data memory instances where the input data should be loaded from. When not specified, this key defaults to the previous layer’s  `out_offset`, or  `0` for the first layer. Example:
`in_offset: 0x2000`        """,  # noqa: E501
    )

    output_width: int = Field(
        default=8,
        json_schema_extra={"optional": True},
        description="""When   using an  `activation` and when  `operation` is    `None`/ `Passthrough`, a layer can output  `32` bits of unclipped data in Q17.14 format. Typically, this is used for the   layer. The default is  `8` bits.  * ... *  `wide=True`  * ... *  `output_width`  * ... *. Example:
`output_width: 32`        """,  # noqa: E501
    )

    data_format: str = Field(
        default="hwc",
        json_schema_extra={"optional": True},
        description="""When specified for the first layer only,  `data_format` can be either  `chw`/ `big` or  `hwc`/ `little`. The default is  `hwc`. Note that the data format interacts with  `processors`, see  .         """,  # noqa: E501
    )

    operation: str = Field(
        default="",
        description="""This key (which can also be specified using  `op`,  `operator`, or  `convolution`) selects a layer’s main operation after the optional input pooling.
When this key is not specified, a warning is displayed, and  `Conv2d` is selected. | Operation                 | Description                                                  |
| :------------------------ | :----------------------------------------------------------- |
|  `Conv1d`                  | 1D convolution over an input composed of several input planes |
|  `Conv2d`                  | 2D convolution over an input composed of several input planes |
|  `ConvTranspose2d`         | 2D transposed convolution (upsampling) over an input composed of several input planes |
|  `None` or  `Passthrough`   | No operation  * ... * |
|  `Linear` or  `FC` or  `MLP` | Linear transformation to the incoming data (fully connected layer) |
|  `Add`                     | Element-wise addition                                        |
|  `Sub`                     | Element-wise subtraction                                     |
|  `BitwiseXor` or  `Xor`     | Element-wise binary XOR                                      |
|  `BitwiseOr` or  `Or`       | Element-wise binary OR                                       |
Element-wise operations default to two operands. This can be changed using the  `operands` key.         """,  # noqa: E501
    )

    eltwise: str = Field(
        default="",
        json_schema_extra={"optional": True},
        description="""Element-wise operations can also be added “in-flight” to  `Conv2d`. In this case, the element-wise operation is specified using the  `eltwise` key.
* ... *Example:
`eltwise: add`        """,  # noqa: E501
    )

    dilation: int = Field(
        default=1,
        json_schema_extra={"optional": True},
        description="""Specifies the dilation for Conv1d/Conv2d operations (default:  `1`).  * ... *Example:
`dilation: 7`        """,  # noqa: E501
    )

    groups: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""By default, Conv1d and Conv2d are configured with  * ... * a “full” convolution. On MAX78002 only, depthwise separable convolutions can be specified using groups=channels (input channels must match the output channels). Example:
`op: conv2d`
`groups: 64`        """,  # noqa: E501
    )

    pool_first: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""When using both pooling and element-wise operations, pooling is performed first by default. Optionally, the element-wise operation can be performed before the pooling operation by setting  `pool_first` to  `False`. Example:
`pool_first: false`        """,  # noqa: E501
    )

    operands: int = Field(
        default=2,
        json_schema_extra={"optional": True},
        description="""For any element-wise  `operation`, this key configures the number of operands from  `2` to  `16` inclusive. The default is  `2`. Example:
`operation: add`
`operands: 4`        """,  # noqa: E501
    )

    activate: str = Field(
        default="None",
        json_schema_extra={"optional": True},
        description="""This key describes whether to activate the layer output (the default is to not activate). When specified, this key must be  `ReLU`,  `Abs` or  `None` (the default).  * ... *`output_shift` can be used for (limited) “linear” activation.         """,  # noqa: E501
    )

    quantization: str | int = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="""This key describes the width of the weight memory in bits and can be  `1`,  `2`,  `4`,  `8`, or  `binary` (MAX78002 only). Specifying a  `quantization` that is smaller than what the weights require results in an error message. The default value is based on the  `weight_bits` field in  `state_dict` read from the quantized checkpoint for the layer. * ... *Example:
`quantization: 4`        """,  # noqa: E501
    )

    output_shift: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""When  `output_width` is 8, the 32-bit intermediate result can be shifted left or right before reduction to 8-bit. The value specified here is cumulative with the value generated from and used by  `quantization`. Note that  `output_shift` is not supported for passthrough layers. The 32-bit intermediate result is multiplied by $2^{totalshift}$, where the total shift count must be within the range $[-15, +15]$, resulting in a factor of $[2^{–15}, 2^{15}]$ or $[0.0000305176$ to $32768]$.
| weight quantization | shift used by quantization | available range for  `output_shift` |
| ------------------- | -------------------------- | ---------------------------------- |
| 8-bit               | 0                          | $[-15, +15]$                       |
| 4-bit               | 4                          | $[-19, +11]$                       |
| 2-bit               | 6                          | $[-21, +9]$                        |
| 1-bit               | 7                          | $[-22, +8]$                        |
Using  `output_shift` can help normalize data, particularly when using small weights. By default,  `output_shift` is generated by the training software, and it is used for batch normalization as well as quantization-aware training. * ... * When using 32-bit wide outputs in the final layer, no hardware shift is performed and  `output_shift` is ignored. Example:
`output_shift: 2`        """,  # noqa: E501
    )

    kernel_size: str = Field(
        default="",
        json_schema_extra={"optional": True},
        description="""* For `Conv2d`, this key must be `3x3` (the default) or `1x1`.
* For `Conv1d`, this key must be `1` through `9`.
* For `ConvTranspose2d`, this key must be `3x3` (the default).
Example:
`kernel_size: 1x1`        """,  # noqa: E501
    )

    stride: int | tuple[int, int] = Field(
        default=1,
        json_schema_extra={"optional": True},
        description="""This key must be  `1` or  `[1, 1]`.         """,
    )

    pad: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`pad` sets the padding for the convolution.         """,
    )

    max_pool: int | tuple[int, int] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""When specified, performs a  `MaxPool` before the convolution. The pooling size can be specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order  `[H, W]`. Example:
`max_pool: 2`        """,  # noqa: E501
    )

    avg_pool: int | tuple[int, int] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""When specified, performs an  `AvgPool` before the convolution. The pooling size can be specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order  `[H, W]`. Example:
`avg_pool: 2`        """,  # noqa: E501
    )

    pool_dilation: int | tuple[int, int] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""When performing a pooling operation  * ... *, this key describes the pool dilation. The pooling dilation can be specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order  `[H, W]`. The default is  `1` or  `[1, 1]`. Example:
`pool_dilation: [2, 1]`        """,  # noqa: E501
    )

    pool_stride: int | tuple[int, int] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""When performing a pooling operation, this key describes the pool stride. The pooling stride can be specified as an integer (when the value is identical for both dimensions, or for 1D convolutions), or as two values in order  `[H, W]`, where both values must be identical. The default is  `1` or  `[1, 1]`. Example:
`pool_stride: [2, 2]`        """,  # noqa: E501
    )

    in_channels: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`in_channels` specifies the number of channels of the input data. This is usually automatically computed based on the weights file and the layer sequence. This key allows overriding of the number of channels. See also:  `in_dim`. Example:
`in_channels: 8`        """,  # noqa: E501
    )

    in_dim: int | tuple[int, int] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`in_dim` specifies the dimensions of the input data. This is usually automatically computed based on the output of the previous layer or the layer(s) referenced by  `in_sequences`. This key allows overriding of the automatically calculated dimensions.  `in_dim` must be used when changing from 1D to 2D data or vice versa. 1D dimensions can be specified as a tuple  `[L, 1]` or as an integer  `L`. See also:  `in_channels`,  `in_crop`. Examples:
`in_dim: [64, 64]`
`in_dim: 32`        """,  # noqa: E501
    )

    in_crop: int | tuple[int, int] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`in_crop` specifies a number of rows (2D) or data bytes (1D) to skip (crop) when using the previous layer's output as input. By also adjusting  `in_offset`, this provides the means to crop the top/bottom of an image or the beginning/end of 1D data. The dimensions and offsets are validated to match (minus the crop amount). See also:  `in_dim`,  `in_offset`. Example (1D cropping):
`# Output data had L=512`
`in_offset: 0x000c  # Skip 3 (x4 processors) at beginning`
`in_dim: 506  # Target length = 506`
`in_crop: [3, 3]  # Crop 3 at the beginning, 3 at the end`        """,  # noqa: E501
    )
    in_sequences: tuple[int, int] | tuple[str, str] = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""By default, a layer’s input is the output of the previous layer.  `in_sequences` can be used to point to the output of one or more arbitrary previous layers, for example when processing the same data using two different kernel sizes, or when combining the outputs of several prior layers.  `in_sequences` can be specified as a single item (for a single input) or as a list (for multiple inputs). Both layer sequence numbers as well as layer names can be used. As a special case,  `-1` or  `input` refer to the input data. The  `in_offset` and  `out_offset` must be set to match the specified sequence.
`in_sequences` is used to specify the inputs for a multi-operand element-wise operator (for example,  `add`). The output processors for all arguments of the sequence must match. `in_sequences` can also be used to specify concatenation similar to  `torch.cat()`. The output processors must be adjacent for all sequence arguments, and arguments listed earlier must use lower processor numbers than arguments listed later. The order of arguments of  `in_sequences` must match the model. The following code shows an example  `forward` function for a model that concatenates two values: In this case,  `in_sequences` would be  `[1, 0]` since  `y` (the output of layer 1) precedes  `x` (the output of layer 0) in the  `torch.cat()` statement. Examples:
`in_sequences: [2, 3]`
`in_sequences: [expand_1x1, expand_3x3]`""",  # noqa: E501
    )

    in_skip: bool = Field(
        default=False,
        validation_alias=AliasChoices("in_skip", "read_gap"),
        serialization_alias="in_skip",
    )

    out_channels: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`out_channels` specifies the number of channels of the output data. This is usually automatically computed based on the weights and layer sequence. This key allows overriding the number of output channels. Example:
`out_channels: 8`        """,  # noqa: E501
    )

    streaming: bool = Field(
        default=False,
        json_schema_extra={"optional": True},
        description="""`streaming` specifies that the layer is using  . This is necessary when the input data dimensions exceed the available data memory. When enabling  `streaming`, all prior layers have to enable  `streaming` as well.  `streaming` can be enabled for up to 8 layers. Example:
`streaming: true`        """,  # noqa: E501
    )

    flatten: bool = Field(
        default=False,
        json_schema_extra={"optional": True},
        description="""`flatten` specifies that 2D input data should be transformed to 1D data for use by a  `Linear` layer.  * ... *Example:
`flatten: true`        """,  # noqa: E501
    )

    write_gap: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""`write_gap` specifies the number of channels that should be skipped during write operations (this value is multiplied with the output multi-pass, i.e., write every  * ... *th word where  * ... *). This creates an interleaved output that can be used as the input for subsequent layers that use an element-wise operation, or to concatenate multiple inputs to form data with more than 64 channels. The default is 0. Set  `write_gap` to  `1` to produce output for a subsequent two-input element-wise operation. Example:
`write_gap: 1`        """,  # noqa: E501
    )

    read_gap: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""On MAX78002 only, when multi-pass is not used (i.e., typically 64 input channels or less), data can be fetched while skipping bytes. This allows a layer to directly consume data produced by a layer with  `write_gap` without the need for intermediate layers. The default is 0. Example:
`read_gap: 1`        """,  # noqa: E501
    )

    bias_group: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""For layers that use a bias, this key can specify one or more bias memories that should be used. By default, the software uses a “Fit First Descending (FFD)” allocation algorithm that considers the largest bias lengths first, and then the layer number, and places each bias in the available quadrant with the most available space, descending to the smallest bias length. “Available quadrants” is the complete list of quadrants used by the network (in any layer).  `bias_group` must reference one or more of these available quadrants. `bias_group` can be a list of integers or a single integer. Example:
`bias_group: 0`        """,  # noqa: E501
    )

    output: bool = Field(
        default=False,
        json_schema_extra={"optional": True},
        description="""By default, the final layer is used as the output layer. Output layers are checked using the known-answer test, and they are copied from hardware memory when  `cnn_unload()` is called. The tool also checks that output layer data isn’t overwritten by any later layers. When specifying  `output: true`, any layer (or a combination of layers) can be used as an output layer.
* ... * When  `unload:` is used, output layers are not used for generating  `cnn_unload()`. Example:
`output: true`        """,  # noqa: E501
    )

    weight_source: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="""Certain networks share weights between layers. The tools automatically deduplicate weights and bias values (unless  `--no-deduplicate-weights` is specified). When the checkpoint file does  * ... * contain duplicated weights,  `weight_source: layer` is needed in the YAML configuration for the layer(s) that reuse(s) weights.  `layer` can be specified as layer number or name. Example:
`weight_source: conv1p3`        """,  # noqa: E501
    )

    buffer_shift: int = Field(
        default=0,
        description="""The buffer is shifted  `n` places using  `buffer_shift: n`.  `in_offset` and  `in_dim` are required. Example:         """,  # noqa: E501
    )

    buffer_insert: int = Field(
        default=0,
        description="""New data is added using  `buffer_insert: n`. Example:         """,  # noqa: E501
    )

    convolution: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    conv_groups: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    in_channel_skip: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    activation: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    op: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    operator: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    snoop_sequence: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    simulated_sequence: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    bypass: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    bias_quadrant: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    calcx4: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    readahead: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    output_pad: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )
    tcalc: None = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="defined but not documented",  # noqa: E501
    )


class AI8XizeConfigUnloadItem(BaseModel):
    processors: int = Field(
        description="The processors data is copied from",
    )
    channels: int = Field(
        default=None,
        description="Data channel count",
    )
    dim: int = Field(
        default=None,
        description="Data dimensions (1D or 2D)",
    )
    offset: int = Field(
        default=None,
        description="Data source offset",
    )
    width: int = Field(
        default=8,
        json_schema_extra={"optional": True},
        description="either 8 or 32",
    )
    write_gap: int = Field(
        default=0,
        json_schema_extra={"optional": True},
        description="Gap between data words",
    )


class AI8XizeConfig(BaseModel):
    """The AI8XizeConfig class is used to define the configuration for the AI8X izer tool.
    https://github.com/analogdevicesinc/ai8x-synthesis/blob/424b4577c1ac7095fbf1ded442443c00e8d7572f/izer/yamlcfg.py
    """  # noqa: E501

    arch: str = Field(
        description="arch specifies the network architecture, for example ai84net5. This key is matched against the architecture embedded in the checkpoint file.",  # noqa: E501
    )
    bias: list[int] = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="The bias configuration is only used for test data. To use bias with trained networks, use the bias parameter in PyTorch’s nn.Module.Conv2d() function. The converter tool will then automatically add bias parameters as needed.",  # noqa: E501
    )
    dataset: str = Field(
        description="dataset configures the data set for the network. Data sets are for example mnist, fashionmnist, and cifar-10. This key is descriptive only, it does not configure input or output dimensions or channel count.",  # noqa: E501
    )
    output_map: int = Field(
        default=0x0000000000000FF0,
        json_schema_extra={"optional": True},
        description="The global output_map, if specified, overrides the memory instances where the last layer outputs its results. If not specified, this will be either the output_processors specified for the last layer, or, if that key does not exist, default to the number of processors needed for the output channels, starting at 0. Please also see output_processors.",  # noqa: E501
    )
    unload: list[AI8XizeConfigUnloadItem] = Field(
        default=None,
        json_schema_extra={"optional": True},
        description="By default, the function cnn_unload() is automatically generated from the network’s output layers (typically, the final layer). unload can be used to override the shape and sequence of data copied from the CNN. It is possible to specify multiple unload list items, and they will be processed in the order they are given.",  # noqa: E501
    )
    data_buffer: object = Field(
        default=0,
        description="""The data buffer is allocated and named using the global  `data_buffer` configuration key.  `processor`,   `dim` (1D or 2D),  `channels`, and  `offset` must be defined. Example:         """,  # noqa: E501
    )
    layers: list[AI8XizeConfigLayer] = Field(
        description="Layers",
    )
