__version__ = "0.1"


from .configuration_flow_mirror import FlowmirrorConfig, FlowmirrorDecoderConfig
from .modeling_flow_mirror import (
    FlowmirrorForCausalLM,
    FlowmirrorForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
)

from .dac_wrapper import DACConfig, DACModel
from transformers import AutoConfig, AutoModel


AutoConfig.register("cac", DACConfig)
AutoModel.register(DACConfig, DACModel)
