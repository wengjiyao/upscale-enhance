
import torch
import coremltools as ct
from basicsr.archs.rrdbnet_arch import RRDBNet
import os

# 1. Load your PyTorch model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = os.path.expanduser("~/.cache/realesrgan/RealESRGAN_x4plus.pth")
loadnet = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(loadnet['params_ema'], strict=True)
model.eval()

# 2. Trace the model with a sample input
example_input = torch.rand(1, 3, 64, 64) # Example input size
traced_model = torch.jit.trace(model, example_input)

# 3. Convert the model to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, ct.RangeDim(1, None), ct.RangeDim(1, None)))],
    outputs=[ct.ImageType(name="output")],
    convert_to="mlprogram"
)

# 4. Save the Core ML model
mlmodel.save("RealESRGAN_x4plus.mlmodel")

print("Successfully converted the 4x model to RealESRGAN_x4plus.mlmodel")
