import cv2
import torch
import numpy as np 
from PIL import Image  
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
 
image = Image.open("0.png")  
image = image.convert('RGB')  
  
tensor = torch.from_numpy(np.array(image)).float().div(255.0)
tensor = tensor.unsqueeze(0)  
tensor = tensor.permute(0, 3, 1, 2)  
latents = vae.encode(tensor,return_dict=False)
latents = latents[0].mean
 
def quantize(latents):
  quantized_latents = (latents / (255 * 0.18215) + 0.5).clamp(0,1)
  quantized = quantized_latents.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
  quantized = (quantized * 255.0 + 0.5).astype(np.uint8)
  return quantized
 
zx = quantize(latents)   
imageOut = cv2.cvtColor(zx, cv2.COLOR_RGBA2BGRA)  
cv2.imwrite('2.png', imageOut)


decodeOut = vae.decode(latents,return_dict=False)
decodeOut  = decodeOut[0].permute(2, 3, 1, 0).squeeze()
numpy_img = decodeOut.detach().numpy()
numpy_img = numpy_img*255
numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('3.png', numpy_img)