from torch.utils.data import DataLoader
from dataset import ManipulationDataset
from loss import TruForLoss
import torch
from model import EmbeddingEncoderDecoder
from tqdm import tqdm
import cv2
device='cuda'
vaemodel=EmbeddingEncoderDecoder()
vaemodel.load_state_dict(torch.load('vae_parameters.pth',map_location='cpu'))
vaemodel.to(device)
vaemodel.eval()
with torch.no_grad():
    valmask=cv2.imread('4.png',cv2.IMREAD_GRAYSCALE)/255.0
    valmask=torch.tensor((valmask>0.1).astype(int))
    valmask=valmask.to(device)
    valmask=valmask.unsqueeze(0)
    pred=vaemodel(valmask)
    pred=torch.softmax(pred,dim=1)[:,1,:,:]
    print(pred.shape)
    cv2.imwrite('2.png',pred[0].cpu().numpy()*255)