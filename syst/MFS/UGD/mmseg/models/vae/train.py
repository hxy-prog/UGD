from torch.utils.data import DataLoader
from dataset import ManipulationDataset
from loss import TruForLoss
import torch
import torch.nn.functional as F
from model import EmbeddingEncoderDecoder
from tqdm import tqdm
import cv2
epochs=5
device='cuda'
mydataset=ManipulationDataset()
traindataloader= DataLoader(mydataset,
                        batch_size=8,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)


criterion = TruForLoss(weights=torch.tensor([0.5,2.5]).to(device), ignore_index=-1)
vaemodel=EmbeddingEncoderDecoder()
vaemodel=vaemodel.to(device)
total_params = 0
for name, param in vaemodel.named_parameters():
    param_count = param.numel()  # 计算参数数量
    total_params += param_count
print(f"Total parameters in the model: {total_params}")

optimizer = torch.optim.Adam(vaemodel.parameters())
for epoch in range(0,epochs):
    vaemodel.train()
    pbar = tqdm(traindataloader, desc='Training Epoch {}/{}'.format(epoch + 1, epochs), unit='steps')
    for step, masks in enumerate(pbar):
        masks=masks.to(device)
        optimizer.zero_grad()
        pred = vaemodel(masks)
        
        recon_loss = criterion(pred, masks)
        
        loss = recon_loss 
        loss.backward()
        optimizer.step()

    vaemodel.eval()
    with torch.no_grad():
        valmask=cv2.imread('0.png',cv2.IMREAD_GRAYSCALE)/255.0
        valmask=torch.tensor((valmask>0.1).astype(int))
        valmask=valmask.to(device)
        valmask=valmask.unsqueeze(0)
        pred=vaemodel(valmask)
        pred=torch.softmax(pred,dim=1)[:,1,:,:]
        print(pred.shape)
        cv2.imwrite('1.png',pred[0].cpu().numpy()*255)
torch.save(vaemodel.state_dict(), 'vae_parameters.pth')
print("Model parameters saved.")
        