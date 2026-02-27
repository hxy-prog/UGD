from DnCNN import make_net
import torch
import torchvision.transforms.functional as TF
import cv2
num_levels = 17
out_channel = 1
dncnn = make_net(3, kernels=[3, ] * num_levels,
                features=[64, ] * (num_levels - 1) + [out_channel],
                bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                acts=['relu', ] * (num_levels - 1) + ['linear', ],
                dilats=[1, ] * num_levels,
                bn_momentum=0.1, padding=1)

checkpoint=torch.load('/home/hexingyang/DDP/segmentation/ckpts/trufor.pth.tar',map_location='cpu')
newcheck=dict()
for k,v in checkpoint['state_dict'].items():
    if k.startswith('dncnn'):
        newcheck[k.replace('dncnn.','')]=v
dncnn.load_state_dict(newcheck,strict=True)
with torch.no_grad():
    dncnn.eval()
    device='cuda:3'
    dncnn.to(device)
    img=cv2.cvtColor(cv2.imread('tampered1.png'), cv2.COLOR_BGR2RGB).transpose(2,0,1)/255.0

    img=torch.tensor(img).to(torch.float).unsqueeze(0)
    # img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # img=img*torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)+torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    img=img.to(device)
    re=dncnn(img)
    re = torch.tile(re, (3, 1, 1))
cv2.imwrite('re.jpg',re[0][0].cpu().numpy()*255)