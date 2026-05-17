import torch, numpy as np
from torchvision import transforms, datasets
import numpy as np

from torch.utils.data import DataLoader
from poison_eval import SimpleCNN, CIFAR_MEAN, CIFAR_STD  # import model/consts

EXP = './poison_labelflip_10'
CKPT = f'{EXP}/chk_ep30.pth'   # pick saved model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ck = torch.load(CKPT, map_location='cpu')
model = SimpleCNN(num_classes=10)
model.load_state_dict(ck['model_state'])
model = model.to(device).eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
test = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
loader = DataLoader(test, batch_size=64, shuffle=False)

conf = np.zeros((10,10), dtype=int)  # rows=gt, cols=pred
total=0; correct=0
with torch.no_grad():
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(1)
        for gt,p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            conf[gt,p] += 1
            total += 1
        correct += (preds==y).sum().item()
print("Clean acc:", correct/total)
# show top confusions
import numpy as np
pairs=[]
for i in range(10):
    for j in range(10):
        if i!=j:
            pairs.append(((i,j), conf[i,j]))
pairs = sorted(pairs, key=lambda z: -z[1])
print("Top confusions (gt->pred):", pairs[:10])


