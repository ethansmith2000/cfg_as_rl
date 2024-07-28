import torch
import torch.nn as nn
import requests
import os
import transformers
import torchvision.transforms as T
import requests


class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            # nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class AestheticClassifier(nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.clip = transformers.CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).to(dtype)
        self.transforms = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize((-1), (2)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.aesthetic_mlp = AestheticMLP().to(device).to(dtype)
    
    def forward(self, image_tensor):
        x = self.transforms(image_tensor)
        x = self.clip.get_image_features(x)
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.aesthetic_mlp(x)
        return x


url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
def download_aesthetic_classifier(path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

def load_aesthetic_classifier(folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "sac+logos+ava1-l14-linearMSE.pth")
    global_rank = int(os.environ.get('RANK', 0))
    if global_rank == 0:
        if not os.path.exists(path):
            download_aesthetic_classifier(path)
    
    model = AestheticClassifier().requires_grad_(False)
    state_dict = torch.load(path, map_location=torch.device('cpu'))

    model.aesthetic_mlp.load_state_dict(state_dict)
    model.eval()
    return model

