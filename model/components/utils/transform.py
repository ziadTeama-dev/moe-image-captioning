import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


# ============ 1) TRAIN TRANSFORMS (Augmentations) ============
train_transform = transforms.Compose([
  
    transforms.Resize(
        size=232,
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.RandomCrop(224),
    
    # 2. Augmentations 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10), 
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), 
    
    # 3. Augmentations (Color/Lighting)
    transforms.RandomApply(p=.3, transforms=
                           [        
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.2, 
        hue=0.1
    )],),

    
#    Tensor
    transforms.ToTensor(),
    
    
    # 6. Normalization 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
