# from torchvision import transforms


# IMG_SIZE = 224

# IMAGENET_MEAN = [
#     0.485,
#     0.456,
#     0.406
# ]

# IMAGENET_STD = [
#     0.229,
#     0.224,
#     0.225
# ]


# train_transform = transforms.Compose([
#     transforms.Resize(
#         (IMG_SIZE, IMG_SIZE)
#     ),

#     transforms.RandomHorizontalFlip(
#         p=0.5
#     ),

#     transforms.RandomRotation(
#         degrees=10
#     ),

#     transforms.ColorJitter(
#         brightness=0.15,
#         contrast=0.15,
#         saturation=0.15
#     ),

#     transforms.ToTensor(),

#     transforms.Normalize(
#         mean=IMAGENET_MEAN,
#         std=IMAGENET_STD
#     )
# ])


# valid_transform = transforms.Compose([
#     transforms.Resize(
#         (IMG_SIZE, IMG_SIZE)
#     ),

#     transforms.ToTensor(),

#     transforms.Normalize(
#         mean=IMAGENET_MEAN,
#         std=IMAGENET_STD
#     )
# ])


# def get_train_transform():

#     return transforms.Compose([

#         transforms.Resize((224, 224)),

#         transforms.RandomHorizontalFlip(),

#         transforms.RandomRotation(10),

#         transforms.ToTensor(),

#         transforms.Normalize(

#             mean=[0.485, 0.456, 0.406],

#             std=[0.229, 0.224, 0.225]

#         )

#     ])

# def get_valid_transform():

#     return transforms.Compose([

#         transforms.Resize((224, 224)),

#         transforms.ToTensor(),

#         transforms.Normalize(

#             mean=[0.485, 0.456, 0.406],

#             std=[0.229, 0.224, 0.225]

#         )

#     ])


from torchvision import transforms


def get_train_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_valid_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])