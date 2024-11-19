# import paramiko
# from scp import SCPClient
# import os
# import pexpect

# def createSSHClient(server, port, user, password):
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
#     client.connect(server, port, user, password)

#     return client

# ssh = createSSHClient("10.1.29.28", 31931, "root", "whitekirin")

# # os.mkdir("Original_ResNet101V2_with_NPC_Augmentation_Image")
# # with open("Original_ResNet101V2_with_NPC_Augmentation_Image_train3.txt", "w") as file:
# #     pass

# with SCPClient(ssh.get_transport()) as scp:
#     scp.get("/mnt/c/張晉嘉/stomach_cancer/Original_ResNet101V2_with_NPC_Augmentation_Image_train3.txt", "/raid/whitekirin/stomach_cancer/Model_result/save_the_train_result(2024-10-05)/Original_ResNet101V2_with_NPC_Augmentation_Image_train3.txt")


from Training_Tools.Tools import Tool

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

tool = Tool()
tool.Set_Labels() # 要換不同資料集就要改
tool.Set_Save_Roots()
Train, Test, Validation = tool.Get_Save_Roots(1)

train_dataset = ImageFolder(root=Train, transform=transform)

balanced_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4)
tests = []
for i, (test, labels) in enumerate(balanced_loader):
    tests.append(labels)

print(tests)