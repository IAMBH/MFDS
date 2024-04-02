import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageOps
from PIL import Image
import cv2 as cv
import os
import getBaseDir
import os, sys
import argparse

import pymssql
from datetime import datetime

server = '20.200.213.94'
username = 'sa'
password = 'Retailtech1@#$'
database = 'SharedKitchen'


# from models.sign_verification.model import SigNet
# from models.sign_verification.inference import Signature
from model import SigNet

class Signature(object):
    def __init__(self, root, threshold=0.0198):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.ckpt_path = os.path.join(root, 'best.pt')

    def model_setup(self):
        model = SigNet().to(self.device)
        model.load_state_dict(torch.load(self.ckpt_path)["model"])
        model.eval()
        return model

    def process_image(self, img_path):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        y, x, h, w = (0, 0, img.shape[0], img.shape[1])

        criteria = max(h, w)

        w_x = (criteria - (w - x)) / 2
        h_y = (criteria - (h - y)) / 2

        if (w_x < 0):
            w_x = 125
        elif (h_y < 0):
            h_y = 125

        M = np.float32([[1, 0, w_x], [0, 1, h_y]])
        img = cv.warpAffine(img, M, (criteria, criteria), borderMode=cv.BORDER_CONSTANT, borderValue=255)

        img = Image.fromarray(img)

        transform = transforms.Compose([
            transforms.Resize(224),
            ImageOps.invert,
            transforms.ToTensor()])
        x = transform(img)

        x = x.unsqueeze(1).to(self.device)

        return x

    def inference(self, model, img1, img2):
        x1 = self.process_image(img1)
        x2 = self.process_image(img2)

        op1, op2 = model(x1, x2)
        euc_dist = F.pairwise_distance(op1, op2).item()
        if euc_dist > self.threshold:
            result = "Forged"
            origin = False
        else:
            result = "Original"
            origin = True
        print(result)

        return origin

def get_args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', required=True, help='Directory to the image of user original signature')
    parser.add_argument('--login', required=True, help='Directory to the image of user login signature')

    return parser

if __name__ == "__main__":
    args = get_args_parse().parse_args()
    
    # ROOT = "./models/sign_verification/ckpt" # 누나가 쓸 때 이거쓰세용
    ROOT = "./ckpt" # exe파일 돌리기위한 ROOT

    # img1 = './models/sign_verification/cedar.png' # 누나가 쓸 때 이거쓰세용
    # img2 = './models/sign_verification/cedar2.png' # 누나가 쓸 때 이거쓰세용
    # img1 = './cedar.png'
    # img2 = './cedar2.png'
    img1 = args.origin
    img2 = args.login


    verifier = Signature(root=ROOT)
    model = verifier.model_setup()
    origin = verifier.inference(model, img1, img2)  # 진짜이면 rue, 위조이면 False return
    
    # DB 접근
    conn = pymssql.connect(server, username, password, database)
    cur = conn.cursor()
    if origin:
        cur.execute('INSERT INTO T_SIGN (SIGN_TF, SIGN_DT, KITCHEN_CD, ROOM_NO, USER_ID) VALUES(%s, %s, %s, %s, %s)', ('TRUE', datetime.today(), '1000', '101', 'USER0'))
        conn.commit()
        print('입실 체크리스트')
    else:
        cur.execute('INSERT INTO T_SIGN (SIGN_TF, SIGN_DT, KITCHEN_CD, ROOM_NO, USER_ID) VALUES(%s, %s)', ('FALSE', datetime.today(), '1000', '101', 'USER0'))
        conn.commit()
        print('위조입니당')
        
    conn.close()