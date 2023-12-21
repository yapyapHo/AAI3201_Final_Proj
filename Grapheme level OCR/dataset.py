from jamo import h2j, j2hcj
import random
import glob
import PIL
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

init_label = {'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6, 'ㅂ': 7, 'ㅃ': 8, 'ㅅ': 9, 'ㅆ': 10,
              'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13, 'ㅊ': 14, 'ㅋ': 15, 'ㅌ': 16, 'ㅍ': 17, 'ㅎ': 18}
medial_label = {'ㅏ': 0, 'ㅐ': 1, 'ㅑ': 2, 'ㅒ': 3, 'ㅓ': 4, 'ㅔ': 5, 'ㅕ': 6, 'ㅖ': 7, 'ㅗ': 8, 'ㅘ': 9, 'ㅙ': 10,
                'ㅚ': 11, 'ㅛ': 12, 'ㅜ': 13, 'ㅝ': 14, 'ㅞ': 15, 'ㅟ': 16, 'ㅠ': 17, 'ㅡ': 18, 'ㅢ': 19, 'ㅣ': 20}
final_label = {'': 0, 'ㄱ': 1, 'ㄲ': 2, 'ㄳ': 3, 'ㄴ': 4, 'ㄵ': 5, 'ㄶ': 6, 'ㄷ': 7, 'ㄹ': 8, 'ㄺ': 9, 'ㄻ': 10,
               'ㄼ': 11, 'ㄽ': 12, 'ㄾ': 13, 'ㄿ': 14, 'ㅀ': 15, 'ㅁ': 16, 'ㅂ': 17, 'ㅄ': 18, 'ㅅ': 19, 'ㅆ': 20,
               'ㅇ': 21, 'ㅈ': 22, 'ㅊ': 23, 'ㅋ': 24, 'ㅌ': 25, 'ㅍ': 26, 'ㅎ': 27}

# 정답 json 에서 가져오기
f = open('handwriting.json', 'r', encoding='UTF-8')
json_data = json.load(f)

transform = transforms.Compose([
    transforms.Resize(size=(225,225)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, file_path, num_data):
        files = glob.glob(file_path + '/*.png') # 파일 이름
        self.x_data = random.sample(files, num_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        file_name = self.x_data[idx]
        img = PIL.Image.open(file_name)
        img = transform(img)
        x = torch.FloatTensor(img)

        file_name = file_name.strip('.png')
        file_name = file_name.split('\\')[-1]
        text = json_data[file_name]

        y = list(j2hcj(h2j(text))) # 자음 모음으로 쪼개기
        # 종성이 없을때 빈 토큰 추가
        if len(y) == 2:
            y.append('')

        # 자모 라벨링
        y[0] = init_label[y[0]]
        y[1] = medial_label[y[1]]
        y[2] = final_label[y[2]]

        y = torch.LongTensor(y)
        return x, y