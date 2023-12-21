from model import OCR

import torch
import os
import PIL
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(size=(225,225)),
    transforms.ToTensor()
])

init_label = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
medial_label = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
final_label = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ',
               'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def evaluate(model, path, device):
    img = PIL.Image.open(path)
    img = transform(img)
    img = torch.FloatTensor(img)
    with torch.no_grad():
        x = img.to(device)
        x = x.unsqueeze(0)

        output = model.forward(x)
        _, init_index = torch.max(output[0], 1)
        _, medial_index = torch.max(output[1], 1)
        _, final_index = torch.max(output[2], 1)

        tmp = []
        tmp.append(init_label[init_index.item()])
        tmp.append(medial_label[medial_index.item()])
        tmp.append(final_label[final_index.item()])
        print(tmp)

    return tmp

def main():
    device = torch.device('cpu')
    if not os.path.exists('model/model.pth'):
        Infer = False
        tmp = input('model not exist')
    else:
        model = torch.load('model/model.pth', map_location=device)
        Infer = True
        model.eval()

    while Infer:
        user_input = input('파일 경로를 입력해주세요: ')

        # 종료
        if user_input == 'q':
            tmp = input('종료합니다')
            break

        # 파일 없음
        if not os.path.exists(user_input):
            print('파일이 존재하지 않음')
            continue

        evaluate(model, user_input, device)


if __name__ == '__main__':
    main()