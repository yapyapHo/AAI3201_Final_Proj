from test import evaluate
from txt2br import text
from os.path import exists, join
import torch
import os
from Hangeul_OCR import CNN

def main():
    device = torch.device('cpu')

    model_path = 'TrainModel/CNN'
    model_name = 'model_epoch10.pth'
    if not exists(join(model_path, model_name)):
        tmp = input('model not exist')
        Infer = False
    else:
        # 모델 불러오기
        model = CNN()
        ckpt = torch.load(join(model_path, model_name))
        model.load_state_dict(ckpt['net'])
        Infer = True

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

        letter = evaluate(model, user_input, device)
        result = text(letter)
        print(result)


if __name__ == '__main__':
    main()