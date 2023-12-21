from test import evaluate
from txt2br import text
import torch
import os
from jamo import j2h

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

        grapheme = evaluate(model, user_input, device)
        letter = j2h(grapheme[0], grapheme[1], grapheme[2])

        result = text(letter)
        print(result)


if __name__ == '__main__':
    main()