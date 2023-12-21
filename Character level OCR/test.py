from model import CNN
import torch
import PIL
from os.path import exists, join
import torchvision.transforms as transforms
from utility import txt_to_dict

transform = transforms.Compose([
    transforms.Resize(size=(32,32)),
    transforms.ToTensor()
])

def evaluate(model, path, device):
    img = PIL.Image.open(path)
    img = transform(img)
    img = torch.FloatTensor(img)

    idx_to_syllable = txt_to_dict('', 'KSX1001.txt')[0]
    with torch.no_grad():
        x = img.to(device)
        x = x.unsqueeze(0)

        output = model.forward(x)
        _, predicted = torch.max(output.data, 1)
        predicted = idx_to_syllable[predicted.item()]
        print(predicted)

    return predicted


def main():
    device = torch.device('cpu')
    model_path = 'TrainModel/CNN'
    model_name = 'model_epoch10.pth'
    if not exists(join(model_path, model_name)):
        tmp = input('model not exist')
    else:
        # 모델 불러오기
        model = CNN()
        ckpt = torch.load(join(model_path, model_name))
        model.load_state_dict(ckpt['net'])

    Infer = True
    while Infer:
        user_input = input('파일을 입력해주세요: ')

        # 종료
        if user_input == 'q':
            tmp = input('종료합니다')
            break

        # 파일 없음
        if not exists(user_input):
            print('파일이 존재하지 않음')
            continue

        evaluate(model, user_input, device)


if __name__ == '__main__':
    main()