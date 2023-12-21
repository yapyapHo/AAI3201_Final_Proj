from model import OCR
from dataset import CustomDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE = 200
EPOCH = 4

############ Dataset ###############

NUM_DATA = 100000  # 훈련할 데이터 개수
path = 'hangul'   # 데이터 경로
dataset = CustomDataset(path, NUM_DATA)

# 8:1:1비율로 데이터 나누기
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
val_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iter = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

################# evaluate ###################
def func_eval(model, data_iter, device):
    with torch.no_grad():
        model.eval()

        total, correct = 0, 0
        for batch_in, batch_out in data_iter:
            x = batch_in.to(device)
            y = batch_out.to(device)

            # 초성 중성 종성 가장 확률이 높은거 선택
            output = model.forward(x)
            _, init_index = torch.max(output[0], 1)
            _, medial_index = torch.max(output[1], 1)
            _, final_index = torch.max(output[2], 1)

            # Tensor 형태 바꾸기
            init_index = init_index.view(init_index.size(0), -1)
            medial_index = medial_index.view(medial_index.size(0), -1)
            final_index = final_index.view(final_index.size(0), -1)

            output = torch.concatenate((init_index, medial_index, final_index), 1)
            total += batch_out.size(0)
            # 정답 개수 카운트
            s = (output==y).sum(1)
            try:
                count = torch.bincount(s)[3].to(device)
            except IndexError:
                count = torch.tensor(0).to(device)
            correct += count

        val_accr = (correct / total)

        model.train()

    return val_accr.item()
################## train ######################
# pretrained 모델 있으면 가져오기
if os.path.exists('model.pth'):
    print('resume training')
    model = torch.load('model.pth', map_location=device)
else:
    print('creating a new model')
    model = OCR(device).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

writer = SummaryWriter('logs/')

print('start training')
for i in range(EPOCH):
    loss_val_sum = 0
    for j, [image, label] in enumerate(train_iter):
        x = image.to(device)
        y = label.to(device)
        optimizer.zero_grad()

        loss = 0
        output = model.forward(x)
        for k in range(3):
            loss += loss_func(output[k], y[:, k])
        loss = loss/3
        loss.backward()
        optimizer.step()

        loss_val_sum += loss
        step = i * BATCH_SIZE * len(train_iter) + j * BATCH_SIZE
        print('step: [%d], loss : [%.3f]' % (step, loss.item()))

        if (step % 5000 == 0):
            train_accr = func_eval(model, train_iter, device)
            val_accr = func_eval(model, val_iter, device)
            test_accr = func_eval(model, test_iter, device)

            # 텐서보드 로그 작성
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", train_accr, step)
            writer.add_scalar("Accuracy/val", val_accr, step)
            writer.add_scalar("Accuracy/test", test_accr, step)

            writer.flush()

    torch.save(model, 'model.pth')
    loss_val_avg = loss_val_sum / len(train_iter)

    print("epoch:[%d] loss:[%.3f] train_accr:[%.3f] val_accr:[%.3f] test_accr:[%.3f]." %
          (i, loss_val_avg, train_accr, val_accr, test_accr))
writer.close()