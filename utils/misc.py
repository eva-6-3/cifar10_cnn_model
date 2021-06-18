import torch
import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary


def is_cuda(debug=True):
    cuda = torch.cuda.is_available()
    if debug:
        print("[INFO] Cuda Avaliable : ", cuda)
    return cuda


def get_device():
    use_cuda = is_cuda(debug=False)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] device : ", device)
    return device


def set_seed(seed=1):
    cuda = is_cuda(debug=False)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    print(f"[INFO] seed set {seed}")


def show_random_images_for_each_class(
    train_data,
    num_images_per_class=16
):
    for c, cls in enumerate(train_data.classes):
        rand_targets = random.sample([
            n
            for n, x in enumerate(train_data.targets)
            if x==c
        ], k=num_images_per_class)
        show_img_grid(
            np.transpose(train_data.data[rand_targets], axes=(0, 3, 1, 2))
        )
        plt.title(cls)
    

def show_img_grid(data):
    try:
        grid_img = torchvision.utils.make_grid(data.cpu().detach())
    except:
        data = torch.from_numpy(data)
        grid_img = torchvision.utils.make_grid(data)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    

def show_random_images(data_loader):
    data, target  = next(iter(data_loader))
    show_img_grid(data)


def show_model_summary(model, input_size=(1, 28, 28)):
    summary(model, input_size=input_size)


def get_wrong_predictions(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    wrong_correct = []
    wrong_predicted = []
    wrong_image_data = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            status = pred.eq(target.view_as(pred))
            # correct += status.sum().item()

            mistakes, _ = torch.where(status==False)
            if len(mistakes):
                m_data = data[mistakes]
                m_target = target[mistakes]
                m_output = output[mistakes]
                m_pred = pred[mistakes]
                correct = [x.item() for x in m_target.cpu().detach()]
                predicted = [x.item() for x in m_pred.cpu().detach()]
                image_data = [x for x in m_data.cpu().detach()]

                wrong_correct.extend(correct)
                wrong_predicted.extend(predicted)
                wrong_image_data.extend(image_data)
    
    return wrong_correct, wrong_predicted, wrong_image_data


def show_grid(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def show_wrong_images(targets, predicts, images, size=20, grid=(5, 4)):
    img_data_temp = []
    wps = []
    for n, (wc, wp, wi) in enumerate(zip(targets, predicts, images)):
        wps.append(wp)
        img_data_temp.append(wi)
        if n>18:
            break
    
    wrong_images_temp = torch.stack(img_data_temp)
    print()
    print(f"Mistakenly predicted as {wps}")

    grid_img = torchvision.utils.make_grid(wrong_images_temp, nrow=grid[0])
    show_grid(grid_img)