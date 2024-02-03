import torch as th
import torchvision as tv
import pro_gan_pytorch.PRO_GAN as pg

# select the device to be used for training
device = th.device("cuda" if th.cuda.is_available() else "cpu")
data_path = "cifar10/"

def setup_data(download=False):
    """
    setup the CIFAR-10 dataset for training the CNN
    :param batch_size: batch_size for sgd
    :param num_workers: num_readers for data reading
    :param download: Boolean for whether to download the data
    :return: classes, trainloader, testloader => training and testing data loaders
    """
    # data setup:
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    transforms = tv.transforms.ToTensor()

    trainset = tv.datasets.CIFAR10(root=data_path,
                                   download=download)

    testset = tv.datasets.CIFAR10(root=data_path,
                                  transform=transforms, train=False,
                                  download=False)

    return trainset


if __name__ == '__main__':

    # some parameters:
    depth = 4
    # hyper-parameters per depth (resolution)
    num_epochs = [10, 20, 20, 20]
    fade_ins = [50, 50, 50, 50]
    batch_sizes = [128, 128, 128, 128]
    latent_size = 128

    # get the data. Ignore the test data and their classes
    dataset = setup_data(download=False)

    print(type(dataset))


    print(dataset[0][1])
    print(dataset[1][1])
    print(dataset[2][1])
    print(dataset[3][1])



