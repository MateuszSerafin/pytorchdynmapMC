import os

import torchvision
import PIL
from torch.utils.data import Dataset, DataLoader
import torch
import Model
import losses
import random
import tqdm



class Trainer():
    def __init__(self, dataloader):
        self.iteration = 0

        # setup data set and data loader
        self.dataloader = dataloader

        self.adv_loss = losses.smgan()

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]

        self.netG = Model.InpaintGenerator().cuda()
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=0.001, betas=(0.05, 0.999))

        self.netD = Model.Discriminator().cuda()
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=0.001, betas=(0.05, 0.999))


    def train(self):
        for epoch in range(100):
            counter = 0
            for images,masks in tqdm.tqdm(self.dataloader):
                images, masks = images.cuda(), masks.cuda()
                images_masked = (images * (1 - masks).float()) + masks

                pred_img = self.netG(images_masked.float(), masks)
                comp_img = (1 - masks) * images + masks * pred_img

                # adversarial loss
                dis_loss, gen_loss = self.adv_loss(self.netD, comp_img, images, masks)

                # backforward
                self.optimG.zero_grad()
                self.optimD.zero_grad()
                gen_loss.backward()
                dis_loss.backward()
                self.optimG.step()
                self.optimD.step()


                print(f"""Generator loss {gen_loss.sum()}, Discriminator loss {dis_loss.sum()}""")

                #if self.args.tensorboard:
                #    self.writer.add_image('mask', make_grid(masks), self.iteration)
                #    self.writer.add_image('orig', make_grid((images+1.0)/2.0), self.iteration)
                #    self.writer.add_image('pred', make_grid((pred_img+1.0)/2.0), self.iteration)
                #    self.writer.add_image('comp', make_grid((comp_img+1.0)/2.0), self.iteration)

                torchvision.transforms.functional.to_pil_image(pred_img[:3, :, :])[0].save(f"""{epoch}_{counter}.jpg""")
                counter += 1


def generate_random_mask_pil():
    x_rand = random.randint(0, 48)
    y_rand = random.randint(0, 48)
    #this should be done through numpy but i had PIL code so i just recyclded that also its for testing i dont care about performacne
    grayscale = PIL.Image.new("L", (64,64))
    grayscale.paste(PIL.Image.new("L", (16,16), "white"), (x_rand, y_rand))
    return grayscale


class CustomDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].float()

        mask = torchvision.transforms.functional.pil_to_tensor(generate_random_mask_pil()).float()

        return image, mask

# Specify the image folder
image_folder = '/mnt/2tb/Projects/inpainting/world/'


images = []
for image in os.listdir(image_folder):
    print(image)
    img = PIL.Image.open(os.path.join(image_folder, image))
    copy = img.copy()
    img.close()
    images.append(torchvision.transforms.functional.pil_to_tensor(copy.resize((64,64))))



# Create the dataset and data loader
dataset = CustomDataset(images)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__=="__main__":
    Trainer(data_loader).train()

