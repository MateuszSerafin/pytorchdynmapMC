import numpy
from pythae.models import AutoModel
from pythae.samplers import NormalSampler
my_trained_vae = AutoModel.load_from_folder(
    'checkpoint_epoch_42'
)
my_samper = NormalSampler(
model=my_trained_vae
)
gen_data = my_samper.sample(
num_samples=50,
batch_size=10,
output_dir=None,
return_gen=True
)
from PIL import Image
import os
import numpy as np
# Iterate through the array and save each element as an image
for i, data_item in enumerate(gen_data):
    data_item = data_item.cpu().numpy()
    # Assuming the data is in the range [0, 1]
    image_array = (data_item * 255).astype(np.uint8)
    # Create a PIL Image from the array
    image = Image.fromarray(numpy.moveaxis(image_array, 0, 2))
    # Save the image to a file
    image_path = os.path.join(f"data_{i}.png")
    image.save(image_path)