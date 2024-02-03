## Main idea
I looked at dynmap and just though what about if it would be really huge.
I thought to use model to generate images. I though it would be faster but probably the fastest way would be to rewrite generation of chunks from minecraft to c++ and also render from dynmap rather than train models. It would be cool if user in browser could use like paint tool to generate custom chunks with custom biomes. At like 10k+ chunks per second. Realistically this project is pretty useless but might look cool at the end.
## What to do
1. Setup minecraft server use chunky pregenerate 100-200Gb minecraft world
2. Use dynmap generate shader chunk perspective and your normal perspective
3. Run my script it will remove water chunks.
<br> TODO Make actal script that will generate me 3x3 dataset rather than just extract both shader and normal perspective jpgs 
4. I don't know i am currently at this stage testing stuff. 



## Currently
Currently i tried some of things. <br>
https://github.com/alexeyhorkin/ProGAN-PyTorch.git <br>
This was the only implementation that converge from my i partly understand why. (Couldn't try really other models because i just have 8gb vram gpu). The issue with above implementation is 
from my understanding i need to put previous frames to the latent space. But it goes quickly with vram.<br> <br>
I changed architecture of it currently i am trying to get encoder decoder architecture working. Then using encoder to generate me data into another model which will understand which chunks were previousely generated and nicely generate me a new chunk. It feels like when i was doing java at first. I just had some clue what i am doing. 
<br>
Not trying to reinvent the wheel using this https://github.com/clementchadebec/benchmark_VAE to check what's working and what not then i will adjust
<br>
<br> I also did some more stuff such as rewriting ProGAN to use float16 instead of float32 but loss function did not particular like it. Unsure if i did something wrong. Still my progress on this is pretty slow.