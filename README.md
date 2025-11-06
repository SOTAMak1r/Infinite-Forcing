<p align="center">
<img src="assets/logo.png" width=20%>
<p>

<p align="center">
<h1 align="center">Towards Infinite-Long Video Generation</h1>
</p>

<div align="center">

[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-huggingface-yellow)](https://huggingface.co/SOTAMak1r/Infinite-Forcing/tree/main)&nbsp;

</div>


<div align="center">

<video src="https://github.com/user-attachments/assets/9d155d04-39a5-4419-bd83-bb892947d702" autoplay muted loop playsinline></video>

</div>


## 👀 Preliminary: Self-Forcing

Self Forcing trains autoregressive video diffusion models by **simulating the inference process during training**, performing autoregressive rollout with KV caching. It resolves the train-test distribution mismatch and enables **real-time, streaming video generation on a single RTX 4090** while matching the quality of state-of-the-art diffusion models.


## 🦄 Difference with Self-Forcing: V-sink

> <b>TL;DR: (1) Treat the initially generated `sink_size` frames as V-sink; (2) Incorporate V-sink into training; (3) Apply RoPE operation after retrieving KV cache. </b>

We found that the Self-Forcing repository has implemented `sink_size`, but the specific usage method is not discussed in their paper.  By printing the attention map, we did not observe phenomena similar to the [attention_sink](https://github.com/mit-han-lab/streaming-llm/tree/main) in LLMs. The experimental configuration for the attention map below is in `configs/self_forcing_dmd_frame1.yaml`. 

<table style="text-align: center; margin-left: auto; margin-right: auto;">
<tr>
<td>
<img src="./assets/frame5_step_0_pooled.png"></img>
</td>
<td>
<img src="./assets/frame10_step_0_pooled.png"></img>
</td>
<td>
<img src="./assets/frame20_step_0_pooled.png"></img>
</td>
<td>
<img src="./assets/frame40_step_0_pooled.png"></img>
</td>
</tr>

<tr>
<td style="text-align: center;">frame 5</td>
<td>frame 10</td>
<td>frame 20</td>
<td>frame 40</td>
</tr>
</table>

When `sink_size = 1` and inference exceeds the training length (> 20 frame), it can be observed that the model assigns a larger attention weight to the frame serving as the sink only in the final block. However, in experiments, we found that even without the appearance of an attention map similar to that in LLMs, using the first `sink_size` frames as sinks can **significantly improve visual quality**.

> We interpret this as the first `sink_size` frames providing a larger memory context, enabling the model to access earlier memories, which helps mitigate exposure bias (AKA drift).

Moreover, the implementation in the Self-Forcing repository applies the RoPE operation before storing the KV cache. We observed that when the inference length becomes too long, the effectiveness of the sink diminishes. Therefore, adopting an approach similar to [streamingLLM](https://github.com/mit-han-lab/streaming-llm/tree/main), we incorporated the sink frame (which we refer to as `V-sink`) into the training process and **moved the RoPE operation to after retrieving the KV cache**. The specific implementation can be found in the `CausalWanSelfAttention` class in `wan/modules/causal_model.py`.

> V-sink differs from attention sink in LLMs. V-sink is a complete frame, whereas in LLMs the sink is the first token of the sequence (typically \<bos\>). Thus, their working mechanisms are distinct.  


## Comparison

We compared the inference performance of three methods:
  - the original Self-Forcing implementation (Left)
  - Self-Forcing w/ V-sink (Mid)
  - **Infinite-Forcing (Right)**


<video src="https://github.com/user-attachments/assets/6c8bc09d-5955-47af-a824-f45f4bf4b6ef" autoplay muted loop playsinline></video>

<video src="https://github.com/user-attachments/assets/5aca1280-e957-4124-9e56-c2d00f2b583b" autoplay muted loop playsinline></video>

<video src="https://github.com/user-attachments/assets/82fd5270-c1ec-46c7-8da5-32b1a8b18fcb" autoplay muted loop playsinline></video>

<video src="https://github.com/user-attachments/assets/d514caa4-20fc-4a13-b5d7-5f5f164a51e5" autoplay muted loop playsinline></video>

<video src="https://github.com/user-attachments/assets/c8ecb294-dcb2-452a-ab06-2f3c8e3116fe" autoplay muted loop playsinline></video>

## 📹️ Gallery


<table class="center" border="0" style="width: 100%; text-align: left;">
<!-- 1 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/9afefed1-e8a0-4f8d-9bb0-68800703ed22" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/fbb029f8-ffba-474d-aa4f-7e32f342186c" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 2 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/04cc84f2-94d7-477b-a954-1bf83f9ff7ae" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/10046e9e-f5e4-4591-b3a4-7fd0932b4465" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 3 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/4cf89920-21d2-4b38-b1e5-13eea321d4c7" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/b532e1a9-8010-45ed-9725-2fc8732c918c" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 4 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/5f15141d-5d03-4dfa-8ec5-4043a61fae94" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/e5f845a4-eede-4b70-8bbd-d96585e842cc" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 5 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/03cd1169-536d-4f09-9792-5b039ff05015" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/0126f5f2-1cce-4cc6-a724-223cb2d816b4" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 6 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/47bed94a-413f-428a-b72d-a976eda303e3" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/184759e7-d4d3-4401-bc60-ff8241de5b4d" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 7 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/0cf44b7c-bcb0-41f2-a46c-2b873eb0932f" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/2b3c7102-b440-4c24-ad7c-968c41208af5" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 8 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/986c6d77-30e5-414c-9e51-862f60f0808a" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/4f33e58c-53c6-4e6f-84c0-e94abeab4f5e" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 9 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/df3ba546-4be9-4800-ad1b-7a3095295c46" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/6c45ec9c-4e06-443b-86d6-7800f0d32f9e" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 10 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/f51fda74-3ef7-45d2-b49b-d820a2811116" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/1e9daea8-7669-4860-8e0e-fcf5890ebf79" autoplay muted loop playsinline></video>
</td>
</tr>
<!-- 11 -->
<tr>
<td>
<video src="https://github.com/user-attachments/assets/5d107da0-74fd-4ce9-a942-1c6b2b7d10b1" autoplay muted loop playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/eafa278e-63d3-4550-b326-9d0b85564e3e" autoplay muted loop playsinline></video>
</td>
</tr>

</table>



## Application: interactive video generation

Since Infinite-Forcing / Self-Forcing ultimately produces a causal autoregressive video generation model, we can modify the text prompts during the generation process to control the video output in real-time.

We initially used the **initial prompt** and switched to the **interaction prompt** midway through inference.


- demo1

  - **Initial prompt**: Wide shot of an elegant urban café: A sharp-dressed businessman in a navy suit focused on a sleek silver laptop, steam rising from a white ceramic coffee cup beside him. Soft morning light streams through large windows, casting warm reflections on polished wooden tables. Background features blurred patrons and baristas in motion, creating a sophisticated yet bustling atmosphere. Cinematic shallow depth of field, muted earth tones, 4K realism.

  - **Interaction prompt**: Medium close-up in a medieval tavern: An elderly wizard with a long grey beard studies an ancient leather-bound spellbook under flickering candlelight. A luminous purple crystal ball pulses on the rough oak table, casting dancing shadows on stone walls. Smoky atmosphere with floating dust particles, barrels and copper mugs in background. Dark fantasy style, volumetric lighting, mystical glow, detailed texture of wrinkled hands and aged parchment, 35mm film grain.

<video src="https://github.com/user-attachments/assets/81fdbfe5-3974-4191-ba1c-51803a0f677c" autoplay muted loop playsinline></video>


- demo2

  - **Initial prompt**: A cinematic wide shot of a serene forest river, its crystal-clear water flowing gently over smooth stones. Sunlight filters through the canopy, creating dancing caustics on the riverbed. The camera tracks slowly alongside the flow.


  - **Interaction prompt**: In the same wide shot, a wave of freezing energy passes through the frame. The flowing water instantly crystallizes into a solid, glistening sheet of ice, trapping air bubbles inside. The camera continues its track, now revealing a solitary red fox cautiously stepping onto the frozen surface, its breath visible in the suddenly cold air.

<video src="https://github.com/user-attachments/assets/70d15211-3fd4-47d3-a71f-439c51bd79a5" autoplay muted loop playsinline></video>

- demo3

  - **Initial prompt**: A dynamic drone shot circles a bustling medieval town square at high noon. People in colorful period clothing crowd the market stalls. The sun is bright, casting sharp, short shadows. Flags flutter in a gentle breeze.


  - **Interaction prompt**: From the same drone perspective, day rapidly shifts to a deep, starry night. The scene is now illuminated by the warm glow of torches in iron sconces and a cool, full moon. The bustling crowd is replaced by a few mysterious, cloaked figures moving through the long, dramatic shadows. The earlier gentle breeze is now a visible mist of cold breath.

<video src="https://github.com/user-attachments/assets/2650ba15-6ab1-44bb-8d2b-48451992c2a5" autoplay muted loop playsinline></video>

- demo4

  - **Initial prompt**: A macro, still-life shot of a half-full glass of water on a rustic wooden table. Morning light streams in from a window, highlighting the condensation on the glass. A few coffee beans and a newspaper are also on the table.


  - **Interaction prompt**: In the same macro shot, gravity reverses. The water elegantly pulls upwards out of the glass, morphing into a perfect, wobbling sphere that hovers mid-air. The coffee beans and a corner of the newspaper also begin to float upwards in a slow, graceful ballet. The condensation droplets on the glass now drift away like tiny planets.

<video src="https://github.com/user-attachments/assets/9e6f666c-80d9-4950-b123-88c9c3e79e47" autoplay muted loop playsinline></video>


## 🛠️ Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## 🚀 Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download SOTAMak1r/Infinite-Forcing --local-dir checkpoints/
```

### CLI Inference
Example inference script using the chunk-wise autoregressive checkpoint trained with DMD:
```
python inference.py \
    --config_path configs/self_forcing_dmd_vsink1.yaml \
    --output_folder videos/self_forcing_dmd_vsink1 \
    --checkpoint_path path/to/your/pt/checkpoint.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

## 🚂 Training
### Download text prompts and ODE initialized checkpoint 

Follow Self-Forcing

### Infinite Forcing Training with V-sink
```
torchrun --nnodes=2 --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd_vsink1.yaml \
  --logdir logs/self_forcing_dmd_vsink1 \
  --disable-wandb
```

Due to resource constraints, we trained the model using 16 A800 GPUs with a gradient accumulation of 4 to simulate the original Self-Forcing configuration.

## 💬 Discussion
- [2025.9.30] We observed that incorporating V-sink results in a reduction of dynamic motion in the generated frames, and this issue worsens as training steps increase. We hypothesize that this occurs because the base model's target distribution includes a portion of static scenes, causing the model to "cut corners" by learning that static videos yield lower loss values—ultimately converging to a suboptimal sub-distribution. Additionally, the introduction of V-sink tends to make subsequent videos overly resemble the initial frames (Brainstorm: Could this potentially serve as a memory mechanism for video-generation-based world models?).  

- [2025.10.22] Thanks to [Xianglong](https://github.com/XianglongHe) for helping validate the effectiveness of our method on [matrix-game-v2](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2). Compared to their Self-Forcing-based baseline, it achieved fewer error accumulations and maintained consistency in scenario style! (Video below at 4x speed)


<div align="center">

<video src="https://github.com/user-attachments/assets/25feb2a7-bf91-4399-b372-6da865545095" autoplay muted loop playsinline></video>

</div>

We will continue to update! Stay tuned!


## Acknowledgements
This codebase is built on top of the open-source implementation of [Self-Forcing](https://github.com/guandeh17/Self-Forcing) repo.


## Citation
If you find this codebase useful for your research, please kindly cite:
```

@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}

@misc{infinite-forcing,
    Author = {Junyi Chen, Zhoujie Fu},
    Year = {2025},
    Note = {https://github.com/SOTAMak1r/Infinite-Forcing},
    Title = {Infinite-Forcing: Towards Infinite-Long Video Generation}
}

```
