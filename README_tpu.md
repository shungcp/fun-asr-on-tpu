# fun-asr-on-tpu

This is a hobby project to try debugging and running ASR model inference on Trillium(v6e) and Ironwood(late 2026).

这是一个个人爱好者项目，欢迎有兴趣移植和优化的同学加入。

目的是为了调试和运行一些流行ASR/TTS模型在TPU上的推理任务。

当然也会在移植的过程中，收获学习和成长的乐趣。

## Progress:

### 2026-02-02 

port the model Fun-ASR-Nano-2512 on v6e-1 with torchax, metrics:  1.26it/s

[guess] this is just for one request, will be almost the same for batch mode
[TODO] develop a batch mode inference API
[ISSUE] Output is not the same as ctc_text, need to figure out the RCA.

```
--- Run 1 ---
  0%|                                                                                       | 0/1 [00:00<?, ?it/s]
DEBUG: encode input speech: torch.Size([1, 94, 560]), lengths: tensor([94], dtype=torch.int32)
DEBUG: encode output encoder_out: torch.Size([1, 94, 512]), lens type: <class 'torch.Tensor'>
DEBUG: encode output lens shape: torch.Size([1])
DEBUG: encode output lens val: tensor([94], dtype=torch.int32)
`torch_dtype` is deprecated! Use `dtype` instead!
rtf_avg: 4.424: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:24<00:00, 24.95s/it]
Output (text field only): 开饭时间早上九点开饭时间早上九点开饭时间早上
Run 1 Duration: 25.6478s

--- Run 2 ---
  0%|                                                                                       | 0/1 [00:00<?, ?it/s]
DEBUG: encode input speech: torch.Size([1, 94, 560]), lengths: tensor([94], dtype=torch.int32)
DEBUG: encode output encoder_out: torch.Size([1, 94, 512]), lens type: <class 'torch.Tensor'>
DEBUG: encode output lens shape: torch.Size([1])
DEBUG: encode output lens val: tensor([94], dtype=torch.int32)
rtf_avg: 0.149: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.19it/s]
Output (text field only): 开饭时间早上九点开饭时间早上九点开饭时间早上
Run 2 Duration: 1.5870s

--- Run 3 ---
  0%|                                                                                       | 0/1 [00:00<?, ?it/s]
DEBUG: encode input speech: torch.Size([1, 94, 560]), lengths: tensor([94], dtype=torch.int32)
DEBUG: encode output encoder_out: torch.Size([1, 94, 512]), lens type: <class 'torch.Tensor'>
DEBUG: encode output lens shape: torch.Size([1])
DEBUG: encode output lens val: tensor([94], dtype=torch.int32)
rtf_avg: 0.141: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.26it/s]
Output (text field only): 开饭时间早上九点开饭时间早上九点开饭时间早上
Run 3 Duration: 1.5562s
```

### 2026-02-01

v0.1 end-to-end works,but slow w/ xla compilation  ~66s/it

```
--- TPU Demo Clean Start ---
[TPU Patch] Applying Qwen2 Attention Patch...
[TPU Patch] Qwen2 Attention Patch TEMPORARILY DISABLED for isolation
[TPU Patch] SANM Patch DISABLED (Confirmed)
[TPU Patch] Patches Applied.
Loading model from /home/admin_shunwang_altostrat_com/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512...
funasr version: 1.3.1.
WARNING:root:trust_remote_code: True
Loading remote code successfully: ./model.py
[Debug] Audio Adaptor Name: Transformer
[Debug] Audio Adaptor Class: <class 'funasr.models.llm_asr.adaptor.Transformer'>
[Debug] Forcing downsample_rate to 8 (was 1) because use_low_frame_rate is True.
[Debug] Audio Adaptor Conf: {'downsample_rate': 1, 'use_low_frame_rate': True, 'ffn_dim': 2048, 'llm_dim': 1024, 'encoder_dim': 512, 'n_layer': 2, 'freeze': True}
Compiling Audio Encoder with Torchax...
[TorchaxWrapper] Compiling SenseVoiceEncoderSmall with static_kwargs=[]...
[TorchaxWrapper] Registered 0 hidden tensors as buffers.
WARNING:root:Duplicate op registration for aten.__and__
Compiling LLM with Torchax...
[TorchaxWrapper] Compiling LLMForwardWrapper with static_kwargs=[]...
[TorchaxWrapper] Registered 0 hidden tensors as buffers.
`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
Running inference on /home/admin_shunwang_altostrat_com/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/example/zh.mp3...

--- Run 1 ---
  0%|                                                                                                                                                   | 0/1 [00:00<?, ?it/s]
DEBUG: encode input speech: torch.Size([1, 94, 560]), lengths: tensor([94], dtype=torch.int32)
DEBUG: encode output encoder_out: torch.Size([1, 94, 512]), lens type: <class 'torch.Tensor'>
DEBUG: encode output lens shape: torch.Size([1])
DEBUG: encode output lens val: tensor([94], dtype=torch.int32)
`torch_dtype` is deprecated! Use `dtype` instead!
rtf_avg: 11.902: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:07<00:00, 67.13s/it]
Output: [{'key': 'zh', 'text': '开饭时间早上九点至下午五点。', 'text_tn': '开饭时间早上九点至下午五点', 'label': 'null', 'ctc_text': '开饭时间早上九点至下午五点', 'ctc_timestamps': [{'token': '开', 'start_time': 0.78, 'end_time': 0.84, 'score': 0.939}, {'token': '饭', 'start_time': 1.08, 'end_time': 1.14, 'score': 0.796}, {'token': '时', 'start_time': 1.38, 'end_time': 1.44, 'score': 0.992}, {'token': '间', 'start_time': 1.56, 'end_time': 1.62, 'score': 0.992}, {'token': '早', 'start_time': 1.98, 'end_time': 2.04, 'score': 0.969}, {'token': '上', 'start_time': 2.22, 'end_time': 2.28, 'score': 0.999}, {'token': '九', 'start_time': 2.64, 'end_time': 2.7, 'score': 0.998}, {'token': '点', 'start_time': 3.0, 'end_time': 3.06, 'score': 0.991}, {'token': '至', 'start_time': 3.48, 'end_time': 3.54, 'score': 0.996}, {'token': '下', 'start_time': 4.08, 'end_time': 4.14, 'score': 0.997}, {'token': '午', 'start_time': 4.32, 'end_time': 4.38, 'score': 0.999}, {'token': '五', 'start_time': 4.68, 'end_time': 4.74, 'score': 0.993}, {'token': '
  ', 'start_time': 4.92, 'end_time': 4.98, 'score': 0.988}], 'timestamps': [{'token': '开', 'start_time': 0.78, 'end_time': 0.84, 'score': 0.939}, {'token': '饭', 'start_time': 1.08, 'end_time': 1.14, 'score': 0.796}, {'token': '时', 'start_time': 1.38, 'end_time': 1.44, 'score': 0.992}, {'token': '间', 'start_time': 1.56, 'end_time': 1.62, 'score': 0.992}, {'token': '早', 'start_time': 1.98, 'end_time': 2.04, 'score': 0.969}, {'token': '上', 'start_time': 2.22, 'end_time': 2.28, 'score': 0.999}, {'token': '九', 'start_time': 2.64, 'end_time': 2.7, 'score': 0.998}, {'token': '点', 'start_time': 3.0, 'end_time': 3.06, 'score': 0.991}, {'token': '至', 'start_time': 3.48, 'end_time': 3.54, 'score': 0.996}, {'token': '下', 'start_time': 4.08, 'end_time': 4.14, 'score': 0.997}, {'token': '午', 'start_time': 4.32, 'end_time': 4.38, 'score': 0.999}, {'token': '五', 'start_time': 4.68, 'end_time': 4.74, 'score': 0.993}, {'token': '点', 'start_time': 4.92, 'end_time': 4.98, 'score': 0.988}, {'token': '。', 'start_time': 5.1, 'end_time': 5.16, 'score': 0.0}]}]
Run 1 Duration: 67.7782s

```


## 2026-02-xx in progress

optimize the performance in batch 

