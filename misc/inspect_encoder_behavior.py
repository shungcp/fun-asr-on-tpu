
import os
import torch
import torch.nn as nn
from funasr import AutoModel

def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    print(f"Loading model from {model_dir}...")
    
    # Load model (CPU is fine for inspecting shapes)
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu",
        hub="ms"
    )
    
    nano_model = model.model
    # Ensure encoder in eval mode
    nano_model.audio_encoder.eval()
    
    # SenseVoice/Whisper usually takes fbank [B, T, 80]
    # BUT LFR (Low Frame Rate) might stack them. The error said expected 560.
    B = 2
    T = 150 # Reduced time for speed
    D = 560 # 80 * 7 = 560
    
    speech = torch.randn(B, T, D)
    # Lengths: [150, 75]
    speech_lengths = torch.tensor([150, 75], dtype=torch.int32)
    
    print("Running audio_encoder...")
    with torch.no_grad():
        encoder_out, encoder_out_lens = nano_model.audio_encoder(speech, speech_lengths)
    
    print(f"Output Type: {type(encoder_out_lens)}")
    if torch.is_tensor(encoder_out_lens):
        print(f"Output Shape: {encoder_out_lens.shape}")
        print(f"Output Dtype: {encoder_out_lens.dtype}")
        print(f"Output Values: {encoder_out_lens}")
    else:
        print(f"Output Value: {encoder_out_lens}")

    # Check Adaptor Expectation
    print("\nRunning audio_adaptor...")
    try:
        adaptor_out, adaptor_out_lens = nano_model.audio_adaptor(encoder_out, encoder_out_lens)
        print(f"Adaptor Output Shape: {adaptor_out.shape}")
        print(f"Adaptor Output Lens: {adaptor_out_lens}")
    except Exception as e:
        print(f"Adaptor Failed: {e}")

if __name__ == "__main__":
    main()
