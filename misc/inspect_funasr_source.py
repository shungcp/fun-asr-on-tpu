
import inspect
from funasr.models.sense_voice import model

print("Source of EncoderLayerSANM:")
try:
    print(inspect.getsource(model.EncoderLayerSANM))
except Exception as e:
    print(f"Could not get source of EncoderLayerSANM: {e}")

print("\nSource of MultiHeadedAttentionSANM:")
try:
    print(inspect.getsource(model.MultiHeadedAttentionSANM))
except Exception as e:
    print(f"Could not get source of MultiHeadedAttentionSANM: {e}")
