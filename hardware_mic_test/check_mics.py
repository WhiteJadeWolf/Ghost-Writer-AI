import sounddevice as sd
print(sd.query_devices())
print("\n--- YOUR DEFAULT INPUT DEVICE ---")
print(sd.query_devices(kind='input'))