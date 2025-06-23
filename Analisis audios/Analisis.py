#%%

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context("poster")

#AUDIO
import librosa
import librosa.display
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio, display

# FUNCIÓN ESPECTROGRAMA
def graficar_espectrograma(audio, sr, titulo):
    plt.figure(figsize=(10, 4))
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title(titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frec. [Hz]")
    plt.tight_layout()
    plt.show()

# AUDIO A 8-BITS (BAJAR CALIDAD)
def convertir_a_8bit(audio):
    return np.round(audio * 127) / 127


print("\n+----------------+")
print("| AUDIO ORIGINAL |")
print("+----------------+\n")

nombre_audio1 = "AnalisisTextos2.wav"

# CARGAR AUDIO NORMAL
audio, sr = sf.read(nombre_audio1)

# INFORMACIÓN DEL AUDIO
print("\n📄INFORMACIÓN DEL AUDIO\n")
print("🔹Nombre del archivo: ",nombre_audio1)
print("🔹Largo del vector: ", len(audio))
print("🔹Frecuencia de muestreo: ", sr, "Hz.")
print("🔹Duración del audio: ", len(audio)/sr, "s.")

# VECTOR AUDIO NORMAL
print("🔹Vector: ", audio)
np.savetxt("Vector_Audio_Original.txt", audio) #GUARDA EL VECTOR COMPLETO EN UN TXT

# GRÁFICO DE LA SEÑAL SONORA
tiempo = np.linspace(0, len(audio)/sr, num=len(audio))

plt.figure(figsize=(10, 4))
plt.plot(tiempo, audio, color='steelblue')
plt.title("Waveform - Audio Original")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# REPRODUCIR EL AUDIO NORMAL
print("▶ Audio original:")
display(Audio(audio, rate=sr))


print("\n+--------------------------------------------+")
print("| AUDIO CON FRECUENCIA DE MUESTREO DUPLICADA |")
print("+--------------------------------------------+\n")

# GUARDAR AUDIO CON FRECUENCIA DE MUESTREO MÁS ALTA
sf.write("OutputAudiox2.wav", audio, samplerate=sr*2)

nombre_audio2 = "OutputAudiox2.wav"

# CARGAR AUDIO CON FRECUENCIA DUPLICADA
audio2, sr2 = sf.read(nombre_audio2)

# INFO AUDIO CON FRECUENCIA DUPLICADA
print("\n📄INFORMACIÓN DEL AUDIO\n")
print("🔹Nombre del archivo: ", nombre_audio2)
print("🔹Largo del Array: ", len(audio2))
print("🔹Frecuencia de muestreo: ", sr2, "Hz.")
print("🔹Duración del audio: ", len(audio2)/sr2, "s.")

# VECTOR AUDIO CON FRECUENCIA DUPLICADA
print("🔹Vector: ", audio2)

# GRÁFICO AUDIO CON FRECUENCIA DUPLICADA
tiempo = np.linspace(0, len(audio2)/sr2, num=len(audio2))

plt.figure(figsize=(10, 4))
plt.plot(tiempo, audio2, color='crimson')
plt.title("Waveform - Audio x2")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# AUDIO CON FRECUENCIA DUPLICADA
print("▶ Audio con frecuencia de muestreo duplicada:")
display(Audio(audio2, rate=sr2))


print("\n+-------------------------------------------+")
print("| AUDIO CON FRECUENCIA DE MUESTREO REDUCIDA |")
print("+-------------------------------------------+\n")

# GUARDAR AUDIO CON FRECUENCIA DE MUESTREO MÁS BAJA
sf.write("OutputAudiox0.5.wav", audio, samplerate=int(sr*0.5))

nombre_audio3 = "OutputAudiox0.5.wav"

# CARGAR AUDIO CON FRECUENCIA REDUCIDA
audio3, sr3 = sf.read(nombre_audio3)

# INFORMACIÓN DEL AUDIO CON FRECUENCIA REDUCIDA
print("\n📄INFORMACIÓN DEL AUDIO\n")
print("🔹Nombre del archivo: ", nombre_audio3)
print("🔹Largo del Array: ", len(audio3))
print("🔹Frecuencia de muestreo: ", sr3, "Hz.")
print("🔹Duración del audio: ", len(audio3)/sr3, "s.")

# VECTOR DEL AUDIO CON FRECUENCIA REDUCIDA
print("🔹Vector: ", audio3)

# GRÁFICO DEL AUDIO CON FRECUENCIA REDUCIDA
tiempo = np.linspace(0, len(audio3)/sr3, num=len(audio3))

plt.figure(figsize=(10, 4))
plt.plot(tiempo, audio3, color='darkorange')
plt.title("Waveform - Audio x0.5")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# AUDIO CON FRECUENCIA REDUCIDA
print("▶ Audio con frecuencia de muestreo reducida:")
display(Audio(audio3, rate=sr3))


print("\n+------------------------------------+")
print("| AUDIO DE CALIDAD REDUCIDA (8-BITS) |")
print("+------------------------------------+\n")

# CONVERTIR AUDIO ORIGINAL A 8-BITS
audio8bits = convertir_a_8bit(audio)

# GUARDAR AUDIO 8-BITS
sf.write("OutputAudio8bits.wav", audio8bits, samplerate=sr)

nombre_audio4 = "OutputAudio8bits.wav"

# CARGAR AUDIO 8-BITS
audio8bits, sr = sf.read(nombre_audio4)

# INFO AUDIO 8-BITS
print("\n📄INFORMACIÓN DEL AUDIO\n")
print("🔹Nombre del archivo: ", nombre_audio4)
print("🔹Largo del Array: ", len(audio8bits))
print("🔹Frecuencia de muestreo: ", sr, "Hz.")
print("🔹Duración del audio: ", len(audio8bits)/sr, "s.")

# VECTOR DEL AUDIO CON CALIDAD REDUCIDA
print("🔹Vector: ", audio8bits)

# GRAFICO AUDIO 8-BITS
tiempo = np.linspace(0, len(audio8bits)/sr, num=len(audio8bits))

plt.figure(figsize=(10, 4))
plt.plot(tiempo, audio8bits, color="lightsteelblue")
plt.title("Waveform - Audio baja calidad")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# AUDIO CON CALIDAD BAJA
print("▶ Audio de calidad reducida:")
display(Audio(audio8bits, rate=sr))


# COMPARACIÓN DE WAVEFORMS SEGÚN FRECUENCIA DE MUESTREO
print("\n+---------------------------------------+")
print("| COMPARACIÓN DE FRECUENCIA DE MUESTREO |")
print("+---------------------------------------+\n")

# EJES DE TIEMPO PARA CADA AUDIO
tiempo_original = np.linspace(0, len(audio)/sr, num=len(audio))
tiempo_x2 = np.linspace(0, len(audio2)/sr2, num=len(audio2))
tiempo_x05 = np.linspace(0, len(audio3)/sr3, num=len(audio3))

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(tiempo_original, audio, label="Original (16 kHz)".format(sr), color="steelblue")
ax.plot(tiempo_x2, audio2, label="x2 Frecuencia (32 kHz)".format(sr2), color="crimson", alpha=0.7)
ax.plot(tiempo_x05, audio3, label="x0.5 Frecuencia (8 kHz)".format(sr3), color="darkorange", alpha=0.7)

ax.set_title("Comparación de Waveforms con Diferente Frecuencia de Muestreo")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Amplitud")
ax.grid(True)

# AJUSTES DE CUADRO DE REFERENCIAS
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

plt.tight_layout()
plt.show()


# COMPARACIÓN DE WAVEFORMS: ORIGINAL vs 8 BITS
print("\n+----------------------------------------------------+")
print("| COMPARACIÓN DE WAVEFORMS - AUDIO ORIGINAL Y 8 BITS |")
print("+----------------------------------------------------+\n")

tiempo = np.linspace(0, len(audio)/sr, num=len(audio))

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(tiempo, audio, label='Original (16 bits)', color='steelblue')
ax.plot(tiempo, audio8bits, label='8 Bits (simulado)', color='lightsteelblue', alpha=0.7)

ax.set_title("Comparación de Waveforms - Audio Original vs 8 Bits")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Amplitud")
ax.grid(True)

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

plt.tight_layout()
plt.show()


# ESPECTROGRAMAS COMPARATIVOS AUDIO ORIGINAL Y 8 BITS
print("\n+----------------+")
print("| ESPECTROGRAMAS |")
print("+----------------+\n")

graficar_espectrograma(audio, sr, "Espectrograma - Audio original")
graficar_espectrograma(audio8bits, sr, "Espectrograma - Audio 8-bits")
