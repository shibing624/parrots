from pydub import AudioSegment
from pydub.silence import split_on_silence
import json
from pprint import pprint
from pathlib import Path
import sys
with open('mapping.json') as data_file:    
    data = json.load(data_file)

# keys = sorted(data.keys())
if len(sys.argv) != 2:
	print("usage:")
	print("      python process.py {key}")
	print("to process ./recording/{key}.wav")
	sys.exit(1)

key = sys.argv[1:][0]

syllables = data[key]

path = "./recording/"+key+".wav"

file = Path(path)
if not file.is_file():
	raise Exception(path+" doesn't exist")

sound_file = AudioSegment.from_wav(path)

audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least 300ms
    min_silence_len=300,
    # consider it silent if quieter than -48 dBFS
    silence_thresh=-48
)

flag = False
if len(syllables)*5 != len(audio_chunks):
	flag = True

for i, chunk in enumerate(audio_chunks):
	syllable = syllables[i//5]
	print(syllable)
	j = i % 5
	if j != 4: # 1st, 2nd, 3rd, 4th tone
		out_file = "./pre/"+ syllable + str(j+1)+".wav"
	else: # neutrual tone
		out_file = "./pre/"+ syllable +".wav"


	# out_file = "./pre/"+ str(i)+".wav"
	print("exporting", out_file)
	chunk.export(out_file, format="wav")

print(key, len(audio_chunks))
if flag:
	print(key, "doesn't match expected audio chunks length")
	print("expected:", len(syllables)*5, "actual:", len(audio_chunks))



