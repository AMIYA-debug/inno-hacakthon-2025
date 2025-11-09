input_file="model.tflite"
output_file="model_data.h"

with open(input_file,"rb") as f:
    data=f.read()

with open(output_file,"w") as f:
    f.write("const unsigned char model_tflite[]={\n")
    for i,b in enumerate(data):
        if i%12==0:
            f.write("    ")
        f.write(f"0x{b:02x}, ")
        if (i+1)%12==0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const int model_tflite_len={len(data)};\n")

