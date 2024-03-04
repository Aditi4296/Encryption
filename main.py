import random

import cv2
from math import log10, sqrt
import numpy as np
from PIL import Image
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from numpy import bitwise_xor, count_nonzero
import math
from nistrng import pack_sequence, unpack_sequence

import pandas

#message to deliver
input_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. In faucibus interdum augue, nec varius eros laoreet id.  

"""

c_array_text =""

global_random_bits = []

def convert_to_8_bit(arr):
    min_val = min(arr)
    max_val = max(arr)

    # Step 2: Normalize the range
    scale_factor = 255 / (max_val - min_val)

    # Step 3: Scale and round
    int_values = abs(np.round((arr - min_val) * scale_factor).astype(np.int8))

    return int_values



# Function to convert integer to 8-bit binary string
def int_to_8bit_binary(n):
    return format(n, '08b')



def convert_into_single_string(result_array):
    new = ""
    for x in result_array:
        new+=x
    return new


def encryption(input_text):
    binary_secret_text = ''.join(format(ord(char), '08b') for char in input_text)
    text_length = len(binary_secret_text)  # Desired length in bits
    seed = 0.6  # Initial value (seed) for logistic map
    random_bits = generate_random_bits(seed, text_length)
    random_bits = [round(x, 2) for x in random_bits]

    print("random bits",random_bits)  # x0,x1,x2


    # Combine values with their indices
    combined = list(enumerate(random_bits))

    # Sort based on the values
    sorted_values_logistic = sorted(combined, key=lambda x: x[1])

    # Extract sorted values and indices separately
    sorted_indices_logistic, sorted_values_logistic = zip(*sorted_values_logistic)
    print("sorted:",sorted_values_logistic)
    print("index:",sorted_indices_logistic)       # key

    binary_array = [int(bit) for bit in binary_secret_text]
    print("binary_secret_text:", binary_secret_text)
    # print(binary_array)

    permutated_binary_text = []   #sorting text_binary_array according to key # (P array)

    print("text binary array", binary_array)


    for x in sorted_indices_logistic:
        permutated_binary_text.append(binary_array[x])


    print("permute",permutated_binary_text)

    converted_values = convert_to_8_bit(np.array(random_bits))
    # print(converted_values)
    converted_random_bits = []
    for val in converted_values:
        myString = int_to_8bit_binary(val).strip("-")
        converted_random_bits.append(myString)
    print("converted random bits", converted_random_bits)   #x array in binary form

    converted_text = convert_to_8_bit(np.array(permutated_binary_text))
    converted_text_bits = []
    # print(converted_text)
    for val in converted_text:
        converted_text_bits.append(int_to_8bit_binary(val))
    print("converted text bits", converted_text_bits)


    # Initialize the result array
    c_array = []

    # for index in range(len(converted_random_bits)):
    #     if(index == len(converted_random_bits) - 1):
    #         result_array.append(str(int(converted_random_bits[index]) ^ int(converted_text_bits[index]) ^ int(converted_text_bits[0])).zfill(8))
    #     else:
    #         result_array.append(str(int(converted_random_bits[index]) ^ int(converted_text_bits[index]) ^ int(converted_text_bits[index+1])).zfill(8))
    for index in range(len(converted_random_bits)):
        c_array.append(str(int(converted_random_bits[index]) ^ int(converted_text_bits[index])).zfill(8))

    print("c_array: ",c_array)


    d_array = []
    for index in range(len(c_array)):
        if(index == len(c_array) - 1):
            d_array.append(xor(c_array[index],c_array[0],8).zfill(8))
        else:
            d_array.append(xor(c_array[index],c_array[index + 1],8).zfill(8))


    print("d_array: ",d_array)

    encrypted_binary_text = convert_into_single_string(c_array)

    global global_random_bits
    global_random_bits = random_bits

    return encrypted_binary_text



# Define tent map function (used in embedding part)
def tent_map(x, a):
    if x < a:
        return x / a
    else:
        return (1 - x) / (1 - a)


# Generate 500x500 random values using tent map
size = (500, 500)
a = 0.7  # Adjust this value for different results
random_values = np.zeros(size)

# Make a random valued array (matrix) of size 500*500
for i in range(size[0]):
    for j in range(size[1]):
        random_values[i, j] = tent_map(random_values[i - 1, j - 1] if i > 0 and j > 0 else np.random.random(), a)

# Flatten the matrix and store indices
flattened_values = random_values.flatten()
indices = np.arange(len(flattened_values))

# Sort the values and corresponding indices
sorted_indices = np.argsort(flattened_values)
sorted_values = flattened_values[sorted_indices]
sorted_indices = indices[sorted_indices]

# Reshape the sorted values and indices back to 2D
sorted_values = sorted_values.reshape(size)
sorted_indices = sorted_indices.reshape(size)


def convert_flat_indices_to_matrix_indices(flat_indices, image_width):
    rows, cols = (500, 500)  # Corrected, should be 500x500
    mat_indices = [[0] * cols for _ in range(rows)]
    i = -1
    for row in flat_indices:
        i = i + 1
        j = -1
        for index in row:
            j = j + 1
            mat_indices[i][j] = (index // image_width, index % image_width)
    return mat_indices



image_width = 500
flat_indices = sorted_indices

matrix_indices = convert_flat_indices_to_matrix_indices(flat_indices, image_width)

text_binary_array = []
def hide_text_with_matrix(image_path, output_path, index_matrix, c_array):
    image = Image.open(image_path).convert('L')
    image.save("grayscale.png")


    key_len = len(c_array)


    single_string_carray = convert_into_single_string(c_array)
    image_capacity = image.width * image.height
    if len(single_string_carray) > image_capacity:
        raise ValueError("Image does not have sufficient capacity to hide the secret text.")

    print("single string carray:",single_string_carray)

    pixels = image.load()
    secret_text_index = 0

    for row in index_matrix:
        for pix in row:
            i = pix[0]
            j = pix[1]
            if (0 <= i) and (i < image.width) and (0 <= j) and (j < image.height):
                pixel = pixels[i, j]
                if secret_text_index > len(single_string_carray) - 1:
                    break

                myString = single_string_carray[secret_text_index].strip("-")
                pixel = (pixel & 0xFE) | int(float(myString))
                secret_text_index += 1
                pixels[i, j] = pixel

    image.save(output_path)
    return key_len


print("text_binary_array",text_binary_array)   #bits array of secret text

def extract_text_from_image(image_path, index_matrix, length):
    image = Image.open(image_path)
    pixels = image.load()

    extracted_bits = ""

    for row in index_matrix:
        for pix in row:
            i = pix[0]
            j = pix[1]

            if (0 <= i) and (i < image.width) and (0 <= j) and (j < image.height):
                pixel = pixels[i, j]
                extracted_bits += str(pixel & 1)
                if len(extracted_bits) >= length * 8:  # Check if we've extracted enough bits
                    break

        if len(extracted_bits) >= length * 8:  # Check if we've extracted enough bits
            break

    #converting extracyed bits into readable message
    extracted_text = ""
    for i in range(0, len(extracted_bits), 8):
        byte = extracted_bits[i:i + 8]
        extracted_text += chr(int(byte, 2))
        if extracted_text[-len(input_text):] == input_text:
            break

    return extracted_bits #the text we obtain from the image


def logistic_map(r, x):
    return r * x * (1 - x)

def generate_random_bits(seed, length):
    bits = []
    x = seed
    for _ in range(length):
        x = logistic_map(3.7, x)  # Example value of r
        bits.append(x)
    return bits




# def decode_binary_string(s):
#     return ''.join(chr(int(s[i*8:i*8+8],2)) for i in range(len(s)//8))

def decode_binary_string(s):
    result = ""
    for i in range(0, len(s), 8):
        binary_chunk = s[i:i+8]
        decimal_value = int(binary_chunk, 2)
        character = chr(decimal_value)
        result += character
    return result


def decryption(string_encrypt, random_bits_decrypt):
    chunk_size = 8
    chunks = [string_encrypt[i:i + chunk_size] for i in range(0, len(string_encrypt), chunk_size)]

    # Print the array of chunks
    print("chunks:", chunks)

    # Combine values with their indices
    combined = list(enumerate(random_bits_decrypt))

    # Sort based on the values
    sorted_values_logistic = sorted(combined, key=lambda x: x[1])

    # Extract sorted values and indices separately
    sorted_indices_logistic, sorted_values_logistic = zip(*sorted_values_logistic)
    print("sorted in decrypt:", sorted_values_logistic)
    print("index in decrypt:", sorted_indices_logistic)  # key

    converted_values = convert_to_8_bit(np.array(random_bits_decrypt))
    print(converted_values)
    converted_random_bits = []
    for val in converted_values:
        converted_random_bits.append(int_to_8bit_binary(val))
    print("converted random bits in decrypt", converted_random_bits)  #x array in binary form

    #calculating permutated binary text (calculating p array)
    p_decrypt = []

    for index in range(len(converted_random_bits)):
        # print("converted_random_bits: ",converted_random_bits)
        # print("chuncks:", chunks[index])
        p_decrypt.append(str(int(converted_random_bits[index]) ^ int(chunks[index])))

    print("p_decrypt",p_decrypt)

    combined = list(enumerate(sorted_indices_logistic))

    # Sort based on the values
    sorted_indices_value = sorted(combined, key=lambda x: x[1])

    # Extract sorted values and indices separately
    sorted_indices_key, sorted_indices_value = zip(*sorted_indices_value)
    print("sorted indices values in decrypt:", sorted_indices_value)
    print("indices index in decrypt:", sorted_indices_key)  # key's key

    t_decrypt =[]

    for x in sorted_indices_key:
        t_decrypt.append(p_decrypt[x])

    print("text_depermute", t_decrypt)

    single_string = convert_into_single_string(t_decrypt)

    return (decode_binary_string(single_string))


def xor(a, b, n):
    ans = ""

    # Loop to iterate over the
    # Binary Strings
    for i in range(n):

        # If the Character matches
        if (a[i] == b[i]):
            ans += "0"
        else:
            ans += "1"
    return ans


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def psnrutil():
    original = cv2.imread("grayscale.png")
    compressed = cv2.imread("output_image.png", 1)
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")



final_encrypted_text = encryption(input_text)
c_array_text = final_encrypted_text

txt_length = hide_text_with_matrix("input_image.jpg", "output_image.png", matrix_indices, c_array_text)
#txt_length gives the length of embedded binary text

extracted_text = extract_text_from_image("output_image.png", matrix_indices, txt_length)
print("Extracted Text:", extracted_text)

final_decrypt_text = decryption(extracted_text, global_random_bits)

print("final decrypt text: ",final_decrypt_text)


#PSNR
psnrutil()

#SSIM
# Load the images
image1 = io.imread("grayscale.png")
image2 = io.imread("output_image.png")

# Convert the images to grayscale if needed
# image1_gray = color.rgb2gray(image1)
# image2_gray = color.rgb2gray(image2)

# Calculate SSIM
ssim_original_stego = ssim(image1, image2)

# Print the result
print(f"SSIM between the images: {ssim_original_stego}")

#BIT ERROR RATE
original_image = np.array(Image.open("grayscale.png"))
final_image = np.array(Image.open("output_image.png"))

original_bits = original_image.flatten()
final_bits = final_image.flatten()

ber = count_nonzero(bitwise_xor(original_bits, final_bits)) / original_bits.size
print(f"Bit Error Rate: {ber}")




"""Lorem ipsum dolor sit amet, consectetur adipiscing elit. In faucibus interdum augue, nec varius eros laoreet id. Vivamus cursus nisi vitae eleifend tempus. Nulla commodo lorem metus, sit amet elementum nulla condimentum vestibulum. Nullam semper lacinia suscipit. Cras tristique eros sem, et vulputate purus mattis id. Vivamus lobortis ac risus nec ultricies. Sed at pulvinar justo. Aliquam a vehicula eros, dictum interdum lacus. Nulla at felis velit. Praesent consequat, ex sed bibendum condimentum, ex tellus tincidunt orci, sed pellentesque dui augue at ante. Aliquam blandit sem felis, sed tempor nunc aliquam mollis. Proin pharetra est nec quam accumsan, sed auctor mi eleifend. Fusce ac tellus eleifend, volutpat felis at, lacinia mauris.

Sed in urna dapibus, venenatis tellus id, placerat lacus. Suspendisse a massa odio. Vivamus eu semper nulla. Praesent vitae augue eu dui eleifend bibendum. In hac habitasse platea dictumst. Duis sed elementum lacus. Sed ac ipsum est. Nullam ornare nunc sed mauris dignissim, ut interdum turpis congue. Quisque ligula enim, eleifend non quam sit amet, euismod ultrices ligula.

Sed vel tristique leo. Fusce nec scelerisque lectus. Duis pellentesque diam id velit ultricies rhoncus. Nam sed hendrerit mauris. Mauris tincidunt blandit pretium. Vestibulum pharetra cursus est, vitae consequat elit. Morbi id imperdiet erat, quis semper odio. Praesent sit amet lacinia lacus. Nunc nulla purus, iaculis eget congue ac, imperdiet a augue. Mauris mollis tincidunt congue. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Phasellus interdum enim a sem feugiat, et commodo est molestie. Integer posuere eros ac justo porttitor consequat. Donec non ligula nec ipsum placerat gravida nec ut lacus. Nunc maximus, nulla eu ornare posuere, ex quam placerat mi, vel tempus quam risus in nisl.

Aenean vel leo vehicula, pharetra purus quis, dictum turpis. Nullam arcu felis, facilisis nec fringilla vel, commodo non dolor. Vestibulum facilisis malesuada ligula vitae congue. Mauris facilisis, enim lobortis varius condimentum, massa elit dictum magna, vel posuere enim ante ut dui. Curabitur at rhoncus risus. Curabitur auctor lectus ex, eu egestas tellus malesuada aliquet. Sed lacus ex, auctor a lectus quis, gravida consequat orci. Praesent quis massa ac ante efficitur elementum.

Ut sit amet finibus quam. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque rhoncus, ante in pulvinar vehicula, nulla ante dignissim justo, id pharetra ipsum enim nec tortor. In vehicula vestibulum tortor, sit amet malesuada orci imperdiet quis. Pellentesque convallis nunc hendrerit, auctor libero vitae, placerat ipsum. Integer at vehicula urna. Vivamus faucibus a nisl ac rutrum. Cras et fringilla leo. Cras imperdiet in turpis quis luctus. Nullam in pretium est, vitae ultricies odio. Fusce ipsum enim, luctus ac elit non, consequat laoreet libero. Phasellus efficitur in magna quis euismod. Integer mollis lacus id diam finibus, ac aliquet mi consectetur. Quisque sagittis erat in mi vehicula, ac accumsan arcu facilisis.

Curabitur at nisi a justo malesuada cursus. Suspendisse fermentum justo ex, nec vehicula neque cursus in. Aenean hendrerit erat lectus, sed tristique mi luctus non. Suspendisse convallis metus at egestas feugiat. Nullam non varius justo, at volutpat velit. Nam sollicitudin, tellus sit amet condimentum rhoncus, erat tellus gravida erat, ac ultricies est mi ut orci. Vivamus at pretium justo, et commodo dolor. Etiam finibus nunc libero, mattis auctor mauris euismod eu. Cras a ullamcorper orci. Proin bibendum augue sed ex condimentum sagittis a eget urna. Ut non felis augue. In neque lectus, varius id neque vel, bibendum porttitor est. Donec sed volutpat leo, ac cursus felis. Vivamus nisl dui, laoreet sit amet dui efficitur, egestas faucibus neque.

Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Pellentesque sollicitudin viverra nisl vel pellentesque. Nunc gravida tellus ex, non dignissim eros imperdiet eget. In malesuada ut arcu ac bibendum. Fusce accumsan efficitur libero, eu pulvinar dolor condimentum a. Duis posuere elementum dolor vel placerat. Nam gravida finibus nunc, ut ultrices erat hendrerit vel. Sed porta neque et convallis tincidunt. Cras vel vulputate dui. Quisque eu elit erat. Vestibulum rhoncus ultricies nunc, auctor ultrices nunc tristique quis.

In condimentum lacus eget massa interdum, quis molestie leo bibendum. Integer vel pulvinar elit, eu euismod mauris. Quisque ac bibendum libero. Pellentesque a urna a leo porta commodo sit amet at mauris. Quisque et ex tincidunt, mattis sem sed, rhoncus orci. Aenean scelerisque risus ut quam auctor, ut malesuada augue posuere. Duis ac luctus dui.

Aenean vitae sem risus. Vivamus in sem et ex sollicitudin bibendum. Nam hendrerit sit amet lorem ut pellentesque. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Maecenas fringilla eget arcu egestas tincidunt. Nullam venenatis convallis quam ac accumsan. Pellentesque laoreet nisl lacus, in iaculis ipsum euismod id. Morbi ultrices accumsan dapibus. Sed facilisis porttitor vulputate. Fusce et varius enim. Suspendisse ultricies tempor erat at efficitur. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nulla pulvinar suscipit massa vitae blandit.

Mauris finibus elit eget arcu placerat sollicitudin. Suspendisse sit amet cursus est. Mauris efficitur tempor dictum. Maecenas porta eu tellus at iaculis. In ut volutpat enim, ac congue ex. Nunc placerat tortor blandit semper bibendum. Aliquam quam lacus, pulvinar vitae ultricies vel, finibus rhoncus enim. Proin eget tellus dapibus, mollis ante et, volutpat magna. Aenean dictum diam eu nibh interdum, a auctor velit tincidunt.

Curabitur eget maximus libero. Morbi non diam at ex consequat dapibus vel id magna. Etiam ut turpis placerat, porta nisi eget, vehicula ante. Ut dui eros, blandit vitae orci ut, semper gravida arcu. Ut non aliquam mi. Integer gravida turpis ut ligula pharetra dictum. Vivamus quis risus lacus. Pellentesque vel lectus felis. Quisque non augue pellentesque odio ullamcorper blandit.

Nullam laoreet a nibh quis accumsan. Morbi ornare ullamcorper ipsum, sed mollis dui aliquet id. Quisque a nibh gravida, dignissim leo sit amet, consequat lectus. Integer rutrum magna eget blandit viverra. Maecenas bibendum, mi vel mattis aliquam, quam justo convallis sapien, sodales dignissim erat nibh nec nisl. Fusce consequat, mauris fermentum congue vulputate, tortor eros malesuada ante, quis molestie dolor augue vel mi. Cras placerat sit amet metus commodo ullamcorper. Aenean vel nibh lorem. Quisque quis suscipit urna. Pellentesque lobortis pellentesque tempor. Fusce viverra libero et gravida porta. Curabitur eros ipsum, hendrerit quis arcu vel, pretium lobortis justo. Fusce non magna feugiat, tincidunt felis dictum, interdum nunc. Fusce sed rhoncus ex, vel bibendum nisl.

Etiam mauris tellus, placerat ut lectus vitae, tincidunt fringilla nisi. Aliquam vel eros vel metus dapibus condimentum posuere sed ex. Donec tortor mi, sollicitudin id volutpat non, porta et justo. Mauris accumsan orci in lectus cursus, sed fermentum lectus tempor. Aenean dignissim luctus nulla. Morbi lacinia erat vel semper ullamcorper. Proin congue tempor accumsan. Maecenas eu velit tortor. Ut nec ex nec leo imperdiet fermentum id eget lacus.

In sed nisl massa. Morbi auctor turpis vel eros feugiat, non tincidunt sem volutpat. Suspendisse potenti. Suspendisse lacus nisi, ultricies nec varius ac, pellentesque nec neque. Proin id massa dolor. Mauris rutrum, nibh ut consectetur finibus, turpis tellus porta urna, nec pellentesque arcu nisl nec massa. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nullam non fermentum purus, quis scelerisque ante. Vestibulum nec sem in risus varius luctus. Aliquam non erat et velit rhoncus ullamcorper.

In sit amet eros at dui malesuada vehicula at ut lorem. Proin consectetur, justo nec dignissim vehicula, tellus ipsum lacinia erat, ut dignissim erat lorem vitae diam. Duis suscipit nisl arcu, sed sagittis lorem rutrum eget. Sed sit amet eros enim. Sed ullamcorper dui in justo venenatis tempor. Nunc quis lacinia ipsum. Nulla massa velit, blandit non nisi ac, auctor vulputate lacus. Curabitur eget aliquet eros. Duis lacus est, fringilla vel diam eget, luctus convallis massa. Sed mauris elit, lacinia nec augue et, ultrices viverra mi. Cras commodo blandit lorem, et condimentum velit pharetra vel. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.

Vivamus quis metus vel leo gravida dignissim. Curabitur dapibus erat at mi maximus, non pharetra elit porta. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis convallis tellus eu ultricies auctor. Vivamus massa nisl, fermentum ut laoreet vel, cursus id nunc. Vestibulum dapibus facilisis scelerisque. Quisque nec leo sed metus venenatis porttitor. Aliquam imperdiet purus odio, sit amet elementum augue vestibulum quis. In fermentum tellus lacinia dui accumsan, in aliquet urna tincidunt.

Sed ut magna et dui porta sollicitudin nec eu ante. Integer ut felis quis felis feugiat ultrices sed id diam. Vestibulum non ex volutpat, accumsan sapien nec, venenatis diam. Curabitur commodo dapibus malesuada. Mauris lobortis urna et orci gravida, sit amet laoreet erat pulvinar. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed nunc ex, hendrerit sit amet posuere a, interdum sit amet libero. Proin ut dapibus nunc, vel convallis est. Ut maximus diam est, non ullamcorper mi fringilla nec.

Nullam orci dolor, fermentum a rutrum in, tincidunt vitae urna. Morbi eu felis sed erat finibus semper sollicitudin nec diam. In aliquam enim pellentesque porttitor mollis. Suspendisse tincidunt vehicula nulla. Morbi dui metus, auctor ac aliquet sit amet, molestie quis erat. Phasellus congue odio neque, maximus tempus felis ultricies quis. Fusce suscipit congue tortor, blandit mattis lacus congue ac. Sed congue ac nunc in pharetra. Aliquam erat volutpat. Donec lacus leo, pretium nec odio a, faucibus egestas neque. Phasellus nec lacinia neque.

Sed lacinia, ante vitae congue ultricies, quam lacus varius nulla, mattis gravida leo ligula in risus. Cras arcu purus, ullamcorper eu eros vel, placerat vestibulum ex. Aenean gravida mattis risus nec blandit. Nullam eu condimentum tellus. Pellentesque iaculis tortor vitae scelerisque lacinia. Integer sed nisi sollicitudin, fermentum quam vel, bibendum lectus. Etiam congue velit quis congue lacinia. Donec ornare vehicula tortor, a ullamcorper turpis auctor eu. Nam est lacus, cursus ut volutpat sit amet, consectetur vel elit.

Donec sit amet leo ultrices, euismod nunc sit amet, varius diam. Vivamus aliquet turpis velit, ut tempor dolor laoreet vel. Nullam id tempus ligula. Praesent ornare velit ut nibh laoreet, quis pretium sem scelerisque. Vivamus ornare arcu non eleifend sollicitudin. Integer ultricies molestie magna, id volutpat purus vehicula vel. Aliquam ac lacus sit amet turpis fermentum malesuada. Nunc varius sem nec feugiat maximus. Donec ac consectetur sapien. Nulla dictum imperdiet dui nec dapibus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Maecenas pretium fringilla ligula, ut hendrerit mi sollicitudin efficitur. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.

Cras tempor purus quis dictum condimentum. Quisque dapibus purus id velit gravida, in faucibus ante rhoncus. Nulla neque dolor, fringilla a condimentum a, tincidunt eget tortor. Nulla facilisi. Sed sollicitudin nulla eu turpis blandit commodo. Etiam laoreet dignissim maximus. Pellentesque ultrices ante ac ultrices tristique. Donec convallis ante et ipsum pretium mollis. In et urna augue. Suspendisse in elementum neque, a porttitor sapien. Sed rhoncus fringilla enim, non blandit neque facilisis at. Phasellus sed cursus purus. Nullam odio risus, blandit a consequat nec, faucibus in nisi. Vestibulum sed lacus non nunc pulvinar bibendum ut nec lorem. Phasellus laoreet neque est, accumsan pulvinar turpis placerat a. Praesent sodales est et nisl lacinia vestibulum.

Praesent massa quam, semper et felis iaculis, varius finibus sem. Donec velit lectus, lacinia ut pretium sit amet, sollicitudin id tortor. Aenean scelerisque nulla quis feugiat finibus. In hac habitasse platea dictumst. Aliquam erat volutpat. Ut vel risus ac ex finibus pretium vel in lacus. Maecenas ultrices porttitor auctor. Suspendisse ullamcorper nisi sodales, feugiat dui eget, iaculis libero. Aenean sem leo, suscipit in pellentesque euismod, imperdiet nec nulla. Mauris est sem, blandit auctor lorem et, euismod tincidunt lectus.

Vivamus sit amet viverra ipsum. Nullam tortor tellus, dictum in lacus ut, accumsan imperdiet nibh. Donec at augue eget orci lacinia dictum quis et arcu. In sit amet tellus et orci bibendum sodales. Praesent enim velit, placerat vel auctor eu, ultricies eget eros. Duis ultricies gravida mauris, vel dignissim libero rutrum in. Suspendisse et magna est. Pellentesque in fringilla ligula. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Suspendisse bibendum velit sit amet pretium feugiat. Vivamus lobortis diam a augue ultrices semper. Pellentesque convallis justo sit amet dui dapibus, eget pellentesque erat iaculis. Aliquam elementum odio quis velit aliquam, a faucibus augue vulputate.

Nulla id fringilla sapien, ac eleifend magna. Mauris volutpat tempor nunc nec faucibus. Maecenas at metus ut risus aliquam tristique a a est. Fusce vitae suscipit elit. Interdum et malesuada fames ac ante ipsum primis in faucibus. Aliquam nunc orci, blandit sit amet pretium at, ultrices ut mi. Phasellus et luctus leo, ac eleifend nunc.

Quisque pharetra nec leo sollicitudin aliquam. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Maecenas tristique dui leo, a viverra ligula pellentesque in. Cras condimentum neque nibh, at rutrum elit euismod id. Curabitur rhoncus sapien vitae tortor pretium interdum. Vivamus in mollis magna, eget molestie eros. Fusce eget velit et diam pulvinar condimentum. Duis efficitur porttitor libero, vitae blandit mi tempus sed. Praesent in mauris ut tortor consequat iaculis nec et velit. Nam porta justo eget ex lobortis ultricies. Phasellus libero nulla, laoreet non porttitor in, tincidunt vulputate erat. Aenean vestibulum in augue vitae pellentesque. Aenean non libero ante.

Aenean in mauris vel quam feugiat convallis. Duis non volutpat sem. Cras consequat, nisi et dictum suscipit, sapien ante ultrices sapien, sed volutpat ligula dolor sed massa. Suspendisse suscipit, lectus in viverra consectetur, est est convallis ante, ut rhoncus leo enim sed justo. Donec id iaculis mi. Nam rutrum nulla eu arcu pharetra, et congue ex dapibus. Ut sagittis tellus et eleifend volutpat. Nunc nec tortor ut mi lobortis tincidunt. Sed dolor nisi, blandit sit amet fermentum at, tempus a leo. Mauris sit amet nibh consequat, condimentum diam et, interdum mauris. Suspendisse tristique felis non turpis accumsan volutpat. Sed vel elit auctor, consectetur sapien vel, euismod dui. Curabitur vel luctus quam, sit amet pulvinar dui. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae;

Donec non tortor et nibh finibus imperdiet. In et leo ac lectus efficitur commodo pellentesque quis tellus. Maecenas blandit diam in vehicula convallis. Etiam placerat quis est sit amet accumsan. Sed sollicitudin dui ligula, id aliquam nisl dignissim non. Aliquam vehicula dignissim diam quis ornare. Maecenas porttitor, turpis non luctus feugiat, est dui congue elit, placerat mattis lorem nisi in diam.

Nunc mauris orci, tristique at lectus in, volutpat tincidunt ante. Curabitur eget quam vel libero volutpat venenatis quis eu ante. Quisque interdum nisl at bibendum tempor. Mauris eleifend tempus aliquet. Donec non eros lorem. Integer sit amet quam a odio finibus accumsan vel vitae risus. Nullam a mi ex. Ut id ante arcu. Morbi tellus nisi, condimentum vitae elementum sed, congue id erat.

Vivamus cursus est vel congue malesuada. Quisque mattis nibh eu gravida ornare. Sed convallis tempor urna vitae dictum. Vestibulum aliquet tincidunt augue, semper faucibus eros porta eget. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Ut hendrerit metus at ligula vestibulum, ut venenatis massa hendrerit. Donec blandit lacinia lectus, nec vulputate nunc commodo in. Nulla porta nec erat vitae vulputate. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Proin fermentum interdum quam in molestie. Cras nec elit et neque sagittis ornare porta ut nibh. Ut id pharetra purus, ut pharetra lacus. Sed cursus ex nec magna fermentum, vel ornare magna commodo.

Suspendisse eget cursus dui. Pellentesque viverra ex ante, at vulputate nunc pulvinar a. Ut dapibus dignissim neque eget bibendum. Vivamus ornare convallis nunc auctor pretium. Pellentesque sit amet scelerisque nisi. Cras at velit vitae nibh eleifend blandit. Nunc fringilla turpis non dignissim mollis. Vivamus porttitor eros sit amet turpis porta convallis. Suspendisse tincidunt pharetra ante, in facilisis mauris pretium sit amet. Nam eleifend libero ac suscipit dignissim. Nullam vel nunc vel nunc auctor posuere non sit amet sem. Suspendisse vehicula ex eu ullamcorper semper. Duis euismod non purus sit amet molestie. Donec fringilla malesuada sapien, nec faucibus turpis tincidunt sit amet. Donec fermentum libero et dolor dapibus consequat.

Proin nec sollicitudin ipsum. Mauris interdum mollis nisi in dictum. Nunc finibus neque nec magna laoreet porta. Vivamus pulvinar, mauris vel lobortis blandit, ipsum nisi lacinia orci, eu ornare ipsum orci eu ipsum. Cras ac ultrices nunc, eget tristique nisi. Ut sodales convallis nisi, et gravida dolor dignissim et. Phasellus semper posuere finibus. Praesent eleifend, nisi vitae bibendum dapibus, mauris purus pellentesque neque, vitae auctor massa leo eu velit. Suspendisse in lacus a orci dignissim venenatis at eget dui. Sed facilisis at felis sit amet eleifend. Pellentesque posuere orci sed ornare imperdiet.

Nam suscipit sagittis mi. Nullam et sem nec ligula venenatis finibus eu aliquam quam. Vivamus aliquam condimentum eleifend. Ut pellentesque, quam eget efficitur lobortis, nibh quam laoreet eros, eget commodo felis nunc non lacus. Nunc sollicitudin tortor nec pretium vestibulum. Nulla semper justo orci, in interdum risus laoreet eu. Praesent lobortis tincidunt sem et sodales. Fusce pellentesque augue sit amet eros tristique, condimentum elementum nunc congue. Proin fringilla lacinia sem, eget scelerisque urna hendrerit sit amet. Integer vel massa gravida, dapibus metus ac, ultricies lorem. Cras interdum justo sed vulputate tempor. Aliquam at arcu ac felis elementum malesuada. Vivamus in odio augue. Morbi efficitur molestie diam, rhoncus consequat dolor lobortis id. Aliquam nunc sapien, elementum et quam in, dapibus aliquam velit.

Morbi eu tortor ligula. Vivamus a magna porta, interdum nibh ac, dignissim est. Curabitur scelerisque ipsum ut eros luctus congue. Nunc ornare laoreet nisl at tincidunt. Aliquam velit magna, vehicula at elementum nec, accumsan quis ante. Etiam dictum posuere nibh vitae vulputate. Cras sed gravida lorem. Duis non fermentum libero, vel cursus nisi. Duis euismod posuere urna et egestas. Vivamus tempor in metus vel efficitur. Nam tristique sapien ut molestie aliquam. Vivamus eu sem nisi. Maecenas rutrum varius mauris nec mattis. Curabitur tincidunt ut augue at volutpat. Sed nunc turpis, lobortis vitae ligula eu, cursus malesuada augue.

Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean eu mi arcu. Nunc in pulvinar felis. Suspendisse consectetur justo purus, a facilisis massa commodo eget. Praesent molestie eros vitae velit ornare egestas. Praesent in nisl sodales, tristique quam non, sollicitudin urna. Ut efficitur libero nec semper bibendum. Nam sem nibh, mollis pretium tortor lobortis, mollis aliquam purus. Quisque sed volutpat tellus. Praesent sed rhoncus est. Vivamus at vehicula sapien. Aenean lorem nisl, cursus vitae leo eu, dignissim porta sem. Vivamus pretium accumsan gravida. Morbi mauris lacus, maximus quis neque ac, tristique fermentum ligula. Quisque porttitor nulla vitae metus dapibus consectetur. Nullam ultricies dolor quis neque placerat ultrices.

Morbi a dictum purus. Nullam sit amet arcu ex. Nam et sollicitudin diam. Vestibulum auctor condimentum dignissim. Duis ornare felis sit amet ipsum aliquet scelerisque sit amet et sem. Proin malesuada lectus vitae est pretium porttitor vel ut felis. Aenean a tellus blandit, cursus mauris sit amet, lobortis lacus. Praesent bibendum, dolor a posuere luctus, elit purus dignissim purus, quis faucibus lectus nisi et nunc. Quisque mattis dapibus ultricies. Aliquam ac commodo augue, nec mollis ex. Mauris vitae metus hendrerit, rhoncus nisi at, egestas odio. Ut vestibulum volutpat sagittis. Integer viverra elit at quam pellentesque eleifend.

Maecenas ac ullamcorper eros. Praesent quam magna, sollicitudin vitae nunc et, euismod tempor tortor. Integer sodales nisi non metus sodales pretium. Fusce vitae elit lacus. Duis bibendum velit sit amet justo dictum facilisis. Vestibulum at risus ac leo pellentesque elementum. Phasellus hendrerit neque a eros imperdiet tristique. Proin luctus maximus sapien ut lacinia.

Donec rutrum nisl purus, ac imperdiet massa luctus ut. Sed maximus mauris ac odio dapibus imperdiet. Praesent vitae elementum nisi. Cras gravida urna non ultricies rhoncus. Mauris id libero et neque posuere consectetur pellentesque sit amet urna. Integer id arcu nisi. Donec ut tempor mauris. Proin vel dui sed orci ullamcorper venenatis. Nulla tempus augue consequat augue pretium posuere. Quisque eget interdum ipsum, ut lobortis ex. Nulla eros mi, condimentum ac ornare vitae, iaculis id ipsum. Nam viverra pharetra enim sed euismod.

Vivamus sagittis, massa at sollicitudin accumsan, ipsum lacus facilisis risus, sit amet rhoncus sapien sem eget quam. Fusce consectetur ligula eu pretium facilisis. Nunc eu risus nec mi dapibus sodales. Ut suscipit nunc at ullamcorper finibus. Aliquam laoreet ullamcorper magna, laoreet consectetur sapien pharetra ut. Pellentesque auctor augue vel arcu pulvinar mattis. Phasellus in convallis tellus. Proin diam erat, blandit ut orci at, suscipit volutpat ante.

Quisque maximus tortor ac orci dictum, non viverra mauris consectetur. Nunc purus ante, accumsan ut tincidunt at, tristique malesuada nulla. In fringilla, eros sit amet imperdiet pretium, dolor libero facilisis diam, at pellentesque mauris mauris eget mi. Nam condimentum, ipsum sed dignissim placerat, orci libero consequat nunc, pulvinar gravida justo turpis quis mauris. Vivamus non convallis odio, a viverra nisi. Ut vulputate fermentum lobortis. Quisque sodales dapibus scelerisque. Praesent eu justo nec neque suscipit iaculis. Donec elementum erat in libero commodo, ut fringilla diam rutrum. Donec leo augue, feugiat in nibh ac, blandit elementum tellus. Donec consectetur metus sit amet nulla ultricies tempus. Suspendisse nec mauris eu nibh finibus aliquam et non arcu. Curabitur tellus nunc, aliquam accumsan tincidunt ac, maximus eget libero. Integer vitae nulla quis orci mattis euismod. In hac habitasse platea dictumst. Nam quis nunc vel lacus dignissim auctor quis non ipsum.

Donec lacus neque, porta at sem id, tristique bibendum purus. Nunc scelerisque est in dui cursus, vel rhoncus erat ultrices. In mollis sit amet tellus in commodo. Duis condimentum ipsum interdum tincidunt accumsan. Donec ultricies arcu est, sit amet pretium sem maximus eget. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras tempus erat eget viverra dapibus. Aliquam scelerisque, enim a iaculis posuere, tellus lorem consectetur felis, ut dignissim dui magna eget sem. Suspendisse eget bibendum ipsum. In tempus pellentesque cursus. Nunc volutpat, lorem ac convallis lacinia, est magna dictum nulla, vitae lacinia eros quam vel tellus.

Cras commodo libero a metus volutpat blandit. Quisque at luctus lacus. Etiam semper at enim pharetra placerat. Cras tempus, erat vel volutpat euismod, velit felis blandit est, et consequat lacus eros quis urna. Morbi nulla odio, convallis et enim sed, pharetra elementum nunc. Phasellus a dolor sagittis, rhoncus sapien sed, maximus diam. Ut mollis ipsum quis libero accumsan, in dignissim enim consectetur. Suspendisse facilisis semper leo. Donec semper ex ut vestibulum vulputate. Praesent vel rutrum mauris, ac dapibus sem.

Cras egestas est turpis, vitae elementum mi pellentesque et. Donec fringilla nisi ac ex tempor, vel semper est fermentum. Nullam pharetra placerat eros eu maximus. Maecenas venenatis id lectus eu congue. Phasellus quis lacus sem. Aenean lacinia dictum fermentum. Donec ut volutpat diam, sit amet faucibus eros. Ut vel felis in erat tempor maximus a et felis. In lectus lectus, efficitur at mauris congue, maximus iaculis libero.

Aliquam tortor dui, hendrerit quis lacus in, rutrum feugiat libero. Nam eleifend nec mauris vel condimentum. Nulla at nisl cursus, scelerisque lectus iaculis, ultricies ante. Aenean et lectus neque. Morbi interdum mi vitae venenatis mollis. Praesent eu posuere quam. Quisque porttitor lobortis mauris, sit amet posuere erat tempor quis. Proin tincidunt varius orci, a mollis arcu placerat a. Sed vitae ipsum vestibulum, vestibulum nisl ac, facilisis nibh.

Vestibulum non justo diam. Duis viverra felis nec sapien lacinia bibendum. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum ullamcorper vitae sapien at dictum. Nam accumsan bibendum aliquet. In ultricies nec sem congue hendrerit. Donec id justo orci. Proin lobortis nulla a est pellentesque vehicula. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

In ac nunc quis justo imperdiet consectetur sollicitudin eget turpis. Phasellus porttitor urna posuere, scelerisque risus non, elementum tortor. In finibus condimentum velit quis pulvinar. Mauris dictum erat eu porta euismod. Sed a venenatis purus. Vivamus in nibh condimentum nunc congue faucibus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Fusce neque lacus, tempor ac augue ut, tincidunt eleifend quam. Aliquam vitae mollis felis.

Ut faucibus diam eu justo commodo, eget iaculis lacus sollicitudin. Nunc dignissim arcu neque, vulputate auctor mauris luctus a. Etiam magna diam, dignissim nec purus vel, dictum ullamcorper est. Sed tincidunt tellus id aliquet vestibulum. Donec varius fermentum tempor. Duis nec ipsum sit amet lacus egestas porta. Maecenas pellentesque dui tellus, nec commodo ligula dignissim eu. Mauris leo purus, condimentum ut fringilla ut, fringilla quis sem. Curabitur a placerat sapien, fringilla eleifend nunc. Proin sodales odio ornare, blandit tortor et, finibus nisi.

Praesent efficitur metus a maximus semper. Nullam id nunc nec mauris hendrerit gravida quis eget ante. Sed sed eros eget nibh fermentum dapibus ut interdum quam. Etiam tincidunt finibus sem sit amet blandit. Nunc odio justo, pretium a maximus pharetra, tempor a nisl. Aenean sed posuere nulla, et varius quam. In quis magna nunc. Proin id nisi eu est tincidunt porttitor quis eu dui. Cras efficitur hendrerit lacinia. Pellentesque diam turpis, facilisis eget arcu at, consequat aliquet lacus. Donec eleifend, purus vel commodo lobortis, nunc dolor consequat orci, vitae vestibulum ante tellus id quam. Donec vulputate volutpat turpis, a semper nibh ultricies vel. Donec tristique malesuada ex sed ultricies. Suspendisse vulputate felis sit amet arcu interdum rutrum.

Ut ac ornare erat. Sed fermentum nibh odio, non posuere quam luctus nec. Phasellus eleifend ultricies quam eu vestibulum. Maecenas a varius magna. Aliquam in molestie massa, nec laoreet urna. Nam molestie dictum sapien nec rutrum. Sed ultrices quis turpis eget fringilla. Pellentesque ut justo nunc. In hac habitasse platea dictumst. Donec porta quis arcu eget venenatis.

Proin sed nulla ultricies, consectetur augue non, auctor metus. Nullam quis ipsum mollis, aliquam neque vitae, posuere velit. Vestibulum condimentum mi quis lorem sollicitudin, eget condimentum lacus volutpat. Suspendisse potenti. Cras turpis odio, eleifend fermentum eleifend sit amet, varius non magna. Aliquam et condimentum elit. Integer rutrum arcu eros, quis posuere nunc volutpat tempus.

Suspendisse potenti. Duis tincidunt libero ac interdum ultrices. Curabitur consectetur ex ex, nec imperdiet ligula eleifend maximus. Pellentesque pretium convallis mauris vitae consequat. Pellentesque quis convallis ipsum. Nullam eleifend elit nisi, ac tempus est tincidunt et. Mauris nunc orci, lobortis a cursus vel, euismod ac justo. Mauris feugiat sodales magna eu dapibus. Phasellus enim mi, molestie vitae augue eget, tempus scelerisque sapien. Sed semper, diam sit amet fringilla efficitur, justo ligula vulputate velit, id finibus felis nibh at mi. In hac habitasse platea dictumst. Nam a sem enim. Ut ut leo vitae tellus accumsan porta eget vehicula libero. Duis ullamcorper arcu dictum justo tempor elementum.

Integer eget ante id lectus cursus interdum. Sed eget dolor at dolor egestas suscipit. Suspendisse erat sem, efficitur quis posuere sed, tincidunt in ex. Maecenas fermentum egestas sapien, ac mollis odio hendrerit sed. Maecenas consequat diam id dignissim volutpat. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce sit amet sem quam. Vestibulum a elementum massa, nec commodo leo.

Sed tincidunt nibh at viverra interdum. Pellentesque consectetur massa quis tristique vestibulum. Integer eleifend fermentum malesuada. Mauris rutrum metus id pharetra feugiat. Fusce nunc quam, faucibus vitae vestibulum vel, ultrices sit amet mauris. Sed molestie mollis ex, a euismod tellus congue sed. Donec ex odio, laoreet at tellus vel, egestas dictum velit. Duis quis arcu in tellus lobortis porta. Nullam malesuada nulla quam, sit amet pellentesque ante sodales non. Donec bibendum ullamcorper lorem ut facilisis. Mauris non neque lacus.

Vestibulum malesuada purus at nibh luctus tristique eget quis lorem. Cras sed aliquam urna. Etiam quis urna blandit, eleifend dolor eu, dictum justo. Maecenas eu purus vitae neque molestie tincidunt ut ac tortor. Quisque iaculis malesuada felis quis euismod. Sed varius lorem turpis, a vestibulum lorem iaculis ut. Praesent aliquam tempor ante eu porta. Sed cursus ullamcorper lectus sit amet blandit. Aliquam a sapien non turpis finibus finibus sed eu tellus. Phasellus sit amet blandit metus, tempus elementum diam. Nam a rutrum ipsum, venenatis tristique erat. Mauris sit amet velit in ex facilisis volutpat sed ut risus. Fusce vitae dolor porttitor, dignissim morbi.

"""
