import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd

# Инициализация CUDA
cuda.init()

# Получение информации о текущем GPU
device = cuda.Device(0)  # Используем первый GPU
name = device.name()
num_cores = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)  # Количество SM
clock_rate = device.get_attribute(cuda.device_attribute.CLOCK_RATE)  # Частота в кГц
clock_rate_ghz = clock_rate / 1e6  # Преобразуем в ГГц

# Определяем архитектуру и параметры для расчета FLOPS
compute_capability = device.compute_capability()
if compute_capability >= (8, 0):  # Ampere (RTX 30xx, A100, H100)
    if "A100" in name:  # Специализированные GPU для обучения
        tensor_cores_per_sm = 4  # tensor cores
        cores_per_sm = 192

        fp64_ratio = 1 / 2
        fp16_ratio = 4
        fp64_tensor_ratio = 1
        tf_32_ratio = 8
        bf_16_ratio = 16

    elif "H100" in name:  # Специализированные GPU для обучения  # !!!! для h100 вообще cores per sm 256
        tensor_cores_per_sm = 4  # tensor cores
        cores_per_sm = 256

        fp64_ratio = 1 / 2
        fp16_ratio = 4
        fp64_tensor_ratio = 0
        tf_32_ratio = 0
        bf_16_ratio = 0

    else:  # Обычные Ampere GPU (RTX 30xx) и 40xx
        tensor_cores_per_sm = 4  # tensor cores
        cores_per_sm = 128

        fp64_ratio = 1 / 64
        fp16_ratio = 1
        fp64_tensor_ratio = 0
        tf_32_ratio = 0
        bf_16_ratio = 0

elif compute_capability == (7, 5):  # Turing (RTX 20xx, Quadro RTX)
    if "GTX 16" in name:  # GTX 16xx без Tensor Cores
        tensor_cores_per_sm = 0
        cores_per_sm = 64

        fp64_ratio = 1 / 32
        fp16_ratio = 2
        fp64_tensor_ratio = 0
        tf_32_ratio = 0
        bf_16_ratio = 0

    else:  # RTX 20xx и Quadro RTX                                            #!!! чекнуть про 20 серию и из тюринга про fp
        tensor_cores_per_sm = 8  # тут сверяться с gpu database насчет fp ratio остальные fp расчитываются засчет дроби и fp32
        cores_per_sm = 64

        fp64_ratio = 1 / 32
        fp16_ratio = 2
        fp64_tensor_ratio = 0
        tf_32_ratio = 0
        bf_16_ratio = 0

elif compute_capability == (7, 0):  # Volta (Tesla V100)
    tensor_cores_per_sm = 8
    cores_per_sm = 128

    fp64_ratio = 1 / 2
    fp16_ratio = 2
    fp64_tensor_ratio = 0
    tf_32_ratio = 0
    bf_16_ratio = 0

else:
    tensor_cores_per_sm = 0
    cores_per_sm = 64

    fp64_ratio = 1 / 32
    fp16_ratio = 0
    fp64_tensor_ratio = 0
    tf_32_ratio = 0
    bf_16_ratio = 0

# Подсчет общего количества CUDA-ядер
total_cores = num_cores * cores_per_sm
# Подсчет общего количества тензор-ядер
num_tensor_cores = num_cores * tensor_cores_per_sm

# FP32 FLOPS
fp32_flops = total_cores * 2 * clock_rate_ghz / 1000  # для терафлопс                   # это основные flops это тут база
fp64_flops = fp32_flops * fp64_ratio
fp16_flops = fp32_flops * fp16_ratio
fp64_tensor_flops = fp64_flops * fp64_tensor_ratio
tf32_flops = fp64_flops * tf_32_ratio
bf16_flops = fp32_flops * bf_16_ratio

dict = {}
dict['GPU Name'] = name
dict['Compute Capability (cuda version)'] = f'{compute_capability}'

dict['Number of SMs'] = num_cores
dict['Cores per SM'] = cores_per_sm
dict['Total CUDA Cores'] = total_cores
dict['Number of Tensor Cores'] = num_tensor_cores

dict['Clock Rate (GHz) boost'] = clock_rate_ghz

dict['FP32 FLOPS (TFLOPS)'] = fp32_flops
dict['TF32 FLOPS (TFLOPS)'] = tf32_flops

dict['FP16 FLOPS (TFLOPS)'] = fp16_flops
dict['BF16 FLOPS (TFLOPS)'] = bf16_flops

dict['FP64 FLOPS (GFLOPS)'] = fp64_flops * 1000
dict['FP64 TENSOR FLOPS (TFLOPS)'] = fp64_tensor_flops

# Вывод результатов
result = pd.DataFrame(dict, index=[0])
result.to_csv('data/gpu_config.csv', index=False)