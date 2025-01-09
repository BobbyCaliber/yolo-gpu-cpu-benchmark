import psutil
import cpuinfo
import pandas as pd

# Получение общей информации о CPU
cpu_info = cpuinfo.get_cpu_info()
dict = {}
processor = cpu_info.get('brand_raw')
freq = psutil.cpu_freq().current
freq_max = psutil.cpu_freq().max
cores = psutil.cpu_count(logical=False)
threads = psutil.cpu_count(logical=True)
l1_cache = cpu_info.get('l1_data_cache_size', 'Unknown')
l1_instruction_cache_size = cpu_info.get('l1_instruction_cache_size', 'Unknown')
l2_cache_per_core = cpu_info.get('l2_cache_size', 'Unknown') / cores / 1e3
l2_cache_all = cpu_info.get('l2_cache_size', 'Unknown') / 1e3
l3_cache_all = cpu_info.get('l3_cache_size', 'Unknown') / 1e6
smp = psutil.cpu_count(logical=True) > psutil.cpu_count(logical=False)

dict['Processor'] = processor
dict['Frequency (MHz)'] = freq
dict['Max Frequency (MHz)'] = freq_max
dict['Cores'] = cores
dict['Threads'] = threads
dict['l1_cache_size (KB)'] = l1_cache
dict['l1_instruction_cache_size (KB)'] = l1_instruction_cache_size
dict['l2_cache_size_per_core (KB)'] = l2_cache_per_core
dict['l2_cache_size (KB)'] = l2_cache_all
dict['l3_cache_size (MB)'] = l3_cache_all
dict['SMP Supported'] = smp


result = pd.DataFrame(dict, index=[0])
result.to_csv('data/cpu_config.csv', index=False)