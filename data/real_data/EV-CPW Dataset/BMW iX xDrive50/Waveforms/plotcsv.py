import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV，跳过前 4 行（Trigger_Date ~ Microseconds_Per_Sample）
df = pd.read_csv("/home/changhong/prj/finn_dev/finn/script/DATE2025/TinyHarmonic/data/real_data/EV-CPW Dataset/BMW iX xDrive50/Waveforms/Waveform_1.csv", skiprows=4)

# 打印前几行检查
print(df.head())

# 绘制波形
plt.figure(figsize=(10, 6))

# 电压
#plt.plot(df["Time (ms)"], df["Voltage (V)"], label="Voltage (V)", color="blue")

# 电流
plt.plot(df["Time (ms)"], df["Current (A)"], label="Current (A)", color="red")

# 图形修饰
plt.title("Voltage & Current Waveforms")
plt.xlabel("Time (ms)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.show()
