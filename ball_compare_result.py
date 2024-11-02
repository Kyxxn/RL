import matplotlib.pyplot as plt

def PrintResult(data, title):
    print(f"<< {title} >>")
    print(f"sum : {sum(data)}")
    print(f"avg : {sum(data) / len(data):.2f}")
    print(f"count : {data.count(1000)}")

file = open("DQN_Result/one_replay_memory_01.txt", "r")
one_replay_memory = eval(file.read())
file.close()

file = open("DQN_Result/multi_replay_memory_05.txt", "r")
multiple_replay_memory = eval(file.read())
file.close()

PrintResult(one_replay_memory, "One Replay Memory")
PrintResult(multiple_replay_memory, "Multiple Replay Memory")

window_size = 1000
x1 = range(window_size, 100001)
y_one = [sum(one_replay_memory[i-window_size:i]) / window_size for i in range(window_size, 100001)]
y_multiple = [sum(multiple_replay_memory[i-window_size:i]) / window_size for i in range(window_size, 100001)]
plt.plot(x1, y_one)
plt.plot(x1, y_multiple)

plt.title("1000-episode Moving Average Step Count")
plt.xlim(0, 101000)
plt.ylim(0, 350)
plt.xlabel("episode")
plt.ylabel("step count")

plt.show()