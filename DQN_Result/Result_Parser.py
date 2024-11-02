# Result_Parser.py

def read_steps(filename):
    steps = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 각 줄에서 'step' 다음에 오는 숫자만 추출
            parts = line.split(', ')
            for part in parts:
                if part.startswith('step:'):
                    # 숫자만 추출하기 위해 문자열을 정리
                    step_str = part.split(': ')[1].strip().rstrip(',')
                    try:
                        step_value = int(step_str)
                        steps.append(step_value)
                    except ValueError:
                        print(f"Error converting '{step_str}' to int.")
                    break
    return steps

def save_steps_to_file(steps, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(str(steps))

if __name__ == "__main__":
    steps = read_steps('Episode 별 분류/DQN_RESULT.txt')
    save_steps_to_file(steps, 'multi_replay_memory_02.txt')
    print("Steps saved to multi_replay_memory_05.txt")