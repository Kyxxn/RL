# count_1000.py

import ast

def count_1000_in_file(filename):
    count = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 문자열을 리스트로 변환
                data_list = ast.literal_eval(line.strip())
                # 리스트 안의 1000의 개수를 세기
                count += data_list.count(1000)
            except (SyntaxError, ValueError):
                print(f"Error parsing line: {line}")
    return count

if __name__ == "__main__":
    count_1000 = count_1000_in_file('multi_replay_memory_05.txt')
    print(f"Number of occurrences of 1000: {count_1000}")