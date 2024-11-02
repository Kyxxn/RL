import tkinter as tk
import math
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout

result_file = open("DQN_Result/Episode 별 분류/DQN_RESULT2.txt", "w", encoding="utf-8")
step_1000_count = 0

class Ball:
    def __init__(self, canvas, color, x, y, r, speed_x, speed_y):
        self.canvas = canvas
        self.color = color
        self.x = x
        self.y = y
        self.r = r
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.id = canvas.create_oval(x - r, y - r, x + r, y + r, fill=color)

    def Move(self):
        self.canvas.move(self.id, self.speed_x, self.speed_y)
        self.canvas.update()

        # 현재 좌표 업데이트
        (x1, y1, x2, y2) = self.canvas.coords(self.id)
        self.x, self.y = x1 + self.r, y1 + self.r

    def CheckCollisionWall(self):
        collision = False

        # 좌우 벽 충돌 체크
        if self.x - self.r <= 0 or self.x + self.r >= self.canvas.winfo_width():
            self.speed_x = -self.speed_x
            collision = True

        # 상하 벽 충돌 체크
        if self.y - self.r <= 0 or self.y + self.r >= self.canvas.winfo_height():
            self.speed_y = -self.speed_y
            collision = True
        return collision

    def CheckCollisionBall(self, ball):
        distance = math.hypot(self.x - ball.x, self.y - ball.y)
        return distance < self.r + ball.r

    def SetSpeed(self, speed_x, speed_y):
        self.speed_x = speed_x
        self.speed_y = speed_y

    def Delete(self):
        self.canvas.delete(self.id)

class MovingBallsEnv:
    def __init__(self, window):
        self.ball_count = 5
        self.state_size = 4 + self.ball_count * 4
        self.action_size = 4
        self.window = window

        # 캔버스 생성
        self.canvas = tk.Canvas(self.window, width=800, height=400, background="white")
        self.canvas.pack(expand=1, fill=tk.BOTH)
        self.player = None

    def Reset(self):
        self.Clear()
        self.player = Ball(self.canvas, "green", 400, 200, 25, 10, 0)
        self.balls = []
        for i in range(self.ball_count):
            x_pos = random.choice([random.randint(50, 350), random.randint(450, 750)])
            y_pos = random.choice([random.randint(50, 150), random.randint(250, 350)])
            x_speed = random.choice([random.randint(-10, -1), random.randint(1, 10)])
            y_speed = random.choice([random.randint(-10, -1), random.randint(1, 10)])
            self.balls.append(Ball(self.canvas, "red", x_pos, y_pos, 25, x_speed, y_speed))

        return self.MakeState()

    def Clear(self):
        if self.player:
            self.player.Delete()
            for ball in self.balls:
                ball.Delete()

    def Move(self):
        done_type = 0
        done = False
        reward = 1
        self.player.Move()

        # 벽 충돌 체크
        if self.player.CheckCollisionWall():
            done = True
            reward = 0
            done_type = 1

        # 적 공들 이동 및 충돌 체크
        for ball in self.balls:
            ball.Move()
            ball.CheckCollisionWall()
            if self.player.CheckCollisionBall(ball):
                done = True
                reward = 0  # 적 공 충돌 패널티
                done_type = 2
                break

        return self.MakeState(), reward, done, done_type

    def MakeState(self):
        state = [self.player.x, self.player.y, self.player.speed_x, self.player.speed_y]
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.speed_x, ball.speed_y])
        return np.array(state)

    def Step(self, action):
        if action == 0:  # 동
            self.player.SetSpeed(10, 0)
        elif action == 1:  # 서
            self.player.SetSpeed(-10, 0)
        elif action == 2:  # 남
            self.player.SetSpeed(0, 10)
        elif action == 3:  # 북
            self.player.SetSpeed(0, -10)
        return self.Move()

class DQNAgent:
    def __init__(self, state_size, action_size):
        # 에이전트 초기 설정
        self.state_size = state_size
        self.action_size = action_size

        # 리플레이 메모리 초기화
        self.ball_collision_memory = deque(maxlen=20000)
        self.wall_collision_memory = deque(maxlen=20000)
        self.no_collision_memory = deque(maxlen=20000)

        self.no_collision_batch_size = 32
        self.wall_collision_batch_size = 64
        self.ball_collision_batch_size = 64

        self.gamma = 0.95
        self.epsilon_start = 0.3
        self.epsilon = self.epsilon_start
        self.epsilon_min = 0.01 # 기존 0.01 -> 0.001
        self.epsilon_decay = 0.9995 # 기존 0.995 -> 0.9995
        self.learning_rate = 0.0001     # 황준하

        # 모델 생성
        self.model = self._build_model()
        self.target_model = self._build_model()

        # 타겟 모델 초기화
        self.update_target_model()

        # 성능 기록용 리스트
        self.scores = []
        self.steps = []
        self.update_target_counter = 0

    def _build_model(self):
        model = Sequential()
        initializer = tf.initializers.glorot_normal()
        model.add(Dense(4096, input_dim=self.state_size, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(2048, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(1024, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(512, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(256, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(128, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(64, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(32, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(16, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer=initializer, bias_initializer=initializer))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # 타겟 모델 업데이트
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, collision_type):
        # 리플레이 메모리에 저장
        if collision_type == "ball":
            self.ball_collision_memory.append((state, action, reward, next_state, done))
        elif collision_type == "wall":
            self.wall_collision_memory.append((state, action, reward, next_state, done))
        else:
            self.no_collision_memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.no_collision_memory, self.no_collision_batch_size)

        ball_batch_size = min(len(self.ball_collision_memory), self.ball_collision_batch_size)
        ball_mini_batch = random.sample(self.ball_collision_memory, ball_batch_size)
        mini_batch.extend(ball_mini_batch)

        wall_batch_size = min(len(self.wall_collision_memory), self.wall_collision_batch_size)
        wall_mini_batch = random.sample(self.wall_collision_memory, wall_batch_size)
        mini_batch.extend(wall_mini_batch)

        return mini_batch

    def train(self, mini_batch):
        batch_size = len(mini_batch)

        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        target = self.model.predict(states, verbose = 0)
        target_val = self.target_model.predict(next_states, verbose = 0)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_val[i])

        self.model.fit(states, target, epochs = 1, verbose = 0)

    # 황준하
    def SetEpsilon(self, episode, max_episode, decay_ratio=0.5):
        decay_episodes = max_episode * decay_ratio
        epsilon = 1 - episode / decay_episodes
        epsilon *= self.epsilon_start - self.epsilon_min
        epsilon += self.epsilon_min
        self.epsilon = max(self.epsilon_min, epsilon)

def main():
    global env, agent, result_file, step_1000_count

    score = 0
    step_n = 0
    max_steps = 1000
    max_episode = 100000
    batch_size = 64
    steps_per_episode = []
    scores = []
    steps = []

    for episode in range(max_episode):
        score = 0
        state = env.Reset()
        state = np.reshape(state, [1, env.state_size])
        agent.SetEpsilon(episode, max_episode)
        step_n = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, done_type = env.Step(action)
            next_state = np.reshape(next_state, [1, env.state_size])

            # 충돌 유형 판단
            collision_type = "no_collision"
            if done_type == 1:
                collision_type = "wall"
            elif done_type == 2:
                collision_type = "ball"

            # 경험 저장
            agent.remember(state, action, reward, next_state, done, collision_type)
            state = next_state
            score += reward
            step_n += 1
            if step_n >= 1000:
                done = True

            # 에피소드 종료 처리
            if done:
                # 에피소드 기록
                agent.scores.append(score)
                agent.steps.append(step_n)
                steps_per_episode.append(step_n)

                if len(agent.no_collision_memory) >= agent.no_collision_batch_size:
                    minibatch = agent.replay(batch_size)
                    agent.train(minibatch)

                # 타겟 모델 업데이트
                agent.update_target_counter += 1
                if agent.update_target_counter % 5 == 0:
                    agent.update_target_model()

                scores.append(score)
                steps.append(step_n)

                if step_n == 1000:
                    step_1000_count += 1

                # 에피소드 결과를 파일에 저장
                result_file.write(f"에피소드: {episode}, step: {step_n}, step이 1000번 넘은 횟수: {step_1000_count}\n")
                result_file.flush()

                break

window = tk.Tk()
env = MovingBallsEnv(window)
agent = DQNAgent(env.state_size, env.action_size)
batch_size = 64

window.after(1000, main)
window.mainloop()
