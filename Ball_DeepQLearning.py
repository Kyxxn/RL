import tkinter as tk
import math
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import datetime

# Ball 클래스 정의
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
        (x1, y1, x2, y2) = self.canvas.coords(self.id)
        self.x, self.y = x1 + self.r, y1 + self.r

    def CheckCollisionWall(self):
        collision = False
        if self.x - self.r <= 0 or self.x + self.r >= self.canvas.winfo_width():
            self.speed_x = -self.speed_x
            collision = True
        if self.y - self.r <= 0 or self.y + self.r >= self.canvas.winfo_height():
            self.speed_y = -self.speed_y
            collision = True
        return collision

    def CheckCollisionBall(self, ball):
        distance = math.sqrt((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2)
        if distance < self.r + ball.r:
            return True
        return False

    def SetSpeed(self, speed_x, speed_y):
        self.speed_x = speed_x
        self.speed_y = speed_y

    def Delete(self):
        self.canvas.delete(self.id)

# 환경 클래스 정의
class MovingBallsEnv:
    def __init__(self, window):
        self.ball_count = 5
        self.state_size = 4 + self.ball_count * 4  # 상태 크기
        self.action_size = 4  # 행동 공간 크기: 동, 서, 남, 북
        self.window = window
        self.canvas = tk.Canvas(self.window, width=800, height=400, background="white")
        self.canvas.pack(expand=1, fill=tk.BOTH)
        self.player = None

    def Reset(self):
        self.Clear()

        # player 생성
        self.player = Ball(self.canvas, "green", 400, 200, 25, 10, 0)

        # ball들 생성
        self.balls = []
        for i in range(self.ball_count):
            x_pos = random.randint(50, 350) if random.randint(0, 1) == 0 else random.randint(450, 750)
            y_pos = random.randint(50, 150) if random.randint(0, 1) == 0 else random.randint(250, 350)
            x_speed = random.randint(-10, -1) if random.randint(0, 1) == 0 else random.randint(1, 10)
            y_speed = random.randint(-10, -1) if random.randint(0, 1) == 0 else random.randint(1, 10)
            self.balls.append(Ball(self.canvas, "red", x_pos, y_pos, 25, x_speed, y_speed))

        return self.MakeState()

    def Clear(self):
        if self.player:
            self.player.Delete()
            for ball in self.balls:
                ball.Delete()

    def Move(self):
        done = False
        reward = 1  # 생존 보상

        self.player.Move()
        if self.player.CheckCollisionWall():
            done = True
            reward = -2  # 벽과 충돌 시 페널티

        for ball in self.balls:
            ball.Move()
            ball.CheckCollisionWall()
            if self.player.CheckCollisionBall(ball):
                done = True
                reward = -1  # 빨간 공과 충돌 시 페널티
                break

        return self.MakeState(), reward, done

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

# DQN 에이전트 클래스 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.ball_collision_memory = deque(maxlen=20000)
        self.wall_collision_memory = deque(maxlen=20000)
        self.no_collision_memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # 타겟 모델 초기화
        self.scores = []
        self.steps = []
        self.update_target_counter = 0  # 타겟 모델 업데이트 카운터

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, collision_type):
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
        # 모든 메모리에서 샘플링
        total_samples = min(len(self.ball_collision_memory) + len(self.wall_collision_memory) + len(self.no_collision_memory), batch_size)
        if total_samples == 0:
            return

        ball_batch_size = min(len(self.ball_collision_memory), total_samples // 3)
        wall_batch_size = min(len(self.wall_collision_memory), total_samples // 3)
        no_collision_batch_size = total_samples - ball_batch_size - wall_batch_size

        ball_batch = random.sample(self.ball_collision_memory, ball_batch_size) if ball_batch_size > 0 else []
        wall_batch = random.sample(self.wall_collision_memory, wall_batch_size) if wall_batch_size > 0 else []
        no_collision_batch = random.sample(self.no_collision_memory, no_collision_batch_size) if no_collision_batch_size > 0 else []

        minibatch = ball_batch + wall_batch + no_collision_batch

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            target = np.reshape(target, [1, self.action_size])
            self.model.fit(state, target, epochs=1, verbose=0)

        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def SetEpsilon(self, episode, max_episode, decay_ratio=0.5):
        decay_episodes = max_episode * decay_ratio
        if episode >= decay_episodes:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = self.epsilon - (self.epsilon - self.epsilon_min) * (episode / decay_episodes)

    def save_score(self, score, step):
        self.scores.append(score)
        self.steps.append(step)
        log_message = f"Score: {score}, Best Score: {max(self.scores)}, Avg Score: {np.mean(self.scores)}, Steps: {step}\n"
        with open("1002_DeepQLearning_Log.txt", "a") as log_file:
            log_file.write(log_message)

def main():
    global env, agent
    state = env.Reset()
    state = np.reshape(state, [1, env.state_size])
    score = 0
    step_n = 0
    max_steps = 1000
    episode = 0
    max_episode = 1000  # 최대 에피소드 수 설정
    batch_size = 64

    def step():
        nonlocal state, score, step_n, episode

        action = agent.act(state)
        next_state, reward, done = env.Step(action)
        next_state = np.reshape(next_state, [1, env.state_size])

        collision_type = "no_collision"
        if done and reward == -100:
            for ball in env.balls:
                if env.player.CheckCollisionBall(ball):
                    collision_type = "ball"
                    break
            else:
                collision_type = "wall"
        agent.remember(state, action, reward, next_state, done, collision_type)

        state = next_state
        score += reward
        step_n += 1

        if done or step_n >= max_steps:
            agent.save_score(score, step_n)
            if len(agent.no_collision_memory) + len(agent.ball_collision_memory) + len(agent.wall_collision_memory) > batch_size:
                agent.replay(batch_size)
            # 타겟 모델 업데이트
            agent.update_target_counter += 1
            if agent.update_target_counter % 5 == 0:  # 5 에피소드마다 업데이트
                agent.update_target_model()
            # 다음 에피소드를 위해 초기화
            score = 0
            step_n = 0
            state = env.Reset()
            state = np.reshape(state, [1, env.state_size])
            episode += 1
            agent.SetEpsilon(episode, max_episode)

        if episode < max_episode:
            window.after(50, step)
        else:
            print("Training completed")
            window.destroy()

    step()

window = tk.Tk()
env = MovingBallsEnv(window)
agent = DQNAgent(env.state_size, env.action_size)
batch_size = 64

window.after(1000, main)
window.mainloop()