import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
import math

class Ball:
    def __init__(self, x, y, r, speed_x, speed_y):
        self.x = x                              # 중심 좌표
        self.y = y
        self.r = r                              # 반지름
        self.speed_x = speed_x                  # 속도
        self.speed_y = speed_y

    def Move(self):
        self.x += self.speed_x
        self.y += self.speed_y

    def CheckCollisionWall(self, width, height):
        collision = False                          # 벽과의 충돌 여부
        if self.x - self.r <= 0 or self.x + self.r >= width:
            self.speed_x = -self.speed_x
            collision = True
        if self.y - self.r <= 0 or self.y + self.r >= height:
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

class MovingBallsEnv:
    def __init__(self):
        self.width = 800
        self.height = 400
        self.ball_count = 5
        self.state_size = 4 + self.ball_count * 4  # 인공 신경망 입력
        self.action_size = 4  # 행동 공간: 동, 서, 남, 북

        self.player = None
        self.balls = []

    def Reset(self):
        self.player = Ball(400, 200, 25, 10, 0)

        self.balls = []
        for _ in range(self.ball_count):
            self.balls.append(Ball(random.randint(100, 700), random.randint(100, 300), 25,
                              random.randint(1, 10), random.randint(1, 10)))

        return self.MakeState()

    def Move(self):
        done = False
        reward = 1

        self.player.Move()
        if self.player.CheckCollisionWall(self.width, self.height):
            done = True
            reward = 0

        for ball in self.balls:
            ball.Move()
            ball.CheckCollisionWall(self.width, self.height)

        for ball in self.balls:
            if self.player.CheckCollisionBall(ball):
                done = True
                reward = 0
                break

        return self.MakeState(), reward, done

    def MakeState(self):
        state = [self.player.x, self.player.y, self.player.speed_x, self.player.speed_y]
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.speed_x, ball.speed_y])
        return np.array(state)

    def Step(self, action):
        if action == 0:
            self.player.SetSpeed(10, 0)
        elif action == 1:
            self.player.SetSpeed(-10, 0)
        elif action == 2:
            self.player.SetSpeed(0, 10)
        elif action == 3:
            self.player.SetSpeed(0, -10)

        return self.Move()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = MovingBallsEnv()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.Reset()
        state = np.reshape(state, [1, state_size])

        for time_t in range(500):
            action = agent.act(state)
            next_state, reward, done = env.Step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time_t}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(f"dqn_ball_e{e}.h5")
