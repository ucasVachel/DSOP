import numpy as np
import gym
from gymnasium import spaces
import time
import random
import math
import pygame
import os
import json


class ZJEnv(gym.Env):
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 50
    }

    # def __init__(self, stepsize):
    def __init__(self, map_id):
        self.h = 229
        self.w = 240
        self.stepsize = 80.0
        self.cost_k = 0.05

        self.destination_range = 15
        self._max_episode_steps = 10
        self.path = []

        with open('old\\annotations.json') as file:
            # map id
            self.data = json.load(file)[map_id]

        self.name = self.data['rs_name'][:-4]

        self.obstacles = np.load(os.path.join('old\\paper_obs', self.name) + '.npy')

        self.origin = (self.data['start'][1], self.data['start'][0])
        self.destination = (self.data['end'][1], self.data['end'][0])
        self.befor_currrnt = None

        self.origin_distance = self.calculate_distance(self.origin, self.destination)

        self.action_space = spaces.Box(low=-self.stepsize, high=self.stepsize, shape=(2,), dtype=np.float32)

        self.single_action_space = spaces.Box(low=-self.stepsize,
                                              high=self.stepsize,
                                              shape=(2,),
                                              dtype=np.float32)

        self.single_observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]),
                                       high=np.array([self.h, self.w, self.calculate_distance(self.origin, self.destination)]), dtype=np.float32)

    def generate_buffer_line(self, new_x, new_y):
        self.line_buffer = round(self.calculate_distance((new_x, new_y), (self.current)) * 0.2)

        cost = 0

        dx = new_x - self.current[0]
        dy = new_y - self.current[1]
        slope = dy / dx if dx != 0 else 0
        intercept = self.current[1] - slope * self.current[0]

        # 边界
        min_x = round(min(self.current[0], new_x))
        # min_y = round(min(self.current[1], new_y))
        max_x = round(max(self.current[0], new_x))
        # max_y = round(max(self.current[1], new_y))

        # 扩宽边界
        min_x = min_x if (min_x - self.line_buffer < 0) else (min_x - self.line_buffer)
        # min_y = min_y if (min_y - self.line_buffer < 0) else (min_y - self.line_buffer)
        max_x = max_x if (max_x + self.line_buffer >= self.h) else (max_x + self.line_buffer)
        # max_y = max_y if (max_y + self.line_buffer > self.w) else (max_y + self.line_buffer)

        for i in range(min_x, max_x):
            test_y = int(slope * i + intercept)

            if test_y < 0 or test_y > self.w:
                continue

            # 遍历y方向上的点
            min_y = test_y if test_y - self.line_buffer < 0 else test_y - self.line_buffer
            max_y = test_y if test_y + self.line_buffer > self.w else test_y + self.line_buffer

            for j in range(min_y, max_y):
                if self.obstacles[i][j] == 2:
                    cost += 1
        return cost

    def terminated(self, new_x, new_y):
        # 是否出界
        out_boundry = new_x < 0 or new_x >= self.h or new_y < 0 or new_y >= self.w
        if out_boundry:
            return True, "out"

        # 是否在障碍里放了杆塔
        if self.obstacles[new_x][new_y] > 0:
            return True, "building"

        # 是否穿过了建筑
        # 计算截距和斜率
        dx = new_x - self.current[0]
        dy = new_y - self.current[1]
        slope = dy / dx if dx != 0 else 0
        intercept = self.current[1] - slope * self.current[0]
        # 这条线x的开始到结束
        for i in range(min(round(self.current[0]), round(new_x)), max(round(self.current[0]), round(new_x))):
            test_y = int(slope * i + intercept)
            # 是否穿过了建筑
            if self.obstacles[i][test_y] == 2:
                return True, "across"

        if self.episodes >= self._max_episode_steps:
            return True, "max"

        if np.sqrt((self.destination[0] - new_x) ** 2 + (self.destination[1] - new_y) ** 2) <= self.destination_range:
            return True, "Get!"

        return False, "ok"

    def distance_point_to_line(self, slope, intercept, point):
        x0, y0 = point
        distance = abs(slope * x0 - y0 + intercept) / math.sqrt(slope ** 2 + 1)
        return distance

    def around_cost(self, new_x, new_y):
        cost = self.generate_buffer_line(new_x, new_y)
        cost = cost * self.cost_k
        return cost

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    # 计算夹角
    def includedAngle(self, cur, new, des):
        import numpy as np

        # 定义两条直线上的两个点的坐标
        line1_point1 = np.array([cur[0], cur[1]])  # 第一条直线上的第一个点的坐标
        line1_point2 = np.array([des[0], des[1]])  # 第一条直线上的第二个点的坐标

        line2_point1 = np.array([new[0], new[1]])  # 第二条直线上的第一个点的坐标
        line2_point2 = np.array([des[0], des[1]])  # 第二条直线上的第二个点的坐标

        # 计算两条直线的向量表示
        line1_vector = line1_point2 - line1_point1
        line2_vector = line2_point2 - line2_point1

        # 计算向量的夹角（弧度）
        cos_theta = np.dot(line1_vector, line2_vector) / (np.linalg.norm(line1_vector) * np.linalg.norm(line2_vector))
        angle_radians = np.arccos(cos_theta)

        # 将弧度转换为角度
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

    def step(self, action, test):

        if test:
            # 测试再加步长限制
            self.episodes += 1

        ax, ay = action[0], action[1]

        # 新的点
        new_x = int(self.current[0] + ax)
        new_y = int(self.current[1] + ay)

        reward = 0
        # 每一步要离终点越来越近才对
        if self.calculate_distance((new_x, new_y), self.destination) < self.calculate_distance(self.current, self.destination):
            # 离得近你也别转圈圈啊
            if self.includedAngle(self.current, (new_x, new_y), self.destination) > 45:
                reward += -20
            # 别走的大起大落
            if self.current != self.origin:
                if self.includedAngle(self.befor_currrnt, (new_x, new_y), self.current) > 45:
                    reward += -30

            reward += -1/(self.w*self.h)
        else:
            reward += -100



        observation = np.array([new_x, new_y, self.calculate_distance((new_x, new_y), self.destination)])

        terminated, info_ = self.terminated(new_x, new_y)

        if terminated:
            if info_ == 'Get!':
                reward += 300
            # out和max是最接受不了的两种情况
            elif info_ == 'out' or info_ == 'max':
                reward += -100
            elif info_ == 'building':
                reward += -50
            else:
                # 我宁愿你撞墙你都别乱窜啊
                reward += -30

        if not terminated:
            self.befor_currrnt = self.current
            self.current = (new_x, new_y)

        if test:
            self.current = (new_x, new_y)
            self.path.append(self.current)

        return observation, reward, terminated, info_


    def reset(self):

        self.current = self.origin
        # record the parent node foe rendering
        self.current_parent = self.origin
        self.done = False
        self.episodes = 0

        self.path = []
        self.path.append(self.origin)

        distance = self.calculate_distance(self.origin, self.destination)

        return np.array([self.current[0], self.current[1], distance])

    def init_pygame(self):
        self.MapWindowName = 'DRL Overhead path path planning'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.w, self.h))

    def render(self, mode='rgb_array'):
        self.map.fill((253, 251, 239))
        # obstacles
        for y, row in enumerate(self.obstacles):
            for x, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(self.map, (201, 218, 253), (x, y, 1, 1))
                if cell == 2:
                    pygame.draw.rect(self.map, (182, 215, 168), (x, y, 1, 1))

        # original current
        # 黄
        pygame.draw.circle(self.map, (255, 197, 0), (self.origin[1], self.origin[0]), 3, 0)

        # destination
        pygame.draw.circle(self.map, (217, 142, 124), (self.destination[1], self.destination[0]), 3, 0)

        if len(self.path) >= 2:
            for i in range(len(self.path) - 1):
                # 灰蓝色
                pygame.draw.circle(self.map, (152, 178, 194), (self.path[i][1], self.path[i][0]), 2, 0)
                # 红色
                pygame.draw.line(self.map, (255, 0, 0), (self.path[i][1], self.path[i][0]),
                                 (self.path[i + 1][1], self.path[i + 1][0]), 1)
            pygame.draw.circle(self.map, (152, 178, 194), (self.path[i+1][1], self.path[i+1][0]), 2, 0)

        # scale the map
        sacled_map = pygame.transform.scale(self.map, (self.map.get_width() * 4, self.map.get_height() * 4))

        pygame.display.update()
        pygame.event.clear()
        pygame.time.delay(100)
        # pygame.event.wait(0)

        result = np.transpose(np.array(pygame.surfarray.array3d(sacled_map)), axes=(1, 0, 2))

        if self.done:
            pygame.quit()
        return result

