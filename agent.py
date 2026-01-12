#agent.py
import numpy as np
import random

def quantize_state(state):
    residualCI, turbidity, ph, remaining_ci, current_step = state
    #pH 상태 양자화
    if ph < 5.8: #pH 낮음
        ph_state = 0
    elif ph > 8.6: #pH 높음
        ph_state = 2
    else: #pH 정상
        ph_state = 1
        
    #탁도 2.8 이하면 0
    #탁도 상태 양자화
    turbidity_state = 0 if turbidity <= 2.8 else 1
    
    #잔류염소 상태 양자화
    #잔류염소
    if residualCI < 0.4: #잔류염소 낮음
        residualCI_state = 0
    elif residualCI > 2.0: #잔류염소 높음
        residualCI_state = 2
    else: #잔류염소 정상
        
        residualCI_state = 1
    
    #남은 염소 상태 양자화
    # if remaining_ci < 20: #20kg 미만일 때 부족으로 인식
    #     remaining_ci_state = 0
    # elif remaining_ci < 50: #20kg ~ 50kg
    #     remaining_ci_state = 1
    # elif remaining_ci < 100: #50kg ~ 100kg
    #     remaining_ci_state = 2
    # else: #100kg 이상
    #     remaining_ci_state = 3
    
    # 남은 염소 상태 양자화(더 촘촘하게 20kg 단위로 구분)
    # if remaining_ci < 20:
    #     remaining_ci_state = 0
    # elif remaining_ci < 40:
    #     remaining_ci_state = 1
    # elif remaining_ci < 60:
    #     remaining_ci_state = 2
    # elif remaining_ci < 80:
    #     remaining_ci_state = 3
    # elif remaining_ci < 100:
    #     remaining_ci_state = 4
    # elif remaining_ci < 120:
    #     remaining_ci_state = 5
    # elif remaining_ci < 140:
    #     remaining_ci_state = 6
    # elif remaining_ci < 160:
    #     remaining_ci_state = 7
    # elif remaining_ci < 180:
    #     remaining_ci_state = 8
    # else:
    #     remaining_ci_state = 9
        
    remaining_ci_state = min(9, int(remaining_ci // 20))
        
    #시간 양자화(아침, 오후, 저녁)
    hour = 9 + (int(current_step) * 10) // 60
    if 9 <= hour < 12:
        time_state = 0  #아침
    elif 12 <= hour < 17:
        time_state = 1  #오후
    else:
        time_state = 2  #저녁
        
    return (residualCI_state, turbidity_state, ph_state, remaining_ci_state, time_state)

class QAgent:
    def __init__(self, state_shape=(3,2,3,10,3), n_actions=5, alpha=0.1, gamma=0.95, epsilon=0.1, epsilon_decay=0.0, epsilon_min=0.05): #n_action : 0kg, 5kg, 15kg, 25kg, 35kg
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_table = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.Q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q_table[state + (action,)]
        self.Q_table[state + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

# class FixedIntervalPolicy:
#     def __init__(self):
#         self.max_steps = 60  #하루 스텝 수
#         self.n_pulses = 10   #하루에 10번 투입
#         self.step_interval = self.max_steps // self.n_pulses  #투입 간격

#     def choose_action(self, state):
#         _, _, _, remaining_ci, current_step = state
#         if remaining_ci > 0 and int(current_step) % self.step_interval == 0:
#             return 2  #20kg 행동
#         return 0      #0kg 행동

# agent.py
class FixedIntervalPolicy:
    def __init__(self):
        #3스탭(30분)마다 수질을 측정하여 염소 투입량을 결정
        self.interval = 3

    def choose_action(self, state):
        residualCI, turbidity, ph, remaining_ci, current_step = state
        #3스탭 간격이 아니면 0kg 투입
        if int(current_step) % self.interval != 0:
            return 0  
        #0: 0kg, 1: 5kg, 2: 15kg, 3: 25kg, 4: 35kg
        if (residualCI < 1.2 or turbidity > 1.5): 
            return 4  #35kg(수질이 나쁠 때)
        elif (residualCI < 1.5):
            return 3  #25kg(수질이 보통일 때)
        else:
            return 1  #5kg(수질이 좋을 때 유지)

class RandomPolicy:
    def __init__(self):
        self.interval = 3 #30분(3스텝) 주기 설정

    def choose_action(self, state):
        _, _, _, _, current_step = state #현재 스텝 정보
        
        #주기가 아니면 아무것도 안 함(0kg)
        if int(current_step) % self.interval != 0:
            return 0
        
        #주기일 때만 5가지 액션 중 랜덤 선택
        return random.randint(0, 4)
    
class GreedyPolicy:
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4]  # 0kg, 5kg, 15kg, 25kg, 35kg
    
    def choose_action(self, state):
        residualCI, turbidity, ph, remaining_ci, current_step = state
        
        # 자원이 없으면 0kg
        if remaining_ci < 5:
            return 0
        
        # 즉각적인 수질 문제가 심각하면 최대 투입
        if residualCI < 0.5 or turbidity > 2.5 or ph < 6.0 or ph > 8.4:
            return 4  # 35kg
        
        # 수질이 약간 나쁘면 중간 투입
        elif residualCI < 0.8 or turbidity > 1.8:
            return 3  # 25kg
        
        # 수질이 괜찮으면 유지만
        elif residualCI < 1.5:
            return 2  # 15kg
        
        elif residualCI > 1.8:
            return 0
        
        # 수질이 매우 좋으면 최소 투입
        else:
            return 1  # 5kg