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
    if residualCI < 0.4: #잔류염소 낮음
        residualCI_state = 0
    elif residualCI > 2.0: #잔류염소 높음
        residualCI_state = 2
    else: #잔류염소 정상
        
        residualCI_state = 1
    
    #남은 염소 상태 양자화
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

class FixedIntervalPolicy:
    def __init__(self):
        #3스탭(30분)마다 수질을 측정하여 염소 투입량을 결정
        self.interval = 3

    def choose_action(self, state):
        residualCI, turbidity, ph, remaining_ci, current_step = state
        #3스탭 간격이 아니면 0kg
        if int(current_step) % self.interval != 0:
            return 0  
        if (residualCI < 1.2 or turbidity > 1.5): 
            return 3
        elif (residualCI < 1.5):
            return 2
        else:
            return 1

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
        self.actions = [0, 1, 2, 3, 4]
    
    def choose_action(self, state):
        residualCI, turbidity, ph, remaining_ci, current_step = state
        
        #자원이 없으면 0kg
        if remaining_ci < 5:
            return 0
        
        #즉각적인 수질 문제가 심각하면 최대 투입
        if residualCI < 0.5 or turbidity > 2.5 or ph < 6.0 or ph > 8.4:
            return 4  # 35kg
        
        #수질이 약간 나쁘면 중간 투입
        elif residualCI < 0.8 or turbidity > 1.8:
            return 3  # 25kg
        
        #수질이 괜찮으면 유지만
        elif residualCI < 1.5:
            return 2  # 15kg
        
        elif residualCI > 1.8:
            return 0
        
        #수질이 매우 좋으면 최소 투입
        else:
            return 1  # 5kg