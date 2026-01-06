#train.py
import numpy as np
import matplotlib.pyplot as plt

from env import WaterParkEnv
from agent import QAgent, FixedIntervalPolicy, RandomPolicy, quantize_state

def run_policy_full(env, policy, quantize=False, episodes=5000):
    total_rewards, usage_counts, safeties = [], [], []
    for ep in range(episodes):
        state = env.reset()
        rewards = 0
        done = False
        safe = True
        while not done:
            s = quantize_state(state) if quantize else state
            action = policy.choose_action(s)
            state, reward, done, info = env.step(action)
            rewards += reward
            if state[0] > 0.5 or state[1] > 2.8 or state[2] < 5.8 or state[2] > 8.6:
                safe = False
        #에피소드 종료 후 남은 자원 기반 보너스
        bonus = 0.2 * state[3]   #state[3] = remaining_ci
        rewards += bonus
        total_rewards.append(rewards)
        usage_counts.append(env.usedCI_count)
        safeties.append(safe)
    return total_rewards, usage_counts, safeties

def train_qlearning_full(env, agent, episodes=5000):
    rewards, usages, safeties = [], [], []
    reward_parts_log = {"resource": [], "quality": [], "ci": [], "turbidity": [], "ph": []}
    state_log = {"ci": [], "turbidity": [], "ph": []}
    
    for ep in range(episodes):
        state = env.reset()
        state_disc = quantize_state(state)
        done = False
        total_reward = 0
        safe = True
        
        #에피소드별 리워드 파트 합계
        ep_reward_parts = {"resource": 0, "quality": 0, "ci": 0, "turbidity": 0, "ph": 0}
        
        while not done:
            action = agent.choose_action(state_disc)
            next_state, reward, done, info = env.step(action)
            next_state_disc = quantize_state(next_state)
            agent.learn(state_disc, action, reward, next_state_disc)
            state_disc = next_state_disc
            state = next_state
            total_reward += reward
            
            #리워드 기여도 기록
            for k,v in info["reward_parts"].items():
                reward_parts_log[k].append(v)
                ep_reward_parts[k] += v #수정
            state_log["ci"].append(info["residualCI"])
            state_log["turbidity"].append(info["turbidity"])
            state_log["ph"].append(info["ph"])
            
        #에피소드 종료 후 남은 자원 기반 보너스
        bonus = 0.1 * state[3] #조정 필요
        total_reward += bonus
        
        rewards.append(total_reward)
        usages.append(env.usedCI_count)
        safeties.append(safe)
        
        #에피소드 진행도에 따라 epsilon_min 조정
        if ep > episodes * 0.8:
            agent.epsilon_min = 0.01  #80% 이상 진행 시 1%로 제한
        elif ep > episodes * 0.5:
            agent.epsilon_min = 0.05  #50% 이상 진행 시 5%로 제한

        agent.decay_epsilon()
        
        # if (ep+1) % 100 == 0:
        #     print(f"Episode {ep+1} / Epsilon: {agent.epsilon:.4f}")
        #     print(f"  - Total Reward: {total_reward:.2f}")
        #     print(f"  - Total CI Usage: {env.usedCI_count} kg")
        #     print("-" * 30)
        print(f"Episode {ep+1}")
        print(f"  자원소모 패널티: {ep_reward_parts['resource']:.2f}") #자원
        print(f"  남은 자원 보너스: {bonus:.2f}") #자원
        print(f"  수질 기준 충족 리워드: {ep_reward_parts['quality']:.2f}") #수질
        print(f"  잔류염소 리워드: {ep_reward_parts['ci']:.2f}") #수질
        print(f"  탁도 리워드: {ep_reward_parts['turbidity']:.2f}") #수질
        print(f"  pH 리워드: {ep_reward_parts['ph']:.2f}") #수질
        print(f"  총 리워드: {total_reward:.2f}")
        print("-" * 40)
            
    return rewards, usages, safeties, reward_parts_log, state_log

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')
if __name__ == "__main__":
    env = WaterParkEnv()

    #Q-러닝 학습, epsilon 0.1로 시작, decay로 점차 감소
    q_agent = QAgent(epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.001)
    fixed_policy = FixedIntervalPolicy()
    random_policy = RandomPolicy()

    #Fixed Policy
    fixed_rewards, fixed_usage, fixed_safety = run_policy_full(env, fixed_policy, quantize=False, episodes=3000) 
    
    #Random Policy
    random_rewards, random_usage, random_safety = run_policy_full(env, random_policy, quantize=False, episodes=3000)

    #Q-Learning
    q_rewards, q_usage, q_safety, reward_parts_log, state_log = train_qlearning_full(env, q_agent, episodes=3000)

    plt.figure(figsize=(14, 5))

    #전체 리워드(왼쪽)
    plt.subplot(1, 2, 1)
    plt.plot(moving_average(fixed_rewards), label="Fixed Policy")
    plt.plot(moving_average(q_rewards), label="Q-Learning")
    plt.plot(moving_average(random_rewards), label="Random Policy")
    
    plt.title("Policy Performance Comparison")
    plt.ylabel("Mean Total Reward (Moving Average)")
    plt.legend()

    #자원 소모량(오른쪽)
    plt.subplot(1, 2, 2)
    plt.plot(moving_average(fixed_usage), label="Fixed Policy")
    plt.plot(moving_average(q_usage), label="Q-Learning")
    plt.plot(moving_average(random_usage), label="Random Policy")
    
    plt.ylim(0, 210)
    plt.title("Resource Usage Comparison")
    plt.ylabel("Chlorine Usage (kg, Moving Average)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    #Water Quality
    plt.figure(figsize=(15, 10))

    #잔류염소 그래프(왼쪽 위)
    plt.subplot(2, 2, 1)
    plt.plot(moving_average(state_log['ci'], window=100), label="Residual Chlorine (mg/L)", color='b')
    plt.axhline(y=0.4, color='b', linestyle='--', label='CI Min')
    plt.axhline(y=2.0, color='b', linestyle='--', label='CI Max')
    plt.title("Residual Chlorine Changes During Q-Learning")
    plt.xlabel("Training Step (Moving Average)")
    plt.ylabel("Value (mg/L)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    #탁도 그래프(오른쪽 위)
    plt.subplot(2, 2, 2)
    plt.plot(moving_average(state_log['turbidity'], window=100), label="Turbidity (NTU)", color='orange')
    plt.axhline(y=2.8, color='orange', linestyle='--', label='Turbidity Max')
    plt.title("Turbidity Changes During Q-Learning")
    plt.xlabel("Training Step (Moving Average)")
    plt.ylabel("Value (NTU)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    #pH 그래프(왼쪽 아래)
    plt.subplot(2, 2, 3)
    plt.plot(moving_average(state_log['ph'], window=100), label="pH", color='g')
    plt.axhline(y=5.8, color='g', linestyle='--', label='pH Min')
    plt.axhline(y=8.6, color='g', linestyle='--', label='pH Max')
    plt.title("pH Changes During Q-Learning")
    plt.xlabel("Training Step (Moving Average)")
    plt.ylabel("Value (pH)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()