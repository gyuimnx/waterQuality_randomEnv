import numpy as np
import matplotlib.pyplot as plt
import os

from env import WaterParkEnv
from agent import QAgent, FixedIntervalPolicy, RandomPolicy, quantize_state, GreedyPolicy


def run_policy_full(env, policy, quantize=False, episodes=5000):
    total_rewards, usage_counts, safeties = [], [], []
    
    state_log = {"ci": [], "turbidity": [], "ph": []}
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
            
            state_log["ci"].append(info["residualCI"])
            state_log["turbidity"].append(info["turbidity"])
            state_log["ph"].append(info["ph"])
            
            if state[0] > 0.4 or state[1] > 2.8 or state[2] < 5.8 or state[2] > 8.6:
                safe = False
        # 에피소드 종료 후 남은 자원 기반 보너스
        bonus = 0.05 * state[3]  # state[3] = remaining_ci
        rewards += bonus
        total_rewards.append(rewards)
        usage_counts.append(env.usedCI_count)
        safeties.append(safe)
    return total_rewards, usage_counts, safeties, state_log


def train_qlearning_full(env, agent, episodes=5000):
    rewards, usages, safeties = [], [], []
    reward_parts_log = {"resource": [], "quality": [],
                        "ci": [], "turbidity": [], "ph": []}
    state_log = {"ci": [], "turbidity": [], "ph": []}

    for ep in range(episodes):
        state = env.reset()
        state_disc = quantize_state(state)
        done = False
        total_reward = 0
        safe = True

        # 에피소드별 리워드 파트 합계
        ep_reward_parts = {"resource": 0, "quality": 0, "ci": 0, "turbidity": 0, "ph": 0}

        while not done:
            action = agent.choose_action(state_disc)
            next_state, reward, done, info = env.step(action)
            next_state_disc = quantize_state(next_state)
            agent.learn(state_disc, action, reward, next_state_disc)
            state_disc = next_state_disc
            state = next_state
            total_reward += reward

            # 리워드 기여도 기록
            for k, v in info["reward_parts"].items():
                reward_parts_log[k].append(v)
                ep_reward_parts[k] += v  # 수정
            state_log["ci"].append(info["residualCI"])
            state_log["turbidity"].append(info["turbidity"])
            state_log["ph"].append(info["ph"])

        # 에피소드 종료 후 남은 자원 기반 보너스
        bonus = 0.05 * state[3]  # 조정 필요
        total_reward += bonus

        rewards.append(total_reward)
        usages.append(env.usedCI_count)
        safeties.append(safe)

        # 에피소드 진행도에 따라 epsilon_min 조정
        if ep > episodes * 0.8:
            agent.epsilon_min = 0.01  # 80% 이상 진행 시 1%로 제한
        elif ep > episodes * 0.5:
            agent.epsilon_min = 0.05  # 50% 이상 진행 시 5%로 제한

        agent.decay_epsilon()

        print(f"Episode {ep+1}")
        print(f"  자원소모 패널티: {ep_reward_parts['resource']:.2f}")  # 자원
        print(f"  남은 자원 보너스: {bonus:.2f}")  # 자원
        print(f"  수질 기준 충족 리워드: {ep_reward_parts['quality']:.2f}")  # 수질
        print(f"  잔류염소 리워드: {ep_reward_parts['ci']:.2f}")  # 수질
        print(f"  탁도 리워드: {ep_reward_parts['turbidity']:.2f}")  # 수질
        print(f"  pH 리워드: {ep_reward_parts['ph']:.2f}")  # 수질
        print(f"  총 리워드: {total_reward:.2f}")
        print("-" * 40)

    return rewards, usages, safeties, reward_parts_log, state_log


def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def get_action_distribution(env, policy, episodes=500):
    actions = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # Q-Agent는 양자화된 상태 필요, 나머지는 연속 상태 사용
            if hasattr(policy, 'Q_table'): 
                s = quantize_state(state)
                action = policy.choose_action(s)
            else: 
                action = policy.choose_action(state) 
            actions.append(action)
            state, _, done, _ = env.step(action)
    return actions


if __name__ == "__main__":
    N_SIMULATIONS = 5  #n번 반복
    
    # 결과를 저장할 폴더 이름(없으면 자동으로 만듦)
    save_dir = "simulation_results"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(1, N_SIMULATIONS + 1):
        print(f"\n" + "="*40)
        print(f" [Simulation {i} / {N_SIMULATIONS}] 시작...")
        print(f" (파일명: {i}-1.png, {i}-2.png ... 로 저장됨)")
        print("="*40)
        
        env = WaterParkEnv()
        
        # Q-러닝 학습, epsilon 1.0로 시작, decay로 점차 감소
        q_agent = QAgent(epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95)
        fixed_policy = FixedIntervalPolicy()
        random_policy = RandomPolicy()
        greedy_policy = GreedyPolicy()

        # Fixed Policy
        fixed_rewards, fixed_usage, fixed_safety, fixed_state_log = run_policy_full(env, fixed_policy, quantize=False, episodes=3000)

        # Random Policy
        random_rewards, random_usage, random_safety, _ = run_policy_full(
            env, random_policy, quantize=False, episodes=3000)

        # Greedy Policy
        greedy_rewards, greedy_usage, greedy_safety, _ = run_policy_full(
            env, greedy_policy, quantize=False, episodes=3000)

        # Q-Learning
        q_rewards, q_usage, q_safety, reward_parts_log, state_log = train_qlearning_full(
            env, q_agent, episodes=3000)

        # 전체 리워드 비교
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(fixed_rewards), label="Fixed Policy")
        plt.plot(moving_average(random_rewards), label="Random Policy")
        plt.plot(moving_average(greedy_rewards), label="Greedy Policy")
        plt.plot(moving_average(q_rewards), label="Q-Learning")

        plt.title("Learning Performance: Total Reward")
        plt.ylabel("Total Reward (Moving Average)")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # plt.show()
        filename = f"{i}-1_Reward.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

        # # 자원 소모량 비교(y축 0부터)
        # plt.figure(figsize=(10, 6))
        # plt.plot(moving_average(fixed_usage), label="Fixed Policy")
        # plt.plot(moving_average(random_usage), label="Random Policy")
        # plt.plot(moving_average(greedy_usage), label="Greedy Policy")
        # plt.plot(moving_average(q_usage), label="Q-Learning")

        # plt.ylim(bottom=0)
        # plt.title("Resource Usage Comparison")
        # plt.ylabel("Average. Chlorine Consumption (kg)")
        # plt.xlabel("Episodes")
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.5)

        # # plt.show()
        # plt.savefig(os.path.join(save_dir, f"{i}-2_Resource.png"))
        # plt.close()
        
        # 자원 소모량 비교(자동 스케일링)
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(fixed_usage), label="Fixed Policy")
        plt.plot(moving_average(random_usage), label="Random Policy")
        plt.plot(moving_average(greedy_usage), label="Greedy Policy")
        plt.plot(moving_average(q_usage), label="Q-Learning")
                
        plt.title("Resource Usage Comparison")
        plt.ylabel("Average. Chlorine Consumption (kg)")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(save_dir, f"{i}-2_Resource_AutoScaled.png"))
        plt.close()

        # 수질 안정성 및 안전 기준 준수 비교 그래프
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Comparison of Water Quality Stability and Safety Compliance: Q-Learning vs. Fixed Policy", fontsize=16)

        # 데이터 절반 자르기 (Convergence가 빠르므로 앞부분 집중)
        # moving_average 결과의 길이를 반으로 줄임
        q_ci = moving_average(state_log['ci'], window=100)
        q_turb = moving_average(state_log['turbidity'], window=100)
        q_ph = moving_average(state_log['ph'], window=100)
        
        f_ci = moving_average(fixed_state_log['ci'], window=100)
        f_turb = moving_average(fixed_state_log['turbidity'], window=100)
        f_ph = moving_average(fixed_state_log['ph'], window=100)
        
        cut_idx = len(q_ci) // 2  # 절반 인덱스
        
        # 공통 스타일 정의
        x_label = "Training Steps"
        
        # ---------------------------
        # Row 1: Q-Learning (Blue/Orange/Green)
        # ---------------------------
        
        # 1-1. 잔류염소 (Q-Learning)
        ax = axes[0, 0]
        ax.plot(q_ci[:cut_idx], label="Q-Learning", color='blue')
        ax.axhline(y=0.4, color='black', linestyle=':', label='Min (0.4)')
        ax.axhline(y=2.0, color='black', linestyle=':', label='Max (2.0)')
        ax.set_title("Residual Chlorine (Q-Learning)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_xticklabels([]) # 위쪽 그래프 X축 라벨 숨김
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 1-2. 탁도 (Q-Learning)
        ax = axes[0, 1]
        ax.plot(q_turb[:cut_idx], label="Q-Learning", color='orange')
        ax.axhline(y=2.8, color='black', linestyle=':', label='Max (2.8)')
        ax.set_title("Turbidity (Q-Learning)")
        ax.set_ylabel("Turbidity (NTU)")
        ax.set_xticklabels([])
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 1-3. pH (Q-Learning)
        ax = axes[0, 2]
        ax.plot(q_ph[:cut_idx], label="Q-Learning", color='green')
        ax.axhline(y=5.8, color='black', linestyle=':', label='Min (5.8)')
        ax.axhline(y=8.6, color='black', linestyle=':', label='Max (8.6)')
        ax.set_title("pH (Q-Learning)")
        ax.set_ylabel("pH Value")
        ax.set_xticklabels([])
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # ---------------------------
        # Row 2: Fixed Policy
        # ---------------------------

        # 2-1. 잔류염소 (Fixed)
        ax = axes[1, 0]
        ax.plot(f_ci[:cut_idx], label="Fixed Policy", color='blue')
        ax.axhline(y=0.4, color='black', linestyle=':', label='Min (0.4)')
        ax.axhline(y=2.0, color='black', linestyle=':', label='Max (2.0)')
        ax.set_title("Residual Chlorine (Fixed Policy)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_xlabel(x_label)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 2-2. 탁도 (Fixed)
        ax = axes[1, 1]
        ax.plot(f_turb[:cut_idx], label="Fixed Policy", color='orange')
        ax.axhline(y=2.8, color='black', linestyle=':', label='Max (2.8)')
        ax.set_title("Turbidity (Fixed Policy)")
        ax.set_ylabel("Turbidity (NTU)")
        ax.set_xlabel(x_label)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 2-3. pH (Fixed)
        ax = axes[1, 2]
        ax.plot(f_ph[:cut_idx], label="Fixed Policy", color='green')
        ax.axhline(y=5.8, color='black', linestyle=':', label='Min (5.8)')
        ax.axhline(y=8.6, color='black', linestyle=':', label='Max (8.6)')
        ax.set_title("pH (Fixed Policy)")
        ax.set_ylabel("pH Value")
        ax.set_xlabel(x_label)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 저장
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"{i}-3_Quality_Integrated.png"))
        plt.close()