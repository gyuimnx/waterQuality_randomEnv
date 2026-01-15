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
        plt.plot(moving_average(fixed_rewards), label="Fixed")
        plt.plot(moving_average(random_rewards), label="Random")
        plt.plot(moving_average(greedy_rewards), label="Greedy(Gamma=0)")
        plt.plot(moving_average(q_rewards), label="Q-Learning")

        plt.title("Policy Performance Comparison")
        plt.ylabel("Mean Total Reward(Moving Average)")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # plt.show()
        filename = f"{i}-1_Reward.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

        # 자원 소모량 비교(y축 0부터)
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(fixed_usage), label="Fixed")
        plt.plot(moving_average(random_usage), label="Random")
        plt.plot(moving_average(greedy_usage), label="Greedy(Gamma=0)")
        plt.plot(moving_average(q_usage), label="Q-Learning")

        plt.ylim(bottom=0)
        plt.title("Resource Usage Comparison")
        plt.ylabel("Chlorine Usage(kg, Moving Average)")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # plt.show()
        plt.savefig(os.path.join(save_dir, f"{i}-2_Resource.png"))
        plt.close()
        
        # 자원 소모량 비교(자동 스케일링)
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(fixed_usage), label="Fixed")
        plt.plot(moving_average(random_usage), label="Random")
        plt.plot(moving_average(greedy_usage), label="Greedy")
        plt.plot(moving_average(q_usage), label="Q-Learning")
                
        plt.title(f"[{i}th Sim] Resource Usage Comparison (Auto Scaled)")
        plt.ylabel("Chlorine Usage (kg, Moving Average)")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 파일명 끝에 _AutoScaled를 붙여서 따로 저장
        plt.savefig(os.path.join(save_dir, f"{i}-2-2_Resource_AutoScaled.png"))
        plt.close()

        # 수질 상태 변화
        plt.figure(figsize=(15, 10))

        # 잔류염소
        plt.subplot(2, 2, 1)
        plt.plot(moving_average(state_log['ci'], window=100), label="Residual Chlorine(mg/L)", color='b')
        plt.axhline(y=0.4, color='b', linestyle='--', label='CI Min')
        plt.axhline(y=2.0, color='b', linestyle='--', label='CI Max')
        plt.title("Residual Chlorine Changes During Q-Learning")
        plt.ylabel("Value(mg/L)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 탁도
        plt.subplot(2, 2, 2)
        plt.plot(moving_average(
            state_log['turbidity'], window=100), label="Turbidity(NTU)", color='orange')
        plt.axhline(y=2.8, color='orange', linestyle='--', label='Turbidity Max')
        plt.title("Turbidity Changes During Q-Learning")
        plt.ylabel("Value(NTU)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # pH
        plt.subplot(2, 2, 3)
        plt.plot(moving_average(state_log['ph'], window=100), label="pH", color='g')
        plt.axhline(y=5.8, color='g', linestyle='--', label='pH Min')
        plt.axhline(y=8.6, color='g', linestyle='--', label='pH Max')
        plt.title("pH Changes During Q-Learning")
        plt.xlabel("Training Step (Moving Average)")
        plt.ylabel("Value(pH)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # plt.tight_layout()
        # plt.show()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"{i}-3_Quality_Q.png"))
        plt.close()

        # -------------------------------------------------------
        # [Graph 3-2] Fixed Policy 수질 상태 변화 (단독)
        # -------------------------------------------------------
        plt.figure(figsize=(15, 10))
        plt.suptitle("Water Quality Changes - Fixed Policy (Interval Control)", fontsize=16)

        # 잔류염소
        plt.subplot(2, 2, 1)
        # Fixed 로그 데이터 사용 (fixed_state_log)
        plt.plot(moving_average(fixed_state_log['ci'], window=100), label="Residual CI", color='r')
        plt.axhline(y=0.4, color='black', linestyle=':', label='Min/Max')
        plt.axhline(y=2.0, color='black', linestyle=':')
        plt.title("Residual Chlorine (mg/L)")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 탁도
        plt.subplot(2, 2, 2)
        plt.plot(moving_average(fixed_state_log['turbidity'], window=100), label="Turbidity", color='darkorange')
        plt.axhline(y=2.8, color='black', linestyle=':', label='Max')
        plt.title("Turbidity (NTU)")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # pH
        plt.subplot(2, 2, 3)
        plt.plot(moving_average(fixed_state_log['ph'], window=100), label="pH", color='darkgreen')
        plt.axhline(y=5.8, color='black', linestyle=':', label='Min/Max')
        plt.axhline(y=8.6, color='black', linestyle=':')
        plt.title("pH")
        plt.ylabel("Value")
        plt.xlabel("Training Steps (Moving Average)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"{i}-4_Quality_Fixed.png"))
        plt.close()


        # -------------------------------------------------------
        # [Graph 4] 행동 분포 비교 (4개 정책 전체 비교 - 비율 %)
        # -------------------------------------------------------
        


        q_actions = get_action_distribution(env, q_agent)
        greedy_actions = get_action_distribution(env, greedy_policy)
        fixed_actions = get_action_distribution(env, fixed_policy)
        random_actions = get_action_distribution(env, random_policy)

        plt.figure(figsize=(12, 6))
        plt.hist([q_actions, greedy_actions, fixed_actions, random_actions], bins=np.arange(6)-0.5, label=['Q-Learning', 'Greedy', 'Fixed', 'Random'], color=['green', 'red', 'blue', 'gray'], alpha=0.7, rwidth=0.85, density=True)
        plt.xticks(range(5), ['0kg', '5kg', '15kg', '25kg', '35kg'])
        plt.title(f"[{i}th Sim] Action Distribution Comparison (%)")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(save_dir, f"{i}-5_Action.png"))
        plt.close()


    # #Epsilon Min 값에 따른 성능 비교 (0.01 vs 0.1 vs 0.3)

    # if __name__ == "__main__":
    #     #환경 생성(모든 에이전트가 동일한 환경 설정 공유)
    #     env = WaterParkEnv()

    #     #Min Epsilon = 0.01
    #     agent_001 = QAgent(gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    #     rewards_001, usage_001, _, _, _ = train_qlearning_full(env, agent_001, episodes=3000)

    #     #Min Epsilon = 0.1
    #     agent_01 = QAgent(gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1)
    #     rewards_01, usage_01, _, _, _ = train_qlearning_full(env, agent_01, episodes=3000)

    #     #Min Epsilon = 0.3
    #     agent_03 = QAgent(gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.3)
    #     rewards_03, usage_03, _, _, _ = train_qlearning_full(env, agent_03, episodes=3000)


    #     # 결과 시각화
    #     plt.figure(figsize=(15, 6))

    #     #전체 리워드 비교
    #     plt.subplot(1, 2, 1)
    #     plt.plot(moving_average(rewards_001), label="Min Epsilon = 0.01", color='red')
    #     plt.plot(moving_average(rewards_01), label="Min Epsilon = 0.1", color='orange')
    #     plt.plot(moving_average(rewards_03), label="Min Epsilon = 0.3", color='green', alpha=0.6)

    #     plt.title("Reward Comparison by Epsilon Min")
    #     plt.xlabel("Episodes")
    #     plt.ylabel("Total Reward")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)

    #     #자원 소모량 비교
    #     plt.subplot(1, 2, 2)
    #     plt.plot(moving_average(usage_001), label="Min Epsilon = 0.01", color='red')
    #     plt.plot(moving_average(usage_01), label="Min Epsilon = 0.1", color='orange')
    #     plt.plot(moving_average(usage_03), label="Min Epsilon = 0.3", color='green', alpha=0.6)

    #     plt.title("Resource Usage Comparison")
    #     plt.xlabel("Episodes")
    #     plt.ylabel("Chlorine Usage (kg)")
    #     plt.ylim(0, 210)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)

    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
