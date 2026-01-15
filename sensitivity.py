import numpy as np
import matplotlib.pyplot as plt
from env import WaterParkEnv
from agent import QAgent, FixedIntervalPolicy, RandomPolicy
from train import train_qlearning_full, run_policy_full

def run_policy_comparison():
    #비교할 시나리오
    guest_scenarios = [2500, 5000, 7500, 10000, 15000, 20000, 25000]
    
    #결과 저장용 리스트
    usage_fixed = []
    usage_random = []
    usage_qlearning = []

    for guests in guest_scenarios:
        print(f"[Scenario] 일일 방문객: {guests}명")
        
        # 환경 생성
        env = WaterParkEnv(daily_guests=guests)
        
        # Fixed Policy
        fixed_policy = FixedIntervalPolicy()
        _, f_usages, _, _ = run_policy_full(env, fixed_policy, episodes=1000)
        avg_f = np.mean(f_usages) # 고정 정책은 학습이 없으므로 전체 평균
        usage_fixed.append(avg_f)
        
        # Random Policy
        random_policy = RandomPolicy()
        _, r_usages, _, _ = run_policy_full(env, random_policy, episodes=1000)
        avg_r = np.mean(r_usages)
        usage_random.append(avg_r)

        # Q-Learning
        agent = QAgent(epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
        _, q_usages, _, _, _ = train_qlearning_full(env, agent, episodes=3000)
        # 학습 완료된 후반부 500판의 평균 사용량
        avg_q = np.mean(q_usages[-500:]) 
        usage_qlearning.append(avg_q)
        
        print(f"   [Fixed]: {avg_f:.1f}kg | [Random]: {avg_r:.1f}kg | [Q-Learning]: {avg_q:.1f}kg")
        print("-" * 70)

    # 결과
    plt.figure(figsize=(10, 6))

    # 그래프 스타일 설정
    plt.title("Resource Usage Comparison by Policy", fontsize=16, pad=20)
    plt.xlabel('Daily Guests (People)', fontsize=14)
    plt.ylabel('Average Chlorine Usage (kg)', fontsize=14)
    plt.xticks(guest_scenarios)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Fixed Policy(파란색)
    plt.plot(guest_scenarios, usage_fixed, label='Fixed', color='tab:blue', marker='o', linewidth=2, markersize=8)

    # Random Policy(주황색)
    plt.plot(guest_scenarios, usage_random, label='Random', color='tab:orange', marker='o', linewidth=2, markersize=8)

    # Q-Learning(초록색)
    plt.plot(guest_scenarios, usage_qlearning, label='Q-Learning', color='tab:green', marker='o', linewidth=2, markersize=8)

    # Y축 0부터 시작
    plt.ylim(bottom=0)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 출력
    plt.show()
    plt.close()

if __name__ == "__main__":
    run_policy_comparison()