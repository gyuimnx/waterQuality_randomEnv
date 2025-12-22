#env.py
import numpy as np
import random

#시간대별 오염 계수
def get_pollution_factor(hour):
    if 14 <= hour < 17:
        return 1.0
    elif 9 <= hour < 12:
        return 0.7
    elif 12 <= hour < 14 or 17 <= hour < 19:
        return 0.3
    else:
        return 0.0

class WaterParkEnv:
    def __init__(self, max_steps=60, max_ci=200):
        self.max_steps = max_steps
        self.max_ci = max_ci #하루 최대 염소 사용량(kg)
        #0kg, 5kg, 15kg, 25kg
        self.action_ci = [0, 5, 15, 25]
        self.reset()

    def get_current_guests(self, step):
        hour = 9 + (step * 10) // 60
        if 9 <= hour < 12: #아침(18스탭동안 1스탭당 30% 유입)
            return int(10000 * 0.3 / 18)
        elif 12 <= hour < 17: #오후(30스탭동안 1스탭당 50% 유입)
            return int(10000 * 0.5 / 30)
        elif 17 <= hour < 19: #저녁(12스탭동안 1스탭당 20% 유입)
            return int(10000 * 0.2 / 12)
        else:
            return 0 #영업시간 외 유입x

    def reset(self):
        self.state = np.array([
            random.uniform(0.4, 2.0), #잔류염소
            random.uniform(0, 2.8), #탁도
            random.uniform(5.8, 8.6), #pH
            self.max_ci, #남은 염소
            0 #스탭
        ])
        self.steps = 0
        self.usedCI_count = 0 #누적 염소 사용량
        self.done = False
        return self.state.copy()

    def step(self, action):
        residualCI, turbidity, ph, remaining_ci, current_step = self.state #잔류염소, 탁도, pH, 남은염소, 현재스탭
        reward = 0.0 #기본보상
        done = False
        #각각의 기본 보상
        reward_resource = 0.0
        reward_quality = 0.0
        reward_ci = 0.0
        reward_turb = 0.0
        reward_ph = 0.0
        
        #염소 투입
        ci_to_add = self.action_ci[action] #선택한 행동을 실제 염소 사용량으로 변환
        if remaining_ci >= ci_to_add:
            remaining_ci -= ci_to_add
            self.usedCI_count += ci_to_add
            #염소 투입 시 잔류염소 증가(10kg당 0.2mg 가정)
            residualCI += (ci_to_add / 10.0) * 0.2 
            #자원 소모 패널티
            # reward -= 0.1 * ci_to_add
            reward_resource -= 0.4 * ci_to_add
            
            #탁도 감소(염소가 탁도를 어느정도 낮춘다고 가정, 10kg당 0.2 감소)
            turbidity -= ci_to_add * 0.2
            #ph는 7.2를 기준으로 조정(ph 조절 약품으로 조절한다고 가정)
            if ph < 7.2:
                ph += ci_to_add * 0.1 #7.2보다 낮으면 올림
            elif ph > 7.2:
                ph -= ci_to_add * 0.1 #7.2보다 높으면 내림
            #ph == 7.2 인 경우 변화 없음
        else:
            #남은 염소 부족 패널티(이거 바꿀 예정)
            # reward -= 0.2
            reward_resource -= 0.2
        
        #추가한거-----------------------------
        #정상 수질 보상
        if 0.4 <= residualCI <= 2.0 and turbidity <= 2.8 and 5.8 <= ph <= 8.6:
            reward_quality += 0.3
        else:
            reward_quality -= 0.1
        
        #오염 증가(인원 유입/스탭마다)
        hour = 9 + (int(current_step) * 10) // 60
        pollution_factor = get_pollution_factor(hour)

        ph += random.uniform(-0.1, 0.1) * pollution_factor
        turbidity += random.uniform(0.5, 1.0) * pollution_factor
        residualCI -= random.uniform(0.05, 0.1) * pollution_factor

        #자연 복원(환경 회복)
        turbidity -= random.uniform(0.1, 0.3)  #입자 가라앉음
        if ph > 7.2:  #염기성에서 산성
            ph -= random.uniform(0.01, 0.05)
        elif ph < 7.2:  #산성에서 염기성
            ph += random.uniform(0.01, 0.05)

        #음수 방지
        residualCI = max(0.0, residualCI)
        turbidity = max(0.0, turbidity)
        ph = max(0.0, ph)

        #-------------보상 계산-------------
        # # 정상 상태 보상 (잔류염소, 탁도, pH 모두 정상 범위일 때)------------------------------------------------
        # if 0.4 <= residualCI <= 2.0 and turbidity <= 2.8 and 5.8 <= ph <= 8.6:
        #     reward += 0.3
        # else:
        #     reward -= 0.2
            
        #잔류염소 패널티
        if residualCI > 2.0:
            reward_ci -= (residualCI - 2.0) *5.0
        elif residualCI < 0.4:
            reward_ci -= (0.4 - residualCI) *5.0
        else:
            reward_ci += 0.2

        #탁도 패널티
        if turbidity > 2.8:
            reward_turb -= (turbidity - 2.8) *4.0
        else:
            reward_turb += 0.2

        #pH 패널티
        if ph > 8.6:
            reward_ph -= (ph - 8.6) *5.0
        elif ph < 5.8:
            reward_ph -= (5.8 - ph) *5.0
        else:
            reward_ph += 0.2

        #자원 초과 사용 패널티
        if self.usedCI_count > self.max_ci:
            excess = self.usedCI_count - self.max_ci
            reward -= excess * 0.05
            
        #총합 리워드
        reward = reward_resource + reward_ci + reward_turb + reward_ph
        
        #-------------상태 업데이트-------------
        current_step += 1
        self.state = np.array([
            residualCI,
            turbidity,
            ph,
            remaining_ci,
            current_step
        ])
        self.steps += 1

        #종료 조건
        if current_step >= self.max_steps or self.steps >= self.max_steps: #하루가 끝났으면 에피소드 종료
            done = True
        self.done = done

        # print(f"[STEP {current_step}] 현재 잔류염소={residualCI:.2f}, 현재 탁도={turbidity:.2f}, 현재 pH={ph:.2f}, "f"투입 염소={ci_to_add}kg, 계산된 Reward={reward:.2f}")

        #디버깅
        #print(f"[STEP {current_step}] Action={action}, Reward={reward:.3f}, CI={residualCI:.2f}, Turb={turbidity:.2f}, pH={ph:.2f}")

        return self.state.copy(), reward, done, {
            'residualCI': residualCI,
            'turbidity': turbidity,
            'ph': ph,
            'remaining_ci': remaining_ci,
            'step': current_step,
            'used_ci': self.usedCI_count,
            'reward_parts': {
                'resource': reward_resource,
                'quality': reward_quality,
                'ci': reward_ci,
                'turbidity': reward_turb,
                'ph': reward_ph
            }
        }