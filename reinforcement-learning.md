# Reinforcement Learning Cheat Sheet

> Quick reference for writing pseudocode. Symbols, shapes, and update rules are stated explicitly so you can translate directly into code.

---

## Table of Contents

0. [Algorithm Quick Reference](#algorithm-quick-reference)
1. [Life Cycle of RL Agents](#1-life-cycle-of-rl-agents)
   - [Metrics Overview](#metrics-overview)
2. [RL Algorithms](#2-rl-algorithms)
   - [Multi-Armed Bandits](#21-multi-armed-bandits)
   - [Dynamic Programming](#22-dynamic-programming)
   - [Monte Carlo Methods](#23-monte-carlo-methods)
   - [SARSA](#24-sarsa)
   - [Q-Learning](#25-q-learning)
   - [Deep Q-Network (DQN)](#26-deep-q-network-dqn)
   - [REINFORCE](#27-reinforce)
   - [Actor-Critic](#28-actor-critic)
   - [PPO](#29-ppo)
   - [DDPG](#210-ddpg)
3. [Dictionary](#3-dictionary) — [A–Z index](#action)

---

## Algorithm Quick Reference

> **Quick Reference** — Pick an algorithm by scenario. Start with **Best choice**, fall back to **Alternatives** if constraints apply.

### By Problem Type

| Problem | Best choice | Alternatives |
|---------|-------------|--------------|
| **No state** — pick best arm | [Multi-Armed Bandits](#21-multi-armed-bandits) | A/B testing baselines |
| **Known model** — full MDP | [Dynamic Programming](#22-dynamic-programming) | — |
| **Unknown model, tabular, small state space** | [Q-Learning](#25-q-learning) | [SARSA](#24-sarsa), [Monte Carlo](#23-monte-carlo-methods) |
| **Unknown model, large state space** | [DQN](#26-deep-q-network-dqn) | [PPO](#29-ppo) |
| **Discrete actions, deep RL** | [DQN](#26-deep-q-network-dqn) → [PPO](#29-ppo) | [Actor-Critic](#28-actor-critic) |
| **Continuous actions** | [DDPG](#210-ddpg) | [PPO](#29-ppo) (Gaussian policy) |
| **On-policy, stable training** | [PPO](#29-ppo) | [SARSA](#24-sarsa), [Actor-Critic](#28-actor-critic) |
| **Off-policy, sample efficient** | [Q-Learning](#25-q-learning) → [DQN](#26-deep-q-network-dqn) | [DDPG](#210-ddpg) |
| **Stochastic policy, direct optimization** | [REINFORCE](#27-reinforce) → [PPO](#29-ppo) | [Actor-Critic](#28-actor-critic) |

### By Scenario

| Scenario | Best choice | Why |
|----------|-------------|-----|
| **Simple exploration benchmark** | [Multi-Armed Bandits](#21-multi-armed-bandits) | No state transitions; regret analysis |
| **Full MDP known** (simulator model) | [Dynamic Programming](#22-dynamic-programming) | Exact solution via Bellman |
| **Episodic tasks, no bootstrapping** | [Monte Carlo](#23-monte-carlo-methods) | Unbiased return estimates |
| **Online learning, safe on-policy** | [SARSA](#24-sarsa) | Uses actual next action |
| **Off-policy, max reward seeking** | [Q-Learning](#25-q-learning) | Learns optimal Q regardless of behavior |
| **High-dim states** (images, vectors) | [DQN](#26-deep-q-network-dqn) | Function approximation + replay |
| **Stable deep RL default** | [PPO](#29-ppo) | Clipped objective; widely robust |
| **Continuous control** (robotics) | [DDPG](#210-ddpg) | Deterministic policy gradient |
| **Need fast prototyping** | [Q-Learning](#25-q-learning) (tabular) | Minimal dependencies |
| **Production robotics / games** | [PPO](#29-ppo) | Stable, supports continuous & discrete |

### Decision Flow

```text
START
├─ No states (just actions)?
│   └─ Multi-Armed Bandits
│
├─ MDP model known?
│   └─ Dynamic Programming (Value / Policy Iteration)
│
├─ State space small (tabular)?
│   ├─ On-policy safe? → SARSA
│   └─ Off-policy optimal? → Q-Learning
│
└─ State space large / continuous?
    ├─ Discrete actions? → DQN → PPO
    ├─ Continuous actions? → DDPG → PPO
    └─ Policy gradient preferred? → REINFORCE → Actor-Critic → PPO
```

---

## 1. Life Cycle of RL Agents

### Life Cycle

#### 1. Problem Formulation

| Component | Meaning |
|-----------|---------|
| **State** $s$ | Agent's observation of the environment |
| **Action** $a$ | Choice made by the agent |
| **Reward** $r$ | Scalar feedback signal |
| **Policy** $\pi$ | Strategy: $\pi(a \mid s)$ or $\pi(s)$ |
| **Goal** | Maximize cumulative (discounted) reward |

```text
FUNCTION define_mdp():
    RETURN (S, A, P, R, γ)
    # S: states, A: actions, P: transition dynamics, R: reward function, γ: discount
```

---

#### 2. Environment Setup

| Step | Pseudocode skeleton |
|------|---------------------|
| **Create env** | `env ← Gym.make("CartPole-v1")` |
| **Reset** | `s ← env.reset()` |
| **Step** | `s', r, done, info ← env.step(a)` |
| **Wrap** | `normalize_obs`, `frame_stack`, `reward_shaping` |

```text
FUNCTION interact(env, policy, n_episodes):
    FOR ep IN 1..n_episodes:
        s ← env.reset()
        done ← False
        WHILE NOT done:
            a ← policy(s)
            s, r, done, info ← env.step(a)
            store_transition(s, a, r, s', done)
```

---

#### 3. Exploration Strategy

| Strategy | Pseudocode idea | Use when |
|----------|-----------------|----------|
| **ε-greedy** | random action with prob ε | Q-Learning, DQN |
| **Softmax / Boltzmann** | sample proportional to $\exp(Q(s,a)/\tau)$ | stochastic exploration |
| **UCB** | pick $\arg\max_a [\hat{\mu}_a + c\sqrt{\log t / n_a}]$ | bandits, optimism |
| **Entropy bonus** | add $-\beta \log \pi(a \mid s)$ to loss | PPO, Actor-Critic |

---

#### 4. Training Loop

```text
FUNCTION train(agent, env, n_episodes):
    FOR ep IN 1..n_episodes:
        trajectory ← collect_episode(env, agent.policy)
        agent.update(trajectory)          # or online per-step
        log(episode_return, loss)
    RETURN agent
```

**Hyperparameter Tuning**

| Method | Pseudocode idea | Use when |
|--------|-----------------|----------|
| **Grid Search** | sweep learning rate, γ, ε | small search space |
| **Random Search** | sample hyperparameter combos | faster exploration |
| **Eval callback** | train N steps → evaluate M episodes | deep RL (PPO, DQN) |

---

#### 5. Evaluation & Deployment

```text
FUNCTION evaluate(agent, env, n_episodes):
    returns ← []
    FOR ep IN 1..n_episodes:
        G ← run_episode(env, agent, explore=False)
        returns.append(G)
    RETURN mean(returns), std(returns)
```

##### Metrics Overview

A <a href="#metric">metric</a> measures how well an agent performs. Unlike supervised learning, labels come from **interaction**, not a fixed dataset.

**Return / Reward metrics**

**Episode Return** $G_t$

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$$

Total discounted reward from step $t$. Higher = better performance.

**Average Episode Reward** — mean return over evaluation episodes; primary deployment metric.

**Success Rate** — fraction of episodes meeting a goal (e.g., reach target, win game).

**Regret** (bandits) — $R_T = T \mu^* - \sum_{t=1}^{T} r_t$; cumulative gap vs. best arm.

**Learning diagnostics**

**TD Error** — $\delta_t = r + \gamma V(s') - V(s)$; should decrease during training.

**Policy Entropy** — $H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$; tracks exploration level.

**Value / Policy Loss** — MSE on value targets or policy gradient loss; monitor for divergence.

---

#### 6. Deployment

| Channel | Pattern |
|---------|---------|
| **Inference loop** | `a ← policy(s)` in real-time control |
| **Model export** | save weights + env normalization stats |
| **Sim-to-real** | train in simulator; fine-tune on real env |

```text
FUNCTION deploy(agent, env):
    save(agent.weights, "policy.pt")
    save(normalizer, "obs_rms.pkl")

FUNCTION serve(s, obs):
    s ← normalizer(obs)
    a ← agent.act(s, explore=False)
    RETURN a
```

---

## 2. RL Algorithms

### 2.1 Multi-Armed Bandits

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | No state; choose among $K$ actions (arms); exploration benchmark |
| **Cons** | No MDP structure; non-stationary arms need sliding windows |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $K$ | number of arms |
| $a_t$ | arm pulled at time $t$ |
| $r_t$ | reward observed |
| $\hat{\mu}_a$ | estimated mean reward of arm $a$ |
| $n_a$ | times arm $a$ was pulled |
| $\varepsilon$ | exploration rate |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Goal** — maximize $\sum_{t=1}^{T} r_t$; minimize <a href="#regret">regret</a>

2. **True mean** — $\mu_a = \mathbb{E}[R \mid a]$

3. **ε-greedy action**  
   $a_t = \begin{cases} \arg\max_a \hat{\mu}_a & \text{w.p. } 1-\varepsilon \\ \text{random } a & \text{w.p. } \varepsilon \end{cases}$

4. **UCB action**  
   $a_t = \arg\max_a \left[\hat{\mu}_a + c\sqrt{\dfrac{\log t}{n_a}}\right]$

5. **Incremental mean update**  
   $\hat{\mu}_a \leftarrow \hat{\mu}_a + \dfrac{1}{n_a}(r_t - \hat{\mu}_a)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **Initialize** $\hat{\mu}_a = 0$, $n_a = 0$ for all arms

2. **For** $t = 1, \ldots, T$:
   - Select arm $a_t$ (ε-greedy or UCB)
   - Observe reward $r_t$
   - $n_{a_t} \leftarrow n_{a_t} + 1$
   - Update $\hat{\mu}_{a_t}$ via incremental mean

3. **Output** — $\arg\max_a \hat{\mu}_a$ as best arm

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| ε | 0.01 – 0.2 | decay over time for ε-greedy |
| c (UCB) | 1 – 2 | optimism bonus |
| T | problem-dependent | total rounds |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Cumulative reward, <a href="#regret">regret</a>, best-arm identification accuracy. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Initialize
K ← 10
mu_hat ← zeros(K); n ← zeros(K)

# Run
FOR t IN 1..T:
    IF random() < epsilon:
        a ← random_int(0, K-1)
    ELSE:
        a ← argmax(mu_hat)
    r ← pull_arm(a)
    n[a] ← n[a] + 1
    mu_hat[a] ← mu_hat[a] + (r - mu_hat[a]) / n[a]

best_arm ← argmax(mu_hat)
```

</details>

---

### 2.2 Dynamic Programming

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Full MDP known ($P$, $R$); planning; baseline for theory |
| **Cons** | Requires complete model; infeasible for large/unknown state spaces |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $S$ | finite state space |
| $A$ | action space |
| $P(s' \mid s, a)$ | transition probability |
| $R(s, a)$ or $R(s,a,s')$ | expected reward |
| $V(s)$ | state value function |
| $V^*(s)$ | optimal value |
| $\gamma$ | <a href="#discount-factor">discount factor</a> |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Bellman expectation equation**  
   $V^\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)\bigl[R + \gamma V^\pi(s')\bigr]$

2. **Bellman optimality equation**  
   $V^*(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma V^*(s')\bigr]$

3. **Optimal policy**  
   $\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma V^*(s')\bigr]$

4. **Value Iteration update**  
   $V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)\bigl[R + \gamma V(s')\bigr]$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

**Value Iteration:**

1. Initialize $V(s) = 0$ for all $s$

2. **Repeat** until convergence:
   - For each $s$: $V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)[R + \gamma V(s')]$

3. Extract $\pi(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R + \gamma V(s')]$

**Policy Iteration:** evaluate $V^\pi$ → improve $\pi$ → repeat until $\pi$ stable.

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| γ | 0.9 – 0.999 | horizon weighting |
| tol | 1e-6 – 1e-4 | convergence threshold |
| max_iter | 100 – 10000 | safety cap |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Value convergence $\max_s |V_{k+1}(s) - V_k(s)|$; policy stability. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Given P[s,a,s'], R[s,a] for all s,a,s'
V ← zeros(|S|)

REPEAT:
    delta ← 0
    FOR each state s:
        v_old ← V[s]
        V[s] ← max_a sum_s'( P[s,a,s'] * (R[s,a,s'] + gamma * V[s']) )
        delta ← max(delta, |v_old - V[s]|)
UNTIL delta < tol

FOR each s:
    pi[s] ← argmax_a sum_s'( P[s,a,s'] * (R[s,a,s'] + gamma * V[s']) )
```

</details>

---

### 2.3 Monte Carlo Methods

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Episodic tasks; no model needed; unbiased value estimates |
| **Cons** | High variance; must wait until episode end; no bootstrapping |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $G_t$ | return from step $t$ |
| $N(s)$ | times state $s$ visited |
| $N(s,a)$ | times $(s,a)$ visited |
| $Q(s,a)$ | action-value estimate |
| first-visit / every-visit | MC averaging variant |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Return**  
   $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$

2. **Value estimate (first-visit)**  
   $V(s) = \text{avg}(G_t \mid s_t = s \text{ first visit})$

3. **Action-value**  
   $Q(s,a) = \text{avg}(G_t \mid s_t = s, a_t = a)$

4. **ε-greedy policy from Q**  
   $\pi(a|s) = 1-\varepsilon + \varepsilon/|A|$ if $a = \arg\max Q(s,\cdot)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Generate episode following $\pi$

2. **For each** first-visit $(s, a)$ in episode with return $G$:
   - $N(s,a) \leftarrow N(s,a) + 1$
   - $Q(s,a) \leftarrow Q(s,a) + \dfrac{1}{N(s,a)}(G - Q(s,a))$

3. Update $\pi$ to be ε-greedy w.r.t. $Q$

4. Repeat for many episodes

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| γ | 0.9 – 0.99 | |
| ε | 0.1 – 0.2 | exploration |
| n_episodes | 1000+ | more = lower variance |
| visit_type | first-visit, every-visit | first-visit more common |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, <a href="#success-rate">success rate</a>, value estimate variance. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
Q ← defaultdict(0); N ← defaultdict(0)

FOR ep IN 1..n_episodes:
    episode ← generate_episode(env, epsilon_greedy(Q, eps))
    visited ← set()
    G ← 0
  FOR (s, a, r) IN reversed(episode):
        G ← gamma * G + r
        IF (s,a) NOT IN visited:          # first-visit
            visited.add((s,a))
            N[s,a] ← N[s,a] + 1
            Q[s,a] ← Q[s,a] + (G - Q[s,a]) / N[s,a]
```

</details>

---

### 2.4 SARSA

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | On-policy TD control; online learning; safer near cliffs/obstacles |
| **Cons** | Must follow learning policy; can be conservative vs. optimal |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $Q(s,a)$ | action-value function |
| $\delta_t$ | <a href="#td-error">TD error</a> |
| $\alpha$ | learning rate |
| $s, a, r, s', a'$ | current transition + next action |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **On-policy TD target** — uses actual next action $a'$

2. **TD error**  
   $\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$

3. **Update**  
   $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t$

4. **Converges to** $Q^\pi$ for policy $\pi$ (under conditions)

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Initialize $Q(s,a)$ arbitrarily; $\pi$ from $Q$ (ε-greedy)

2. **For each** step:
   - Observe $s, a, r, s'$
   - Choose $a' \sim \pi(s')$
   - $\delta \leftarrow r + \gamma Q(s', a') - Q(s, a)$
   - $Q(s,a) \leftarrow Q(s,a) + \alpha \delta$
   - $s \leftarrow s'$; $a \leftarrow a'$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| α | 0.01 – 0.5 | decay over time |
| γ | 0.9 – 0.99 | |
| ε | 0.1 → 0.01 | decay schedule |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, <a href="#td-error">TD error</a> magnitude, <a href="#success-rate">success rate</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
Q ← zeros(|S|, |A|)
s ← env.reset()
a ← epsilon_greedy(Q, s, eps)

WHILE NOT done:
    s_next, r, done ← env.step(a)
    a_next ← epsilon_greedy(Q, s_next, eps)
    delta ← r + gamma * Q[s_next, a_next] - Q[s, a]
    Q[s, a] ← Q[s, a] + alpha * delta
    s, a ← s_next, a_next
```

</details>

---

### 2.5 Q-Learning

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Off-policy; learn optimal Q without following optimal policy; tabular baseline |
| **Cons** | Can overestimate; unstable with function approximation (use DQN fixes) |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $Q(s,a)$ | action-value estimate |
| $Q^*(s,a)$ | optimal action-value |
| $\max_{a'} Q(s', a')$ | greedy next-state value |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Bellman optimality (Q form)**  
   $Q^*(s,a) = \mathbb{E}\bigl[R + \gamma \max_{a'} Q^*(s', a')\bigr]$

2. **TD error (off-policy)**  
   $\delta_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$

3. **Update**  
   $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t$

4. **Greedy policy** — $\pi(s) = \arg\max_a Q(s,a)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Initialize $Q(s,a)$

2. **For each** step:
   - Select $a$ via ε-greedy on $Q$ (behavior policy)
   - Observe $r, s'$
   - $\delta \leftarrow r + \gamma \max_{a'} Q(s', a') - Q(s, a)$
   - $Q(s,a) \leftarrow Q(s,a) + \alpha \delta$

3. Repeat; converge to $Q^*$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| α | 0.01 – 0.5 | |
| γ | 0.9 – 0.99 | |
| ε | 0.1 → 0.01 | exploration decay |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a> (greedy eval), <a href="#td-error">TD error</a>, convergence of $Q$. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
Q ← zeros(|S|, |A|)
s ← env.reset()

WHILE training:
    a ← epsilon_greedy(Q, s, eps)
    s_next, r, done ← env.step(a)
    best_next ← max_a Q[s_next, a]
    delta ← r + gamma * best_next - Q[s, a]
    Q[s, a] ← Q[s, a] + alpha * delta
    s ← s_next IF NOT done ELSE env.reset()
```

</details>

---

### 2.6 Deep Q-Network (DQN)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Large/continuous state spaces; discrete actions; images or vectors as input |
| **Cons** | Discrete actions only; can be unstable; sample inefficient vs. policy methods |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $s$ | state (vector or image) | (d,) or (C,H,W) |
| $Q(s,a;\theta)$ | neural Q-function | scalar per action |
| $\theta$ | network weights | — |
| $\mathcal{D}$ | <a href="#replay-buffer">replay buffer</a> | — |
| $\theta^-$ | <a href="#target-network">target network</a> weights | — |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Q-network** — approximates $Q^*(s,a)$ with neural net

2. **TD target (with target net)**  
   $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$

3. **Loss (Huber or MSE)**  
   $\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\bigl[(y - Q(s,a;\theta))^2\bigr]$

4. **ε-greedy behavior** for exploration

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Initialize Q-network $\theta$, target network $\theta^- \leftarrow \theta$, replay buffer $\mathcal{D}$

2. **For each** step:
   - $a \leftarrow$ ε-greedy on $Q(s,\cdot;\theta)$
   - Execute $a$; store $(s,a,r,s')$ in $\mathcal{D}$
   - Sample mini-batch from $\mathcal{D}$
   - $y \leftarrow r + \gamma \max_{a'} Q(s',a';\theta^-)$
   - $\theta \leftarrow \theta - \alpha \nabla_\theta (y - Q(s,a;\theta))^2$
   - Every C steps: $\theta^- \leftarrow \theta$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| learning_rate | 1e-4 – 1e-3 | Adam optimizer |
| γ | 0.99 | |
| ε | 1.0 → 0.01 | decay over 1e6 steps |
| buffer_size | 1e5 – 1e6 | |
| batch_size | 32 – 128 | |
| target_update_freq | 1000 – 10000 | steps |
| double_dqn | True | reduces overestimation |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, loss, <a href="#td-error">TD error</a>, Q-value magnitude. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
Q_net ← QNetwork(state_dim, n_actions)
Q_target ← copy(Q_net)
buffer ← ReplayBuffer(capacity=100000)

s ← env.reset()
FOR step IN 1..total_steps:
    a ← epsilon_greedy(Q_net(s), eps)
    s_next, r, done ← env.step(a)
    buffer.store(s, a, r, s_next, done)

    batch ← buffer.sample(64)
    y ← r + gamma * max_a Q_target(s_next, a) * (1 - done)
    loss ← MSE(Q_net(s, a), y)
    update(Q_net, loss)
    IF step % 1000 == 0: Q_target ← copy(Q_net)
    s ← s_next IF NOT done ELSE env.reset()
```

</details>

---

### 2.7 REINFORCE

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Direct policy optimization; episodic tasks; discrete/continuous actions |
| **Cons** | High variance; slow convergence; on-policy only |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $\pi(a \mid s; \theta)$ | parameterized stochastic policy |
| $\theta$ | policy parameters |
| $G_t$ | return from step $t$ |
| $\nabla_\theta J$ | policy gradient |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Objective**  
   $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$

2. **Policy gradient theorem**  
   $\nabla_\theta J = \mathbb{E}\bigl[\nabla_\theta \log \pi(a|s;\theta) \cdot G_t\bigr]$

3. **Log-derivative trick** — enables gradient via sampled trajectories

4. **Baseline** (variance reduction)  
   $\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot (G_t - b(s_t))$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **For each** episode:
   - Generate trajectory $(s_0,a_0,r_1,\ldots,s_T)$ using $\pi_\theta$
   - Compute returns $G_t$ for each step
   - **For each** $t$: $\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \log \pi(a_t|s_t;\theta)$

2. Repeat over episodes

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| learning_rate | 1e-4 – 1e-2 | |
| γ | 0.99 | |
| baseline | learned value fn | reduces variance |
| entropy_coef | 0.01 | optional bonus |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, gradient norm, <a href="#policy-entropy">policy entropy</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
policy ← SoftmaxPolicy(state_dim, n_actions)
optimizer ← Adam(policy.parameters(), lr=1e-3)

FOR ep IN 1..n_episodes:
    trajectory ← collect_episode(env, policy)
    FOR t IN 0..len(trajectory)-1:
        G ← discounted_return(trajectory, t, gamma)
        loss ← -log_prob(policy, s_t, a_t) * G
        optimizer.step(loss)
```

</details>

---

### 2.8 Actor-Critic

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Lower variance than REINFORCE; online updates; continuous or discrete actions |
| **Cons** | Two networks to tune; can be unstable |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $\pi(a|s;\theta)$ | actor (policy) |
| $V(s;w)$ or $Q(s,a;w)$ | critic (value) |
| $A(s,a)$ | <a href="#advantage">advantage</a> $= Q - V$ |
| TD error $\delta_t$ | critic learning signal |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Critic TD error**  
   $\delta_t = r_{t+1} + \gamma V(s_{t+1}; w) - V(s_t; w)$

2. **Advantage estimate**  
   $A_t \approx \delta_t$ (or n-step / GAE)

3. **Critic update**  
   $w \leftarrow w + \alpha_w \delta_t \nabla_w V(s_t; w)$

4. **Actor update**  
   $\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi(a_t|s_t;\theta)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **For each** step $(s, a, r, s')$:
   - $a' \sim \pi(s')$
   - $\delta \leftarrow r + \gamma V(s'; w) - V(s; w)$
   - Critic: $w \leftarrow w + \alpha_w \delta \nabla_w V(s; w)$
   - Actor: $\theta \leftarrow \theta + \alpha_\theta \delta \nabla_\theta \log \pi(a|s;\theta)$

2. Repeat

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| actor_lr | 1e-4 – 1e-3 | |
| critic_lr | 1e-3 – 1e-2 | often higher |
| γ | 0.99 | |
| n_steps (A2C) | 5 – 20 | rollout length |
| gae_lambda | 0.95 | GAE advantage |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, critic loss, <a href="#td-error">TD error</a>, <a href="#policy-entropy">entropy</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
actor ← PolicyNet(); critic ← ValueNet()

s ← env.reset()
WHILE training:
    a ← sample(actor, s)
    s_next, r, done ← env.step(a)
    delta ← r + gamma * critic(s_next) - critic(s)
    critic.loss ← delta^2; update(critic)
    actor.loss ← -log_prob(actor, s, a) * delta.detach()
    update(actor)
    s ← s_next IF NOT done ELSE env.reset()
```

</details>

---

### 2.9 PPO

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Stable deep RL default; discrete or continuous; robotics, games |
| **Cons** | On-policy (less sample-efficient); many hyperparameters |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $\pi_\theta$ | current policy |
| $\pi_{\theta_{\text{old}}}$ | policy before update |
| $r_t(\theta)$ | probability ratio $\pi_\theta(a|s) / \pi_{\theta_{\text{old}}}(a|s)$ |
| $\hat{A}_t$ | advantage estimate (GAE) |
| $\varepsilon$ | clip range |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Probability ratio**  
   $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$

2. **Clipped surrogate objective**  
   $L^{\text{CLIP}} = \mathbb{E}\bigl[\min\bigl(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\bigr]$

3. **Total loss**  
   $L = L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2 H(\pi_\theta)$

4. **GAE advantage**  
   $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Collect rollout of $T$ steps with $\pi_{\theta_{\text{old}}}$

2. Compute advantages $\hat{A}_t$ via GAE

3. **For** K epochs on same batch:
   - $r_t \leftarrow \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$
   - $L \leftarrow$ clipped surrogate + value loss $-$ entropy bonus
   - $\theta \leftarrow \theta + \alpha \nabla_\theta L$

4. $\theta_{\text{old}} \leftarrow \theta$; repeat

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| clip ε | 0.1 – 0.3 | trust region |
| learning_rate | 1e-4 – 3e-4 | |
| γ | 0.99 | |
| gae_lambda | 0.95 | |
| n_steps | 128 – 2048 | rollout length |
| n_epochs | 3 – 10 | per batch |
| entropy_coef | 0.0 – 0.01 | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, clip fraction, KL divergence, value loss, <a href="#policy-entropy">entropy</a>. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
agent ← PPO(policy_net, value_net, clip_eps=0.2)

FOR iteration IN 1..N:
    rollout ← collect_rollout(env, agent, n_steps=2048)
    advantages ← compute_gae(rollout, gamma, lam=0.95)
    FOR epoch IN 1..10:
        ratio ← pi_theta(a|s) / pi_old(a|s)
        surr1 ← ratio * advantages
        surr2 ← clip(ratio, 1-eps, 1+eps) * advantages
        loss ← -min(surr1, surr2) + 0.5*value_loss - 0.01*entropy
        update(agent, loss)
    agent.sync_old_policy()
```

</details>

---

### 2.10 DDPG

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Continuous action spaces; control tasks (robotics, physics) |
| **Cons** | Off-policy instability; sensitive to hyperparameters; exploration tricky |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $\mu(s;\theta^\mu)$ | deterministic actor | action dim |
| $Q(s,a;\theta^Q)$ | critic | scalar |
| $\mathcal{N}$ | exploration noise | — |
| $\theta^-$ | target network weights | — |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Deterministic policy gradient**  
   $\nabla_{\theta^\mu} J \approx \mathbb{E}\bigl[\nabla_a Q(s,a;\theta^Q)\big|_{a=\mu(s)} \nabla_{\theta^\mu} \mu(s;\theta^\mu)\bigr]$

2. **Critic loss**  
   $L = \mathbb{E}\bigl[(r + \gamma Q(s', \mu(s';\theta^{\mu-});\theta^{Q-}) - Q(s,a;\theta^Q))^2\bigr]$

3. **Exploration** — $a = \mu(s) + \mathcal{N}$ (Gaussian noise)

4. **Soft target update**  
   $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Select $a = \mu(s;\theta^\mu) + \text{noise}$; observe $r, s'$

2. **Critic** — minimize TD error vs. target networks

3. **Actor** — maximize $Q(s, \mu(s;\theta^\mu))$ via deterministic policy gradient

4. **Soft update** target networks with $\tau \ll 1$

5. Repeat

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| actor_lr | 1e-4 – 1e-3 | |
| critic_lr | 1e-3 – 1e-2 | |
| γ | 0.99 | |
| τ (soft update) | 0.001 – 0.01 | |
| noise_std | 0.1 – 0.3 | Ornstein-Uhlenbeck |
| buffer_size | 1e6 | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

<a href="#episode-return">Episode return</a>, critic loss, Q-value, action noise scale. See <a href="#metrics-overview">Metrics Overview</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
actor ← DeterministicPolicy(state_dim, action_dim)
critic ← QNetwork(state_dim, action_dim)
actor_t, critic_t ← copy(actor), copy(critic)
buffer ← ReplayBuffer(1e6)

s ← env.reset()
FOR step IN 1..total_steps:
    a ← actor(s) + noise
    s_next, r, done ← env.step(clip(a))
    buffer.store(s, a, r, s_next, done)
    batch ← buffer.sample(64)
    a_next ← actor_t(s_next)
    y ← r + gamma * critic_t(s_next, a_next)
    update(critic, MSE(critic(s,a), y))
    update(actor, -mean(critic(s, actor(s))))
    soft_update(actor_t, actor, tau); soft_update(critic_t, critic, tau)
    s ← s_next IF NOT done ELSE env.reset()
```

</details>

---

## 3. Dictionary

Click any term below, or follow links throughout the cheat sheet.

**A–E:** [Action](#action) · [Actor](#actor) · [Advantage](#advantage) · [Bellman Equation](#bellman-equation) · [Critic](#critic) · [Discount Factor](#discount-factor) · [Episode Return](#episode-return) · [Exploration vs. Exploitation](#exploration-vs-exploitation)

**G–P:** [GAE](#gae) · [Hyperparameter](#hyperparameter) · [MDP](#mdp) · [Metric](#metric) · [Off-Policy](#off-policy) · [On-Policy](#on-policy) · [Policy](#policy) · [Policy Entropy](#policy-entropy) · [Q-Function](#q-function)

**R–Z:** [Regret](#regret) · [Replay Buffer](#replay-buffer) · [Reward](#reward) · [State](#state) · [Success Rate](#success-rate) · [Target Network](#target-network) · [TD Error](#td-error) · [Value Function](#value-function)

---

<a id="action"></a>
### Action

Choice $a \in \mathcal{A}$ made by the agent given state $s$. Can be discrete (left/right) or continuous (torque).

<a id="actor"></a>
### Actor

The <a href="#policy">policy</a> network in actor-critic methods. Outputs actions or action distributions. Updated via policy gradient.

<a id="advantage"></a>
### Advantage

How much better action $a$ is vs. average: $A(s,a) = Q(s,a) - V(s)$. Used in <a href="#29-ppo">PPO</a> and <a href="#28-actor-critic">Actor-Critic</a> to reduce variance.

<a id="bellman-equation"></a>
### Bellman Equation

Recursive relationship for value functions. Optimal form: $V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V^*(s')]$.

<a id="critic"></a>
### Critic

Value network estimating $V(s)$ or $Q(s,a)$. Provides baseline / TD error for actor updates.

<a id="discount-factor"></a>
### Discount Factor

$\gamma \in [0,1)$. Weights future rewards; $\gamma \to 0$ = myopic, $\gamma \to 1$ = far-sighted.

<a id="episode-return"></a>
### Episode Return

$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$. Total discounted reward from step $t$ to episode end.

<a id="exploration-vs-exploitation"></a>
### Exploration vs. Exploitation

Trade-off between trying new actions (explore) and using known best actions (exploit). ε-greedy, UCB, entropy bonus address this.

<a id="gae"></a>
### GAE

**G**eneralized **A**dvantage **E**stimation. $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$. Balances bias-variance in advantage estimates for <a href="#29-ppo">PPO</a>.

<a id="hyperparameter"></a>
### Hyperparameter

Config set before training — learning rate, γ, ε, clip range, buffer size.

<a id="mdp"></a>
### MDP

**M**arkov **D**ecision **P**rocess. Tuple $(S, \mathcal{A}, P, R, \gamma)$ defining an RL problem.

<a id="metric"></a>
### Metric

Number measuring agent performance — <a href="#episode-return">return</a>, <a href="#success-rate">success rate</a>, <a href="#regret">regret</a>, <a href="#td-error">TD error</a>.

<a id="off-policy"></a>
### Off-Policy

Learn about one policy while following another. <a href="#25-q-learning">Q-Learning</a>, <a href="#26-deep-q-network-dqn">DQN</a>, <a href="#210-ddpg">DDPG</a>.

<a id="on-policy"></a>
### On-Policy

Learn only from data generated by the current policy. <a href="#24-sarsa">SARSA</a>, <a href="#27-reinforce">REINFORCE</a>, <a href="#29-ppo">PPO</a>.

<a id="policy"></a>
### Policy

Strategy $\pi(a|s)$ mapping states to actions (stochastic) or $\pi(s)$ (deterministic). Goal: find optimal $\pi^*$.

<a id="policy-entropy"></a>
### Policy Entropy

$H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$. Higher = more random; used as exploration bonus.

<a id="q-function"></a>
### Q-Function

Action-value $Q(s,a)$ = expected return starting from $s$, taking $a$, then following $\pi$. Optimal $Q^*$ satisfies Bellman optimality.

<a id="regret"></a>
### Regret

$Cumulative\ gap = T\mu^* - \sum_t r_t$ in bandits. Measures exploration cost vs. always picking best arm.

<a id="replay-buffer"></a>
### Replay Buffer

Stores transitions $(s,a,r,s')$ for off-policy training. Breaks correlation in <a href="#26-deep-q-network-dqn">DQN</a> / <a href="#210-ddpg">DDPG</a>.

<a id="reward"></a>
### Reward

Scalar signal $r_t$ from environment. Agent maximizes cumulative (discounted) reward.

<a id="state"></a>
### State

Observation $s \in \mathcal{S}$ of the environment at time $t$. Markov property: future depends only on current $s$.

<a id="success-rate"></a>
### Success Rate

Fraction of episodes achieving a goal condition. Common evaluation metric for episodic tasks.

<a id="target-network"></a>
### Target Network

Copy of Q-network with delayed weights $\theta^-$. Stabilizes <a href="#26-deep-q-network-dqn">DQN</a> / <a href="#210-ddpg">DDPG</a> training targets.

<a id="td-error"></a>
### TD Error

$\delta_t = r + \gamma V(s') - V(s)$. Drives TD learning updates; should decrease during training.

<a id="value-function"></a>
### Value Function

$V(s)$ = expected return from state $s$ following policy $\pi$. Critic networks approximate $V$ or $Q$.

---

## Quick Pseudocode Template (End-to-End)

```text
# 1. Setup
env ← Gym.make("EnvName")
agent ← Algorithm(hyperparams)
buffer ← ReplayBuffer() IF off_policy ELSE None

# 2. Train
FOR episode OR step IN training_range:
    s ← env.reset()
    done ← False
    WHILE NOT done:
        a ← agent.select_action(s, explore=True)
        s_next, r, done, info ← env.step(a)
        agent.store(s, a, r, s_next, done)
        agent.update()                   # per-step or per-episode
        s ← s_next

# 3. Evaluate
mean_return ← evaluate(agent, env, n_episodes=100, explore=False)
PRINT mean_return

# 4. Deploy
save(agent.policy, "policy.pt")
serve(agent, real_env)
```
