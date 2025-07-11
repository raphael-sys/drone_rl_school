import numpy as np


def train(agent, env, writer, cfg,
        start_episode=0, best_score=float('-inf'),
        buffer=None, visualize=False, store_model=False):

    metrics = []
    rewards = []
    for current_episode in range(start_episode, start_episode + cfg.agent.episodes_to_train):
        obs = env.reset()
        ep_rewards = []
        ep_metrics = []
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)

            # Add the step to the experience buffer if one exists
            if buffer is not None:
                buffer.push(obs, action, reward, next_obs, done)

            # The training
            if buffer is None:
                agent.update(obs, action, reward, next_obs)
            elif len(buffer) >= cfg.agent.buffer_warmup_size:
                agent.update(buffer)

            
            obs = next_obs
            ep_rewards.append(reward)
            ep_metrics.append(env.metric())
            
            if visualize:
                env.render()

        # Store the average epoch metric and reward
        metrics.append(np.mean(ep_metrics))
        rewards.append(np.mean(ep_rewards))

        # Run the requested per episode decays
        if cfg.agent.epsilon_decay:
            agent.decay_epsilon(current_episode)
        if cfg.agent.lr_decay:
            agent.decay_alpha(current_episode)

        # Log values
        if writer:
            writer.add_scalar('metric', metrics[-1], current_episode)
            writer.add_scalar('reward', rewards[-1], current_episode)
            writer.add_scalar('epsilon', agent.epsilon, current_episode)
            writer.add_scalar('alpha_median', np.median(agent.lr), current_episode)
            writer.add_scalar('alpha_mean', np.mean(agent.lr), current_episode)
            writer.add_scalar('alpha_min', np.min(agent.lr), current_episode)
            writer.add_scalar('alpha_max', np.max(agent.lr), current_episode)
            writer.add_scalar('alpha_sum', np.sum(agent.lr), current_episode)

        # Update target net (for dqn)
        if cfg.agent.type == 'dqn' and current_episode % cfg.agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        # Print a training overview
        n = 100
        if (current_episode + 1) % n == 0:
            print(f"Episode {current_episode + 1 - n} to {current_episode + 1} \t \
                  Mean of Metric: {np.mean(metrics[-n:]):.2f}")
            if store_model and np.mean(metrics[-n:]) > best_score:
                best_score = np.mean(metrics[-n:])
                agent.save_model(best_score, current_episode)
                print(f"New best agent saved.")

    return current_episode, rewards, best_score
