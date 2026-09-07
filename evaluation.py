env_eval = SatelliteGraphEnv(...)
env_eval.seed(SEED)
env_eval.generate_environment(graph_topology, listofDemands)

agent = DQNAgent(env_eval, hparams['batch_size'])

checkpoint_dir = "./modelssample_DQN_agent"
checkpoint = tf.train.Checkpoint(model=agent.primary_network)
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    checkpoint.restore(latest_ckpt).expect_partial()
    agent.target_network.set_weights(agent.primary_network.get_weights())
    print("Restored from", latest_ckpt)
else:
    print("No checkpoint found, using random weights")

# 然后写一个只评估的 loop
state, demand, source, destination = env_eval.reset()
...
