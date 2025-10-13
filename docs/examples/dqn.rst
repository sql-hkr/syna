Deep Q-Network
===================

Solve CartPole-v1 using the DQN algorithm.

.. code-block:: python

    from syna.algo.dqn import DQNAgent
    from syna.rl import Trainer

    trainer = Trainer(
        env_name="CartPole-v1",
        num_episodes=300,
        agent=DQNAgent(
            lr=3e-4,
        ),
    )
    trainer.train()
