from src.utils.logger import Logger
import os

logger = Logger()

# Simulate 10 episodes
for i in range(10):
    logger.log_episode(success=i % 2, reward=i * 0.5)
    logger.log_loss(loss=0.1 * i)

# Simulate evaluation
logger.log_eval(success=7, collision=3)

# Save file
os.makedirs("results/test", exist_ok=True)
logger.save("results/test/test_log.npz")

print("Logger test saved!")