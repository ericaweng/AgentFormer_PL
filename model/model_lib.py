from model.agentformer import AgentFormer
from model.dlow import DLow
from model.trajectory_optimizer import TrajectoryOptimizer


model_dict = {
    'agentformer': AgentFormer,
    'dlow': DLow,
    'trajectory_optimizer': TrajectoryOptimizer
}