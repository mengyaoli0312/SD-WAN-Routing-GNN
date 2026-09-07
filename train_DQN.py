# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # ✅ 必须最早
os.environ["PYTHONHASHSEED"] = "37"
import numpy as np
import gc
import random
import mpnn as gnn
import tensorflow as tf
from collections import deque
import time as tt
from environment1 import SatelliteGraphEnv
from topology_builder import build_satellite_graph
import logging
import json
import hashlib

import pickle
import gzip
import glob
import re
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #no gpu
graph_topology = 0 
SEED = 37
ITERATIONS = 0
TRAINING_EPISODES = 6000
EVALUATION_EPISODES = 5
FIRST_WORK_TRAIN_EPISODE = 10

MULTI_FACTOR_BATCH = 3 # Number of batches used in training
TAU = 0.08 # Only used in soft weights copy

differentiation_str = "sample_DQN_agent"
checkpoint_dir = "./models"+differentiation_str
store_loss = 3 # Store the loss every store_loss batches

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

tf.random.set_seed(SEED)

train_dir = "./"+differentiation_str
summary_writer = tf.summary.create_file_writer(train_dir)
# Three discrete bandwidth requirements (8/32/64).
listofDemands = [2,8,16]
copy_weights_interval = 10
evaluation_interval = 40
epsilon_start_decay = FIRST_WORK_TRAIN_EPISODE


hparams = {
    'l2': 0.001,
    'dropout_rate': 0.05,
    'link_state_dim': 20,
    'readout_units': 64,
    'learning_rate': 0.0005,
    'batch_size': 64,
    'T': 4, 
    'num_demands': len(listofDemands)
}

MAX_QUEUE_SIZE = 3000

# ----------------- Logging 配置 -----------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 清空默认 handler
logger.handlers.clear()

# 只写日志到文件，不往终端打..
os.makedirs("./Logs", exist_ok=True)
file_handler = logging.FileHandler("./Logs/train_debug.log", "a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def cummax(alist, extractor):
    #This is used to process the index offset of
    # nodes/edges when batching multiple images into a GNN. 
    # `alist`: Each element is a dictionary of graph features, e.g., `{'first': ..., 'second': ...}`.
    #`extractor`: Given `v`, return one column (first or second).
    #`maxes[k]`: The maximum value at that index in the `k`-th graph + 1 ≈ number of nodes/edges.
    # `cummaxes`: Prefix and offset values, e.g.:
    
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

AUX_VERSION = 1

def _to_np(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x

def serialize_transition(tr):
    return tuple(_to_np(v) for v in tr)

def deserialize_transition(tr_np):
    (s_ls, s_gid, s_f, s_s, s_ne,
     a, r, sp_ls, sp_gid, d, sp_f, sp_s, sp_ne) = tr_np

    return (
        tf.convert_to_tensor(s_ls, dtype=tf.float32),
        tf.convert_to_tensor(s_gid, dtype=tf.int32),
        tf.convert_to_tensor(s_f, dtype=tf.int32),
        tf.convert_to_tensor(s_s, dtype=tf.int32),
        tf.convert_to_tensor(s_ne, dtype=tf.int32),
        tf.convert_to_tensor(a, dtype=tf.int32),
        tf.convert_to_tensor(r, dtype=tf.float32),
        tf.convert_to_tensor(sp_ls, dtype=tf.float32),
        tf.convert_to_tensor(sp_gid, dtype=tf.int32),
        tf.convert_to_tensor(d, dtype=tf.float32),
        tf.convert_to_tensor(sp_f, dtype=tf.int32),
        tf.convert_to_tensor(sp_s, dtype=tf.int32),
        tf.convert_to_tensor(sp_ne, dtype=tf.int32),
    )

def aux_path_for_ckpt(ckpt_dir: str, ckpt_path: str) -> str:
    """
    ckpt_path 形如: ./modelssample_DQN_agent/ckpt_last-412
    aux 文件名: aux_ckpt_last-412.pkl.gz
    """
    base = os.path.basename(ckpt_path)
    return os.path.join(ckpt_dir, f"aux_{base}.pkl.gz")

def save_aux_state(ckpt_dir: str, ckpt_path: str, agent, max_reward: float) -> str:
    data = {
        "version": AUX_VERSION,
        "ckpt_path": str(ckpt_path),
        "episode": int(getattr(agent, "last_episode", 0)) if hasattr(agent, "last_episode") else 0,
        "epsilon": float(agent.epsilon),
        "max_reward": float(max_reward),

        "py_random_state": random.getstate(),
        "np_random_state": np.random.get_state(),

        "memory_maxlen": int(agent.memory.maxlen),
        "memory_len": int(len(agent.memory)),
        "memory": [serialize_transition(tr) for tr in agent.memory],
    }
    path = aux_path_for_ckpt(ckpt_dir, ckpt_path)
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_aux_state(path, agent):
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    if data.get("version", None) != AUX_VERSION:
        raise RuntimeError(f"AUX version mismatch: {data.get('version')} vs {AUX_VERSION}")

    # restore RNG
    random.setstate(data["py_random_state"])
    np.random.set_state(data["np_random_state"])

    # restore replay buffer
    agent.memory = deque(maxlen=int(data.get("memory_maxlen", MAX_QUEUE_SIZE)))
    mem_np = data.get("memory", [])
    for tr_np in mem_np:
        agent.memory.append(deserialize_transition(tr_np))

    # restore scalars
    agent.epsilon = float(data.get("epsilon", agent.epsilon))
    max_reward = float(data.get("max_reward", -1e9))
    ep = int(data.get("episode", 0))
    return ep, max_reward

def make_checkpoint(agent, ckpt_state):
    return tf.train.Checkpoint(
        primary=agent.primary_network,
        target=agent.target_network,
        optimizer=agent.optimizer,
        episode=ckpt_state["episode"],
        max_reward=ckpt_state["max_reward"],
        epsilon=ckpt_state["epsilon"],
    )

class DQNAgent:
    def __init__(self, env, batch_size):
        self.env = env
        self.memory = deque(maxlen=MAX_QUEUE_SIZE) #bufffer
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.writer = None
        self.K = 15 # K-paths
        self.listQValues = None
        self.numbersamples = batch_size
        self.action = None
        self.capacity_feature = None
        self.bw_allocated_feature = np.zeros((env.numEdges,len(env.listofDemands)))

        self.global_step = 0
        self.primary_network = gnn.myModel(hparams)
        self.primary_network.build()
        self.target_network = gnn.myModel(hparams)
        self.target_network.build()
        hparams['learning_rate'] = 1e-4  # 原来 5e-4 稍微保守一点

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7)
        hparams['learning_rate'] = 0.0005
        self.K = int(getattr(env, "K_paths", 15))

        self.r_clip_min = -2.0
        self.r_clip_max =  2.0

        # ✅ horizon：每个 episode 的最大步数（你 env 是 requests_per_episode=20）
        self.horizon = int(getattr(env, "requests_per_episode", 20))

        # ✅ 理论 y 上界（在 reward 被 clip 的情况下）
        # sum_{t=0..H-1} gamma^t * r_max = r_max * (1 - gamma^H) / (1 - gamma)
        self.y_clip = float(self.r_clip_max * (1.0 - (self.gamma ** self.horizon)) / (1.0 - self.gamma + 1e-8))

        self.nan_exit = True
        self.emergency_ckpt_prefix = None
    @staticmethod
    def _stat_dict_from_vec(vec, prefix: str):
        """vec: 1D tensor"""
        return {
            f"{prefix}_mean": tf.reduce_mean(vec),
            f"{prefix}_min":  tf.reduce_min(vec),
            f"{prefix}_max":  tf.reduce_max(vec),
        }

    @staticmethod
    def _compute_path_delay(env, path_idx):
        """根据 env.link_delay 计算一条路径的总时延（ms）"""
        if path_idx is None or path_idx == "GROUND":
            return None
        total = 0.0
        for u_idx, v_idx in zip(path_idx[:-1], path_idx[1:]):
            key = f"{u_idx}:{v_idx}"
            pos = env.edgesDict.get(key)
            if pos is None:
                return None
            total += float(env.link_delay[pos])
        return total
    
    def act(self, env, state, demand, source, destination, flagEvaluation):
        """
        Given a demand stored in the environment it allocates the K=4 shortest paths on the current 'state'
        and predicts the q_values of the K=4 different new graph states by using the GNN model.
        Picks the state according to epsilon-greedy approach. The flag=TRUE indicates that we are testing
        the model and thus, it won't activate the drop layers.
        """
        # Set to True if we need to compute K=4 q-values and take the maxium
        takeMax_epsilon = False
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # Initialize action
        action = 0

        # We get the K-paths between source-destination
        pathList = self._get_k_paths(env, source, destination)
        path = 0

        # 1. Implement epsilon-greedy to pick allocation
        # If flagEvaluation==TRUE we are EVALUATING => take always the action that the agent is saying has higher q-value
        # Otherwise, we are training with normal epsilon-greedy strategy
        if flagEvaluation:
            # If evaluation, compute K=4 q-values and take the maxium value
            takeMax_epsilon = True
        else:
            # If training, compute epsilon-greedy
            z = np.random.random()
            if z > self.epsilon:
                # Compute K=4 q-values and pick the one with highest value
                # In case of multiple same max values, return the first one
                takeMax_epsilon = True
            else:
                # Pick a random path and compute only one q-value
                path = np.random.randint(0, len(pathList))
                action = path

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            currentPath = pathList[path]

            state_copy = np.copy(state)
            alloc_mark = np.zeros(env.numEdges, dtype=np.float32)

            if currentPath != "GROUND":
                i, j = 0, 1
                while j < len(currentPath):
                    u = currentPath[i]; v = currentPath[j]
                    edge_idx = env.edgesDict[f"{u}:{v}"]
                    alloc_mark[edge_idx] = float(demand)
                    i += 1; j += 1

            # ✅ 临时扣减：让候选图在 capacity 上体现“选这条路会消耗多少”
            # 注意：这里按你的语义，down/up 都扣同一份 demand（如果你环境是双向独立，也可以只扣一个方向）
            state_copy[:, 0] = np.maximum(0.0, state_copy[:, 0] - alloc_mark)  # down_rem
            state_copy[:, 1] = np.maximum(0.0, state_copy[:, 1] - alloc_mark)  # up_rem

            features = self.get_graph_features(env, state_copy, alloc_mark)
            list_k_features.append(features)

            if not takeMax_epsilon:
                # If this is a random exploration, return to the current path Q in advance; there's no need to calculate the remaining paths.
                break

            path += 1

        #Then, the K candidate images are packaged into a batch and fed into MPNN:
        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        # Predict qvalues for all graphs within tensors
        self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                        tensors['second'], tensors['num_edges'], training=False).numpy()

        if takeMax_epsilon:
            action = int(np.argmax(self.listQValues))
            return action, list_k_features[action]
        else:
            # ✅ 随机选了 action，就返回 action 对应的特征
            return int(action), list_k_features[0]

    
    def select_action(self, env, state, demand, source, destination):
        action, _ = self.act(env, state, demand, source, destination, flagEvaluation=True)

        paths = self._get_k_paths(env, source, destination)  # ✅ 与 act() 完全一致
        p = paths[action]

        if p == "GROUND":
            real_path = env._compute_ground_path(source, destination)
        else:
            real_path = p

        approx_delay = self._compute_path_delay(env, real_path)
        return int(action), approx_delay
    
    def load_model(self, ckpt_dir):
        """
        ckpt_dir: 训练时的 checkpoint_dir，比如 "./modelssample_DQN_agent"
        """
        ckpt = tf.train.Checkpoint(model=self.primary_network)
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        ckpt.restore(latest).expect_partial()
        print(f"[LOAD] Loaded model weights from {latest}")

    def get_graph_features(self, env, copyGraph, alloc_mark=None):
        """
        构造 MPNN 输入特征（edge-level）。
        copyGraph: shape [numEdges, 3]，语义来自环境：
            copyGraph[:,0] = remaining_down_capacity
            copyGraph[:,1] = remaining_up_capacity
            copyGraph[:,2] = used_flag (可选，不一定用)
        alloc_mark: shape [numEdges]，表示“本次动作在该边分配的 demand 值”
            - 例如 demand=5 的边 alloc_mark[e]=5
            - 没分配的边 alloc_mark[e]=0
            - 用它来构造 bw_onehot（不会破坏 up/down capacity）
        """
        num_edges = env.numEdges
        num_demands = len(env.listofDemands)

        # -------- 0) alloc_mark 默认全 0 --------
        if alloc_mark is None:
            alloc_mark = np.zeros(num_edges, dtype=np.float32)
        else:
            alloc_mark = np.asarray(alloc_mark, dtype=np.float32)
            if alloc_mark.shape[0] != num_edges:
                raise ValueError(f"alloc_mark shape mismatch: {alloc_mark.shape} vs num_edges={num_edges}")

        # -------- 1) bw one-hot --------
        if (self.bw_allocated_feature is None or
            self.bw_allocated_feature.shape[0] != num_edges or
            self.bw_allocated_feature.shape[1] != num_demands):
            self.bw_allocated_feature = np.zeros((num_edges, num_demands), dtype=np.float32)
        else:
            self.bw_allocated_feature.fill(0.0)

        for e_idx, d in enumerate(alloc_mark):
            if d <= 0:
                continue
            for k, demand_value in enumerate(env.listofDemands):
                if d == demand_value:
                    self.bw_allocated_feature[e_idx, k] = 1.0
                    break

        # -------- 2) down/up remaining capacity --------
        down_rem = copyGraph[:, 0].astype(np.float32)
        up_rem   = copyGraph[:, 1].astype(np.float32)

        # numpy 层先消毒（非常关键）
        down_rem = np.nan_to_num(down_rem, nan=0.0, posinf=0.0, neginf=0.0)
        up_rem   = np.nan_to_num(up_rem,   nan=0.0, posinf=0.0, neginf=0.0)

        max_cap = float(getattr(env, "max_capacity", 1.0) + 1e-8)
        down_feat = down_rem / max_cap
        up_feat   = up_rem   / max_cap

        # cap_feat：取 min(down, up)
        cap_feat = np.minimum(down_feat, up_feat)

        # -------- 3) other features --------
        delay = np.asarray(getattr(env, "link_delay", np.zeros(num_edges)), dtype=np.float32)
        delay = np.nan_to_num(delay, nan=0.0, posinf=0.0, neginf=0.0)
        max_delay = float(getattr(env, "max_delay", 1.0) + 1e-8)
        delay_feat = delay / max_delay

        weather = np.asarray(getattr(env, "link_weather", np.zeros(num_edges)), dtype=np.float32)
        weather = np.nan_to_num(weather, nan=0.0, posinf=0.0, neginf=0.0)
        w_mean = float(getattr(env, "weather_mean", 0.0))
        w_std  = float(getattr(env, "weather_std", 1.0) + 1e-8)
        weather_feat = (weather - w_mean) / w_std

        cap_penalty_arr = np.asarray(getattr(env, "link_cap_penalty", np.zeros(num_edges)), dtype=np.float32)
        cap_penalty_arr = np.nan_to_num(cap_penalty_arr, nan=0.0, posinf=0.0, neginf=0.0)
        cap_penalty_feat = cap_penalty_arr / max_cap

        dl_degrade_arr = np.asarray(getattr(env, "link_download_deg", np.zeros(num_edges)), dtype=np.float32)
        dl_degrade_arr = np.nan_to_num(dl_degrade_arr, nan=0.0, posinf=0.0, neginf=0.0)
        dl_degrade_feat = dl_degrade_arr  # 通常 0~1

        link_type = np.asarray(getattr(env, "link_type_onehot", np.zeros((num_edges, 3))), dtype=np.float32)
        link_type = np.nan_to_num(link_type, nan=0.0, posinf=0.0, neginf=0.0)

        bet = np.asarray(getattr(env, "between_feature", np.zeros(num_edges)), dtype=np.float32)
        bet = np.nan_to_num(bet, nan=0.0, posinf=0.0, neginf=0.0)

        # -------- 4) concat -> hiddenStates --------
        cap_t      = tf.reshape(tf.convert_to_tensor(cap_feat[:num_edges], dtype=tf.float32), [num_edges, 1])
        up_t       = tf.reshape(tf.convert_to_tensor(up_feat[:num_edges], dtype=tf.float32),  [num_edges, 1])
        down_t     = tf.reshape(tf.convert_to_tensor(down_feat[:num_edges], dtype=tf.float32),[num_edges, 1])
        delay_t    = tf.reshape(tf.convert_to_tensor(delay_feat[:num_edges], dtype=tf.float32),[num_edges, 1])
        weather_t  = tf.reshape(tf.convert_to_tensor(weather_feat[:num_edges], dtype=tf.float32),[num_edges, 1])
        bet_t      = tf.reshape(tf.convert_to_tensor(bet[:num_edges], dtype=tf.float32),      [num_edges, 1])
        cap_pen_t  = tf.reshape(tf.convert_to_tensor(cap_penalty_feat[:num_edges], dtype=tf.float32), [num_edges, 1])
        dl_deg_t   = tf.reshape(tf.convert_to_tensor(dl_degrade_feat[:num_edges], dtype=tf.float32),   [num_edges, 1])

        type_t = tf.convert_to_tensor(link_type[:num_edges, :], dtype=tf.float32)
        bw_t   = tf.convert_to_tensor(self.bw_allocated_feature[:num_edges, :], dtype=tf.float32)

        hiddenStates = tf.concat(
            [cap_t, up_t, down_t, delay_t, weather_t, bet_t, cap_pen_t, dl_deg_t, type_t, bw_t],
            axis=1
        )

        # Tensor 层最终消毒：任何 NaN/Inf -> 0
        hiddenStates = tf.where(tf.math.is_finite(hiddenStates), hiddenStates, tf.zeros_like(hiddenStates))

        feature_dim = int(hiddenStates.shape[1])
        if feature_dim > hparams['link_state_dim']:
            raise ValueError(
                f"edge feature dim {feature_dim} > hparams['link_state_dim']={hparams['link_state_dim']}. "
                f"请增大 link_state_dim 或减少特征。"
            )

        # pad 到 link_state_dim
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - feature_dim]])
        link_state = tf.pad(hiddenStates, paddings, mode="CONSTANT")

        # -------- 5) Pack inputs for MPNN --------
        first = tf.convert_to_tensor(env.first[:env.firstTrueSize], dtype=tf.int32)
        second = tf.convert_to_tensor(env.second[:env.firstTrueSize], dtype=tf.int32)

        inputs = {
            'link_state': link_state,
            'first': first,
            'second': second,
            'num_edges': tf.convert_to_tensor(env.numEdges, dtype=tf.int32),
        }
        return inputs
    
    # def _write_tf_summary(self, gradients, loss):
    #     with summary_writer.as_default():
    #         tf.summary.scalar(name="loss", data=loss[0], step=self.global_step)
    #         tf.summary.histogram(name='gradients_5', data=gradients[5], step=self.global_step)
    #         tf.summary.histogram(name='gradients_7', data=gradients[7], step=self.global_step)
    #         tf.summary.histogram(name='gradients_9', data=gradients[9], step=self.global_step)
    #         tf.summary.histogram(name='FirstLayer/kernel:0', data=self.primary_network.variables[0], step=self.global_step)
    #         tf.summary.histogram(name='FirstLayer/bias:0', data=self.primary_network.variables[1], step=self.global_step)
    #         tf.summary.histogram(name='kernel:0', data=self.primary_network.variables[2], step=self.global_step)
    #         tf.summary.histogram(name='recurrent_kernel:0', data=self.primary_network.variables[3], step=self.global_step)
    #         tf.summary.histogram(name='bias:0', data=self.primary_network.variables[4], step=self.global_step)
    #         tf.summary.histogram(name='Readout1/kernel:0', data=self.primary_network.variables[5], step=self.global_step)
    #         tf.summary.histogram(name='Readout1/bias:0', data=self.primary_network.variables[6], step=self.global_step)
    #         tf.summary.histogram(name='Readout2/kernel:0', data=self.primary_network.variables[7], step=self.global_step)
    #         tf.summary.histogram(name='Readout2/bias:0', data=self.primary_network.variables[8], step=self.global_step)
    #         tf.summary.histogram(name='Readout3/kernel:0', data=self.primary_network.variables[9], step=self.global_step)
    #         tf.summary.histogram(name='Readout3/bias:0', data=self.primary_network.variables[10], step=self.global_step)
    #         summary_writer.flush()
    #         self.global_step = self.global_step + 1

    @tf.function
    def _forward_pass(self, x):
        q_state = self.primary_network(x[0], x[1], x[2], x[3], x[4], training=True)
        q_next  = self.target_network(x[7], x[8], x[10], x[11], x[12], training=False)
        q_next  = tf.stop_gradient(q_next)
        return q_state, q_next

    def _train_step(self, batch):
    # ============================================================
    # 0) dbg: 永远保证 replay() 需要的 key 存在（防 KeyError）
    # ============================================================
        dbg = {
            "has_bad": tf.constant(False),
            "has_bad_nt": tf.constant(False),
            "has_bad_term": tf.constant(False),

            "num_done_in_batch": tf.constant(0, tf.int32),
            "mean_abs_y_minus_r_done": tf.constant(0.0, tf.float32),
            "y_saturation_ratio": tf.constant(0.0, tf.float32),
            "y_mean_batch_nt": tf.constant(0.0, tf.float32),
            "y_std_batch_nt": tf.constant(0.0, tf.float32),

            "bad_idx": tf.constant(0, tf.int32),
            "bad_q_sa": tf.constant(0.0, tf.float32),
            "bad_y": tf.constant(0.0, tf.float32),
            "bad_q_next_max": tf.constant(0.0, tf.float32),
            "bad_r": tf.constant(0.0, tf.float32),
            "bad_done": tf.constant(0.0, tf.float32),

            # replay() 里 dump 用
            "qn_vec": tf.zeros([1], tf.float32),
            "y_vec": tf.zeros([1], tf.float32),
            "r_vec": tf.zeros([1], tf.float32),
            "d_vec": tf.zeros([1], tf.float32),

            # 统计（先给默认，后面会覆盖）
            "q_sa_mean": tf.constant(0.0, tf.float32),
            "q_sa_min":  tf.constant(0.0, tf.float32),
            "q_sa_max":  tf.constant(0.0, tf.float32),
            "y_mean": tf.constant(0.0, tf.float32),
            "y_min":  tf.constant(0.0, tf.float32),
            "y_max":  tf.constant(0.0, tf.float32),
            "q_next_max_mean": tf.constant(0.0, tf.float32),
            "q_next_max_min":  tf.constant(0.0, tf.float32),
            "q_next_max_max":  tf.constant(0.0, tf.float32),
            "r_mean": tf.constant(0.0, tf.float32),
            "r_min":  tf.constant(0.0, tf.float32),
            "r_max":  tf.constant(0.0, tf.float32),
            "done_mean": tf.constant(0.0, tf.float32),

            "loss": tf.constant(float("nan"), tf.float32),
        }

        # ============================================================
        # 1) forward + build targets
        # ============================================================
        with tf.GradientTape() as tape:
            q_sa_list, y_list, q_next_max_list, r_list, done_list = [], [], [], [], []
            bad_found = False

            for x in batch:
                # x layout (你 add_sample 的顺序)：
                # 0 s_link_state
                # 1 s_graph_id
                # 2 s_first
                # 3 s_second
                # 4 s_num_edges
                # 5 action (int32)
                # 6 reward (float32)
                # 7 sp_link_state
                # 8 sp_graph_id
                # 9 done (float32)
                # 10 sp_first
                # 11 sp_second
                # 12 sp_num_edges

                # --- Q(s,a) ---
                q_state_all = self.primary_network(x[0], x[1], x[2], x[3], x[4], training=True)
                q_state_all = tf.reshape(q_state_all, [-1])

                # 你的网络输出通常是 1 个图一个 Q（因为你传进去的是“执行动作后的图”）
                # 保险起见：空/异常直接判 bad
                if tf.size(q_state_all) <= 0:
                    bad_found = True
                    break
                q_sa = q_state_all[0]

                # --- reward/done ---
                r = tf.cast(x[6], tf.float32)
                r = tf.clip_by_value(r, self.r_clip_min, self.r_clip_max)
                done = tf.cast(x[9], tf.float32)

                # --- Double DQN：a* from online, value from target ---
                q_next_online_all = self.primary_network(x[7], x[8], x[10], x[11], x[12], training=False)
                q_next_target_all = self.target_network(x[7], x[8], x[10], x[11], x[12], training=False)

                q_next_online_all = tf.reshape(q_next_online_all, [-1])
                q_next_target_all = tf.reshape(q_next_target_all, [-1])
                q_next_target_all = tf.stop_gradient(q_next_target_all)

                # K 可能对不齐，取 min
                k_eff = tf.minimum(tf.shape(q_next_online_all)[0], tf.shape(q_next_target_all)[0])
                if k_eff <= 0:
                    bad_found = True
                    break
                q_next_online_all = q_next_online_all[:k_eff]
                q_next_target_all = q_next_target_all[:k_eff]

                a_star = tf.argmax(q_next_online_all, output_type=tf.int32)
                q_next_max = tf.gather(q_next_target_all, a_star)

                # --- TD target ---
                y = r + self.gamma * (1.0 - done) * q_next_max
                y = tf.clip_by_value(y, -self.y_clip, self.y_clip)
                y = tf.stop_gradient(y)

                # --- finite check（任何一个非 finite => bad）---
                finite_ok = tf.math.is_finite(q_sa) & tf.math.is_finite(y) & tf.math.is_finite(q_next_max) & tf.math.is_finite(r) & tf.math.is_finite(done)
                if not bool(finite_ok.numpy()):
                    bad_found = True
                    break

                q_sa_list.append(q_sa)
                y_list.append(y)
                q_next_max_list.append(tf.stop_gradient(q_next_max))
                r_list.append(r)
                done_list.append(done)

            # 如果 batch 内出现坏样本或没样本：返回 None + dbg(带齐字段)
            if bad_found or len(q_sa_list) == 0:
                dbg["has_bad"] = tf.constant(True)
                # loss 保持 NaN，用于 replay 的 emergency 逻辑
                dbg["loss"] = tf.constant(float("nan"), tf.float32)

                # dump vec 给 1 个 NaN，避免你切片 [:10] 报错
                dbg["qn_vec"] = tf.constant([float("nan")], tf.float32)
                dbg["y_vec"]  = tf.constant([float("nan")], tf.float32)
                dbg["r_vec"]  = tf.constant([float("nan")], tf.float32)
                dbg["d_vec"]  = tf.constant([float("nan")], tf.float32)

                dbg["bad_idx"] = tf.constant(0, tf.int32)
                dbg["bad_q_sa"] = tf.constant(float("nan"), tf.float32)
                dbg["bad_y"] = tf.constant(float("nan"), tf.float32)
                dbg["bad_q_next_max"] = tf.constant(float("nan"), tf.float32)
                dbg["bad_r"] = tf.constant(float("nan"), tf.float32)
                dbg["bad_done"] = tf.constant(float("nan"), tf.float32)

                return None, dbg["loss"], dbg

            # stack -> vectors
            q_sa_vec = tf.stack(q_sa_list, axis=0)
            y_vec    = tf.stack(y_list, axis=0)
            qn_vec   = tf.stack(q_next_max_list, axis=0)
            r_vec    = tf.stack(r_list, axis=0)
            d_vec    = tf.stack(done_list, axis=0)

            # loss: huber + reg
            loss = tf.keras.losses.Huber()(y_vec, q_sa_vec)
            loss = tf.reduce_mean(loss)
            if self.primary_network.losses:
                loss = loss + tf.add_n(self.primary_network.losses)

        # ============================================================
        # 2) loss 非有限：直接返回（dbg 也要齐）
        # ============================================================
        if not bool(tf.math.is_finite(loss).numpy()):
            dbg["has_bad"] = tf.constant(True)
            dbg["loss"] = tf.cast(loss, tf.float32)

            dbg["qn_vec"] = tf.where(tf.math.is_finite(qn_vec), qn_vec, tf.zeros_like(qn_vec))
            dbg["y_vec"]  = tf.where(tf.math.is_finite(y_vec),  y_vec,  tf.zeros_like(y_vec))
            dbg["r_vec"]  = tf.where(tf.math.is_finite(r_vec),  r_vec,  tf.zeros_like(r_vec))
            dbg["d_vec"]  = tf.where(tf.math.is_finite(d_vec),  d_vec,  tf.zeros_like(d_vec))

            return None, dbg["loss"], dbg

        # ============================================================
        # 3) gradients + finite clean + clip + apply
        # ============================================================
        grads = tape.gradient(loss, self.primary_network.trainable_variables)

        safe_grads = []
        for g in grads:
            if g is None:
                safe_grads.append(None)
            else:
                safe_grads.append(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)))
        grads = safe_grads

        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.primary_network.trainable_variables))

        # ============================================================
        # 4) dbg stats（保证 replay() 全字段可用）
        # ============================================================
        dbg["loss"] = tf.cast(loss, tf.float32)

        # vec 保存（replay 的 dump 用）
        dbg["qn_vec"] = qn_vec
        dbg["y_vec"]  = y_vec
        dbg["r_vec"]  = r_vec
        dbg["d_vec"]  = d_vec

        # 基础统计
        dbg["q_sa_mean"] = tf.reduce_mean(q_sa_vec)
        dbg["q_sa_min"]  = tf.reduce_min(q_sa_vec)
        dbg["q_sa_max"]  = tf.reduce_max(q_sa_vec)

        dbg["y_mean"] = tf.reduce_mean(y_vec)
        dbg["y_min"]  = tf.reduce_min(y_vec)
        dbg["y_max"]  = tf.reduce_max(y_vec)

        dbg["q_next_max_mean"] = tf.reduce_mean(qn_vec)
        dbg["q_next_max_min"]  = tf.reduce_min(qn_vec)
        dbg["q_next_max_max"]  = tf.reduce_max(qn_vec)

        dbg["r_mean"] = tf.reduce_mean(r_vec)
        dbg["r_min"]  = tf.reduce_min(r_vec)
        dbg["r_max"]  = tf.reduce_max(r_vec)

        dbg["done_mean"] = tf.reduce_mean(d_vec)

        # terminal correctness: done=1 => y == r
        done_mask = tf.cast(d_vec > 0.5, tf.bool)
        num_done = tf.reduce_sum(tf.cast(done_mask, tf.int32))
        dbg["num_done_in_batch"] = num_done

        abs_y_minus_r = tf.abs(y_vec - r_vec)
        dbg["mean_abs_y_minus_r_done"] = tf.cond(
            num_done > 0,
            lambda: tf.reduce_mean(tf.boolean_mask(abs_y_minus_r, done_mask)),
            lambda: tf.constant(0.0, tf.float32),
        )
        has_bad_term = tf.cond(
            num_done > 0,
            lambda: tf.reduce_any(tf.boolean_mask(abs_y_minus_r, done_mask) > 1e-5),
            lambda: tf.constant(False),
        )
        dbg["has_bad_term"] = has_bad_term

        # NaN/Inf health check（理论上不会有，但保险）
        finite_all = (
            tf.reduce_all(tf.math.is_finite(q_sa_vec)) &
            tf.reduce_all(tf.math.is_finite(y_vec)) &
            tf.reduce_all(tf.math.is_finite(qn_vec)) &
            tf.reduce_all(tf.math.is_finite(r_vec)) &
            tf.reduce_all(tf.math.is_finite(d_vec))
        )
        has_bad_nan = tf.logical_not(finite_all)

        dbg["y_saturation_ratio"] = tf.reduce_mean(
            tf.cast(tf.abs(y_vec) > (self.y_clip * 0.98), tf.float32)
        )

        # 选择 worst idx（用于 BAD_SAMPLE 日志 + dump）
        td_err = tf.abs(tf.stop_gradient(y_vec) - q_sa_vec)
        bad_score = td_err
        bad_score = tf.where(done_mask & (abs_y_minus_r > 1e-5), tf.constant(1e9, tf.float32), bad_score)
        bad_score = tf.where(~tf.math.is_finite(td_err), tf.constant(1e10, tf.float32), bad_score)
        worst_idx = tf.argmax(bad_score, output_type=tf.int32)

        dbg["bad_idx"] = worst_idx
        dbg["bad_q_sa"] = q_sa_vec[worst_idx]
        dbg["bad_y"] = y_vec[worst_idx]
        dbg["bad_q_next_max"] = qn_vec[worst_idx]
        dbg["bad_r"] = r_vec[worst_idx]
        dbg["bad_done"] = d_vec[worst_idx]

        # 最终 has_bad
        dbg["has_bad"] = has_bad_nan | has_bad_term

        return grads, dbg["loss"], dbg

    def _get_k_paths(self, env, source, destination):
        key = f"{source}:{destination}"
        paths = list(env.allPaths.get(key, []))

        if not paths:
            return ["GROUND"]  # 至少给一个动作

        # 保证 GROUND 在最后（如果你需要这个约定）
        paths_no_ground = [p for p in paths if p != "GROUND"]
        has_ground = any(p == "GROUND" for p in paths)
        paths = paths_no_ground + (["GROUND"] if has_ground else [])

        return paths[:self.K]


    def replay(self, episode):
        if len(self.memory) < self.numbersamples:
            return True

        ok_updates = 0
        last_dbg = None
        last_loss_val = None

        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(self.memory, self.numbersamples)
            grads, loss, dbg = self._train_step(batch)
            last_dbg = dbg

            loss_val = float(loss.numpy()) if hasattr(loss, "numpy") else float(loss)
            last_loss_val = loss_val

            if grads is None or (not np.isfinite(loss_val)):
                logger.info(f"[WARN] Bad batch skipped: loss={loss_val}")
                continue

            ok_updates += 1

            if i % store_loss == 0:
                self.fileLogs.write(".," + f"{loss_val:.9f}" + ",\n")
                self.fileLogs.flush()

                logger.info(
                    f"[TRAIN] episode={episode}, batch_iter={i}, loss={loss_val:.6f}, epsilon={self.epsilon:.4f} | "
                    f"q_sa(mean/min/max)={float(dbg['q_sa_mean']):.3f}/{float(dbg['q_sa_min']):.3f}/{float(dbg['q_sa_max']):.3f} | "
                    f"y(mean/min/max)={float(dbg['y_mean']):.3f}/{float(dbg['y_min']):.3f}/{float(dbg['y_max']):.3f} | "
                    f"q_next_max(mean/min/max)={float(dbg['q_next_max_mean']):.3f}/{float(dbg['q_next_max_min']):.3f}/{float(dbg['q_next_max_max']):.3f} | "
                    f"r(mean/min/max)={float(dbg['r_mean']):.3f}/{float(dbg['r_min']):.3f}/{float(dbg['r_max']):.3f} | "
                    f"done(mean)={float(dbg['done_mean']):.3f} | "
                    f"num_done={int(dbg['num_done_in_batch'].numpy())} | "
                    f"mean|y-r|(done=1)={float(dbg['mean_abs_y_minus_r_done']):.6f} | "
                    f"y_sat_ratio={float(dbg['y_saturation_ratio']):.3f}"
                )

            if bool(dbg["has_bad"].numpy()):
                logger.info(
                    f"[BAD_SAMPLE] episode={episode}, batch_iter={i} | "
                    f"idx={int(dbg['bad_idx'].numpy())}, bad_q_sa={float(dbg['bad_q_sa']):.3f}, "
                    f"bad_y={float(dbg['bad_y']):.3f}, bad_q_next_max={float(dbg['bad_q_next_max']):.3f}, "
                    f"bad_r={float(dbg['bad_r']):.3f}, done={float(dbg['bad_done']):.0f}"
                )
                qn = dbg["qn_vec"].numpy()
                yy = dbg["y_vec"].numpy()
                rr = dbg["r_vec"].numpy()
                dd = dbg["d_vec"].numpy()
                logger.info(f"[DUMP] q_next_max_vec(first10)={qn[:10]}")
                logger.info(f"[DUMP] y_vec(first10)={yy[:10]}")
                logger.info(f"[DUMP] r_vec(first10)={rr[:10]}")
                logger.info(f"[DUMP] done_vec(first10)={dd[:10]}")

        if ok_updates == 0:
            logger.info(f"[FATAL] replay failed: all batches NaN/invalid at episode={episode}, last_loss={last_loss_val}")
            return False

        if episode % copy_weights_interval == 0:
            self.target_network.set_weights(self.primary_network.get_weights())

        gc.collect()
        return True

            
    
    def add_sample( self, env_training, state_action, action,  reward, done, new_state, new_demand, new_source, new_destination,
):
        """
        存 replay buffer 的 transition：
        (s, a, r, s', done)
        其中 s' 需要构造 K 条候选动作对应的图输入（和 act() 一样的方式）。
        """
        # 当前 state_action 是 act() 返回的 features（对应当前执行的 action）
        # 给它补 graph_id=0（单图）
        state_action["graph_id"] = tf.fill([tf.shape(state_action["link_state"])[0]], 0)

        # -------- s'：为 next request 构造 K 个候选图输入 --------
        key = f"{new_source}:{new_destination}"
        pathList = self._get_k_paths(env_training, new_source, new_destination)
        list_k_features = []

        for p in pathList:
            # 注意：new_state_copy 只是环境的真实状态副本（down/up remain），不能被 demand 覆盖
            new_state_copy = np.copy(new_state)

            # alloc_mark 仅用于“本次动作在哪些边写了 demand”，不污染 new_state_copy
            alloc_mark = np.zeros(env_training.numEdges, dtype=np.float32)

            if p != "GROUND":
                i, j = 0, 1
                while j < len(p):
                    u = p[i]
                    v = p[j]
                    edge_idx = env_training.edgesDict[f"{u}:{v}"]
                    alloc_mark[edge_idx] = float(new_demand)
                    i += 1
                    j += 1
            new_state_copy[:, 0] = np.maximum(0.0, new_state_copy[:, 0] - alloc_mark)
            new_state_copy[:, 1] = np.maximum(0.0, new_state_copy[:, 1] - alloc_mark)
            features = self.get_graph_features(env_training, new_state_copy, alloc_mark)
            list_k_features.append(features)

        # -------- 把 K 个图打包成 batch（和 act() 同样的 cummax 逻辑）--------
        vs = list_k_features

        graph_ids = [tf.fill([tf.shape(vs[it]["link_state"])[0]], it) for it in range(len(vs))]
        first_offset = cummax(vs, lambda v: v["first"])
        second_offset = cummax(vs, lambda v: v["second"])

        tensors = {
            "graph_id": tf.concat(graph_ids, axis=0),
            "link_state": tf.concat([v["link_state"] for v in vs], axis=0),
            "first": tf.concat([v["first"] + m for v, m in zip(vs, first_offset)], axis=0),
            "second": tf.concat([v["second"] + m for v, m in zip(vs, second_offset)], axis=0),
            "num_edges": tf.math.add_n([v["num_edges"] for v in vs]),
        }

        # -------- 存 transition --------
        self.memory.append(
            (
                state_action["link_state"],                    # x[0]
                state_action["graph_id"],                      # x[1]
                state_action["first"],                         # x[2]
                state_action["second"],                        # x[3]
                tf.convert_to_tensor(state_action["num_edges"]),  # x[4]
                tf.convert_to_tensor(int(action), tf.int32),   # x[5]
                tf.convert_to_tensor(float(reward), tf.float32),# x[6]
                tensors["link_state"],                         # x[7]
                tensors["graph_id"],                           # x[8]
                tf.convert_to_tensor(1.0 if done else 0.0, tf.float32),
                tensors["first"],                              # x[10]
                tensors["second"],                             # x[11]
                tf.convert_to_tensor(tensors["num_edges"]),     # x[12]
            )
        )

def rollback_to_good(checkpoint, ckpt_path, agent, aux_path, logger, ckpt_state, lr_scale=0.2, drop_recent=500):
    logger.info(f"[ROLLBACK] restoring ckpt: {ckpt_path}")
    checkpoint.restore(ckpt_path).expect_partial()

    # restore aux (rng + memory)
    if aux_path and os.path.exists(aux_path):
        ep, maxr = load_aux_state(aux_path, agent)
        logger.info(f"[ROLLBACK] restored aux: {aux_path} (ep={ep}, mem={len(agent.memory)})")
    else:
        logger.info("[ROLLBACK] no aux found -> clear memory")
        agent.memory.clear()

    # reset optimizer（会换对象 -> 必须重建 checkpoint）
    old_lr = float(agent.optimizer.learning_rate.numpy()) if agent.optimizer is not None else 1e-4
    new_lr = old_lr * lr_scale
    agent.optimizer = tf.keras.optimizers.Adam(learning_rate=new_lr, epsilon=1e-7)
    logger.info(f"[ROLLBACK] optimizer reset: lr {old_lr} -> {new_lr}")

    # drop recent samples
    if drop_recent > 0 and len(agent.memory) > 0:
        for _ in range(min(drop_recent, len(agent.memory))):
            agent.memory.pop()
        logger.info(f"[ROLLBACK] dropped recent samples: {drop_recent}, now mem={len(agent.memory)}")

    # sync target
    agent.target_network.set_weights(agent.primary_network.get_weights())

    # ✅ 重建 checkpoint（关键）
    checkpoint = make_checkpoint(agent, ckpt_state)
    return checkpoint

def _all_finite(tensors) -> bool:
    for t in tensors:
        if t is None:
            continue

        # t 可能不是 Tensor（比如 optimizer.variables() 里混有其他对象）
        try:
            dt = t.dtype
        except Exception:
            continue

        # 只对浮点类型做 is_finite；int/bool 不需要检查
        if dt.is_floating:
            if not tf.reduce_all(tf.math.is_finite(t)).numpy():
                return False
        else:
            # int/bool/其它类型：跳过
            continue

    return True


def check_and_fix_optimizer(agent, logger, lr=1e-4):
    if not _all_finite(agent.primary_network.trainable_variables):
        logger.info("[FATAL] primary network has non-finite weights.")
        return False, False
    if not _all_finite(agent.target_network.trainable_variables):
        logger.info("[FATAL] target network has non-finite weights.")
        return False, False

    changed = False
    try:
        opt_vars = agent.optimizer.variables()
    except Exception:
        opt_vars = []

    if opt_vars and (not _all_finite(opt_vars)):
        logger.info("[WARN] optimizer has non-finite slot vars -> RESET optimizer.")
        agent.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7)
        changed = True
        return True, changed

    try:
        norms = [tf.linalg.global_norm([v]).numpy() for v in opt_vars if v is not None]
        if norms and (np.max(norms) > 1e6):
            logger.info("[WARN] optimizer slot norm too large -> RESET optimizer.")
            agent.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7)
            changed = True
    except Exception:
        pass

    return True, changed


if __name__ == "__main__":
    # ====== 1) Env ======
    epsilon_decay_start_episode = epsilon_start_decay
    env_training = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=listofDemands,
        K_paths=15,
        requests_per_episode=20,
        shuffle_requests=True,
        epoch="2025-12-05 06:00:00",
    )
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    env_training.seed(SEED)
    env_training.generate_environment(graph_topology, listofDemands)

    env_eval = SatelliteGraphEnv(
        tle_path="iridium_66_main.tle",
        en_csv="analysis_data_Enschede.csv",
        osn_csv="analysis_data_Osnabrück.csv",
        listofDemands=listofDemands,
        K_paths=4,
        requests_per_episode=20,
        shuffle_requests=False,
        epoch="2025-12-05 06:00:00",
    )
    env_eval.seed(SEED)
    env_eval.generate_environment(graph_topology, listofDemands)

    # ====== 2) Agent ======
    batch_size = hparams["batch_size"]
    agent = DQNAgent(env_training, batch_size)

    # ====== 3) Logs ======
    os.makedirs("./Logs", exist_ok=True)
    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")
    agent.fileLogs = fileLogs

    # ====== 4) Checkpoint state ======
    ckpt_state = {
        "episode": tf.Variable(0, dtype=tf.int64, name="episode"),
        "max_reward": tf.Variable(-1e9, dtype=tf.float32, name="max_reward"),
        "epsilon": tf.Variable(1.0, dtype=tf.float32, name="epsilon"),
    }

    # ====== 5) Checkpoint object (必须先创建再 restore) ======
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = make_checkpoint(agent, ckpt_state)

    checkpoint_prefix_best = os.path.join(checkpoint_dir, "ckpt_best")
    checkpoint_prefix_last = os.path.join(checkpoint_dir, "ckpt_last")
    agent.emergency_ckpt_prefix = os.path.join(checkpoint_dir, "ckpt_nan")

    # ====== 6) Restore  ======
    forced_ckpt = os.path.join(checkpoint_dir, "ckpt_last-410")
    state_path = os.path.join(checkpoint_dir, "train_state.json")

    start_episode = 1
    max_reward = -1e9
    agent.epsilon = 1.0

    if tf.io.gfile.exists(forced_ckpt + ".index"):
        logger.info(f"[CKPT] Restoring from {forced_ckpt}")
        checkpoint.restore(forced_ckpt).expect_partial()

        saved_ep   = int(ckpt_state["episode"].numpy())
        saved_eps  = float(ckpt_state["epsilon"].numpy())
        saved_maxr = float(ckpt_state["max_reward"].numpy())
        logger.info(f"[DBG] ckpt_state: saved_ep={saved_ep}, saved_eps={saved_eps}, saved_maxr={saved_maxr}")

        if saved_ep > 0:
            start_episode = saved_ep + 1
            agent.epsilon = saved_eps
            max_reward = saved_maxr
            logger.info(f"[CKPT] Resume: start_episode={start_episode}, eps={agent.epsilon:.4f}, max_reward={max_reward:.4f}")
        else:
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    train_state = json.load(f)
                start_episode = int(train_state.get("last_episode", 0)) + 1
                agent.epsilon = float(train_state.get("epsilon", 1.0))
                max_reward = float(train_state.get("max_reward", -1e9))
                logger.info(f"[CKPT] Resume from train_state.json: start_episode={start_episode}, eps={agent.epsilon:.4f}, max_reward={max_reward:.4f}")
            else:
                logger.info("[CKPT] No train_state.json -> start from scratch.")

        # restore 后同步 target
        agent.target_network.set_weights(agent.primary_network.get_weights())
    else:
        logger.info(f"[CKPT] Not found: {forced_ckpt}.index -> training from scratch")

    # epsilon sanity
    if not np.isfinite(agent.epsilon) or agent.epsilon <= 0.0:
        logger.info(f"[WARN] bad epsilon restored: {agent.epsilon}, force to 0.05")
        agent.epsilon = 0.05
    ckpt_state["epsilon"].assign(float(agent.epsilon))

    logger.info(f"[PATH] checkpoint_dir = {os.path.abspath(checkpoint_dir)}")
    logger.info(f"[PATH] state_path     = {os.path.abspath(state_path)}")
    logger.info(f"[PATH] forced_restore_ckpt = {forced_ckpt}")
    logger.info(f"[DBG] FINAL restored: start_episode={start_episode}, epsilon={agent.epsilon:.4f}, max_reward={max_reward:.4f}")

    # ====== restore aux by ckpt basename ======
    forced_aux = aux_path_for_ckpt(checkpoint_dir, forced_ckpt)
    if os.path.exists(forced_aux):
        ep_aux, maxr_aux = load_aux_state(forced_aux, agent)
        logger.info(f"[AUX] Restored {forced_aux} (ep_aux={ep_aux}, mem={len(agent.memory)})")
    else:
        logger.info(f"[AUX] Missing {forced_aux} -> memory empty")

    # ====== optimizer check (may reset optimizer -> must rebuild checkpoint) ======
    ok_opt, changed = check_and_fix_optimizer(agent, logger, lr=1e-4)
    if not ok_opt:
        raise SystemExit(2)
    if changed:
        checkpoint = make_checkpoint(agent, ckpt_state)
        logger.info("[OPT] checkpoint rebuilt after optimizer reset")

    last_good_ckpt = forced_ckpt
    last_good_aux  = forced_aux
    # ====== 7) Training loop ======
    try:
        save_last_interval = 10
        for train_ep in range(start_episode, TRAINING_EPISODES + 1):
            episode_ok = True

            state, demand, source, destination = env_training.reset()
            done = False
            ep_reward = 0.0
            step = 0

            while not done:
                action, state_action = agent.act(
                    env_training, state, demand, source, destination, flagEvaluation=False
                )

                new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(
                    state, action, demand, source, destination
                )

                agent.add_sample(
                    env_training, state_action, action, reward, done,
                    new_state, new_demand, new_source, new_destination
                )

                state, demand, source, destination = new_state, new_demand, new_source, new_destination
                ep_reward += float(reward)
                step += 1

                if train_ep >= FIRST_WORK_TRAIN_EPISODE and len(agent.memory) >= batch_size * MULTI_FACTOR_BATCH:
                    episode_ok = agent.replay(train_ep)
                    if not episode_ok:
                        logger.info(f"[FATAL] NaN/Inf detected in training: episode={train_ep}")
                        checkpoint = rollback_to_good(
                            checkpoint, last_good_ckpt, agent, last_good_aux, logger,
                            ckpt_state=ckpt_state, lr_scale=0.2, drop_recent=500
                        )
                        raise SystemExit(2)

            # episode end
            if train_ep >= epsilon_decay_start_episode and agent.epsilon > agent.epsilon_min:
                agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

            agent.last_episode = int(train_ep)
            
            if episode_ok and (ep_reward > max_reward):
                max_reward = ep_reward
                ckpt_path = checkpoint.save(checkpoint_prefix_best)
                aux_path  = save_aux_state(checkpoint_dir, ckpt_path, agent, max_reward)
                last_good_ckpt = ckpt_path
                last_good_aux  = aux_path
                logger.info(f"[CKPT] Saved BEST at ep={train_ep}, reward={ep_reward:.4f}")

            ckpt_state["episode"].assign(train_ep)
            ckpt_state["epsilon"].assign(float(agent.epsilon))
            ckpt_state["max_reward"].assign(float(max_reward))

            if episode_ok and (train_ep % save_last_interval == 0):
                ckpt_path = checkpoint.save(checkpoint_prefix_last)
                aux_path  = save_aux_state(checkpoint_dir, ckpt_path, agent, max_reward)
                last_good_ckpt = ckpt_path
                last_good_aux  = aux_path
                logger.info(f"[CKPT] Saved LAST at ep={train_ep}")

            logger.info(f"[TRAIN_EP] ep={train_ep}, steps={step}, reward={ep_reward:.4f}, epsilon={agent.epsilon:.4f}")

            # optional evaluation
            if train_ep % evaluation_interval == 0:
                rewards_test = np.zeros(EVALUATION_EPISODES, dtype=np.float32)
                for eval_ep in range(EVALUATION_EPISODES):
                    s, d, src, dst = env_eval.reset()
                    done_eval = False
                    total_r = 0.0
                    while not done_eval:
                        a, _ = agent.act(env_eval, s, d, src, dst, flagEvaluation=True)
                        s2, r, done_eval, d2, src2, dst2 = env_eval.make_step(s, a, d, src, dst)
                        s, d, src, dst = s2, d2, src2, dst2
                        total_r += r
                    rewards_test[eval_ep] = total_r

                mean_eval_reward = float(np.mean(rewards_test))
                logger.info(f"[EVAL_DONE] train_ep={train_ep}, mean_reward={mean_eval_reward:.4f}")

    except SystemExit as e:
        code = int(getattr(e, "code", 1))
        logger.info(f"[EXIT] SystemExit({code}) caught in main.")

        try:
            emergency_path = checkpoint.save(agent.emergency_ckpt_prefix)
            logger.info(f"[EMERGENCY] Saved emergency checkpoint to {emergency_path}")
        except Exception as ex:
            logger.info(f"[EMERGENCY] Failed to save emergency checkpoint: {ex}")

        raise
    finally:
        try:
            fileLogs.close()
        except Exception:
            pass

    logger.info("[TRAINING FINISHED]")