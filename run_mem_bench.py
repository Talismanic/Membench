import os
from benchmark.MembenchAgent import MemBenchAgent
from benchmark.env.Membenenv import MemBenchEnv

config = {
    # two splits exist in MemData/FirstAgent: "roles" and "events"
    "dataset_path": {"FirstAgent": "MemData/FirstAgent/simple.json"},
    "dataset_type": ["roles", "events"],  # evaluate both categories
    # pick one of the memory configs below
    "memory_config": None,  # to be set later
    "LLM_config": {
        "type": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0,
    },
}

# Memory configs to choose from
FULL_MEMORY_CONFIG = {
    "type": "FullMemory",  # stores everything, truncates recall to max_words
    "args": {"max_words": 4000},
}

RECENT_MEMORY_CONFIG = {
    "type": "RecentMemory",  # keeps only the latest observations up to max_words
    "args": {"max_words": 4000},
}

MEMORY_BANK_CONFIG = {
    "type": "MemoryBank",
    "args": {
        "max_words": 4000,
        # retrieval/summarization internals; adjust as needed
        "embedding_dim": 768,
        "embedding_model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "window_size": 10,
        "threshold": 0.5,
    },
}

RETRIEVAL_MEMORY_CONFIG = {
    "type": "RetrievalMemory",
    "args": {
        "max_words": 4000,
        "embedding_dim": 384,
        "embedding_model_path": "intfloat/multilingual-e5-small",
    },
}

GAM_MEMORY_CONFIG = {
    "type": "GAMemory",
    "args": {
        "max_words": 4000,
        "recency_decay": 0.5,
        "recency_coef": 1.0,
        "importance_coef": 1.0,
        "relevance_coef": 1.0,
        "reflect_threshold": 1.5,
        "reflect_max_words": 400,
        "reflect_question_num": 3,
        "reflect_retrieval_topk": 5,
        "reflect_insight_num": 3,
        "embedding_dim": 768,
        "embedding_model_path": "sentence-transformers/all-MiniLM-L6-v2",
        # placeholder LLM configs; reuse your main LLM or point to smaller/cheaper models
        "reflector_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
        "importance_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
    },
}

MG_MEMORY_CONFIG = {
    "type": "MGMemory",
    "args": {
        "max_words": 4000,
        "embedding_dim": 768,
        "embedding_model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "summarizer_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
        "controller_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
    },
}

SC_MEMORY_CONFIG = {
    "type": "SCMemory",
    "args": {
        "max_words": 4000,
        "embedding_dim": 768,
        "embedding_model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "summarizer_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
        "controller_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
    },
}

RF_MEMORY_CONFIG = {
    "type": "RFMemory",
    "args": {
        "max_words": 4000,
        "embedding_dim": 768,
        "embedding_model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "summarizer_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
        "controller_LLM_config": {
            "type": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0,
        },
    },
}

ROAMPAL_MEMORY_CONFIG = {
    "type": "RoampalMemory",
    "args": {
        "max_words": 4000,
        "endpoint": "http://localhost:8000/api/memory-bank",
        "add_path": "/add",
        "search_path": "/search",
        "limit": 10,
        "collections": "memory_bank",
        "tags": [],
        "importance": 1.0,
    },
}

# Choose which memory config to use
config["memory_config"] = ROAMPAL_MEMORY_CONFIG

def run_once(env, agent, traj_idx):
    obs, reward, done, info = env.reset(traj_idx)
    agent.reset()
    step = 0
    while not done:
        action = agent.response(obs, reward, done, info, mode="FirstAgent", step=step)
        obs, reward, done, info, recall = env.step(action, mode="FirstAgent")
        step += 1
    return reward  # 1 if correct, 0 otherwise

if __name__ == "__main__":
    env = MemBenchEnv(config, path_i="FirstAgent")
    agent = MemBenchAgent(config)
    total = 0
    for i in range(len(env.dataset)):
        total += run_once(env, agent, i)
    print(f"Accuracy: {total} / {len(env.dataset)}")
