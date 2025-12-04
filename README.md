> **Attribution:** This repository is derived from [https://github.com/import-myself/Membench](https://github.com/import-myself/Membench).
> The goal of this mirror is to facilitate the benchmarking of other memory mechanisms on top of the original Membench paper.

<h1 align="center"> MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents </h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2506.21605)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT)


Data location:

https://pan.baidu.com/s/1HqwY0nu5bltSAJ2TbnxcFQ?pwd=yzsj Extraction code: yzsj

or

https://drive.google.com/file/d/112Zraj4pTPH4Idph6i1uMOLA_LPFdGr0/view?usp=sharing

If you want to directly use the data sampled in the paper, you can find the 0-10k and 100k datasets in the **data2test** directory. These represent the lengths of the entire conversation. Note that if your test setup is to mimic the **memory flow**, you can use these directly. However, if you want to test the model's ability with **long contexts**, you should sample more examples.

## Data Details

We provide two versions of the data: **Categorical data** and  **data**.

**Categorical data** represents data for each category under the first-person (Participation) and third-person (Observation), with category information and details available in Appendix Table 6.

The **data** represents the full datasets after categorization, as mentioned in the paper, including:

- Participation-Reflective (FirstAgentHighLevel)
- Participation-Factual (FirstAgentLowLevel)
- Observation-Reflective (ThirdAgentHighLevel)
- Observation-Factual (ThirdAgentLowLevel)

## Noise data

We also provide a noise dataset, NoiseData, to extend the length of dialogues or information flows, including:

- FirstNoise (FirstAgent)
- ThirdNoise (ThirdAgent)

You can use `makenoise.py` to generate extended and sampled Complete Data. You can modify the main function in the source code to implement this. For each additional unit of noise length, the token count increases by about 1k on average.

You can refer to the following code:

```python
MakeNoiseMessageHighLevel('data/ThirdAgentDataHighLevel.json', 'data2test', length=10, sample_num=100)  ## Add noise to the third-person high-level

MakeNoiseMessage('data/ThirdAgentDataLowLevel.json', 'data2test', length=10, sample_num=100)  ## Add noise to the third-person low-level

MakeNoiseSession('data/FirstAgentDataLowLevel.json', 'data2test', length=10, sample_num=100)  ## Add noise to the first-person low-level

MakeNoiseSession('data/FirstAgentDataHighLevel.json', 'data2test', length=10, sample_num=100)  ## Add noise to the first-person high-level
```

## Easy Benchmarking with `run_mem_bench.py`

This repository includes a `run_mem_bench.py` script to simplify the process of benchmarking different memory mechanisms.

### Quick Start

1.  **Install Dependencies**: Ensure you have the required packages installed (see `requirements.txt`).
2.  **Configure**: Open `run_mem_bench.py` and adjust the `config` dictionary. You can select the dataset, LLM settings, and the memory configuration you want to test.
3.  **Run**:
    ```bash
    python run_mem_bench.py
    ```

### Supported Memory Systems

The benchmark supports various memory architectures, ranging from simple baselines to complex, agentic memory systems.

*   **FullMemory**: Keeps the entire interaction history, truncating only when the context window is exceeded.
*   **RecentMemory**: Retains only the most recent observations, simulating a sliding window context.
*   **RetrievalMemory**: A standard RAG (Retrieval-Augmented Generation) system using vector embeddings (e.g., `multilingual-e5-small`) to retrieve relevant past information.
*   **GAMemory**: Based on **Generative Agents**, incorporating recency, importance, and relevance scoring, along with a reflection mechanism to synthesize high-level insights.
*   **MemoryBank**: Implements the **MemoryBank** architecture, utilizing an Ebbinghaus forgetting curve and hierarchical summarization (daily/global summaries).
*   **MGMemory**: Inspired by **MemGPT**, featuring a tiered memory structure (Working Context, FIFO Queue, Recursive Summary) and tool-use for memory management (archival/recall).
*   **SCMemory**: **Self-Controlled Memory**, where a "Controller" LLM dynamically decides when to retrieve or summarize information based on the query's needs.
*   **RFMemory**: Based on **Reflexion**, allowing the agent to "reflect" on past execution trails to improve performance in subsequent attempts.

### Roampal Integration

We have introduced **RoampalMemory** to facilitate integration with [Roampal](https://github.com/roampal-ai/roampal), a dedicated memory-as-a-service solution.

*   **RoampalMemory**: Offloads memory storage and retrieval to a running Roampal instance. This allows you to benchmark the performance of Roampal's decoupled memory management against other embedded memory mechanisms.
    *   **Configuration**: Set the `endpoint` in `ROAMPAL_MEMORY_CONFIG` to point to your Roampal API (default: `http://localhost:8000/api/memory-bank`).






