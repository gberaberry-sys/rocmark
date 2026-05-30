"""Standardized test prompt sets for benchmarking."""

import random

SHORT_PROMPTS = [
    "What is machine learning?",
    "Explain Python decorators in one sentence.",
    "What does GPU stand for?",
    "Summarize the theory of relativity briefly.",
    "What is a neural network?",
    "Define API in simple terms.",
    "What is Docker used for?",
    "Explain REST in one paragraph.",
    "What is the difference between CPU and GPU?",
    "How does garbage collection work?",
]

MEDIUM_PROMPTS = [
    "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
    "Explain the differences between TCP and UDP protocols. When would you use each one?",
    "Describe the architecture of a transformer model. What are attention mechanisms?",
    "How does a distributed consensus algorithm like Raft work? Explain the leader election process.",
    "Write a SQL query to find the top 5 customers by total order value from an e-commerce database.",
    "Explain the CAP theorem in distributed systems. Give examples of each trade-off.",
    "Describe the differences between multiprocessing and multithreading in Python.",
    "How does backpropagation work in neural networks? Explain the chain rule application.",
    "Write a function to detect cycles in a directed graph using DFS.",
    "Explain the concept of database indexing. When should you use B-tree vs hash indexes?",
]

LONG_PROMPTS = [
    "Implement a complete LRU cache in Python with O(1) get and put operations. Use a doubly linked list and hash map. Include proper error handling, thread safety considerations, and comprehensive unit tests.",
    "Design a rate limiter using the token bucket algorithm. It should support multiple API endpoints with different rate limits, handle concurrent requests, and provide metrics. Write the implementation in Python with proper abstractions.",
    "Explain the complete lifecycle of an HTTP request from browser to server response. Cover DNS resolution, TCP handshake, TLS negotiation, HTTP headers, server processing, and response rendering.",
    "Write a Python decorator that implements retry logic with exponential backoff, jitter, circuit breaker pattern, and configurable retry policies. Include logging and metrics collection.",
    "Implement a thread-safe producer-consumer queue with bounded buffer, graceful shutdown, and dead letter queue support in Python. Use proper synchronization primitives.",
]

CODE_PROMPTS = [
    "```python\n# Optimize this function for large inputs\ndef find_pairs(nums, target):\n    result = []\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):\n            if nums[i] + nums[j] == target:\n                result.append((nums[i], nums[j]))\n    return result\n```\n\nRewrite this with O(n) time complexity.",
    "```python\n# What does this code do? Explain and suggest improvements.\nclass DataPipeline:\n    def __init__(self):\n        self.steps = []\n    def add_step(self, fn):\n        self.steps.append(fn)\n        return self\n    def run(self, data):\n        for step in self.steps:\n            data = step(data)\n        return data\n```",
    "Write a Python context manager for database transactions that handles commit, rollback, connection pooling, and nested transactions using savepoints.",
]

MULTILINGUAL_PROMPTS = [
    "Jelaskan perbedaan antara machine learning dan deep learning dalam bahasa Indonesia.",
    "在Python中，装饰器是什么？请用中文解释并给出一个实际例子。",
    "Объясните разницу между REST и GraphQL API на русском языке.",
    "¿Qué es la programación funcional? Explícalo en español con ejemplos.",
]


def get_test_prompts(count: int, seed: int = 42) -> list[str]:
    """Get a mixed set of test prompts.

    Args:
        count: Number of prompts to return.
        seed: Random seed for reproducibility.

    Returns:
        List of prompt strings with mixed difficulty.
    """
    rng = random.Random(seed)
    all_prompts = (
        SHORT_PROMPTS * 3 +
        MEDIUM_PROMPTS * 2 +
        LONG_PROMPTS +
        CODE_PROMPTS * 2 +
        MULTILINGUAL_PROMPTS
    )
    rng.shuffle(all_prompts)
    return all_prompts[:count]
