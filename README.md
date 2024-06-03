# Pallas Puzzles

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdyro/Pallas-Puzzles/blob/main/Pallas-Puzzles.ipynb)

*This is a direct fork of the [Triton Puzzles](https://github.com/srush/Triton-Puzzles), but adapted for the Pallas
language for using a Triton-like kernel language in JAX.*

<p align="center">
<img style="max-width:400px;width:100%" src="pallas_puzzles_logo.svg">
</p>

By Robert Dyro, Ed Schmerling

Programming for accelerators such as GPUs is critical for modern AI systems.
This often means programming directly in proprietary low-level languages such as CUDA. [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) is an alternative [Triton](https://github.com/triton-lang/triton)-like language that allows you to code at a higher-level and compile to accelerators like GPU and TPU for [JAX](https://jax.readthedocs.io/).

Coding for Pallas is very similar to Numpy and JAX in both syntax and semantics. However, as a lower-level language there are a lot of details that you need to keep track of. In particular, one area that learners have trouble with is memory loading and storage which is critical for speed on low-level devices.

This set of puzzles is meant to teach you how to use Triton from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks. These puzzles **do not** need to run on GPU since they can use the Pallas (CPU-based) interpreter.

<!-- TODO: Discord: https://discord.gg/cudamode #triton-puzzles -->

<!-- TODO: ![image](https://github.com/srush/Triton-Puzzles/assets/35882/3e18a47d-1311-43d0-a025-ed1f593f919e) -->


---

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

<p align="center">
Original [Triton Puzzles](https://github.com/srush/Triton-Puzzles) repository below
</p>

# Triton Puzzles

w/ Tejas Ramesh and [Keren Zhou](https://www.jokeren.tech/) based on [Triton-Viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb)


Programming for accelerators such as GPUs is critical for modern AI systems.
This often means programming directly in proprietary low-level languages such as CUDA. [Triton](https://github.com/openai/triton/) is an alternative open-source language that allows you to code at a higher-level and compile to accelerators like GPU.

Coding for Triton is very similar to Numpy and PyTorch in both syntax and semantics. However, as a lower-level language there are a lot of details that you need to keep track of. In particular, one area that learners have trouble with is memory loading and storage which is critical for speed on low-level devices.

This set is puzzles is meant to teach you how to use Triton from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks. These puzzles **do not** need to run on GPU since they use a Triton interpreter.

Discord: https://discord.gg/cudamode #triton-puzzles

![image](https://github.com/srush/Triton-Puzzles/assets/35882/3e18a47d-1311-43d0-a025-ed1f593f919e)



If you are into this kind of thing, this is 7th in a series of these puzzles.

* https://github.com/srush/gpu-puzzles
* https://github.com/srush/tensor-puzzles
* https://github.com/srush/autodiff-puzzles
* https://github.com/srush/transformer-puzzles
* https://github.com/srush/GPTworld
* https://github.com/srush/LLM-Training-Puzzles
