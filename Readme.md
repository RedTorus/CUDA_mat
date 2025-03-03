# GPU Homework Instructions

This repository contains three different CUDA projects, each with its own Makefile for compilation. Below, you will find instructions on how to compile and run the code in each folder, along with performance analysis using `nsys` or `nvprof`.

## Prerequisites

Ensure you have the following installed:

- NVIDIA CUDA Toolkit
- NVIDIA Nsight Systems (`nsys`)
- `nvprof` (for performance profiling)

## Directory Overview

- **`startup/`** - Copies seed data into GPU memory.
- **`mmm/`** - Implements efficient matrix multiplication using a single thread block.
- **`banks/`** - Contains various implementations to optimize matrix transposition and minimize bank conflicts.

---

## 1. Running the `startup` Code

### Compilation
```bash
cd startup
make
```
### Execution
```bash
./startup.x
```
### Profiling
To get detailed performance metrics:
```bash
nsys nvprof ./startup.x
```
### Cleaning Up
```bash
make clean
```

## 2. Running the `mmm` Code

### Compilation
```bash
cd mmm
make
```
### Execution
```bash
./mmm.x
```
### Profiling
```bash
nsys nvprof ./mmm.x
```
### Cleaning Up
```bash
make clean
```

## 3. Running the `banks` Code

This folder contains several implementations of an efficient matrix transpose designed to minimize bank conflicts.

### Available Targets

**Default Implementation:**
```bash
make default
./bank.x
```
**Shared Memory Optimized Implementation:**
```bash
make shared
./bankS.x
```
**Multi-Kernel Implementation:**
```bash
make multik
./bankMK.x
```
**Multi-SM Implementation:**
```bash
make multis
./bankMS.x
```

### Profiling

For any executable in this folder, run:
```bash
nsys nvprof ./<executable_name>.x
```
For example, to profile the shared memory version:
```bash
nsys nvprof ./bankS.x
```

### Cleaning Up
```bash
make clean
```



