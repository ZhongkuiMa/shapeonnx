# 🔷 **ShapeONNX**: Shape Inference for ONNX Models

**ShapeONNX** is a lightweight and handy tool for performing **shape inference** on ONNX models—especially in cases where the standard ONNX inference tools fall short. 🧠📐

> 🧩 Sure, you can use [`onnx.shape_inference.infer_shapes`](https://onnx.ai/onnx/api/shape_inference.html), and in most cases, it does the job.  
> But ONNX is an **open and loosely defined protocol**. With its **inconsistent versions**, **non-standard conversions** between frameworks (hello, PyTorch 👋), and the way shapes are sometimes handled as constants, sometimes as inputs... things quickly get messy 😵‍💫.  
> That’s where **ShapeONNX** comes in—filling the gap, providing flexibility, and keeping you sane! 🛠️✨

---

## ❓ Why Do You Need ShapeONNX?

Sometimes, ONNX models involve shape manipulation logic that goes beyond the basic inference—often encountered in real-world exported models from frameworks like PyTorch.

Here are some scenarios where ShapeONNX shines:

- 🧬 You use the `Shape` operator to extract the dimensions of a tensor, and then:
    - Perform operations on the resulting shape, such as `Gather`, `Slice`, `Add`, `Sub`, `Mul`, `Sub`.
- 🔄 The **shape itself is passed as an input** to the model—making it a *dynamic shape* model.
- 🧠 You’re building tools like **SlimONNX**, or working on **neural network verification**, and need **precise and static shape information** for nodes that ONNX can't infer automatically.
- 🧪 You need shape info to simulate or analyze how tensors flow through complex control structures, even before the actual data is available.

In short, if you're working with **non-trivial ONNX models**, you’ll eventually hit a wall where native shape inference isn't enough. That’s when **ShapeONNX** becomes your best friend. 🤝

---

## ⚙️ How It Works

ShapeONNX statically simulates and traces shape-related computations in your model. Rather than just looking at the node and guessing its shape, it **executes a mini shape computation graph** that tracks how shapes are created and modified.

- ✅ It follows shape-related paths like `Shape → Gather → Add`, and propagates static values where possible.
- 🪄 It replaces intermediate shape tensors with constants if their values can be resolved.
- 💥 It simplifies dynamic shape operations into static ones when the logic allows it.

---

## 🚀 Usage Guide

### 💻 Installation

To get started with ShapeONNX, make sure your environment meets the following requirements:

- Python ≥ **3.10** (We recommend Python **3.12** 🐍)
- Install the minimal dependencies:

```bash
pip install onnx==1.17.0 numpy==2.2.4
```

📌 Note: Other versions may work, but we recommend not going too far below ONNX version 1.17.0.

> ⚠️ Internally, we treat ONNX IR version 22.0.0 as the baseline for shape-related logic. We use the newest version of ONNX operators and take them as the baseline because the attributes and inputs of a node are not consistent in different versions 😰. We highly recommend converting your model to this version using `onnx.version_converter` before using ShapeONNX.

## 🤝 Contributing

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 Please note: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
