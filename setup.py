from setuptools import setup, find_packages

setup(
    name="er-map",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "openenv-core>=0.2.3",
        "groq>=0.4.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "flask>=3.0.0",
        "pydantic>=2.0.0",
        "elevenlabs>=1.0.0",
        "edge-tts>=6.1.0",
        "pygame>=2.5.0",
    ],
    extras_require={
        "training": [
            "torch>=2.0.0",
            "transformers>=4.38.0",
            "trl>=0.8.0",
            "peft>=0.9.0",
            "accelerate>=0.27.0",
            "datasets>=2.16.0",
            "unsloth>=2024.1",
            "wandb>=0.16.0",
        ],
    },
    author="ER-MAP Team",
    description="Emergency Response Multi-Agent Pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
